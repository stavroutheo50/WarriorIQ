from __future__ import annotations

import logging
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from core.config import SETTINGS
from core.video import open_capture


def sam_sampling_stride(source_fps: float, total_source_frames: int) -> int:
    """Bound SAM work by frequency, an absolute frame budget, and the clock.

    The first two bounds are both absolute, and neither knows how long the
    video is. That is what made the continuous pass unaffordable on short
    fights: four samples per second of footage at 0.173 s each is 0.69 s of SAM
    for every second of video, so a seventy second round spent forty-five
    seconds here and had twenty-five left for everything else. The same
    settings cost 5% of a twenty minute fight.

    The third bound fixes that by being relative. The product's budget is the
    video's own length, so the continuous pass gets a share of it -
    sam_budget_fraction - and the stride follows from that. It can only raise
    the stride, never lower it, so a long fight where the cap does not bind is
    unchanged and no configuration gets quietly overridden.

    This bounds the SWEEP, not the safety net. Recovering a fighter the
    detector lost runs off sam_recovery_enabled and is untouched.
    """
    fps_stride = max(1, round(float(source_fps) / max(1.0, SETTINGS.sam_continuous_fps)))
    budget_stride = max(1, int(np.ceil(max(0, total_source_frames) / max(1, SETTINGS.sam_continuous_max_frames))))

    fraction = float(SETTINGS.sam_budget_fraction)
    cost = float(SETTINGS.sam_frame_cost_seconds)
    clock_stride = 1
    if fraction > 0 and cost > 0 and source_fps > 0 and total_source_frames > 0:
        segment_seconds = float(total_source_frames) / float(source_fps)
        affordable_frames = max(1.0, segment_seconds * fraction / cost)
        clock_stride = max(1, int(np.ceil(total_source_frames / affordable_frames)))
    return max(fps_stride, budget_stride, clock_stride)


class _DecodedClip:
    """Frames already in memory, in the form SAM2's own loader would return.

    SAM2 reads a clip from a folder of JPEGs or from an MP4, and both mean the
    frames leave memory and come back. The continuous path decoded every frame
    with OpenCV, wrote it out as a quality-86 JPEG, and SAM2 then reopened it
    with PIL and resized it again. Measured on a sixty-second bout that round
    trip cost 8.5 s of a 131 s analysis - and it also put a lossy re-encode in
    front of the one component that finds a fighter the detector missed, on
    footage where a fighter is already only sixty pixels tall.

    The tensor allocated here is the same one SAM2 allocates: frames are
    resized and normalised straight into it, one chunk at a time, so nothing
    is held that the JPEG path did not already hold.
    """

    def __init__(self, image_size: int, capacity: int, height: int, width: int):
        self.image_size = max(1, int(image_size))
        self.images = torch.zeros(max(1, int(capacity)), 3,
                                  self.image_size, self.image_size, dtype=torch.float32)
        self.height, self.width = int(height), int(width)
        self.count = 0

    @property
    def full(self) -> bool:
        return self.count >= int(self.images.shape[0])

    @staticmethod
    def prepare(frame_bgr: np.ndarray, image_size: int) -> np.ndarray:
        """The per-frame image work, apart so the sweep's helper thread can do it."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # PIL resizes with bicubic by default, which is what SAM2's loader got
        # from Image.resize, so the model sees the filter it was tuned on.
        return np.ascontiguousarray(cv2.resize(rgb, (int(image_size), int(image_size)),
                                               interpolation=cv2.INTER_CUBIC))

    def add(self, frame_bgr: np.ndarray) -> None:
        self.add_prepared(self.prepare(frame_bgr, self.image_size))

    def add_prepared(self, resized_rgb: np.ndarray) -> None:
        if self.full:
            return
        frame = torch.from_numpy(resized_rgb)
        self.images[self.count] = frame.permute(2, 0, 1).float().div_(255.0)
        self.count += 1

    def taken(self) -> torch.Tensor:
        return self.images[: self.count]


_END = object()


class _SampledFrames:
    """The frames the sweep keeps, decoded and prepared on a helper thread.

    2026-10-20, measured on the owner's RTX 5060: the sweep was 43 s of an
    84 s analysis (52%) and 70 s of 213 s. It read() every source frame -
    decode and colour conversion - to keep one in `stride` (about one in
    fifteen), and decoded while the card waited. The frame pass already
    grab()s the frames it skips and decodes ahead (core/frame_feed.py; grab
    measured -61% on 1080p60). This brings the sweep the same two things.

    The results do not change: the same frames, in the same order, stopping
    at the same frame a read() loop would stop at, prepared by the same code.
    OpenCV and torch release the GIL, so decoding runs beside propagation.
    """

    def __init__(self, cap, start_frame: int, end_frame: int, stride: int, image_size: int,
                 depth: int = 16):
        self._cap, self._start, self._end = cap, int(start_frame), int(end_frame)
        self._stride, self._image_size = max(1, int(stride)), int(image_size)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._stopped = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="warrioriq-sam-frames", daemon=True)
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stopped.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            source = self._start
            while source < self._end and not self._stopped.is_set():
                # read() is grab() + retrieve(); a failed grab is where read()
                # would have stopped too.
                if not self._cap.grab():
                    break
                if (source - self._start) % self._stride == 0:
                    ok, frame = self._cap.retrieve()
                    if not ok or frame is None:
                        break
                    if not self._put((source, _DecodedClip.prepare(frame, self._image_size), frame.shape[:2])):
                        return
                source += 1
        except BaseException as exc:  # noqa: BLE001 - handed to the reading thread
            self._error = exc
        finally:
            self._put(_END)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is _END:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def close(self) -> None:
        self._stopped.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5.0)


def _install_in_memory_loader() -> None:
    """Teach SAM2's frame loader to accept a clip we already hold.

    Patching rather than reimplementing init_state: everything else that call
    does is bookkeeping this module has no business copying, and a future SAM2
    that adds a field would silently lose it.
    """
    import sam2.sam2_video_predictor as predictor_module

    if getattr(predictor_module, "_warrioriq_in_memory_loader", False):
        return
    original = predictor_module.load_video_frames

    def load_video_frames(video_path, image_size, offload_video_to_cpu,
                          img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225),
                          async_loading_frames=False, compute_device=None):
        if not isinstance(video_path, _DecodedClip):
            return original(
                video_path=video_path, image_size=image_size,
                offload_video_to_cpu=offload_video_to_cpu, img_mean=img_mean,
                img_std=img_std, async_loading_frames=async_loading_frames,
                compute_device=compute_device if compute_device is not None else torch.device("cuda"),
            )
        images = video_path.taken()
        mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        if not offload_video_to_cpu and compute_device is not None:
            images = images.to(compute_device)
            mean, std = mean.to(compute_device), std.to(compute_device)
        # In place: the clip is finished with, and a copy of this tensor is
        # gigabytes on a long chunk.
        images = images.sub_(mean).div_(std)
        return images, video_path.height, video_path.width

    predictor_module.load_video_frames = load_video_frames
    predictor_module._warrioriq_in_memory_loader = True


def _silence_progress_bar() -> None:
    """Stop SAM2 drawing a progress bar nobody reads.

    `propagate_in_video` wraps its frame loop in `tqdm(..., desc="propagate in
    video")`. Under the worker nothing is attached to stdout - the output goes
    to a pipe or a log - so every refresh is a blocking write that renders for
    no one.

    **It is not free, and it is not small.** Profiling a real 1080p bout
    (2026-09-14, 156 analysed frames, 110 s total) put `tqdm` at **5.3 s, or
    4.9% of the entire analysis** - 81 refreshes at 66 ms each. For comparison
    the YOLO pose detector, the part everyone assumes is the cost, was 4.2 s.
    A progress bar was more expensive than the pose model.

    Patched at the module SAM2 resolves it from, so the loop still iterates
    exactly the same objects in the same order and only the drawing is gone.
    """
    import sam2.sam2_video_predictor as predictor_module

    if getattr(predictor_module, "_warrioriq_quiet_progress", False):
        return
    predictor_module.tqdm = lambda iterable=None, *args, **kwargs: (
        iterable if iterable is not None else ()
    )
    predictor_module._warrioriq_quiet_progress = True


LOGGER = logging.getLogger(__name__)

# One built predictor kept for the life of the worker process, so a job does
# not pay for building SAM2 again (core/config.py sam_keep_warm). Between uses
# it waits in host memory: the frame pass after the sweep needs the card's
# memory, which is why release() existed. Taken out by one user at a time.
_WARM_LOCK = threading.Lock()
_warm: dict = {"model_id": None, "predictor": None}


def _build_predictor():
    from sam2.build_sam import build_sam2_video_predictor_hf

    return build_sam2_video_predictor_hf(SETTINGS.sam_model_id, device="cuda")


def _take_warm():
    """The parked predictor on the card, or None when there is none to reuse."""
    with _WARM_LOCK:
        predictor, model_id = _warm["predictor"], _warm["model_id"]
        _warm["predictor"] = _warm["model_id"] = None
    if predictor is None or model_id != SETTINGS.sam_model_id:
        return None
    try:
        return predictor.to("cuda")
    except Exception as exc:  # noqa: BLE001 - a failed move means rebuild, as before
        LOGGER.warning("sam_warm_reuse_failed error=%s", type(exc).__name__)
        return None


def _park(predictor) -> None:
    """Keep the predictor in host memory for the next use, freeing the card."""
    if predictor is None or not SETTINGS.sam_keep_warm:
        return
    try:
        predictor = predictor.to("cpu")
    except Exception as exc:  # noqa: BLE001 - cannot park it: drop it, as before
        LOGGER.warning("sam_park_failed error=%s", type(exc).__name__)
        return
    with _WARM_LOCK:
        _warm["predictor"], _warm["model_id"] = predictor, SETTINGS.sam_model_id


def preload() -> bool:
    """Build SAM2 while the worker is idle and park it, so the first job is warm."""
    if not (SETTINGS.sam_recovery_enabled and SETTINGS.sam_keep_warm and torch.cuda.is_available()):
        return False
    if str(SETTINGS.sam_backend).strip().lower() != "sam2":
        return False
    with _WARM_LOCK:
        if _warm["predictor"] is not None:
            return True
    started = time.perf_counter()
    _park(_build_predictor())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.info("sam_preload_complete seconds=%.2f", time.perf_counter() - started)
    return True


class SamRecovery:
    """Best-effort short-window SAM2.1 identity recovery.

    This module is deliberately lazy-loaded and is never used as the normal
    frame-by-frame tracker. It only propagates a known fighter box across a
    very short buffered window after the fast identity manager becomes
    uncertain.
    """

    def __init__(self):
        self.predictor = None
        self.available = None
        self.failure_reason = None
        self.continuous_frames = 0
        self.continuous_failure_reason = None
        # Told each frame the sweep finishes, in order (SweepInBackground).
        self.frame_done = None
        # Where the sweep's time goes (2026-10-13): on an RTX 5060 it took
        # 0.23 s a frame, more than the 0.173 s the stride planner assumes,
        # and the report could not say whether decoding, SAM2's own frame
        # loading, the propagation or copying masks back was the cost.
        self.sweep_seconds = {"decoding": 0.0, "loading_frames": 0.0, "propagation": 0.0, "mask_to_box": 0.0}
        # Seconds spent getting a predictor onto the card this run, every
        # load included (core/analyzer.py stage_seconds).
        self.load_seconds = 0.0

    def _load(self) -> bool:
        if self.available is not None:
            return self.available
        if not SETTINGS.sam_recovery_enabled:
            self.available = False
            self.failure_reason = "disabled"
            return False
        started = time.perf_counter()
        try:
            predictor = _take_warm() if SETTINGS.sam_keep_warm else None
            self.predictor = predictor if predictor is not None else _build_predictor()
            self.available = True
        except Exception as exc:  # optional dependency must never kill analysis
            self.available = False
            self.failure_reason = f"{type(exc).__name__}: {exc}"
        self.load_seconds += time.perf_counter() - started
        return bool(self.available)

    def release(self) -> None:
        """Free the card after continuous propagation, keeping the model warm.

        The predictor is parked in host memory (_park) rather than dropped, so
        the next load - a frame-pass rescue in this job, or the next job - moves
        it back instead of building SAM2 again.
        """
        predictor, self.predictor = self.predictor, None
        self.available = None
        _park(predictor)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _logits_to_box(logits: torch.Tensor) -> np.ndarray | None:
        """_mask_to_box, worked out where the mask already is.

        2026-10-20, the owner's RTX 5060: turning masks into boxes was 7.4 s
        of a 37 s sweep - 0.1 s a mask - because each full-resolution mask was
        copied off the card and every pixel searched with numpy. On the card
        the same answer is four numbers and a count: the same box, the same
        "fewer than four pixels is no box" rule.
        """
        mask = (logits.detach() > 0.0).squeeze()
        if mask.ndim != 2:
            return None
        rows = torch.nonzero(mask.any(dim=1)).flatten()
        cols = torch.nonzero(mask.any(dim=0)).flatten()
        if rows.numel() == 0:
            return None
        found = torch.stack([mask.sum(), cols[0], rows[0], cols[-1], rows[-1]]).cpu().numpy()
        if int(found[0]) < 4:
            return None
        return found[1:].astype(np.float32)

    @staticmethod
    def _mask_to_box(mask) -> np.ndarray | None:
        array = np.squeeze(mask)
        if array.ndim != 2:
            return None
        ys, xs = np.where(array)
        if len(xs) < 4 or len(ys) < 4:
            return None
        return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    def recover(self, frames: list[np.ndarray], seed_box) -> np.ndarray | None:
        if not frames or seed_box is None or not self._load():
            return None
        # The other propagating path. Same reasoning as in track_segment.
        _silence_progress_bar()

        # Bound recovery work even if the caller supplied a larger buffer.
        frames = frames[-SETTINGS.sam_buffer_frames :]
        temp_dir = Path(tempfile.mkdtemp(prefix="warrioriq_sam_recovery_"))
        try:
            for i, frame in enumerate(frames):
                path = temp_dir / f"{i:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    return None

            state = self.predictor.init_state(
                video_path=str(temp_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
                async_loading_frames=True,
            )
            try:
                self.predictor.reset_state(state)
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=np.asarray(seed_box, dtype=np.float32),
                )

                final_box = None
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        for _, object_ids, mask_logits in self.predictor.propagate_in_video(state):
                            for j, object_id in enumerate(object_ids):
                                if int(object_id) != 1:
                                    continue
                                mask = (mask_logits[j] > 0.0).detach().cpu().numpy()
                                box = self._mask_to_box(mask)
                                if box is not None:
                                    final_box = box
                return final_box
            finally:
                try:
                    self.predictor.reset_state(state)
                except Exception:
                    pass
                del state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            return None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _propagate_clip(self, clip, source_frames, seeds, tracks,
                        progress_callback, completed, total) -> None:
        """Run one chunk and carry its last good boxes forward as the next seed."""
        timing = self.sweep_seconds
        loading_started = time.perf_counter()
        state = self.predictor.init_state(
            video_path=clip,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        timing["loading_frames"] += time.perf_counter() - loading_started
        propagation_started = time.perf_counter()
        boxing = 0.0
        try:
            self.predictor.reset_state(state)
            for object_id, name in ((1, "A"), (2, "B")):
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=object_id,
                    box=seeds[name],
                )
            last_guided: dict[str, np.ndarray] = {}
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for frame_index, object_ids, mask_logits in self.predictor.propagate_in_video(state):
                    if not (0 <= int(frame_index) < len(source_frames)):
                        continue
                    guided: dict[str, np.ndarray] = {}
                    boxing_started = time.perf_counter()
                    for j, object_id in enumerate(object_ids):
                        box = self._logits_to_box(mask_logits[j])
                        if box is not None:
                            guided["A" if int(object_id) == 1 else "B"] = box
                    boxing += time.perf_counter() - boxing_started
                    if guided:
                        tracks[source_frames[int(frame_index)]] = guided
                        last_guided = guided
                    if self.frame_done is not None:
                        self.frame_done(source_frames[int(frame_index)], guided or None)
                    if progress_callback is not None and int(frame_index) % 15 == 0:
                        progress_callback(min(total, completed + int(frame_index) + 1), total)
            for name in ("A", "B"):
                if name in last_guided:
                    seeds[name] = last_guided[name]
        finally:
            # The first .cpu() of a frame waits for the GPU, so "mask_to_box"
            # includes that frame's GPU work; "propagation" is the rest.
            timing["mask_to_box"] += boxing
            timing["propagation"] += time.perf_counter() - propagation_started - boxing
            try:
                self.predictor.reset_state(state)
            except Exception:
                pass
            del state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def track_segment(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        source_fps: float,
        fighter_a_box,
        fighter_b_box,
        progress_callback=None,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Propagate both selected fighter masks through a sampled segment.

        Returned boxes are identity guidance only. Downstream code still
        requires a detector/pose observation before counting a visible frame.

        Each chunk is propagated as soon as it is full rather than writing the
        whole segment out first. The decode is one forward pass either way, and
        only one chunk is ever resident - which is what the folder of JPEGs was
        buying, at the price of encoding every frame, writing it to disk and
        having SAM2 decode it again.
        """
        if not SETTINGS.sam_continuous_enabled or not self._load():
            return {}
        _install_in_memory_loader()
        _silence_progress_bar()
        self.sweep_seconds = {key: 0.0 for key in self.sweep_seconds}
        start_frame, end_frame = int(start_frame), int(end_frame)
        stride = sam_sampling_stride(source_fps, end_frame - start_frame)
        chunk_frames = max(1, SETTINGS.sam_continuous_chunk_frames)
        # Only used to size chunks and to drive the progress bar. Frames can
        # run out early on a truncated file, which shortens the last chunk.
        expected = max(0, (end_frame - start_frame + stride - 1) // stride)
        image_size = int(getattr(self.predictor, "image_size", 1024) or 1024)
        tracks: dict[int, dict[str, np.ndarray]] = {}
        seeds = {"A": np.asarray(fighter_a_box, dtype=np.float32),
                 "B": np.asarray(fighter_b_box, dtype=np.float32)}
        cap = open_capture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("Could not open video for SAM2 propagation")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            saved = 0
            completed = 0
            clip = None
            chunk_sources: list[int] = []
            frames = _SampledFrames(cap, start_frame, end_frame, stride, image_size)
            try:
                # "decoding" is now the time the sweep waited for its next
                # frame; decoding that overlapped propagation costs nothing.
                waiting_since = time.perf_counter()
                for source, prepared, shape in frames:
                    self.sweep_seconds["decoding"] += time.perf_counter() - waiting_since
                    if clip is None:
                        capacity = min(chunk_frames, max(1, expected - saved))
                        clip = _DecodedClip(image_size, capacity, height or shape[0], width or shape[1])
                        chunk_sources = []
                    clip.add_prepared(prepared)
                    chunk_sources.append(source)
                    saved += 1
                    if clip.full:
                        self._propagate_clip(clip, chunk_sources, seeds, tracks,
                                             progress_callback, completed, expected)
                        completed += len(chunk_sources)
                        clip, chunk_sources = None, []
                    waiting_since = time.perf_counter()
            finally:
                frames.close()
            if clip is not None and clip.count:
                self._propagate_clip(clip, chunk_sources, seeds, tracks,
                                     progress_callback, completed, expected)
            self.continuous_frames = len(tracks)
            return tracks
        except Exception as exc:
            self.continuous_failure_reason = f"{type(exc).__name__}: {exc}"
            return {}
        finally:
            cap.release()


def nearest_guidance(
    tracks: dict[int, dict[str, np.ndarray]],
    source_frame: int,
    tolerance: int,
) -> dict[str, np.ndarray] | None:
    """Return the closest propagated masks without bridging a large gap."""
    if not tracks:
        return None
    nearest = min(tracks, key=lambda frame: abs(frame - source_frame))
    return tracks[nearest] if abs(nearest - source_frame) <= max(0, tolerance) else None


class SweepInBackground:
    """The continuous sweep on a helper thread while the frame pass runs.

    2026-10-21, the owner's RTX 5060: a 197 s fight took 200 s - the sweep
    75 s, then the frame pass 120 s, one after the other. The frame pass keeps
    the card busy about 40% of each frame; the rest is CPU work. Running the
    sweep beside it fills that gap.

    The frame pass gets exactly the guidance it would have got: guidance()
    waits until the sweep has finished every frame up to `source_frame +
    tolerance`, so nearest_guidance sees every key that could be the answer
    (keys further on are further than the tolerance). Frames are finished in
    order. When the sweep is over, the answer comes from its returned tracks
    - which are empty if it failed, as they were when it ran first.

    The predictor is released on the sweep's own thread before the sweep is
    marked over, so a frame-pass rescue (only after an empty sweep) never
    meets a running sweep. One difference from running first, and only on a
    sweep that fails part-way or finds nothing at all: frames analysed before
    that was known used the guidance there was, and the fallback frame buffer
    for rescues was not kept.
    """

    def __init__(self, recovery: SamRecovery, video_path: str, start_frame: int, end_frame: int,
                 source_fps: float, fighter_a_box, fighter_b_box):
        self._recovery = recovery
        self._condition = threading.Condition()
        self._tracks: dict[int, dict[str, np.ndarray]] = {}
        self._reached = int(start_frame) - 1
        self._done = False
        self.result: dict[int, dict[str, np.ndarray]] = {}
        self.available: bool | None = None
        self.breakdown: dict[str, float] = {}
        self.frames = 0
        self.seconds = 0.0
        self.waited = 0.0
        recovery.frame_done = self._frame_done
        self._thread = threading.Thread(
            target=self._run, name="warrioriq-sam-sweep", daemon=True,
            args=(video_path, start_frame, end_frame, source_fps, fighter_a_box, fighter_b_box))
        self._thread.start()

    def _frame_done(self, source_frame: int, guided) -> None:
        with self._condition:
            if guided:
                self._tracks[int(source_frame)] = guided
            self._reached = max(self._reached, int(source_frame))
            self._condition.notify_all()

    def _run(self, *args) -> None:
        started = time.perf_counter()
        result: dict = {}
        try:
            result = self._recovery.track_segment(*args) or {}
        except Exception as exc:  # noqa: BLE001 - the frame pass must still finish
            self._recovery.continuous_failure_reason = f"{type(exc).__name__}: {exc}"
            result = {}
        finally:
            self._recovery.frame_done = None
            available = self._recovery.available
            breakdown = dict(getattr(self._recovery, "sweep_seconds", {}) or {})
            frames = int(getattr(self._recovery, "continuous_frames", 0) or 0)
            try:
                self._recovery.release()
            finally:
                with self._condition:
                    self.result = result
                    self.available, self.breakdown, self.frames = available, breakdown, frames
                    self.seconds = time.perf_counter() - started
                    self._done = True
                    self._condition.notify_all()

    def guidance(self, source_frame: int, tolerance: int) -> dict[str, np.ndarray] | None:
        """nearest_guidance, once every frame that could be the answer is finished."""
        started = time.perf_counter()
        target = int(source_frame) + max(0, int(tolerance))
        with self._condition:
            self._condition.wait_for(lambda: self._done or self._reached >= target)
            self.waited += time.perf_counter() - started
            return nearest_guidance(self.result if self._done else self._tracks, source_frame, tolerance)

    def finished_empty(self) -> bool:
        """The sweep is over and found nothing: what "no tracks" meant before."""
        with self._condition:
            return self._done and not self.result

    def join(self) -> dict[int, dict[str, np.ndarray]]:
        self._thread.join()
        return self.result
