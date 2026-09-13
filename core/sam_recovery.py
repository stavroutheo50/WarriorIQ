from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from core.config import SETTINGS


def sam_sampling_stride(source_fps: float, total_source_frames: int) -> int:
    """Bound SAM work by both target frequency and an absolute frame budget."""
    fps_stride = max(1, round(float(source_fps) / max(1.0, SETTINGS.sam_continuous_fps)))
    budget_stride = max(1, int(np.ceil(max(0, total_source_frames) / max(1, SETTINGS.sam_continuous_max_frames))))
    return max(fps_stride, budget_stride)


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

    def add(self, frame_bgr: np.ndarray) -> None:
        if self.full:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # PIL resizes with bicubic by default, which is what SAM2's loader got
        # from Image.resize, so the model sees the filter it was tuned on.
        resized = cv2.resize(rgb, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_CUBIC)
        frame = torch.from_numpy(np.ascontiguousarray(resized))
        self.images[self.count] = frame.permute(2, 0, 1).float().div_(255.0)
        self.count += 1

    def taken(self) -> torch.Tensor:
        return self.images[: self.count]


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

    def _load(self) -> bool:
        if self.available is not None:
            return self.available
        if not SETTINGS.sam_recovery_enabled:
            self.available = False
            self.failure_reason = "disabled"
            return False
        try:
            from sam2.build_sam import build_sam2_video_predictor_hf

            self.predictor = build_sam2_video_predictor_hf(
                SETTINGS.sam_model_id,
                device="cuda",
            )
            self.available = True
        except Exception as exc:  # optional dependency must never kill analysis
            self.available = False
            self.failure_reason = f"{type(exc).__name__}: {exc}"
        return bool(self.available)

    def release(self) -> None:
        """Release the large video predictor after continuous propagation."""
        self.predictor = None
        self.available = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        state = self.predictor.init_state(
            video_path=clip,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
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
                    for j, object_id in enumerate(object_ids):
                        box = self._mask_to_box((mask_logits[j] > 0.0).detach().cpu().numpy())
                        if box is not None:
                            guided["A" if int(object_id) == 1 else "B"] = box
                    if guided:
                        tracks[source_frames[int(frame_index)]] = guided
                        last_guided = guided
                    if progress_callback is not None and int(frame_index) % 15 == 0:
                        progress_callback(min(total, completed + int(frame_index) + 1), total)
            for name in ("A", "B"):
                if name in last_guided:
                    seeds[name] = last_guided[name]
        finally:
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
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("Could not open video for SAM2 propagation")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            source = start_frame
            saved = 0
            completed = 0
            clip = None
            chunk_sources: list[int] = []
            while source < end_frame:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if (source - start_frame) % stride == 0:
                    if clip is None:
                        capacity = min(chunk_frames, max(1, expected - saved))
                        clip = _DecodedClip(image_size, capacity,
                                            height or frame.shape[0], width or frame.shape[1])
                        chunk_sources = []
                    clip.add(frame)
                    chunk_sources.append(source)
                    saved += 1
                    if clip.full:
                        self._propagate_clip(clip, chunk_sources, seeds, tracks,
                                             progress_callback, completed, expected)
                        completed += len(chunk_sources)
                        clip, chunk_sources = None, []
                source += 1
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
