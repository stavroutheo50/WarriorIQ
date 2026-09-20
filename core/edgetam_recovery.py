"""EdgeTAM as an alternative to SAM2 for the continuous identity sweep.

SAM2's sweep is the largest single cost in a short analysis - 53.8% of a
70-second round - and the obvious saving, a smaller SAM2 checkpoint, was
measured and is nearly not there: `sam2.1-hiera-tiny` against `-small` is
1.01x -> 0.97x end to end, about 12% off SAM2's own time.

EdgeTAM (arXiv 2501.07256, CVPR 2025, Apache 2.0) explains why, and it is the
reason this module exists rather than another checkpoint swap: *"they all focus
on compressing the image encoder, while our benchmark shows that the newly
introduced memory attention blocks are also the latency bottleneck."* It
replaces the dense frame memory with a 2D Spatial Perceiver - a fixed set of
learned queries, split into global and patch-level groups - so cost stops
scaling with how much memory has accumulated.

**Measured here, same fight, same seeds, same sampled frames:**

    SAM2      0.197 s/frame
    EdgeTAM   0.078 s/frame     2.5x faster, both fighters held on 60/60

That is meaningful and it is not the paper's 22x, which is an iPhone figure
against SAM2 on the same phone. On a desktop GPU the encoder is not the
bottleneck the perceiver removes, so expect this ratio rather than that one.

## Why this is a separate class and not a branch inside SamRecovery

EdgeTAM's own repository is a fork of SAM2 and installs over the `sam2` package
this project pins, which would replace the working path to add an experimental
one. The HuggingFace port does not: `EdgeTamVideoModel` lives in `transformers`
and shares nothing with `sam2` at runtime, so both can be installed at once and
either can be selected.

The cost is that the API is genuinely different - `init_video_session` /
`add_inputs_to_inference_session` / `propagate_in_video_iterator` rather than
`init_state` / `add_new_points_or_box` / `propagate_in_video` - so this is a
reimplementation of the sweep, not a checkpoint string. It takes frames as
arrays directly, which means the `load_video_frames` monkeypatch and the
JPEG round trip that `core/sam_recovery.py` exists partly to avoid are simply
not needed on this path.

`track_segment` returns the same thing SamRecovery's does and means the same
thing by it: **identity guidance only.** Downstream still requires a detector
or pose observation overlapping the box before a frame counts as visible.

Selected with `WARRIORIQ_SAM_BACKEND=edgetam`. SAM2 remains the default until
this is measured on identity as well as on speed.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from core.config import SETTINGS
from core.sam_recovery import sam_sampling_stride

LOGGER = logging.getLogger("warrioriq.edgetam")


class EdgeTamRecovery:
    """The continuous sweep, on EdgeTAM. Same contract as SamRecovery's."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.available: bool | None = None
        self.failure_reason: str | None = None
        # The report reads both of these off whichever backend ran, so they
        # are part of the contract rather than SAM2 bookkeeping.
        self.continuous_frames = 0
        self.continuous_failure_reason: str | None = None

    def _load(self) -> bool:
        if self.available is not None:
            return bool(self.available)
        if not SETTINGS.sam_recovery_enabled:
            self.available = False
            self.failure_reason = "disabled"
            return False
        try:
            from transformers import EdgeTamVideoModel, Sam2VideoProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = EdgeTamVideoModel.from_pretrained(
                SETTINGS.edgetam_model_id).to(device).eval()
            self.processor = Sam2VideoProcessor.from_pretrained(SETTINGS.edgetam_model_id)
            self.available = True
            LOGGER.info("edgetam_loaded model=%s device=%s", SETTINGS.edgetam_model_id, device)
        except Exception as exc:  # optional dependency must never kill analysis
            self.available = False
            self.failure_reason = "%s: %s" % (type(exc).__name__, exc)
            LOGGER.warning("edgetam_unavailable %s", self.failure_reason)
        return bool(self.available)

    def release(self) -> None:
        self.model = None
        self.processor = None
        self.available = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _mask_to_box(mask) -> np.ndarray | None:
        array = np.squeeze(np.asarray(mask))
        if array.ndim != 2:
            return None
        rows = np.any(array > 0, axis=1)
        cols = np.any(array > 0, axis=0)
        if not rows.any() or not cols.any():
            return None
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return np.asarray([x1, y1, x2, y2], dtype=np.float32)

    def recover(self, frames, last_box):
        """Not implemented on this backend; the continuous sweep replaces it.

        The on-demand rescue only runs when the sweep produced nothing
        (`not sam_tracks` in core/analyzer.py), so returning None here means a
        failed sweep degrades to no guidance rather than to a different
        backend's guidance - which is the honest failure, not a silent one.
        """
        return None

    def track_segment(self, video_path: str, start_frame: int, end_frame: int,
                      source_fps: float, fighter_a_box, fighter_b_box,
                      progress_callback=None) -> dict[int, dict[str, np.ndarray]]:
        tracks: dict[int, dict[str, np.ndarray]] = {}
        if not SETTINGS.sam_continuous_enabled or not self._load():
            return tracks

        stride = sam_sampling_stride(source_fps, max(0, end_frame - start_frame))
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return tracks
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[np.ndarray] = []
        source_frames: list[int] = []
        index = start_frame
        budget = max(1, int(SETTINGS.sam_continuous_max_frames))
        while index < end_frame and len(frames) < budget:
            # grab() decodes; retrieve() converts YUV to BGR into a fresh
            # buffer. At stride 16 this loop walks fifteen frames for every one
            # it keeps, and calling read() on all of them pays the conversion
            # fifteen times over for a buffer that is dropped on the next line.
            # core/analyzer.py's own decode loop makes this split and measured
            # it at -61% on 1080p; the same argument applies here and the sweep
            # samples far more sparsely than the pose pass does.
            if not capture.grab():
                break
            if (index - start_frame) % max(1, stride) == 0:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                source_frames.append(index)
            index += 1
        capture.release()
        if len(frames) < 2:
            return tracks

        seeds = [[float(v) for v in np.asarray(fighter_a_box, dtype=np.float32)],
                 [float(v) for v in np.asarray(fighter_b_box, dtype=np.float32)]]
        try:
            session = self.processor.init_video_session(
                video=frames,
                inference_device=("cuda" if torch.cuda.is_available() else "cpu"),
                # Mirrors SAM2's offload_video_to_cpu=True: the frame buffer is
                # the large allocation and it does not belong on an 8 GB card
                # that is also holding a TensorRT pose context.
                video_storage_device="cpu",
                dtype=torch.float32)
            self.processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=0, obj_ids=[1, 2],
                input_boxes=[seeds])

            with torch.inference_mode():
                # Segment the prompted frame before propagating. Without this
                # the iterator cannot work out where to start and raises.
                self.model(inference_session=session, frame_idx=0)
                for output in self.model.propagate_in_video_iterator(session):
                    position = int(output.frame_idx)
                    if not (0 <= position < len(source_frames)):
                        continue
                    masks = self.processor.post_process_masks(
                        [output.pred_masks],
                        original_sizes=[[session.video_height, session.video_width]],
                        binarize=True)[0]
                    guided: dict[str, np.ndarray] = {}
                    for slot, object_id in enumerate(session.obj_ids):
                        box = self._mask_to_box(masks[slot].detach().cpu().numpy())
                        if box is not None:
                            guided["A" if int(object_id) == 1 else "B"] = box
                    if guided:
                        tracks[source_frames[position]] = guided
                    if progress_callback is not None and position % 15 == 0:
                        progress_callback(min(len(frames), position + 1), len(frames))
        except Exception as exc:  # noqa: BLE001 - guidance must never kill a run
            self.continuous_failure_reason = "%s: %s" % (type(exc).__name__, exc)
            LOGGER.warning("edgetam_sweep_failed error=%s detail=%s",
                           type(exc).__name__, str(exc)[:200])
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if progress_callback is not None:
            progress_callback(len(frames), len(frames))
        self.continuous_frames = len(tracks)
        return tracks


def build_recovery():
    """The sweep backend this run should use. SAM2 unless asked otherwise."""
    from core.sam_recovery import SamRecovery

    if str(SETTINGS.sam_backend).strip().lower() == "edgetam":
        LOGGER.info("sam_backend=edgetam")
        return EdgeTamRecovery()
    return SamRecovery()
