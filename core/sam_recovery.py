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
        """
        if not SETTINGS.sam_continuous_enabled or not self._load():
            return {}
        stride = sam_sampling_stride(source_fps, end_frame - start_frame)
        temp_dir = Path(tempfile.mkdtemp(prefix="warrioriq_sam_primary_"))
        frame_chunks: list[tuple[Path, list[int]]] = []
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("Could not open video for SAM2 propagation")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            source = start_frame
            saved = 0
            while source < end_frame:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if (source - start_frame) % stride == 0:
                    chunk_index = saved // max(1, SETTINGS.sam_continuous_chunk_frames)
                    local_index = saved % max(1, SETTINGS.sam_continuous_chunk_frames)
                    if chunk_index >= len(frame_chunks):
                        chunk_dir = temp_dir / f"chunk_{chunk_index:04d}"
                        chunk_dir.mkdir(parents=True, exist_ok=True)
                        frame_chunks.append((chunk_dir, []))
                    chunk_dir, chunk_sources = frame_chunks[chunk_index]
                    if not cv2.imwrite(str(chunk_dir / f"{local_index:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]):
                        raise RuntimeError("Could not prepare frames for SAM2")
                    chunk_sources.append(source)
                    saved += 1
                source += 1
            if not frame_chunks:
                return {}
            tracks: dict[int, dict[str, np.ndarray]] = {}
            seeds = {"A": np.asarray(fighter_a_box, dtype=np.float32), "B": np.asarray(fighter_b_box, dtype=np.float32)}
            total = sum(len(sources) for _, sources in frame_chunks)
            completed = 0
            for chunk_dir, source_frames in frame_chunks:
                state = self.predictor.init_state(
                    video_path=str(chunk_dir),
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
                                progress_callback(completed + int(frame_index) + 1, total)
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
                completed += len(source_frames)
            self.continuous_frames = len(tracks)
            return tracks
        except Exception as exc:
            self.continuous_failure_reason = f"{type(exc).__name__}: {exc}"
            return {}
        finally:
            cap.release()
            shutil.rmtree(temp_dir, ignore_errors=True)


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
