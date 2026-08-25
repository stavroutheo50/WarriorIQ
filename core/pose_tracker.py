from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from core.config import SETTINGS
from ultralytics import YOLO
from core.identity import appearance_hist, pose_signature
from core.types import PersonObservation


class QualityController:
    """Adaptive analysis quality while respecting the <= video-length target."""

    def __init__(self, source_fps: float):
        self.source_fps = max(1.0, float(source_fps))
        self.target_fps = min(self.source_fps, SETTINGS.target_tracking_fps)
        self.min_fps = min(self.source_fps, SETTINGS.min_tracking_fps)
        self.max_fps = min(self.source_fps, SETTINGS.max_tracking_fps)
        self.stride = max(1, round(self.source_fps / self.target_fps))
        self.imgsz = SETTINGS.default_imgsz
        self.mode = "balanced"
        self.last_adjust = 0

    @property
    def effective_fps(self) -> float:
        return self.source_fps / self.stride

    def maybe_adjust(self, analyzed_index: int, processed_seconds: float, elapsed_seconds: float) -> tuple[int, int, str]:
        if not SETTINGS.adaptive_quality or elapsed_seconds < 2.0 or analyzed_index - self.last_adjust < 60:
            return self.stride, self.imgsz, self.mode

        speed = processed_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
        fps = self.effective_fps

        if speed < 0.92:
            # Behind budget: reduce expensive inference before compromising
            # identity safeguards.
            fps = max(self.min_fps, fps * 0.82)
            self.imgsz = max(SETTINGS.min_imgsz, self.imgsz - 64)
            self.mode = "deadline"
        elif speed < 1.08:
            fps = max(self.min_fps, fps * 0.92)
            self.imgsz = max(SETTINGS.min_imgsz, self.imgsz - 32)
            self.mode = "economy"
        elif speed > 1.65:
            fps = min(self.max_fps, fps * 1.12)
            self.imgsz = min(SETTINGS.default_imgsz, self.imgsz + 32)
            self.mode = "high"
        elif speed > 1.20:
            self.mode = "balanced"
        else:
            self.mode = "balanced"

        self.stride = max(1, round(self.source_fps / max(self.min_fps, fps)))
        self.last_adjust = analyzed_index
        return self.stride, self.imgsz, self.mode


class PoseTracker:
    def __init__(self):
        requested = str(SETTINGS.device).strip().lower()
        self.device = (0 if torch.cuda.is_available() else "cpu") if requested == "auto" else (int(requested) if requested.isdigit() else requested)
        self.uses_cuda = self.device != "cpu" and torch.cuda.is_available()
        engine_path = Path(SETTINGS.pose_model_engine)
        model_path = str(engine_path) if self.uses_cuda and engine_path.exists() else SETTINGS.pose_model_pt
        self.model_path = model_path
        self.model = YOLO(model_path)
        self._focus_model = None
        self._warmed = False

    def warmup(self, frame) -> None:
        if self._warmed:
            return
        try:
            _ = self.model.predict(
                frame,
                device=self.device,
                imgsz=SETTINGS.default_imgsz if self.uses_cuda else min(416, SETTINGS.default_imgsz),
                conf=SETTINGS.detection_conf,
                classes=[0],
                verbose=False,
            )
        except Exception:
            if not self.model_path.endswith(".engine"):
                raise
            # TensorRT engines are tied to compatible NVIDIA runtimes. Fall
            # back to the original PyTorch checkpoint on another device.
            self.model_path = SETTINGS.pose_model_pt
            self.model = YOLO(self.model_path)
            self._focus_model = None
            _ = self.model.predict(
                frame,
                device=self.device,
                imgsz=SETTINGS.default_imgsz if self.uses_cuda else min(416, SETTINGS.default_imgsz),
                conf=SETTINGS.detection_conf,
                classes=[0],
                verbose=False,
            )
        self._warmed = True

    def reset_tracking(self) -> None:
        """Best-effort reset of Ultralytics tracker state between fights.

        The YOLO model is cached so we do not pay model-loading cost for every
        fight, but BoT-SORT state must never leak from one uploaded video into
        the next. Ultralytics internals can vary by version, so this deliberately
        uses feature checks rather than depending on one private layout.
        """
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return

        trackers = getattr(predictor, "trackers", None)
        if trackers is not None:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception:
                        pass

        # Some Ultralytics tracker callbacks use vid_path to decide whether a
        # source changed. Clearing it prevents a cached ndarray source from
        # inheriting identity state from the previous analysis.
        if hasattr(predictor, "vid_path"):
            try:
                current = predictor.vid_path
                predictor.vid_path = [None] * len(current) if isinstance(current, (list, tuple)) else None
            except Exception:
                pass

    def track(self, frame, imgsz: int | None = None) -> list[PersonObservation]:
        size = int(imgsz or SETTINGS.default_imgsz)
        results = self.model.track(
            frame,
            persist=True,
            tracker=SETTINGS.tracker,
            device=self.device,
            imgsz=size,
            conf=SETTINGS.detection_conf,
            classes=[0],
            verbose=False,
        )
        return self.parse(results[0], frame)

    def recover_from_guidance(
        self,
        frame: np.ndarray,
        guidance: dict[str, np.ndarray] | None,
        existing: list[PersonObservation],
    ) -> list[PersonObservation]:
        """Run focused pose inference where SAM sees a fighter YOLO missed."""
        if not guidance:
            return []
        from core.identity import box_iou

        height, width = frame.shape[:2]
        requests: list[tuple[str, np.ndarray, int, int, np.ndarray]] = []
        for name in ("A", "B"):
            guide = guidance.get(name)
            if guide is None or any(box_iou(guide, person.box) >= 0.12 for person in existing):
                continue
            x1, y1, x2, y2 = map(float, guide)
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx1 = max(0, int(x1 - 0.45 * bw))
            cy1 = max(0, int(y1 - 0.30 * bh))
            cx2 = min(width, int(x2 + 0.45 * bw))
            cy2 = min(height, int(y2 + 0.30 * bh))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.shape[0] >= 20 and crop.shape[1] >= 12:
                requests.append((name, guide, cx1, cy1, crop))
        if not requests:
            return []

        if self._focus_model is None:
            # A separate predictor is essential: predict() on the persistent
            # tracking model changes its internal source geometry and breaks
            # BoT-SORT camera-motion state on the next full frame.
            self._focus_model = YOLO(self.model_path)
        results = self._focus_model.predict(
            [request[4] for request in requests],
            device=self.device,
            imgsz=384 if self.uses_cuda else 320,
            conf=max(0.10, SETTINGS.detection_conf * 0.65),
            classes=[0],
            verbose=False,
        )
        recovered: list[PersonObservation] = []
        for (name, guide, offset_x, offset_y, crop), result in zip(requests, results):
            candidates = self.parse(result, crop)
            best = None
            best_score = -1.0
            for candidate in candidates:
                candidate.box[[0, 2]] += offset_x
                candidate.box[[1, 3]] += offset_y
                if candidate.keypoints is not None:
                    valid = (candidate.keypoints[:, 0] > 0) & (candidate.keypoints[:, 1] > 0)
                    candidate.keypoints[valid, 0] += offset_x
                    candidate.keypoints[valid, 1] += offset_y
                score = 0.75 * box_iou(guide, candidate.box) + 0.25 * candidate.confidence
                if score > best_score:
                    best, best_score = candidate, score
            if best is not None and best_score >= 0.18:
                best.track_id = -1001 if name == "A" else -1002
                best.appearance = appearance_hist(frame, best.box)
                best.pose_signature = pose_signature(best.keypoints, best.box)
                recovered.append(best)
        return recovered

    @staticmethod
    def parse(result, frame) -> list[PersonObservation]:
        people: list[PersonObservation] = []
        if result.boxes is None or len(result.boxes) == 0:
            return people

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        ids = None
        if result.boxes.id is not None:
            ids = result.boxes.id.detach().cpu().numpy().astype(int)

        keypoints_xy = None
        keypoints_conf = None
        if result.keypoints is not None:
            if result.keypoints.xy is not None:
                keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
            if result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.detach().cpu().numpy()

        for i, box in enumerate(boxes):
            kp = np.asarray(keypoints_xy[i], dtype=np.float32) if keypoints_xy is not None else None
            obs = PersonObservation(
                track_id=int(ids[i]) if ids is not None and i < len(ids) else None,
                box=np.asarray(box, dtype=np.float32),
                confidence=float(confs[i]),
                keypoints=kp,
                keypoint_conf=np.asarray(keypoints_conf[i], dtype=np.float32) if keypoints_conf is not None else None,
            )
            obs.appearance = appearance_hist(frame, obs.box)
            obs.pose_signature = pose_signature(kp, obs.box)
            people.append(obs)
        return people


def find_initial_people(manual_a, manual_b, people: list[PersonObservation], frame=None):
    from core.identity import box_iou, normalized_distance

    def match(selection, candidates, excluded=None):
        available = [person for person in candidates if person is not excluded]
        if not available:
            return None, -1.0
        ranked = sorted(
            available,
            key=lambda person: (box_iou(selection, person.box), -normalized_distance(selection, person.box)),
            reverse=True,
        )
        best = ranked[0]
        overlap = box_iou(selection, best.box)
        # The first identity decision is irreversible enough to seed SAM2 and
        # every later pose/action result.  Centre distance alone is unsafe in a
        # fight because the referee often stands immediately beside or between
        # the selected fighters.  A detector candidate therefore has to agree
        # with a meaningful area of the user's box; otherwise the exact manual
        # selection becomes the temporary anchor.
        return (best, overlap) if overlap >= SETTINGS.min_initial_iou else (None, overlap)

    def manual_anchor(selection):
        """Create a temporary identity anchor when first-frame detection misses.

        The user-confirmed box is stronger evidence than aborting the fight.
        A later tracked observation can acquire a real BoT-SORT ID through the
        normal appearance/position recovery path.
        """
        box = np.asarray(selection, dtype=np.float32)
        return PersonObservation(
            track_id=None,
            box=box,
            confidence=0.55,
            keypoints=None,
            keypoint_conf=None,
            appearance=appearance_hist(frame, box) if frame is not None else None,
            pose_signature=None,
        )

    best_a, iou_a = match(manual_a, people)
    best_b, iou_b = match(manual_b, people, best_a)

    if best_a is None:
        best_a, iou_a = manual_anchor(manual_a), 0.0
    if best_b is None:
        best_b, iou_b = manual_anchor(manual_b), 0.0
    if best_a.track_id is not None and best_b.track_id is not None and best_a.track_id == best_b.track_id:
        raise RuntimeError("Fighter A and B received the same initial tracker ID.")
    return best_a, best_b, float(iou_a), float(iou_b)
