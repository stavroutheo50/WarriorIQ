"""A second look at each detected punch, from the video itself.

The action engine (core/action.py) proposes a punch whenever a wrist in the
pose skeleton moves fast enough, far enough, toward the opponent. The skeleton
cannot tell a punch from a hand reaching, a guard adjustment, an arm knocked
back by the opponent's punch, or a wrist the pose model guessed at - they move
the same joints the same way. Rules on the skeleton's own numbers top out near
half of counted punches being real, and a model fitted on those numbers did
not carry over to fighters it had not seen (2026-10-23, V18: it removed real
punches and no false ones).

This looks at the pixels. Sixteen frames around each proposed punch, cropped
to the credited fighter and widened toward the opponent, go through a video
action model pretrained to recognise human actions (torchvision r3d_18,
Kinetics-400). A small linear head on its features, fitted by
tools/fit_strike_check.py on proposals judged from the video, gives the
probability that a person watching would call it a punch by that fighter.

It only ever removes a proposal; it never creates a strike. When the model
cannot be loaded - no torchvision, no network for the first download, no head
file - nothing is scored and the analysis keeps its previous rules.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core.config import MODELS

LOGGER = logging.getLogger("warrioriq.strike_check")

HEAD_PATH = MODELS / "strike_check_head.npz"
FRAMES = 16
FRAME_STEP_SECONDS = 0.05
WINDOW_BEFORE_SECONDS = 0.4
CROP_SIZE = 112
_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
# Kinetics classes whose scores are kept beside the pooled features: the ones
# about striking and fighting, plus near-misses the model may confuse them with.
ACTION_WORDS = ("punch", "boxing", "kick", "wrestl", "fight", "martial", "tai chi", "dancing")

BoxesAt = Callable[[float], tuple]


def window_times(centre: float) -> list[float]:
    """The sixteen moments read for a proposal whose motion peaked at `centre`."""
    return [centre - WINDOW_BEFORE_SECONDS + FRAME_STEP_SECONDS * k for k in range(FRAMES)]


def crop_box(me, opponent, width: int, height: int) -> tuple[int, int, int, int]:
    """A square around the credited fighter, widened toward the opponent so a
    punch reaching them stays in the picture."""
    if me is not None:
        x1, y1, x2, y2 = (float(v) for v in me)
        if opponent is not None:
            opponent_x = (float(opponent[0]) + float(opponent[2])) / 2.0
            if opponent_x > x2:
                x2 = min(opponent_x, x2 + (x2 - x1))
            elif opponent_x < x1:
                x1 = max(opponent_x, x1 - (x2 - x1))
    elif opponent is not None:
        x1, y1, x2, y2 = (float(v) for v in opponent)
    else:
        x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)
    side = max(x2 - x1, y2 - y1) * 1.1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
            int(min(width, cx + side / 2)), int(min(height, cy + side / 2)))


def read_window(capture, times: list[float]) -> list[np.ndarray] | None:
    """The decoded frames nearest each of `times`, reading forward once."""
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, times[0] - 0.05) * 1000.0)
    decoded: list[tuple[float, np.ndarray]] = []
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        decoded.append((stamp, frame))
        if stamp >= times[-1] + 0.05:
            break
    if not decoded:
        return None
    stamps = np.array([stamp for stamp, _ in decoded])
    return [decoded[int(np.argmin(np.abs(stamps - t)))][1] for t in times]


def clip_array(frames: list[np.ndarray], times: list[float], boxes_at: BoxesAt) -> np.ndarray:
    """(3, 16, 112, 112) float32, normalised as the pretrained model expects."""
    crops = []
    for frame, moment in zip(frames, times):
        me, opponent = boxes_at(moment)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = crop_box(me, opponent, width, height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            x1, y1, x2, y2 = 0, 0, width, height
        crop = cv2.resize(frame[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    clip = (np.stack(crops).astype(np.float32) / 255.0 - _MEAN) / _STD     # T, H, W, C
    return np.ascontiguousarray(clip.transpose(3, 0, 1, 2))                 # C, T, H, W


class _Backbone:
    """The pretrained video model, loaded once per process."""

    def __init__(self) -> None:
        import torch
        import torchvision

        weights = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torchvision.models.video.r3d_18(weights=weights).eval().to(self.device)
        categories = weights.meta["categories"]
        self.kept = [i for i, name in enumerate(categories) if any(word in name for word in ACTION_WORDS)]
        self._pooled = None
        self.model.avgpool.register_forward_hook(lambda _m, _i, out: setattr(self, "_pooled", out.flatten(1)))

    def features(self, clips: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(clips).to(self.device))
            scores = torch.softmax(logits, dim=1)[:, self.kept]
            return torch.cat([self._pooled, scores], dim=1).float().cpu().numpy()


_backbone: _Backbone | None = None
_backbone_failed = False


def backbone() -> _Backbone | None:
    global _backbone, _backbone_failed
    if _backbone is None and not _backbone_failed:
        try:
            _backbone = _Backbone()
        except Exception as exc:  # noqa: BLE001 - optional: the analysis keeps its rules
            _backbone_failed = True
            LOGGER.warning("strike_check_unavailable error=%s detail=%s", type(exc).__name__, str(exc)[:200])
    return _backbone


def clip_features(video_path: str, centres: list[float], boxes_for: list[BoxesAt],
                  batch_size: int = 16) -> list[np.ndarray | None]:
    """Backbone features for each proposal. Shared by the analysis and the fitting tool."""
    from core.video import open_capture

    model = backbone()
    if model is None:
        return [None] * len(centres)
    capture = open_capture(video_path)
    out: list[np.ndarray | None] = [None] * len(centres)
    try:
        pending: list[tuple[int, np.ndarray]] = []
        order = sorted(range(len(centres)), key=lambda i: centres[i])
        for position, index in enumerate(order):
            times = window_times(centres[index])
            frames = read_window(capture, times)
            if frames is not None:
                pending.append((index, clip_array(frames, times, boxes_for[index])))
            if pending and (len(pending) >= batch_size or position == len(order) - 1):
                values = model.features(np.stack([clip for _, clip in pending]))
                for (slot, _), value in zip(pending, values):
                    out[slot] = value
                pending = []
    finally:
        capture.release()
    return out


class Head:
    """Projection and logistic weights fitted by tools/fit_strike_check.py."""

    def __init__(self, path: Path | None = None):
        data = np.load(path or HEAD_PATH)
        self.mean, self.projection = data["mean"], data["projection"]
        self.scale, self.weights, self.bias = data["scale"], data["weights"], float(data["bias"])

    def probability(self, features) -> float:
        z = ((np.asarray(features, dtype=np.float64) - self.mean) @ self.projection) / self.scale
        return float(1.0 / (1.0 + np.exp(-float(np.clip(z @ self.weights + self.bias, -30, 30)))))


def boxes_from_track(track: list[tuple[float, list | None, list | None]]) -> Callable[[str], BoxesAt]:
    """Box lookups from (seconds, box A, box B) rows, nearest row within half a second."""
    times = np.array([row[0] for row in track]) if track else np.array([])

    def for_fighter(fighter: str) -> BoxesAt:
        mine, theirs = (1, 2) if fighter == "A" else (2, 1)

        def at(moment: float):
            if not len(times):
                return None, None
            i = int(np.argmin(np.abs(times - moment)))
            if abs(times[i] - moment) > 0.5:
                return None, None
            return track[i][mine], track[i][theirs]
        return at
    return for_fighter


def score_punches(video_path: str, events: list, boxes_at_for: Callable[[str], BoxesAt]) -> dict:
    """Write evidence["video_punch_probability"] on every punch proposal.

    `boxes_at_for(fighter)` gives a function returning (that fighter's box, the
    opponent's box) at a moment of the video, from the analysis's own tracking.
    Returns a summary for the report: how many were scored, and why not if none.
    """
    started = time.perf_counter()
    punches = [event for event in events if getattr(event, "family", None) == "punch"]
    if not punches:
        return {"scored": 0, "of": 0, "reason": "no_punches"}
    if not HEAD_PATH.exists():
        return {"scored": 0, "of": len(punches), "reason": "no_head_file"}
    try:
        head = Head()
        centres = [float(e.evidence.get("action_peak_time", e.peak_time)) for e in punches]
        values = clip_features(video_path, centres, [boxes_at_for(e.fighter) for e in punches])
    except Exception as exc:  # noqa: BLE001 - optional: the analysis keeps its rules
        LOGGER.warning("strike_check_failed error=%s detail=%s", type(exc).__name__, str(exc)[:200])
        return {"scored": 0, "of": len(punches), "reason": type(exc).__name__}
    scored = 0
    for event, value in zip(punches, values):
        if value is not None:
            event.evidence["video_punch_probability"] = round(head.probability(value), 4)
            scored += 1
    reason = None if scored else ("model_unavailable" if backbone() is None else "frames_unreadable")
    return {"scored": scored, "of": len(punches), "seconds": round(time.perf_counter() - started, 2),
            "reason": reason}
