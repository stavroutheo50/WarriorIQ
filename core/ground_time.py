"""Time on the ground in MMA, second by second (owner, 2026-11-02).

The tracked fighter boxes cannot carry this: on the ground the pose model
cannot pull two tangled bodies apart, so the old reading (core/grappling.py,
which needs both fighters' hips and shoulders) caught 5 of 60 labelled
ground seconds on one bout. This reads the scene instead - every upright or
flat person the detector sees in a frame, the largest first - and asks one
question per second: are the fighters on the ground?

The people alone were not enough: from a low ringside camera the pose model
often misses the two tangled fighters entirely and finds only the referee and
the front row (MMA297: 14 of 42 ground seconds). So each sampled frame is also
described by a pretrained image model (torchvision resnet50, ImageNet, its
2048 pooled numbers), and the head reads both.

Fitted on five CC BY pankration bouts labelled second by second
(dataset/regression/grappling_pankration, tools/fit_ground_time.py), and
scored there with each bout judged by a model fitted on the other four: 646 of
674 seconds right, and every bout's ground time within 10% of its labels. All
five are from one promoter's venues, so other venues are not yet measured. It
says nothing about who is on top or about takedowns: neither was measured well
enough to publish.

Frames are sampled at SAMPLE_FPS per second whatever the analysis stride, so
the per-second mean / max / min the model was fitted on mean the same thing
at any frame rate.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

HEAD_PATH = Path(__file__).resolve().parents[1] / "models" / "ground_time_head.npz"
SAMPLE_FPS = 3.0
MIN_PERSON_CONFIDENCE = 0.35
MIN_PERSON_HEIGHT = 0.12          # share of the frame height; the crowd is smaller
LARGEST_PEOPLE = 4
MIN_SAMPLES_PER_SECOND = 2
PICTURE_SIZE = (384, 224)         # width, height the image model sees
PICTURE_BATCH = 8
_MEAN = np.array([.485, .456, .406], dtype=np.float32)
_STD = np.array([.229, .224, .225], dtype=np.float32)


def _person(box, keypoints, keypoint_conf, width: float, height: float) -> dict:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)

    def mid(i: int, j: int):
        if keypoints is None or keypoint_conf is None:
            return None
        if float(keypoint_conf[i]) < .3 or float(keypoint_conf[j]) < .3:
            return None
        return ((float(keypoints[i][0]) + float(keypoints[j][0])) / 2.0,
                (float(keypoints[i][1]) + float(keypoints[j][1])) / 2.0)

    shoulders, hips = mid(5, 6), mid(11, 12)
    angle = None
    if shoulders and hips:
        dx, dy = shoulders[0] - hips[0], shoulders[1] - hips[1]
        # 0 is upright, 90 flat, above 90 upside down.
        angle = (math.degrees(math.atan2(abs(dx), -dy)) if dy < 0
                 else 90.0 + math.degrees(math.atan2(dy, abs(dx) + 1e-6)))
    ankles = []
    if keypoints is not None and keypoint_conf is not None:
        ankles = [float(keypoints[i][1]) for i in (15, 16) if float(keypoint_conf[i]) > .3]
    hip_to_ankle = (max(ankles) - hips[1]) / h if (ankles and hips) else None
    return {"hw": h / w, "h": h / height, "angle": angle, "hip_to_ankle": hip_to_ankle,
            "box": (x1, y1, x2, y2), "area": w * h / max(1.0, width * height)}


def _overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(1.0, smaller)


def frame_features(people: list[dict], width: float, height: float) -> list[float]:
    """Ten numbers about the people in one frame.

    ``people``: dicts with box, conf, kp (17 x 2) and kc (17), as the pose
    model gives them.
    """
    found = [_person(p["box"], p.get("kp"), p.get("kc"), width, height)
             for p in people if float(p.get("conf", 1.0)) >= MIN_PERSON_CONFIDENCE]
    found = [p for p in found if p["h"] >= MIN_PERSON_HEIGHT]
    found.sort(key=lambda p: -p["area"])
    top = found[:LARGEST_PEOPLE]
    low = [p for p in top if p["hw"] < 1.25 or (p["angle"] is not None and p["angle"] > 45)]
    flat = [p for p in top if p["angle"] is not None and p["angle"] > 60]
    pair_overlap = max([_overlap(a["box"], b["box"]) for i, a in enumerate(top) for b in top[i + 1:]] or [0.0])
    largest = top[0] if top else None
    return [
        len(low) / max(1, len(top)), float(bool(low)), float(bool(flat)),
        min([p["hw"] for p in top] or [3.0]), float(np.mean([p["hw"] for p in top])) if top else 3.0,
        max([p["angle"] or 0.0 for p in top] or [0.0]),
        float(largest is not None and largest["hw"] < 1.3), pair_overlap, float(len(top)),
        min([p["hip_to_ankle"] for p in top if p["hip_to_ankle"] is not None] or [1.0]),
    ]


def second_features(frames: list[list[float]], pictures: list[np.ndarray]) -> np.ndarray:
    """One second: the mean picture, then the people's mean / max / min."""
    a = np.asarray(frames, dtype=np.float64)
    return np.concatenate([np.mean(np.asarray(pictures, dtype=np.float64), 0), a.mean(0), a.max(0), a.min(0)])


class _Pictures:
    """The pretrained image model, loaded once per process."""

    def __init__(self) -> None:
        import torch
        import torchvision

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)

    def features(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2

        batch = np.stack([(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), PICTURE_SIZE).astype(np.float32) / 255.0
                           - _MEAN) / _STD for f in frames_bgr]).transpose(0, 3, 1, 2)
        with self.torch.inference_mode():
            return self.model(self.torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)).float().cpu().numpy()


_pictures: _Pictures | None = None
_pictures_failed = False


def pictures() -> _Pictures | None:
    global _pictures, _pictures_failed
    if _pictures is None and not _pictures_failed:
        try:
            _pictures = _Pictures()
        except Exception as exc:  # noqa: BLE001 - optional: the report then leaves ground time out
            _pictures_failed = True
            LOGGER.warning("ground_time_unavailable error=%s detail=%s", type(exc).__name__, str(exc)[:200])
    return _pictures


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> dict:
    """Class-balanced logistic regression on standardised features."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    share = float(y.mean())
    weight = np.where(y == 1, .5 / max(share, 1e-3), .5 / max(1 - share, 1e-3))
    for _ in range(4000):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        grad = (p - y) * weight
        w -= .05 * (Z.T @ grad / len(y) + l2 * w / len(y))
        b -= .05 * float(grad.mean())
    return {"mu": mu, "sd": sd, "w": w, "b": np.array(b)}


def predict(head: dict, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(((np.atleast_2d(X) - head["mu"]) / head["sd"]) @ head["w"] + float(head["b"]))))


def load_head() -> dict | None:
    if not HEAD_PATH.exists():
        return None
    stored = np.load(HEAD_PATH)
    return {key: stored[key] for key in ("mu", "sd", "w", "b")}


class GroundWatch:
    """Collects scene features while the analysis runs; summarises per round."""

    def __init__(self, width: float, height: float, picture_model=None):
        self.width, self.height = float(width), float(height)
        self.frames: dict[int, list[list[float]]] = {}
        self.pictures: dict[int, list[np.ndarray]] = {}
        self.rounds: dict[int, int | None] = {}
        self._next_sample = -1.0
        self._picture_model = picture_model
        self._waiting: list[tuple[int, np.ndarray]] = []
        self.failed = False

    def _describe_waiting(self) -> None:
        if not self._waiting or self.failed:
            return
        model = self._picture_model or pictures()
        if model is None:
            self.failed = True
            return
        try:
            described = model.features([frame for _, frame in self._waiting])
        except Exception as exc:  # noqa: BLE001 - never fail the analysis over ground time
            LOGGER.warning("ground_time_failed error=%s detail=%s", type(exc).__name__, str(exc)[:200])
            self.failed = True
            return
        for (second, _), vector in zip(self._waiting, described):
            self.pictures.setdefault(second, []).append(np.asarray(vector, dtype=np.float32))
        self._waiting = []

    def observe(self, seconds: float, people, round_number: int | None, frame: np.ndarray | None = None) -> None:
        if seconds < self._next_sample or frame is None or self.failed:
            return
        # On a fixed grid, so a stride that does not divide it still keeps
        # SAMPLE_FPS frames a second, as in the bouts the head was fitted on.
        self._next_sample = (math.floor(seconds * SAMPLE_FPS + 1e-6) + 1) / SAMPLE_FPS - 1e-6
        rows = []
        for person in people or []:
            keypoints = getattr(person, "keypoints", None)
            rows.append({"box": [float(v) for v in person.box], "conf": float(person.confidence),
                         "kp": None if keypoints is None else keypoints[:, :2].tolist(),
                         "kc": None if getattr(person, "keypoint_conf", None) is None
                         else [float(v) for v in person.keypoint_conf]})
        second = int(seconds)
        self.frames.setdefault(second, []).append(frame_features(rows, self.width, self.height))
        self.rounds.setdefault(second, round_number)
        # Kept small until described: the frame itself is not held.
        import cv2
        self._waiting.append((second, cv2.resize(frame, PICTURE_SIZE)))
        if len(self._waiting) >= PICTURE_BATCH:
            self._describe_waiting()

    def summary(self, head: dict | None = None) -> dict | None:
        """{"rounds": {round: {"read_seconds", "ground_seconds"}}, ...}, or None without a model."""
        head = head if head is not None else load_head()
        if head is None:
            return None
        self._describe_waiting()
        if self.failed:
            return None
        seconds = [s for s, f in self.frames.items()
                   if len(f) >= MIN_SAMPLES_PER_SECOND and len(self.pictures.get(s, [])) == len(f)]
        if not seconds:
            return {"rounds": {}, "ground_seconds": 0, "read_seconds": 0}
        X = np.stack([second_features(self.frames[s], self.pictures[s]) for s in seconds])
        if X.shape[1] != len(head["w"]):
            LOGGER.warning("ground_time_head_mismatch features=%s head=%s", X.shape[1], len(head["w"]))
            return None
        on_ground = predict(head, X) >= .5
        rounds: dict = {}
        for second, ground in zip(seconds, on_ground):
            line = rounds.setdefault(self.rounds.get(second), {"read_seconds": 0, "ground_seconds": 0})
            line["read_seconds"] += 1
            line["ground_seconds"] += int(bool(ground))
        return {"rounds": rounds, "read_seconds": len(seconds), "ground_seconds": int(on_ground.sum()),
                "ground_seconds_list": [int(s) for s, g in zip(seconds, on_ground) if g]}
