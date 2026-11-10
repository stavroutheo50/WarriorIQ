"""Is this one frame the peak of a punch, a kick, or neither?

The temporal strike model fired on 0-16% of the actions the rules detected in
WarriorIQ's own fights (tools/test_model_on_own_footage.py, 2026-10-10). Every
positive it learned from was a clip from another dataset with another pose
model, so it learned those datasets. What was missing was strikes seen by
**our** pose model.

The Combat Sports Dataset (Evan Quinn, Zenodo doi:10.5281/zenodo.15349809,
CC BY 4.0) is still images, not clips, so it cannot teach motion. It can teach
the shape of a strike at its peak: tools/import_combat_sports.py runs the same
RTMPose model the live analysis uses on each labelled person, so these
skeletons come from our pose model. Strikes and the "none" examples (guards)
come from the same videos, so source cannot separate them.

The feature is the position half of core.action._feature_vector - 17 joints
x (x, y, confidence) relative to the person box - so a frame from a live
analysis and a frame from the importer are described identically.

The model is a small two-layer network in plain numpy: it needs no new
dependency, trains in seconds on a few thousand frames, and loads anywhere the
analysis runs. It is measured, never trusted: nothing in the live analysis
reads it until it passes on our own footage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

PEAK_CLASSES = ("none", "punch", "kick")
FEATURES = 51
# Dataset label -> what the model is asked. Guards are "none": a fighter
# standing ready, not striking.
FAMILY = {"cross": "punch", "hook": "punch", "kick": "kick", "high-guard": "none", "low-guard": "none"}


def pose_features(box, keypoints, conf) -> np.ndarray:
    """51 numbers for one person on one frame, as the live analysis computes them."""
    from core.action import Sample, _feature_vector

    sample = Sample(
        frame=0, time=0.0, round_number=None, box=np.asarray(box, dtype=np.float32),
        keypoints=np.asarray(keypoints, dtype=np.float32),
        conf=None if conf is None else np.asarray(conf, dtype=np.float32),
        opponent_box=None, opponent_keypoints=None, opponent_conf=None,
        identity_confidence=1.0, opponent_identity_confidence=1.0)
    return _feature_vector(sample, None)[:FEATURES]


def mirror_features(x: np.ndarray) -> np.ndarray:
    """The same frames seen in a mirror: a southpaw's version of each pose."""
    from core.temporal_augment import MIRROR_JOINTS

    pos = np.asarray(x, dtype=np.float32).reshape(len(x), 17, 3)[:, MIRROR_JOINTS].copy()
    seen = np.abs(pos).sum(axis=2) > 0
    pos[..., 0] = np.where(seen, 1.0 - pos[..., 0], 0.0)
    return pos.reshape(len(x), FEATURES)


class PeakPoseModel:
    """Two layers: 51 -> hidden (ReLU) -> 3. Inputs are standardised."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}

    def logits(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float32) - self.w["mean"]) / self.w["std"]
        h = np.maximum(0.0, z @ self.w["w1"] + self.w["b1"])
        return h @ self.w["w2"] + self.w["b2"]

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        s = self.logits(x)
        s = np.exp(s - s.max(axis=1, keepdims=True))
        return s / s.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> list[str]:
        return [PEAK_CLASSES[i] for i in self.probabilities(x).argmax(axis=1)]

    def save(self, path: Path, **meta) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(handle, classes=np.asarray(PEAK_CLASSES), **self.w,
                     **{"meta_" + k: np.asarray(v) for k, v in meta.items()})
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> "PeakPoseModel":
        with np.load(path, allow_pickle=False) as blob:
            if tuple(str(c) for c in blob["classes"]) != PEAK_CLASSES:
                raise ValueError(f"{path} was trained for other classes: {list(blob['classes'])}")
            return cls({k: blob[k] for k in ("mean", "std", "w1", "b1", "w2", "b2")})


def train(x: np.ndarray, y: np.ndarray, *, hidden: int = 64, epochs: int = 400, lr: float = 0.01,
          weight_decay: float = 1e-4, seed: int = 0) -> PeakPoseModel:
    """Full-batch Adam on class-balanced cross-entropy. ``y`` holds PEAK_CLASSES indices.

    Classes are weighted by inverse frequency so the rare kicks are not simply
    ignored. Deterministic for a given seed.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-3
    z = (x - mean) / std
    k = len(PEAK_CLASSES)
    counts = np.bincount(y, minlength=k).astype(np.float32)
    class_weight = np.where(counts > 0, len(y) / (k * np.maximum(counts, 1)), 0.0)
    sample_weight = class_weight[y] / class_weight[y].sum()
    params = {
        "w1": rng.normal(0, np.sqrt(2.0 / FEATURES), (FEATURES, hidden)).astype(np.float32),
        "b1": np.zeros(hidden, np.float32),
        "w2": rng.normal(0, np.sqrt(1.0 / hidden), (hidden, k)).astype(np.float32),
        "b2": np.zeros(k, np.float32),
    }
    m = {n: np.zeros_like(p) for n, p in params.items()}
    v = {n: np.zeros_like(p) for n, p in params.items()}
    onehot = np.eye(k, dtype=np.float32)[y]
    for step in range(1, epochs + 1):
        pre = z @ params["w1"] + params["b1"]
        h = np.maximum(0.0, pre)
        s = h @ params["w2"] + params["b2"]
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        ds = (p - onehot) * sample_weight[:, None]
        grads = {"w2": h.T @ ds, "b2": ds.sum(axis=0)}
        dh = (ds @ params["w2"].T) * (pre > 0)
        grads["w1"] = z.T @ dh
        grads["b1"] = dh.sum(axis=0)
        for n in params:
            g = grads[n] + (weight_decay * params[n] if n.startswith("w") else 0.0)
            m[n] = 0.9 * m[n] + 0.1 * g
            v[n] = 0.999 * v[n] + 0.001 * g * g
            params[n] -= lr * (m[n] / (1 - 0.9 ** step)) / (np.sqrt(v[n] / (1 - 0.999 ** step)) + 1e-8)
    return PeakPoseModel({"mean": mean, "std": std, **params})


def scores(truth: list[str], predicted: list[str]) -> dict[str, dict]:
    """Precision and recall per class, with the counts behind them."""
    out = {}
    for name in PEAK_CLASSES:
        tp = sum(t == name and p == name for t, p in zip(truth, predicted))
        called = sum(p == name for p in predicted)
        real = sum(t == name for t in truth)
        out[name] = {"true": real, "called": called, "correct": tp,
                     "precision": tp / called if called else None,
                     "recall": tp / real if real else None}
    return out
