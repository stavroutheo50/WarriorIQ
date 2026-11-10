"""Training-time variations of a strike window, so a model learns strikes rather than one dataset.

A model trained on BoxingVI's tidy clips fired on 3-24% of the actions the
rules detected in WarriorIQ's own fights (tools/test_model_on_own_footage.py,
2026-10-09): the skeletons there come from another pose model, another frame
rate and another camera. These variations are the cheap part of closing that
gap. Each keeps the label true:

* **mirror** - left and right swapped, as a southpaw would throw it. Sided
  classes swap with it (left_hook <-> right_hook); jab and cross stay, since
  they name the lead and rear hand, not a side.
* **speed** - the window replayed 0.8-1.25x as fast; velocities scale with it.
* **jitter** - small noise on the joint positions and speeds.
* **dropout** - a joint missing for the whole window, as when it is hidden.

The window is the trainer's 102-dim layout from core.action._feature_vector:
17 x (x, y, confidence) relative to the box, then 17 x (vx, vy, speed) in
body lengths per second. A joint that was not seen is all zeros and stays so.

Used by tools/train_temporal_model.py on training windows only; validation and
the exam always see windows exactly as they were made.
"""

from __future__ import annotations

import numpy as np

from core.temporal_model import ACTION_CLASSES

JOINTS = 17
# COCO keypoints: nose, then (left, right) pairs of eyes, ears, shoulders,
# elbows, wrists, hips, knees, ankles.
MIRROR_JOINTS = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _mirror_class(index: int) -> int:
    name = ACTION_CLASSES[index]
    if name.startswith("left_"):
        name = "right_" + name[len("left_"):]
    elif name.startswith("right_"):
        name = "left_" + name[len("right_"):]
    return ACTION_CLASSES.index(name)


MIRROR_CLASS = [_mirror_class(i) for i in range(len(ACTION_CLASSES))]


def _split(x: np.ndarray):
    pos = x[:, :JOINTS * 3].reshape(len(x), JOINTS, 3).copy()
    vel = x[:, JOINTS * 3:].reshape(len(x), JOINTS, 3).copy()
    return pos, vel


def _join(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return np.concatenate([pos.reshape(len(pos), -1), vel.reshape(len(vel), -1)], axis=1).astype(np.float32)


def _seen(pos: np.ndarray) -> np.ndarray:
    """(T, 17) True where the joint was observed (anything but all zeros)."""
    return np.abs(pos).sum(axis=2) > 0


def mirror(x: np.ndarray, y: int) -> tuple[np.ndarray, int]:
    pos, vel = _split(x)
    seen = _seen(pos)
    pos, vel, seen = pos[:, MIRROR_JOINTS], vel[:, MIRROR_JOINTS], seen[:, MIRROR_JOINTS]
    pos[..., 0] = np.where(seen, 1.0 - pos[..., 0], 0.0)
    vel[..., 0] = -vel[..., 0]
    return _join(pos, vel), MIRROR_CLASS[int(y)]


def speed(x: np.ndarray, factor: float) -> np.ndarray:
    """The window as if played ``factor`` times as fast, ending on the same frame."""
    pos, vel = _split(x)
    frames = len(x)
    # Sample backwards from the last frame (the strike's peak in our windows).
    times = np.clip((frames - 1) - (np.arange(frames)[::-1] * factor), 0, frames - 1)
    low = np.floor(times).astype(int)
    high = np.minimum(low + 1, frames - 1)
    w = (times - low)[:, None, None]
    new_pos = pos[low] * (1 - w) + pos[high] * w
    new_vel = (vel[low] * (1 - w) + vel[high] * w) * factor
    # A joint missing on either side of an interpolated frame stays missing.
    missing = ~(_seen(pos)[low] & _seen(pos)[high])
    new_pos[missing] = 0.0
    new_vel[missing] = 0.0
    return _join(new_pos, new_vel)


def augment(x: np.ndarray, y: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """One random variation of a training window; the label follows it."""
    x = np.asarray(x, dtype=np.float32)
    if rng.random() < 0.5:
        x, y = mirror(x, y)
    if rng.random() < 0.5:
        x = speed(x, float(rng.uniform(0.8, 1.25)))
    pos, vel = _split(x)
    seen = _seen(pos)
    pos[..., :2] += rng.normal(0.0, 0.01, pos[..., :2].shape) * seen[..., None]
    vel[..., :2] += rng.normal(0.0, 0.05, vel[..., :2].shape) * seen[..., None]
    vel[..., 2] = np.where(seen, np.hypot(vel[..., 0], vel[..., 1]), 0.0)
    hidden = rng.random(JOINTS) < 0.05
    pos[:, hidden] = 0.0
    vel[:, hidden] = 0.0
    return _join(pos, vel), int(y)
