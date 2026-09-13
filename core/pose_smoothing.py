"""Refusing a joint position the body could not have reached.

The pose model does not fail by being uncertain. It fails by returning a
confidently wrong skeleton: core/rtm_pose.py records frames where the torso is
squeezed to a sliver and both legs converge on a single point while ankle
confidence reads 0.93 to 0.97. Nothing downstream can catch that, because every
downstream check is a confidence threshold and the confidence is high.

Time can catch it. A joint that teleports and comes back has moved at a speed
no body reaches, and that is measurable without a single label.

**The limits are read off the footage, not chosen.** Measured over 15,164 joint
transitions on fight 1, in body-heights per second - the frame rate the
analysis samples at varies, so a per-frame displacement would mean nothing:

    median 0.47    p90 1.74    p99 6.02    p99.9 12.58    max 25.30

The important part is not the overall shape but that it differs by joint. Both
ankles reach a p99 of 8.2 and 9.5, which is a kick: a foot genuinely travels
around 15 m/s and the fastest recorded competition kicks are quicker than that.
The head does not. Yet nose, eyes, ears and shoulders all reach 10 to 15
body-heights per second at p99.9 - about 20 m/s, or 73 km/h, for somebody's
face. A head cannot do that and a hip cannot do that. That asymmetry is the
whole signal: the same number means "kick" at the ankle and "the skeleton
collapsed" at the nose.

So the limit is per joint, set well above what that joint really does and well
below what the failure produces.

**Which joints get refused is the check on the whole idea**, and it was run on
two bouts rather than the one the limits came from:

    fight 1   86 joints over  18 frames   0.64% of updates   4.8 joints/frame
    fight 3  233 joints over  31 frames   1.45% of updates   7.5 joints/frame

    head and torso   70% / 74%      elbows and knees   13% / 18%
    wrists and ankles   9% / 8%     wrists alone, fight 1: none at all

Hips, shoulders, ears and noses are what this refuses - joints that cannot
travel that fast under any circumstances - and it refuses several of them in
the same frame, which is a skeleton collapsing rather than a limb moving. If
the limits had been clipping real kicks the tail would be wrists and ankles,
and it is not. Fight 3 refuses more because it is the harder footage; the
proportions hold.

Coverage and event counts are unchanged on both bouts, which is by design: this
runs after the identity manager has already chosen, so it changes what is
measured about a fighter and never which fighter was picked.

This gates rather than smooths, deliberately. An EMA over the keypoints would
blur genuine fast motion - and fast motion is the entire subject, since a kick
is the thing being measured. Holding the last believable position for a joint
leaves every ordinary frame exactly as the model returned it.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = ("nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
               "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
               "l_knee", "r_knee", "l_ankle", "r_ankle")

# COCO-17 order, which is what the pose model and everything reading it use.
_HEAD_AND_TORSO = (0, 1, 2, 3, 4, 5, 6, 11, 12)   # nose, eyes, ears, shoulders, hips
_MID_LIMB = (7, 8, 13, 14)                        # elbows, knees
_EXTREMITY = (9, 10, 15, 16)                      # wrists, ankles

# Body-heights per second. A body height is about 1.7 m, so 6 is 10 m/s at the
# hip and 14 is 24 m/s at the ankle - past the fastest kick anyone has recorded.
_LIMIT_HEAD_TORSO = 6.0
_LIMIT_MID_LIMB = 9.0
_LIMIT_EXTREMITY = 14.0

# Longer than this between two sightings and there is nothing to compare
# against: the fighter may genuinely be somewhere else.
_MAX_GAP_SECONDS = 0.75

# A joint may only be held for so long. If the model keeps insisting, it is
# more likely right than this gate is, and a fighter who really did move must
# not be pinned to where they were.
_MAX_CONSECUTIVE_HOLDS = 3


def _limits() -> np.ndarray:
    limits = np.full(17, _LIMIT_MID_LIMB, dtype=np.float32)
    limits[list(_HEAD_AND_TORSO)] = _LIMIT_HEAD_TORSO
    limits[list(_MID_LIMB)] = _LIMIT_MID_LIMB
    limits[list(_EXTREMITY)] = _LIMIT_EXTREMITY
    return limits


class JointGate:
    """Per-fighter history, refusing joint positions no body could reach.

    Applied after the identity manager has committed a fighter, so it corrects
    what the metrics, the action engine and the defence engine read while
    leaving identity matching on the raw observation - identity's job is to
    decide who this is, and it should see what the model actually returned.
    """

    def __init__(self):
        self._last: dict[str, tuple[np.ndarray, float]] = {}
        self._holds: dict[str, np.ndarray] = {}
        self._limits = _limits()
        self.gated_joints = 0
        self.gated_frames = 0
        self.examined_joints = 0
        # Which joints get refused is the check on this whole idea. Head and
        # torso mean a collapsed skeleton, which is the failure. A tail of
        # ankles and wrists would mean the limits were clipping real kicks
        # instead, and the thresholds would be wrong.
        self.per_joint = np.zeros(17, dtype=np.int64)

    def reset(self, name: str) -> None:
        self._last.pop(name, None)
        self._holds.pop(name, None)

    def apply(self, name: str, seconds: float, observation) -> None:
        """Correct this fighter's keypoints in place, if any need it."""
        if observation is None or observation.keypoints is None:
            # Losing the fighter ends the comparison. Holding a stale pose
            # across a gap would fight the re-acquisition rather than help it.
            self.reset(name)
            return
        points = np.asarray(observation.keypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] != 17:
            return
        box = np.asarray(observation.box, dtype=np.float32)
        scale = float(max(1.0, box[3] - box[1]))
        previous = self._last.get(name)
        current = points[:, :2].copy()
        if previous is not None:
            last_points, last_seconds = previous
            gap = float(seconds) - float(last_seconds)
            if 0.0 < gap <= _MAX_GAP_SECONDS:
                speed = np.linalg.norm(current - last_points, axis=1) / scale / gap
                holds = self._holds.get(name, np.zeros(17, dtype=np.int16))
                refuse = (speed > self._limits) & (holds < _MAX_CONSECUTIVE_HOLDS)
                self.examined_joints += 17
                if refuse.any():
                    current[refuse] = last_points[refuse]
                    points[refuse, :2] = last_points[refuse]
                    observation.keypoints = points
                    # The signature is derived from the keypoints, so a stale
                    # one would describe the pose this just refused.
                    observation.pose_signature = None
                    self.gated_joints += int(refuse.sum())
                    self.gated_frames += 1
                    self.per_joint += refuse.astype(np.int64)
                holds = np.where(refuse, holds + 1, 0).astype(np.int16)
                self._holds[name] = holds
            else:
                self._holds[name] = np.zeros(17, dtype=np.int16)
        self._last[name] = (current, float(seconds))

    def summary(self) -> dict:
        return {
            "gated_joints": self.gated_joints,
            "gated_frames": self.gated_frames,
            "examined_joints": self.examined_joints,
            "gated_share": (
                round(self.gated_joints / self.examined_joints, 5)
                if self.examined_joints else 0.0
            ),
            "gated_by_joint": dict(zip(JOINT_NAMES, (int(n) for n in self.per_joint))),
        }
