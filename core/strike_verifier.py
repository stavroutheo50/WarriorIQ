"""A second look at each detected punch: was it a punch at all?

The action engine (core/action.py) proposes a punch whenever a wrist moves fast
enough, far enough, toward the opponent. Its own confidence does not separate
real punches from the rest - guard adjustments, an arm knocked back by the
opponent's punch, a wrist the pose model could not see and guessed at. Each
proposal already carries what was measured about it (its evidence); this reads
those measurements together and gives the probability that a person watching
would call it a punch.

The weights are a logistic regression fitted by tools/fit_strike_verifier.py
on detected punches that were each judged from the video: did the credited
fighter throw a punch in these frames? Each clip was scored by a model that
had not seen it, and the numbers in WEIGHTS_SOURCE are from that. It is a
filter in front of the count, never a source of strikes: it can only remove a
proposal, and it reads nothing a proposal does not already carry.

Kicks and knees are not judged here; too few were judged to fit anything.
"""

from __future__ import annotations

import math

import numpy as np

SHOULDER, ELBOW, WRIST = {"left": 5, "right": 6}, {"left": 7, "right": 8}, {"left": 9, "right": 10}

# Filled in by tools/fit_strike_verifier.py; see WEIGHTS_SOURCE.
FEATURES: tuple[str, ...] = ()
MEAN: tuple[float, ...] = ()
SCALE: tuple[float, ...] = ()
WEIGHTS: tuple[float, ...] = ()
BIAS = 0.0
WEIGHTS_SOURCE = ""


def _angle(a, b, c) -> float:
    v1, v2 = np.subtract(a, b), np.subtract(c, b)
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm <= 0.0:
        return 90.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))))


def features(event) -> dict[str, float]:
    """The measurements a punch proposal carries, as numbers the model reads."""
    evidence = getattr(event, "evidence", None) or {}
    side = "left" if str(getattr(event, "limb", "")).startswith("left") else "right"
    keypoints = evidence.get("peak_attacker_keypoints")
    box = evidence.get("peak_attacker_box")
    height = max(1.0, float(box[3]) - float(box[1])) if box else 1.0
    if keypoints is not None and len(keypoints) > WRIST[side]:
        elbow = _angle(keypoints[SHOULDER[side]], keypoints[ELBOW[side]], keypoints[WRIST[side]])
        reach = float(np.linalg.norm(np.subtract(keypoints[WRIST[side]], keypoints[SHOULDER[side]]))) / height
    else:
        elbow, reach = 90.0, 0.0
    samples = evidence.get("contact_samples") or []
    arm = evidence.get("arm_confidence")
    guard = evidence.get("defender_guard_distance")
    ratio = evidence.get("family_travel_ratio")
    return {
        "arm_confidence": 0.5 if arm is None else float(arm),
        "speed": min(6.0, float(evidence.get("max_speed_body_lengths_per_s") or 0.0)),
        "extension_gain": min(0.6, float(evidence.get("extension_gain") or 0.0)),
        "detector_confidence": float(getattr(event, "confidence", 0.0)),
        "contact_distance": min(2.0, float(evidence.get("contact_distance_body_lengths") or 2.0)),
        "guard_distance": min(2.0, 2.0 if guard is None else float(guard)),
        "hand_to_foot_travel": math.log(max(0.05, min(20.0, 1.0 if ratio is None else float(ratio)))),
        "opponent_seen_share": (evidence.get("opponent_observed_samples") or 0) / max(1, len(samples)),
        "duration": float(getattr(event, "end_time", 0.0)) - float(getattr(event, "start_time", 0.0)),
        "contact_frames": float(min(3, int(evidence.get("contact_exact_frames") or 0))),
        "near_contact_frames": float(min(5, int(evidence.get("contact_support_frames") or 0))),
        "elbow_straightness": elbow / 180.0,
        "reach": reach,
        "outcome_missed": float(getattr(event, "outcome", None) == "missed"),
        "outcome_uncertain": float(getattr(event, "outcome", None) == "uncertain"),
    }


def punch_probability(event) -> float | None:
    """How likely a person would call this proposal a punch; None when it cannot say."""
    if getattr(event, "family", None) != "punch" or not FEATURES:
        return None
    measured = features(event)
    z = sum(((measured[name] - mean) / scale) * weight
            for name, mean, scale, weight in zip(FEATURES, MEAN, SCALE, WEIGHTS)) + BIAS
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
