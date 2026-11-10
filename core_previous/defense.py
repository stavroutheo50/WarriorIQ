from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from core.types import DefenseEvent, PersonObservation, StrikeEvent

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class DefensePose:
    frame: int
    time: float
    fighter: str
    keypoints: np.ndarray
    box: np.ndarray


def _p(kp, i):
    if kp is None or len(kp) <= i:
        return None
    p = np.asarray(kp[i], dtype=np.float32)[:2]
    return p if p[0] > 0 and p[1] > 0 else None


def _body(kp, box):
    ls, rs, lh, rh = (_p(kp, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    if all(x is not None for x in (ls, rs, lh, rh)):
        return max(20.0, float(np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2)) * 2.15)
    return max(20.0, float(box[3] - box[1]))


class DefenseEngine:
    """Evidence-linked defense classifier.

    Blocks/checks come directly from contact geometry. Slips/evades/parries are
    only emitted when temporal defender movement supports them; otherwise the
    event stays simply 'missed' rather than inventing a defense.
    """

    def __init__(self):
        self.history = {"A": deque(maxlen=20), "B": deque(maxlen=20)}

    def update_pose(self, fighter: str, frame: int, seconds: float, obs: PersonObservation | None) -> None:
        if obs is None or obs.keypoints is None:
            return
        self.history[fighter].append(DefensePose(frame, seconds, fighter, obs.keypoints.copy(), obs.box.copy()))

    def _nearest(self, fighter: str, seconds: float) -> list[DefensePose]:
        return sorted(self.history[fighter], key=lambda x: abs(x.time - seconds))[:4]

    def classify(self, event: StrikeEvent) -> DefenseEvent | None:
        defender = event.opponent
        if event.outcome == "blocked":
            return DefenseEvent(defender, event.fighter, event.round_number, event.peak_time, event.peak_frame, "block", max(0.70, event.contact_confidence), event.technique)
        if event.outcome == "checked":
            return DefenseEvent(defender, event.fighter, event.round_number, event.peak_time, event.peak_frame, "check", max(0.72, event.contact_confidence), event.technique)
        if event.outcome != "missed":
            return None

        poses = self._nearest(defender, event.peak_time)
        if len(poses) < 2:
            return None
        before, after = min(poses, key=lambda x: x.time), max(poses, key=lambda x: x.time)
        body = _body(after.keypoints, after.box)
        head_before, head_after = _p(before.keypoints, NOSE), _p(after.keypoints, NOSE)
        if head_before is None or head_after is None:
            return None
        head_move = (head_after - head_before) / body
        horizontal = abs(float(head_move[0]))
        vertical = abs(float(head_move[1]))

        # Hand moving across the head line during a missed punch is a parry clue.
        if event.family == "punch":
            lb, rb = _p(before.keypoints, L_WRIST), _p(before.keypoints, R_WRIST)
            la, ra = _p(after.keypoints, L_WRIST), _p(after.keypoints, R_WRIST)
            hand_motion = 0.0
            for p0, p1 in ((lb, la), (rb, ra)):
                if p0 is not None and p1 is not None:
                    hand_motion = max(hand_motion, float(np.linalg.norm(p1 - p0)) / body)
            if hand_motion > 0.16 and horizontal < 0.10:
                return DefenseEvent(defender, event.fighter, event.round_number, event.peak_time, event.peak_frame, "parry", 0.62, event.technique)
            if horizontal > 0.10 and horizontal > vertical * 1.4:
                return DefenseEvent(defender, event.fighter, event.round_number, event.peak_time, event.peak_frame, "slip", min(0.88, 0.55 + horizontal), event.technique)

        if horizontal > 0.16 or vertical > 0.14:
            return DefenseEvent(defender, event.fighter, event.round_number, event.peak_time, event.peak_frame, "evade", min(0.85, 0.52 + max(horizontal, vertical)), event.technique)
        return None
