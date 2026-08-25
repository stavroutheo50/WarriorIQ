from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from core.types import KnockdownEvent, PersonObservation, StrikeEvent

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def _p(kp, idx):
    if kp is None or len(kp) <= idx:
        return None
    p = np.asarray(kp[idx], dtype=np.float32)[:2]
    return p if p[0] > 0 and p[1] > 0 else None


@dataclass
class FallSample:
    frame: int
    seconds: float
    round_number: int | None
    horizontal: bool
    low: bool


class KnockdownDetector:
    """Conservative knockdown evidence detector.

    It requires a sustained near-horizontal torso/box state AND a recent clean
    opponent strike. If those conditions are not met, WarriorIQ reports no
    knockdown rather than guessing from a single crouch/slip frame.
    """

    def __init__(self, frame_height: int):
        self.frame_height = max(1, frame_height)
        self.history = {"A": deque(maxlen=12), "B": deque(maxlen=12)}
        self.last_emitted = {"A": -999.0, "B": -999.0}

    def _fall_shape(self, obs: PersonObservation | None) -> tuple[bool, bool]:
        if obs is None:
            return False, False
        x1, y1, x2, y2 = map(float, obs.box)
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        box_horizontal = w / h > 1.15
        kp = obs.keypoints
        shoulders = [_p(kp, L_SHOULDER), _p(kp, R_SHOULDER)]
        hips = [_p(kp, L_HIP), _p(kp, R_HIP)]
        shoulders = [p for p in shoulders if p is not None]
        hips = [p for p in hips if p is not None]
        torso_horizontal = False
        if shoulders and hips:
            s = np.mean(shoulders, axis=0)
            hp = np.mean(hips, axis=0)
            delta = hp - s
            torso_horizontal = abs(float(delta[0])) > abs(float(delta[1])) * 1.25
        center_y = (y1 + y2) / 2.0
        low = center_y > self.frame_height * 0.63 or y2 > self.frame_height * 0.90
        return bool(box_horizontal or torso_horizontal), bool(low)

    def update(self, fighter: str, frame: int, seconds: float, round_number: int | None, obs: PersonObservation | None, events: list[StrikeEvent]) -> KnockdownEvent | None:
        horizontal, low = self._fall_shape(obs)
        self.history[fighter].append(FallSample(frame, seconds, round_number, horizontal, low))
        if seconds - self.last_emitted[fighter] < 3.0:
            return None
        samples = list(self.history[fighter])[-6:]
        if len(samples) < 4:
            return None
        strong_fall = sum(s.horizontal and s.low for s in samples) >= 4
        if not strong_fall:
            return None
        opponent = "B" if fighter == "A" else "A"
        recent = [
            e for e in events
            if e.fighter == opponent
            and e.outcome in {"clean", "likely_landed"}
            and 0.0 <= seconds - e.peak_time <= 2.0
        ]
        if not recent:
            return None
        cause = max(recent, key=lambda e: e.peak_time)
        confidence = 0.72 + min(0.20, cause.contact_confidence * 0.15)
        self.last_emitted[fighter] = seconds
        return KnockdownEvent(
            fighter=fighter,
            caused_by=opponent,
            round_number=round_number,
            source_frame=frame,
            time_seconds=seconds,
            confidence=float(min(0.93, confidence)),
            evidence={"causing_strike_time": cause.peak_time, "causing_technique": cause.technique},
        )
