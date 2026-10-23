"""A punch counts only when the pose model saw the arm that threw it (2026-10-23).

Measured against every strike labelled in four public-domain phone sparring
clips (tools/benchmark_dense_strikes.py) and the competitor-labelled Kick
Light bout: falsely counted punches came from arms the model was guessing.
core/action.py arm_confidence, core/analyzer.py PUNCH_ARM_CONFIDENCE.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.action import arm_confidence
from core.analyzer import PUNCH_ARM_CONFIDENCE, _attempt_drop_reason


def _sample(elbow: float, wrist: float, side: str = "right"):
    conf = np.ones(17, dtype=np.float32)
    elbow_i, wrist_i = (7, 9) if side == "left" else (8, 10)
    conf[elbow_i], conf[wrist_i] = elbow, wrist
    return SimpleNamespace(conf=conf)


def test_the_weaker_joint_of_the_arm_is_what_counts():
    samples = [_sample(0.9, 0.2), _sample(0.4, 0.8)]
    assert abs(arm_confidence(samples, "right") - 0.3) < 1e-6
    # The other arm is not this one.
    assert arm_confidence(samples, "left") == 1.0
    assert arm_confidence([SimpleNamespace(conf=None)], "right") is None


def _punch(seen, family="punch"):
    return SimpleNamespace(attempted=True, family=family, peak_time=4.0, confidence=1.0,
                           outcome="landed", metadata={}, evidence={"arm_confidence": seen})


def test_a_punch_from_an_arm_nobody_saw_is_not_counted():
    assert _attempt_drop_reason(_punch(PUNCH_ARM_CONFIDENCE - 0.05)) == "arm_not_seen"
    assert _attempt_drop_reason(_punch(0.84)) is None


def test_kicks_and_older_events_are_left_alone():
    assert _attempt_drop_reason(_punch(0.05, family="kick")) is None
    # An event analysed before arm confidence was recorded.
    assert _attempt_drop_reason(_punch(None)) is None
