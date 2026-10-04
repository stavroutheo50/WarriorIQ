"""No kick or knee without a visible leg.

QA, 2026-10-04: a waist-up boxing clip under K-1 rules produced 33 kicks,
14 knees and 3 punches for one fighter, and two mirror-image fighters got
different counts. The pose model returns knee and ankle positions for legs
outside the picture - low-confidence guesses that jitter like strikes.
"""

from dataclasses import replace

import numpy as np
import pytest

from core import temporal_model
from core.action import ActionEngine
from core.config import SETTINGS
from core.types import PersonObservation

LEG_JOINTS = (13, 14, 15, 16)


def _body(right_ankle, *, wrist_x=180.0, leg_conf=1.0, jitter=0.0):
    points = np.asarray([
        [200, 100], [195, 98], [205, 98], [190, 102], [210, 102],
        [185, 140], [215, 140], [180, 150], [220, 150],
        [wrist_x, 155], [220, 155], [190, 220], [210, 220],
        [190, 270], [210 + jitter, 270 - jitter], [190, 320], right_ankle,
    ], dtype=np.float32)
    conf = np.ones(17, dtype=np.float32)
    conf[list(LEG_JOINTS)] = leg_conf
    return PersonObservation(1, np.asarray([160, 80, 300, 330], dtype=np.float32), .95,
                             keypoints=points, keypoint_conf=conf)


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setattr(temporal_model, "SETTINGS",
                        replace(SETTINGS, temporal_checkpoint=str(tmp_path / "absent.pt")))
    return ActionEngine()


OPPONENT = PersonObservation(2, np.asarray([400, 80, 600, 330], dtype=np.float32), .95)
# The right foot rises and drives toward the opponent, then comes back.
KICK = [[210, 320], [250, 290], [300, 250], [350, 220], [380, 210], [330, 240], [260, 300]]


def _run(engine, frames, fighter="A"):
    events = []
    for index, observation in enumerate(frames):
        events.extend(engine.update(fighter, index, index / 10, 1, observation, OPPONENT))
    return events


def test_a_visible_kick_is_still_counted(engine):
    events = _run(engine, [_body(ankle) for ankle in KICK])
    assert [e.family for e in events if e.family in {"kick", "knee"}], events


def test_the_same_kick_with_guessed_legs_is_not(engine):
    events = _run(engine, [_body(ankle, leg_conf=0.15) for ankle in KICK])
    assert [e for e in events if e.family in {"kick", "knee"}] == []
    assert engine.states["A"].legs_hidden_discards >= 1


def test_jittery_guessed_legs_do_not_turn_a_punch_into_a_kick(engine):
    # Waist-up: the hand punches while the guessed knee jumps about.
    frames = [_body([210, 320], wrist_x=x, leg_conf=0.1, jitter=j)
              for x, j in zip([180, 220, 260, 300, 270, 230], [0, 40, -35, 45, -40, 30])]
    events = _run(engine, frames)
    assert all(e.family == "punch" for e in events), [e.family for e in events]


def test_mirror_fighters_with_hidden_legs_both_get_no_leg_strikes(engine):
    for fighter in ("A", "B"):
        events = _run(engine, [_body(ankle, leg_conf=0.2, jitter=(-1) ** i * 30)
                               for i, ankle in enumerate(KICK)], fighter)
        assert [e for e in events if e.family in {"kick", "knee"}] == []
