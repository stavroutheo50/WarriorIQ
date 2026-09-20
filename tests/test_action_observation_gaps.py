from dataclasses import replace

import numpy as np
import pytest

from core import temporal_model
from core.action import ActionEngine
from core.config import SETTINGS
from core.types import PersonObservation


def _fighter(wrist_x):
    points = np.asarray([
        [200, 100], [195, 98], [205, 98], [190, 102], [210, 102],
        [185, 140], [215, 140], [180, 150], [220, 150],
        [wrist_x, 155], [220, 155], [190, 220], [210, 220],
        [190, 270], [210, 270], [190, 320], [210, 320],
    ], dtype=np.float32)
    return PersonObservation(
        1, np.asarray([160, 80, 300, 330], dtype=np.float32), .95,
        keypoints=points, keypoint_conf=np.ones(17, dtype=np.float32),
    )


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setattr(
        temporal_model, "SETTINGS",
        replace(SETTINGS, temporal_checkpoint=str(tmp_path / "absent.pt")),
    )
    return ActionEngine()


def _update(engine, fighter, frame, seconds, wrist_x):
    opponent = PersonObservation(2, np.asarray([400, 80, 600, 330]), .95)
    return engine.update(fighter, frame, seconds, 1, _fighter(wrist_x), opponent)


def test_unobserved_gap_does_not_complete_a_pre_gap_punch(engine):
    for frame, wrist_x in enumerate([180, 220, 260]):
        assert _update(engine, "A", frame, frame / 10, wrist_x) == []
    assert "left_hand" in engine.states["A"].active

    # Retraction is observed 7.8 seconds later: its intervening motion is unknown.
    assert _update(engine, "A", 80, 8.0, 240) == []
    state = engine.states["A"]
    assert len(state.samples) == len(state.features) == 1
    assert not state.active
    np.testing.assert_array_equal(state.features[-1][51:], np.zeros(51))


def test_contiguous_six_fps_punch_is_preserved(engine):
    events = []
    for frame, wrist_x in enumerate([180, 220, 260, 240]):
        events.extend(_update(engine, "A", frame, frame / 6, wrist_x))

    assert len(events) == 1
    assert events[0].fighter == "A"
    assert events[0].family == "punch"
    assert events[0].start_time == 0
    assert events[0].peak_time == pytest.approx(2 / 6)
    assert events[0].end_time == pytest.approx(3 / 6)


def test_gap_restarts_only_the_affected_fighters_sequence(engine):
    for frame, wrist_x in enumerate([180, 220, 260]):
        _update(engine, "A", frame, frame / 10, wrist_x)
        _update(engine, "B", 76 + frame, 7.6 + frame / 10, wrist_x)
    b_state = engine.states["B"]
    b_samples = list(b_state.samples)
    b_candidate = b_state.active["left_hand"]

    _update(engine, "A", 80, 8.0, 240)
    assert len(engine.states["A"].samples) == 1
    assert len(b_state.samples) == len(b_samples)
    assert all(before is after for before, after in zip(b_samples, b_state.samples))
    assert b_state.active["left_hand"] is b_candidate
    events = _update(engine, "B", 79, 7.9, 240)
    assert len(events) == 1
    assert events[0].fighter == "B"
