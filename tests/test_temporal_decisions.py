from dataclasses import replace
import math

import numpy as np
import pytest

from core import action, temporal_model
from core.config import SETTINGS
from core.temporal_model import ACTION_CLASSES, TemporalModel
from core.types import PersonObservation


@pytest.fixture
def model(monkeypatch, tmp_path):
    settings = replace(SETTINGS, temporal_checkpoint=str(tmp_path / "absent.pt"), action_window=4)
    monkeypatch.setattr(temporal_model, "SETTINGS", settings)
    monkeypatch.setattr(action, "SETTINGS", settings)
    return TemporalModel()


def _output(model, label, confident=True, probability=None):
    import torch

    class Output(torch.nn.Module):
        def forward(self, sequence):
            logits = torch.zeros((1, len(ACTION_CLASSES)))
            if confident:
                logits[0, ACTION_CLASSES.index(label)] = (
                    math.log(probability * (len(ACTION_CLASSES) - 1) / (1 - probability))
                    if probability is not None else 10
                )
            return logits

    model.available = True
    model.status = "ready"
    model.device = torch.device("cpu")
    model.model = Output()


@pytest.mark.parametrize("label,confident,status", [
    ("none", True, "no_action"),
    ("jab", True, "strike"),
    ("none", False, "uncertain"),
])
def test_prediction_distinguishes_non_strike_from_uncertainty(model, label, confident, status):
    _output(model, label, confident)
    decision = model.predict_decision(np.zeros((4, 102), dtype=np.float32))
    assert decision.status == status
    assert decision.label == label
    assert 0 <= decision.confidence <= 1
    # Existing callers keep the tuple-or-None interface.
    legacy = model.predict(np.zeros((4, 102), dtype=np.float32))
    assert (legacy is not None) == (status == "strike")


def test_unavailable_prediction_is_not_a_negative(model):
    decision = model.predict_decision(np.zeros((4, 102), dtype=np.float32))
    assert decision.status == "unavailable"
    assert decision.label is None and decision.confidence is None
    assert decision.reason == "checkpoint_missing"


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


def _punch(engine, offset=0):
    opponent = PersonObservation(2, np.asarray([400, 80, 600, 330]), .95)
    events = []
    for index, wrist_x in enumerate([180, 220, 260, 240]):
        frame = index + offset
        events.extend(engine.update("A", frame, frame / 10, 1, _fighter(wrist_x), opponent))
    return events


def test_confident_non_strike_vetoes_candidate_without_blocking_next_strike(model):
    engine = action.ActionEngine()
    engine.temporal = model
    _output(model, "none")
    assert _punch(engine) == []
    assert not engine.states["A"].active
    assert "left_hand" not in engine.states["A"].last_event_time
    engine.interrupt("A")
    _output(model, "jab")
    events = _punch(engine, offset=4)
    assert len(events) == 1
    assert events[0].technique == "jab"
    assert events[0].model_source == "warrioriq_temporal_model"
    assert events[0].evidence["temporal_decision"]["status"] == "strike"


@pytest.mark.parametrize("case,status,reason", [
    ("missing", "unavailable", "checkpoint_missing"),
    ("uncertain", "uncertain", "below_threshold"),
    ("family", "uncertain", "family_or_side_disagreement"),
    ("side", "uncertain", "family_or_side_disagreement"),
    ("short", "unavailable", "incomplete_window"),
])
def test_unconfirmed_candidates_never_gain_model_provenance(model, monkeypatch, case, status, reason):
    engine = action.ActionEngine()
    engine.temporal = model
    if case != "missing":
        _output(model, "right_round_kick" if case == "family" else "right_hook" if case == "side" else "jab", case != "uncertain")
    if case == "short":
        monkeypatch.setattr(action, "SETTINGS", replace(action.SETTINGS, action_window=20))
    events = _punch(engine)
    assert len(events) == 1
    assert events[0].model_source == "temporal_rules"
    decision = events[0].evidence["temporal_decision"]
    assert decision["status"] == status
    assert decision["reason"] == reason


def test_failed_inference_is_unavailable_not_no_action(model):
    _output(model, "none")
    decision = model.predict_decision(np.full((4, 102), np.nan, dtype=np.float32))
    assert decision.status == "unavailable"
    assert decision.reason == "inference_failed"
    assert model.inference_failures == 1
    assert not model.available


def test_rule_score_cannot_inflate_model_confidence(model):
    engine = action.ActionEngine()
    engine.temporal = model
    _output(model, "jab", probability=.65)
    events = _punch(engine)
    assert len(events) == 1
    assert events[0].confidence == pytest.approx(.65)
