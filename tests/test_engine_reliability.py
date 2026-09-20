from dataclasses import replace

import numpy as np
import pytest

from core import temporal_model
from core.action import ActionEngine
from core.config import SETTINGS
from core.model_validation import (
    audit_dataset_split,
    audit_sequence_directory,
    classification_metrics,
)
from core.temporal_model import ACTION_CLASSES, TemporalModel
from core.types import PersonObservation
from core.video import SourceTimestampClock


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_source_clock_rejects_invalid_frame_rates(fps):
    with pytest.raises(ValueError, match="frame rate"):
        SourceTimestampClock(fps)


def test_source_clock_uses_variable_frame_presentation_times():
    clock = SourceTimestampClock(30)
    actual = [clock.seconds(frame, pts) for frame, pts in enumerate([0, 41, 92, 123])]
    assert actual == pytest.approx([0, .041, .092, .123])
    assert clock.fallback_frames == 0


def test_source_clock_missing_pts_stays_anchored_to_last_real_time(caplog):
    clock = SourceTimestampClock(25)
    assert clock.seconds(100, 5500) == 5.5
    assert clock.seconds(101, float("nan")) == pytest.approx(5.54)
    assert clock.seconds(104, 0) == pytest.approx(5.66)
    assert clock.seconds(105, 5500) == pytest.approx(5.70)
    assert clock.seconds(106, 5800) == 5.8
    assert clock.fallback_frames == 3
    assert caplog.text.count("video_timestamp_fallback") == 1


@pytest.fixture
def missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        temporal_model, "SETTINGS",
        replace(SETTINGS, temporal_checkpoint=str(tmp_path / "absent.pt")),
    )


def test_missing_temporal_checkpoint_has_explicit_diagnostics(missing_checkpoint):
    model = TemporalModel()
    assert not model.available
    assert model.predict(np.zeros((SETTINGS.action_window, 102))) is None
    assert model.diagnostics() == {
        "status": "checkpoint_missing", "error_type": None, "inference_failures": 0,
    }


def test_invalid_checkpoint_contract_fails_closed(monkeypatch, tmp_path, caplog):
    import torch

    checkpoint = tmp_path / "invalid-contract.pt"
    torch.save({"input_dim": 99, "state_dict": {}}, checkpoint)
    monkeypatch.setattr(
        temporal_model, "SETTINGS", replace(SETTINGS, temporal_checkpoint=str(checkpoint)),
    )
    model = TemporalModel()
    assert not model.available
    assert model.model is None
    assert model.diagnostics() == {
        "status": "checkpoint_load_failed", "error_type": "ValueError", "inference_failures": 0,
    }
    assert "checkpoint_load_failed" in caplog.text
    assert str(checkpoint) not in caplog.text


@pytest.mark.parametrize("failure", ["exception", "wrong_shape", "nonfinite", "bad_input"])
def test_temporal_inference_failure_is_not_retried(missing_checkpoint, failure, caplog):
    import torch

    class InferenceBoundary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            if failure == "exception":
                raise RuntimeError("private credential must not appear in logs")
            if failure == "wrong_shape":
                return torch.zeros((1, 2))
            return torch.full((1, len(ACTION_CLASSES)), float("nan"))

    model = TemporalModel()
    boundary = InferenceBoundary()
    model.available = True
    model.model = boundary
    model.device = torch.device("cpu")
    model.status = "ready"
    sequence = np.zeros((SETTINGS.action_window, 102), dtype=np.float32)
    if failure == "bad_input":
        sequence[0, 0] = float("nan")
    assert model.predict(sequence) is None
    assert not model.available
    assert model.predict(np.zeros_like(sequence)) is None
    assert boundary.calls == (0 if failure == "bad_input" else 1)
    assert model.diagnostics() == {
        "status": "inference_failed",
        "error_type": "RuntimeError" if failure == "exception" else "ValueError",
        "inference_failures": 1,
    }
    assert "private credential" not in caplog.text


def _fighter(wrist_x=180):
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


def _active_engine():
    engine = ActionEngine()
    opponent = PersonObservation(2, np.asarray([400, 80, 600, 330]), .95)
    assert engine.update("A", 0, 0, 1, _fighter(), opponent) == []
    assert engine.update("A", 1, .1, 1, _fighter(220), opponent) == []
    assert "left_hand" in engine.states["A"].active
    engine.update("B", 1, .1, 1, _fighter(), opponent)
    return engine, opponent


@pytest.mark.parametrize("missing", ["person", "pose", "short_pose"])
def test_missing_pose_discards_unfinished_action_without_affecting_opponent(missing_checkpoint, missing):
    engine, opponent = _active_engine()
    observation = None if missing == "person" else _fighter()
    if missing == "pose":
        observation.keypoints = None
    elif missing == "short_pose":
        observation.keypoints = observation.keypoints[:8]
    assert engine.update("A", 2, .2, 1, observation, opponent) == []
    state = engine.states["A"]
    assert not state.samples and not state.features and not state.active
    assert len(engine.states["B"].samples) == 1
    assert engine.update("A", 3, .3, 1, _fighter(280), opponent) == []
    assert not state.active
    np.testing.assert_array_equal(state.features[-1][51:], np.zeros(51))


@pytest.mark.parametrize("seconds,round_number", [(.2, 2), (.1, 1), (.05, 1)])
def test_round_change_or_clock_reversal_restarts_action_sequence(missing_checkpoint, seconds, round_number):
    engine, opponent = _active_engine()
    assert engine.update("A", 2, seconds, round_number, _fighter(280), opponent) == []
    state = engine.states["A"]
    assert len(state.samples) == len(state.features) == 1
    assert not state.active
    np.testing.assert_array_equal(state.features[-1][51:], np.zeros(51))


def test_completely_missed_class_counts_as_zero_in_macro_f1():
    metrics = classification_metrics(
        ["jab", "cross"], ["jab", "none"], classes=["none", "jab", "cross", "left_hook"],
    )
    assert metrics["per_class"]["cross"]["precision"] is None
    assert metrics["per_class"]["cross"]["f1"] == 0
    assert metrics["per_class"]["left_hook"]["f1"] is None
    assert metrics["macro_action_f1"] == pytest.approx(.5)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)


def _sequence(path, label, fight_id, value=0):
    np.savez_compressed(
        path, x=np.full((SETTINGS.action_window, 102), value, dtype=np.float32),
        y=np.asarray(label), fight_id=np.asarray(fight_id),
    )


def test_identical_features_with_different_labels_are_duplicates(tmp_path):
    _sequence(tmp_path / "one.npz", 1, "first-fight")
    _sequence(tmp_path / "two.npz", 2, "second-fight")
    result = audit_sequence_directory(tmp_path)
    assert result["valid_sequences"] == 2
    assert result["duplicate_sequences"] == 1
    assert not result["experimental_train_ready"]


def test_relabelled_copy_cannot_be_an_untouched_test_fight(tmp_path):
    development = tmp_path / "development"
    untouched = tmp_path / "test"
    development.mkdir()
    untouched.mkdir()
    _sequence(development / "original.npz", 1, "first-fight")
    _sequence(untouched / "relabeled.npz", 2, "renamed-fight")
    result = audit_dataset_split(development, untouched)
    assert result["fight_overlap"] == []
    assert result["content_overlap"] == 1
    assert not result["leakage_safe"]
    assert not result["untouched_test_ready"]


def test_fractional_class_index_is_rejected_not_truncated(tmp_path):
    _sequence(tmp_path / "fractional.npz", 1.5, "fight-one")
    result = audit_sequence_directory(tmp_path)
    assert result["valid_sequences"] == 0
    assert result["invalid_sequences"] == 1
    assert result["class_support"]["jab"] == 0
    assert "integer class index" in result["issues"][0]["reason"]
