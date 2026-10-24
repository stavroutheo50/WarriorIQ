"""The video check on proposed punches (core/strike_verifier.py, 2026-10-23).

The pretrained backbone is replaced by a stand-in here: the suite never
downloads a model. What is pinned is the plumbing around it - which frames are
read, how the crop follows the fighter, how the head turns features into a
probability, and that a scored punch is judged by the check alone.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core import strike_verifier as check
from core.analyzer import _attempt_drop_reason
from core.config import SETTINGS
from core.contact import resolve_simultaneous_attribution


@contextmanager
def _threshold(value):
    previous = SETTINGS.punch_check_threshold
    object.__setattr__(SETTINGS, "punch_check_threshold", value)
    try:
        yield
    finally:
        object.__setattr__(SETTINGS, "punch_check_threshold", previous)


def test_sixteen_moments_around_the_peak():
    times = check.window_times(10.0)
    assert len(times) == 16 and times[0] == pytest.approx(9.6) and times[-1] == pytest.approx(10.35)


def test_the_crop_reaches_toward_the_opponent():
    alone = check.crop_box([100, 100, 200, 300], None, 1000, 500)
    toward = check.crop_box([100, 100, 200, 300], [500, 100, 600, 300], 1000, 500)
    assert toward[2] > alone[2]                      # widened to the right, where the opponent is
    assert check.crop_box(None, None, 640, 360) == (0, 0, 640, 360)


def test_box_lookup_takes_the_nearest_frame_and_gives_up_far_away():
    lookup = check.boxes_from_track([(1.0, [0, 0, 1, 1], [2, 2, 3, 3]), (2.0, [5, 5, 6, 6], None)])
    assert lookup("A")(1.1) == ([0, 0, 1, 1], [2, 2, 3, 3])
    assert lookup("B")(1.9) == (None, [5, 5, 6, 6])
    assert lookup("A")(9.0) == (None, None)


def _head(path, dims=3):
    np.savez(path, mean=np.zeros(dims), projection=np.eye(dims)[:, :1], scale=np.ones(1),
             weights=np.array([2.0]), bias=np.array(0.0))
    return check.Head(path)


def test_the_head_is_a_logistic_on_the_projection(tmp_path):
    head = _head(tmp_path / "h.npz")
    assert head.probability([0, 0, 0]) == pytest.approx(0.5)
    assert head.probability([3, 0, 0]) > 0.99 and head.probability([-3, 0, 0]) < 0.01


def _video(path, seconds=2.0, fps=25):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(int(seconds * fps)):
        frame = np.full((120, 160, 3), (i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class _StandIn:
    """Feature = mean brightness of the clip, so frames really are read."""

    def features(self, clips):
        return np.array([[float(c.mean()), 0.0, 0.0] for c in clips])


def _punch(fighter, peak, **evidence):
    return SimpleNamespace(family="punch", fighter=fighter, peak_time=peak, attempted=True, confidence=1.0,
                           outcome="missed", metadata={}, evidence={"action_peak_time": peak, **evidence})


def test_every_punch_is_scored_from_its_own_frames(tmp_path, monkeypatch):
    video = _video(tmp_path / "v.mp4")
    _head(tmp_path / "h.npz")
    monkeypatch.setattr(check, "HEAD_PATH", tmp_path / "h.npz")
    monkeypatch.setattr(check, "backbone", lambda: _StandIn())
    events = [_punch("A", 0.6), _punch("B", 1.4),
              SimpleNamespace(family="kick", fighter="A", peak_time=1.0, evidence={})]
    summary = check.score_punches(str(video), events, check.boxes_from_track([]))
    assert summary["scored"] == 2 and summary["of"] == 2
    a, b = (e.evidence["video_punch_probability"] for e in events[:2])
    assert a != b                                    # different moments, different frames
    assert "video_punch_probability" not in events[2].evidence


def test_without_a_model_nothing_is_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "HEAD_PATH", tmp_path / "missing.npz")
    events = [_punch("A", 0.6)]
    assert check.score_punches("x.mp4", events, check.boxes_from_track([]))["reason"] == "no_head_file"
    assert "video_punch_probability" not in events[0].evidence


def test_a_scored_punch_is_judged_by_the_check_alone():
    far_miss = dict(defender_guard_distance=1.5)     # the old rule called this "thrown at nothing"
    with _threshold(0.6):
        assert _attempt_drop_reason(_punch("A", 1.0, video_punch_probability=0.9, **far_miss)) is None
        assert _attempt_drop_reason(_punch("A", 1.0, video_punch_probability=0.3, **far_miss)) == "not_a_punch"
        # Not scored (no model): the old rule still applies.
        assert _attempt_drop_reason(_punch("A", 1.0, **far_miss)) == "thrown_at_nothing"
    with _threshold(0.0):
        # Switched off: the score is ignored and the old rules are back.
        assert _attempt_drop_reason(_punch("A", 1.0, video_punch_probability=0.9, **far_miss)) == "thrown_at_nothing"


def test_the_pair_rule_leaves_scored_punches_alone():
    near = _punch("A", 5.0, contact_distance_body_lengths=0.1, video_punch_probability=0.8)
    far = _punch("B", 5.1, contact_distance_body_lengths=0.9, video_punch_probability=0.8)
    kept, dropped = resolve_simultaneous_attribution([near, far])
    assert dropped == 0 and len(kept) == 2
    near.evidence.pop("video_punch_probability")
    far.evidence.pop("video_punch_probability")
    kept, dropped = resolve_simultaneous_attribution([near, far])
    assert dropped == 1 and kept == [near]


def test_only_sports_it_was_measured_on_are_checked():
    # Boxing only (2026-10-24): on Muay Thai it let more false strikes through.
    assert check.applies_to("BOXING")
    for ruleset in ("K1", "KICK_LIGHT", "MUAY_THAI", "ITF_TAEKWONDO", "WT_TAEKWONDO", "MMA", None):
        assert not check.applies_to(ruleset)
