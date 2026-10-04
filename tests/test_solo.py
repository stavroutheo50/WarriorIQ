"""Solo sessions: one person, movement and guard only, no opponent, no score.

QA, 2026-10-04: a one-person video (shadowboxing, bag, pads) was a dead end -
the same person could not be picked as A and B.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core import solo
from core.types import AnalysisRequest, PersonObservation


def _person(track_id, x, appearance=None, keypoints=True):
    box = np.asarray([x, 60, x + 80, 300], dtype=np.float32)
    kp = None
    conf = None
    if keypoints:
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[:, 0] = np.linspace(x + 10, x + 70, 17)
        kp[:, 1] = np.linspace(70, 290, 17)
        kp[9] = kp[10] = [x + 40, 90]   # wrists up by the face: guard up
        conf = np.ones(17, dtype=np.float32)
    return PersonObservation(track_id, box, 0.9, keypoints=kp, keypoint_conf=conf,
                             appearance=None if appearance is None else np.asarray(appearance, dtype=np.float32))


RED, BLUE = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]


def _samples(people_at):
    return [(i, i / 10, people_at(i)) for i in range(30)]


def test_seed_person_needs_a_person_in_the_box():
    people = [_person(1, 100)]
    found, iou = solo.seed_person(people, [100, 60, 180, 300])
    assert found is people[0] and iou > 0.9
    assert solo.seed_person(people, [400, 60, 480, 300])[0] is None


def test_the_person_is_followed_both_ways_from_the_seed_by_track_id():
    samples = _samples(lambda i: [_person(7, 100 + i, RED), _person(9, 400, BLUE)])
    seed = samples[15][2][0]
    followed = solo.follow(samples, 15, seed)
    assert all(obs is not None and obs.track_id == 7 for obs in followed)


def test_a_lost_track_is_picked_up_nearby_within_a_second():
    # Track id changes at 2.0 s; the new id is where the person was.
    samples = _samples(lambda i: [_person(7 if i < 20 else 8, 100 + i, RED)])
    followed = solo.follow(samples, 5, samples[5][2][0])
    assert all(obs is not None for obs in followed)


def test_a_pad_holder_is_not_adopted_when_the_person_leaves():
    # The person is gone from 1.5 s; only the holder (different look, far away)
    # remains. Position alone may not hand the identity over.
    samples = _samples(lambda i: ([_person(7, 100, RED)] if i < 15 else []) + [_person(9, 450, BLUE)])
    followed = solo.follow(samples, 0, samples[0][2][0])
    assert all(obs is None or obs.track_id == 7 for obs in followed)
    assert followed[20] is None


def test_after_a_long_gap_only_someone_who_looks_like_them_is_picked_up():
    def people(i):
        if 5 <= i < 25:
            return [_person(9, 450, BLUE)]
        return [_person(7 if i < 5 else 11, 120, RED), _person(9, 450, BLUE)]
    samples = _samples(people)
    followed = solo.follow(samples, 0, samples[0][2][0])
    assert followed[26] is not None and followed[26].track_id == 11
    assert all(followed[i] is None for i in range(5, 25))


class _FakeTracker:
    model_path = "fake.pt"

    def __init__(self):
        self.frames = 0

    def warmup(self, frame):
        pass

    def reset_tracking(self):
        pass

    def track(self, frame, imgsz=None):
        self.frames += 1
        return [_person(1, 100 + (self.frames % 20), RED)]


def _video(path: Path, seconds=3, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 320))
    for index in range(seconds * fps):
        frame = np.full((320, 320, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (100 + index % 20, 60), (180 + index % 20, 300), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def test_a_solo_analysis_covers_the_whole_video_and_scores_nothing(tmp_path):
    from core import analyzer

    video = _video(tmp_path / "bag.avi")
    req = AnalysisRequest(video_path=str(video), fighter_a_box=[100, 60, 180, 300], fighter_b_box=[],
                          analysis_target="A", solo=True, selection_seconds=2.0,
                          output_dir=str(tmp_path / "run"), persist_result=False)
    with patch.object(analyzer, "get_pose_tracker", return_value=_FakeTracker()):
        report = analyzer.analyze(req)
    assert report["mode"] == "solo"
    span = report["video"]["analysed_span"]
    assert span["start_seconds"] == 0.0 and span["whole_video"] is True
    assert report["scorecard"]["available"] is False
    assert report["statistics"]["fighters"] == {}
    assert set(report["metrics"]) == {"A"}
    assert "pressure_index" not in report["metrics"]["A"] and "attacks" not in report["metrics"]["A"]
    assert report["metrics"]["A"]["pose_coverage"] == pytest.approx(1.0)
    assert report["integrity"]["identity_evidence_trusted"] is True
    assert report["analysis_build"]["analysis_version"] >= 3
    run = tmp_path / "run"
    assert json.loads((run / "report.json").read_text())["mode"] == "solo"
    assert len((run / "tracking.jsonl").read_text().splitlines()) == report["tracking"]["analyzed_frames"]


def test_a_box_around_nobody_gives_an_untrusted_report(tmp_path):
    from core import analyzer

    video = _video(tmp_path / "card.avi")
    req = AnalysisRequest(video_path=str(video), fighter_a_box=[230, 10, 310, 120], fighter_b_box=[],
                          analysis_target="A", solo=True, selection_seconds=1.0,
                          output_dir=str(tmp_path / "run"), persist_result=False)
    with patch.object(analyzer, "get_pose_tracker", return_value=_FakeTracker()):
        report = analyzer.analyze(req)
    assert report["integrity"]["identity_evidence_trusted"] is False
    assert "could not find the person" in report["integrity"]["solo_note"]


def test_the_solo_page_renders_movement_and_no_strikes():
    from test_web import _real_template_env

    class Stub:
        def __init__(self, **kw): self.__dict__.update(kw)
        def __getattr__(self, k): return Stub()
        def __str__(self): return ""
        def __bool__(self): return False

    report = {"mode": "solo", "integrity": {"identity_evidence_trusted": True},
              "metrics": {"A": {"pose_coverage": 0.9, "footwork_body_lengths_per_second": 0.42,
                                "guard_index": 0.61, "balance_index": 0.8}}}
    html = _real_template_env().get_template("solo_result.html").render(
        request=Stub(state=Stub(csrf_token="t" * 43, account=None), url=Stub(path="/result/x")),
        report=report, job_id="x", asset_version="t",
        analysed_span={"label": "Whole video · 0:30", "note": None},
        analysis_build={"analysis_version": 3, "commit": "abc", "outdated": False})
    assert "61%" in html and "80%" in html and "Whole video" in html
    for claim in ("strikes thrown", "Estimated score", "Fighter B", "opponent's"):
        assert claim not in html
