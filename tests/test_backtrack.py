"""The selection frame seeds identity; it no longer decides where analysis starts.

QA, 2026-09: picking a frame at 0:30 of a 1:56 video analysed only 1:26 - the
selection time was stored as the analysis start - and the automatically picked
frame silently cut the opening off too. These tests pin the hand-off that
follows the fighters back from the chosen frame, the check that confirms it,
and the request fields that carry the two times separately.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core import analyzer, backtrack
from core.types import AnalysisRequest, PersonObservation

FPS = 30.0


def _person(track_id, x1, x2, reid=None):
    return PersonObservation(
        track_id=track_id, box=np.asarray([x1, 80, x2, 310], dtype=np.float32),
        confidence=0.9, reid=None if reid is None else np.asarray(reid, dtype=np.float32))


def _initial(a_box, b_box, people):
    def nearest(box):
        return max(people, key=lambda p: -abs(float(p.box[0]) - float(box[0])))
    a = nearest(a_box)
    b = nearest(b_box) if nearest(b_box) is not a else None
    return a, b, 0.9, 0.9


RED, BLUE = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]


def _walk(frames, *, a_track=1, b_track=2, drop=(), swap_from=None, reids=(RED, BLUE)):
    """Samples seed-first, descending, with both fighters drifting apart."""
    for index, frame in enumerate(frames):
        people = []
        shift = index * 2.0
        ra, rb = reids
        if swap_from is not None and frame <= swap_from:
            ra, rb = rb, ra
        if frame not in drop:
            people.append(_person(a_track, 100 - shift, 200 - shift, ra))
        people.append(_person(b_track, 400 + shift, 500 + shift, rb))
        yield frame, people


A_BOX, B_BOX = [100, 80, 200, 310], [400, 80, 500, 310]


def test_samples_run_from_the_seed_down_to_the_requested_start():
    frames = backtrack.sample_frames(0, 900, FPS)
    assert frames[0] == 900 and frames[-1] == 0
    assert all(later > earlier for later, earlier in zip(frames, frames[1:]))
    # Ten a second at 30 fps.
    assert frames[1] == 897


def test_both_fighters_followed_to_the_start():
    frames = backtrack.sample_frames(0, 300, FPS)
    handoff = backtrack.follow_back(_walk(frames), A_BOX, B_BOX, FPS, 0, find_initial=_initial)
    assert handoff.moved
    assert handoff.frame == 0
    assert handoff.reason is None
    # A stays the left-hand person all the way back.
    assert handoff.a_box[0] < handoff.b_box[0]


def test_a_camera_cut_stops_the_hand_off_at_the_cut():
    frames = backtrack.sample_frames(0, 300, FPS)
    handoff = backtrack.follow_back(_walk(frames), A_BOX, B_BOX, FPS, 0,
                                    find_initial=_initial, cuts={150})
    assert handoff.reason == "camera_cut"
    assert handoff.frame == 150


def test_a_fighter_missing_too_long_stops_it():
    frames = backtrack.sample_frames(0, 300, FPS)
    gone = {f for f in frames if f < 200}
    handoff = backtrack.follow_back(_walk(frames, drop=gone), A_BOX, B_BOX, FPS, 0,
                                    find_initial=_initial)
    assert handoff.reason == "fighter_lost"
    assert handoff.frame >= 200


def test_a_short_gap_is_bridged():
    frames = backtrack.sample_frames(0, 300, FPS)
    handoff = backtrack.follow_back(_walk(frames, drop={150, 147, 144}), A_BOX, B_BOX, FPS, 0,
                                    find_initial=_initial)
    assert handoff.reason is None and handoff.frame == 0


def test_appearance_saying_they_swapped_stops_it():
    frames = backtrack.sample_frames(0, 300, FPS)
    handoff = backtrack.follow_back(_walk(frames, swap_from=120), A_BOX, B_BOX, FPS, 0,
                                    find_initial=_initial)
    assert handoff.reason == "unsure_who_is_who"
    assert handoff.frame > 120


def test_identical_kits_do_not_trigger_a_swap():
    """Same appearance on both: no evidence either way, so motion carries it."""
    frames = backtrack.sample_frames(0, 300, FPS)
    handoff = backtrack.follow_back(_walk(frames, reids=(RED, RED)), A_BOX, B_BOX, FPS, 0,
                                    find_initial=_initial)
    assert handoff.reason is None and handoff.frame == 0


def test_no_track_at_the_seed_means_no_hand_off():
    frames = backtrack.sample_frames(0, 300, FPS)
    handoff = backtrack.follow_back(_walk(frames, a_track=None), A_BOX, B_BOX, FPS, 0,
                                    find_initial=_initial)
    assert not handoff.moved
    assert handoff.reason == "not_detected_at_seed"


@pytest.mark.parametrize("a_box,b_box,expected", [
    (A_BOX, B_BOX, "match"),
    (B_BOX, A_BOX, "swapped"),
    ([700, 80, 800, 310], B_BOX, "unknown"),
])
def test_seed_verdict(a_box, b_box, expected):
    a = _person(1, a_box[0], a_box[2])
    b = _person(2, b_box[0], b_box[2])
    assert backtrack.seed_verdict(a, b, A_BOX, B_BOX) == expected


def test_seed_verdict_with_one_fighter_missing_is_partial():
    assert backtrack.seed_verdict(_person(1, 100, 200), None, A_BOX, B_BOX) == "partial"


# --- the analyser's use of it --------------------------------------------------

def _request(**kw):
    base = AnalysisRequest(video_path="fight.mp4", fighter_a_box=A_BOX, fighter_b_box=B_BOX,
                           start_seconds=0.0, selection_seconds=30.0)
    return replace(base, **kw)


INFO = SimpleNamespace(fps=FPS, duration=116.0, width=1280, height=720, frame_count=3480)


def _run(handoff, analyze_side_effect=None):
    calls = []

    def fake_analyze(req, progress_callback=None, **kwargs):
        calls.append((req, kwargs))
        if analyze_side_effect and len(calls) == 1:
            raise analyze_side_effect
        return {"ok": True}

    with patch.object(analyzer, "get_video_info", return_value=INFO), \
         patch.object(analyzer, "get_pose_tracker", return_value=object()), \
         patch.object(analyzer._backtrack, "backtrack", return_value=handoff), \
         patch.object(analyzer, "_analyze", side_effect=fake_analyze):
        analyzer._analyze_from_seed(_request(), None)
    return calls


def test_a_successful_hand_off_analyses_from_the_start_and_checks_the_seed():
    handoff = backtrack.Handoff(seed_frame=900, requested_start_frame=0, frame=0,
                                a_box=[90, 80, 190, 310], b_box=[410, 80, 510, 310])
    (req, kwargs), = _run(handoff)
    assert req.start_seconds == 0.0
    assert req.fighter_a_box == [90, 80, 190, 310]
    assert kwargs["seed_check"]["frame"] == 900
    assert kwargs["seed_check"]["a_box"] == A_BOX
    assert kwargs["seed_seconds"] == 30.0


def test_a_failed_seed_check_reruns_from_the_seed_and_says_why():
    handoff = backtrack.Handoff(seed_frame=900, requested_start_frame=0, frame=0,
                                a_box=[90, 80, 190, 310], b_box=[410, 80, 510, 310])
    calls = _run(handoff, analyze_side_effect=analyzer.SeedCheckFailed("swapped"))
    assert len(calls) == 2
    rerun, kwargs = calls[1]
    assert rerun.start_seconds == 30.0
    assert rerun.fighter_a_box == A_BOX
    assert kwargs["excluded_reason"] == "unverified"
    assert "seed_check" not in kwargs


def test_no_hand_off_starts_at_the_seed_and_records_the_reason():
    handoff = backtrack.Handoff(seed_frame=900, requested_start_frame=0, frame=900, reason="camera_cut")
    (req, kwargs), = _run(handoff)
    assert req.start_seconds == 30.0
    assert kwargs["excluded_reason"] == "camera_cut"


def test_analysed_span_is_stated_plainly():
    span = analyzer._analysed_span(INFO, 30.0, 116.0, 30.0, None, "camera_cut")
    assert span["whole_video"] is False
    assert span["excluded_start_seconds"] == 30.0
    assert span["excluded_reason_text"] == backtrack.REASONS["camera_cut"]
    whole = analyzer._analysed_span(INFO, 0.0, 116.0, 30.0, None, None)
    assert whole["whole_video"] is True and whole["excluded_reason"] is None


# --- the job fields ------------------------------------------------------------

def test_web_request_analyses_the_whole_video_and_seeds_at_the_frame():
    import app.main as web

    job = {"video_path": "v.mp4", "fight_type": "competition", "ruleset": "K1",
           "start_seconds": 30.0, "selection_seconds": 30.0, "requested_start_seconds": 0.0,
           "round_count": 1, "round_duration_seconds": 116.0, "break_duration_seconds": 0.0}
    req = web._analysis_request("abc", job, A_BOX, B_BOX, "A")
    assert req.start_seconds == 0.0 and req.selection_seconds == 30.0


def test_a_job_stored_before_the_change_is_analysed_from_the_start_too():
    import app.main as web

    legacy = {"video_path": "v.mp4", "fight_type": "competition", "ruleset": "K1",
              "start_seconds": 30.0, "round_count": 1, "round_duration_seconds": 116.0,
              "break_duration_seconds": 0.0}
    req = web._analysis_request("abc", legacy, A_BOX, B_BOX, "A")
    assert req.start_seconds == 0.0 and req.selection_seconds == 30.0


def test_remote_payload_keeps_old_workers_on_the_old_behaviour():
    import app.main as web
    import worker

    job = {"video_path": "v.mp4", "fight_type": "competition", "ruleset": "K1",
           "start_seconds": 30.0, "selection_seconds": 30.0, "requested_start_seconds": 0.0,
           "fighter_a_box": A_BOX, "fighter_b_box": B_BOX, "analysis_run_id": "ab12"}
    payload = web._remote_job_payload("abc", job)
    # An old worker reads only start_seconds and seeds there, as it always did.
    assert payload["start_seconds"] == 30.0
    req = worker._request_from_job("abc", {**payload, "video_path": "v.mp4"})
    assert req.start_seconds == 0.0 and req.selection_seconds == 30.0


def test_report_says_what_it_covers():
    import app.main as web

    whole = web._analysed_span_summary({"video": {"analysed_span": {
        "start_seconds": 0.0, "end_seconds": 116.0, "video_duration_seconds": 116.0}}})
    assert whole["label"] == "Whole video · 1:56" and whole["note"] is None
    cut = web._analysed_span_summary({"video": {"analysed_span": {
        "start_seconds": 30.0, "end_seconds": 116.0, "video_duration_seconds": 116.0,
        "excluded_reason_text": backtrack.REASONS["camera_cut"]}}})
    assert cut["label"] == "0:30–1:56 of 1:56"
    assert "first 0:30" in cut["note"] and "cuts to a different shot" in cut["note"]


def test_an_older_truncated_report_is_labelled_too():
    import app.main as web

    old = {"setup": {"start_seconds": 30.0}, "rounds": [{"start_seconds": 30.0, "end_seconds": 116.0}],
           "performance": {"segment_duration_seconds": 86.0}}
    summary = web._analysed_span_summary(old)
    assert summary["label"] == "0:30–1:56 of 1:56"
    assert "first 0:30" in summary["note"]
