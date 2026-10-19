"""Audit of 14 Oct 2026, phase 2: fighter identity.

#2  re-identification after leaving the frame, an official class, camera-motion
    compensation, SAM2 on merged boxes, and low-coverage reports kept at low
    confidence instead of failing
#11 close fighters can be selected; a hand-drawn box is never refused
#12 a tap where boxes overlap picks the smallest box, ties to the nearest centre
#13 the default selection frame falls back to two confident, apart people
#24 a failed report names the real cause and the matching fix
#41 Fight Camp offers "Re-tag and re-run" on failed fights
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.config import SETTINGS
from core.identity import IdentityManager
from core.types import PersonObservation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "report_sample.json"


def _person(track_id, x1, x2, look, *, y1=80, y2=310, reid=None, referee=None):
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = np.linspace(x1 + 4, x2 - 4, 17)
    kp[:, 1] = np.linspace(y1 + 20, y2 - 20, 17)
    return PersonObservation(track_id=track_id, box=np.asarray([x1, y1, x2, y2], dtype=np.float32),
                             confidence=0.95, keypoints=kp, keypoint_conf=np.ones((17,), dtype=np.float32),
                             appearance=np.asarray(look, dtype=np.float32),
                             reid=None if reid is None else np.asarray(reid, dtype=np.float32),
                             referee_prob=referee)


def _look(index):
    vector = np.zeros((64,), dtype=np.float32)
    vector[index] = 1.0
    return vector


A_LOOK, B_LOOK, REF_LOOK = _look(1), _look(2), _look(3)


# ---------------------------------------------------------------------- #2
def test_a_fighter_who_left_the_frame_is_reidentified_by_appearance_where_they_return():
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0, source_fps=30)
    for frame in range(1, 6):
        manager.update([_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK)], frame)
    # A leaves the picture for two seconds.
    for frame in range(6, 66, 3):
        a, b = manager.update([_person(2, 400, 500, B_LOOK)], frame)
        assert a is None and b is not None
    # ...and walks back in on the far side, as a new tracker track.
    a, b = manager.update([_person(7, 900, 1000, A_LOOK), _person(2, 400, 500, B_LOOK)], 70)
    assert a is not None and a.track_id == 7
    assert b is not None and b.track_id == 2
    assert manager.reentries["A"] == 1


def test_a_far_stranger_who_does_not_look_like_the_fighter_is_still_refused():
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0, source_fps=30)
    for frame in range(1, 66, 3):
        manager.update([_person(2, 400, 500, B_LOOK)], frame)
    a, _ = manager.update([_person(8, 900, 1000, REF_LOOK), _person(2, 400, 500, B_LOOK)], 70)
    assert a is None


def test_a_brief_absence_does_not_open_the_distance_gate():
    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0, source_fps=30)
    manager.update([_person(2, 400, 500, B_LOOK)], 3)
    a, _ = manager.update([_person(7, 900, 1000, A_LOOK), _person(2, 400, 500, B_LOOK)], 6)
    assert a is None  # 0.2 s away: too soon to have left the picture and come back


def test_an_official_track_is_never_assigned_to_a_fighter_even_when_its_score_dips():
    manager = IdentityManager(_person(1, 100, 200, A_LOOK, referee=0.01),
                              _person(2, 400, 500, B_LOOK, referee=0.02), 0, source_fps=30)
    # The referee stands by for a few frames and is classed an official.
    for frame in range(1, 5):
        manager.update([_person(1, 100, 200, A_LOOK, referee=0.01), _person(2, 400, 500, B_LOOK, referee=0.02),
                        _person(9, 250, 350, REF_LOOK, referee=0.95)], frame)
    assert 9 in manager.official_tracks
    # A vanishes; the referee steps where A was with a frame scored only 0.3,
    # below the per-frame gate - he must still not become Fighter A.
    a, _ = manager.update([_person(2, 400, 500, B_LOOK, referee=0.02),
                           _person(9, 100, 200, A_LOOK, referee=0.3)], 5)
    assert a is None
    assert manager.rejections.get("official", 0) >= 1


def test_a_track_id_handed_to_a_fighter_is_not_refused_as_an_official():
    manager = IdentityManager(_person(1, 100, 200, A_LOOK, referee=0.01),
                              _person(2, 400, 500, B_LOOK, referee=0.02), 0, source_fps=30)
    manager.official_tracks.add(1)
    # Fighters score at most 0.084 on the probe; such a frame is the fighter's.
    a, _ = manager.update([_person(1, 100, 200, A_LOOK, referee=0.03), _person(2, 400, 500, B_LOOK)], 1)
    assert a is not None


def _scene(shift_x=0, shift_y=0):
    rng = np.random.default_rng(3)
    base = (rng.random((360, 640)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    frame = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(frame, matrix, (640, 360), borderMode=cv2.BORDER_REFLECT)


def test_the_camera_shift_between_two_frames_is_measured():
    from core.camera_motion import GlobalMotion

    motion = GlobalMotion()
    assert motion.update(_scene()) is None
    shift = motion.update(_scene(12, -5))
    assert shift is not None and shift.significant
    assert shift.dx == pytest.approx(12, abs=1.0) and shift.dy == pytest.approx(-5, abs=1.0)


def test_a_still_camera_is_not_compensated():
    from core.camera_motion import GlobalMotion

    motion = GlobalMotion()
    motion.update(_scene())
    shift = motion.update(_scene())
    assert shift is None or not shift.significant


def test_a_pan_moves_the_remembered_boxes_so_the_fighter_is_not_too_far():
    from core.camera_motion import CameraShift

    manager = IdentityManager(_person(1, 100, 200, A_LOOK), _person(2, 400, 500, B_LOOK), 0, source_fps=30)
    # The camera pans 450 px: both fighters move together in the picture.
    pan = CameraShift(matrix=np.float64([[1, 0, 450], [0, 1, 0]]), inliers=80)
    people = [_person(1, 550, 650, A_LOOK), _person(2, 850, 950, B_LOOK)]
    uncompensated = deepcopy(manager)
    a_raw, _ = uncompensated.update([_person(5, 550, 650, A_LOOK), _person(6, 850, 950, B_LOOK)], 1)
    assert manager.compensate_camera(pan)
    a, b = manager.update(people, 1)
    assert a is not None and b is not None
    assert a_raw is None  # without compensation the same pan lost the fighter
    assert manager.camera_compensations == 1


def test_a_fighter_swallowed_by_the_other_fighters_box_asks_for_sam_at_once():
    manager = IdentityManager(_person(1, 100, 220, A_LOOK), _person(2, 180, 300, B_LOOK), 0, source_fps=30)
    # Clinch: one detection around both, matched to B; A is missing.
    manager.update([_person(2, 100, 300, B_LOOK)], 1)
    assert manager.a.missing_frames == 1
    assert manager.merged_into_other(manager.a)
    previous = SETTINGS.sam_recovery_enabled
    object.__setattr__(SETTINGS, "sam_recovery_enabled", True)
    try:
        assert manager.needs_recovery(manager.a, analyzed_index=50)
    finally:
        object.__setattr__(SETTINGS, "sam_recovery_enabled", previous)


def _report(cov_a, cov_b, **tracking):
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report["tracking"].update({"fighter_A_coverage": cov_a, "fighter_B_coverage": cov_b, **tracking})
    return report


def test_low_coverage_is_reported_at_low_confidence_instead_of_failing():
    from core.report import identity_verdict, refresh_identity_integrity

    report = refresh_identity_integrity(_report(0.92, 0.32))
    verdict = identity_verdict(report)
    assert verdict["trusted"] is True and verdict["partial"] is True
    assert verdict["confidence"] == {"A": "high", "B": "low"}
    assert report["integrity"]["identity_partial"] is True
    assert report["integrity"]["identity_evidence_trusted"] is True


def test_too_little_coverage_still_fails():
    from core.report import identity_verdict

    verdict = identity_verdict(_report(0.92, 0.10))
    assert verdict["trusted"] is False and verdict["cause"]["cause"] == "coverage"


def test_only_rounds_where_both_were_followed_are_scored():
    from core.report import followed_rounds
    from core.types import RoundSpec

    tracking = {"coverage_windows": [
        {"start_seconds": 0, "end_seconds": 10, "fighter_A_coverage": 0.95, "fighter_B_coverage": 0.9, "analyzed_frames": 100},
        {"start_seconds": 10, "end_seconds": 20, "fighter_A_coverage": 0.95, "fighter_B_coverage": 0.2, "analyzed_frames": 100},
        {"start_seconds": 20, "end_seconds": 30, "fighter_A_coverage": 0.9, "fighter_B_coverage": 0.9, "analyzed_frames": 100},
    ]}
    rounds = [RoundSpec(1, 0, 10), RoundSpec(2, 10, 20), RoundSpec(3, 20, 30)]
    assert followed_rounds(tracking, rounds, 0.0, 0.85) == [1, 3]


def test_the_report_page_says_low_confidence():
    template = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
    assert "report.integrity.identity_partial" in template
    assert "Every number below comes only from the parts of the fight where they were followed." in template


# --------------------------------------------------------------------- #11
def test_two_close_tapped_fighters_are_allowed():
    from core.selection_check import pair_verdict

    a, b = [100, 50, 260, 400], [150, 50, 300, 400]   # IoU ~0.63, a clinch
    assert pair_verdict(a, b, both_detected=True)["verdict"] == "warn"
    assert pair_verdict([100, 50, 220, 400], [180, 50, 300, 400])["verdict"] == "ok"


def test_a_hand_drawn_box_is_never_refused():
    from core.selection_check import pair_verdict

    same = [100, 50, 260, 400]
    assert pair_verdict(same, list(same), b_drawn=True)["verdict"] == "warn"
    assert pair_verdict(same, list(same), a_drawn=True)["verdict"] == "warn"
    assert pair_verdict(same, list(same))["verdict"] == "block"


def test_start_accepts_overlapping_boxes_when_one_was_drawn(monkeypatch):
    """The server applies the same rule: a drawn pair past the old 28% rule starts."""
    from core.selection_check import pair_verdict

    a, b = [100, 50, 260, 400], [140, 50, 300, 400]
    old_shared = (120 * 350) / (160 * 350)
    assert old_shared >= 0.28                 # the old rule refused this pair
    assert pair_verdict(a, b, b_drawn=True)["verdict"] != "block"
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "if shared / max(1.0, smallest) >= 0.28" not in source


def test_select_page_sends_how_each_box_was_made():
    page = (ROOT / "app" / "templates" / "select.html").read_text(encoding="utf-8")
    assert "fighter_a_drawn:boxADrawn,fighter_b_drawn:boxBDrawn,both_detected:bothDetected()" in page
    assert "The two boxes cover the same person" not in page
    assert "if(iou>.6)return(boxADrawn||boxBDrawn||bothDetected())?'warn':'block'" in page


# --------------------------------------------------------------------- #12
def test_a_tap_on_overlapping_boxes_picks_the_smallest_not_the_merged_box():
    from core.selection_check import box_at

    merged = [100, 40, 400, 420]
    left, right = [110, 50, 250, 410], [240, 50, 390, 410]
    assert box_at((180, 200), [merged, left, right]) == left
    assert box_at((330, 200), [merged, left, right]) == right


def test_equal_boxes_go_to_the_nearest_centre():
    from core.selection_check import box_at

    first, second = [100, 50, 250, 400], [200, 50, 350, 400]
    assert box_at((240, 200), [first, second]) == second   # centres 175 and 275
    assert box_at((210, 200), [first, second]) == first
    assert box_at((210, 200), [first, second], exclude=second) == first


def test_select_page_breaks_ties_by_centre():
    page = (ROOT / "app" / "templates" / "select.html").read_text(encoding="utf-8")
    assert "tied=hits.filter(box=>area(box)<=smallest*1.15)" in page


# --------------------------------------------------------------------- #13
def test_without_a_clear_moment_the_frame_with_two_confident_apart_people_is_chosen(monkeypatch, tmp_path):
    from core import person_detect

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for index in range(60):
        frame = np.full((240, 320, 3), index * 4 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    def detect(frame):
        shade = int(frame[0, 0, 0])
        # Head-and-shoulders crops: never a strict "clear start".
        if shade < 80:
            return [{"box": [10, 0, 100, 239], "confidence": 0.9}]                       # one person
        return [{"box": [10, 0, 100, 239], "confidence": 0.8},
                {"box": [60 if shade < 160 else 180, 0, 160 if shade < 160 else 300, 239], "confidence": 0.8}]

    monkeypatch.setattr(person_detect, "_load", lambda: object())
    monkeypatch.setattr(person_detect, "detect_people", detect)
    moment = person_detect.find_clear_moment(video, samples=6)
    assert moment is not None and moment.get("relaxed")
    boxes = [moment["people"][i]["box"] for i in moment["pair"]]
    assert boxes[1][0] == 180   # the pair that does not overlap


# --------------------------------------------------------------------- #24
def test_the_failed_report_shows_the_cause_and_its_own_fix():
    template = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
    assert "Fix it in a few seconds" not in template
    for fix in ("The fix: steady the camera", "The fix: keep both fighters in the picture",
                "The fix: re-tap the fighters", "Re-tag and trim"):
        assert fix in template
    assert "identity_failure.headline" in template


def test_coverage_failure_is_its_own_cause():
    from core.report import identity_failure

    cause = identity_failure({"fighter_A_coverage": 0.92, "fighter_B_coverage": 0.10,
                              "fighter_A_seed_source": "pose_detector", "fighter_B_seed_source": "pose_detector",
                              "initial_iou_A": 0.9, "initial_iou_B": 0.9})
    assert cause["cause"] == "coverage"


# --------------------------------------------------------------------- #41
def test_fight_camp_offers_retag_and_rerun_on_failed_fights():
    template = (ROOT / "app" / "templates" / "camp.html").read_text(encoding="utf-8")
    assert template.count("Re-tag and re-run") >= 3
    assert 'href="/select/{{ item.job_id }}" data-retag-rerun' in template
