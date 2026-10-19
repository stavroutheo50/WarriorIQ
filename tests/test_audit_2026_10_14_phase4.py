"""Audit of 14 Oct 2026, phase 4: metrics honesty.

#5  guard is wrist-to-chin over shoulder width, both hands, smoothed, only on
    confident keypoints - tested on synthetic poses - and labelled Experimental
#6  Fight Camp no longer calls the figures "real"
#7  "Why this score", and no score while the strike numbers are withheld
#8  one rule for strike numbers on report, replay, Compare and Fight Camp
#28 drill targets only for reliable measurements, at their reference line
#35 one noise-based rule for "level" on Compare
#38 "Fighters told apart only N% of the time"
#39 changes in percentages are percentage points
#40 every change carries a unit
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# COCO keypoint indices used by the guard.
NOSE, L_EYE, R_EYE, L_SHOULDER, R_SHOULDER = 0, 1, 2, 5, 6
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12


def pose(left_wrist=(95, 150), right_wrist=(125, 150), *, wrist_conf=0.9, scale=1.0, shoulder_conf=0.9):
    """A front-on fighter: shoulders 60 px apart at y=180, nose at (110, 120), hips at y=300.

    The chin falls 35% of the way from the head to the neck: about (110, 141).
    """
    points = np.zeros((17, 2), dtype=np.float32)
    conf = np.full((17,), 0.05, dtype=np.float32)
    def put(index, xy, c):
        points[index] = np.asarray(xy, dtype=np.float32) * scale
        conf[index] = c
    put(NOSE, (110, 120), 0.9)
    put(L_EYE, (104, 115), 0.9)
    put(R_EYE, (116, 115), 0.9)
    put(L_SHOULDER, (80, 180), shoulder_conf)
    put(R_SHOULDER, (140, 180), shoulder_conf)
    put(L_HIP, (90, 300), 0.9)
    put(R_HIP, (130, 300), 0.9)
    if left_wrist is not None:
        put(L_WRIST, left_wrist, wrist_conf)
    if right_wrist is not None:
        put(R_WRIST, right_wrist, wrist_conf)
    return points, conf


# ---------------------------------------------------------------------- #5
def test_both_hands_at_the_chin_is_hands_up():
    from core.guard import is_up, reading

    value, why = reading(*pose())
    assert why is None and is_up(value)


def test_one_hand_dropped_is_not_a_guard():
    from core.guard import is_up, reading

    # Left hand at the chin, right hand at the hip.
    value, why = reading(*pose(right_wrist=(130, 300)))
    assert why is None and not is_up(value)


def test_the_reading_is_in_shoulder_widths_not_pixels():
    from core.guard import reading

    small, _ = reading(*pose(right_wrist=(130, 300)))
    big, _ = reading(*pose(right_wrist=(130, 300), scale=3.0))
    assert big == pytest.approx(small, rel=1e-4)


def test_low_confidence_keypoints_give_no_reading():
    from core.guard import UNCLEAR, reading

    assert reading(*pose(wrist_conf=0.2)) == (None, UNCLEAR)
    assert reading(*pose(shoulder_conf=0.2)) == (None, UNCLEAR)
    # One wrist not seen: the frame cannot say whether both hands were up.
    assert reading(*pose(right_wrist=None)) == (None, UNCLEAR)


def test_a_fighter_too_small_in_the_picture_is_not_measured():
    from core.guard import TOO_SMALL, reading

    # Shoulders 3 px apart, torso 6 px: well under the 8 px scale floor.
    assert reading(*pose(scale=0.05))[1] == TOO_SMALL


def test_one_misplaced_frame_is_smoothed_away():
    from core.guard import reading, summarise

    up, _ = reading(*pose())
    down, _ = reading(*pose(right_wrist=(130, 300)))
    # Ten frames a second for two seconds, hands up, with one bad frame.
    readings = [(i / 10, up) for i in range(20)]
    readings[10] = (1.0, down)
    assert summarise(readings)["share"] == 1.0


def test_guard_is_marked_experimental_and_never_targeted():
    from core.metric_catalog import BY_KEY, is_experimental, reference_target

    assert is_experimental("guard_index") and BY_KEY["guard_index"].label == "Guard (experimental)"
    assert reference_target("guard_index", 0.07) is None


def test_every_page_showing_guard_carries_the_badge():
    templates = ROOT / "app" / "templates"
    for name in ("result.html", "camp.html", "compare.html", "shared.html", "solo_result.html"):
        assert "experimental_badge" in (templates / name).read_text(encoding="utf-8"), name


# ---------------------------------------------------------------------- #6
def test_fight_camp_no_longer_calls_the_figures_real():
    camp = (ROOT / "app" / "templates" / "camp.html").read_text(encoding="utf-8")
    assert "are real" not in camp
    assert "Guard is experimental" in camp


# ---------------------------------------------------------------------- #7
def test_why_this_score_names_what_was_counted_the_weights_and_the_rounds():
    from core.scoring import score_explanation

    card = {"available": True, "ruleset": "K1", "ruleset_label": "K-1",
            "rounds": [{"round": 1, "fighter_A": 10, "fighter_B": 10, "winner": "EVEN"}]}
    lines = score_explanation(card, {"A": "Theo", "B": "Opponent"})
    assert lines[0].startswith("The score counts landed strikes only, not every attempt")
    assert any("kick counts 1.15" in line for line in lines)
    assert score_explanation({"available": False}) == []


def test_why_this_score_walks_through_ten_point_rounds():
    from core.scoring import RULESETS, score_explanation

    key = next(key for key, profile in RULESETS.items() if profile.ten_point_must and not profile.round_margins)
    card = {"available": True, "ruleset": key, "rounds": [
        {"round": 1, "fighter_A": 10, "fighter_B": 8, "winner": "A", "effective_action_A": 9, "effective_action_B": 2},
        {"round": 2, "fighter_A": 9, "fighter_B": 10, "winner": "B", "effective_action_A": 3, "effective_action_B": 4},
        {"round": 3, "fighter_A": 9, "fighter_B": 10, "winner": "B", "effective_action_A": 3, "effective_action_B": 4}]}
    text = " ".join(score_explanation(card, {"A": "Theo", "B": "Opponent"}))
    assert "round 2 to Opponent (3 to 4), 9-10" in text
    assert "can lead with fewer strikes overall" in text


def test_the_report_shows_why_this_score():
    template = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
    assert "<strong>Why this score</strong>" in template


def test_no_score_while_the_strike_numbers_are_withheld():
    import app.main as web

    report = {"scorecard": {"available": True, "sport": "kickboxing", "totals": {"A": 29, "B": 28},
                            "rounds": [{"round": 1}]},
              "integrity": {"identity_evidence_trusted": True}}
    with mock.patch.object(web, "strike_numbers_shown", return_value=()):
        web._withhold_score_while_counts_are_off(report)
    assert report["scorecard"]["available"] is False and report["scorecard"]["totals"] == {"A": None, "B": None}


# ---------------------------------------------------------------------- #8
def test_one_strike_rule_for_every_page():
    from core.report import strike_numbers_shown

    trusted = {"integrity": {"identity_evidence_trusted": True}, "scorecard": {"sport": "kickboxing"}}
    failed = {"integrity": {"identity_evidence_trusted": False}, "scorecard": {"sport": "kickboxing"}}
    assert strike_numbers_shown(failed) == ()
    shown = strike_numbers_shown(trusted)
    import app.main as web
    from core.progress_insights import build_progress

    # Compare and Fight Camp read the same function as the report and replay.
    assert "strikes_shown" in (ROOT / "app" / "templates" / "compare.html").read_text(encoding="utf-8")
    assert "hidden until WarriorIQ can count strikes accurately" not in (
        ROOT / "app" / "templates" / "compare.html").read_text(encoding="utf-8")
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert source.count("strike_numbers_shown(") >= 4
    record = [{"created_at": "2026-10-01", "job_id": "x", "report": {
        **trusted, "video": {"focus_fighter": "A"}, "setup": {"ruleset": "K1"},
        "metrics": {"A": {"pose_coverage": .9, "attacks": {"families": {"punch": 3, "kick": 2}}}}}}]
    assert build_progress(record, "A")["latest"]["action_trusted"] is bool(shown)
    assert web.strike_numbers_shown is strike_numbers_shown


# --------------------------------------------------------------------- #28
def test_targets_come_from_the_reference_line_not_plus_eight_points():
    from core.coaching import metric_goal
    from core.metric_catalog import BALANCE_WEAKNESS

    goal, show = metric_goal("balance_index", 0.30)
    assert goal == BALANCE_WEAKNESS and show(goal) == "48.0%"
    assert metric_goal("balance_index", 0.70)[0] is None          # already past the line
    assert metric_goal("guard_index", 0.073)[0] is None           # experimental
    assert metric_goal("ring_center_control", 0.40)[0] is None    # no reference range
    assert metric_goal("footwork_body_lengths_per_second", 0.8)[0] is None


def test_the_plan_says_why_there_is_no_number():
    from core.coaching import build_training_plan

    drill = {"name": "Guard-return audit", "prescription": "4 x 90 sec", "metric": "guard_index",
             "label": "Guard (experimental)", "measured": 0.073, "opponent": 0.2}
    goal = build_training_plan({"drills": [drill], "evidence_type": "pose_only"}, "A", {})[0]["goal"]
    assert "15.3%" not in goal and "No number target: guard is experimental" in goal


# --------------------------------------------------------------------- #35
def _side(**metrics):
    return {"video": {"focus_fighter": "A"}, "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9},
            "metrics": {"A": metrics}}


def test_level_follows_each_measurements_noise_not_its_scale():
    from core.squad import compare_movement

    rows = {row["key"]: row for row in compare_movement([
        _side(balance_index=0.52, footwork_body_lengths_per_second=0.8),
        _side(balance_index=0.55, footwork_body_lengths_per_second=0.7)])["rows"]}
    # 3 points of balance (noise 4) and 0.1 body lengths a second (noise
    # 0.15) are both inside their measurement's noise.
    assert rows["balance_index"]["leader"] == "level"
    assert rows["footwork_body_lengths_per_second"]["leader"] == "level"
    rows = {row["key"]: row for row in compare_movement([
        _side(balance_index=0.40, footwork_body_lengths_per_second=0.5),
        _side(balance_index=0.60, footwork_body_lengths_per_second=1.2)])["rows"]}
    assert rows["balance_index"]["leader"] == "b"
    assert rows["footwork_body_lengths_per_second"]["leader"] == "b"


def test_a_measured_spread_can_keep_a_bigger_gap_level():
    from core.squad import compare_movement

    noisy = {"balance_index": {"standard_error": 0.08}}
    rows = {row["key"]: row for row in compare_movement([
        _side(balance_index=0.50, spread=noisy), _side(balance_index=0.62, spread=noisy)])["rows"]}
    assert rows["balance_index"]["leader"] == "level"     # 12 points, but 1.96 x 0.113 = 22


# ------------------------------------------------------------ #38 #39 #40
def test_failed_fights_say_how_often_the_fighters_were_told_apart():
    camp = (ROOT / "app" / "templates" / "camp.html").read_text(encoding="utf-8")
    assert "Fighters told apart only" in camp
    assert "pts since last fight" not in camp
    assert "percentage point{{'' if delta_pts|abs == 1 else 's'}} since last fight" in camp
    assert '<abbr title="percentage point' in camp


def test_changes_carry_their_unit():
    from core.progress_insights import change_words

    assert change_words(1, "percent") == "+1 percentage point"
    assert change_words(-13, "percent") == "-13 percentage points"
    assert change_words(7, "pressure") == "+7 on the 0-100 scale"
    assert change_words(0.3, "rate") == "+0.3 body lengths a second"
