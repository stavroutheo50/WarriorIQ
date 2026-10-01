"""A strength or weakness is only named when the difference is real.

QA 2026-09: a taekwondo report said "Work on: Guard 7% - You 7%, them 8% -
behind your opponent here." A one-point gap is noise, and guard height means
something different in taekwondo than in boxing or K-1.
"""

from __future__ import annotations

import numpy as np

from core.coaching import build_pose_coaching, build_training_plan, clear_difference
from core.metrics import MetricsAccumulator


def _fighter(guard, balance, guard_error=None, balance_error=None):
    own = {"guard_index": guard, "balance_index": balance}
    if guard_error is not None:
        own["spread"] = {"guard_index": {"blocks": 30, "standard_error": guard_error},
                         "balance_index": {"blocks": 30, "standard_error": balance_error}}
    return own


def test_a_one_point_gap_is_not_a_weakness():
    coaching = build_pose_coaching("A", _fighter(0.07, 0.70), _fighter(0.08, 0.70))
    titles = [item["title"] for item in coaching["improvements"]]
    assert not any(title.startswith("Work on") for title in titles), titles
    assert titles == ["Nothing behind your opponent"]
    assert coaching["drills"] == []
    assert "behind your opponent" not in " ".join(item["detail"] for item in coaching["improvements"])


def test_a_large_gap_is_named_and_drilled():
    coaching = build_pose_coaching("A", _fighter(0.07, 0.70), _fighter(0.25, 0.70))
    assert coaching["improvements"][0]["title"] == "Work on: Guard 7%"
    assert "behind your opponent" in coaching["improvements"][0]["detail"]
    assert build_training_plan(coaching, "A", _fighter(0.07, 0.70))


def test_a_gap_inside_the_noise_is_not_named():
    """Ten points apart, but each average is uncertain by eight."""
    noisy = build_pose_coaching("A", _fighter(0.10, 0.70, 0.08, 0.01), _fighter(0.20, 0.70, 0.08, 0.01))
    assert [item["title"] for item in noisy["improvements"]] == ["Nothing behind your opponent"]
    steady = build_pose_coaching("A", _fighter(0.10, 0.70, 0.01, 0.01), _fighter(0.20, 0.70, 0.01, 0.01))
    assert steady["improvements"][0]["title"] == "Work on: Guard 10%"


def test_clear_difference_needs_both_size_and_certainty():
    assert not clear_difference("guard_index", 0.07, 0.08)
    assert clear_difference("guard_index", 0.07, 0.20)
    assert not clear_difference("guard_index", 0.07, 0.20, {"spread": {"guard_index": {"standard_error": 0.1}}},
                                {"spread": {"guard_index": {"standard_error": 0.1}}})


def test_a_strength_needs_a_real_lead():
    level = build_pose_coaching("A", _fighter(0.21, 0.70), _fighter(0.20, 0.70))
    assert level["strengths"] == []
    ahead = build_pose_coaching("A", _fighter(0.30, 0.70), _fighter(0.20, 0.70))
    assert ahead["strengths"][0]["title"] == "Guard: ahead of your opponent"


def test_taekwondo_guard_is_style_not_a_fault():
    low_hands = _fighter(0.05, 0.70)
    opponent = _fighter(0.30, 0.70)
    boxing = build_pose_coaching("A", low_hands, opponent, "boxing")
    taekwondo = build_pose_coaching("A", low_hands, opponent, "taekwondo")
    assert boxing["improvements"][0]["title"] == "Work on: Guard 5%"
    assert not any("Guard" in item["title"] for item in taekwondo["improvements"] + taekwondo["strengths"])
    assert all(drill["metric"] != "guard_index" for drill in taekwondo["drills"])
    assert "taekwondo" in taekwondo["improvements"][0]["detail"]
    # Still measured and shown, just not ranked.
    assert "guard 5%" in taekwondo["baseline_summary"]


def test_one_fighter_is_only_told_about_numbers_well_outside_the_usual_range():
    ordinary = build_pose_coaching("A", _fighter(0.15, 0.70))
    assert ordinary["improvements"][0]["title"] == "Nothing clearly below the usual range"
    assert ordinary["drills"] == []
    low = build_pose_coaching("A", _fighter(0.02, 0.70))
    assert low["improvements"][0]["title"] == "Work on: Guard 2%"


def test_metrics_carry_a_block_standard_error():
    metrics = MetricsAccumulator(width=640, height=480)
    rng = np.random.default_rng(3)
    samples = [(index / 30.0, 0.2 + 0.05 * rng.standard_normal()) for index in range(30 * 20)]
    spread = metrics._spread(samples)
    assert spread["blocks"] == 10
    # Over blocks, not frames: 600 frames would claim an error ~sqrt(60) times smaller.
    assert 0.0 < spread["standard_error"] < 0.02
    assert metrics._spread(samples[:10]) is None
