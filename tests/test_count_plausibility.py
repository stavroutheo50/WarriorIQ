"""Counts no real fight could produce are withheld, with the reason."""

from unittest import mock

from core import count_plausibility, report
from core.fight_stats import summarize_fight_events


def _strikes(fighter, n, landed=0):
    return [{"fighter": fighter, "family": "punch", "technique": "jab", "round_number": 1,
             "time_seconds": i * 0.5 + (0 if fighter == "A" else 0.25),
             "outcome": "landed" if i < landed else "missed"} for i in range(n)]


def test_a_real_busy_fight_passes():
    # 120 thrown in five minutes is the 99th percentile of UFC rounds: real, not withheld.
    fighters = {"A": {"attempts": 120, "landed": 60}, "B": {"attempts": 40, "landed": 20}}
    assert count_plausibility.check(fighters, 300) == {"implausible": False, "reasons": []}


def test_an_impossible_rate_is_caught():
    result = count_plausibility.check({"A": {"attempts": 400, "landed": None}, "B": {}}, 300)
    assert result["implausible"] and "80 strikes a minute" in result["reasons"][0]


def test_a_short_clip_is_never_judged_on_rate():
    assert not count_plausibility.check({"A": {"attempts": 40}}, 20)["implausible"]


def test_an_impossible_landed_share_is_caught_only_with_enough_strikes():
    assert count_plausibility.check({"A": {"attempts": 30, "landed": 30}}, 300)["implausible"]
    assert count_plausibility.check({"A": {"attempts": 30, "landed": 0}}, 300)["implausible"]
    assert not count_plausibility.check({"A": {"attempts": 10, "landed": 10}}, 300)["implausible"]
    assert not count_plausibility.check({"A": {"attempts": 30, "landed": None}}, 300)["implausible"]


def test_statistics_carry_the_verdict_and_withhold_counts():
    stats = summarize_fight_events(_strikes("A", 400) + _strikes("B", 10), {"A": 100, "B": 100}, 100,
                                   trusted=False, processed_seconds=120)
    assert stats["plausibility"]["implausible"]
    assert stats["attempt_counts_available"] is False and stats["action_labels_available"] is False
    normal = summarize_fight_events(_strikes("A", 40) + _strikes("B", 30), {"A": 100, "B": 100}, 100,
                                    trusted=False, processed_seconds=300)
    assert normal["attempt_counts_available"] is True and not normal["plausibility"]["implausible"]


def test_an_implausible_report_shows_no_strike_families_even_with_counts_published():
    bad = {"statistics": {"plausibility": {"implausible": True, "reasons": ["x"]}}}
    with mock.patch.object(report, "STRIKE_COUNTS_PUBLISHED", True):
        assert report.published_families("kickboxing", bad) == ()
        assert report.published_families("kickboxing", {"statistics": {}}) != ()
        assert report.published_families("kickboxing") != ()


def test_an_implausible_report_loses_its_estimated_score():
    from app import main

    scorecard = {"available": True, "totals": {"A": 30, "B": 27}, "rounds": [1], "winner_estimate": "A"}
    bad = {"scorecard": dict(scorecard), "statistics": {"plausibility": {"implausible": True}}}
    with mock.patch.object(main, "STRIKE_COUNTS_PUBLISHED", True):
        main._withhold_score_while_counts_are_off(bad)
        good = {"scorecard": dict(scorecard), "statistics": {}}
        main._withhold_score_while_counts_are_off(good)
    assert bad["scorecard"]["available"] is False and bad["scorecard"]["status"] == "strike_counts_implausible"
    assert good["scorecard"]["available"] is True
