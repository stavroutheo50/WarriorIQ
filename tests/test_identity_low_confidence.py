"""A fighter over the hand-off limit passes, labelled lower confidence, when the
boxes are shown to have stayed on the two people who fought.

2026-10-14, Athens hall, handheld phone: B read 14.9 suspicious hand-offs a
minute (limit 12) and the whole scorecard was withheld, although the boxes
followed both fighters about 97% of the time and 28 of 32 strikes checked by
eye were on the right fighter.
"""

from __future__ import annotations

from core.report import identity_low_confidence, identity_ready_by_fighter, label_low_confidence


def _tracking(**overrides):
    tracking = {
        "fighter_A_seed_source": "manual_anchor", "fighter_B_seed_source": "pose_detector",
        "initial_iou_A": 0.0, "initial_iou_B": 0.88,
        "fighter_A_coverage": 0.94, "fighter_B_coverage": 0.935,
        "fighters_separable": True, "fighter_pair_similarity": 0.144,
        "pair_similarity_method": "kit_regions_lab_v1", "kit_check_attempted": True,
        "identity_confusions": 0, "identity_confusions_per_minute": 0.0,
        "fighter_A_suspicious_handoffs_per_minute": 7.3,
        "fighter_B_suspicious_handoffs_per_minute": 14.9,
        "followed_share": [0.978, 0.964],
    }
    tracking.update(overrides)
    return tracking


def test_the_athens_fight_passes_with_lower_confidence():
    tracking = _tracking()
    assert identity_ready_by_fighter(tracking) == {"A": True, "B": True}
    assert identity_low_confidence(tracking) == {"A": False, "B": True}


def test_without_corroboration_the_gate_holds_as_before():
    # An older report, with no measure of whether the boxes stayed on the fighters.
    tracking = _tracking()
    del tracking["followed_share"]
    assert identity_ready_by_fighter(tracking)["B"] is False
    # One of the people who fought was not followed most of the time.
    assert identity_ready_by_fighter(_tracking(followed_share=[0.97, 0.6]))["B"] is False
    # Far over the limit: the wrong fights measured read 20 and more.
    assert identity_ready_by_fighter(_tracking(fighter_B_suspicious_handoffs_per_minute=20.1))["B"] is False


def test_the_scorecard_says_it_is_lower_confidence():
    scorecard = {"available": True, "totals": {"A": 29, "B": 28}}
    label_low_confidence(scorecard, _tracking())
    assert scorecard["confidence"] == "low"
    assert "Fighter B" in scorecard["confidence_note"]
    steady = {"available": True}
    label_low_confidence(steady, _tracking(fighter_B_suspicious_handoffs_per_minute=4.0))
    assert "confidence" not in steady
    withheld = {"available": False}
    label_low_confidence(withheld, _tracking())
    assert "confidence" not in withheld


def test_unchecked_outcome_calls_are_counted_as_estimates_only():
    from core.fight_stats import summarize_fight_events

    feed = [{"fighter": "A", "family": "punch", "time_seconds": t, "outcome": "unclassified",
             "verification": "observed", "estimated_outcome": o}
            for t, o in [(1, "landed"), (5, "missed"), (9, "uncertain"), (13, "blocked"), (17, "landed")]]
    a = summarize_fight_events(feed, None, 0, False)["fighters"]["A"]
    assert a["estimated"] == {"landed": 2, "missed": 1, "blocked": 1, "evaded": 0, "uncertain": 1}
    # The validated fields stay empty: an estimate is never presented as a checked outcome.
    assert a["landed"] is None and a["accuracy"] is None
    # Reports from before strikes carried the call show no estimate.
    for item in feed:
        del item["estimated_outcome"]
    assert summarize_fight_events(feed, None, 0, False)["fighters"]["A"]["estimated"] is None


def test_the_report_page_says_lower_confidence_beside_the_score():
    import json
    from pathlib import Path

    from tests_support import render_result

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
    scorecard = report.setdefault("scorecard", {})
    scorecard.update({"available": True, "totals": {"A": 29, "B": 28}, "winner_estimate": "A",
                      "rounds": [{"round": 1, "fighter_A": 29, "fighter_B": 28, "winner": "A"}]})
    label_low_confidence(scorecard, _tracking())
    page = render_result(report=report, numbers={"state": "ok"}, job_id="abc")
    assert "Lower confidence &middot; camera moved a lot" in page
    assert scorecard["confidence_note"] in page
