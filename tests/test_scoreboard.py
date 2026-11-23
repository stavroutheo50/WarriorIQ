"""tools/scoreboard.py: the pass/fail rules of the 95% target (2026-11-23)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import scoreboard  # noqa: E402


def _row(real, counted, labelled, a=(0, 0), b=(0, 0)):
    return {"real": real, "counted": counted, "labelled": labelled,
            "counted_A": a[0], "labelled_A": a[1], "counted_B": b[0], "labelled_B": b[1]}


def test_identity_needs_95_percent_and_no_swap_at_all():
    good = {"right": 96, "partial": 0, "swapped": 0, "other": 2, "missing": 2, "frames": 100}
    assert scoreboard.identity_verdict(good)["passes"] is True
    assert scoreboard.identity_verdict({**good, "right": 95, "swapped": 1})["passes"] is False   # one swap fails it
    assert scoreboard.identity_verdict({**good, "right": 90, "other": 8})["passes"] is False
    assert scoreboard.identity_verdict({"frames": 0})["passes"] is None                         # not measured


def test_ground_time_is_the_share_of_seconds_right():
    assert scoreboard.ground_verdict({"right": 646, "seconds": 674})["passes"] is True        # 95.8%
    assert scoreboard.ground_verdict({"right": 600, "seconds": 674})["passes"] is False
    assert scoreboard.ground_verdict({"seconds": 0})["passes"] is None


def test_machine_labelled_fights_never_count_toward_the_target():
    rows = {"claude_only": _row(10, 10, 10, (5, 5), (5, 5))}
    summary = scoreboard.strike_summary(rows, person=set())
    assert summary["passes"] is None and summary["machine"]["precision"] == 1.0


def test_strikes_need_precision_recall_and_totals():
    passing = {"f1": _row(96, 100, 100, (50, 50), (50, 50))}
    assert scoreboard.strike_summary(passing, {"f1"})["passes"] is True
    low_recall = {"f1": _row(46, 50, 100, (25, 50), (25, 50))}
    result = scoreboard.strike_summary(low_recall, {"f1"})
    assert result["passes"] is False and result["score"] == 0.46
    # Today's measured state (dataset/regression/human_strikes): 38% precision, 46% recall.
    today = {"human": _row(51, 136, 111, (70, 55), (66, 56))}
    assert scoreboard.strike_summary(today, {"human"})["passes"] is False


def test_a_total_is_close_enough_within_10_percent_or_2_strikes():
    assert scoreboard._within(12, 10) and not scoreboard._within(13, 10)
    assert scoreboard._within(108, 100) and not scoreboard._within(111, 100)


def test_the_table_says_not_measured_instead_of_inventing_a_number():
    board = {"measured_at": "2026-11-23T00:00:00+00:00", "target": 0.95,
             "identity": {"score": None, "passes": None, "note": "no marked frames"},
             "ground": {"score": 0.958, "passes": True, "note": "646/674"},
             "strikes": {"score": 0.38, "passes": False, "note": "precision 38%"},
             "movement": {"score": None, "passes": None, "note": "not measured yet"}}
    text = scoreboard.render(board)
    assert "not measured" in text and "PASS" in text and "below" in text and "95.8%" in text
