"""The report's head-to-head stat sheet (core/stat_sheet.py, 2026-10-28)."""

from __future__ import annotations

from core.stat_sheet import build_stat_sheet


def _report(**overrides):
    report = {
        "statistics": {"fighters": {
            "A": {"punch_attempts": 13, "kick_attempts": 1, "knee_attempts": 0, "activity_rate": 9.33},
            "B": {"punch_attempts": 9, "kick_attempts": 1, "knee_attempts": 0, "activity_rate": 6.67},
        }},
        "metrics": {
            "A": {"footwork_body_lengths_per_second": .71, "pressure_index": .12, "ring_center_control": .41,
                  "balance_index": .68, "pose_coverage": .93,
                  "numbers": {"distance_body_lengths": 58.4, "forward_seconds": 19.7, "backward_seconds": 16.2,
                              "hands_up_share": .037, "off_balance_count": 8,
                              "series": {"points": [{"t": 0, "speed": .2}, {"t": 5, "speed": .9}, {"t": 10, "speed": .4}]}}},
            "B": {"footwork_body_lengths_per_second": .55, "pressure_index": -.1, "ring_center_control": .3,
                  "balance_index": .7, "pose_coverage": .93,
                  "numbers": {"distance_body_lengths": 40.0, "forward_seconds": 12.0, "backward_seconds": 20.0,
                              "hands_up_share": .15, "off_balance_count": 5,
                              "series": {"points": [{"t": 0, "speed": .1}, {"t": 5, "speed": .5}, {"t": 10, "speed": 1.2}]}}},
        },
        "events": [{"fighter": "A", "attempted": True, "family": "punch", "peak_time": 3.0},
                   {"fighter": "A", "attempted": True, "family": "punch", "peak_time": 4.0},
                   {"fighter": "B", "attempted": True, "family": "kick", "peak_time": 15.0},
                   {"fighter": "B", "attempted": False, "family": "punch", "peak_time": 16.0}],
        "defenses": [{"fighter": "B"}, {"fighter": "A"}, {"fighter": "B"}],
        "rounds": [{"end_seconds": 25.0}],
    }
    report.update(overrides)
    return report


def _rows(sheet, group):
    return {row["label"]: row for row in sheet[group]}


def test_strike_rows_add_up_the_published_families_and_keep_the_estimate_label():
    sheet = build_stat_sheet(_report(), ["punch", "kick"], True)
    rows = _rows(sheet, "strikes")
    assert (rows["Strikes thrown"]["a"], rows["Strikes thrown"]["b"]) == (14, 10)
    assert rows["Strikes thrown"]["note"] == "estimate"
    assert (rows["Punches"]["a_text"], rows["Kicks"]["b_text"]) == ("13", "1")
    assert "Knees" not in rows                        # not a family this sport publishes
    assert rows["Strikes thrown"]["share_a"] == round(14 / 24 * 100, 1)


def test_no_strike_numbers_when_counts_are_not_published_or_fighters_not_told_apart():
    for families, trusted in ((["punch"], False), ([], True)):
        sheet = build_stat_sheet(_report(), families, trusted)
        assert sheet["strikes"] == [] and sheet["timeline"] is None
        assert sheet["movement"]                       # movement does not depend on strike counts


def test_movement_rows_use_the_reports_own_scales():
    rows = _rows(build_stat_sheet(_report(), ["punch"], True), "movement")
    assert rows["Pressure"]["a_text"] == "56"          # (0.12 + 1) / 2 * 100, as on the report tile
    assert rows["Time in the centre"]["a_text"] == "41%"
    assert rows["Guard up"]["note"] == "experimental"
    assert (rows["Defences seen"]["a"], rows["Defences seen"]["b"]) == (1, 2)


def test_a_stat_either_fighter_lacks_is_left_out_not_shown_as_zero():
    report = _report()
    del report["metrics"]["B"]["balance_index"]
    rows = _rows(build_stat_sheet(report, ["punch"], True), "movement")
    assert "Balance" not in rows


def test_the_timeline_counts_attempts_per_ten_seconds():
    timeline = build_stat_sheet(_report(), ["punch", "kick"], True)["timeline"]
    assert [(b["A"], b["B"]) for b in timeline["buckets"]] == [(2, 0), (0, 1), (0, 0)]
    assert timeline["peak"] == 2 and timeline["buckets"][0]["a_height"] == 100.0


def test_the_pace_graph_is_drawn_from_both_fighters_series():
    pace = build_stat_sheet(_report(), [], True)["pace"]
    assert pace["max_speed"] == 1.2 and len(pace["a"].split()) == 3 and pace["end_label"] == "0:10"


def test_an_empty_report_has_no_sheet():
    assert build_stat_sheet({}, ["punch"], True)["available"] is False


def test_the_timeline_counts_the_same_strikes_as_the_totals():
    feed = [{"kind": "strike", "fighter": "A", "family": "punch", "time_seconds": 1.0, "confidence": .5},
            {"kind": "strike", "fighter": "A", "family": "punch", "time_seconds": 1.2, "confidence": .6},   # the same punch twice
            {"kind": "strike", "fighter": "B", "family": "kick", "time_seconds": 12.0, "confidence": .5}]
    report = _report(event_feed=feed)
    timeline = build_stat_sheet(report, ["punch", "kick"], True)["timeline"]
    assert sum(b["A"] for b in timeline["buckets"]) == 1 and sum(b["B"] for b in timeline["buckets"]) == 1
