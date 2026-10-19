"""The public sample report (audit 2026-10-14, #54).

EVERY NUMBER HERE IS INVENTED. Nothing in this file comes from an analysis,
and nothing may be read from it into a real report, the Fight Camp, a
comparison or any statistic. It exists so a visitor can see what a report
looks like before creating an account, and the page that renders it says, at
the top and beside the numbers, that it is a sample with made-up data.

It shows only the measurements a real report publishes (core/metric_catalog.py)
under the same names, definitions and experimental labels, so the sample does
not promise anything the product does not do: no strike counts and no score,
which a real report shows only when the footage supports them.
"""
from __future__ import annotations

from core.metric_catalog import CATALOG

SAMPLE_NOTICE = ("Sample report with made-up data. The fighters and every number below are invented to "
                 "show the layout of a WarriorIQ report. They are not from a real fight.")

SAMPLE_REPORT = {
    "sport": "Kickboxing",
    "ruleset": "K-1 rules",
    "rounds": 3,
    "round_minutes": 2,
    "fighters": {"A": "Sample Fighter A", "B": "Sample Fighter B"},
    "focus": "A",
    # How well the two people were followed - a real report always shows this
    # before any measurement.
    "tracking": {"seen_A": 0.86, "seen_B": 0.81, "told_apart": 0.93},
    "metrics": {
        "A": {"pose_coverage": 0.86, "guard_index": 0.58, "balance_index": 0.74,
              "ring_center_control": 0.55, "footwork_body_lengths_per_second": 0.42, "pressure_index": 0.18},
        "B": {"pose_coverage": 0.81, "guard_index": 0.66, "balance_index": 0.69,
              "ring_center_control": 0.45, "footwork_body_lengths_per_second": 0.37, "pressure_index": -0.18},
    },
    "rounds_detail": [
        {"round": 1, "note": "Both fighters seen for most of the round; A held the centre more often."},
        {"round": 2, "note": "A's hands dropped after exchanges; B gave ground along the ropes."},
        {"round": 3, "note": "Tracking dipped for B near the corner camera, so this round carries less weight."},
    ],
    "priorities": [
        ("Bring the hands back after you throw",
         "Guard dropped most often in the two seconds after an exchange in round 2."),
        ("Keep your stance when you step in",
         "Balance was lowest on forward steps."),
    ],
    "plan": [
        ("Week 1", "Shadow boxing: 3 x 2 min, hands back to the chin after every combination."),
        ("Week 2", "Pad rounds: partner taps the open side when the guard drops."),
        ("Week 3", "Footwork ladder: step in and out without the rear heel lifting."),
        ("Week 4", "Sparring: film it and upload it again to compare against this report."),
    ],
}


def sample_rows() -> list[dict]:
    """One row per published measurement, with both fighters' made-up values."""
    rows = []
    for metric in CATALOG:
        a = SAMPLE_REPORT["metrics"]["A"].get(metric.key)
        b = SAMPLE_REPORT["metrics"]["B"].get(metric.key)
        if a is None or b is None:
            continue
        rows.append({"metric": metric, "A": _shown(metric, a), "B": _shown(metric, b)})
    return rows


def _shown(metric, value: float) -> str:
    if metric.key == "pressure_index":
        return f"{round((value + 1) * 50)} / 100"
    if metric.unit == "percent":
        return f"{value * 100:.0f}%"
    return f"{value:.2f}"
