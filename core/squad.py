"""Pull several fights into one view, for the person who has to triage a squad.

A coach and an athlete want opposite things from the same analysis. An athlete
wants one page answering "what do I fix this week": one priority, one drill,
one target, and whether they are better than last time. A coach wants breadth -
every fighter side by side, every fight in order, and which way each of them is
moving - because their job is deciding who needs attention, not fixing one
person.

One report served both and served neither, which is most of why it read as
pointless.

This reads the saved reports and answers the coach's question. It reads only
what the analysis can stand behind: tracking coverage, movement, pressure,
centre control and the movement scorecard. Nothing here depends on strike
detection, so nothing here is withheld for the reason the striking scorecard is.
"""
from __future__ import annotations

import json
from pathlib import Path

# Below this the fight was not watched well enough for its numbers to belong in
# a comparison. Showing it anyway would let a badly tracked fight look like a
# fighter having a bad week.
_MIN_COVERAGE = 0.85
# A change smaller than this is noise between two fights, not a trend.
_MEANINGFUL_CHANGE = 0.05


def _metric(report: dict, fighter: str, key: str) -> float | None:
    block = ((report.get("metrics") or {}).get(fighter) or {})
    value = block.get(key)
    if value is None:
        value = (block.get("baselines") or {}).get(key)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def summarize_fight(report: dict, fight: dict) -> dict | None:
    """One row: what this fight says about the fighter it was focused on."""
    video = report.get("video") or {}
    focus = video.get("focus_fighter") or video.get("analysis_target") or "A"
    if focus not in {"A", "B"}:
        focus = "A"
    tracking = report.get("tracking") or {}
    coverage = {
        side: float(tracking.get(f"fighter_{side}_coverage", 0.0) or 0.0)
        for side in ("A", "B")
    }
    movement = report.get("movement_scorecard") or {}
    verdict = None
    if movement.get("available"):
        totals = movement.get("totals") or {}
        other = "B" if focus == "A" else "A"
        if totals.get(focus) is not None and totals.get(other) is not None:
            if totals[focus] > totals[other]:
                verdict = "ahead on movement"
            elif totals[focus] < totals[other]:
                verdict = "behind on movement"
            else:
                verdict = "level on movement"

    coaching = ((report.get("coaching") or {}).get(focus) or {})
    improvements = coaching.get("improvements") or []
    scorecard = report.get("scorecard") or {}
    return {
        "job_id": fight.get("job_id"),
        "sport": scorecard.get("sport") or "unknown",
        "sport_label": scorecard.get("sport_label") or scorecard.get("sport") or "—",
        "name": fight.get("original_name"),
        "created_at": fight.get("created_at"),
        "ruleset": fight.get("ruleset"),
        "focus": focus,
        "coverage": round(min(coverage["A"], coverage["B"]), 3),
        "usable": min(coverage["A"], coverage["B"]) >= _MIN_COVERAGE,
        "movement_verdict": verdict,
        "pressure": _metric(report, focus, "pressure_index"),
        "centre": _metric(report, focus, "ring_center_control"),
        "footwork": _metric(report, focus, "footwork_body_lengths_per_second"),
        "priority": (improvements[0] or {}).get("title") if improvements else None,
    }


def _direction(newer: float | None, older: float | None) -> str:
    if newer is None or older is None:
        return "unknown"
    change = newer - older
    if abs(change) < _MEANINGFUL_CHANGE:
        return "steady"
    return "up" if change > 0 else "down"


def build_squad_view(fights: list[dict], limit: int = 25) -> dict:
    """Every fight in order, plus which way each fighter is moving.

    A fight the analysis could not watch properly is listed but marked, never
    silently folded into a trend: a tracking failure is not a fighter having a
    bad week, and a coach making selection decisions must be able to tell them
    apart.
    """
    rows: list[dict] = []
    for fight in fights[:limit]:
        path = Path(str(fight.get("report_path") or ""))
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        row = summarize_fight(report, fight)
        if row is not None:
            rows.append(row)

    usable = [row for row in rows if row["usable"]]
    trend = {}
    trend_sport = None
    # The trend follows the newest fight's sport, for the same reason the
    # athlete comparison does: these numbers are not comparable across sports.
    # Kept separate from `usable` so the counts above still describe every
    # fight in the list, not just the one sport being trended.
    comparable = usable
    if usable:
        trend_sport = usable[0].get("sport")
        comparable = [row for row in usable if row.get("sport") == trend_sport]
    if len(comparable) >= 2:
        # list_fights returns newest first.
        newest, previous = comparable[0], comparable[1]
        for key in ("pressure", "centre", "footwork"):
            trend[key] = {
                "direction": _direction(newest.get(key), previous.get(key)),
                "now": newest.get(key),
                "before": previous.get(key),
            }

    return {
        "fights": rows,
        "usable_count": len(usable),
        "unusable_count": len(rows) - len(usable),
        "trend": trend,
        "trend_available": bool(trend),
        "trend_sport": trend_sport,
        "note": (
            "Compared across the two most recent fights in the same sport that WarriorIQ "
            "tracked well enough. Movement only - striking is not included."
        ),
    }


def compare_with_previous(report: dict, fights: list[dict], job_id: str) -> dict:
    """The athlete's half: is this fight better than the last one?

    One comparison, against the most recent earlier fight WarriorIQ tracked
    well enough to compare against. A fight it could not watch properly is
    skipped rather than compared, because "you got worse" is a serious thing to
    tell somebody and a tracking failure is not evidence of it.
    """
    current = None
    previous = None
    for fight in fights:
        path = Path(str(fight.get("report_path") or ""))
        if not path.exists():
            continue
        try:
            other = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        row = summarize_fight(other, fight)
        if row is None:
            continue
        if fight.get("job_id") == job_id:
            current = summarize_fight(report, fight)
            continue
        # Same sport only. Pressure and centre control mean different things in
        # a taekwondo bout and a kickboxing bout, so comparing across them and
        # calling the difference progress is meaningless - and it is worse than
        # meaningless when it tells somebody they got worse.
        if current is not None and row["usable"] and row.get("sport") == current.get("sport"):
            previous = row
            break

    if current is None or previous is None or not current["usable"]:
        return {"available": False}

    changes = []
    for key, label, higher_is_better in (
        ("pressure", "Pressure", True),
        ("centre", "Holding the middle", True),
        ("footwork", "Footwork", True),
    ):
        direction = _direction(current.get(key), previous.get(key))
        if direction == "unknown":
            continue
        better = direction == "steady" or ((direction == "up") == higher_is_better)
        changes.append({
            "label": label, "direction": direction, "better": better,
            "now": current.get(key), "before": previous.get(key),
        })
    if not changes:
        return {"available": False}

    improved = sum(1 for item in changes if item["better"] and item["direction"] != "steady")
    worse = sum(1 for item in changes if not item["better"])
    if improved > worse:
        headline = "Better than your last fight."
    elif worse > improved:
        headline = "Down on your last fight."
    else:
        headline = "About the same as your last fight."
    return {
        "available": True,
        "headline": headline,
        "changes": changes,
        "previous_name": previous.get("name"),
        "previous_date": (previous.get("created_at") or "")[:10],
    }
