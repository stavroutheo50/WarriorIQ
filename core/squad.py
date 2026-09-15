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
from datetime import datetime
from pathlib import Path

from core.config import RULESET_LABELS

# Below this the fight was not watched well enough for its numbers to belong in
# a comparison. Showing it anyway would let a badly tracked fight look like a
# fighter having a bad week.
_MIN_COVERAGE = 0.85
# A change smaller than this is noise between two fights, not a trend.
_MEANINGFUL_CHANGE = 0.05


def _short_date(created_at: str | None) -> str:
    """"14 Sep" from a stored ISO timestamp, or "" if there is not one.

    Stored values carry a timezone; a malformed or missing one is not worth an
    exception on a page listing somebody's whole squad.
    """
    if not created_at:
        return ""
    try:
        return datetime.fromisoformat(str(created_at)).strftime("%-d %b")
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(str(created_at)).strftime("%d %b").lstrip("0")
        except (TypeError, ValueError):
            return str(created_at)[:10]


def fight_label(sport_label: str | None, ruleset: str | None, created_at: str | None) -> str:
    """What to call a fight in front of a person.

    The squad table printed the uploaded file's name - IMG_4554.mov, 2.mp4, a
    camera's hash - while /pricing carries the check-marked promise "No video
    filename shown". The filename stays in the database, where reprocessing
    needs it; it stops being what a coach reads.
    """
    parts = [
        (sport_label or "").strip(),
        RULESET_LABELS.get(ruleset or "", (ruleset or "").replace("_", " ").title()).strip(),
        _short_date(created_at),
    ]
    return " · ".join(part for part in parts if part) or "Fight analysis"


def fight_choice_label(ruleset: str | None, created_at: str | None,
                       fight_type: str | None) -> str:
    """What to call a fight in a dropdown, built from what differs between them.

    The /compare selects listed "Fight analysis · 2026-09-02" six times and
    "· 2026-09-01" three times, which is not a choice but nine identical rows.
    Ruleset, time of day and fight type are all stored on the row already and
    are exactly what tells one from another.
    """
    stamp = ""
    if created_at:
        try:
            moment = datetime.fromisoformat(str(created_at))
            stamp = f"{_short_date(created_at)}, {moment:%H:%M}"
        except (TypeError, ValueError):
            stamp = str(created_at)[:10]
    parts = [
        RULESET_LABELS.get(ruleset or "", (ruleset or "").replace("_", " ").title()).strip(),
        stamp,
        (fight_type or "").replace("_", " ").strip(),
    ]
    return " · ".join(part for part in parts if part) or "Fight analysis"


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
        "fighter_id": fight.get("fighter_id"),
        "fighter_name": fight.get("fighter_name"),
        "sport": scorecard.get("sport") or "unknown",
        "sport_label": scorecard.get("sport_label") or scorecard.get("sport") or "—",
        # Kept for anything that needs the source file; not for display.
        "name": fight.get("original_name"),
        "created_at": fight.get("created_at"),
        "ruleset": fight.get("ruleset"),
        "ruleset_label": RULESET_LABELS.get(
            fight.get("ruleset") or "",
            str(fight.get("ruleset") or "").replace("_", " ").title()),
        "label": fight_label(
            scorecard.get("sport_label") or scorecard.get("sport"),
            fight.get("ruleset"), fight.get("created_at")),
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
        trend_fighter = usable[0].get("fighter_id")
        comparable = [
            row for row in usable
            if row.get("sport") == trend_sport
            and trend_fighter is not None
            and row.get("fighter_id") == trend_fighter
        ]
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
        "trend_fighter": (usable[0].get("fighter_name") if usable else None),
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
        # Same fighter, same sport. The roster settles the first: a workspace
        # can hold a squad, and comparing two different people and calling the
        # difference progress is not a fact about either of them. Where the
        # fight predates the roster and has no fighter, nothing is compared
        # rather than falling back to a guess.
        if current is None or not row["usable"]:
            continue
        if current.get("fighter_id") is None or row.get("fighter_id") != current.get("fighter_id"):
            continue
        if row.get("sport") != current.get("sport"):
            continue
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


# The five identity-safe pose dimensions, in the order core/coaching.py
# declares them. Higher is better for all five: build_pose_coaching ranks them
# with reverse=True and calls the top one the fighter's strength, so a
# comparison treating any of them as lower-is-better would contradict the
# coaching shown on the same fight.
_MOVEMENT_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("pressure_index", "Walking them down"),
    ("ring_center_control", "Holding the middle"),
    ("footwork_body_lengths_per_second", "Moving your feet"),
    ("guard_index", "Guard"),
    ("balance_index", "Balance"),
)


def _display_value(key: str, value: float | None) -> tuple[str, float | None]:
    """Format one pose metric the way the report and coaching already do.

    The same three conventions as core/coaching.py's _phrase: pressure mapped
    onto 0-100 where 50 is neither forward nor back, footwork in body lengths a
    second, everything else a percentage. A second scale for the same number
    would make the comparison page disagree with the report it compares.
    """
    if value is None:
        return "—", None
    if key == "pressure_index":
        shown = (value + 1) / 2 * 100
        return f"{shown:.0f}", shown
    if key == "footwork_body_lengths_per_second":
        return f"{value:.1f}", value
    return f"{value * 100:.0f}%", value * 100


def compare_movement(reports: list) -> dict:
    """Fight 1 against fight 2 on the movement numbers, for /compare.

    That page withholds the strike-by-strike half until strike counting is
    release-validated, told the reader movement was "compared below", and then
    rendered nothing below. These are the numbers that survive the gate: they
    come from pose and tracking alone, so none of them depends on the action
    labels being trustworthy.

    Deliberately not phrased as progress. The two fights are whichever two the
    reader picked, in whatever order, so "improved" would be a claim about
    chronology this page has no basis for. It reports which fight leads each
    number and by how much, and leaves the reading to the coach.
    """
    if len(reports) != 2 or not all(reports):
        return {"available": False}

    sides = []
    for report in reports:
        video = report.get("video") or {}
        focus = video.get("focus_fighter") or video.get("analysis_target") or "A"
        if focus not in {"A", "B"}:
            focus = "A"
        tracking = report.get("tracking") or {}
        coverage = min(
            float(tracking.get("fighter_A_coverage", 0.0) or 0.0),
            float(tracking.get("fighter_B_coverage", 0.0) or 0.0),
        )
        sides.append({
            "report": report, "focus": focus,
            "coverage": round(coverage, 3), "usable": coverage >= _MIN_COVERAGE,
        })

    rows = []
    for key, label in _MOVEMENT_DIMENSIONS:
        first = _metric(sides[0]["report"], sides[0]["focus"], key)
        second = _metric(sides[1]["report"], sides[1]["focus"], key)
        if first is None and second is None:
            continue
        first_text, first_shown = _display_value(key, first)
        second_text, second_shown = _display_value(key, second)
        delta_text, leader = "—", "level"
        if first_shown is not None and second_shown is not None:
            gap = second_shown - first_shown
            # The same noise floor the trend uses, read on the displayed scale
            # so a 5% band means 5 points of what the reader can actually see.
            floor = _MEANINGFUL_CHANGE * (1.0 if key == "footwork_body_lengths_per_second" else 100.0)
            if abs(gap) < floor:
                delta_text, leader = "level", "level"
            else:
                leader = "b" if gap > 0 else "a"
                sign = "+" if gap > 0 else "−"
                digits = 1 if key == "footwork_body_lengths_per_second" else 0
                delta_text = f"{sign}{abs(gap):.{digits}f}"
        rows.append({
            "key": key, "label": label,
            "a": first_text, "b": second_text,
            "delta": delta_text, "leader": leader,
        })

    if not rows:
        return {"available": False}
    return {
        "available": True,
        "rows": rows,
        # Surfaced rather than used to suppress: a fight the tracker could not
        # follow is still worth showing beside one it could, so long as the
        # reader is told which is which.
        "coverage": [sides[0]["coverage"], sides[1]["coverage"]],
        "untrusted": [not sides[0]["usable"], not sides[1]["usable"]],
    }
