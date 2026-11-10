"""What the athletes someone follows have been doing, as feed items.

Only things that already happened and are already on the athlete's own page:
a fight they posted (its stats-only card), a level they reached, a run of
training weeks. Each is dated by the record behind it - the ledger entry that
crossed the level, the first counted session of the week that made the run -
so nothing is announced that the data does not show.

Pure functions over the ledger and session rows, so they are tested without a
database. Who may see whose items is decided by the caller (core/social.py).
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.camp import LEVEL_POINTS, RANKS

# Runs of training weeks worth telling followers about.
STREAK_MILESTONES = (2, 4, 8, 12, 26, 52)


def _rank(level: int) -> str:
    return [name for at, name in RANKS if level >= at][-1]


def _when(stamp) -> datetime:
    moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def level_ups(points: list[dict]) -> list[dict]:
    """One item per level reached, dated by the entry that reached it.

    Levels follow points earned (camp_standing), so spending never moves them.
    """
    earned = 0
    items = []
    ordered = sorted(points, key=lambda item: (str(item.get("created_at") or ""), int(item.get("id") or 0)))
    for entry in ordered:
        gained = int(entry["points"])
        if gained <= 0:
            continue
        before = 1 + earned // LEVEL_POINTS
        earned += gained
        after = 1 + earned // LEVEL_POINTS
        for level in range(before + 1, after + 1):
            items.append({"kind": "level", "level": level, "rank": _rank(level),
                          "new_rank": _rank(level) != _rank(level - 1), "at": entry["created_at"]})
    return items


def streak_milestones(sessions: list[dict]) -> list[dict]:
    """One item when a run of weeks with a counted session reaches a milestone."""
    first_in_week: dict[tuple[int, int], str] = {}
    for session in sessions:
        if session.get("verdict") != "counted":
            continue
        week = tuple(_when(session["created_at"]).isocalendar()[:2])
        if week not in first_in_week or str(session["created_at"]) < first_in_week[week]:
            first_in_week[week] = str(session["created_at"])
    items = []
    run = 0
    previous = None
    for week in sorted(first_in_week):
        monday = datetime.fromisocalendar(week[0], week[1], 1)
        run = run + 1 if previous is not None and (monday - previous).days == 7 else 1
        previous = monday
        if run in STREAK_MILESTONES:
            items.append({"kind": "streak", "weeks": run, "at": first_in_week[week]})
    return items


def newest_first(items: list[dict], limit: int) -> list[dict]:
    return sorted(items, key=lambda item: _when(item["at"]), reverse=True)[:limit]
