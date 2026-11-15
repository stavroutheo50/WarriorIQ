"""Progress over the last 30, 60 and 90 days (owner's brief, 2026-11-14).

Fight Camp already charts every measure across all of an athlete's fights
(core/progress_insights.py trend_cards). This narrows the same points to a
period, so "how have I changed this month" has an answer, and adds how much the
athlete trained in it.

Nothing new is measured or estimated. The points are the ones build_progress
already accepted - fights whose report could tell who was who - and a change
needs two fights with that measure inside the period, exactly as the all-time
cards do. A period with less says so instead of showing a number.

In app/ rather than core/ because it is presentation for one page, and every
core/*.py is part of the analysis-code fingerprint the worker must match.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.progress_insights import trend_cards

PERIODS = (30, 60, 90)


def _moment(value) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def period_summary(points: list[dict], sessions: list[dict], now: datetime | None = None) -> list[dict]:
    """One entry per period: fights measured, counted training sessions, and each measure's change."""
    now = now or datetime.now(timezone.utc)
    summary = []
    for days in PERIODS:
        start = now - timedelta(days=days)
        inside = [point for point in points if (moment := _moment(point.get("created_at"))) and moment >= start]
        trained = [item for item in sessions
                   if item.get("verdict") == "counted" and (moment := _moment(item.get("created_at")))
                   and moment >= start]
        active_weeks = {(_moment(item["created_at"]).isocalendar()[:2]) for item in trained}
        summary.append({
            "days": days,
            "fights": len(inside),
            "sessions": len(trained),
            # Weeks in the period with at least one counted session: a plain
            # measure of consistency, out of the period's length in weeks.
            "active_weeks": len(active_weeks),
            "weeks": round(days / 7),
            "cards": trend_cards(inside),
        })
    return summary
