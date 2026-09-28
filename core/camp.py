"""Fight Camp missions: what to train next, from the athlete's latest fight.

A mission is one of the report's own drills - an exercise chosen because a
measured number was behind - with the target the training plan set for it
and the moments in the fight where it showed. Nothing is invented here: a
fight whose report prescribed no drill gives no mission, and the page says
why instead of filling the space with generic advice.

Only a fight whose identity check held can give missions. Every drill is a
claim about one fighter, and on a fight where WarriorIQ could not tell the
two apart it may be the opponent's weakness.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

MAX_MISSIONS = 3


def _strip_fighter(name: str) -> str:
    """"Fighter A · Guard-return audit" -> "Guard-return audit"."""
    head, sep, tail = name.partition(" · ")
    return tail if sep and head.startswith("Fighter ") else name


def _strip_block(focus: str) -> str:
    """"Block 1: Fighter A · Guard-return audit" -> "Guard-return audit"."""
    return _strip_fighter(focus.split(": ", 1)[1] if focus.startswith("Block ") and ": " in focus else focus)


def missions_from_report(report: dict, fighter: str) -> dict:
    """{"missions": [...], "reason": str | None} for one fighter of one fight."""
    if not (report.get("integrity") or {}).get("identity_evidence_trusted", True):
        return {"missions": [], "reason": "identity"}
    coaching = (report.get("coaching") or {}).get(fighter) or {}
    drills = coaching.get("drills") or []
    if not drills:
        return {"missions": [], "reason": "no_drill"}
    plan = (report.get("training_plan") or {}).get(fighter) or []
    evidence = {item.get("detail"): item.get("evidence_times") or []
                for item in coaching.get("improvements") or []}
    missions = []
    for index, drill in enumerate(drills[:MAX_MISSIONS]):
        block = plan[index] if index < len(plan) else {}
        times = [float(t) for t in evidence.get(drill.get("why"), []) if t is not None]
        missions.append({
            "title": _strip_fighter(str(drill.get("name") or _strip_block(str(block.get("focus") or "")))),
            "exercise": drill.get("prescription"),
            "why": drill.get("why"),
            "target": block.get("goal"),
            "metric": drill.get("metric"),
            "label": drill.get("label"),
            "measured": drill.get("measured"),
            "evidence_seconds": times[0] if times else None,
        })
    return {"missions": missions, "reason": None}


def fight_camp_missions(records: list[dict], assignments: list[dict]) -> dict:
    """Missions from the newest fight that can give them.

    `records` are the athlete's fights newest first, each {"job_id",
    "report", "fighter", "created_at"}. The newest fight is used unless its
    identity check failed, in which case the one before it is - an athlete
    whose last video was filmed badly still has something to train.
    `assignments` marks the missions already taken on, by title.
    """
    taken = {str(item.get("title") or "").strip().lower(): item.get("status") for item in assignments}
    skipped_for_identity = None
    for record in records:
        found = missions_from_report(record["report"], record["fighter"])
        if found["reason"] == "identity":
            skipped_for_identity = skipped_for_identity or record
            continue
        for mission in found["missions"]:
            mission["status"] = taken.get(mission["title"].strip().lower())
        return {
            "job_id": record["job_id"], "fighter": record["fighter"], "created_at": record.get("created_at"),
            "missions": found["missions"], "reason": found["reason"],
            "skipped_newer_fight": skipped_for_identity is not None,
        }
    return {"job_id": None, "fighter": None, "created_at": None, "missions": [],
            "reason": "identity" if skipped_for_identity else "no_fight", "skipped_newer_fight": False}


# ---- Points --------------------------------------------------------------
#
# Sized to what each step proves. A counted session proves a real video with
# someone moving in it (core/training_check.py) - not who, not which drill -
# so it is worth little and capped per day. Finishing a mission after
# training for it is worth more. The big award is the only one the athlete
# cannot give themselves: their next fight showing the number moved.
SESSION_POINTS = 10
DAILY_COUNTED_SESSIONS = 2
MISSION_DONE_POINTS = 25
IMPROVED_POINTS = 100
LEVEL_POINTS = 100

# How far a number has to move to count as improved rather than noise, in
# the units it is stored in: a share (guard 0.10 -> 0.13 is 3 points), the
# pressure index (-1..1, so 0.06 is 3 on the 0-100 scale shown) and footwork
# in body lengths a second.
_IMPROVED_BY = {"pressure_index": 0.06, "footwork_body_lengths_per_second": 0.1}
_DEFAULT_IMPROVED_BY = 0.03


def improved_by(metric: str) -> float:
    return _IMPROVED_BY.get(metric, _DEFAULT_IMPROVED_BY)


def mission_result(mission: dict, later: list[dict]) -> dict | None:
    """The first later fight that measured the mission's number, and whether it moved.

    `later` are fights analysed after the mission was taken, oldest first,
    each {"job_id", "report", "fighter"}. Drills are only prescribed for a
    number that was behind, and for every such number higher is better.
    A fight whose identity check failed says nothing about this athlete.
    """
    for record in later:
        report = record["report"]
        if not (report.get("integrity") or {}).get("identity_evidence_trusted", True):
            continue
        value = ((report.get("metrics") or {}).get(record["fighter"]) or {}).get(mission["metric"])
        if value is None:
            continue
        value = float(value)
        return {"job_id": record["job_id"], "before": float(mission["measured"]), "after": value,
                "improved": value >= float(mission["measured"]) + improved_by(mission["metric"])}
    return None


def camp_standing(points: list[dict], sessions: list[dict], today: date) -> dict:
    """Points, level and weekly streak from the ledger and the sessions.

    The streak is the number of weeks in a row, up to this week or last,
    with at least one counted session: a week not over yet does not break it.
    """
    total = sum(int(item["points"]) for item in points)
    weeks = set()
    for session in sessions:
        if session.get("verdict") == "counted":
            year, week, _ = _day(session["created_at"]).isocalendar()
            weeks.add((year, week))
    streak = 0
    year, week, _ = today.isocalendar()
    cursor = today
    if (year, week) not in weeks:
        cursor = today - timedelta(days=7)
    while cursor.isocalendar()[:2] in weeks:
        streak += 1
        cursor = cursor - timedelta(days=7)
    return {"points": total, "level": 1 + total // LEVEL_POINTS,
            "into_level": total % LEVEL_POINTS, "level_size": LEVEL_POINTS, "streak_weeks": streak}


def _day(stamp: str) -> date:
    return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).date()
