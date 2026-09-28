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
