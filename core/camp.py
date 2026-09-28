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

from core.coaching import POSE_DIMENSIONS, metric_goal

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
# Spending points: one extra analysis for REDEEM_COST, at most REDEEM_PER_MONTH
# a month. Each one is real processing time on the analysis machine, and the
# cap keeps the points worth earning rather than a second, free plan. Points
# can be spent but never bought or cashed out.
REDEEM_COST = 200
REDEEM_PER_MONTH = 2
# Counted sessions a week that make a good week. A target, not a cap: every
# session still counts for points up to the daily cap.
WEEKLY_TARGET = 3
# A name for each stretch of levels, reached by earning points. Cosmetic: it
# says how much training someone has put in, not how good they are.
RANKS = ((1, "Rookie"), (3, "Prospect"), (5, "Contender"), (8, "Challenger"), (12, "Champion"))
# What each kind of ledger entry is called on the page (core.db.award_points
# reasons, and redeem_points_for_analysis's spend).
_POINT_REASONS = {"session": "Training session", "mission_done": "Mission finished",
                  "improved": "Your fight showed the improvement", "redeem_analysis": "Extra analysis"}

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
    # The level follows what was earned, so spending points never takes a
    # level away; the balance is what is left to spend.
    total = sum(int(item["points"]) for item in points if int(item["points"]) > 0)
    balance = sum(int(item["points"]) for item in points)
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
    level = 1 + total // LEVEL_POINTS
    rank = [name for at, name in RANKS if level >= at][-1]
    next_rank = next(((at, name) for at, name in RANKS if at > level), None)
    this_week = sum(1 for session in sessions if session.get("verdict") == "counted"
                    and _day(session["created_at"]).isocalendar()[:2] == (year, week))
    return {"points": total, "balance": balance, "level": level,
            "into_level": total % LEVEL_POINTS, "level_size": LEVEL_POINTS, "streak_weeks": streak,
            "rank": rank, "next_rank": next_rank[1] if next_rank else None,
            "next_rank_level": next_rank[0] if next_rank else None,
            "week_sessions": this_week, "week_target": WEEKLY_TARGET}


def points_history(points: list[dict], limit: int = 5) -> list[dict]:
    """The newest ledger entries, named, so an athlete can see where points came from."""
    newest = sorted(points, key=lambda item: (str(item.get("created_at") or ""), int(item.get("id") or 0)),
                    reverse=True)
    return [{"points": int(item["points"]), "label": _POINT_REASONS.get(str(item.get("reason")), "Points"),
             "created_at": item.get("created_at")} for item in newest[:limit]]


def _day(stamp: str) -> date:
    return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).date()


def paid_sessions_on(sessions: list[dict], day: date) -> int:
    """Sessions that earned points on one day, which the daily cap counts."""
    stamp = day.isoformat()
    return sum(1 for session in sessions
               if int(session.get("points") or 0) > 0 and str(session.get("created_at") or "").startswith(stamp))


# ---- The board -------------------------------------------------------------
#
# One card per mission, whatever stage it has reached. Missions from the
# latest fight and the athlete's own list used to be two sections, and a
# mission taken on in the first reappeared in the second: three steps and two
# places for one thing.

_LABELS = {key: label for key, label, *_ in POSE_DIMENSIONS}
# Cards the athlete is working on first, finished ones last.
_GROUP = {"training": 0, "new": 1, "waiting": 2, "not_yet": 3, "improved": 3, "done": 3}


def mission_progress(metric: str | None, measured: float | None, result: dict | None = None) -> dict | None:
    """Where a mission's number was, where the plan aims it, and where the next fight put it.

    The goal is the training plan's own (core.coaching.metric_goal), so the bar
    and the plan's sentence name the same number. The tick marks where the
    improvement award starts (improved_by), which is the rule mission_result
    judges by. The bar only ever fills from a fight that measured the number:
    before one has, it is empty rather than guessed.
    """
    if not metric or measured is None:
        return None
    before = float(measured)
    goal, show = metric_goal(metric, before)
    span = goal - before
    if span <= 0:
        return None
    unlock = before + improved_by(metric)
    progress = {
        "label": _LABELS.get(metric, "This number"),
        "before": show(before), "goal": show(goal), "unlock": show(unlock),
        "tick": round(min(1.0, (unlock - before) / span) * 100),
        "after": None, "fill": 0, "improved": None,
    }
    if result is not None:
        after = float(result["after"])
        progress.update(after=show(after), improved=bool(result["improved"]),
                        fill=round(max(0.0, min(1.0, (after - before) / span)) * 100))
    return progress


def mission_board(camp: dict, assignments: list[dict], linked: dict, results: dict,
                  sessions_by_item: dict) -> list[dict]:
    """Every mission as one card with its stage.

    `camp` is fight_camp_missions' answer; `assignments` the athlete's list;
    `linked` the camp_missions rows by assignment id (which fight and number a
    taken mission came from); `results` mission_result by assignment id; and
    `sessions_by_item` counted sessions by assignment id.

    Stages: "new" (from the latest fight, not taken yet), "training" (on the
    list), "waiting" (finished training, no fight has measured it since),
    "improved" or "not_yet" (a later fight measured it), and "done" (the
    athlete's own item, ticked off - nothing measures those).
    """
    missions = camp.get("missions") or []
    latest = {str(m.get("title") or "").strip().lower(): m for m in missions}
    cards = []
    for item in assignments:
        mission = latest.get(str(item.get("title") or "").strip().lower(), {})
        link = linked.get(item["id"])
        result = results.get(item["id"])
        if result is not None:
            stage = "improved" if result["improved"] else "not_yet"
        elif item.get("status") == "active":
            stage = "training"
        elif link is not None:
            stage = "waiting"
        else:
            stage = "done"
        cards.append({
            "stage": stage, "assignment": item, "index": None,
            "title": item["title"], "detail": item.get("detail"), "why": mission.get("why"),
            "evidence_seconds": mission.get("evidence_seconds"),
            "sessions": int(sessions_by_item.get(item["id"], 0)),
            "progress": mission_progress(link["metric"], link["measured"], result) if link else None,
        })
    for index, mission in enumerate(missions):
        if mission.get("status") is not None:
            continue     # already on the list, as the card above
        cards.append({
            "stage": "new", "assignment": None, "index": index,
            "title": mission["title"], "detail": mission.get("exercise"), "why": mission.get("why"),
            "target": mission.get("target"), "evidence_seconds": mission.get("evidence_seconds"),
            "sessions": 0, "progress": mission_progress(mission.get("metric"), mission.get("measured")),
        })

    def group(card: dict) -> int:
        # Still on the list, whatever a fight has said since: keep it with the
        # work in hand rather than among the finished.
        if card["assignment"] is not None and card["assignment"].get("status") == "active":
            return 0
        return _GROUP[card["stage"]]

    for card in cards:
        card["finished"] = group(card) == _GROUP["done"]
    return sorted(cards, key=group)


def next_step(camp: dict, board: list[dict], paid_today: int, has_fight: bool) -> dict:
    """The one thing to do next, for the top of the page.

    In order: train for something on the list (or rest, once today's points
    are in), take on a mission, analyse the fight that will judge the
    training, and otherwise analyse a fight.
    """
    training = [card for card in board
                if card["assignment"] is not None and card["assignment"].get("status") == "active"]
    if training:
        if paid_today < DAILY_COUNTED_SESSIONS:
            # The item trained least so far: min() keeps the board's order on a tie.
            return {"kind": "upload", "card": min(training, key=lambda card: card["sessions"])}
        return {"kind": "rested", "card": None}
    new = [card for card in board if card["stage"] == "new"]
    if new:
        return {"kind": "take", "card": new[0]}
    waiting = [card for card in board if card["stage"] == "waiting"]
    if waiting:
        return {"kind": "next_fight", "card": waiting[0]}
    if not has_fight:
        return {"kind": "first_fight", "card": None}
    return {"kind": "analyse", "card": None, "reason": camp.get("reason")}
