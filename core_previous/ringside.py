"""The free Ringside Scorer: a person taps each landed strike, WarriorIQ scores it.

Owner, 2026-11-02: something useful on WarriorIQ that works with no video and
no analysis computer, and brings coaches and fighters to the site. A coach at
ringside taps a corner's button each time that fighter lands; the score is
worked out by the same scorer the report's estimated scorecard uses
(core.scoring.score_fight), under the same federation rules.

Nothing here is detected or estimated. Every action is one the person
recorded, so the score is exactly as good as their taps - which the page says.

Which buttons a ruleset gets is not hand-written per sport. Every candidate
technique is built as a landed strike and kept only if the real scorer would
score it: that drops what a ruleset makes illegal (a kick in boxing, a head
punch in WT taekwondo, a knee in Low Kick) and what it values at nothing (a
leg in ITF taekwondo). So the buttons cannot offer a strike the score then
ignores.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.scoring import RULESETS, _table_points, is_estimated_scoring_event, normalize_ruleset, score_fight
from core.types import KnockdownEvent, StrikeEvent

# Elbows are a scoring strike under IFMA's full rules and the ABC Unified
# Rules, and illegal under no-elbows Muay Thai and everywhere else here. The
# analysis cannot see them, but a person at ringside can.
ELBOW_RULESETS = frozenset({"MUAY_THAI", "MMA"})

MAX_ROUNDS = 15
MAX_TAPS_PER_CORNER_ROUND = 400


@dataclass(frozen=True)
class _Candidate:
    key: str
    label: str
    family: str
    target: str
    technique: str
    jumping: bool = False
    spinning: bool = False


# Every technique the scorer distinguishes, in the order the buttons appear.
_CANDIDATES = (
    _Candidate("punch_head", "Punch to the head", "punch", "head", "cross"),
    _Candidate("punch_body", "Punch to the body", "punch", "body", "cross"),
    _Candidate("kick_body", "Body kick", "kick", "body", "round_kick"),
    _Candidate("kick_head", "Head kick", "kick", "head", "round_kick"),
    _Candidate("kick_leg", "Leg kick", "kick", "leg", "round_kick"),
    _Candidate("knee_body", "Knee", "knee", "body", "knee"),
    _Candidate("elbow", "Elbow", "elbow", "body", "elbow"),
    _Candidate("jump_kick_body", "Jumping kick to the body", "kick", "body", "round_kick", jumping=True),
    _Candidate("jump_kick_head", "Jumping kick to the head", "kick", "head", "round_kick", jumping=True),
    _Candidate("turn_kick_body", "Turning kick to the body", "kick", "body", "round_kick", spinning=True),
    _Candidate("turn_kick_head", "Turning kick to the head", "kick", "head", "round_kick", spinning=True),
)
_BY_KEY = {candidate.key: candidate for candidate in _CANDIDATES}


def _event(candidate: _Candidate, fighter: str, round_number: int, seconds: float) -> StrikeEvent:
    frame = int(round(seconds * 30))
    return StrikeEvent(
        fighter=fighter, opponent="B" if fighter == "A" else "A", round_number=round_number,
        start_frame=frame, peak_frame=frame, end_frame=frame,
        start_time=seconds, peak_time=seconds, end_time=seconds,
        technique=candidate.technique, family=candidate.family, limb=f"tap_{frame}",
        outcome="clean", landed=True, target=candidate.target,
        confidence=1.0, contact_confidence=1.0, model_source="ringside_scorer",
        evidence={"jumping": candidate.jumping, "spinning": candidate.spinning,
                  "foot_lift_torsos": 1.0},
    )


def _scores(candidate: _Candidate, ruleset: str) -> int | None:
    """Points the real scorer gives this technique, None where it weighs rounds
    rather than counting points, 0 where it does not score it at all."""
    profile = RULESETS[ruleset]
    event = _event(candidate, "A", 1, 0.0)
    if candidate.family == "elbow" and ruleset not in ELBOW_RULESETS:
        return 0
    if not is_estimated_scoring_event(event, ruleset):
        return 0
    points = _table_points(event, profile)
    if points is None:
        return None if not candidate.jumping and not candidate.spinning else 0
    return points


def buttons(ruleset: str) -> list[dict]:
    """The techniques a person can tap under this ruleset, with their value."""
    key = normalize_ruleset(ruleset)
    profile = RULESETS[key]
    out = []
    for candidate in _CANDIDATES:
        points = _scores(candidate, key)
        if points == 0:
            continue
        if points is not None and (candidate.jumping or candidate.spinning):
            # Only offered where it is worth more than the plain kick.
            plain = _BY_KEY[f"kick_{candidate.target}"]
            if points <= (_scores(plain, key) or 0):
                continue
        out.append({"key": candidate.key, "label": candidate.label, "points": points})
    if profile.ten_point_must:
        # Weighed round by round, so where the target makes no difference to
        # the scorer a single button per family is enough.
        families: dict[str, dict] = {}
        for button in out:
            family = _BY_KEY[button["key"]].family
            if profile.sport == "boxing":
                families[button["key"]] = button
            elif family not in families or _BY_KEY[button["key"]].target == "body":
                # The body variant, so a family is worth exactly its
                # ruleset weight with no head or leg bonus folded in.
                families[family] = {**button, "label": {
                    "punch": "Punch", "kick": "Kick", "knee": "Knee", "elbow": "Elbow"}[family]}
        out = list(families.values())
    return out


def rules_summary(ruleset: str) -> str:
    """How this ruleset turns taps into a score, in one or two sentences."""
    profile = RULESETS[normalize_ruleset(ruleset)]
    if profile.ten_point_must and profile.round_margins:
        bands = ", ".join(f"a lead of up to {limit} wins 10-{points}" for limit, points in profile.round_margins)
        return (f"Ten-point must, round by round. Every scoring strike counts the same; {bands}. "
                "A knockdown wins the round.")
    if profile.ten_point_must:
        return ("Ten-point must, round by round: the fighter who lands more wins the round 10-9, "
                "a dominant round is 10-8, and a knockdown wins the round.")
    return "Points: every landed technique scores its federation value, and the totals decide the bout."


def score(ruleset: str, rounds: list[dict]) -> dict:
    """Score the taps. `rounds` is one dict per round:
    {"A": [button keys], "B": [button keys], "knockdowns": {"A": n, "B": n}},
    where a knockdown counts for the fighter who scored it."""
    key = normalize_ruleset(ruleset)
    allowed = {button["key"] for button in buttons(key)}
    rounds = list(rounds or [])[:MAX_ROUNDS] or [{}]
    events: list[StrikeEvent] = []
    knockdowns: list[KnockdownEvent] = []
    seconds = 0.0
    counts = []
    for number, taps in enumerate(rounds, start=1):
        taps = taps if isinstance(taps, dict) else {}
        row = {}
        for fighter in ("A", "B"):
            keys = [k for k in (taps.get(fighter) or [])[:MAX_TAPS_PER_CORNER_ROUND] if k in allowed]
            row[fighter] = len(keys)
            for tap in keys:
                seconds += 1.0          # one second apart: the scorer never merges two taps
                events.append(_event(_BY_KEY[tap], fighter, number, seconds))
            if RULESETS[key].ten_point_must:
                scored = max(0, min(5, int((taps.get("knockdowns") or {}).get(fighter) or 0)))
                for _ in range(scored):
                    seconds += 1.0
                    # KnockdownEvent.fighter is the one who went down.
                    knockdowns.append(KnockdownEvent(fighter="B" if fighter == "A" else "A", caused_by=fighter,
                                                     round_number=number, source_frame=int(seconds * 30),
                                                     time_seconds=seconds, confidence=1.0))
        counts.append(row)
    card = score_fight(events, key, range(1, len(rounds) + 1), knockdowns, reliable=True, estimated=True)
    return {
        "ruleset": key,
        "ruleset_label": card["ruleset_label"],
        "sport_label": card["sport_label"],
        "ten_point_must": RULESETS[key].ten_point_must,
        "rounds": [{"round": item["round"], "A": item["fighter_A"], "B": item["fighter_B"],
                    "winner": item["winner"], "landed_A": counts[item["round"] - 1]["A"],
                    "landed_B": counts[item["round"] - 1]["B"]} for item in card["rounds"]],
        "totals": card["totals"],
        "winner": card.get("winner_estimate"),
    }
