"""The free Ringside Scorer: a person scores a fight by tapping points.

Owner, 2026-11-02: something useful on WarriorIQ that works with no video and
no analysis computer. Owner, 2026-11-09: no technique buttons - just +1 and +2
for the blue corner and the red corner, and a point deduction under each.

Nothing here is detected or estimated. Every point is one the person tapped,
so the score is exactly as good as their taps - which the page says.

How a round is scored depends on the rules picked:

  points rules (Kick Light, Point Fighting, taekwondo...)
      each corner's points, less its deductions, added up over the rounds;
  ten-point must (boxing, K-1, Muay Thai, MMA...)
      the corner with more points takes the round 10-9, level is 10-10, and
      each deduction takes one point off that corner's score for the round.
"""

from __future__ import annotations

from core.scoring import RULESETS, normalize_ruleset

TAPS = (1, 2)
MAX_ROUNDS = 15
MAX_POINTS_PER_CORNER_ROUND = 400
MAX_DEDUCTIONS_PER_CORNER_ROUND = 10


def rules_summary(ruleset: str) -> str:
    """How this ruleset turns taps into a score, in one sentence."""
    if RULESETS[normalize_ruleset(ruleset)].ten_point_must:
        return ("Ten-point must: the corner with more points takes the round 10-9, a level round is 10-10, "
                "and each deduction takes a point off that corner's round.")
    return "Points: every point you tap counts, each deduction takes one off, and the totals decide the bout."


def _count(value, limit: int) -> int:
    try:
        return max(0, min(limit, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def _corner(taps: dict, fighter: str) -> tuple[int, int]:
    side = taps.get(fighter) if isinstance(taps, dict) else None
    side = side if isinstance(side, dict) else {}
    return (_count(side.get("points"), MAX_POINTS_PER_CORNER_ROUND),
            _count(side.get("deductions"), MAX_DEDUCTIONS_PER_CORNER_ROUND))


def score(ruleset: str, rounds: list[dict]) -> dict:
    """Score the taps. `rounds` is one dict per round:
    {"A": {"points": n, "deductions": n}, "B": {"points": n, "deductions": n}}
    with A the blue corner and B the red corner."""
    key = normalize_ruleset(ruleset)
    profile = RULESETS[key]
    rounds = list(rounds or [])[:MAX_ROUNDS] or [{}]
    rows = []
    totals = {"A": 0, "B": 0}
    for number, taps in enumerate(rounds, start=1):
        (points_a, minus_a), (points_b, minus_b) = _corner(taps, "A"), _corner(taps, "B")
        if profile.ten_point_must:
            a = 10 if points_a >= points_b else 9
            b = 10 if points_b >= points_a else 9
            a, b = a - minus_a, b - minus_b
        else:
            a, b = points_a - minus_a, points_b - minus_b
        totals["A"] += a
        totals["B"] += b
        rows.append({"round": number, "A": a, "B": b, "winner": "A" if a > b else "B" if b > a else "EVEN",
                     "points_A": points_a, "points_B": points_b,
                     "deductions_A": minus_a, "deductions_B": minus_b})
    winner = "A" if totals["A"] > totals["B"] else "B" if totals["B"] > totals["A"] else "EVEN"
    return {"ruleset": key, "ten_point_must": profile.ten_point_must, "rounds": rows,
            "totals": totals, "winner": winner}
