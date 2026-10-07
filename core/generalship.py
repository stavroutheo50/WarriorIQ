"""Score the parts of a round that judging actually asks about and tracking
can actually measure.

A ten-point-must round is judged on four things: clean effective striking,
effective aggression, ring generalship, and defence. WarriorIQ cannot see the
first one - the pose model returns a confident standing pose through a kick at
tournament resolution, so what the action stage reports as strikes is mostly
footwork. Every attempt to score a fight on strikes has therefore produced
nothing, or worse, a scorecard built from people walking.

The other three are movement, and movement is the one thing this system does
measure well: both fighters track above 93% coverage on real footage.

So this scores those three and says so. It is not a substitute for a judge and
it does not pretend the striking criterion was considered. What it gives an
athlete disputing a decision is better than an opinion: a timestamped,
reproducible measurement of who pressed, who held the middle, and who gave
ground, which anyone can check against the video.

  effective aggression  who moved at the other, and who moved away
  ring generalship      who held the middle of the action
  territory             who advanced and who was backed up

Everything here is derived from positions and directions only. No strike, no
contact, no target.

Only rulesets judged on a ten-point-must card (boxing, Muay Thai, MMA) are
judged on criteria like these at all. Taekwondo and every WAKO discipline are
decided by counting techniques, so for them this is a plain movement
comparison with neutral wording, and nothing here is ever a score unless a
ten-point-must ruleset asks for one while scoring is on. See judge_fight and
movement_comparison.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from core.metrics import ring_frame

# A round has to be clearly one fighter's before it is called. Below this the
# two were equal on what could be measured, and saying so is the honest answer
# rather than splitting hairs to produce a winner.
_DECISIVE_MARGIN = 0.06
# Nothing is judged from a handful of samples.
_MIN_SAMPLES_PER_FIGHTER = 40


@dataclass(frozen=True)
class RoundJudgement:
    number: int
    aggression: dict[str, float]
    generalship: dict[str, float]
    territory: dict[str, float]
    winner: str | None
    margin: float
    note: str

    def score(self, fighter: str) -> int:
        """Ten-point-must on the criteria that were measured."""
        if self.winner is None:
            return 10
        return 10 if fighter == self.winner else 9


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _share(a: float | None, b: float | None, low: float, high: float) -> tuple[float, float] | None:
    """Turn two comparable readings into shares of one round.

    Shares rather than raw values because the raw numbers carry the camera in
    them - a wide hall shot and a tight ring shot give different pixel
    distances for the same fight - while the split between two fighters seen
    by the same camera does not.

    Each reading is placed on its own known scale first. Normalising against
    the pair instead - subtracting whichever was lower - is what a first
    version did, and it forces the loser to exactly zero every time: two
    fighters who pressed 0.02 and -0.03 came out as a 100%/0% split, which
    reads as a wipeout and means almost nothing.
    """
    if a is None or b is None or high <= low:
        return None
    span = high - low
    a_scaled = min(1.0, max(0.0, (a - low) / span))
    b_scaled = min(1.0, max(0.0, (b - low) / span))
    total = a_scaled + b_scaled
    if total <= 1e-9:
        return 0.5, 0.5
    return a_scaled / total, b_scaled / total


def judge_round(
    number: int,
    pressure: dict[str, list[float]],
    centre: dict[str, list[float]],
    advance: dict[str, list[float]],
) -> RoundJudgement | None:
    """Judge one round from movement alone. None when there is too little to see."""
    for fighter in ("A", "B"):
        if len(pressure.get(fighter, [])) < _MIN_SAMPLES_PER_FIGHTER:
            return None

    # Each reading has its own natural range. Pressure is a cosine of movement
    # direction against the line to the opponent, so it runs -1 to 1. Centre
    # control is already a 0-1 closeness. Territory is ground taken as a share
    # of the round's own spread, and half a spread either way is a lot.
    parts: list[tuple[str, tuple[float, float]]] = []
    for name, source, low, high in (
        ("aggression", pressure, -1.0, 1.0),
        ("generalship", centre, 0.0, 1.0),
        ("territory", advance, -0.5, 0.5),
    ):
        split = _share(_mean(source.get("A", [])), _mean(source.get("B", [])), low, high)
        if split is not None:
            parts.append((name, split))
    if not parts:
        return None

    scored = {name: {"A": round(split[0], 3), "B": round(split[1], 3)} for name, split in parts}
    overall_a = statistics.fmean([split[0] for _, split in parts])
    margin = abs(overall_a - 0.5) * 2.0
    if margin < _DECISIVE_MARGIN:
        winner, note = None, "Too close to separate on movement alone."
    else:
        winner = "A" if overall_a > 0.5 else "B"
        leading = max(parts, key=lambda item: abs(item[1][0] - 0.5))[0]
        note = {
            "aggression": "Carried the round by pressing forward more of the time.",
            "generalship": "Carried the round by holding the middle of the action.",
            "territory": "Carried the round by advancing while the other gave ground.",
        }[leading]

    return RoundJudgement(
        number=number,
        aggression=scored.get("aggression", {}),
        generalship=scored.get("generalship", {}),
        territory=scored.get("territory", {}),
        winner=winner,
        margin=round(margin, 3),
        note=note,
    )


def _round_slices(samples, rounds):
    """Bucket timestamped samples into the rounds that were detected."""
    buckets: dict[int, dict[str, list]] = {}
    for spec in rounds:
        buckets[spec.number] = {"A": [], "B": []}
    for record in samples:
        seconds, fighter = record[0], record[1]
        for spec in rounds:
            if spec.start_seconds <= seconds < spec.end_seconds:
                buckets[spec.number][fighter].append(record)
                break
    return buckets


def judge_fight(metrics, rounds, coverage: dict[str, float], minimum_coverage: float,
                score_rounds: bool = False) -> dict:
    """A round-by-round movement comparison, and a movement score only on request.

    ``score_rounds`` adds ten-point-must round scores and totals. The analyzer
    asks for them only for a ruleset judged on a ten-point-must card while
    scoring is on (see core.scoring.RuleProfile.ten_point_must). Without it
    nothing here is a score: QA, 2026-10-07 found "Movement scorecard 10-10"
    beside "Score: Not scored" on WT taekwondo and point-fighting reports,
    sports that are decided by counting techniques and never scored 10-10.

    Withheld entirely when tracking was not good enough, for the same reason
    the striking scorecard is withheld: a comparison of a fight the system did
    not watch properly is worse than none.
    """
    worst = min(float(coverage.get("A", 0.0)), float(coverage.get("B", 0.0)))
    if worst < minimum_coverage:
        return {
            "available": False,
            "status": "insufficient_tracking",
            "reason": (
                f"Movement judging needs at least {minimum_coverage * 100:.0f}% tracking "
                f"coverage on both fighters. This analysis produced "
                f"A {coverage.get('A', 0) * 100:.0f}% and B {coverage.get('B', 0) * 100:.0f}%."
            ),
            "rounds": [],
        }

    pressure_by_round = _round_slices(metrics.timed_pressure, rounds)
    positions_by_round = _round_slices(metrics.timed_positions, rounds)

    judgements: list[RoundJudgement] = []
    for spec in rounds:
        if not spec.selected:
            continue
        pressure = {f: [record[2] for record in pressure_by_round[spec.number][f]] for f in ("A", "B")}

        # Ring generalship and territory both come from where the fighters were,
        # measured against the middle of this round's own action so the camera
        # cannot decide the answer.
        points = [record for side in ("A", "B") for record in positions_by_round[spec.number][side]]
        centre = {"A": [], "B": []}
        advance = {"A": [], "B": []}
        # The middle of the area this round used, not the average of the two
        # fighters' positions - that average is the point between them, which
        # scored every round's "held the middle" at about 50/50. See
        # core.metrics.ring_frame.
        frame = ring_frame([(record[2], record[3]) for record in points]) if len(points) >= 10 else None
        if frame is not None:
            (middle_x, middle_y), spread = (float(frame[0][0]), float(frame[0][1])), frame[1]
            for fighter in ("A", "B"):
                own = positions_by_round[spec.number][fighter]
                for record in own:
                    distance = ((record[2] - middle_x) ** 2 + (record[3] - middle_y) ** 2) ** 0.5
                    centre[fighter].append(max(0.0, 1.0 - distance / spread))
                # Territory: did they finish the round nearer the middle than
                # they started it? Ground taken, rather than ground held.
                if len(own) >= 20:
                    half = len(own) // 2
                    early = statistics.fmean([
                        (((r[2] - middle_x) ** 2 + (r[3] - middle_y) ** 2) ** 0.5) for r in own[:half]
                    ])
                    late = statistics.fmean([
                        (((r[2] - middle_x) ** 2 + (r[3] - middle_y) ** 2) ** 0.5) for r in own[half:]
                    ])
                    advance[fighter].append((early - late) / spread)

        judged = judge_round(spec.number, pressure, centre, advance)
        if judged is not None:
            judgements.append(judged)

    if not judgements:
        return {
            "available": False,
            "status": "insufficient_movement_samples",
            "reason": "Not enough tracked movement in any round to judge it.",
            "rounds": [],
        }

    led = {f: sum(1 for item in judgements if item.winner == f) for f in ("A", "B")}
    card = {
        "available": True,
        "status": "movement_comparison",
        # Rounds in which one fighter was clearly ahead on movement. A
        # comparison, not rounds won.
        "rounds_led": led,
        "rounds": [
            {
                "number": item.number,
                "leader": item.winner, "margin": item.margin,
                "aggression": item.aggression,
                "generalship": item.generalship,
                "territory": item.territory,
            }
            for item in judgements
        ],
    }
    if not score_rounds:
        return card
    totals = {f: sum(item.score(f) for item in judgements) for f in ("A", "B")}
    card.update({
        "status": "movement_criteria_only",
        "totals": totals,
        "rounds_won": led,
        "leader": None if totals["A"] == totals["B"] else ("A" if totals["A"] > totals["B"] else "B"),
        "criteria_scored": ["effective aggression", "ring generalship", "ground taken"],
        "criteria_excluded": ["clean effective striking"],
    })
    for row, item in zip(card["rounds"], judgements):
        row.update({"A": item.score("A"), "B": item.score("B"), "winner": item.winner, "note": item.note})
    return card


# What each comparison is called, and the judging criterion it speaks to where
# the ruleset really is judged round by round. Ground taken is not a judging
# criterion in any of them, so it carries none.
COMPARISONS = ("aggression", "generalship", "territory")
_NEUTRAL_LABELS = {"aggression": "Moved forward", "generalship": "In the middle", "territory": "Gained ground"}
_JUDGED_LABELS = {"aggression": "Pressed forward", "generalship": "Held the middle", "territory": "Took ground"}
_JUDGED_CRITERIA = {"aggression": "effective aggression", "generalship": "ring generalship", "territory": None}
_READS = {
    "aggression": "{name} moved forward more of the time.",
    "generalship": "{name} spent more of the round in the middle.",
    "territory": "{name} gained more ground.",
}


def _leading_comparison(row: dict) -> str | None:
    shares = [(key, row.get(key) or {}) for key in COMPARISONS]
    shares = [(key, share) for key, share in shares if "A" in share]
    if not shares:
        return None
    return max(shares, key=lambda item: abs(float(item[1]["A"]) - 0.5))[0]


def movement_comparison(card: dict | None, ruleset: str | None, scoring_available: bool,
                        names: dict | None = None) -> dict | None:
    """What the report shows of a stored movement card, for any report age.

    Reports saved before 2026-10-07 carry 10-10 round scores and "judging
    criteria" for every ruleset; this decides at render time what may be
    shown, so they are corrected without being re-run:

    * a score - round scores and totals - only for a ten-point-must ruleset
      while scoring is on, and only when the card holds one;
    * judging-criteria wording only for a ten-point-must ruleset; every other
      ruleset is decided by counting techniques, so it gets neutral wording.
    """
    if not card:
        return None
    from core.scoring import RULESETS, normalize_ruleset

    profile = RULESETS.get(normalize_ruleset(ruleset or ""))
    judged = bool(profile and profile.ten_point_must)
    names = names or {}
    labels = _JUDGED_LABELS if judged else _NEUTRAL_LABELS
    view = {
        "available": bool(card.get("available")),
        "reason": card.get("reason"),
        "title": "Movement comparison",
        "judged": judged,
        "columns": [{"key": key, "label": labels[key],
                     "criterion": _JUDGED_CRITERIA[key] if judged else None} for key in COMPARISONS],
    }
    sport = (profile.sport_label if profile and profile.sport != "kickboxing" else profile.label) if profile else None
    if judged:
        view["basis"] = (
            f"{sport} rounds are judged as a whole. This compares the part of that judging movement "
            "can show - who pressed forward and who held the middle - plus who gained ground. Clean "
            "striking is not included, because WarriorIQ cannot yet count strikes reliably, so this "
            "is not a score and does not replace the judges.")
    else:
        view["basis"] = (
            "Movement only: who moved forward, who spent more time in the middle of the area used, "
            "and who gained ground. "
            + (f"{sport} is decided by counting scoring techniques, so none of this is how the bout "
               "is judged, and it is not a score." if sport else "It is not a score."))
    show_scores = judged and bool(scoring_available) and isinstance(card.get("totals"), dict)
    view["scores"] = ({"totals": card["totals"], "leader": card.get("leader")} if show_scores else None)
    rounds = []
    for row in card.get("rounds") or []:
        leader = row.get("leader", row.get("winner"))
        key = _leading_comparison(row)
        if leader in ("A", "B") and key:
            read = _READS[key].format(name=names.get(leader) or f"Fighter {leader}")
        else:
            read = "Too close to separate on movement."
        shown = {"number": row.get("number"), "read": read,
                 "shares": {key: row.get(key) or None for key in COMPARISONS}}
        if show_scores:
            shown["score"] = {"A": row.get("A"), "B": row.get("B")}
        rounds.append(shown)
    view["rounds"] = rounds
    return view
