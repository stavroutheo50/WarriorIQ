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
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

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


def judge_fight(metrics, rounds, coverage: dict[str, float], minimum_coverage: float) -> dict:
    """A scorecard for the three criteria that movement can evidence.

    Withheld entirely when tracking was not good enough, for the same reason
    the striking scorecard is withheld: a score from a fight the system did not
    watch properly is worse than no score.
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
        if len(points) >= 10:
            middle_x = statistics.fmean([record[2] for record in points])
            middle_y = statistics.fmean([record[3] for record in points])
            spread = statistics.fmean([
                ((record[2] - middle_x) ** 2 + (record[3] - middle_y) ** 2) ** 0.5 for record in points
            ]) or 1.0
            for fighter in ("A", "B"):
                own = positions_by_round[spec.number][fighter]
                for record in own:
                    distance = ((record[2] - middle_x) ** 2 + (record[3] - middle_y) ** 2) ** 0.5
                    centre[fighter].append(max(0.0, 1.0 - distance / (spread * 2.0)))
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

    totals = {f: sum(item.score(f) for item in judgements) for f in ("A", "B")}
    won = {f: sum(1 for item in judgements if item.winner == f) for f in ("A", "B")}
    leader = None if totals["A"] == totals["B"] else ("A" if totals["A"] > totals["B"] else "B")
    return {
        "available": True,
        "status": "movement_criteria_only",
        "totals": totals,
        "rounds_won": won,
        "leader": leader,
        "criteria_scored": ["effective aggression", "ring generalship", "territory"],
        "criteria_excluded": ["clean effective striking"],
        "disclaimer": (
            "Scored on movement only: effective aggression, ring generalship and territory. "
            "Clean striking is not included, because WarriorIQ cannot yet detect strikes "
            "reliably enough to score them. This is not an official result and does not "
            "replace the judges - it is a reproducible record of who pressed, who held the "
            "middle, and who gave ground, which can be checked against the video."
        ),
        "rounds": [
            {
                "number": item.number,
                "A": item.score("A"), "B": item.score("B"),
                "winner": item.winner, "margin": item.margin, "note": item.note,
                "aggression": item.aggression,
                "generalship": item.generalship,
                "territory": item.territory,
            }
            for item in judgements
        ],
    }
