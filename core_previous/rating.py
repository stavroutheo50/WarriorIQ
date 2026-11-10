"""A rating that only moves on evidence the scorecard already trusts.

Reads what core/ has already published for a fight and turns it into one
number per athlete over time. It consumes the report; it never recomputes or
second-guesses it, and it changes nothing about how a scorecard is produced.

**The gate is the whole design.** `integrity.automated_evidence_trusted` says
whether the action model's output has passed release validation. When it has
not - which is the current state of this project, and the reason punch counts
are withheld from every report - a strike-derived rating would be a confident
number resting on labels the product itself refuses to show. So the session is
still recorded, because the history is worth keeping, and the rating does not
move. A session that cannot be rated says so instead of being averaged in.

Movement is treated differently and deliberately: pose and tracking numbers do
not depend on action labels, which is exactly why /compare already shows them
while withholding the strike table. They can move a rating when the strike
numbers cannot.

Elo needs two opponents. There is only one athlete here, so the comparison is
against the athlete's own current rating: a session that beats what the rating
predicts raises it, one that falls short lowers it, and the size of the move
falls as the rating earns its confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Where an unrated athlete starts, and what a session is measured against.
BASELINE_RATING = 1500.0

# Sessions needed before the number is presented as a rating rather than a
# guess. Five is chosen because a single fight's coverage varies enough on this
# project's own footage - 28% between re-encodes of the same bout - that four
# would still be mostly noise.
PROVISIONAL_SESSIONS = 5

# How far one session can move the rating. Larger while provisional so a new
# athlete converges instead of crawling, then settled.
K_PROVISIONAL = 48.0
K_SETTLED = 24.0

# Elo's scale: a 400-point gap means the stronger side is expected to win ~10
# times out of 11.
RATING_SCALE = 400.0


@dataclass(frozen=True)
class SessionSignals:
    """What one analysed fight says about one athlete.

    Every field is optional because every one of them can be absent from a
    real report - an untrusted action model, a fight the tracker lost, a
    metric the sport does not produce. Absent is not zero.
    """

    job_id: str = ""
    fighter: str = "A"
    accuracy: float | None = None
    attempts: int = 0
    landed: int = 0
    output_per_round: float | None = None
    defensive_lapses: int | None = None
    pose_coverage: float | None = None
    evidence_trusted: bool = False
    movement_available: bool = False
    used: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ratable(self) -> bool:
        """Whether anything here may move a rating."""
        return bool(self.used)


def extract_session(report: dict, fighter: str = "A", job_id: str = "") -> SessionSignals:
    """Read one athlete's session out of a finished report.

    Purely a reader. Nothing here writes to the report or recomputes anything
    the scorecard decided.
    """
    integrity = (report or {}).get("integrity") or {}
    trusted = bool(integrity.get("automated_evidence_trusted"))
    metrics = ((report or {}).get("metrics") or {}).get(fighter) or {}
    attacks = metrics.get("attacks") or {}

    attempts = int(attacks.get("attempts") or 0)
    landed = int(attacks.get("landed") or 0)
    accuracy = attacks.get("accuracy")
    accuracy = float(accuracy) if isinstance(accuracy, (int, float)) else None

    coverage = metrics.get("pose_coverage")
    coverage = float(coverage) if isinstance(coverage, (int, float)) else None

    rounds = len((report or {}).get("rounds") or []) or 1
    output = (attempts / rounds) if attempts else None

    lapses = metrics.get("vulnerability_techniques")
    if isinstance(lapses, dict):
        lapses = int(sum(int(v or 0) for v in lapses.values()))
    elif isinstance(lapses, (int, float)):
        lapses = int(lapses)
    else:
        lapses = None

    # Which signals are allowed to move the rating. Strike-derived numbers
    # only once the scorecard's own gate says they may be believed; movement
    # numbers regardless, because they do not rest on action labels.
    used: list[str] = []
    movement = bool(metrics.get("advanced_metrics_available"))
    if movement and coverage is not None:
        used.append("movement")
    if trusted and accuracy is not None and attempts > 0:
        used.append("accuracy")
        if output is not None:
            used.append("output")
        if lapses is not None:
            used.append("defence")

    return SessionSignals(
        job_id=job_id, fighter=fighter, accuracy=accuracy, attempts=attempts,
        landed=landed, output_per_round=output, defensive_lapses=lapses,
        pose_coverage=coverage, evidence_trusted=trusted,
        movement_available=movement, used=tuple(used),
    )


def performance_score(signals: SessionSignals) -> float | None:
    """One 0..1 number for how the session went, or None if unratable.

    An average of whatever the gate allowed, so a session rated on movement
    alone is not silently compared against one rated on movement and strikes
    as though they measured the same thing. What each was rated on is recorded
    alongside it.
    """
    if not signals.ratable:
        return None
    parts: list[float] = []
    if "movement" in signals.used and signals.pose_coverage is not None:
        # Coverage is how much of the fight the athlete was actually followed
        # for. It is a floor on how much the rest can be believed.
        parts.append(max(0.0, min(1.0, signals.pose_coverage)))
    if "accuracy" in signals.used and signals.accuracy is not None:
        parts.append(max(0.0, min(1.0, signals.accuracy)))
    if "output" in signals.used and signals.output_per_round:
        # Twenty attempts a round is a busy round in this project's footage;
        # beyond that the extra volume says little about quality.
        parts.append(max(0.0, min(1.0, signals.output_per_round / 20.0)))
    if "defence" in signals.used and signals.defensive_lapses is not None:
        # Fewer lapses is better, so this one is inverted. Ten lapses is the
        # point past which the difference stops being informative.
        parts.append(max(0.0, 1.0 - min(1.0, signals.defensive_lapses / 10.0)))
    if not parts:
        return None
    return sum(parts) / len(parts)


def expected_score(rating: float, baseline: float = BASELINE_RATING) -> float:
    """What a rating of this size is expected to score against the baseline."""
    return 1.0 / (1.0 + 10.0 ** ((baseline - rating) / RATING_SCALE))


def next_rating(rating: float, sessions: int, score: float) -> tuple[float, float]:
    """(new rating, delta) after one session scoring `score` in 0..1."""
    k = K_PROVISIONAL if sessions < PROVISIONAL_SESSIONS else K_SETTLED
    delta = k * (score - expected_score(rating))
    return rating + delta, delta


def is_provisional(sessions: int) -> bool:
    """Whether the number should be shown as a rating or as a placeholder."""
    return sessions < PROVISIONAL_SESSIONS


def describe(rating: float, sessions: int) -> str:
    """What to put on screen.

    A number carries an authority it has not earned after one fight, so until
    there are enough sessions it is not shown as one.
    """
    if sessions <= 0:
        return "Unrated"
    if is_provisional(sessions):
        return "Provisional (%d of %d sessions)" % (sessions, PROVISIONAL_SESSIONS)
    return str(int(round(rating)))
