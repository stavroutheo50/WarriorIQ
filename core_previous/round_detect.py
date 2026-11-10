"""Work out a fight's round structure from the fight itself.

Rounds and round length were typed in by hand and defaulted to three twos. That
default is wrong for most footage people actually upload - a 94-second clip was
being cut into three two-minute rounds - and asking is the wrong shape of
question anyway: the video already contains the answer.

A round is when the two fighters are engaging. A break is when they are not,
and a break does not look like a lull in the action. During a break the pair
separate to their corners and stay apart, for a long time, and neither closes
the distance. Both halves matter: fighters back off constantly inside a round,
but only for a second or two, and they come back.

So a break is a sustained stretch where the fighters are far apart AND staying
that way. Everything between breaks is a round.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.config import SETTINGS

# A gap must last this long to be a break rather than a reset. Amateur rounds
# are separated by a minute; even a rushed corner change takes far longer than
# the two-second breathers that happen constantly inside a round.
_MIN_BREAK_SECONDS = 12.0
# A round shorter than this is a fragment of one, not a round of its own.
_MIN_ROUND_SECONDS = 25.0


@dataclass(frozen=True)
class DetectedRound:
    number: int
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


class RoundDetector:
    """Records how far apart the fighters were, second by second."""

    def __init__(self) -> None:
        self._samples: list[tuple[float, float | None]] = []

    def observe(self, seconds: float, separation_body_lengths: float | None) -> None:
        self._samples.append((float(seconds), separation_body_lengths))

    def _apart_seconds(self) -> list[tuple[float, bool]]:
        """One reading a second: were the fighters out of engagement range?

        Sampled per second rather than per frame so a single missed detection
        cannot punch a hole in a round, and so the thresholds below mean the
        same whatever frame rate the analysis ran at.
        """
        buckets: dict[int, list[float | None]] = {}
        for seconds, separation in self._samples:
            buckets.setdefault(int(seconds), []).append(separation)
        readings = []
        for second in sorted(buckets):
            values = [v for v in buckets[second] if v is not None]
            if not values:
                # Nobody located. Not evidence of a break: fighters are lost to
                # occlusion mid-exchange all the time.
                readings.append((float(second), False))
                continue
            median = sorted(values)[len(values) // 2]
            readings.append((float(second), median > SETTINGS.max_engagement_body_lengths))
        return readings

    def rounds(self) -> list[DetectedRound]:
        """The fighting segments, or an empty list when nothing is clear.

        Returning nothing is a real answer. A single continuous round and a
        video the detector could not read look identical from here, and both
        are served correctly by analysing the whole thing as one round.
        """
        readings = self._apart_seconds()
        if len(readings) < 60:
            return []

        breaks: list[tuple[float, float]] = []
        run_start: float | None = None
        for second, apart in readings:
            if apart and run_start is None:
                run_start = second
            elif not apart and run_start is not None:
                if second - run_start >= _MIN_BREAK_SECONDS:
                    breaks.append((run_start, second))
                run_start = None
        if run_start is not None and readings[-1][0] - run_start >= _MIN_BREAK_SECONDS:
            breaks.append((run_start, readings[-1][0]))

        if not breaks:
            return []

        start = readings[0][0]
        end = readings[-1][0]
        segments: list[tuple[float, float]] = []
        cursor = start
        for gap_start, gap_end in breaks:
            if gap_start - cursor >= _MIN_ROUND_SECONDS:
                segments.append((cursor, gap_start))
            cursor = gap_end
        if end - cursor >= _MIN_ROUND_SECONDS:
            segments.append((cursor, end))

        # One segment means the breaks found were at the very edges - a walk-on
        # or a celebration - not a multi-round structure.
        if len(segments) < 2:
            return []
        return [
            DetectedRound(index, round(a, 2), round(b, 2))
            for index, (a, b) in enumerate(segments, start=1)
        ]

    def summary(self) -> dict:
        found = self.rounds()
        if not found:
            return {
                "rounds_detected": None,
                "reason": "No clear break between rounds; the fight was read as one continuous round.",
                "rounds": [],
            }
        return {
            "rounds_detected": len(found),
            "reason": None,
            "rounds": [
                {
                    "number": item.number,
                    "start_seconds": item.start_seconds,
                    "end_seconds": item.end_seconds,
                    "duration_seconds": round(item.duration, 1),
                }
                for item in found
            ],
        }
