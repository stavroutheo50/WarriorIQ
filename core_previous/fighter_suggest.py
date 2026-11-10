"""Work out which two people in the frame are actually fighting.

The motivating failure: a real job tracked two people standing at the edge of
the mat for a whole bout and published a fight feed for them. Nothing in the
report said the boxes were on the wrong people, because nothing knew.

Two measurements separate fighters from everyone else, and neither needs to
know who anyone is:

  * They move. Measured in body lengths travelled per minute, so the number
    means the same at any camera distance, real fighters came in at 24-65
    across every fight measured. The man standing at ringside came in at 9.6.
  * They move at each other. A referee crosses the mat and coaches wander, but
    only the two fighters spend the bout within striking range of one specific
    person.

Scored that way on three tournament fights where the right answer was known by
eye, the correct pair ranked first on all three.

This does not pick the fighters on its own. It runs alongside the analysis and
says when the people being analysed are not the people who fought.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from core.config import SETTINGS

# BoT-SORT issues a new id whenever someone is briefly occluded, so one fighter
# arrives as several tracks. Rejoin them when one ends where the next begins.
_MERGE_GAP_SAMPLES = 25
_MERGE_DISTANCE_BODY_LENGTHS = 1.2


@dataclass
class _Track:
    samples: list[tuple[int, float, float, float]] = field(default_factory=list)

    def add(self, index: int, box: list[float]) -> None:
        self.samples.append(
            (index, (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, max(1.0, box[3] - box[1]))
        )

    @property
    def height(self) -> float:
        return sorted(item[3] for item in self.samples)[len(self.samples) // 2]

    def travel_per_minute(self, samples_per_second: float) -> float:
        if len(self.samples) < 6 or samples_per_second <= 0:
            return 0.0
        distance = sum(
            math.hypot(b[1] - a[1], b[2] - a[2])
            for a, b in zip(self.samples, self.samples[1:])
        )
        span = (self.samples[-1][0] - self.samples[0][0]) / samples_per_second
        if span < 20.0:                      # too short a look to judge anyone by
            return 0.0
        return distance / self.height / (span / 60.0)


class FighterFinder:
    """Accumulates every tracked person, then names the pair that fought.

    Fed from the analysis loop, which already detects everyone in each frame,
    so this costs an append per person per frame and nothing else.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self._index = 0
        self._first_seconds: float | None = None
        self._last_seconds: float | None = None
        # Where the two people actually being analysed were, per sample, so the
        # comparison is trajectory against trajectory. Comparing an average
        # position to a first-frame box is meaningless: fighters drift to the
        # middle of the mat, so the two never line up even when correct.
        self._selected: dict[int, list[tuple[float, float, float]]] = {}

    def observe(self, seconds: float, people) -> None:
        if self._first_seconds is None:
            self._first_seconds = seconds
        self._last_seconds = seconds
        for person in people or ():
            track_id = getattr(person, "track_id", None)
            box = getattr(person, "box", None)
            if track_id is None or box is None:
                continue
            self._tracks.setdefault(int(track_id), _Track()).add(
                self._index, [float(value) for value in box]
            )
        self._index += 1

    def observe_selected(self, *observations) -> None:
        """Record where the analysed fighters were for this same sample."""
        here = []
        for observation in observations:
            box = getattr(observation, "box", None) if observation is not None else None
            if box is None:
                continue
            box = [float(value) for value in box]
            here.append((
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
                max(1.0, box[3] - box[1]),
            ))
        if here:
            self._selected[self._index] = here

    # -- internals ---------------------------------------------------------
    def _samples_per_second(self) -> float:
        if self._first_seconds is None or self._last_seconds is None:
            return 0.0
        span = self._last_seconds - self._first_seconds
        return (self._index / span) if span > 0 else 0.0

    def _merged(self) -> list[_Track]:
        ordered = sorted(
            (t for t in self._tracks.values() if len(t.samples) >= 4),
            key=lambda t: t.samples[0][0],
        )
        merged: list[_Track] = []
        for track in ordered:
            head = track.samples[0]
            for group in merged:
                tail = group.samples[-1]
                gap = head[0] - tail[0]
                if 0 <= gap <= _MERGE_GAP_SAMPLES:
                    reach = max(20.0, (head[3] + tail[3]) / 2.0) * _MERGE_DISTANCE_BODY_LENGTHS
                    if math.hypot(head[1] - tail[1], head[2] - tail[2]) < reach:
                        group.samples.extend(track.samples)
                        group.samples.sort()
                        break
            else:
                merged.append(_Track(list(track.samples)))
        return merged

    def best_pair(self) -> dict | None:
        """The two people who moved, and who moved at each other."""
        per_second = self._samples_per_second()
        if per_second <= 0 or self._index < 40:
            return None

        candidates = []
        for track in self._merged():
            if len(track.samples) < self._index * 0.10:
                continue
            travel = track.travel_per_minute(per_second)
            if travel < SETTINGS.min_candidate_travel_per_minute:
                continue                     # standing at the side, not fighting
            positions = {s: (x, y, h) for s, x, y, h in track.samples}
            xs = [x for _, x, _, _ in track.samples]
            ys = [y for _, _, y, _ in track.samples]
            candidates.append({
                "travel": travel,
                "positions": positions,
                "centre": (sum(xs) / len(xs), sum(ys) / len(ys)),
                "height": track.height,
            })

        best = None
        for a, b in itertools.combinations(candidates, 2):
            shared = sorted(set(a["positions"]) & set(b["positions"]))
            if len(shared) < self._index * 0.08:
                continue
            separations = []
            for sample in shared:
                ax, ay, ah = a["positions"][sample]
                bx, by, bh = b["positions"][sample]
                body = max(20.0, (ah + bh) / 2.0)
                separations.append(math.hypot(ax - bx, ay - by) / body)
            in_range = sum(1 for s in separations if s < SETTINGS.max_engagement_body_lengths)
            close = in_range / len(separations)
            score = min(a["travel"], b["travel"]) * close
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "close_share": round(close, 3),
                    "travel": (round(a["travel"], 1), round(b["travel"], 1)),
                    "centres": (
                        (round(a["centre"][0], 1), round(a["centre"][1], 1)),
                        (round(b["centre"][0], 1), round(b["centre"][1], 1)),
                    ),
                    "heights": (round(a["height"], 1), round(b["height"], 1)),
                    "_tracks": (a["positions"], b["positions"]),
                }
        if best is not None:
            best["followed_share"] = self._followed_share(best.pop("_tracks"))
        return best

    def _followed_share(self, observed) -> tuple[float, float] | None:
        """How often each person who fought had an analysed box on them.

        Scored per fighter and reported as a pair, because the failure that
        matters is asymmetric: seeding one real fighter and one coach still
        follows the fight half the time. Asking "was either box on this
        fighter?" scored that run 0.936 against 0.994 for a correct one, which
        separates nothing. Asking it of each fighter separately does: the
        fighter nobody was watching scores near zero.
        """
        if not self._selected:
            return None
        shares = []
        for positions in observed:
            hits = considered = 0
            for sample, picks in self._selected.items():
                spot = positions.get(sample)
                if spot is None:
                    continue
                considered += 1
                reach = max(20.0, spot[2]) * 1.2
                if any(math.hypot(p[0] - spot[0], p[1] - spot[1]) <= reach for p in picks):
                    hits += 1
            if considered < 40:
                return None
            shares.append(round(hits / considered, 3))
        return (shares[0], shares[1]) if len(shares) == 2 else None


def analysis_missed_the_fight(followed_share) -> bool:
    """True when one of the two people who fought was never being watched.

    Takes the worse of the two shares. A correct run still loses its fighters
    to occlusion and camera pans, so this is not a quality score - it only
    separates "followed both fighters" from "followed one of them and somebody
    else entirely".
    """
    if not followed_share:
        return False
    return min(followed_share) < SETTINGS.min_followed_share
