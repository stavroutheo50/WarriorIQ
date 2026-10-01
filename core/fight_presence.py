"""Is this stretch of video two people fighting?

QA, 2026-09: a news clip that was mostly interviews produced "150 strike
attempts", and a street scene - a man carrying pads through a crowd, with a
bystander boxed as Fighter B - produced a full report: 12 strikes, Pressure 43,
Centre 68%, Guard 6%. Nothing asked whether the footage showed a fight at all.
Every measurement assumed it did.

This module answers that question per window of footage, from what the
analysis already has for each analysed frame - the two selected people - and
nothing else, so it costs no inference:

  * ``both_present``  both selected people were found;
  * ``full_body``     both show their hips, which a head-and-shoulders
                      interview shot or a crowd cut-in does not;
  * ``in_range``      at some point in the window they were close enough to
                      exchange (centre to centre, in body heights);
  * ``moving``        at least one of them is actually moving - limbs or feet -
                      as fighters do and two people standing talking do not.

A window is left out only on positive evidence that it is not fight footage:
nobody in shot (graphics, title cards), the two
selected people far apart all window (a crowd, a street), or the people in
shot shown head-and-shoulders and barely moving (an interview). A window in
which tracking simply lost one of the fighters is kept - losing someone is a
tracking problem, already reported as coverage, and calling a real fight "not
a fight" would be a false claim. Short pauses between fight windows are kept.

Metrics and strike counts are then taken from the kept footage only, and the
report says how much of the video that was: "fight footage analysed: X of Y",
with what was left out and why. When too little is left to measure anything,
the report says so instead of showing numbers.

Thresholds were checked against real labelled clips (UCF101 Punch and
SumoWrestling against Haircut, WalkingWithDog, MilitaryParade,
BoxingPunchingBag, SalsaSpin, BandMarching) - see tests/test_fight_presence.py
for what each threshold has to keep and to reject.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WINDOW_SECONDS = 2.0
# Share of a window's analysed frames in which both people must be found.
MIN_BOTH_PRESENT = 0.5
# Share of those frames in which both show their hips.
MIN_FULL_BODY = 0.5
# Closest the two came during the window, centre to centre, in average body
# heights. Strikes need 2.5 (SETTINGS.max_engagement_body_lengths); a window
# of circling at range still belongs to the fight, hence the margin.
MAX_CLOSEST_SEPARATION = 3.0
# Median limb-and-body speed of the busier of the two, in body heights per
# second. Fighting windows on the labelled clips ran 0.49-2.25; a clinch can be
# slow, so only a pair that is also shown head-and-shoulders is held to the
# higher bar, and anyone below the lower one is standing still.
MIN_ACTIVITY = 0.25
STILL_ACTIVITY = 0.12
# Gaps of up to this many non-fight windows between fight windows are kept as
# part of the fight - a break in an exchange is not a cut to the crowd.
MAX_BRIDGED_WINDOWS = 1
# Below this much fight footage no measurement means anything, and the report
# says that instead of printing numbers.
MIN_FIGHT_SECONDS = 8.0

_HIPS = (11, 12)
_LIMBS = (9, 10, 15, 16)    # wrists and ankles
_KEYPOINT_CONF = 0.3

REASON_TEXT = {
    "nobody": "nobody was in shot (graphics, title cards, empty frames)",
    "not_full_body": "the people in shot were shown head-and-shoulders and barely moving, as in an interview",
    "apart": "the two selected people were never close enough to fight",
    "still": "neither selected person was moving",
}


def _hips_visible(obs) -> bool:
    conf = getattr(obs, "keypoint_conf", None)
    points = getattr(obs, "keypoints", None)
    if conf is None or points is None or len(conf) < 13:
        return False
    return any(float(conf[i]) >= _KEYPOINT_CONF and float(points[i][0]) > 0 and float(points[i][1]) > 0
               for i in _HIPS)


def _height(box) -> float:
    return max(20.0, float(box[3]) - float(box[1]))


def _separation(a, b) -> float:
    ca = np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0], dtype=np.float32)
    cb = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float32)
    return float(np.linalg.norm(ca - cb)) / ((_height(a) + _height(b)) / 2.0)


@dataclass
class _Window:
    frames: int = 0
    both: int = 0
    full_body: int = 0
    closest: float = float("inf")
    activity: list = field(default_factory=list)
    people: list = field(default_factory=list)
    full_people: list = field(default_factory=list)

    def verdict(self) -> tuple[bool, str | None]:
        """(kept, reason left out). Left out only on evidence, never on doubt."""
        if self.frames == 0:
            return True, None
        moving = float(np.median(self.activity)) if self.activity else None
        if self.both / self.frames >= MIN_BOTH_PRESENT:
            if self.closest > MAX_CLOSEST_SEPARATION:
                return False, "apart"
            full = self.full_body / max(1, self.both) >= MIN_FULL_BODY
            if not full and (moving is None or moving < MIN_ACTIVITY):
                return False, "not_full_body"
            if moving is not None and moving < STILL_ACTIVITY:
                return False, "still"
            return True, None
        # The selected pair was not followed through most of this window.
        # That alone says nothing about the footage; who else is in shot does.
        # Two fighters overlapping in a clinch are often found as one person,
        # so "only one person" is not evidence; nobody at all is.
        people = float(np.median(self.people)) if self.people else 0.0
        if people < 0.5:
            return False, "nobody"
        full_people = float(np.median(self.full_people)) if self.full_people else 0.0
        if full_people < 0.5 and (moving is None or moving < MIN_ACTIVITY):
            return False, "not_full_body"
        return True, None


class FightPresence:
    """Classify analysed frames into fight and non-fight windows."""

    def __init__(self, start_seconds: float, end_seconds: float):
        self.start = float(start_seconds)
        self.end = max(float(end_seconds), self.start)
        self._windows: dict[int, _Window] = {}
        self._last: dict[str, tuple[float, np.ndarray, np.ndarray | None]] = {}

    def _index(self, seconds: float) -> int:
        return max(0, int((float(seconds) - self.start) // WINDOW_SECONDS))

    def _activity(self, name: str, seconds: float, obs) -> float | None:
        """Body-heights per second moved by the box centre and the limbs."""
        box = np.asarray(obs.box, dtype=np.float32)
        center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)
        limbs = None
        points, conf = getattr(obs, "keypoints", None), getattr(obs, "keypoint_conf", None)
        if points is not None and conf is not None and len(conf) >= 17:
            limbs = np.array([points[i] if float(conf[i]) >= _KEYPOINT_CONF else (np.nan, np.nan)
                              for i in _LIMBS], dtype=np.float32)
        previous = self._last.get(name)
        self._last[name] = (float(seconds), center, limbs)
        if previous is None or seconds <= previous[0] or seconds - previous[0] > 0.6:
            return None
        dt = float(seconds) - previous[0]
        moved = float(np.linalg.norm(center - previous[1]))
        if limbs is not None and previous[2] is not None:
            deltas = np.linalg.norm(limbs - previous[2], axis=1)
            deltas = deltas[np.isfinite(deltas)]
            if deltas.size:
                moved = max(moved, float(np.mean(deltas)))
        return moved / _height(box) / dt

    def observe(self, seconds: float, fighter_a, fighter_b, people=None) -> None:
        window = self._windows.setdefault(self._index(seconds), _Window())
        window.frames += 1
        if people is not None:
            window.people.append(len(people))
            window.full_people.append(sum(1 for person in people if _hips_visible(person)))
        speeds = []
        for name, obs in (("A", fighter_a), ("B", fighter_b)):
            if obs is None:
                self._last.pop(name, None)
                continue
            speed = self._activity(name, seconds, obs)
            if speed is not None:
                speeds.append(speed)
        if fighter_a is None or fighter_b is None:
            return
        window.both += 1
        if _hips_visible(fighter_a) and _hips_visible(fighter_b):
            window.full_body += 1
        window.closest = min(window.closest, _separation(fighter_a.box, fighter_b.box))
        if speeds:
            window.activity.append(max(speeds))

    def _verdicts(self) -> list[tuple[int, bool, str | None]]:
        count = max(1, int(np.ceil((self.end - self.start) / WINDOW_SECONDS)))
        out = []
        for index in range(count):
            window = self._windows.get(index)
            # A window with no analysed frame at all (a gap the frame pass
            # stepped over) is kept: there is no evidence either way.
            fight, reason = window.verdict() if window is not None else (True, None)
            out.append((index, fight, reason))
        # Bridge short pauses between fight windows.
        fights = [fight for _, fight, _ in out]
        for index in range(len(fights)):
            if fights[index]:
                continue
            before = next((i for i in range(index - 1, -1, -1) if fights[i]), None)
            after = next((i for i in range(index + 1, len(fights)) if fights[i]), None)
            if before is not None and after is not None and after - before - 1 <= MAX_BRIDGED_WINDOWS:
                out[index] = (index, True, None)
        return out

    def segments(self) -> list[tuple[float, float]]:
        spans: list[list[float]] = []
        for index, fight, _ in self._verdicts():
            if not fight:
                continue
            start = self.start + index * WINDOW_SECONDS
            end = min(self.end, start + WINDOW_SECONDS)
            if spans and abs(spans[-1][1] - start) < 1e-6:
                spans[-1][1] = end
            else:
                spans.append([start, end])
        return [(round(a, 3), round(b, 3)) for a, b in spans]

    def is_fight(self, seconds: float, segments: list[tuple[float, float]] | None = None) -> bool:
        for start, end in (segments if segments is not None else self.segments()):
            if start <= seconds < end or (seconds == end == self.end):
                return True
        return False

    def decided_until(self, latest_seconds: float) -> float:
        """Footage before this time sits in windows that are complete."""
        return self.start + self._index(latest_seconds) * WINDOW_SECONDS

    def summary(self) -> dict:
        verdicts = self._verdicts()
        segments = self.segments()
        fight_seconds = round(sum(end - start for start, end in segments), 2)
        analysed = round(self.end - self.start, 2)
        reasons: dict[str, int] = {}
        for _, fight, reason in verdicts:
            if not fight and reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        main_reason = max(reasons, key=reasons.get) if reasons else None
        return {
            "fight_seconds": fight_seconds,
            "analysed_seconds": analysed,
            "share": round(fight_seconds / analysed, 3) if analysed > 0 else 0.0,
            "segments": segments,
            "excluded_seconds": round(max(0.0, analysed - fight_seconds), 2),
            "excluded_windows_by_reason": reasons,
            "main_exclusion_reason": main_reason,
            "main_exclusion_text": REASON_TEXT.get(main_reason) if main_reason else None,
            "sufficient": fight_seconds >= MIN_FIGHT_SECONDS,
            "window_seconds": WINDOW_SECONDS,
        }
