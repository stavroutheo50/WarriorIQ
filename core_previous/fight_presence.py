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

import gzip
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

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

# ---- Is it a fight at all? -------------------------------------------------
# QA, 2026-10-07: two photo cut-outs slid left and right past each other and
# got "Strong observation evidence", guard-drop timestamps, strengths and a
# four-week plan. Every window above passed: both were found, full length,
# close, and "moving" - the box moved, so activity did. Nothing asked whether
# the bodies moved like bodies. These three do, over the whole analysed span,
# from keypoints already on every observation.
#
# Measured on the one real two-fighter track in the repository
# (dataset/regression/kicklight_stavrou_ceschia, 590 frames, ~90 px tall
# fighters): limb positions relative to the hips, in torso lengths, varied by
# a median 0.39 and 0.42 over the bout (0.31 and 0.36 smoothed); each
# fighter's head pointed towards the other in 78-80% of frames and away in
# 4-6%; the gap between them ran
# 0.41 to 1.15 body heights (10th-90th percentile). The lines below sit well
# clear of that, because calling a real fight "not a fight" is a false claim.
#
# Articulation: below this, neither person's arms or legs moved relative to
# their own body - the pose was fixed while the shape slid, as a photo does.
# Each limb is first averaged over +-SMOOTH_SECONDS: real limb movement is
# carried from frame to frame and survives that, while the pose model's
# frame-to-frame jitter on a still body does not. Measured: a real pose slid
# rigidly with 2 px of jitter on an 18 px torso reads 0.08-0.10 smoothed
# (0.18-0.22 unsmoothed, which would have passed); the real bout reads
# 0.31 and 0.34.
MIN_ARTICULATION = 0.12
SMOOTH_SECONDS = 0.4
# Frames a signal needs before it may decide anything.
MIN_SIGNAL_FRAMES = 30
# Head offset from the shoulder line, in torso lengths, that counts as
# pointing towards or away from the other person.
FACING_OFFSET = 0.05
# Both facing away from each other in at least this share of their frames.
MAX_FACING_AWAY = 0.5
# Neither faced the other in even this share of frames...
MIN_FACING_TOWARD = 0.3
# ...and the distance between them never changed by more than this (10th to
# 90th percentile, body heights): no exchange happened.
MIN_GAP_SPREAD = 0.15

_TORSO = (5, 6, 11, 12)
_ARTICULATED = (7, 8, 9, 10, 13, 14, 15, 16)   # elbows, wrists, knees, ankles
_FACE = (0, 1, 2, 3, 4)

PLAUSIBILITY_TEXT = {
    "rigid": ("neither person's arms or legs moved relative to their body: the poses stayed fixed "
              "while the shapes slid across the frame, as photos or cut-outs do"),
    "facing_away": "the two people faced away from each other most of the time",
    "no_engagement": "the two people never faced each other and the distance between them never changed",
}

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


def _confident(conf, points, index) -> bool:
    return (float(conf[index]) >= _KEYPOINT_CONF and float(points[index][0]) > 0
            and float(points[index][1]) > 0)


def _pose_shape(obs) -> np.ndarray | None:
    """Limb positions relative to the hips, in torso lengths; NaN where unseen."""
    points, conf = getattr(obs, "keypoints", None), getattr(obs, "keypoint_conf", None)
    if points is None or conf is None or len(conf) < 17:
        return None
    if not all(_confident(conf, points, i) for i in _TORSO):
        return None
    points = np.asarray(points, dtype=np.float32)[:, :2]
    hips = (points[11] + points[12]) / 2.0
    torso = float(np.linalg.norm((points[5] + points[6]) / 2.0 - hips))
    if torso < 3.0:
        return None
    return np.array([(points[i] - hips) / torso if _confident(conf, points, i) else (np.nan, np.nan)
                     for i in _ARTICULATED], dtype=np.float32)


def _facing(obs, other_box) -> float | None:
    """Head offset from the shoulders towards the other person, in torso lengths."""
    points, conf = getattr(obs, "keypoints", None), getattr(obs, "keypoint_conf", None)
    if points is None or conf is None or len(conf) < 17:
        return None
    if not (_confident(conf, points, 5) and _confident(conf, points, 6)):
        return None
    face = [points[i] for i in _FACE if _confident(conf, points, i)]
    if not face:
        return None
    points = np.asarray(points, dtype=np.float32)[:, :2]
    shoulders = (points[5] + points[6]) / 2.0
    scale = float(np.linalg.norm(shoulders - (points[11] + points[12]) / 2.0)) \
        if _confident(conf, points, 11) and _confident(conf, points, 12) else 0.0
    scale = scale if scale >= 3.0 else max(3.0, abs(float(points[5][0] - points[6][0])))
    head = np.mean(np.asarray(face, dtype=np.float32)[:, :2], axis=0)
    other_x = (float(other_box[0]) + float(other_box[2])) / 2.0
    direction = 1.0 if other_x >= float(shoulders[0]) else -1.0
    return float(head[0] - shoulders[0]) / scale * direction


class FightPresence:
    """Classify analysed frames into fight and non-fight windows."""

    def __init__(self, start_seconds: float, end_seconds: float):
        self.start = float(start_seconds)
        self.end = max(float(end_seconds), self.start)
        self._windows: dict[int, _Window] = {}
        self._last: dict[str, tuple[float, np.ndarray, np.ndarray | None]] = {}
        # The plausibility signals, over the whole analysed span.
        self._shapes: dict[str, list[tuple[float, np.ndarray]]] = {"A": [], "B": []}
        self._facing: dict[str, list[float]] = {"A": [], "B": []}
        self._gaps: list[float] = []

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
        for name, obs in (("A", fighter_a), ("B", fighter_b)):
            shape = _pose_shape(obs) if obs is not None else None
            if shape is not None:
                self._shapes[name].append((float(seconds), shape))
        if fighter_a is None or fighter_b is None:
            return
        for name, obs, other in (("A", fighter_a, fighter_b), ("B", fighter_b, fighter_a)):
            facing = _facing(obs, other.box)
            if facing is not None:
                self._facing[name].append(facing)
        self._gaps.append(_separation(fighter_a.box, fighter_b.box))
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

    def plausibility(self) -> dict:
        """Whether what was tracked moved like two people fighting.

        Fails only on positive evidence (see the constants above); a signal
        with too few frames to judge decides nothing, and the verdict says so.
        """
        articulation = {}
        for name, shapes in self._shapes.items():
            if len(shapes) < MIN_SIGNAL_FRAMES:
                articulation[name] = None
                continue
            times = np.asarray([t for t, _ in shapes], dtype=np.float64)
            stacked = np.stack([shape for _, shape in shapes])
            smoothed = np.empty_like(stacked)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                order = np.argsort(times, kind="stable")
                times, stacked = times[order], stacked[order]
                low = np.searchsorted(times, times - SMOOTH_SECONDS, side="left")
                high = np.searchsorted(times, times + SMOOTH_SECONDS, side="right")
                for index in range(len(times)):
                    smoothed[index] = np.nanmean(stacked[low[index]:high[index]], axis=0)
                spread = np.nanstd(smoothed, axis=0)
            per_limb = np.linalg.norm(spread, axis=1)
            per_limb = per_limb[np.isfinite(per_limb)]
            articulation[name] = round(float(np.median(per_limb)), 3) if per_limb.size else None
        facing = {}
        for name, offsets in self._facing.items():
            if len(offsets) < MIN_SIGNAL_FRAMES:
                facing[name] = None
                continue
            values = np.asarray(offsets, dtype=np.float32)
            facing[name] = {"toward": round(float(np.mean(values > FACING_OFFSET)), 3),
                            "away": round(float(np.mean(values < -FACING_OFFSET)), 3),
                            "frames": int(values.size)}
        gap_spread = None
        if len(self._gaps) >= MIN_SIGNAL_FRAMES:
            gap_spread = round(float(np.percentile(self._gaps, 90) - np.percentile(self._gaps, 10)), 3)

        reasons = []
        measured = [value for value in articulation.values() if value is not None]
        if measured and len(measured) == len(articulation) and max(measured) < MIN_ARTICULATION:
            reasons.append("rigid")
        both_facing = facing["A"] is not None and facing["B"] is not None
        if both_facing and min(facing["A"]["away"], facing["B"]["away"]) >= MAX_FACING_AWAY:
            reasons.append("facing_away")
        elif (both_facing and max(facing["A"]["toward"], facing["B"]["toward"]) < MIN_FACING_TOWARD
              and gap_spread is not None and gap_spread < MIN_GAP_SPREAD):
            reasons.append("no_engagement")
        return {
            "plausible": not reasons,
            "reasons": reasons,
            "text": [PLAUSIBILITY_TEXT[reason] for reason in reasons],
            "signals": {"articulation": articulation, "facing": facing, "gap_spread": gap_spread},
        }

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
            "plausibility": self.plausibility(),
        }


# Verdicts for saved reports, by tracking file and modification time.
_TRACKING_CACHE: dict[str, tuple[int, dict | None]] = {}


def plausibility_from_tracking(path: Path) -> dict | None:
    """The plausibility verdict for a saved analysis, from its tracking.jsonl.

    Reports saved before 2026-10-07 carry no verdict, but the per-frame
    observations the analysis used are kept beside them, so the same check is
    run on the same frames. None when there is no readable track.
    """
    from core.types import PersonObservation

    path = Path(path)
    try:
        modified = path.stat().st_mtime_ns
    except OSError:
        return None
    cached = _TRACKING_CACHE.get(str(path))
    if cached and cached[0] == modified:
        return cached[1]

    def observation(entry):
        observed = (entry or {}).get("observation")
        if not observed or observed.get("box") is None:
            return None
        return PersonObservation(
            track_id=observed.get("track_id"), box=np.asarray(observed["box"], dtype=np.float32),
            confidence=float(observed.get("confidence") or 0.0),
            keypoints=None if observed.get("keypoints") is None else np.asarray(observed["keypoints"], dtype=np.float32),
            keypoint_conf=(None if observed.get("keypoint_conf") is None
                           else np.asarray(observed["keypoint_conf"], dtype=np.float32)))

    verdict = None
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        presence = None
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                seconds = float(row.get("time_seconds") or 0.0)
                if presence is None:
                    presence = FightPresence(seconds, seconds)
                presence.observe(seconds, observation(row.get("fighter_A")), observation(row.get("fighter_B")))
        verdict = presence.plausibility() if presence is not None else None
    except (OSError, ValueError, KeyError, TypeError):
        verdict = None
    _TRACKING_CACHE[str(path)] = (modified, verdict)
    return verdict


def attach_plausibility(report: dict, tracking_path: Path | None) -> dict:
    """Give a saved two-fighter report its plausibility verdict, in memory only."""
    video = report.setdefault("video", {})
    if isinstance(video.get("plausibility"), dict) or report.get("mode") == "solo" or tracking_path is None:
        return report
    verdict = plausibility_from_tracking(tracking_path)
    if verdict is not None:
        video["plausibility"] = verdict
    return report
