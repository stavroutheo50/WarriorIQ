"""Was each strike credited to the right fighter? A second opinion from the shorts.

Shadow mode: this records what it *would* change and changes nothing. A
strike's fighter is still whoever the identity manager says; the report
carries this module's proposals under ``tracking.kit_attribution`` so they
can be checked on real uploads before anything acts on them.

Why the shorts. In bare-chested sports the identity manager's colour check
(core/identity.py appearance_hist) looks at the torso, in hue and saturation
only - so two fighters in white and black shorts look the same to it, and its
swap correction cannot fire. Broadcast camera cuts and fighters crossing
then hand one fighter's box to the other. Measured on UFC 326 Holloway vs
Oliveira 2, rounds 1, 3 and 4 (480x220 screen recordings; 54 strike moments
checked by eye): 14 of 54 strikes were credited to the wrong fighter.

What it reads: the share of near-black pixels (CIE L* below DARK_L) on each
fighter's shorts - the band from the hips 45% of the way to the knees, from
the pose. One fighter in dark shorts and one in light shorts is the common
case this can help; anything else, it stays silent.

Which fighter wears the darker shorts is decided twice, independently, and
must agree: on the frames around the one the person picked the fighters on,
and by the majority of every frame of the video. The selection frame alone
is not safe - on round 3 the fighter's hands covered the shorts and it read
the wrong way round; the majority alone is not safe - on rounds 1 and 4 it was
right by 59-61% only. When both agree, a strike is flagged when the frames
within WINDOW_SECONDS of it (not crossing a camera cut) say the boxes were
the other way round.

On those three rounds, both agreeing on rounds 1 and 4 only: the wrong-
fighter strikes would go 14 -> 5, with no correct strike flagged. One fight
is not enough to act on, hence shadow mode.
"""

from __future__ import annotations

import bisect

import cv2
import numpy as np

_HIPS = (11, 12)
_KNEES = (13, 14)
_JOINT_CONFIDENCE = 0.3
# L* (0-255 OpenCV scale) under which a pixel counts as near-black.
DARK_L = 70
# A frame votes only when the two fighters' dark shares differ by this much.
MIN_FRAME_DIFFERENCE = 0.15
# How far from a strike the frames that judge it may be.
WINDOW_SECONDS = 0.5
# Seconds either side of the selection frame used for the first reading.
SELECTION_WINDOW_SECONDS = 2.0
# The two fighters must differ this much at selection to be told apart.
MIN_SELECTION_DIFFERENCE = 0.25
# A frame whose colour histogram correlates less than this with the previous
# analysed frame starts a new camera shot.
CUT_CORRELATION = 0.6
MIN_PIXELS = 10


def shorts_dark_share(frame: np.ndarray | None, keypoints, keypoint_conf) -> float | None:
    """Share of near-black pixels on the shorts, or None when they are not visible."""
    if frame is None or keypoints is None or keypoint_conf is None:
        return None
    if len(keypoints) <= max(_KNEES) or len(keypoint_conf) <= max(_KNEES):
        return None
    if min(float(keypoint_conf[i]) for i in (*_HIPS, *_KNEES)) < _JOINT_CONFIDENCE:
        return None
    hip_l, hip_r = np.asarray(keypoints[11], float)[:2], np.asarray(keypoints[12], float)[:2]
    knee_l, knee_r = np.asarray(keypoints[13], float)[:2], np.asarray(keypoints[14], float)[:2]
    band = np.array([hip_l, hip_r, hip_r + 0.45 * (knee_r - hip_r), hip_l + 0.45 * (knee_l - hip_l)], np.int32)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [band], 1)
    if int(mask.sum()) < MIN_PIXELS:
        return None
    lightness = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0][mask > 0]
    return float((lightness < DARK_L).mean())


def _shot_signature(frame: np.ndarray) -> np.ndarray:
    small = cv2.cvtColor(cv2.resize(frame, (96, 44)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([small], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _darker(a: float | None, b: float | None) -> str | None:
    """"A" or "B" for whichever box's shorts are clearly darker this frame."""
    if a is None or b is None or abs(a - b) < MIN_FRAME_DIFFERENCE:
        return None
    return "A" if a > b else "B"


class KitAttribution:
    """Fed every analysed frame; at the end, proposes which strikes were miscredited."""

    def __init__(self, selection_seconds: float | None) -> None:
        self.selection_seconds = selection_seconds
        self._times: list[float] = []
        self._shots: list[int] = []
        self._shares: list[tuple[float | None, float | None]] = []
        self._shot = 0
        self._last_signature = None

    def observe(self, seconds: float, frame: np.ndarray | None, fighter_a, fighter_b) -> None:
        if frame is not None:
            signature = _shot_signature(frame)
            if self._last_signature is not None and \
                    cv2.compareHist(self._last_signature, signature, cv2.HISTCMP_CORREL) < CUT_CORRELATION:
                self._shot += 1
            self._last_signature = signature
        shares = tuple(
            shorts_dark_share(frame, getattr(f, "keypoints", None), getattr(f, "keypoint_conf", None))
            if f is not None else None
            for f in (fighter_a, fighter_b))
        self.add(seconds, self._shot, shares[0], shares[1])

    def add(self, seconds: float, shot: int, share_a: float | None, share_b: float | None) -> None:
        """One analysed frame, already measured. observe() calls this; tests call it directly."""
        self._times.append(float(seconds))
        self._shots.append(int(shot))
        self._shares.append((share_a, share_b))

    def _darker_at_selection(self) -> str | None:
        if self.selection_seconds is None:
            return None
        a = [s[0] for t, s in zip(self._times, self._shares)
             if abs(t - self.selection_seconds) <= SELECTION_WINDOW_SECONDS and s[0] is not None]
        b = [s[1] for t, s in zip(self._times, self._shares)
             if abs(t - self.selection_seconds) <= SELECTION_WINDOW_SECONDS and s[1] is not None]
        if not a or not b:
            return None
        median_a, median_b = float(np.median(a)), float(np.median(b))
        if abs(median_a - median_b) < MIN_SELECTION_DIFFERENCE:
            return None
        return "A" if median_a > median_b else "B"

    def _darker_by_majority(self) -> str | None:
        votes = [_darker(a, b) for a, b in self._shares]
        a, b = votes.count("A"), votes.count("B")
        if a == b:
            return None
        return "A" if a > b else "B"

    def _local_darker(self, seconds: float) -> str | None:
        """Whichever box the frames near this moment say wears the darker shorts."""
        if not self._times:
            return None
        k = min(bisect.bisect_left(self._times, seconds), len(self._times) - 1)
        shot = self._shots[k]
        score = 0
        for j in range(len(self._times)):
            if self._shots[j] != shot or abs(self._times[j] - seconds) > WINDOW_SECONDS:
                continue
            vote = _darker(*self._shares[j])
            score += 1 if vote == "A" else -1 if vote == "B" else 0
        return "A" if score > 0 else "B" if score < 0 else None

    def proposals(self, events) -> dict:
        """What this would change, without changing it."""
        at_selection, by_majority = self._darker_at_selection(), self._darker_by_majority()
        agreed = at_selection is not None and at_selection == by_majority
        flagged = []
        if agreed:
            for event in events:
                fighter = getattr(event, "fighter", None)
                peak = getattr(event, "peak_time", None)
                if fighter not in ("A", "B") or peak is None:
                    continue
                local = self._local_darker(float(peak))
                if local is not None and local != at_selection:
                    flagged.append({"seconds": round(float(peak), 3), "credited": fighter,
                                    "proposed": "B" if fighter == "A" else "A",
                                    "technique": getattr(event, "technique", None)})
        return {
            "mode": "shadow",
            "darker_shorts_at_selection": at_selection,
            "darker_shorts_by_majority": by_majority,
            "active": agreed,
            "strikes_checked": len(list(events)) if agreed else 0,
            "would_recredit": flagged,
        }
