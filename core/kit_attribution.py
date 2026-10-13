"""Was each strike credited to the right fighter? A second opinion from the shorts.

Active since 2026-10-13 (``WARRIORIQ_KIT_ATTRIBUTION``, default on; 0 puts
it back in shadow mode, recording what it would change and changing
nothing). A strike it flags is re-credited to the other fighter, and the
report lists every change under ``tracking.kit_attribution``.

Which fighter wears the darker shorts is now read from the boxes the person
drew on the selection frame (``box_shorts_dark_share``), the one place the
identities are known to be right. It acts only when one pair of shorts is
clearly dark and the other clearly light there. Measured on UFC 326 rounds 1
and 4 (every re-credited strike checked by eye): 43 strikes re-credited, 24
fixed, none made wrong, 19 in grappling piles where no one could say who
threw what. Silent on round 3 (shorts too close at selection) and on both
amateur benchmark clips (both fighters in dark shorts, or a box on a shirt).

The earlier rule below - the tracker's own boxes around the selection frame,
agreeing with the whole-video majority - is kept only as a record for
analyses without a drawn-box reading, and never moves a strike. It read round 1 the wrong way round on a rerun: the
tracker had the boxes swapped around the selection frame and on most frames
of the round, and it would have moved 18 correctly credited strikes.

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
import os

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
# The selection-frame reading from the drawn boxes: one pair of shorts must be
# at least this dark and the other at most this light. Measured there: UFC 326
# 0.73/0.05 and 0.63/0.00; both-dark amateur bout 0.97/0.62 (silent).
SELECTION_DARK_MIN = 0.50
SELECTION_LIGHT_MAX = 0.20
# The shorts band inside a drawn box: an upright fighter's hips to mid-thigh.
BOX_BAND_TOP, BOX_BAND_BOTTOM, BOX_BAND_SIDE = 0.50, 0.68, 0.20


def active() -> bool:
    return os.getenv("WARRIORIQ_KIT_ATTRIBUTION", "1").strip().lower() not in {"0", "false", "no", "off"}


def box_shorts_dark_share(frame: np.ndarray | None, box) -> float | None:
    """Share of near-black pixels in the shorts band of a box drawn around an upright fighter."""
    if frame is None or box is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    top, bottom = int(y1 + BOX_BAND_TOP * height), int(y1 + BOX_BAND_BOTTOM * height)
    left, right = int(x1 + BOX_BAND_SIDE * width), int(x2 - BOX_BAND_SIDE * width)
    crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
    if crop.size == 0 or crop.shape[0] * crop.shape[1] < MIN_PIXELS:
        return None
    return float((cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0] < DARK_L).mean())


def darker_at_selection_boxes(share_a: float | None, share_b: float | None) -> str | None:
    """"A"/"B" when one drawn fighter's shorts are clearly dark and the other's clearly light."""
    if share_a is None or share_b is None:
        return None
    if share_a >= SELECTION_DARK_MIN and share_b <= SELECTION_LIGHT_MAX:
        return "A"
    if share_b >= SELECTION_DARK_MIN and share_a <= SELECTION_LIGHT_MAX:
        return "B"
    return None


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
    # Only the band's own rectangle is converted, not the frame: this runs for
    # both fighters on every analysed frame.
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(band[:, 0].min())), max(0, int(band[:, 1].min()))
    x2, y2 = min(width, int(band[:, 0].max()) + 1), min(height, int(band[:, 1].max()) + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(mask, [band - np.array([x1, y1], np.int32)], 1)
    if int(mask.sum()) < MIN_PIXELS:
        return None
    lightness = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB)[..., 0][mask > 0]
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

    def __init__(self, selection_seconds: float | None, selection_darker: str | None = None) -> None:
        self.selection_seconds = selection_seconds
        # From the drawn boxes (darker_at_selection_boxes); decides alone when set.
        self.selection_darker = selection_darker
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
        # Frames arrive in time order, so the window is one slice.
        lo = bisect.bisect_left(self._times, seconds - WINDOW_SECONDS)
        hi = bisect.bisect_right(self._times, seconds + WINDOW_SECONDS)
        for j in range(lo, hi):
            if self._shots[j] != shot:
                continue
            vote = _darker(*self._shares[j])
            score += 1 if vote == "A" else -1 if vote == "B" else 0
        return "A" if score > 0 else "B" if score < 0 else None

    def proposals(self, events, apply: bool = False) -> dict:
        """Which strikes were credited to the wrong fighter; with ``apply``, re-credit them."""
        events = list(events)
        by_majority = self._darker_by_majority()
        if self.selection_darker is not None:
            at_selection, source = self.selection_darker, "drawn_boxes"
            agreed = True
        else:
            at_selection, source = self._darker_at_selection(), "tracked_boxes"
            agreed = at_selection is not None and at_selection == by_majority
        flagged, changed = [], []
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
                    changed.append(event)
        # Only the drawn-box reading may move a strike. The tracked-box reading
        # read a real round backwards (module docstring), so without a drawn-box
        # reading this stays a record of what it would have done.
        apply = apply and source == "drawn_boxes"
        if apply:
            for event in changed:
                event.fighter, event.opponent = event.opponent, event.fighter
                if isinstance(getattr(event, "evidence", None), dict):
                    event.evidence["recredited_by_shorts"] = True
        return {
            "mode": "active" if apply else "shadow",
            "selection_reading": source,
            "darker_shorts_at_selection": at_selection,
            "darker_shorts_by_majority": by_majority,
            "active": agreed,
            "strikes_checked": len(events) if agreed else 0,
            "would_recredit": flagged,
            "recredited": len(changed) if apply else 0,
        }
