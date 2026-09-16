"""Which fighter is the red corner and which is the blue one.

Identity today asks "does this person look like whoever I decided was A?". That
is a comparison against a remembered template, so it drifts, and once it has
drifted it has no way back - the wrong answer becomes the new template. Measured
on this project's own footage, the box is on the right fighter 53-62% of the
time.

Corner colour asks a different question: "is this the red one?" That one is
answerable from scratch on any frame, which means it can re-anchor after a loss
rather than compounding it. It needs no labels, because the answer is in the
competition rules - WAKO, Olympic boxing and taekwondo all mandate red against
blue - so the supervision is free.

**Where the colour lives is not fixed, which is the part that has to be
measured rather than assumed.** Sampled on the four fights in this project:

    fight             torso          head+hands+feet
    5736 (broadcast)  72% clean      51% clean        red .95 / blue .98
    b883 (hall)        0% clean      51% clean        red .38 / blue .64
    f84e (hall)        1% clean      51% clean        red .38 / blue .64
    1.mp4             0% clean      12% clean         weak either way

The first fight carries the corner on the uniform; the others carry it on
gloves, headgear and footgear, which is normal for point-fighting. Sampling
either region alone finds nothing on half the library. So the region is chosen
per fight, from evidence, and a fight that separates on neither is reported as
undecided rather than guessed at - fight 4 above is what that looks like.

Nothing here assigns identity. It reports whether this footage has a usable
corner signal and how strong it is, and scores one detection at a time. What
uses that is core/identity.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# COCO keypoint order, as the pose model emits it.
_NOSE, _EYES, _EARS = 0, (1, 2), (3, 4)
_WRISTS, _ANKLES = (9, 10), (15, 16)
_SHOULDERS, _HIPS = (5, 6), (11, 12)

# The two places a corner colour is ever worn. "gear" is head, hands and feet:
# headguard, gloves and footpads, which is where point-fighting rules put it.
GEAR_JOINTS = (_NOSE,) + _EYES + _EARS + _WRISTS + _ANKLES
REGIONS = ("torso", "gear")

# A pixel only votes on colour if it is actually coloured. Competition mats in
# this library are themselves red and blue, so an unsaturated or dark pixel is
# far more likely to be floor, shadow or motion blur than it is to be kit.
_MIN_SATURATION = 90
_MIN_VALUE = 60
_MIN_VOTING_PIXELS = 8

# OpenCV hue is 0-179. Red wraps the origin, which is why it is two ranges.
_RED_LOW, _RED_HIGH = 10, 170
_BLUE_LOW, _BLUE_HIGH = 100, 135

# What counts as a fighter reading "red" or "blue" rather than neither. Set
# from the measurements above: on footage that genuinely carries corners the
# winning side scores 0.38-0.98, and on footage that does not it scores 0.00.
CLEAR_COLOUR = 0.25

# How much of the fight has to separate cleanly before the signal is trusted.
# The three fights that carry corners land at 51-72%; the one that does not
# lands at 12%. Half is the gap between them.
DECIDED_FRACTION = 0.40

# How many analysed frames of the round to watch before taking the fight's
# answer, when there were no warm-up frames to take it from.
#
# A clip analysed from frame 0 has no warm-up: warm_start == start_frame, the
# warm-up loop never runs, and the reader was handed nothing - so every such
# fight silently fell back to the single seed frame, which is the fragility
# this reader exists to remove. Sampling the round itself costs no extra
# decode, because those frames are being tracked anyway.
#
# Small enough that corners are on for almost all of the fight, and large
# enough that the answer is the fight's rather than one frame's.
DECIDE_AFTER_FRAMES = 40


@dataclass(frozen=True)
class CornerReading:
    """What a whole fight says about its own corner colours."""

    region: str | None = None
    separation: float = 0.0
    frames_scored: int = 0
    frames_clean: int = 0
    decided: bool = False
    reason: str = "not measured"
    per_region: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "region": self.region,
            "separation": round(self.separation, 3),
            "frames_scored": self.frames_scored,
            "frames_clean": self.frames_clean,
            "decided": self.decided,
            "reason": self.reason,
            "per_region": {k: round(v, 3) for k, v in self.per_region.items()},
        }


def _region_pixels(frame, box, keypoints, region: str):
    """The pixels of one body region, or None when the pose does not place it.

    Sampled from keypoints rather than from the box because the mat is red and
    blue: a box-wide sample on this footage measures mostly floor.
    """
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), np.uint8)

    def usable(index):
        if keypoints is None or index >= len(keypoints):
            return None
        point = keypoints[index]
        if point is None or len(point) < 2:
            return None
        if not (np.isfinite(point[0]) and np.isfinite(point[1])):
            return None
        return int(point[0]), int(point[1])

    if region == "torso":
        corners = [usable(i) for i in (_SHOULDERS[0], _SHOULDERS[1], _HIPS[1], _HIPS[0])]
        if any(c is None for c in corners):
            # No torso quad: fall back to the upper middle of the box, which is
            # the same place, just without the pose to prove it.
            x1, y1, x2, y2 = (int(v) for v in box)
            cx, cy = (x1 + x2) // 2, y1 + int((y2 - y1) * 0.35)
            rx, ry = max(2, (x2 - x1) // 5), max(2, (y2 - y1) // 8)
            corners = [(cx - rx, cy - ry), (cx + rx, cy - ry),
                       (cx + rx, cy + ry), (cx - rx, cy + ry)]
        cv2.fillPoly(mask, [np.array(corners, np.int32)], 255)
    elif region == "gear":
        # Discs scaled to the fighter, so a 60px subject and a 300px one sample
        # a comparable share of the glove rather than a fixed pixel count.
        radius = max(3, int((float(box[3]) - float(box[1])) * 0.07))
        placed = 0
        for index in GEAR_JOINTS:
            point = usable(index)
            if point is None:
                continue
            cv2.circle(mask, point, radius, 255, -1)
            placed += 1
        if not placed:
            return None
    else:
        raise ValueError("unknown region %r" % region)

    ys, xs = np.nonzero(mask)
    if len(ys) < _MIN_VOTING_PIXELS:
        return None
    return frame[ys, xs]


def colour_scores(pixels) -> tuple[float, float]:
    """(red, blue) as the share of coloured pixels reading each hue."""
    if pixels is None or len(pixels) < _MIN_VOTING_PIXELS:
        return 0.0, 0.0
    hsv = cv2.cvtColor(np.asarray(pixels).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    hsv = hsv.reshape(-1, 3)
    hue = hsv[:, 0].astype(np.int16)
    voting = (hsv[:, 1] >= _MIN_SATURATION) & (hsv[:, 2] >= _MIN_VALUE)
    if int(voting.sum()) < _MIN_VOTING_PIXELS:
        return 0.0, 0.0
    hue = hue[voting]
    red = float(((hue <= _RED_LOW) | (hue >= _RED_HIGH)).mean())
    blue = float(((hue >= _BLUE_LOW) & (hue <= _BLUE_HIGH)).mean())
    return red, blue


def score_detection(frame, box, keypoints, region: str) -> tuple[float, float]:
    """How red and how blue one detection is, in the region this fight uses."""
    return colour_scores(_region_pixels(frame, box, keypoints, region))


def score_people(frame, people, region: str | None) -> None:
    """Attach corner colours to every detection in a frame, in place.

    Mirrors how core/reid.py and core/referee.py are applied: one pass over
    the frame's people, writing onto each observation, so the identity manager
    never has to hold a frame. A region of None leaves every reading None,
    which is what turns the whole signal off for a fight that has no corner.
    """
    if not region or frame is None or not people:
        return
    for person in people:
        try:
            red, blue = score_detection(
                frame, person.box, getattr(person, "keypoints", None), region)
        except Exception:                       # noqa: BLE001 - never fail a frame
            red = blue = None
        person.corner_red, person.corner_blue = red, blue


@dataclass
class CornerAssignment:
    """Which fighter is the red corner, and what that verdict was read from."""

    corner_a: str | None = None
    corner_b: str | None = None
    frames: int = 0
    red_a: float = 0.0
    blue_a: float = 0.0
    red_b: float = 0.0
    blue_b: float = 0.0

    @property
    def decided(self) -> bool:
        return self.corner_a is not None and self.corner_b is not None


def _decide_pair(red_a, blue_a, red_b, blue_b) -> tuple[str | None, str | None]:
    """Opposite corners, or nothing.

    Both have to read clearly, in opposite colours, with a margin between
    them - the same bar the refusal itself uses, because a pairing this is
    not sure of would put the refusal on the wrong fighter.
    """
    from core.identity import CORNER_MARGIN

    best, strength = (None, None), 0.0
    for a_corner, a_score, a_other, b_corner, b_score, b_other in (
            ("red", red_a, blue_a, "blue", blue_b, red_b),
            ("blue", blue_a, red_a, "red", red_b, blue_b)):
        if a_score < CLEAR_COLOUR or b_score < CLEAR_COLOUR:
            continue
        if a_score - a_other < CORNER_MARGIN or b_score - b_other < CORNER_MARGIN:
            continue
        if min(a_score, b_score) > strength:
            best, strength = (a_corner, b_corner), min(a_score, b_score)
    return best


def assign_corners(frame, box_a, keypoints_a, box_b, keypoints_b,
                   region: str) -> CornerAssignment:
    """Which of these two is the red corner, from this one frame.

    One frame, so one frame's worth of confidence. Measured on real footage
    mid-exchange, a fighter read red 0.41 against blue 0.44 in the gear
    region - both colours present, neither decisive - and no assignment was
    possible at all. CornerReader.assign() is the form that averages over a
    window and is what an analysis should use; this one is for the case where
    there is only a frame to look at.
    """
    red_a, blue_a = score_detection(frame, box_a, keypoints_a, region)
    red_b, blue_b = score_detection(frame, box_b, keypoints_b, region)
    corner_a, corner_b = _decide_pair(red_a, blue_a, red_b, blue_b)
    return CornerAssignment(corner_a=corner_a, corner_b=corner_b, frames=1,
                            red_a=red_a, blue_a=blue_a, red_b=red_b, blue_b=blue_b)


def _frame_separation(frame, people, region: str) -> float:
    """Best red/blue split among the two largest fighters in one frame.

    Returns the weaker of the pair's two scores, so a frame only scores well
    when one fighter reads red *and* the other reads blue. One fighter in
    clear red against a second the sampler cannot colour is not a separation.
    """
    scored = []
    for person in people:
        red, blue = score_detection(frame, person.box, getattr(person, "keypoints", None), region)
        area = (float(person.box[2]) - float(person.box[0])) * (float(person.box[3]) - float(person.box[1]))
        scored.append((area, red, blue))
    if len(scored) < 2:
        return 0.0
    scored.sort(key=lambda item: -item[0])
    (_, red_a, blue_a), (_, red_b, blue_b) = scored[0], scored[1]
    # Either fighter may be the red one; take whichever pairing reads better.
    return max(min(red_a, blue_b), min(red_b, blue_a))


class CornerReader:
    """Accumulates the evidence for a corner decision, one frame at a time.

    Separate from read_corners() so the analyser can feed it frames it is
    already decoding - the identity warm-up pass - instead of making a second
    pass over the video purely to look at colour. Only the per-frame numbers
    are kept; the frames themselves are not held, which matters because the
    warm-up can run to hundreds of them.
    """

    def __init__(self) -> None:
        self._totals: dict[str, list[float]] = {region: [] for region in REGIONS}
        # How the two *identified* fighters read, as opposed to the two largest
        # people in the frame. Kept separately because they answer different
        # questions - which region carries the colour, and which fighter wears
        # which - and because only identity knows which detection is fighter A.
        self._fighters: dict[str, list[tuple[float, float, float, float]]] = {
            region: [] for region in REGIONS}

    def observe(self, frame, people) -> None:
        """Measure one frame. Frames with fewer than two fighters are ignored."""
        if frame is None or people is None or len(people) < 2:
            return
        for region in REGIONS:
            self._totals[region].append(_frame_separation(frame, people, region))

    def observe_fighters(self, frame, observation_a, observation_b) -> None:
        """Record how each identified fighter reads, for the assignment.

        Frames where identity is not holding both fighters are skipped rather
        than half-recorded: a colour read off one fighter alone cannot say
        which corner either of them is in.
        """
        if frame is None or observation_a is None or observation_b is None:
            return
        for region in REGIONS:
            try:
                red_a, blue_a = score_detection(
                    frame, observation_a.box,
                    getattr(observation_a, "keypoints", None), region)
                red_b, blue_b = score_detection(
                    frame, observation_b.box,
                    getattr(observation_b, "keypoints", None), region)
            except Exception:               # noqa: BLE001 - never fail a frame
                continue
            self._fighters[region].append((red_a, blue_a, red_b, blue_b))

    @property
    def frames_scored(self) -> int:
        return len(self._totals[REGIONS[0]])

    def decide(self) -> CornerReading:
        return _decide(self._totals)

    def assign(self, region: str | None) -> CornerAssignment:
        """Which fighter is the red corner, averaged over everything seen.

        The region is a question about the whole fight and is answered from
        the whole window; this is the other half of the same answer and has to
        be read the same way. Deciding it from a single frame was measured
        failing on footage where the fight-level region was unambiguous: mid
        exchange one fighter read red 0.41 against blue 0.44 in the gear
        region, so nothing could be assigned and the whole signal switched
        off. Averaged over the window, a moment like that is outvoted.
        """
        looks = self._fighters.get(region) if region else None
        if not looks:
            return CornerAssignment()
        count = float(len(looks))
        red_a, blue_a, red_b, blue_b = (
            sum(look[index] for look in looks) / count for index in range(4))
        corner_a, corner_b = _decide_pair(red_a, blue_a, red_b, blue_b)
        return CornerAssignment(
            corner_a=corner_a, corner_b=corner_b, frames=len(looks),
            red_a=red_a, blue_a=blue_a, red_b=red_b, blue_b=blue_b)


def read_corners(samples) -> CornerReading:
    """Decide, for one fight, whether corner colour is usable and where.

    `samples` is an iterable of (frame, people), where people are the
    detections that are not the referee. Both regions are measured over the
    whole sample and the better one wins, because which region carries the
    colour depends on the ruleset and cannot be known in advance.
    """
    reader = CornerReader()
    for frame, people in samples:
        reader.observe(frame, people)
    return reader.decide()


def _decide(totals: dict[str, list[float]]) -> CornerReading:
    scored = len(totals[REGIONS[0]])
    if not scored:
        return CornerReading(reason="no frame had two fighters to compare")

    per_region = {}
    for region, values in totals.items():
        clean = sum(1 for value in values if value >= CLEAR_COLOUR)
        per_region[region] = clean / scored

    best = max(per_region, key=per_region.get)
    fraction = per_region[best]
    clean_frames = int(round(fraction * scored))
    if fraction < DECIDED_FRACTION:
        return CornerReading(
            region=None, separation=fraction, frames_scored=scored,
            frames_clean=clean_frames, decided=False, per_region=per_region,
            reason="no region separated the fighters often enough to trust")
    return CornerReading(
        region=best, separation=fraction, frames_scored=scored,
        frames_clean=clean_frames, decided=True, per_region=per_region,
        reason="corner colour read from %s" % best)
