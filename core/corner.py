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


def read_corners(samples) -> CornerReading:
    """Decide, for one fight, whether corner colour is usable and where.

    `samples` is an iterable of (frame, people), where people are the
    detections that are not the referee. Both regions are measured over the
    whole sample and the better one wins, because which region carries the
    colour depends on the ruleset and cannot be known in advance.
    """
    totals = {region: [] for region in REGIONS}
    for frame, people in samples:
        if frame is None or people is None or len(people) < 2:
            continue
        for region in REGIONS:
            totals[region].append(_frame_separation(frame, people, region))

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
