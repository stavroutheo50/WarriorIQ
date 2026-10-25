"""Can two fighters be told apart by what they wear?

QA, 2026-09, found the old answer wrong in both directions:

  * yellow top with red headgear against blue top with blue headgear was
    called "89% alike" and the fight was not scored;
  * black top against blue/purple top was called "79% alike";
  * two identical white doboks passed with no warning;
  * a 1947 black-and-white bout read "100%", and the report blamed the camera.

The old measure was the correlation of hue/saturation histograms over the
upper-middle of each box. Three things made it unfit for this question:

  1. **Brightness was thrown away.** Hue and saturation cannot separate black
     from white or from grey - every achromatic kit is the same to them, which
     is why black-vs-blue scored high and why black-and-white footage is 100%.
  2. **Most of the pixels were not kit.** Skin (both fighters bare-chested, or
     faces and arms), the canvas and the ropes between the arms are shared by
     both crops and dominate a whole-crop histogram, so very different kits
     still correlated strongly.
  3. **One number for the whole body.** Fighters are told apart by whichever
     part differs - headgear, shorts, top - not by the average of everything.

This measures colour where kit is, in a space where distance means what the eye
sees: the median CIE L*a*b* colour of three regions near the centre line of
each box (head and headgear, torso, shorts), compared by Delta E. The pair is
as distinguishable as its *most different* region, so a matching top with
different shorts or headgear still separates them. It also says why a pair
cannot be told apart when it cannot - black-and-white footage, fighters too
small in the picture, or kit that genuinely matches - so the report can name
the real cause instead of one generic sentence.

This is only a colour question. Identity no longer fails on it alone: two
fighters in identical kit are followed by position and motion, and the report
is judged on whether that tracking actually held (core/report.py).
"""

from __future__ import annotations

import cv2
import numpy as np

# (top, bottom) as fractions of the box height, and the central share of its
# width - the body's centre line, away from the background between the arms.
REGIONS = {
    "head": (0.00, 0.16),
    "torso": (0.18, 0.48),
    "shorts": (0.48, 0.66),
}
CENTRE = (0.30, 0.70)
# Delta E at which a region counts as completely different. Two halves of one
# kit under one light differ by 2-8; a red and a blue kit by 60 or more.
DELTA_E_SCALE = 40.0
# Below this chroma (in a*b* units) a colour is grey: white, black or any shade
# between. Every region of both fighters under it means colour cannot help.
ACHROMATIC_CHROMA = 8.0
# Fighters shorter than this in pixels give regions a handful of pixels each.
MIN_BOX_HEIGHT = 60


def _lab(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _region_colour(lab: np.ndarray, box, region: tuple[float, float]) -> np.ndarray | None:
    height, width = lab.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    left = int(max(0, x1 + CENTRE[0] * bw))
    right = int(min(width, x1 + CENTRE[1] * bw))
    top = int(max(0, y1 + region[0] * bh))
    bottom = int(min(height, y1 + region[1] * bh))
    if right - left < 2 or bottom - top < 2:
        return None
    pixels = lab[top:bottom, left:right].reshape(-1, 3)
    return np.median(pixels, axis=0)


def kit_similarity(image: np.ndarray | None, box_a, box_b) -> dict | None:
    """How alike two fighters' kits are on this frame, 0 (nothing alike) to 1.

    Returns None when the frame or boxes cannot be read. The dict carries the
    per-region Delta E and the facts a report needs to explain a "too alike"
    verdict honestly.
    """
    if image is None or box_a is None or box_b is None:
        return None
    lab = _lab(image)
    regions = {}
    chromas = []
    for name, span in REGIONS.items():
        colour_a, colour_b = _region_colour(lab, box_a, span), _region_colour(lab, box_b, span)
        if colour_a is None or colour_b is None:
            continue
        delta = float(np.linalg.norm(colour_a - colour_b))
        regions[name] = {
            "delta_e": round(delta, 1),
            "similarity": round(max(0.0, 1.0 - delta / DELTA_E_SCALE), 3),
        }
        chromas.extend([float(np.hypot(colour_a[1], colour_a[2])), float(np.hypot(colour_b[1], colour_b[2]))])
    if not regions:
        return None
    similarity = min(item["similarity"] for item in regions.values())
    heights = [float(box_a[3]) - float(box_a[1]), float(box_b[3]) - float(box_b[1])]
    most_different = min(regions, key=lambda name: regions[name]["similarity"])
    return {
        "similarity": round(similarity, 3),
        "regions": regions,
        "most_different_region": most_different,
        "achromatic": bool(chromas) and max(chromas) < ACHROMATIC_CHROMA,
        "small": min(heights) < MIN_BOX_HEIGHT,
        "method": "kit_regions_lab_v1",
    }


def alike_reason(kit: dict | None) -> str:
    """Why two fighters look alike, as the cause rather than the symptom."""
    if not kit:
        return "kit"
    if kit.get("small"):
        return "small"
    if kit.get("achromatic"):
        return "black_and_white"
    return "kit"


# ------------------------------------------------- hand-offs checked by kit
# A region is used to tell the two fighters apart only when their kits differ
# there by at least this Delta E at selection: clearly different colours, not
# two shades of one kit under changing light.
HANDOFF_REGION_DELTA_E = 25.0
# A box is wearing a fighter's kit in a region when it is within this share of
# the A-to-B difference of that fighter's own colour, and nearer to it than to
# the opponent's.
HANDOFF_OWN_SHARE = 0.5
# Smallest box a hand-off is read on. Lower than MIN_BOX_HEIGHT, which decides
# whether the report blames fighters too small to compare at all: checking one
# known fighter against two known kits needs fewer pixels than that. On the
# owner's Light Contact fight f604d0c6d027 (568x320, fighters 57-90 px,
# 2026-10-25) 50 cleared 11 of 17 suspicious hand-offs against 9 at 60, and
# confirmed the opponent's box as the fighter 0 times in 40.
HANDOFF_MIN_BOX_HEIGHT = 50


def region_colours(image: np.ndarray | None, box) -> dict[str, list[float]]:
    """The median L*a*b* colour of each kit region of one box, as stored lists."""
    if image is None or box is None:
        return {}
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    left, top = int(max(0, x1)), int(max(0, y1))
    right, bottom = int(min(width, x2)), int(min(height, y2))
    if right - left < 2 or bottom - top < 2:
        return {}
    # Only the box is converted: this runs on frames where a fighter was
    # picked up again, not once per analysis.
    lab = _lab(image[top:bottom, left:right])
    local = (x1 - left, y1 - top, x2 - left, y2 - top)
    out = {}
    for name, span in REGIONS.items():
        colour = _region_colour(lab, local, span)
        if colour is not None:
            out[name] = [round(float(v), 2) for v in colour]
    return out


class HandoffKitCheck:
    """Was a fighter picked up again by a person wearing their kit?

    The identity manager counts a hand-off between tracker tracks as
    suspicious when the person picked up looks different by its torso colour
    histogram or is far from where the last track left them
    (core/identity.py). A hand-held phone at a crowded event does both to the
    right fighter all the time - the camera swings, the fighter is 85 pixels
    tall - and the "could not follow you" gate (core/report.py churn_rate)
    then refused fights the identity manager never confused.

    When the two fighters wear clearly different kit, the kit answers the
    question directly: the box picked up is read in the same regions as the
    boxes the person drew on the frame they chose, and the hand-off is
    confirmed only if every region that tells the two apart matches this
    fighter's colour and not the opponent's. Anything less - kit too alike,
    box too small, a region not readable, closer to the opponent - leaves the
    hand-off counted as suspicious, as before.
    """

    def __init__(self, colours: dict | None):
        colours = colours or {}
        a, b = colours.get("A") or {}, colours.get("B") or {}
        self.reference = {"A": {}, "B": {}}
        self.apart: dict[str, float] = {}
        for region in REGIONS:
            if region in a and region in b:
                delta = float(np.linalg.norm(np.subtract(a[region], b[region])))
                if delta >= HANDOFF_REGION_DELTA_E:
                    self.apart[region] = delta
                    self.reference["A"][region] = np.asarray(a[region], dtype=np.float32)
                    self.reference["B"][region] = np.asarray(b[region], dtype=np.float32)
        self.confirmed = {"A": 0, "B": 0}
        self.checked = {"A": 0, "B": 0}

    @property
    def enabled(self) -> bool:
        return bool(self.apart)

    def confirms(self, image: np.ndarray | None, box, fighter: str) -> bool:
        if not self.enabled or box is None or float(box[3]) - float(box[1]) < HANDOFF_MIN_BOX_HEIGHT:
            return False
        self.checked[fighter] += 1
        seen = region_colours(image, box)
        other = "B" if fighter == "A" else "A"
        for region, delta in self.apart.items():
            if region not in seen:
                return False
            colour = np.asarray(seen[region], dtype=np.float32)
            own = float(np.linalg.norm(colour - self.reference[fighter][region]))
            opponent = float(np.linalg.norm(colour - self.reference[other][region]))
            if own > HANDOFF_OWN_SHARE * delta or own >= opponent:
                return False
        self.confirmed[fighter] += 1
        return True
