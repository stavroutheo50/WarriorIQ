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
