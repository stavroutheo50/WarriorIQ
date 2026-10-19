"""Whether two fighter boxes can start an analysis, and which box a tap means.

Audit 2026-10-14:

#11 The selection step said "The two boxes cover the same person" whenever two
    different fighters stood close - an exchange, or a narrow vertical video -
    even for boxes the user drew. The old rule refused any pair whose overlap
    was 28% of the smaller box, which two people in a clinch always exceed.
    Now only boxes that are nearly the same box (IoU above MAX_IOU) are
    refused, and never when the user drew either one by hand or when both are
    separate people the detector found: those get a warning at most.

#12 Tapping where boxes overlap picked the merged box around both fighters.
    The tap now goes to the smallest box containing it, and boxes of about
    the same size are separated by whose centre is closest to the tap.

app/templates/select.html applies the same two rules in the browser; the
numbers here are the ones it uses.
"""

from __future__ import annotations

import math

MAX_IOU = 0.6
# Above this share of the smaller box, the pair is allowed but the user is
# asked to check each box holds a different fighter.
WARN_SHARED = 0.5
# Two candidate boxes within this ratio of area are "about the same size".
SIZE_TIE = 1.15


def _area(box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _intersection(a, b) -> float:
    return (max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
            * max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))))


def box_iou(a, b) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def pair_verdict(box_a, box_b, *, a_drawn: bool = False, b_drawn: bool = False,
                 both_detected: bool = False) -> dict:
    """{"verdict": "ok" | "warn" | "block", "iou", "shared", "message"} for a pair of boxes.

    ``a_drawn``/``b_drawn``: the user dragged that box by hand. ``both_detected``:
    each box is a different person the detector found (a tap on each).
    """
    iou = box_iou(box_a, box_b)
    shared = _intersection(box_a, box_b) / max(1.0, min(_area(box_a), _area(box_b)))
    result = {"iou": round(iou, 3), "shared": round(shared, 3), "verdict": "ok", "message": None}
    if iou > MAX_IOU:
        if a_drawn or b_drawn or both_detected:
            result.update(verdict="warn", message=(
                "The two boxes overlap a lot. That is fine when the fighters are close together - "
                "just check each box is on a different fighter."))
        else:
            result.update(verdict="block", message=(
                "The two boxes are almost the same box. Put each box on a different fighter."))
    elif shared >= WARN_SHARED:
        result.update(verdict="warn", message=(
            "The boxes overlap. That is fine in an exchange - just check each box is on a different fighter."))
    return result


def box_at(point, boxes, exclude=None):
    """The box a tap at ``point`` means: the smallest containing it, ties to the nearest centre."""
    x, y = float(point[0]), float(point[1])
    hits = [box for box in boxes
            if box is not exclude and float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])]
    if not hits:
        return None
    smallest = min(_area(box) for box in hits)
    tied = [box for box in hits if _area(box) <= smallest * SIZE_TIE]
    return min(tied, key=lambda box: math.hypot((float(box[0]) + float(box[2])) / 2 - x,
                                                (float(box[1]) + float(box[3])) / 2 - y))
