"""Coverage says a fighter was held. This asks whether anybody was in the box.

Coverage counts a frame as held when a fighter has an observation. It never
asks whether that observation is on a person. On fight 1 that gap is large
enough to change what the number means: at f3380 both fighters stand plainly
visible and unboxed while both held boxes sit on bare mat, and the frame still
counts toward 88.8% and 66.9%.

Every identity measurement taken from such a box - appearance similarity, pose
similarity, the anchor comparison - is computed on mat texture. Seven attempts
to fix an apparent identity swap on that fight all read as "flat" for this
reason, so this tool exists before any further identity work does.

**The referee is the crop detector, not the whole frame.** The whole frame
misses people constantly on compressed footage - a median of 1 person found
where a crop of the same region finds 7 - so "nothing overlaps this box" would
prove nothing. A window around the held box, inferred at its own size, is the
most capable look available, which is what makes its silence meaningful.

**The window is calibrated, not guessed.** Five boxes were judged by eye first
and the window chosen to agree with them:

    frame/side  what the eye says            tight   WIDER   widest
    f1528 B     on the dark fighter           0.07    0.82    0.72
    f1578 B     on the dark fighter           0.11    0.93    0.93
    f1661 B     on the dark fighter           0.01    0.01    0.00
    f1528 A     on the blue fighter           0.91    0.90    0.92
    f2678 A     empty mat beside a fighter    0.14    0.15    0.14

A tight crop fills the frame with the person and the detector stops seeing
them, which made the first version of this tool report 60% of fighter B's
boxes as empty when many plainly were not. The wider window agrees with the
eye on four of five.

**Known bias: this over-reports "on nobody".** f1661 is the fifth case - two
fighters adjacent, merged by the detector into one box that overlaps the other
fighter's slot. So treat the empty figure as an upper bound.

Measured 2026-09-21:

    athens_hd, 1920x1080   A 100% on a person, B 100%, median overlap 0.94
    fight 1,   480x220     A  21% on NOBODY,   B  37%, median 0.50 / 0.44

The tracker is not the problem on footage it can see. Read every coverage
number against this one.

    tools/measure_held_boxes.py --trace run.json --video fights/1.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import SETTINGS


def _device():
    """Resolve SETTINGS.device the way PoseTracker does.

    "auto" is this project's word, not Ultralytics', which rejects it outright
    and names every other option in the error while never mentioning that one.
    """
    import torch

    requested = str(SETTINGS.device or "auto")
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return int(requested) if requested.isdigit() else requested

# Chosen by the calibration in the docstring. Wide enough that a fighter does
# not fill the crop, which is where the detector stops finding them.
WINDOW_WIDTHS = 2.5
WINDOW_HEIGHTS = 1.5
ON_SOMEBODY = 0.50
ON_NOBODY = 0.20


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def occupancy(trace_path: Path, video: str, samples: int) -> dict:
    from ultralytics import YOLO

    frames = json.loads(trace_path.read_text(encoding="utf-8"))["frames"]
    model = YOLO(SETTINGS.pose_model_pt)
    capture = cv2.VideoCapture(video)
    results: dict[str, dict] = {}
    try:
        for side in ("a", "b"):
            held = [f for f in frames if f.get("%s_box" % side)]
            if not held:
                results[side.upper()] = {"held": 0, "sampled": 0, "overlaps": []}
                continue
            picks = [held[i] for i in np.linspace(
                0, len(held) - 1, min(samples, len(held))).astype(int)]
            overlaps, empty_frames = [], []
            for row in picks:
                box = [float(v) for v in row["%s_box" % side]]
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
                ok, image = capture.read()
                if not ok:
                    continue
                height, width = image.shape[:2]
                bw, bh = box[2] - box[0], box[3] - box[1]
                x1 = int(max(0, box[0] - WINDOW_WIDTHS * bw))
                y1 = int(max(0, box[1] - WINDOW_HEIGHTS * bh))
                x2 = int(min(width, box[2] + WINDOW_WIDTHS * bw))
                y2 = int(min(height, box[3] + WINDOW_HEIGHTS * bh))
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                found = model.predict(crop, imgsz=640, conf=0.15, classes=[0],
                                      verbose=False, device=_device())[0]
                people = (found.boxes.xyxy.cpu().numpy()
                          if found.boxes is not None else np.zeros((0, 4)))
                best = max([0.0] + [
                    _iou(box, [p[0] + x1, p[1] + y1, p[2] + x1, p[3] + y1]) for p in people])
                overlaps.append(best)
                if best < ON_NOBODY:
                    empty_frames.append(int(row["frame"]))
            results[side.upper()] = {"held": len(held), "sampled": len(overlaps),
                                     "overlaps": overlaps, "empty_frames": empty_frames}
    finally:
        capture.release()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trace", required=True,
                        help="a file from measure_identity.py --trace-out")
    parser.add_argument("--video", required=True)
    parser.add_argument("--samples", type=int, default=150)
    arguments = parser.parse_args()

    results = occupancy(Path(arguments.trace), arguments.video, arguments.samples)
    for side, data in results.items():
        overlaps = np.array(data["overlaps"])
        if not overlaps.size:
            print("fighter %s: nothing held" % side)
            continue
        print("fighter %s: %d of %d held boxes sampled" % (side, data["sampled"], data["held"]))
        print("   clearly on somebody (>= %.2f) : %3d (%.0f%%)"
              % (ON_SOMEBODY, (overlaps >= ON_SOMEBODY).sum(), 100.0 * (overlaps >= ON_SOMEBODY).mean()))
        print("   partly on somebody            : %3d (%.0f%%)"
              % (((overlaps >= ON_NOBODY) & (overlaps < ON_SOMEBODY)).sum(),
                 100.0 * ((overlaps >= ON_NOBODY) & (overlaps < ON_SOMEBODY)).mean()))
        print("   ON NOBODY (< %.2f)             : %3d (%.0f%%)   <- an upper bound, see the docstring"
              % (ON_NOBODY, (overlaps < ON_NOBODY).sum(), 100.0 * (overlaps < ON_NOBODY).mean()))
        print("   median overlap %.2f" % float(np.median(overlaps)))
        if data["empty_frames"]:
            print("   look at these: %s"
                  % ", ".join("f%d" % f for f in data["empty_frames"][:10]))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
