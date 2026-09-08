"""Several clips on one sheet, still large enough to judge.

One clip per image is honest but slow, and 178 of them is a day. The crops are
already tight on the subject and their opponent, so four key frames per clip at
3x is still a 180px-tall fighter - readable - and four clips fit on a sheet.

Frames chosen around the moment, not spread across the window: a strike is
thrown and retracted inside about a third of a second, and the frames that
settle it are the ones either side of the peak.

    tools/contact_sheet.py --job fam3 --ids 16,19,20,21 --out sheet.png
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

from tools.zoom_clip import VIDEOS, _tracking

OFFSETS = (-8, -4, 0, 4, 8)
TARGET_H = 200


def strip(job: str, clip_id: int, scale: int = 3) -> np.ndarray:
    index = json.loads((PROJECT_ROOT / "labelpack" / job / "index.json").read_text(encoding="utf-8"))
    item = next(c for c in index["candidates"] if c["id"] == clip_id)
    fighter, peak = item["fighter"], int(item["peak_frame"])
    other = "B" if fighter == "A" else "A"
    tracking = _tracking(job)

    # Centred on the subject and sized by *their* height, not by how far away
    # the opponent happens to be. Padding rows to a common width let one wide
    # shot shrink every other clip on the sheet to nothing; this way a fighter
    # occupies the same fraction of every panel whatever the camera did.
    centres, heights = [], []
    for off in OFFSETS:
        record = tracking.get(peak + off) or {}
        box = ((record.get("fighter_%s" % fighter) or {}).get("observation") or {}).get("box")
        if box:
            centres.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
            heights.append(max(12.0, box[3] - box[1]))
    cap = cv2.VideoCapture(VIDEOS[job])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if centres:
        cx = float(np.mean([c[0] for c in centres]))
        cy = float(np.mean([c[1] for c in centres]))
        # 2.4 body heights across, which holds a fighter, their limbs at full
        # extension, and whoever is close enough to be hit.
        half_h = 1.2 * float(np.median(heights))
        half_w = half_h
        x1, x2 = int(max(0, cx - half_w)), int(min(width, cx + half_w))
        y1, y2 = int(max(0, cy - half_h)), int(min(height, cy + half_h))
    else:
        x1, y1, x2, y2 = 0, 0, width, height
    if x2 - x1 < 30 or y2 - y1 < 30:
        x1, y1, x2, y2 = 0, 0, width, height

    panels = []
    for off in OFFSETS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak + off)
        ok, frame = cap.read()
        if not ok:
            continue
        record = tracking.get(peak + off) or {}
        view = frame.copy()
        for side, colour, thick in ((other, (150, 150, 150), 1), (fighter, (60, 200, 255), 2)):
            box = ((record.get("fighter_%s" % side) or {}).get("observation") or {}).get("box")
            if box:
                cv2.rectangle(view, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colour, thick)
        crop = view[y1:y2, x1:x2]
        target_h = TARGET_H
        ratio = target_h / max(1, crop.shape[0])
        big = cv2.resize(crop, (max(1, int(crop.shape[1] * ratio)), target_h), interpolation=cv2.INTER_CUBIC)
        if off == 0:
            cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (0, 190, 255), 3)
        panels.append(big)
    cap.release()
    if not panels:
        raise SystemExit("clip %d unreadable" % clip_id)
    tallest = max(p.shape[0] for p in panels)
    panels = [np.pad(p, ((0, tallest - p.shape[0]), (0, 0), (0, 0))) for p in panels]
    row = np.hstack(panels)
    banner = np.zeros((26, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, "#%d  %s  (%s)" % (clip_id, item["proposed"], item["source"]),
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, row])


def main() -> int:
    parser = argparse.ArgumentParser(description="Contact sheet of several clips.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--height", type=int, default=200,
                        help="panel height; raise it when the family is the question")
    args = parser.parse_args()
    globals()['TARGET_H'] = args.height
    rows = [strip(args.job, int(i), args.scale) for i in args.ids.split(",")]
    widest = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, widest - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(args.out, np.vstack(rows))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
