"""Render one clip large enough to actually judge.

The pack's filmstrip is built for speed: six thumbnails, a whole exchange in
one glance, fine for confirming a guess. It is not enough to *settle* a
disagreement between what the eye says and what the pose measured - at 480x220
a fighter is sixty pixels tall, and a jab and a lean look identical at that
size.

This crops to the subject and their opponent, keeps both in frame, and scales
up. More frames, bigger, with the moment marked. Used when the two readings
disagree, which is the only time it is worth the extra look.

    tools/zoom_clip.py --job fam3 --id 0 --out zoom.png
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

VIDEOS = {
    "fam3": "fights/1.mp4",
    "f2_gateh": "fights/0-02-05-5736bb3acb024e3772ad5fd6341d0b6768ccece93b7c329dcfa26f3ec1478f00_d242ef68be9b3ffe.mp4",
    "f3_gateh": "fights/0-02-05-f84ec82f3271783fcb884ca43f15f94aae2ec3086be1ca26774d5784b373e3b9_336b8df8343b952b.mp4",
}


def _tracking(job: str) -> dict[int, dict]:
    records = {}
    for line in (PROJECT_ROOT / "outputs" / job / "tracking.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("source_frame") is not None:
            records[int(record["source_frame"])] = record
    return records


def render(job: str, clip_id: int, frames: int = 8, step: int = 3, scale: int = 4) -> np.ndarray:
    index = json.loads((PROJECT_ROOT / "labelpack" / job / "index.json").read_text(encoding="utf-8"))
    item = next(c for c in index["candidates"] if c["id"] == clip_id)
    fighter = item["fighter"]
    other = "B" if fighter == "A" else "A"
    peak = int(item["peak_frame"])
    tracking = _tracking(job)

    offsets = [(n - frames // 2) * step for n in range(frames)]
    # One crop window for the whole strip, so the subject does not jump about
    # between panels - chosen to hold both fighters across the window.
    xs, ys = [], []
    for off in offsets:
        record = tracking.get(peak + off) or {}
        for side in (fighter, other):
            box = ((record.get("fighter_%s" % side) or {}).get("observation") or {}).get("box")
            if box:
                xs += [box[0], box[2]]
                ys += [box[1], box[3]]
    cap = cv2.VideoCapture(VIDEOS[job])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if xs:
        pad = 0.35 * (max(xs) - min(xs) + 1)
        x1, x2 = int(max(0, min(xs) - pad)), int(min(width, max(xs) + pad))
        y1, y2 = int(max(0, min(ys) - pad)), int(min(height, max(ys) + pad))
    else:
        x1, y1, x2, y2 = 0, 0, width, height
    if x2 - x1 < 40 or y2 - y1 < 40:
        x1, y1, x2, y2 = 0, 0, width, height

    panels = []
    for off in offsets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak + off)
        ok, frame = cap.read()
        if not ok:
            continue
        record = tracking.get(peak + off) or {}
        view = frame.copy()
        for side, colour, thickness in ((other, (150, 150, 150), 1), (fighter, (60, 200, 255), 2)):
            box = ((record.get("fighter_%s" % side) or {}).get("observation") or {}).get("box")
            if box:
                cv2.rectangle(view, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colour, thickness)
        crop = view[y1:y2, x1:x2]
        big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        label = "MOMENT" if off == 0 else ("%+d" % off)
        cv2.rectangle(big, (0, 0), (big.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(big, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 190, 255) if off == 0 else (210, 210, 210), 1, cv2.LINE_AA)
        panels.append(big)
    cap.release()
    if not panels:
        raise SystemExit("no frames could be read for clip %d" % clip_id)
    rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    widest = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, widest - r.shape[1]), (0, 0))) for r in rows]
    return np.vstack(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one clip big.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--step", type=int, default=3)
    args = parser.parse_args()
    cv2.imwrite(args.out, render(args.job, args.id, args.frames, args.step))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
