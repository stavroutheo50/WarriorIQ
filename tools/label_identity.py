"""Make a sheet a person can label, so identity can finally be scored.

Coverage says how often a box was held. It cannot say whether the box was on a
fighter, and this project has now been fooled by that three times: 98% coverage
of a seated spectator, 0.728 of a seated coach, and 0.994 on HD footage where
the boxes were on a cornerman and a spectator's head. Every automatic substitute
tried has failed - the manager's own spread reading is blind to tracks with no
history, which is exactly what a freshly latched bystander is, and box
displacement flags fighters who circle or clinch.

Telling "held a fighter" from "held someone who looks like one" needs someone to
say which is which. That is all this asks for, and it asks in the cheapest form:
the candidates are drawn and numbered, and the answer is two numbers per frame.

    tools/label_identity.py --fight 5736 --frames 40
    # look at outputs/label_5736/*.png, fill in tools/labels_5736.json
    tools/measure_identity.py --fight 5736 --labels tools/labels_5736.json

Frames are sampled evenly along the same stride grid the analysis uses, so each
labelled frame is one the pipeline actually evaluated and the two can be
compared without interpolation.
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

SEEDS_PATH = Path(__file__).resolve().parent / "verified_seeds.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fight", required=True)
    parser.add_argument("--frames", type=int, default=40,
                        help="how many frames to label; 40 is about an hour's honest work")
    parser.add_argument("--stride", type=int, default=3,
                        help="must match the stride the measurement will use")
    # One frame per image, not four. Labelling a 4-up sheet was measured
    # producing wrong labels: on fight 5736 at least four frames were marked
    # "this fighter is not detected" when the pipeline was holding the fighter
    # correctly and a thin candidate box was simply too small to see. Those
    # errors then scored against the pipeline as "held a non-fighter", which is
    # the exact accusation the labelling exists to test. A label that is harder
    # to read than the thing it judges is worse than no label.
    parser.add_argument("--per-sheet", type=int, default=1)
    args = parser.parse_args()

    from core import preflight
    from core.config import SETTINGS
    from ultralytics import YOLO

    fights = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))["fights"]
    if args.fight not in fights:
        raise SystemExit("unknown fight %r - have %s" % (args.fight, ", ".join(sorted(fights))))
    fight = fights[args.fight]

    video = str(PROJECT_ROOT / fight["video"])
    capture = cv2.VideoCapture(video)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = capture.read()
    if not ok:
        raise SystemExit("could not read %s" % video)

    # The frames the analysis actually evaluated, taken from the analysis rather
    # than guessed. Assuming seed_frame + k*stride looks right and is not: the
    # analyser gates each frame on wants_inference and widens the stride outside
    # a selected round, so its grid is a subset. Labelling the guessed grid
    # scored 2 of 8 answers, because six of them were frames nothing ran on.
    capture.release()
    print("running one analysis to learn which frames it evaluates...", flush=True)
    from tools.measure_identity import trace_run

    trace, _ = trace_run(fight, args.stride)
    grid = [t["frame"] for t in trace]
    if not grid:
        raise SystemExit("the analysis evaluated no frames")
    picks = [grid[i] for i in np.linspace(0, len(grid) - 1, min(args.frames, len(grid))).astype(int)]
    capture = cv2.VideoCapture(video)

    imgsz = preflight.probe(video, YOLO(SETTINGS.pose_model_engine)).recommended_inference_size
    model = YOLO(SETTINGS.pose_model_engine)

    out_dir = PROJECT_ROOT / "outputs" / ("label_%s" % args.fight)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, tiles = [], []
    for frame_no in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        result = model.predict(frame, imgsz=imgsz, conf=SETTINGS.detection_conf,
                               verbose=False, device=0)[0]
        boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().numpy().tolist()
        entries.append({
            "frame": int(frame_no),
            "seconds": round(frame_no / max(1.0, fps), 2),
            "candidates": [[round(float(v), 1) for v in b] for b in boxes],
            # Fill these in: the candidate index for each fighter, or null when
            # that fighter is not visible in this frame. "null" is a real
            # answer and an important one - it is how the pipeline is credited
            # for correctly holding nothing.
            "fighter_a": "FILL",
            "fighter_b": "FILL",
        })
        scale = max(2, int(round(1900.0 / max(1, frame.shape[1]))))
        big = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            cv2.rectangle(big, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(big, str(index), (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(big, "f%d  %.1fs" % (frame_no, frame_no / max(1.0, fps)), (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(big)
    capture.release()

    width = min(t.shape[1] for t in tiles)
    height = min(t.shape[0] for t in tiles)
    tiles = [t[:height, :width] for t in tiles]
    for start in range(0, len(tiles), args.per_sheet):
        group = tiles[start:start + args.per_sheet]
        while len(group) % 2:
            group.append(np.zeros_like(group[0]))
        rows = [np.hstack(group[i:i + 2]) for i in range(0, len(group), 2)]
        cv2.imwrite(str(out_dir / ("sheet_%02d.png" % (start // args.per_sheet))),
                    np.vstack(rows))

    labels_path = Path(__file__).resolve().parent / ("labels_%s.json" % args.fight)
    payload = {
        "_how": [
            "For each frame below, replace FILL with the candidate number that is",
            "fighter A and the one that is fighter B, reading them off the matching",
            "sheet in outputs/label_%s/. Use null when that fighter is not visible" % args.fight,
            "or is not detected - that is a real answer, and it is how the pipeline",
            "gets credit for correctly holding nothing.",
            "",
            "A and B are whoever the seed boxes in verified_seeds.json are:",
            fight.get("note", ""),
        ],
        "fight": args.fight,
        "stride": args.stride,
        "frames": entries,
    }
    labels_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("%d frames sampled along the stride-%d grid" % (len(entries), args.stride))
    print("sheets:  %s" % out_dir)
    print("answers: %s" % labels_path)
    print("\nFill in fighter_a and fighter_b for each frame, then run:")
    print("  tools/measure_identity.py --fight %s --labels %s" % (
        args.fight, labels_path.relative_to(PROJECT_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
