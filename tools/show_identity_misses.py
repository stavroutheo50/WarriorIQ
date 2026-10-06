"""Draw every "other person" moment from the last identity benchmark on one sheet.

    python tools/show_identity_misses.py            # after tools/run_identity_benchmark.py

The benchmark says how often a fighter's box was on "somebody else" but not
who that was - the referee, a coach, a spectator, a fighter from the next mat.
The fix depends on which, so this draws each such moment from the run already
on disk (dataset/public/identity_runs/<arm>/<clip>/tracking.jsonl) beside its
video:

  green  fighter A as a person marked them
  blue   fighter B as a person marked them
  red    the box the analysis followed instead, labelled with whose it was meant
         to be and the referee filter's probability for it (core/referee.py)

and writes one picture, logs/identity-misses.png, plus a line per moment.
Nothing is analysed again and nothing outside logs/ is written.
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

from core.identity import box_iou  # noqa: E402
from tools.identity_benchmark import MATCH_IOU, _box, _nearest  # noqa: E402
from tools.run_identity_benchmark import TRUTH, WORK  # noqa: E402

TILE_WIDTH = 480
COLUMNS = 3
GREEN, BLUE, RED = (60, 200, 60), (230, 140, 40), (40, 40, 230)


def misses(records: list[dict], truth: dict) -> list[dict]:
    """The marked standing frames where a fighter's box was on somebody else."""
    found = []
    for frame in truth["frames"]:
        if frame["phase"] != "standing":
            continue
        record = _nearest(records, int(frame["source_frame"]))
        if record is None:
            continue
        for fighter, other in (("A", "B"), ("B", "A")):
            if frame.get(fighter) is None:
                continue
            box = _box(record, fighter)
            if box is None or box_iou(box, frame[fighter]) >= MATCH_IOU:
                continue
            if frame.get(other) is not None and box_iou(box, frame[other]) >= MATCH_IOU:
                continue                                            # a swap, not another person
            found.append({"fighter": fighter, "followed": box, "frame": frame})
    return found


def _draw_box(image, box, colour, label=None):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
    if label:
        cv2.putText(image, label, (x1 + 3, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


def _tile(video: Path, miss: dict, clip: str) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(miss["frame"]["source_frame"]))
    ok, image = capture.read()
    capture.release()
    if not ok:
        return None
    try:
        from core.referee import referee_probability

        probability = referee_probability(image, miss["followed"])
    except Exception:                                               # noqa: BLE001 - the picture matters more
        probability = None
    miss["referee_prob"] = probability
    frame = miss["frame"]
    if frame.get("A") is not None:
        _draw_box(image, frame["A"], GREEN, "A")
    if frame.get("B") is not None:
        _draw_box(image, frame["B"], BLUE, "B")
    referee = "?" if probability is None else f"{probability:.2f}"
    _draw_box(image, miss["followed"], RED, f"{miss['fighter']} followed (ref {referee})")
    scale = TILE_WIDTH / image.shape[1]
    image = cv2.resize(image, (TILE_WIDTH, int(image.shape[0] * scale)))
    caption = np.zeros((28, TILE_WIDTH, 3), np.uint8)
    cv2.putText(caption, f"{clip}  {float(frame['time_seconds']):.0f}s  fighter {miss['fighter']}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([caption, image])


def sheet(tiles: list[np.ndarray]) -> np.ndarray:
    height = max(tile.shape[0] for tile in tiles)
    padded = [np.vstack([t, np.zeros((height - t.shape[0], TILE_WIDTH, 3), np.uint8)]) for t in tiles]
    while len(padded) % COLUMNS:
        padded.append(np.zeros((height, TILE_WIDTH, 3), np.uint8))
    rows = [np.hstack(padded[i:i + COLUMNS]) for i in range(0, len(padded), COLUMNS)]
    return np.vstack(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arm", default="off", help="which benchmark run to read (default: off)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "logs" / "identity-misses.png"))
    args = parser.parse_args(argv)
    tiles = []
    for path in TRUTH:
        truth = json.loads(path.read_text(encoding="utf-8"))
        tracking = WORK / "identity_runs" / args.arm / path.stem / "tracking.jsonl"
        video = WORK / "identity_clips" / (truth.get("file") or f"{truth['archive_item']}_512kb.mp4")
        if not tracking.exists() or not video.exists():
            print(f"{path.stem}: no benchmark run on disk - run tools/run_identity_benchmark.py first")
            continue
        records = [json.loads(line) for line in tracking.read_text(encoding="utf-8").splitlines() if line.strip()]
        for miss in misses(records, truth):
            tile = _tile(video, miss, path.stem)
            if tile is None:
                continue
            tiles.append(tile)
            referee = "?" if miss["referee_prob"] is None else f"{miss['referee_prob']:.2f}"
            print(f"{path.stem:5} {float(miss['frame']['time_seconds']):6.0f}s  fighter {miss['fighter']}  "
                  f"followed box {[round(v) for v in miss['followed']]}  referee probability {referee}")
    if not tiles:
        print("No 'other person' moments found.")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet(tiles))
    print(f"\n{len(tiles)} moments drawn to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
