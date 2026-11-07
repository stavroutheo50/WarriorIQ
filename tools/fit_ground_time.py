"""Fit and measure the MMA ground-time head (core/ground_time.py).

    python tools/fit_ground_time.py            # leave-one-bout-out scores only
    python tools/fit_ground_time.py --write    # also store models/ground_time_head.npz

Labels: dataset/regression/grappling_pankration/<bout>.json, one per second.
Videos: dataset/public/grappling_clips/<bout>.mp4 (archive.org, CC BY 2.0;
see that folder's README). Every person the pose model finds is read at
core.ground_time.SAMPLE_FPS and cached beside the video as
<bout>.poses.jsonl, and the image model's description of the same frames as
<bout>.pictures.npz, so refitting is quick and an interrupted run resumes.

Each bout is scored by a head fitted on the other bouts: the number to read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import ground_time  # noqa: E402

LABELS = PROJECT_ROOT / "dataset" / "regression" / "grappling_pankration"
CLIPS = PROJECT_ROOT / "dataset" / "public" / "grappling_clips"
# Tried 10, 100 and 1000 leave-one-bout-out (2026-11-03); 10 was best on every bout.
L2 = 10.0


def read_poses(bout: str) -> Path:
    import cv2
    from ultralytics import YOLO

    from core.config import SETTINGS

    out = CLIPS / f"{bout}.poses.jsonl"
    done = set()
    if out.exists():
        done = {round(json.loads(line)["t"], 2) for line in out.read_text().splitlines() if line}
    capture = cv2.VideoCapture(str(CLIPS / f"{bout}.mp4"))
    fps, frames = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    step = max(1, int(round(fps / ground_time.SAMPLE_FPS)))
    model = None
    with out.open("a") as handle:
        for index in range(0, frames, step):
            t = round(index / fps, 2)
            if t in done:
                continue
            model = model or YOLO(SETTINGS.pose_model_pt)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, imgsz=640, conf=0.25, classes=[0], verbose=False)[0]
            people = []
            if result.boxes is not None and len(result.boxes):
                for box, kp, kc, conf in zip(result.boxes.xyxy.tolist(), result.keypoints.xy.tolist(),
                                             result.keypoints.conf.tolist(), result.boxes.conf.tolist()):
                    people.append({"box": [round(v, 1) for v in box], "conf": round(conf, 3),
                                   "kp": [[round(x, 1), round(y, 1)] for x, y in kp], "kc": [round(v, 2) for v in kc]})
            handle.write(json.dumps({"t": t, "W": width, "H": height, "people": people}) + "\n")
            handle.flush()
    return out


def read_pictures(bout: str, times: list[float]) -> dict[float, np.ndarray]:
    import cv2

    out = CLIPS / f"{bout}.pictures.npz"
    if out.exists():
        stored = np.load(out)
        if len(stored["t"]) == len(times):
            return {round(float(t), 2): f for t, f in zip(stored["t"], stored["f"])}
    model = ground_time.pictures()
    capture = cv2.VideoCapture(str(CLIPS / f"{bout}.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    vectors = []
    for start in range(0, len(times), ground_time.PICTURE_BATCH):
        frames = []
        for t in times[start:start + ground_time.PICTURE_BATCH]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = capture.read()
            # The analyzer shrinks the frame first; do the same, so both see one picture.
            frames.append(cv2.resize(frame, ground_time.PICTURE_SIZE))
        vectors.extend(model.features(frames))
    np.savez(out, t=np.array(times), f=np.array(vectors, dtype=np.float32))
    return {round(t, 2): f for t, f in zip(times, vectors)}


def bout_rows(bout: str):
    labels = json.loads((LABELS / f"{bout}.json").read_text(encoding="utf-8"))["seconds"]
    records = [json.loads(line) for line in read_poses(bout).read_text().splitlines() if line]
    described = read_pictures(bout, [record["t"] for record in records])
    frames: dict[int, list] = {}
    pictures: dict[int, list] = {}
    for record in records:
        frames.setdefault(int(record["t"]), []).append(
            ground_time.frame_features(record["people"], record["W"], record["H"]))
        pictures.setdefault(int(record["t"]), []).append(described[round(record["t"], 2)])
    X, y = [], []
    for second, code in labels.items():
        second = int(second)
        if code == "X" or len(frames.get(second, [])) < ground_time.MIN_SAMPLES_PER_SECOND:
            continue
        X.append(ground_time.second_features(frames[second], pictures[second]))
        y.append(1.0 if code.startswith("G") else 0.0)
    return np.array(X), np.array(y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    bouts = sorted(p.stem for p in LABELS.glob("*.json") if (CLIPS / f"{p.stem}.mp4").exists())
    data = {bout: bout_rows(bout) for bout in bouts}
    total = {"labelled": 0, "measured": 0, "right": 0, "seconds": 0}
    for bout in bouts:
        X, y = data[bout]
        rest_X = np.concatenate([data[o][0] for o in bouts if o != bout])
        rest_y = np.concatenate([data[o][1] for o in bouts if o != bout])
        said = ground_time.predict(ground_time.fit(rest_X, rest_y, l2=L2), X) >= .5
        truth = y == 1
        print(f"{bout:8} {len(y):4} seconds | ground labelled {int(truth.sum()):4}s, measured {int(said.sum()):4}s | "
              f"seconds right {int((said == truth).sum())}/{len(y)} | false ground {int((said & ~truth).sum())}, "
              f"missed ground {int((~said & truth).sum())}")
        total["labelled"] += int(truth.sum()); total["measured"] += int(said.sum())
        total["right"] += int((said == truth).sum()); total["seconds"] += len(y)
    print(f"TOTAL    ground labelled {total['labelled']}s, measured {total['measured']}s, "
          f"seconds right {total['right']}/{total['seconds']}")
    if args.write:
        X = np.concatenate([data[b][0] for b in bouts]); y = np.concatenate([data[b][1] for b in bouts])
        ground_time.HEAD_PATH.parent.mkdir(exist_ok=True)
        np.savez(ground_time.HEAD_PATH, **ground_time.fit(X, y, l2=L2))
        print("head written to", ground_time.HEAD_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
