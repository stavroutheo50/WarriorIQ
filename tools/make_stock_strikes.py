"""Rebuild the pose tracks for dataset/stock_strikes/, the labelled stock clips.

Each folder there holds strikes.json (every strike, labelled frame by frame)
and source.json (where the clip comes from, its licence, the ruleset and the
moment the two fighters were picked). The pose track the benchmark replays,
track.jsonl.gz, is not committed - pose measurements of identifiable athletes
stay out of this public repository - so this rebuilds it:

    python tools/make_stock_strikes.py                  # every folder
    python tools/make_stock_strikes.py mixkit_40979     # one folder
    python tools/benchmark_dense_strikes.py

It downloads the clip named in source.json, picks the two largest people on
the selection frame as the app's two taps would (left is A, right is B), runs
the real analysis (core.analyzer.analyze) with nothing saved to the database,
and keeps only its tracking.jsonl. Runs in a scratch folder, never the live
data directories.

The labels follow whoever the analysis tracked as A or B at that moment; a
swap is recorded in strikes.json's notes. If a rebuilt track swaps at a
different time, read the notes before trusting the scores.

Your own sparring, from a video on this machine, into dataset/own_sparring/:

    python tools/make_stock_strikes.py --video C:/Users/you/Desktop/sparring.mp4 --name sparring \
        --selection 12.5 --ruleset K1 --gpu

--selection is a moment where both fighters are clearly separated and the
two largest people in frame; check/selection.jpg shows who was picked. The
video is read where it is and never copied into the repository. Afterwards
check/ holds frames with the tracked A (red) and B (blue) boxes drawn, to
spot a swap before labelling. check/ and track.jsonl.gz are not committed.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOCK = ROOT / "dataset" / "stock_strikes"
OWN = ROOT / "dataset" / "own_sparring"
A_COLOUR, B_COLOUR = (0, 0, 255), (255, 128, 0)  # BGR: A red, B blue


def _read_frame(video: Path, seconds: float):
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {video} at {seconds}s")
    return frame


def _draw(frame, boxes: dict, label: str):
    import cv2

    for name, box in boxes.items():
        if box:
            x1, y1, x2, y2 = (int(v) for v in box)
            colour = A_COLOUR if name == "A" else B_COLOUR
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(frame, name, (x1 + 4, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    cv2.putText(frame, label, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def _fighter_boxes(video: Path, seconds: float, preview: Path | None = None) -> list[list[float]]:
    from ultralytics import YOLO

    from core.config import SETTINGS

    frame = _read_frame(video, seconds)
    device = None if SETTINGS.device == "auto" else SETTINGS.device
    result = YOLO(SETTINGS.pose_model_pt).predict(frame, classes=[0], conf=.3, verbose=False, device=device)[0]
    boxes = sorted(result.boxes.xyxy.tolist(), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
    if len(boxes) < 2:
        raise RuntimeError(f"fewer than two people at {seconds}s in {video}")
    a, b = sorted(boxes, key=lambda b: b[0])
    if preview is not None:
        import cv2

        preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview), _draw(frame, {"A": a, "B": b}, f"selection {seconds:.2f}s"))
    return [a, b]


def write_check_frames(video: Path, track: Path, out: Path, count: int) -> None:
    """Draw the tracked A and B boxes on `count` evenly spaced frames, so a
    swap shows up as the red and blue boxes trading people."""
    import cv2

    with gzip.open(track, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if not records or count <= 0:
        return
    out.mkdir(parents=True, exist_ok=True)
    step = max(1, len(records) // count)
    cap = cv2.VideoCapture(str(video))
    for record in records[::step][:count]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(record["source_frame"]))
        ok, frame = cap.read()
        if not ok:
            continue
        boxes = {f: ((record.get(f"fighter_{f}") or {}).get("observation") or {}).get("box") for f in "AB"}
        seconds = float(record["time_seconds"])
        cv2.imwrite(str(out / f"t{seconds:07.2f}.jpg"), _draw(frame, boxes, f"{seconds:.2f}s"))
    cap.release()


def rebuild(folder: Path, scratch: Path, video: Path | None = None, check_frames: int = 0) -> None:
    from core.analyzer import analyze
    from core.types import AnalysisRequest

    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    if video is None:
        video = scratch / f"{folder.name}.mp4"
        if not video.exists():
            urllib.request.urlretrieve(source["download_url"], video)
    selection = float(source.get("selection_seconds", 0.0))
    preview = folder / "check" / "selection.jpg" if check_frames else None
    a, b = _fighter_boxes(video, selection, preview)
    analyze(AnalysisRequest(
        video_path=str(video), fighter_a_box=a, fighter_b_box=b, original_name=folder.name,
        fight_type="sparring", ruleset=source["ruleset"], start_seconds=0.0, selection_seconds=selection,
        round_count=1, round_duration_seconds=600, break_duration_seconds=0,
        job_id=folder.name, profile_id=1, persist_result=False), lambda _patch: None)
    tracking = scratch / "out" / folder.name / "tracking.jsonl"
    with open(tracking, "rb") as raw, gzip.open(folder / "track.jsonl.gz", "wb") as packed:
        shutil.copyfileobj(raw, packed)
    print(folder / "track.jsonl.gz")
    if check_frames:
        write_check_frames(video, folder / "track.jsonl.gz", folder / "check", check_frames)
        print(folder / "check")


def _own_source(folder: Path, args) -> None:
    """Create or update source.json for an own-sparring folder. Facts only
    the person who filmed it knows (date, consent) are left for them to
    confirm; existing values are kept."""
    path = folder / "source.json"
    source = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "title": args.name,
        "filmed_by": None,
        "filmed_on": None,
        "consent": None,
        "labelled_by": None,
    }
    source["video"] = Path(args.video).name  # the file name only: no local paths in Git
    source["ruleset"] = args.ruleset or source.get("ruleset") or "K1"
    if args.selection is not None or "selection_seconds" not in source:
        source["selection_seconds"] = float(args.selection or 0.0)
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="*", help="stock_strikes folders to rebuild (default: all)")
    parser.add_argument("--video", help="a video on this machine: build dataset/own_sparring/<name>/ from it")
    parser.add_argument("--name", help="with --video: the own_sparring folder name")
    parser.add_argument("--selection", type=float, help="seconds into the video to pick the two fighters")
    parser.add_argument("--ruleset", help="BOXING, K1, LOW_KICK, MUAY_THAI, MMA, ... (default K1)")
    parser.add_argument("--gpu", action="store_true", help="run on CUDA and stop if it is not available")
    parser.add_argument("--check-frames", type=int, default=None,
                        help="frames with the tracked boxes to draw into check/ (default 24 with --video, else 0)")
    args = parser.parse_args(argv[1:])

    scratch = Path(tempfile.mkdtemp(prefix="stock_strikes_"))
    # The analysis writes its database and outputs here, not into the app's.
    os.environ.update(WARRIORIQ_DB_PATH=str(scratch / "db.sqlite3"),
                      WARRIORIQ_UPLOADS_DIR=str(scratch / "up"), WARRIORIQ_OUTPUTS_DIR=str(scratch / "out"))
    if args.gpu:
        import torch

        if not torch.cuda.is_available():
            print("--gpu: CUDA is not available to this Python (check the torch build and driver)", file=sys.stderr)
            return 2
        os.environ.setdefault("WARRIORIQ_DEVICE", "cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    sys.path.insert(0, str(ROOT))

    if args.video:
        video = Path(args.video)
        if not video.is_file():
            print(f"no such video: {video}", file=sys.stderr)
            return 2
        if not args.name:
            parser.error("--video needs --name")
        folder = OWN / args.name
        _own_source(folder, args)
        rebuild(folder, scratch, video=video, check_frames=24 if args.check_frames is None else args.check_frames)
        return 0

    names = set(args.names)
    for folder in sorted(p for p in STOCK.iterdir() if (p / "source.json").exists()):
        if not names or folder.name in names:
            rebuild(folder, scratch, check_frames=args.check_frames or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
