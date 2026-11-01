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


def _fighter_boxes(video: Path, seconds: float) -> list[list[float]]:
    import cv2
    from ultralytics import YOLO

    from core.config import SETTINGS

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {video} at {seconds}s")
    result = YOLO(SETTINGS.pose_model_pt).predict(frame, classes=[0], conf=.3, verbose=False)[0]
    boxes = sorted(result.boxes.xyxy.tolist(), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:2]
    if len(boxes) < 2:
        raise RuntimeError(f"fewer than two people at {seconds}s in {video}")
    return sorted(boxes, key=lambda b: b[0])


def rebuild(folder: Path, scratch: Path) -> None:
    from core.analyzer import analyze
    from core.types import AnalysisRequest

    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    video = scratch / f"{folder.name}.mp4"
    if not video.exists():
        urllib.request.urlretrieve(source["download_url"], video)
    selection = float(source.get("selection_seconds", 0.0))
    a, b = _fighter_boxes(video, selection)
    analyze(AnalysisRequest(
        video_path=str(video), fighter_a_box=a, fighter_b_box=b, original_name=folder.name,
        fight_type="sparring", ruleset=source["ruleset"], start_seconds=0.0, selection_seconds=selection,
        round_count=1, round_duration_seconds=600, break_duration_seconds=0,
        job_id=folder.name, profile_id=1, persist_result=False), lambda _patch: None)
    tracking = scratch / "out" / folder.name / "tracking.jsonl"
    with open(tracking, "rb") as raw, gzip.open(folder / "track.jsonl.gz", "wb") as packed:
        shutil.copyfileobj(raw, packed)
    print(folder / "track.jsonl.gz")


def main(argv: list[str]) -> int:
    scratch = Path(tempfile.mkdtemp(prefix="stock_strikes_"))
    # The analysis writes its database and outputs here, not into the app's.
    os.environ.update(WARRIORIQ_DB_PATH=str(scratch / "db.sqlite3"),
                      WARRIORIQ_UPLOADS_DIR=str(scratch / "up"), WARRIORIQ_OUTPUTS_DIR=str(scratch / "out"))
    sys.path.insert(0, str(ROOT))
    names = set(argv[1:])
    for folder in sorted(p for p in STOCK.iterdir() if (p / "source.json").exists()):
        if not names or folder.name in names:
            rebuild(folder, scratch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
