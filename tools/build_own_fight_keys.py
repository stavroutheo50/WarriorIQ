"""Turn bouts labelled from the owner's own video into scoreboard answer keys (2026-11-23).

The labelling page ("WarriorIQ Strike Labelling", a private claude.ai page)
has a "Your own video" mode: the labeller opens a video from their own
computer - it plays from disk and is never uploaded - marks where each bout
starts and ends, taps the blue and the red fighter, and marks every strike.
Only the marks are saved. They are exported into
dataset/regression/own_fights/<date>.labels.json.

This runs on the analysis PC, where the video is:

    python tools/build_own_fight_keys.py dataset/regression/own_fights/2026-11-24.labels.json
    python tools/build_own_fight_keys.py <labels.json> --videos "C:/Users/User/Videos"
    python tools/scoreboard.py

For each finished bout it finds the video by file name, picks the two people
whose boxes hold the points the labeller tapped at the bout's start, runs the
real analysis over that bout only, and writes, in dataset/public/own_fights/<bout>/
(kept out of Git - pose tracks of identifiable people):

  track.jsonl.gz      the analysis's tracking for the bout
  strikes.json        the person's marks, as tools/benchmark_dense_strikes.py reads them
  strikes.human.json  the same: these are a person's marks, so they count toward the target
  source.json         video name, bout window, ruleset, which points were tapped

Times are seconds from the start of the video file on both sides: the page
marks them that way and the analysis tracks them that way. A bout already
built is skipped unless --rebuild.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "dataset" / "public" / "own_fights"
FAMILIES = {"punch", "kick", "knee"}


def read_documents(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    docs = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for doc in docs:
        doc = doc.get("data", doc) if isinstance(doc, dict) else doc
        if isinstance(doc, dict) and doc.get("kind") == "own_video" and doc.get("bout"):
            out.append(doc)
    return out


def bout_name(doc: dict) -> str:
    stem = Path(str(doc["video"]["name"])).stem
    safe = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:40] or "video"
    return f"own_{safe}_b{int(doc['bout']['number']):02d}"


def pick(docs: list[dict]) -> dict[str, dict]:
    """Per bout: the labeller who watched the most of it."""
    chosen: dict[str, dict] = {}
    for doc in docs:
        name = bout_name(doc)
        if name not in chosen or float(doc.get("watched") or 0) > float(chosen[name].get("watched") or 0):
            chosen[name] = doc
    return chosen


def answer_key(doc: dict) -> dict:
    """The person's marks inside the bout, in benchmark_dense_strikes' layout."""
    start, end = float(doc["bout"]["start"]), float(doc["bout"]["end"])
    strikes = sorted(({"t": round(float(m["t"]), 3), "fighter": m["fighter"], "family": m["family"],
                       "sure": bool(m.get("sure", True))}
                      for m in doc.get("marks") or []
                      if m.get("fighter") in {"A", "B"} and m.get("family") in FAMILIES
                      and start <= float(m["t"]) <= end), key=lambda s: s["t"])
    unreadable = [[max(start, float(a)), min(end, float(b))] for a, b in doc.get("unreadable") or []
                  if float(b) > start and float(a) < end]
    return {"covered": [start, end], "unreadable": unreadable, "strikes": strikes,
            "notes": f"Marked by a person on the labelling page's own-video mode; A is the fighter tapped "
                     f"blue, B the one tapped red, at {start:.2f} s."}


def find_video(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if (root / name).is_file():
            return root / name
    for root in roots:
        if root.is_dir():
            for found in root.rglob(name):
                if found.is_file():
                    return found
    return None


def _containing(boxes: list[list[float]], point: tuple[float, float]) -> int | None:
    """The person whose box holds the point; with several, the smallest; with none, the nearest centre."""
    inside = [i for i, b in enumerate(boxes) if b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]]
    if inside:
        return min(inside, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    if not boxes:
        return None
    return min(range(len(boxes)), key=lambda i: ((boxes[i][0] + boxes[i][2]) / 2 - point[0]) ** 2
               + ((boxes[i][1] + boxes[i][3]) / 2 - point[1]) ** 2)


def fighter_boxes(video: Path, seconds: float, blue: list[float], red: list[float]) -> tuple[list, list]:
    """The two people tapped on the page, as boxes in pixels."""
    import cv2
    from ultralytics import YOLO

    from core.config import SETTINGS

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {video.name} at {seconds:.2f} s")
    height, width = frame.shape[:2]
    result = YOLO(SETTINGS.pose_model_pt).predict(frame, classes=[0], conf=.25, verbose=False)[0]
    boxes = result.boxes.xyxy.tolist()
    a = _containing(boxes, (blue[0] * width, blue[1] * height))
    b = _containing([x if i != a else [0, 0, 0, 0] for i, x in enumerate(boxes)], (red[0] * width, red[1] * height))
    if a is None or b is None or a == b:
        raise RuntimeError(f"could not find two different people at the tapped points at {seconds:.2f} s")
    return boxes[a], boxes[b]


def build(doc: dict, video: Path, folder: Path) -> None:
    from core.analyzer import analyze
    from core.types import AnalysisRequest

    bout = doc["bout"]
    start, end = float(bout["start"]), float(bout["end"])
    a, b = fighter_boxes(video, start, bout["blue"], bout["red"])
    name = folder.name
    analyze(AnalysisRequest(
        video_path=str(video), fighter_a_box=a, fighter_b_box=b, original_name=name,
        fight_type="sparring", ruleset=str(bout.get("ruleset") or "K1"), start_seconds=start,
        selection_seconds=start, end_seconds=end, round_count=1, round_duration_seconds=max(1.0, end - start),
        break_duration_seconds=0, job_id=name, profile_id=1, persist_result=False), lambda _patch: None)
    from core.config import OUTPUTS
    folder.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS / name / "tracking.jsonl", "rb") as raw, gzip.open(folder / "track.jsonl.gz", "wb") as packed:
        shutil.copyfileobj(raw, packed)
    key = answer_key(doc)
    for file in ("strikes.json", "strikes.human.json"):
        (folder / file).write_text(json.dumps(key, indent=1), encoding="utf-8")
    (folder / "source.json").write_text(json.dumps({
        "video": doc["video"]["name"], "bout": {k: bout[k] for k in ("number", "start", "end", "blue", "red") if k in bout},
        "ruleset": str(bout.get("ruleset") or "K1"), "labelled_by": "a person, on the labelling page's own-video mode",
        "selection_boxes": {"A": a, "B": b}}, indent=1), encoding="utf-8")
    print(f"  {name}: {len(key['strikes'])} marked strikes, {end - start:.0f} s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("labels", type=Path, help="the labelling page's marks, exported as JSON")
    parser.add_argument("--videos", type=Path, action="append", default=[],
                        help="a folder to look for the video in (repeatable); the usual folders are searched too")
    parser.add_argument("--rebuild", action="store_true", help="build bouts that were already built")
    parser.add_argument("--include-unfinished", action="store_true",
                        help="also bouts not marked finished (their uncovered part counts as missed strikes)")
    args = parser.parse_args(argv)

    home = Path.home()
    roots = args.videos + [ROOT / "dataset" / "public" / "owner_fights", ROOT / "fights",
                           home / "Videos", home / "Desktop", home / "Downloads"]
    chosen = pick(read_documents(args.labels))
    if not chosen:
        print("No own-video bouts in that file.")
        return 0
    scratch = Path(tempfile.mkdtemp(prefix="own_fights_"))
    # The analysis writes its database and outputs here, never into the app's.
    os.environ.update(WARRIORIQ_DB_PATH=str(scratch / "db.sqlite3"), WARRIORIQ_UPLOADS_DIR=str(scratch / "up"),
                      WARRIORIQ_OUTPUTS_DIR=str(scratch / "out"), WARRIORIQ_HARD_REALTIME="0")
    built = skipped = 0
    for name, doc in sorted(chosen.items()):
        bout = doc["bout"]
        if bout.get("end") is None:
            print(f"  {name}: no end marked yet, skipped")
            skipped += 1
            continue
        if not doc.get("finished") and not args.include_unfinished:
            print(f"  {name}: not marked finished yet, skipped")
            skipped += 1
            continue
        folder = OUT / name
        if (folder / "track.jsonl.gz").exists() and not args.rebuild:
            continue
        video = find_video(str(doc["video"]["name"]), roots)
        if video is None:
            print(f"  {name}: {doc['video']['name']} not found; pass its folder with --videos")
            skipped += 1
            continue
        build(doc, video, folder)
        built += 1
    print(f"Built {built} bout(s), skipped {skipped}. Now: python tools/scoreboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
