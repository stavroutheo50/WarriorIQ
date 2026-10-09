"""Analyse one UFC fight with the candidate strike model, for the exam's count check.

The exam (tools/strike_exam.py, core/strike_exam.py) passes a sport only when
WarriorIQ's per-round strike counts on whole fights are close to the official
UFC statistics. That needs UFC fights analysed by **the candidate model**,
listed in dataset/exam/ufc_bouts.json. This does one fight per run, on this
machine only: the live worker and its model are not touched.

Three steps, from the repository root:

1. Find the bout's exact name in the official statistics:

       .venv\\Scripts\\python.exe tools\\exam_ufc_fight.py --find "Adesanya"

2. Pick a moment where both fighters are fully in view (seconds into the
   video). This saves a picture with the people it found numbered:

       .venv\\Scripts\\python.exe tools\\exam_ufc_fight.py --video videos\\fight1.mp4 --at 30

3. Analyse, saying which number is which fighter (names as step 1 printed):

       .venv\\Scripts\\python.exe tools\\exam_ufc_fight.py --video videos\\fight1.mp4 --at 30 ^
           --a 1 --fighter-a "Israel Adesanya" --b 2 --fighter-b "Alex Pereira" ^
           --bout "Israel Adesanya vs. Alex Pereira" --event "UFC 287: Pereira vs. Adesanya 2"

The video should be the whole fight as broadcast, rounds and breaks included:
WarriorIQ finds the rounds from the breaks, and the exam compares them round by
round with the official lines. A round the analysis misses or misnumbers
counts against the model, never for it.

After three fights, run the pipeline's exam again (tools/strike_pipeline.py
--skip-fetch, or tools/strike_exam.py with --ufc-manifest).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATE = ROOT / "models" / "warrioriq_temporal_candidate.pt"
MANIFEST = ROOT / "dataset" / "exam" / "ufc_bouts.json"
OFFICIAL = ROOT / "dataset" / "public" / "ufc_stats" / "ufc_fight_stats.csv"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "fight"


def update_manifest(path: Path, entry: dict) -> list[dict]:
    """Add ``entry``, replacing any earlier entry for the same result folder."""
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    entries = [e for e in entries if e.get("result") != entry["result"]] + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries


def find_bouts(rows: list[dict], text: str) -> list[tuple[str, str, list[str]]]:
    """(event, bout, fighters) for every bout whose name contains ``text``."""
    wanted = text.lower()
    found: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if wanted in str(row.get("BOUT", "")).lower():
            found.setdefault((row.get("EVENT", ""), row.get("BOUT", "")), set()).add(row.get("FIGHTER", ""))
    return [(event, bout, sorted(fighters)) for (event, bout), fighters in sorted(found.items())]


def check_official(rows: list[dict], bout: str, fighters: tuple[str, str], event: str | None) -> None:
    """Refuse before a long analysis when the names do not match the official lines."""
    from core.official_stats import official_rounds

    for name in fighters:
        try:
            found = official_rounds(rows, bout=bout, fighter=name, event=event)
        except LookupError as problem:  # a misspelt name, or a rematch without --event
            raise SystemExit(f"{problem}. Use --find to copy the exact names.") from None
        if not found:
            raise SystemExit(f'No official rounds for "{name}" in "{bout}". Use --find to copy the exact names.')


def frame_at(video: Path, seconds: float):
    import cv2

    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Could not read {video} at {seconds} s.")
    return frame


def people_at(video: Path, seconds: float) -> list[list[float]]:
    from core.person_detect import detect_people

    people = detect_people(frame_at(video, seconds))
    if people is None:
        raise SystemExit("The person detector is not available on this machine.")
    # Left to right, so the numbers read in the order you see them.
    return sorted((p["box"] for p in people), key=lambda b: b[0] + b[2])


def save_pick(video: Path, seconds: float, boxes: list[list[float]], out: Path) -> None:
    import cv2

    frame = frame_at(video, seconds)
    for number, (x1, y1, x2, y2) in enumerate(boxes, 1):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 3)
        cv2.putText(frame, str(number), (int(x1) + 6, int(y1) + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 255), 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--find", help="list official bouts whose name contains this text")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--at", type=float, default=30.0, help="seconds into the video where both fighters show")
    parser.add_argument("--a", type=int, help="number of fighter A in the saved picture")
    parser.add_argument("--b", type=int, help="number of fighter B in the saved picture")
    parser.add_argument("--fighter-a")
    parser.add_argument("--fighter-b")
    parser.add_argument("--bout")
    parser.add_argument("--event")
    parser.add_argument("--checkpoint", type=Path, default=CANDIDATE)
    parser.add_argument("--official", type=Path, default=OFFICIAL)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    from core.official_stats import load_official

    if args.find:
        rows = load_official(args.official)
        for event, bout, fighters in find_bouts(rows, args.find)[:40]:
            print(f'{event}\n    --bout "{bout}"  fighters: {" | ".join(fighters)}')
        return 0

    if not args.video or not args.video.exists():
        raise SystemExit("--video must name a video file on this machine.")
    if args.a is None or args.b is None:
        boxes = people_at(args.video, args.at)
        picture = ROOT / "dataset" / "exam" / f"ufc_pick_{slug(args.video.stem)}.png"
        save_pick(args.video, args.at, boxes, picture)
        print(f"Found {len(boxes)} people at {args.at:g} s. Open {picture.relative_to(ROOT)} and run again with "
              "--a <number> --b <number> and the fighters' names.")
        return 0
    if not (args.fighter_a and args.fighter_b and args.bout):
        raise SystemExit("Give --fighter-a, --fighter-b and --bout as --find printed them.")
    if not args.checkpoint.exists():
        raise SystemExit(f"No candidate model at {args.checkpoint}. Run tools/strike_pipeline.py first.")
    check_official(load_official(args.official), args.bout, (args.fighter_a, args.fighter_b), args.event)

    boxes = people_at(args.video, args.at)
    if not (1 <= args.a <= len(boxes) and 1 <= args.b <= len(boxes)) or args.a == args.b:
        raise SystemExit(f"--a and --b must be two different numbers from 1 to {len(boxes)}.")

    # The candidate, for this process only: config reads it at import.
    os.environ["WARRIORIQ_TEMPORAL_MODEL"] = str(args.checkpoint)
    import cv2

    from core import analyzer
    from core.temporal_model import checkpoint_sha256
    from core.types import AnalysisRequest

    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / max(1.0, fps)
    capture.release()
    job = f"ufcexam_{slug(args.bout)}"
    out = ROOT / "outputs" / job
    report = analyzer.analyze(AnalysisRequest(
        video_path=str(args.video), fighter_a_box=boxes[args.a - 1], fighter_b_box=boxes[args.b - 1],
        ruleset="MMA", fight_type="competition", round_count=1, start_seconds=0.0,
        round_duration_seconds=max(1.0, duration), selection_seconds=args.at,
        job_id=job, profile_id=1, persist_result=False, output_dir=str(out)))

    made_by = (report.get("classifier") or {}).get("temporal_checkpoint_sha256")
    if made_by != checkpoint_sha256(args.checkpoint):
        raise SystemExit("The analysis did not run on the candidate model from start to finish, so the exam "
                         "would refuse it. Check the log above for a model that failed to load.")
    entries = update_manifest(args.manifest, {
        "result": str(out.relative_to(ROOT)), "bout": args.bout, "fighter_a": args.fighter_a,
        "fighter_b": args.fighter_b, **({"event": args.event} if args.event else {})})
    rounds = len(report.get("rounds") or [])
    print(f"Done: {rounds} round(s) found. {len(entries)} fight(s) in {args.manifest.relative_to(ROOT)}; "
          "the exam needs 3 or more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
