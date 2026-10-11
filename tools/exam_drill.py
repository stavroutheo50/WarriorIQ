"""A pad drill with a known answer: does WarriorIQ count the strikes that were called?

Every strike model so far learned from other people's footage and failed on
ours (dataset/EXTERNAL_SOURCES.md). What was missing is footage like ours with
a known answer. A called drill gives one without anybody labelling anything:
the striker throws one strike on each beep, so **the plan is the answer key**.

## Filming it

1. Open tools/drill_caller.html in a browser on a laptop, volume up.
2. Partner holds pads. Film both people from the side, whole bodies in view,
   on a phone close enough to hear the laptop.
3. Start the phone, then press Start on the page. Stop filming after "Finished".

The page plays three high beeps, then blocks of strikes - one strike on every
low beep - with rest blocks between them, where both people move and keep
their guard up but nobody strikes. About seven minutes, 84 strikes.

## Checking it

    .venv\\Scripts\\python.exe tools\\exam_drill.py --video videos\\drill1.mp4 --at 10
    .venv\\Scripts\\python.exe tools\\exam_drill.py --video videos\\drill1.mp4 --at 10 --a 1 --b 2

The first saves a picture with the people numbered; the second names the
striker (A) and the pad holder (B), finds the three start beeps in the
video's sound, analyses the video and compares.

## How it is scored

* A called strike is **found** when the striker has a detected strike that
  peaks from 0.2 s before its beep to 1.5 s after it, one detection per beep.
* **Precision** is found strikes over every strike detected for the striker
  between the start beeps and "Finished" - a detection in a rest block, or a
  second one on a beep, counts against it.
* **Recall** is found strikes over called strikes; the family (punch, kick,
  knee) of each found strike is checked against the call.
* The pad holder threw nothing, so every detection for them is wrong too, and
  is reported separately.

One drill is one person, one camera and one day. It is a check on footage
like ours, not the exam's verdict on a sport.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

PAGE = ROOT / "tools" / "drill_caller.html"
RESULTS = ROOT / "dataset" / "exam"

START_HZ, STRIKE_HZ = 1800.0, 1000.0
START_BEEPS = (0.0, 0.6, 1.2)          # seconds after the first start beep
LEAD_IN = 4.0                          # from the first start beep to the first block
ANNOUNCE = 3.0                         # a block is named this long before its first beep
INTERVAL = 2.5                         # between strike beeps
BEFORE, AFTER = 0.2, 1.5               # a detection this close to a beep answers it
BAR_PRECISION, BAR_STRIKES = 0.80, 50  # core/strike_exam.py BAR

REST_SAY = "Move. Guard up. No strikes."
PLAN = [
    {"rest": 30, "say": "Move around. Guard up. No strikes."},
    {"technique": "jab", "family": "punch", "count": 16, "say": "Jabs. One on every beep."},
    {"rest": 20},
    {"technique": "cross", "family": "punch", "count": 16, "say": "Crosses. One on every beep."},
    {"rest": 20},
    {"technique": "hook", "family": "punch", "count": 12, "say": "Hooks. Either hand. One on every beep."},
    {"rest": 20},
    {"technique": "uppercut", "family": "punch", "count": 12, "say": "Uppercuts. Either hand. One on every beep."},
    {"rest": 20},
    {"technique": "round_kick", "family": "kick", "count": 12, "say": "Round kicks. Either leg. One on every beep."},
    {"rest": 20},
    {"technique": "push_kick", "family": "kick", "count": 8, "say": "Teeps. Push kicks. One on every beep."},
    {"rest": 20},
    {"technique": "knee", "family": "knee", "count": 8, "say": "Knees. One on every beep."},
    {"rest": 30, "say": "Last rest. Move. Guard up. No strikes."},
]


def schedule(plan=PLAN) -> dict:
    """Every beep and block, in seconds after the first start beep."""
    t = LEAD_IN
    blocks, strikes = [], []
    for number, block in enumerate(plan):
        if "rest" in block:
            blocks.append({"kind": "rest", "start": t, "end": t + block["rest"], "say": block.get("say", REST_SAY)})
            t += block["rest"]
            continue
        beeps = [t + ANNOUNCE + i * INTERVAL for i in range(block["count"])]
        end = beeps[-1] + INTERVAL
        blocks.append({"kind": "strike", "start": t, "end": end, "say": block["say"],
                       "technique": block["technique"], "family": block["family"]})
        strikes += [{"time": round(b, 3), "block": number, "technique": block["technique"],
                     "family": block["family"]} for b in beeps]
        t = end
    return {"start_beeps": list(START_BEEPS), "start_hz": START_HZ, "strike_hz": STRIKE_HZ,
            "blocks": [{**b, "start": round(b["start"], 3), "end": round(b["end"], 3)} for b in blocks],
            "strikes": strikes, "finished": round(t, 3)}


def page_schedule(path: Path = PAGE) -> dict:
    """The schedule the caller page actually plays."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="schedule" type="application/json">(.*?)</script>', text, re.S)
    if not match:
        raise SystemExit(f"{path} has no schedule.")
    return json.loads(match.group(1))


# --- finding the drill in the sound ---------------------------------------------------------

RATE, FRAME = 8000, 0.02


def read_audio(video: Path) -> np.ndarray:
    """Mono 8 kHz samples of the video's sound, via ffmpeg."""
    from core.video import _ffmpeg_exe

    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        raise SystemExit("ffmpeg is not available to read the sound. Give --start <seconds> "
                         "(when the first high beep is heard) instead.")
    result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1",
                             "-ar", str(RATE), "-f", "s16le", "-"], capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        raise SystemExit("Could not read the video's sound. Give --start <seconds> instead.")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def tone_share(samples: np.ndarray, hz: float) -> np.ndarray:
    """Per 20 ms frame, the share of its energy at ``hz`` (0 to about 1)."""
    n = int(RATE * FRAME)
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    t = np.arange(n) / RATE
    at = np.abs(frames @ np.exp(-2j * np.pi * hz * t)) ** 2 * 2.0 / n
    total = (frames ** 2).sum(axis=1) + 1e-9
    return np.clip(at / total, 0.0, 1.0)


def find_start(samples: np.ndarray, *, share: float = 0.5) -> float | None:
    """Time of the first start beep: three high beeps 0.6 s apart."""
    on = tone_share(samples, START_HZ) >= share
    step = round(START_BEEPS[1] / FRAME)
    for i in np.flatnonzero(on):
        if all(on[max(0, i + k * step - 1): i + k * step + 2].any() for k in (1, 2)):
            return float(i * FRAME)
    return None


def beeps_heard(samples: np.ndarray, times: list[float], *, share: float = 0.5) -> int:
    """How many of the strike beeps are in the sound where the schedule puts them."""
    on = tone_share(samples, STRIKE_HZ) >= share
    return sum(bool(on[max(0, int(t / FRAME) - 3): int(t / FRAME) + 4].any()) for t in times)


# --- scoring ----------------------------------------------------------------------------------

def score(events: list[dict], plan: dict, start: float) -> dict:
    """Compare detected strikes with the called ones. ``start`` is the first start beep, in video seconds."""
    def peak(event):
        value = event.get("peak_time", event.get("time_seconds"))
        return None if value is None else float(value)

    begin, finish = start, start + plan["finished"]
    detected = {side: sorted(p for e in events if e.get("fighter") == side
                             and (p := peak(e)) is not None and begin <= p <= finish) for side in ("A", "B")}
    family_at = {}
    for e in events:
        if e.get("fighter") == "A" and peak(e) is not None:
            family_at.setdefault(peak(e), e.get("family"))

    free = list(detected["A"])
    found, right_family, per_block = 0, 0, {}
    for call in plan["strikes"]:
        beep = start + call["time"]
        match = next((p for p in free if beep - BEFORE <= p <= beep + AFTER), None)
        row = per_block.setdefault(call["block"], {"technique": call["technique"], "called": 0, "found": 0})
        row["called"] += 1
        if match is not None:
            free.remove(match)
            found += 1
            row["found"] += 1
            right_family += family_at.get(match) == call["family"]
    rests = [b for b in plan["blocks"] if b["kind"] == "rest"]
    in_rest = sum(any(start + b["start"] <= p < start + b["end"] for b in rests) for p in detected["A"])
    for number, block in enumerate(plan["blocks"]):
        if block["kind"] == "strike":
            per_block[number]["detected"] = sum(start + block["start"] <= p < start + block["end"]
                                                for p in detected["A"])
    called = len(plan["strikes"])
    return {
        "called": called, "detected": len(detected["A"]), "found": found,
        "precision": found / len(detected["A"]) if detected["A"] else None,
        "recall": found / called if called else None,
        "family_right": right_family, "in_rest": in_rest,
        "rest_seconds": sum(b["end"] - b["start"] for b in rests),
        "holder_detections": len(detected["B"]),
        "blocks": [per_block[k] for k in sorted(per_block)],
    }


def verdict(result: dict) -> str:
    p = result["precision"]
    if result["called"] < BAR_STRIKES:
        return f"Too few strikes called to judge ({result['called']} < {BAR_STRIKES})."
    if p is None:
        return "No strikes detected for the striker: nothing to count."
    passed = p >= BAR_PRECISION
    return (f"Precision {p:.2f} {'meets' if passed else 'is below'} the {BAR_PRECISION:.2f} bar "
            f"over {result['called']} called strikes.")


def main(argv=None) -> int:
    from exam_ufc_fight import people_at, save_pick, slug

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--at", type=float, default=10.0, help="seconds into the video where both people show")
    parser.add_argument("--a", type=int, help="number of the striker in the saved picture")
    parser.add_argument("--b", type=int, help="number of the pad holder in the saved picture")
    parser.add_argument("--start", type=float, help="seconds into the video of the first high beep, "
                                                    "if it cannot be found in the sound")
    parser.add_argument("--checkpoint", type=Path, help="a candidate temporal model instead of the live one")
    args = parser.parse_args(argv)
    if not args.video.exists():
        raise SystemExit(f"No video at {args.video}.")

    if args.a is None or args.b is None:
        boxes = people_at(args.video, args.at)
        picture = RESULTS / f"drill_pick_{slug(args.video.stem)}.png"
        save_pick(args.video, args.at, boxes, picture)
        print(f"Found {len(boxes)} people at {args.at:g} s. Open {picture.relative_to(ROOT)} and run again "
              "with --a <striker's number> --b <pad holder's number>.")
        return 0

    plan = page_schedule()
    if plan != schedule():
        raise SystemExit("tools/drill_caller.html does not match the plan in this file.")
    samples = None
    if args.start is None:
        samples = read_audio(args.video)
        args.start = find_start(samples)
        if args.start is None:
            raise SystemExit("Could not hear the three high start beeps. Give --start <seconds> "
                             "(when the first one is heard) instead.")
    if samples is not None:
        heard = beeps_heard(samples, [args.start + s["time"] for s in plan["strikes"]])
        print(f"Drill starts at {args.start:.2f} s; {heard} of {len(plan['strikes'])} strike beeps heard "
              "where the plan puts them.")
        if heard < 0.8 * len(plan["strikes"]):
            raise SystemExit("The beeps do not line up with the plan, so the answer key would be wrong. "
                             "Was the whole drill filmed, without pausing?")

    boxes = people_at(args.video, args.at)
    if not (1 <= args.a <= len(boxes) and 1 <= args.b <= len(boxes)) or args.a == args.b:
        raise SystemExit(f"--a and --b must be two different numbers from 1 to {len(boxes)}.")
    if args.checkpoint:
        os.environ["WARRIORIQ_TEMPORAL_MODEL"] = str(args.checkpoint)
    import cv2

    from core import analyzer
    from core.types import AnalysisRequest

    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / max(1.0, fps)
    capture.release()
    job = f"drill_{slug(args.video.stem)}"
    report = analyzer.analyze(AnalysisRequest(
        video_path=str(args.video), fighter_a_box=boxes[args.a - 1], fighter_b_box=boxes[args.b - 1],
        ruleset="K1", fight_type="sparring", round_count=1, start_seconds=0.0,
        round_duration_seconds=max(1.0, duration), selection_seconds=args.at,
        job_id=job, profile_id=1, persist_result=False, output_dir=str(ROOT / "outputs" / job)))

    result = score(report.get("events") or [], plan, args.start)
    result["model"] = (report.get("classifier") or {}).get("temporal_checkpoint_sha256")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{job}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"{'block':12s} {'called':>6s} {'found':>6s} {'detected':>9s}")
    for row in result["blocks"]:
        print(f"{row['technique']:12s} {row['called']:6d} {row['found']:6d} {row['detected']:9d}")
    print(f"Found {result['found']} of {result['called']} called strikes; {result['detected']} detected in all, "
          f"{result['in_rest']} of them in {result['rest_seconds']:.0f} s of rest. "
          f"Right family on {result['family_right']} of the found. Pad holder: {result['holder_detections']} "
          "detections (should be 0).")
    print(verdict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
