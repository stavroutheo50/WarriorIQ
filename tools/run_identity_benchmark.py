"""Measure who-is-who on the hand-marked clips, with and without a change.

    python tools/run_identity_benchmark.py                      # bystander memory off vs on
    python tools/run_identity_benchmark.py --clips ma640 V16    # a subset
    python tools/run_identity_benchmark.py --setting WARRIORIQ_IDENTITY_SIZE_GATE

Downloads the public-domain clips the truth files in
dataset/regression/identity_* describe (archive.org) into
dataset/public/identity_clips/, analyses each one twice - once with
the compared setting (--setting, WARRIORIQ_BYSTANDER_MEMORY unless told
otherwise) off and once on - and scores both against the
frames a person marked (tools/identity_benchmark.py). Prints one table.

Nothing touches the real fight database or outputs: each analysis runs in its
own process with uploads, outputs and the database redirected to
dataset/public/identity_runs/. The models folder, and so the TensorRT engine,
is the normal one. Stop the analysis worker first on an 8 GB card, or both
compete for the GPU.

The phone clips are scored on their first 55 s, the pankration bouts on the
whole marked window. Hard real-time planning is off so both arms analyse the
same frames.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRUTH = sorted((PROJECT_ROOT / "dataset" / "regression").glob("identity_*/*.json"))
WORK = PROJECT_ROOT / "dataset" / "public"
PHONE_SECONDS = 55.0
ARMS = {"off": "0", "on": "1"}


def _video_url(truth: dict) -> tuple[str, str]:
    item = truth["archive_item"]
    # The pankration boxes were marked on archive.org's 320x240 "512kb" copy;
    # the full-size file is 480x360, where every marked box is in the wrong place.
    name = truth.get("file") or f"{item}_512kb.mp4"
    return f"https://archive.org/download/{item}/{urllib.request.quote(name)}", name


def _download(truth: dict) -> Path:
    url, name = _video_url(truth)
    target = WORK / "identity_clips" / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name}", flush=True)
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(target)
    return target


def _analyse(video: Path, truth: dict, out_dir: Path, phone_seconds: float) -> float:
    """One analysis, in its own process so the setting is read fresh."""
    start = float(truth["start_seconds"])
    end = float(truth["end_seconds"]) if "file" not in truth else min(float(truth["end_seconds"]), start + phone_seconds)
    code = (
        "import json, sys\n"
        "from core import analyzer\n"
        "from core.types import AnalysisRequest\n"
        "t = json.loads(sys.argv[1])\n"
        "analyzer.analyze(AnalysisRequest(video_path=t['video'], fighter_a_box=t['a'], fighter_b_box=t['b'],\n"
        "    start_seconds=t['start'], selection_seconds=t['start'], end_seconds=t['end'], round_count=1,\n"
        "    round_duration_seconds=t['end'] - t['start'], output_dir=t['out'], persist_result=False))\n"
    )
    job = {"video": str(video), "a": truth["fighter_a_box"], "b": truth["fighter_b_box"],
           "start": start, "end": end, "out": str(out_dir)}
    started = time.perf_counter()
    subprocess.run([sys.executable, "-c", code, json.dumps(job)], cwd=PROJECT_ROOT, env=os.environ, check=True)
    return time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    from tools.identity_benchmark import score

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--clips", nargs="*", help="truth file names without .json (default: all)")
    parser.add_argument("--setting", default="WARRIORIQ_BYSTANDER_MEMORY",
                        help="the on/off setting to compare (default: WARRIORIQ_BYSTANDER_MEMORY; "
                             "the size gate was WARRIORIQ_IDENTITY_SIZE_GATE)")
    parser.add_argument("--phone-seconds", type=float, default=PHONE_SECONDS,
                        help="how much of each phone clip to analyse (default 55, as measured before)")
    args = parser.parse_args(argv)
    chosen = [p for p in TRUTH if not args.clips or p.stem in args.clips]

    runs = WORK / "identity_runs"
    os.environ.update({
        "WARRIORIQ_HARD_REALTIME": "0",
        "WARRIORIQ_UPLOADS_DIR": str(runs / "uploads"),
        "WARRIORIQ_OUTPUTS_DIR": str(runs / "outputs"),
        "WARRIORIQ_DB_PATH": str(runs / "benchmark.sqlite3"),
    })
    keys = ("right", "swapped", "partial", "other", "missing")
    totals = {arm: {**{key: 0 for key in keys}, "checked": 0} for arm in ARMS}
    print(f"{'clip':8} {'arm':4} {'right':>7} {'swapped':>8} {'partial':>8} {'other':>6} {'missing':>8} {'seconds':>8}")
    for path in chosen:
        truth = json.loads(path.read_text(encoding="utf-8"))
        video = _download(truth)
        marked = dict(truth)
        if "file" in truth:                                       # a phone clip: its first 55 s
            limit = float(truth["start_seconds"]) + args.phone_seconds
            marked["frames"] = [f for f in truth["frames"] if float(f["time_seconds"]) <= limit]
        for arm, value in ARMS.items():
            os.environ[args.setting] = value
            out_dir = runs / arm / path.stem
            seconds = _analyse(video, truth, out_dir, args.phone_seconds)
            records = [json.loads(line) for line in (out_dir / "tracking.jsonl").read_text().splitlines() if line]
            result = score(records, marked)
            row = {**{key: 0 for key in keys}, "checked": 0}
            for fighter in ("A", "B"):
                counts = result[fighter]
                for key in keys:
                    row[key] += int(counts.get(key, 0))
                row["checked"] += int(counts["frames"])
            for key in row:
                totals[arm][key] += row[key]
            print(f"{path.stem:8} {arm:4} {row['right']:>3}/{row['checked']:<3} {row['swapped']:>8} "
                  f"{row['partial']:>8} {row['other']:>6} {row['missing']:>8} {seconds:>8.0f}", flush=True)
    print()
    for arm, row in totals.items():
        print(f"TOTAL {args.setting} {arm:3}: right {row['right']}/{row['checked']}, right person with a smaller box "
              f"{row['partial']}, wrong person {row['swapped'] + row['other']} (swapped {row['swapped']}, "
              f"other {row['other']}), missing {row['missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
