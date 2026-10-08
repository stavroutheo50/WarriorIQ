"""Fetch, import, train, auto-label, retrain and examine the strike model - one command.

Run on the analysis machine (it needs the GPU stack), from the repository root:

    .venv\\Scripts\\python.exe tools\\strike_pipeline.py

and, when the result says a sport passed and you want counts live:

    .venv\\Scripts\\python.exe tools\\strike_pipeline.py --write-verdict

Stages, each skipped with a reason when its input is missing:

1. **Fetch** the commercially usable public sets (tools/fetch_public_datasets.py):
   TKD-Kick3 (CC BY 4.0), StrikeMetrics (MIT), UFC official statistics.
   BoxingVI only with ``--include-boxingvi``: it states no licence.
2. **Import** them into sequences (tools/import_tkd_kick3.py,
   tools/import_strikemetrics.py, tools/import_boxingvi.py).
3. **Negatives from your own footage** (tools/mine_own_negatives.py) when the
   library fights in tools/verified_seeds.json are on this machine. Without
   them a model learns the datasets, not fights: one trained on BoxingVI alone
   fired on 99.8% of real windows.
4. **Hold out the exam.** TKD-Kick3's test split (and, with BoxingVI, three of
   its videos) is copied to dataset/exam/ and never trained on.
5. **Train** (tools/train_temporal_model.py) on everything else.
6. **Auto-label** your finished analyses (tools/auto_label.py) with that model,
   where the rules and the model name the same strike, then **train again**
   with those windows added. ``--rounds`` sets how many times (default 2).
7. **Examine** the final model (tools/strike_exam.py) on the held-out clips
   and, if dataset/exam/ufc_bouts.json lists UFC bouts analysed by this model,
   on their official totals. The verdict is printed; ``--write-verdict``
   writes dataset/strike_exam_verdict.json.

The model is written to models/warrioriq_temporal_candidate.pt. Counts go live
only after **both** of these: the verdict file is committed and deployed to the
web app, and that exact file is copied to models/warrioriq_temporal_best.pt on
the worker (the verdict names its SHA-256, so any other file keeps counts off).

The one step this cannot do for you is the MMA count check: it needs a few UFC
bouts analysed with the candidate model (normal uploads, then list them in
dataset/exam/ufc_bouts.json - see tools/strike_exam.py). Until then the exam
reports the clip results and no sport passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
EXAM = DATASET / "exam"
TRAIN = DATASET / "sequences_pipeline"
CANDIDATE = ROOT / "models" / "warrioriq_temporal_candidate.pt"
UFC_MANIFEST = EXAM / "ufc_bouts.json"
UFC_CSV = DATASET / "public" / "ufc_stats" / "ufc_fight_stats.csv"

# Held out for the exam: never copied into the training directory.
EXAM_FIGHTS = {"tkd_test": "tkd_kick3_test"}
BOXINGVI_EXAM = ("boxingvi_V8", "boxingvi_V9", "boxingvi_V10")


def step(title: str, command: list[str]) -> bool:
    print(f"\n=== {title}\n$ {' '.join(command)}", flush=True)
    result = subprocess.run([sys.executable, *command], cwd=ROOT)
    if result.returncode != 0:
        print(f"--- {title}: failed (exit {result.returncode})", flush=True)
    return result.returncode == 0


def fight_of(path: Path) -> str:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return str(np.asarray(data["fight_id"]).item()) if "fight_id" in data else path.stem.split("__", 1)[0]


def gather(sources: list[Path], exam_fights: dict[str, str]) -> dict:
    """Copy sequences into TRAIN, and the held-out fights into EXAM/<name>."""
    for directory in [TRAIN, *(EXAM / name for name in set(exam_fights.values()))]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    counts = {"train": 0, "exam": 0}
    for source in sources:
        for path in sorted(source.glob("*.npz")):
            fight = fight_of(path)
            if fight in exam_fights:
                shutil.copy2(path, EXAM / exam_fights[fight] / f"{source.name}__{path.name}")
                counts["exam"] += 1
            else:
                shutil.copy2(path, TRAIN / f"{source.name}__{path.name}")
                counts["train"] += 1
    return counts


def finished_jobs() -> list[Path]:
    outputs = ROOT / "outputs"
    return sorted(p for p in outputs.glob("*") if (p / "report.json").exists()
                  and (p / "events.json").exists() and (p / "tracking.jsonl").exists())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-boxingvi", action="store_true", help="BoxingVI states no licence")
    parser.add_argument("--rounds", type=int, default=2, help="train / auto-label rounds")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--write-verdict", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_fetch:
        only = ["tkd_kick3", "strikemetrics", "ufc_stats"] + (["boxingvi"] if args.include_boxingvi else [])
        step("Fetch public datasets", ["tools/fetch_public_datasets.py", "--only", *only])

    sources = []
    tkd_zip = DATASET / "public" / "tkd_kick3" / "TKD-Kick3.zip"
    if tkd_zip.exists() and step("Import TKD-Kick3", ["tools/import_tkd_kick3.py", "--source", str(tkd_zip),
                                                      "--out", "dataset/sequences_tkd_kick3"]):
        sources.append(DATASET / "sequences_tkd_kick3")
    if (DATASET / "public" / "strikemetrics").exists() and step(
            "Import StrikeMetrics", ["tools/import_strikemetrics.py", "--root", "dataset/public/strikemetrics",
                                     "--out", "dataset/sequences_strikemetrics"]):
        sources.append(DATASET / "sequences_strikemetrics")
    exam_fights = dict(EXAM_FIGHTS)
    if args.include_boxingvi and (DATASET / "public" / "boxingvi").exists() and step(
            "Import BoxingVI", ["tools/import_boxingvi.py", "--source", "dataset/public/boxingvi",
                                "--out", "dataset/sequences_boxingvi", "--sided"]):
        sources.append(DATASET / "sequences_boxingvi")
        exam_fights.update({fight: "boxingvi_test" for fight in BOXINGVI_EXAM})
    if step("Negatives from your own footage", ["tools/mine_own_negatives.py", "--reuse"]):
        sources.append(DATASET / "sequences_own_negatives")
    sources = [s for s in sources if s.exists() and any(s.glob("*.npz"))]
    if not sources:
        print("\nNothing to train on: no dataset imported. Check the fetch step above.")
        return 1

    jobs = finished_jobs()
    auto = DATASET / "sequences_auto"
    for round_number in range(1, max(1, args.rounds) + 1):
        training_sources = list(sources)
        if round_number > 1 and auto.exists() and any(auto.glob("*.npz")):
            training_sources.append(auto)
        counts = gather(training_sources, exam_fights)
        print(f"\nRound {round_number}: {counts['train']} training windows, {counts['exam']} held out for the exam")
        if not step(f"Train (round {round_number})", ["tools/train_temporal_model.py", "--data", str(TRAIN),
                                                      "--epochs", str(args.epochs), "--out", str(CANDIDATE),
                                                      "--dataset-version", f"pipeline-round-{round_number}"]):
            print("\nTraining refused or failed. The usual cause is too few \"none\" windows: the public sets "
                  "hold only strikes, so the negatives come from your own footage (the library fights in "
                  "tools/verified_seeds.json must be on this machine) or, from round 2, auto-labels.")
            return 1
        if round_number < args.rounds:
            if not jobs:
                print("\nNo finished analyses in outputs/ to auto-label; stopping after this round.")
                break
            if auto.exists():
                shutil.rmtree(auto)
            step("Auto-label your analyses", ["tools/auto_label.py", "--checkpoint", str(CANDIDATE),
                                              "--jobs", *map(str, jobs), "--out", str(auto)])

    windows = [str(EXAM / name) for name in sorted(set(exam_fights.values()))
               if (EXAM / name).exists() and any((EXAM / name).glob("*.npz"))]
    if not windows:
        print("\nNo held-out exam clips; cannot examine.")
        return 1
    exam = ["tools/strike_exam.py", "--checkpoint", str(CANDIDATE), "--windows", *windows]
    if UFC_MANIFEST.exists() and UFC_CSV.exists():
        exam += ["--ufc-manifest", str(UFC_MANIFEST), "--official", str(UFC_CSV)]
    else:
        print(f"\nNo {UFC_MANIFEST.relative_to(ROOT)}: the MMA count check is skipped, so no sport can pass yet.")
    if args.write_verdict:
        exam.append("--write")
    passed = step("Examine", exam)
    print(f"\nCandidate model: {CANDIDATE.relative_to(ROOT)}")
    if args.write_verdict and passed:
        print("To switch counts on: commit dataset/strike_exam_verdict.json and deploy the web app, and copy "
              "the candidate to models/warrioriq_temporal_best.pt on the worker.")
    print(json.dumps({"rounds": args.rounds, "sources": [s.name for s in sources], "auto_labelled_jobs": len(jobs)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
