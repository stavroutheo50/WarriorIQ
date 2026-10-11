"""Train the peak-pose model and score it on videos it never saw.

    .venv\\Scripts\\python.exe tools\\train_peak_pose.py

Reads what tools/import_combat_sports.py wrote, trains core.peak_pose on the
``train`` videos (with each pose also seen mirrored, as a southpaw would throw
it) and prints precision and recall on the held-out ``exam`` videos. The model
is written to models/peak_pose_candidate.npz with its SHA-256; nothing in the
live analysis reads it. The next step is tools/test_model_on_own_footage.py
--peak-pose, which asks how it behaves on WarriorIQ's own fights.

A class with fewer than ``--min-exam`` (20) held-out examples is printed but
marked "too few to judge" - a precision over a handful of frames is not a
measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.peak_pose import PEAK_CLASSES, mirror_features, scores, train  # noqa: E402

DATA = ROOT / "dataset" / "peak_pose" / "combat_sports_gym.npz"
OUT = ROOT / "models" / "peak_pose_candidate.npz"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-exam", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.data.exists():
        raise SystemExit(f"No data at {args.data}. Run tools/import_combat_sports.py first.")

    with np.load(args.data, allow_pickle=False) as blob:
        x, family, split = blob["features"], [str(f) for f in blob["family"]], blob["split"]
    y = np.asarray([PEAK_CLASSES.index(f) for f in family])
    fit, exam = split == "train", split == "exam"
    if not fit.any() or not exam.any():
        raise SystemExit("Need both train and exam videos in the data.")

    model = train(np.concatenate([x[fit], mirror_features(x[fit])]), np.concatenate([y[fit], y[fit]]),
                  seed=args.seed)
    truth = [family[i] for i in np.flatnonzero(exam)]
    result = scores(truth, model.predict(x[exam]))
    print(f"Held-out videos: {int(exam.sum())} examples. Trained on {int(fit.sum())} (x2 mirrored).")
    for name in PEAK_CLASSES:
        r = result[name]
        p = "-" if r["precision"] is None else f"{r['precision']:.2f}"
        q = "-" if r["recall"] is None else f"{r['recall']:.2f}"
        note = "  too few to judge" if r["true"] < args.min_exam else ""
        print(f"  {name:6s} precision {p:>4s} ({r['correct']}/{r['called']})  recall {q:>4s} "
              f"({r['correct']}/{r['true']}){note}")
    sha = model.save(args.out, seed=args.seed, data=args.data.name, train_examples=int(fit.sum()))
    print(f"Saved {args.out} sha256 {sha[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
