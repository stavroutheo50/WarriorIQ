"""Label strikes in finished analyses where two independent methods agree - no person involved.

    python tools/auto_label.py --checkpoint models/warrioriq_temporal_best.pt \\
        --jobs outputs/<job> [outputs/<job> ...] --out dataset/sequences_auto

WarriorIQ has two ways of reading a strike that do not share a decision:

* the **rules** (core/action.py): limb speed, extension and direction, which
  name a technique from geometry - "left_hook", "right_round_kick";
* the **learned model** (core/temporal_model.py), trained on the public,
  human-labelled datasets, which names a class from the 12-frame window.

A window becomes a label only when both name **the same class** and the model
is at least ``--min-probability`` (0.90) sure. A "none" window is one where the
rules found nothing within ``--quiet-seconds`` (1.0 s) for that fighter and the
model says "none" at ``--min-none-probability`` (0.95). Anything else -
disagreement, a less certain model, a fighter not held for the whole window -
is thrown away rather than guessed.

Only analyses whose identity check passed are read, so a label is about the
fighter it names.

## Training material, never an answer key

Every sequence is written with fight id ``auto_<job>``. tools/strike_exam.py
refuses it as exam evidence: two methods agreeing is not proof, and a model
graded against labels it helped make cannot fail. Agreement can also inherit a
shared blind spot, which is why the exam that decides publication runs only on
labels people made and official totals.

Features are built by core.action._feature_vector from the analysis's own
tracking, the windows end at each strike's peak like
tools/build_labelled_sequences.py, and the label is a class index into
core.temporal_model.ACTION_CLASSES (never the binary 0/1 that tool writes).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import numpy as np  # noqa: E402

from core.action import _feature_vector  # noqa: E402
from core.strike_exam import AUTO_PREFIX  # noqa: E402
from core.temporal_model import ACTION_CLASSES  # noqa: E402


def window_at(samples, times: np.ndarray, at: float, window: int):
    """Features of the ``window`` analysed frames ending nearest ``at``, or None."""
    if not np.isfinite(times).any():
        return None
    index = int(np.nanargmin(np.abs(times - at)))
    chunk = samples[max(0, index - window + 1):index + 1]
    if len(chunk) < window or any(s is None for s in chunk):
        return None
    x = np.stack([_feature_vector(s, chunk[i - 1] if i else None) for i, s in enumerate(chunk)]).astype(np.float32)
    return x if np.isfinite(x).all() else None


def probabilities(model, windows: list[np.ndarray]) -> np.ndarray:
    import torch

    if not windows:
        return np.zeros((0, len(ACTION_CLASSES)), np.float32)
    with torch.inference_mode():
        return torch.softmax(model(torch.from_numpy(np.stack(windows))), dim=-1).numpy()


def label_job(job: Path, model, out: Path, args, rng: np.random.Generator) -> collections.Counter:
    from build_labelled_sequences import samples_for

    tally: collections.Counter = collections.Counter()
    report = json.loads((job / "report.json").read_text(encoding="utf-8"))
    if not (report.get("integrity") or {}).get("identity_evidence_trusted"):
        tally["skipped: identity check did not pass"] += 1
        return tally
    events = json.loads((job / "events.json").read_text(encoding="utf-8"))
    events = events.get("events", events) if isinstance(events, dict) else events
    group = f"{AUTO_PREFIX}{job.name}"
    written = 0
    for side in ("A", "B"):
        samples = samples_for(job, side)
        times = np.array([s.time if s else np.nan for s in samples], dtype=np.float64)
        mine = [e for e in events if e.get("fighter") == side and e.get("technique") in ACTION_CLASSES]
        strike_windows, strike_classes = [], []
        for event in mine:
            x = window_at(samples, times, float(event.get("peak_time") or 0.0), args.window)
            if x is None:
                tally["dropped: fighter not held for the whole window"] += 1
                continue
            strike_windows.append(x)
            strike_classes.append(ACTION_CLASSES.index(event["technique"]))
        for x, rule_class, p in zip(strike_windows, strike_classes, probabilities(model, strike_windows)):
            guess = int(p.argmax())
            if guess != rule_class:
                tally["dropped: rules and model disagree"] += 1
            elif float(p[guess]) < args.min_probability:
                tally["dropped: model not sure enough"] += 1
            else:
                np.savez_compressed(out / f"{group}__{written:05d}.npz", x=x, y=np.int64(guess), fight_id=group)
                written += 1
                tally[ACTION_CLASSES[guess]] += 1
        # Quiet moments: no rule strike near, and the model sure nothing happened.
        peaks = np.array([float(e.get("peak_time") or 0.0) for e in mine])
        held = [t for t in times[np.isfinite(times)] if not len(peaks) or np.min(np.abs(peaks - t)) > args.quiet_seconds]
        rng.shuffle(held)
        quiet = [w for w in (window_at(samples, times, t, args.window) for t in held[:args.max_none * 4]) if w is not None]
        kept = 0
        for x, p in zip(quiet, probabilities(model, quiet)):
            if kept >= args.max_none:
                break
            if int(p.argmax()) == 0 and float(p[0]) >= args.min_none_probability:
                np.savez_compressed(out / f"{group}__{written:05d}.npz", x=x, y=np.int64(0), fight_id=group)
                written += 1
                kept += 1
                tally["none"] += 1
            else:
                tally["dropped: quiet moment the model was not sure of"] += 1
    return tally


def load_model(checkpoint: Path):
    import torch

    from core.temporal_model import build_temporal_network

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if list(payload.get("classes", ACTION_CLASSES)) != ACTION_CLASSES:
        raise SystemExit("checkpoint classes do not match this build")
    model = build_temporal_network(payload.get("architecture", "gru_v1"), int(payload.get("input_dim", 102)),
                                   len(ACTION_CLASSES))
    model.load_state_dict(payload.get("state_dict", payload))
    model.eval()
    return model


def main(argv=None) -> int:
    from core.config import SETTINGS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--jobs", nargs="+", required=True, type=Path, help="analysis result directories")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "sequences_auto")
    parser.add_argument("--min-probability", type=float, default=0.90)
    parser.add_argument("--min-none-probability", type=float, default=0.95)
    parser.add_argument("--quiet-seconds", type=float, default=1.0)
    parser.add_argument("--max-none", type=int, default=40, help="quiet windows kept per fighter per job")
    parser.add_argument("--window", type=int, default=int(SETTINGS.action_window))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    total: collections.Counter = collections.Counter()
    for job in args.jobs:
        tally = label_job(job, model, args.out, args, rng)
        print(f"{job.name}: {dict(tally)}")
        total.update(tally)
    print(f"Total: {dict(total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
