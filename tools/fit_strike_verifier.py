"""Fit the punch check in core/strike_verifier.py from judged punch proposals.

Each fight folder in dataset/public/strike_judged/ holds:

  track.jsonl.gz  the analyser's tracking.jsonl
  judged.json     one answer per detector proposal, found again by fighter,
                  limb and motion-peak time: "yes" (the credited fighter threw
                  a strike in those frames), "no", "unsure" or "wrong_person"

dataset/public/ is not in Git: pose tracks of identifiable athletes stay out
of a public repository. Only "yes" and "no" punches are used.

    python tools/fit_strike_verifier.py             # held-out scores only
    python tools/fit_strike_verifier.py --write     # also store the weights

The held-out score is the one that matters: every fight is scored by a model
fitted on the others, so a fight is never judged by weights that saw it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import strike_verifier  # noqa: E402

ROOT = PROJECT_ROOT / "dataset" / "public" / "strike_judged"
L2 = 1.0


def proposals(track: Path) -> list:
    """Every proposal the action path makes on this track, before any gate."""
    from core.action import ActionEngine
    from core.contact import classify_contact
    from tools.benchmark_labelled_fight import _observation
    import gzip

    engine, out = ActionEngine(), []
    with gzip.open(track, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            a = _observation(record["fighter_A"]["observation"])
            b = _observation(record["fighter_B"]["observation"])
            conf_a = float(record["fighter_A"]["identity_confidence"])
            conf_b = float(record["fighter_B"]["identity_confidence"])
            args = (record["source_frame"], record["time_seconds"], record["round_number"])
            for event in engine.update("A", *args, a, b, conf_a, conf_b) + engine.update("B", *args, b, a, conf_b, conf_a):
                out.append(classify_contact(event))
    return out


def judged_punches() -> list[tuple[str, object, float]]:
    rows = []
    for fight in sorted(p for p in ROOT.iterdir() if (p / "judged.json").exists()):
        answers = json.loads((fight / "judged.json").read_text(encoding="utf-8"))
        events = proposals(fight / "track.jsonl.gz")
        for answer in answers:
            if answer["answer"] not in {"yes", "no"}:
                continue
            match = [e for e in events if e.fighter == answer["fighter"] and e.limb == answer["limb"]
                     and abs(float(e.evidence.get("action_peak_time", e.peak_time)) - answer["peak"]) < 0.01]
            if match and match[0].family == "punch":
                rows.append((fight.name, match[0], float(answer["answer"] == "yes")))
    return rows


def fit(X: np.ndarray, y: np.ndarray, steps: int = 4000, rate: float = 0.1):
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / scale
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        w -= rate * (Z.T @ (p - y) / len(y) + L2 * w / len(y))
        b -= rate * float(np.mean(p - y))
    return mean, scale, w, b


def predict(model, X: np.ndarray) -> np.ndarray:
    mean, scale, w, b = model
    return 1.0 / (1.0 + np.exp(-(((X - mean) / scale) @ w + b)))


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def write_weights(names, model, source: str) -> None:
    mean, scale, w, b = model
    path = PROJECT_ROOT / "core" / "strike_verifier.py"
    text = path.read_text(encoding="utf-8")
    fmt = lambda values: "(" + ", ".join(f"{float(v):.4f}" for v in values) + ",)"  # noqa: E731
    block = (f"FEATURES: tuple[str, ...] = {tuple(names)!r}\n"
             f"MEAN: tuple[float, ...] = {fmt(mean)}\n"
             f"SCALE: tuple[float, ...] = {fmt(scale)}\n"
             f"WEIGHTS: tuple[float, ...] = {fmt(w)}\n"
             f"BIAS = {float(b):.4f}\n"
             f"WEIGHTS_SOURCE = {source!r}\n")
    text = re.sub(r"FEATURES: tuple\[str, \.\.\.\] = .*?\nWEIGHTS_SOURCE = .*?\n", lambda _: block, text, flags=re.S)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="store the weights fitted on every fight")
    args = parser.parse_args()
    rows = judged_punches()
    names = list(strike_verifier.features(rows[0][1]))
    X = np.array([[strike_verifier.features(e)[n] for n in names] for _, e, _ in rows])
    y = np.array([label for *_, label in rows])
    fights = np.array([f for f, *_ in rows])
    held_out = np.zeros(len(y))
    for fight in np.unique(fights):
        held_out[fights == fight] = predict(fit(X[fights != fight], y[fights != fight]), X[fights == fight])
    print(f"{len(y)} judged punches in {len(np.unique(fights))} fights, {int(y.sum())} real; "
          f"held-out AUC {auc(held_out, y):.3f}")
    for fight in np.unique(fights):
        mask = fights == fight
        print(f"  {fight:10} {int(mask.sum()):4} punches  held-out AUC {auc(held_out[mask], y[mask]):.3f}")
    if args.write:
        model = fit(X, y)
        source = (f"{len(y)} judged punch proposals ({int(y.sum())} real) in {len(np.unique(fights))} public-domain "
                  f"phone sparring clips; held-out AUC {auc(held_out, y):.3f}")
        write_weights(names, model, source)
        print("weights written to core/strike_verifier.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
