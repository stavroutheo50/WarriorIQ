"""Fit the punch check in core/strike_verifier.py from judged punch proposals.

Each fight folder in dataset/public/strike_judged/ holds:

  track.jsonl.gz  the analyser's tracking.jsonl (where each fighter's box was)
  judged.json     one answer per detector proposal - fighter, limb, motion
                  peak time and "yes" (the credited fighter threw a strike in
                  those frames), "no", "unsure" or "wrong_person"
  source.json     {"video": "<path to the video>"}

dataset/public/ is not in Git: pose tracks of identifiable athletes stay out
of a public repository. Only punches answered "yes" or "no" are used.

    python tools/fit_strike_check.py             # held-out scores only
    python tools/fit_strike_check.py --write     # also store models/strike_check_head.npz

Every fight is scored by a head fitted on the others, so no fight is judged by
weights that saw it; that is the number to read. Features are computed by the
same function the analysis uses (core.strike_verifier.clip_features) and are
cached beside each fight, keyed by the backbone, so refitting is quick.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import strike_verifier  # noqa: E402

ROOT = PROJECT_ROOT / "dataset" / "public" / "strike_judged"
DIMENSIONS = 16
L2 = 10.0


def _track(fight: Path) -> list[tuple[float, list | None, list | None]]:
    rows = []
    with gzip.open(fight / "track.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            boxes = [((record.get(f"fighter_{f}") or {}).get("observation") or {}).get("box") for f in "AB"]
            rows.append((float(record["time_seconds"]), boxes[0], boxes[1]))
    return rows


def fight_rows(fight: Path) -> tuple[np.ndarray, np.ndarray]:
    answers = [a for a in json.loads((fight / "judged.json").read_text(encoding="utf-8"))
               if a["answer"] in {"yes", "no"} and a.get("family", "punch") == "punch"]
    cache = fight / "features_r3d18.npz"
    if cache.exists():
        stored = np.load(cache)
        if stored["peaks"].shape[0] == len(answers) and np.allclose(stored["peaks"], [a["peak"] for a in answers]):
            return stored["features"], stored["labels"]
    video = json.loads((fight / "source.json").read_text(encoding="utf-8"))["video"]
    lookup = strike_verifier.boxes_from_track(_track(fight))
    values = strike_verifier.clip_features(str(PROJECT_ROOT / video), [a["peak"] for a in answers],
                                           [lookup(a["fighter"]) for a in answers])
    keep = [i for i, v in enumerate(values) if v is not None]
    features = np.array([values[i] for i in keep])
    labels = np.array([float(answers[i]["answer"] == "yes") for i in keep])
    np.savez(cache, features=features, labels=labels, peaks=np.array([answers[i]["peak"] for i in keep]))
    return features, labels


def fit(X: np.ndarray, y: np.ndarray) -> dict:
    mean = X.mean(0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    projection = vt[:DIMENSIONS].T
    Z = (X - mean) @ projection
    scale = Z.std(0) + 1e-6
    Z = Z / scale
    weights, bias = np.zeros(DIMENSIONS), 0.0
    for _ in range(3000):
        p = 1.0 / (1.0 + np.exp(-(Z @ weights + bias)))
        weights -= 0.1 * (Z.T @ (p - y) / len(y) + L2 * weights / len(y))
        bias -= 0.1 * float(np.mean(p - y))
    return {"mean": mean, "projection": projection, "scale": scale, "weights": weights, "bias": np.array(bias)}


def predict(head: dict, X: np.ndarray) -> np.ndarray:
    z = ((X - head["mean"]) @ head["projection"]) / head["scale"]
    return 1.0 / (1.0 + np.exp(-(z @ head["weights"] + float(head["bias"]))))


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="store the head fitted on every fight")
    args = parser.parse_args()
    fights = sorted(p for p in ROOT.iterdir() if (p / "judged.json").exists() and (p / "source.json").exists())
    data = {fight.name: fight_rows(fight) for fight in fights}
    names = list(data)
    for name in names:
        rest_X = np.concatenate([data[o][0] for o in names if o != name])
        rest_y = np.concatenate([data[o][1] for o in names if o != name])
        X, y = data[name]
        print(f"{name:10} {len(y):4} judged punches ({int(y.sum())} real)  "
              f"held-out AUC {auc(predict(fit(rest_X, rest_y), X), y):.3f}")
    if args.write:
        X = np.concatenate([data[n][0] for n in names])
        y = np.concatenate([data[n][1] for n in names])
        np.savez(strike_verifier.HEAD_PATH, **fit(X, y))
        print(f"head written to {strike_verifier.HEAD_PATH.relative_to(PROJECT_ROOT)} "
              f"({len(y)} punches, {int(y.sum())} real, {len(names)} fights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
