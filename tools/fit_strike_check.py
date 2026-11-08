"""Fit the punch check in core/strike_verifier.py from judged punch proposals.

Each fight folder in dataset/public/strike_judged/ holds:

  track.jsonl.gz  the analyser's tracking.jsonl (where each fighter's box was)
  judged.json     one answer per detector proposal - fighter, limb, motion
                  peak time and "yes" (the credited fighter threw a strike in
                  those frames), "no", "unsure" or "wrong_person"
  source.json     {"video": "<path to the video>"}

dataset/public/ is not in Git: pose tracks of identifiable athletes stay out
of a public repository. Only punches answered "yes" or "no" are used.

With --dense, the fights where every strike is labelled
(tools/benchmark_dense_strikes.py) add their punch proposals too. Off by
default: those labels are Claude's, and the owner confirmed only 10 of 30
sampled ones (dataset/regression/label_check, 2026-11-08). A proposal is a
real punch when a sure labelled punch by the same fighter is within the
benchmark's matching window, not one when no label of either kind is near,
and is left out when only an unsure label is. Their features are the ones `benchmark_dense_strikes.py --video` caches
(features_check.npz); run that first.

    python tools/fit_strike_check.py             # held-out scores only
    python tools/fit_strike_check.py --write     # also store models/strike_check_head.npz

Every video is scored by a head fitted on the others, so no fight is judged by
weights that saw any of its frames; that is the number to read. Features are computed by the
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


def video_of(fight: Path) -> str:
    source = json.loads((fight / "source.json").read_text(encoding="utf-8"))
    return source.get("video") or source.get("download_url") or fight.name


def dense_rows(fight: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Punch proposals in a densely labelled fight, as (features, real or not)."""
    from tools import benchmark_dense_strikes as dense

    cache_path = fight / "features_check.npz"
    if not cache_path.exists() or not (fight / "source.json").exists():
        return None
    cache = np.load(cache_path)
    truth = json.loads((fight / "strikes.json").read_text(encoding="utf-8"))
    start, end = truth["covered"]
    spans = truth.get("unreadable") or []
    X, y = [], []
    for event in dense._thrown(fight / "track.jsonl.gz"):
        moment = float(event.peak_time)
        if event.family != "punch" or not start <= moment <= end or dense._inside(moment, spans):
            continue
        key = f"{event.fighter}:{float(event.evidence.get('action_peak_time', event.peak_time)):.3f}"
        if key not in cache:
            continue
        near = [s for s in truth["strikes"]
                if s["fighter"] == event.fighter and abs(s["t"] - moment) <= dense.MATCH_SECONDS]
        if any(s.get("sure", True) and s.get("family", "punch") == "punch" for s in near):
            y.append(1.0)
        elif near:
            continue
        else:
            y.append(0.0)
        X.append(cache[key])
    return (np.array(X), np.array(y)) if X else None


def all_rows(with_dense: bool = False) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Every training row, grouped by the video it came from."""
    from tools import benchmark_dense_strikes as dense

    groups: dict[str, list] = {}
    for fight in sorted(p for p in ROOT.iterdir() if (p / "judged.json").exists() and (p / "source.json").exists()):
        groups.setdefault(video_of(fight), []).append((fight.name, *fight_rows(fight)))
    for fight in (dense.dense_fights() if with_dense else []):
        rows = dense_rows(fight)
        if rows is not None:
            groups.setdefault(video_of(fight), []).append((fight.name, *rows))
    return groups


def training_rows(exclude_video: str | None = None, with_dense: bool = False) -> tuple[np.ndarray, np.ndarray]:
    parts = [(X, y) for video, rows in all_rows(with_dense).items() if video != exclude_video for _, X, y in rows]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


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
    parser.add_argument("--dense", action="store_true", help="also learn from the densely labelled fights (unchecked labels)")
    args = parser.parse_args()
    groups = all_rows(args.dense)
    for video, rows in groups.items():
        rest = [(X, y) for other, more in groups.items() if other != video for _, X, y in more]
        head = fit(np.concatenate([r[0] for r in rest]), np.concatenate([r[1] for r in rest]))
        X = np.concatenate([r[1] for r in rows])
        y = np.concatenate([r[2] for r in rows])
        names = ", ".join(r[0] for r in rows)
        score = f"{auc(predict(head, X), y):.3f}" if 0 < y.sum() < len(y) else "n/a"
        print(f"{names:24} {len(y):4} punch proposals ({int(y.sum())} real)  held-out AUC {score}")
    if args.write:
        X, y = training_rows(with_dense=args.dense)
        np.savez(strike_verifier.HEAD_PATH, **fit(X, y))
        print(f"head written to {strike_verifier.HEAD_PATH.relative_to(PROJECT_ROOT)} "
              f"({len(y)} punches, {int(y.sum())} real, {len(groups)} videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
