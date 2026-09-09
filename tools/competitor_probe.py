"""Is the tracked person actually competing, or standing at the edge of it?

**This is a different question from the one this project kept asking.** Hand
labelling all 178 clips showed 24 identity failures, and of those:

    tracking the referee                     10
    tracking a bystander                      8
    the box jumping between people mid-clip   6
    fighter A confused with fighter B         0

Three quarters are "not a competitor at all". None is the A-versus-B confusion
that months of work went into - and that one may not be solvable here, since
the colour histogram separates the two athletes at AUC 0.667 and pooled ReID at
0.89-0.90 where the same person scores 0.93-0.98.

So this attacks the tractable three quarters.

**Behavioural features, deliberately not appearance.** A competitor probe was
tried before on the ReID embedding and learned "blue singlet on a blue mat"
from 18 positives in one bout. An embedding's easiest generalisation is the
venue. What separates a competitor from a bystander is what they *do*: a
fighter moves, closes distance, and throws limbs; a seated official does not.
The one appearance signal kept is the referee uniform probe, because the
referee is the failure motion cannot catch - he is on the mat, moving, near the
fighters - and that probe already ranks officials at AUC 0.993.

Judged leave-one-fight-out. A probe that scores well pooled and badly held out
has learned the bout, which is what happened last time and is invisible without
the fold.

**RESULT, 2026-09-09: it does not work, and nothing here is wired into the
analysis.** Run on the 178 labelled clips:

    held out          AUC
    fight 1           0.453
    fight 2           skipped - only one class in that fold
    fight 3           0.556
    mean              0.504     <- chance

The reframe was right and the data still is not there. Of 178 clips only
**eleven** are usable non-competitors: 24 identity failures, minus the six
id-swaps which are neither class cleanly, minus those whose track is too short
for a 1.5-second window. Five or six negatives per fold cannot train anything,
whatever the features are.

Pooled, some features do carry signal - `height_share` 0.653 (a bystander is
further away and smaller in frame), `gap_change` 0.274 which is 0.726 inverted.
Pooled is not transfer, and this is exactly the trap the last attempt fell into.

Worth keeping as the harness for when there are more labelled fights. Worth
re-running when there are, and not before: the shortage is examples, not ideas.

One measured surprise for whoever picks this up - **the referee moves more than
a fighter, not less**. `centre_spread` scores 0.360, meaning non-competitors
have the *higher* spread, because the official walks the whole mat while an
athlete works in a small area. Any "bystanders are still" intuition is wrong
here, and only the seated ones behave that way.

    tools/competitor_probe.py --labels <scratch dir with answers_*.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.action import L_ANKLE, L_KNEE, L_WRIST, R_ANKLE, R_KNEE, R_WRIST
from core.referee import referee_probability

VIDEOS = {
    "fam3": "fights/1.mp4",
    "f2_gateh": "fights/0-02-05-5736bb3acb024e3772ad5fd6341d0b6768ccece93b7c329dcfa26f3ec1478f00_d242ef68be9b3ffe.mp4",
    "f3_gateh": "fights/0-02-05-f84ec82f3271783fcb884ca43f15f94aae2ec3086be1ca26774d5784b373e3b9_336b8df8343b952b.mp4",
}
NOT_A_COMPETITOR = {"referee", "wrongsubject"}
COMPETITOR = {"punch", "kick", "knee", "none"}
WINDOW_SECONDS = 1.5
FEATURE_NAMES = (
    "referee_uniform", "centre_spread", "limb_travel", "height_share",
    "opponent_gap", "gap_change", "vertical_position",
)


def _tracking(job: str) -> dict[int, dict]:
    records = {}
    for line in (PROJECT_ROOT / "outputs" / job / "tracking.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("source_frame") is not None:
            records[int(record["source_frame"])] = record
    return records


def _point(keypoints, index):
    if keypoints is None or index >= len(keypoints):
        return None
    x, y = float(keypoints[index][0]), float(keypoints[index][1])
    return np.array([x, y], dtype=np.float32) if np.isfinite(x) and np.isfinite(y) else None


def _travel(samples, index, scale):
    points = [p for p in (_point(k, index) for k in samples) if p is not None]
    if len(points) < 2:
        return 0.0
    return float(sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1))) / max(1.0, scale)


def features(job, fighter, peak_frame, tracking, frame, height, width, fps=30.0):
    """Seven numbers, all scale-free, none of them a colour except the uniform."""
    span = max(2, int(fps * WINDOW_SECONDS))
    other = "B" if fighter == "A" else "A"
    boxes, keypoints, opponent = [], [], []
    for offset in range(-span, span + 1):
        record = tracking.get(peak_frame + offset)
        if not record:
            continue
        own = (record.get("fighter_%s" % fighter) or {}).get("observation") or {}
        if own.get("box"):
            boxes.append(own["box"])
            keypoints.append(own.get("keypoints"))
        opp = (record.get("fighter_%s" % other) or {}).get("observation") or {}
        if opp.get("box"):
            opponent.append(opp["box"])
    if len(boxes) < 4:
        return None

    heights = [max(1.0, b[3] - b[1]) for b in boxes]
    scale = float(np.median(heights))
    centres = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes], dtype=np.float32)

    peak_box = boxes[len(boxes) // 2]
    uniform = referee_probability(frame, peak_box)

    gaps = []
    if opponent:
        opp_centres = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in opponent], dtype=np.float32)
        n = min(len(centres), len(opp_centres))
        gaps = [float(np.linalg.norm(centres[i] - opp_centres[i]) / scale) for i in range(n)]

    return np.array([
        0.0 if uniform is None else float(uniform),
        float(np.linalg.norm(centres.std(axis=0)) / scale),
        max(_travel(keypoints, L_WRIST, scale), _travel(keypoints, R_WRIST, scale),
            _travel(keypoints, L_ANKLE, scale), _travel(keypoints, R_ANKLE, scale),
            _travel(keypoints, L_KNEE, scale), _travel(keypoints, R_KNEE, scale)),
        scale / max(1.0, height),
        float(np.median(gaps)) if gaps else 6.0,
        float(np.std(gaps)) if len(gaps) > 2 else 0.0,
        float(np.median(centres[:, 1]) / max(1.0, height)),
    ], dtype=np.float32)


def build(scratch: Path):
    rows, labels, fights = [], [], []
    for job, video in VIDEOS.items():
        answers = json.loads((scratch / ("answers_%s.json" % job)).read_text(encoding="utf-8"))
        index = json.loads((PROJECT_ROOT / "labelpack" / job / "index.json").read_text(encoding="utf-8"))
        by_id = {c["id"]: c for c in index["candidates"]}
        tracking = _tracking(job)
        cap = cv2.VideoCapture(str(PROJECT_ROOT / video))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        for cid, verdict in answers.items():
            # idswap is neither class cleanly - the box is on a competitor for
            # part of the window and somebody else for the rest.
            if verdict not in NOT_A_COMPETITOR | COMPETITOR:
                continue
            item = by_id[int(cid)]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(item["peak_frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            vector = features(job, item["fighter"], int(item["peak_frame"]), tracking, frame, height, width)
            if vector is None:
                continue
            rows.append(vector)
            labels.append(0 if verdict in NOT_A_COMPETITOR else 1)
            fights.append(job)
        cap.release()
    return np.asarray(rows), np.asarray(labels), fights


def _fit_logistic(X, y, l2=2.0, steps=4000, rate=0.15):
    """Ridge-regularised logistic regression, class-balanced, in numpy.

    Written out rather than pulled in, because this is an experiment that may
    well be reverted and a new runtime dependency should not outlive it. The
    classes are 12:1, so the loss is weighted - unweighted, the model that
    calls everything a competitor is 92% accurate and useless.
    """
    A = np.c_[X, np.ones(len(X))]
    w = np.zeros(A.shape[1])
    weight = np.where(y == 1, 1.0 / max(1, (y == 1).sum()), 1.0 / max(1, (y == 0).sum()))
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        grad = A.T @ (weight * (p - y)) + l2 * np.r_[w[:-1], 0.0] / len(A)
        w -= rate * grad
    return w


def _auc(scores, truth):
    pos, neg = scores[truth == 1], scores[truth == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Competitor-vs-bystander probe, judged across fights.")
    parser.add_argument("--labels", required=True, help="directory holding answers_<job>.json")
    args = parser.parse_args()

    X, y, fights = build(Path(args.labels))
    print("samples: %d  (%d competitors, %d not)" % (len(y), (y == 1).sum(), (y == 0).sum()))
    print()
    print("each feature on its own, pooled:")
    for i, name in enumerate(FEATURE_NAMES):
        print("   %-18s AUC %.3f" % (name, _auc(X[:, i], y)))

    # Standardise on the training fold only, or the held-out fight leaks in.
    print("\nlogistic regression, leave-one-fight-out:")
    aucs = []
    for held in VIDEOS:
        mask = np.array([f == held for f in fights])
        if len(set(y[~mask])) < 2 or len(set(y[mask])) < 2:
            print("   %-10s skipped (only one class in this fold)" % held)
            continue
        mean, std = X[~mask].mean(axis=0), X[~mask].std(axis=0) + 1e-6
        weights = _fit_logistic((X[~mask] - mean) / std, y[~mask])
        scores = np.c_[(X[mask] - mean) / std, np.ones(mask.sum())] @ weights
        auc = _auc(scores, y[mask])
        aucs.append(auc)
        print("   %-10s held out: AUC %.3f  (%d competitors, %d not)"
              % (held, auc, (y[mask] == 1).sum(), (y[mask] == 0).sum()))
    if aucs:
        print("\n   mean held-out AUC: %.3f" % float(np.mean(aucs)))
        print("   (0.5 is chance. Below it means the probe inverts on unseen footage.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
