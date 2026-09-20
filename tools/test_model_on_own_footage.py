"""Two questions a validation score cannot answer, asked with no labels at all.

A temporal checkpoint's validation accuracy is measured on the dataset it was
trained from. That number said 99.3% for a model that then fired on 99.8% of
windows of real tournament footage, and 99.1% for one that fires on 0%. Both
were useless, in opposite directions, and neither was visible from the metric.

So this asks the two questions that matter, on WarriorIQ's own fights, seeded
from tools/verified_seeds.json:

  **quiet**   Over every sliding window of the fight, how often does the model
              fire? A fight is mostly not-punching, so a model that fires
              everywhere cannot say `none` and will bury the report in false
              positives.

  **strike**  On the windows ending at each action the analysis DID detect,
              how often does it fire? A model that is silent here has not
              learned "punch vs not-punch" at all.

Neither needs a label. Together they bracket the failure: firing near 100% on
quiet is one broken model, firing near 0% on strike is the other, and the
interesting checkpoint is the one that separates them.

**Why both are needed, from the run that produced this file.** Training on
BoxingVI punches with BoxingVI negatives gave quiet=99.8%, strike=100% - it
called everything a punch. Adding in-domain negatives mined from this footage
gave quiet=0.3%, strike=**0%** - it now calls everything `none`. The second
model scores 99.1% on its own validation set and never fires on a real fight.

That is a dataset classifier. Every positive came from BoxingVI and every
in-domain negative from WarriorIQ, so "which dataset is this" separates the
training data perfectly and generalises to nothing. No amount of extra
negatives fixes it; the missing ingredient is labelled POSITIVES on this
footage, which is a human task and the reason the label pack exists.

    python tools/test_model_on_own_footage.py --ckpt models/experiments/x.pt --fights 1mp4,b883
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEEDS_PATH = PROJECT_ROOT / "tools" / "verified_seeds.json"


def _samples(job: Path, side: str):
    """One Sample per analysed frame for this fighter, None where not held."""
    from core.action import Sample

    rows = [json.loads(line) for line in
            (job / "tracking.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for row in rows:
        observation = ((row.get("fighter_%s" % side) or {}).get("observation")) or {}
        points, box = observation.get("keypoints"), observation.get("box")
        if not points or not box:
            out.append(None)
            continue
        keypoints = np.asarray(points, dtype=np.float32)
        conf = observation.get("keypoint_conf")
        out.append(Sample(
            frame=int(row["source_frame"]), time=float(row["time_seconds"]),
            round_number=row.get("round_number"),
            box=np.asarray(box, dtype=np.float32), keypoints=keypoints,
            conf=(np.asarray(conf, dtype=np.float32) if conf
                  else np.ones(len(keypoints), dtype=np.float32)),
            opponent_box=None, opponent_keypoints=None, opponent_conf=None,
            identity_confidence=1.0, opponent_identity_confidence=1.0))
    return out


def _features(chunk):
    from core.action import _feature_vector

    stacked = np.stack([
        _feature_vector(s, chunk[i - 1] if i else None)
        for i, s in enumerate(chunk)]).astype(np.float32)
    return stacked if np.isfinite(stacked).all() else None


def _predict(net, batch, classes):
    import torch

    with torch.no_grad():
        predicted = net(torch.from_numpy(np.stack(batch))).argmax(dim=1).numpy()
    return collections.Counter(classes[p] for p in predicted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--fights", default="1mp4,b883")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--reuse", action="store_true",
                        help="reuse an existing trace under outputs/ownfootage_<fight>")
    arguments = parser.parse_args()

    os.environ["WARRIORIQ_FORCE_STRIDE"] = str(int(arguments.stride))

    import cv2
    import torch

    from core import analyzer
    from core.config import SETTINGS
    from core.temporal_model import ACTION_CLASSES, build_temporal_network
    from core.types import AnalysisRequest

    blob = torch.load(PROJECT_ROOT / arguments.ckpt, map_location="cpu", weights_only=False)
    net = build_temporal_network("pose_transformer_v2", 102, len(ACTION_CLASSES))
    net.load_state_dict(blob.get("state_dict", blob) if isinstance(blob, dict) else blob)
    net.eval()

    seeds = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))["fights"]
    window = int(SETTINGS.action_window)

    print("checkpoint: %s" % arguments.ckpt)
    print("%-12s %26s %26s" % ("fight", "quiet (all windows)", "strike (detected actions)"))
    for key in [k.strip() for k in arguments.fights.split(",") if k.strip()]:
        fight = seeds.get(key)
        if not fight or not (PROJECT_ROOT / fight["video"]).exists():
            print("%-12s unknown or missing" % key)
            continue
        video = PROJECT_ROOT / fight["video"]
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()

        job = PROJECT_ROOT / "outputs" / ("ownfootage_%s" % key)
        if arguments.reuse and (job / "tracking.jsonl").exists() and (job / "events.json").exists():
            events = json.loads((job / "events.json").read_text(encoding="utf-8"))
        else:
            report = analyzer.analyze(AnalysisRequest(
                video_path=str(video),
                fighter_a_box=[float(v) for v in fight["fighter_a"]],
                fighter_b_box=[float(v) for v in fight["fighter_b"]],
                ruleset="K1", fight_type="competition", round_count=1,
                start_seconds=float(fight["seed_frame"]) / max(1.0, fps),
                round_duration_seconds=max(
                    1.0, float(total) / max(1.0, fps) - float(fight["seed_frame"]) / max(1.0, fps)),
                job_id="ownfootage_%s" % key, profile_id=1, persist_result=False,
                output_dir=str(job)))
            events = report.get("events") or []

        quiet_batch, strike_batch = [], []
        for side in ("A", "B"):
            samples = _samples(job, side)
            for start in range(0, len(samples) - window + 1):
                chunk = samples[start:start + window]
                if any(s is None for s in chunk):
                    continue
                features = _features(chunk)
                if features is not None:
                    quiet_batch.append(features)

            times = np.array([s.time if s else np.nan for s in samples], dtype=np.float64)
            if not np.isfinite(times).any():
                continue
            for event in events:
                if event.get("fighter") != side or event.get("peak_time") is None:
                    continue
                index = int(np.nanargmin(np.abs(times - float(event["peak_time"]))))
                chunk = samples[max(0, index - window + 1):index + 1]
                if len(chunk) < window or any(s is None for s in chunk):
                    continue
                features = _features(chunk)
                if features is not None:
                    strike_batch.append(features)

        def rate(batch):
            if not batch:
                return "no complete windows"
            tally = _predict(net, batch, ACTION_CLASSES)
            fired = sum(v for k, v in tally.items() if k != "none")
            return "%5d windows, fires %5.1f%%" % (len(batch), 100.0 * fired / len(batch))

        print("%-12s %26s %26s" % (key, rate(quiet_batch), rate(strike_batch)))

    print()
    print("Useful looks like a LOW quiet rate and a HIGH strike rate. Both near")
    print("100%% means it calls everything a punch; both near 0%% means it calls")
    print("everything nothing. Neither is visible in a validation score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
