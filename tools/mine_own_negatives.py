"""Cut `none` sequences from WarriorIQ's own footage, because BoxingVI's do not transfer.

A model trained on BoxingVI punches plus BoxingVI negatives scored 99.3% on
held-out BoxingVI fights and then **fired on 99.8% of windows of real tournament
footage** - 589 of 590, predicting `none` exactly once. It had not learned "is
this a punch". It had learned to tell a jab from a cross among pre-cut punch
clips, and to recognise the one video its negatives came from.

That is a domain shortcut, and the only cure is negatives from the domain the
model will actually run in. This produces them from the library fights, seeded
from tools/verified_seeds.json so the tracked person is a fighter rather than
whatever a rule picked.

## These negatives are heuristic, and the direction of their error is known

There is no ground truth here. A window is called `none` when no candidate
action from the analysis is anywhere near it, with a wide margin either side.
The detector this leans on is 29% precise and misses about a third of strikes,
so some windows called `none` do contain a real strike that nothing detected.

That error is one-directional and it is the safe direction from where the model
now stands: a missed strike taught as `none` makes the model **more** reluctant
to fire. A model that currently fires on 99.8% of windows has no reluctance to
spare, so being pulled toward silence is a correction rather than a new fault.
It would be the wrong trade for a model with the opposite problem.

**So this is a floor, not ground truth.** It cannot be used to claim recall, and
a model trained on it has still never been shown a verified negative. The honest
test remains a human-labelled pack on this footage.

    python tools/mine_own_negatives.py --out dataset/sequences_own_negatives
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


def windows_from_tracking(tracking: Path, window: int, event_times: list[float],
                          margin_seconds: float, step: int = 0):
    """Every run of `window` consecutive analysed frames that no event goes near.

    The scarce resource here is not quiet time, it is *unbroken* tracking: a
    window needs `window` consecutive analysed frames where that fighter was
    held, and coverage on this footage runs 0.6-0.9. Measured across four
    fights, widening the exclusion margin barely matters next to that:

        margin   1.5s  1.0s  0.5s  0.25s  0.0s
        windows    74    98   124    150    175

    So the margin stays generous - it is nearly free - and `step` is what
    buys volume. A step below `window` overlaps consecutive negatives, which
    is weak augmentation rather than new evidence; it is worth a little and
    should not be pushed to 1.
    """
    from core.action import Sample, _feature_vector

    rows = [json.loads(line) for line in
            tracking.read_text(encoding="utf-8").splitlines() if line.strip()]

    produced = []
    for side in ("A", "B"):
        samples: list = []
        for row in rows:
            block = row.get("fighter_%s" % side)
            observation = (block or {}).get("observation") or {}
            points = observation.get("keypoints")
            box = observation.get("box")
            if not points or not box:
                samples.append(None)
                continue
            keypoints = np.asarray(points, dtype=np.float32)
            confidence = observation.get("keypoint_conf")
            samples.append(Sample(
                frame=int(row["source_frame"]), time=float(row["time_seconds"]),
                round_number=row.get("round_number"),
                box=np.asarray(box, dtype=np.float32), keypoints=keypoints,
                conf=(np.asarray(confidence, dtype=np.float32) if confidence
                      else np.ones(len(keypoints), dtype=np.float32)),
                opponent_box=None, opponent_keypoints=None, opponent_conf=None,
                identity_confidence=1.0, opponent_identity_confidence=1.0))

        start = 0
        while start + window <= len(samples):
            chunk = samples[start:start + window]
            if any(s is None for s in chunk):
                start += 1
                continue
            begin, end = chunk[0].time, chunk[-1].time
            if any(begin - margin_seconds <= t <= end + margin_seconds for t in event_times):
                start += 1
                continue
            features = np.stack([
                _feature_vector(s, chunk[i - 1] if i else None)
                for i, s in enumerate(chunk)]).astype(np.float32)
            if np.isfinite(features).all():
                produced.append((side, chunk[0].frame, features))
            start += step if step > 0 else window
    return produced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="dataset/sequences_own_negatives")
    parser.add_argument("--stride", type=int, default=3,
                        help="pinned, so the sampling matches a reproducible run")
    parser.add_argument("--margin", type=float, default=1.5,
                        help="seconds either side of any detected action to avoid")
    parser.add_argument("--fights", default="",
                        help="comma-separated keys from verified_seeds.json; default all")
    parser.add_argument("--per-fight", type=int, default=400)
    parser.add_argument("--step", type=int, default=0,
                        help="frames between negatives; 0 means a full window (no overlap)")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse an existing tracking.jsonl instead of re-analysing")
    arguments = parser.parse_args()

    os.environ["WARRIORIQ_FORCE_STRIDE"] = str(int(arguments.stride))

    import cv2

    from core import analyzer
    from core.config import SETTINGS
    from core.temporal_model import ACTION_CLASSES
    from core.types import AnalysisRequest

    seeds = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))["fights"]
    wanted = [k.strip() for k in arguments.fights.split(",") if k.strip()] or sorted(seeds)
    destination = PROJECT_ROOT / arguments.out
    destination.mkdir(parents=True, exist_ok=True)
    window = int(SETTINGS.action_window)
    none_index = np.int64(ACTION_CLASSES.index("none"))

    per_fight: collections.Counter = collections.Counter()
    for key in wanted:
        fight = seeds.get(key)
        if not fight:
            print("  %-14s unknown key" % key)
            continue
        video = PROJECT_ROOT / fight["video"]
        if not video.exists():
            print("  %-14s video missing" % key)
            continue

        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()
        start_seconds = float(fight["seed_frame"]) / max(1.0, fps)

        job_dir = PROJECT_ROOT / "outputs" / ("negmine_%s" % key)
        cached = (job_dir / "tracking.jsonl").exists() and (job_dir / "events.json").exists()
        if arguments.reuse and cached:
            events = json.loads((job_dir / "events.json").read_text(encoding="utf-8"))
            report = {"events": events}
        else:
            report = analyzer.analyze(AnalysisRequest(
                video_path=str(video),
                fighter_a_box=[float(v) for v in fight["fighter_a"]],
                fighter_b_box=[float(v) for v in fight["fighter_b"]],
                ruleset="K1", fight_type="competition", round_count=1,
                start_seconds=start_seconds,
                round_duration_seconds=max(1.0, float(total) / max(1.0, fps) - start_seconds),
                job_id="negmine_%s" % key, profile_id=1, persist_result=False,
                output_dir=str(job_dir)))

        events = report.get("events") or []
        times = []
        for event in events:
            value = event.get("time_seconds", event.get("peak_time"))
            if value is not None:
                times.append(float(value))

        produced = windows_from_tracking(
            job_dir / "tracking.jsonl", window, times, arguments.margin, arguments.step)
        written = 0
        for side, frame, features in produced[:arguments.per_fight]:
            np.savez_compressed(
                destination / ("own_%s__%s%06d.npz" % (key, side, frame)),
                x=features, y=none_index, fight_id="own_%s" % key)
            written += 1
        per_fight[key] = written
        print("  %-14s %4d events avoided, %5d quiet windows, wrote %4d" % (
            key, len(times), len(produced), written))

    print()
    print("wrote %d negatives from %d fights to %s" % (
        sum(per_fight.values()), len(per_fight), destination))
    print("These are heuristic, not labelled. See the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
