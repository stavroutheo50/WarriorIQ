"""Turn a labelled pack into trainable sequences, from the run that produced it.

A label pack ships image crops - `(16, 96, 64)` uint8 filmstrips for a person to
look at - not the `(T, 102)` feature vectors the temporal trainer consumes. The
features have to be rebuilt from the analysis trace that the pack was cut from,
which is what this does: for each labelled candidate it takes the window of
`action_window` analysed frames ending at that candidate's peak, for that
candidate's fighter, and runs `core.action._feature_vector` over it.

Same function as inference uses, for the reason given in tools/import_boxingvi.py:
a training set whose features are computed by a copy of the inference code
silently drifts from it.

**This writes BINARY labels: `none` or `strike`.** That is not a shortcut, it is
what the labels support. Judging a jab from a cross off six frames is not
reliable even at 1080p, so the pack was labelled to strike/no-strike plus family
and the technique was deliberately left unsaid. Writing a technique here would
invent one.

Binary is also the axis that has actually failed. Two 17-class models trained on
BoxingVI scored over 99% on their own validation and then fired on 99.8% and 0%
of real windows respectively - they could not say `none`, or could say nothing
else. Getting that right is worth more than naming the punch.

`unsure` verdicts are dropped rather than pushed either way.

    python tools/build_labelled_sequences.py --labels tools/labels_athens_hd_claude.json \
        --job athens_hd --pack labelpack/athens_hd --out dataset/sequences_athens_hd
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.action import Sample, _feature_vector
from core.config import SETTINGS

BINARY = {"none": 0, "strike": 1}


def samples_for(job_dir: Path, side: str) -> list:
    """One Sample per analysed frame for this fighter, None where not held."""
    rows = [json.loads(line) for line
            in (job_dir / "tracking.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    out: list = []
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fight-id", default=None,
                        help="group name the trainer splits on; defaults to the job")
    arguments = parser.parse_args()

    payload = json.loads((PROJECT_ROOT / arguments.labels).read_text(encoding="utf-8"))
    verdicts: dict[int, str] = {}
    for entry in payload.get("labels", []):
        verdicts[int(entry["id"])] = str(entry["verdict"])
    for identifier in payload.get("quiet_all_none", []):
        verdicts.setdefault(int(identifier), "none")

    candidates = json.loads(
        (PROJECT_ROOT / arguments.pack / "index.json").read_text(encoding="utf-8"))["candidates"]
    job_dir = PROJECT_ROOT / "outputs" / arguments.job
    destination = PROJECT_ROOT / arguments.out
    destination.mkdir(parents=True, exist_ok=True)
    window = int(SETTINGS.action_window)
    group = arguments.fight_id or ("own_%s" % arguments.job)

    cache: dict[str, list] = {}
    written = 0
    counts: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()

    for candidate in candidates:
        verdict = verdicts.get(int(candidate["id"]))
        if verdict not in BINARY:
            skipped["unsure or unlabelled"] += 1
            continue
        side = str(candidate["fighter"])
        if side not in cache:
            cache[side] = samples_for(job_dir, side)
        samples = cache[side]
        times = np.array([s.time if s else np.nan for s in samples], dtype=np.float64)
        if not np.isfinite(times).any():
            skipped["no trace for this fighter"] += 1
            continue

        index = int(np.nanargmin(np.abs(times - float(candidate["peak_time"]))))
        chunk = samples[max(0, index - window + 1):index + 1]
        if len(chunk) < window or any(s is None for s in chunk):
            skipped["fighter not held for a whole window"] += 1
            continue
        features = np.stack([
            _feature_vector(s, chunk[i - 1] if i else None)
            for i, s in enumerate(chunk)]).astype(np.float32)
        if not np.isfinite(features).all():
            skipped["non-finite features"] += 1
            continue

        np.savez_compressed(
            destination / ("%s__%05d.npz" % (group, int(candidate["id"]))),
            x=features, y=np.int64(BINARY[verdict]), fight_id=group)
        counts[verdict] += 1
        written += 1

    print("wrote %d sequences to %s" % (written, destination))
    print("  group: %s" % group)
    for verdict, count in counts.most_common():
        print("     %-8s %4d" % (verdict, count))
    if skipped:
        print("  skipped:")
        for reason, count in skipped.most_common():
            print("     %-34s %4d" % (reason, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
