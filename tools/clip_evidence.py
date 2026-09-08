"""Measure what the limbs actually did around each label-pack clip.

A second opinion, independent of the eye. Judging a clip from six thumbnails
alone is how the first dataset ended up with a quarter of its labels inverted;
judging it from the pose alone is what the detector already does, and it is
wrong about a third of the time. Requiring the two to agree catches the cases
where either on its own would be confident and wrong.

Everything is in body lengths, not pixels, so a fighter at the far side of the
mat is measured the same as one in the foreground. Path length, not
displacement: a kick's foot goes out and comes back, so where it ends up says
almost nothing about whether it was thrown. The same lesson core/action.py
learned the hard way.

    tools/clip_evidence.py --job fam3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.action import L_ANKLE, L_KNEE, L_WRIST, R_ANKLE, R_KNEE, R_WRIST

# Just over half a second either side of the moment: long enough to contain a
# strike from chamber to retraction, short enough not to swallow the next one.
WINDOW_SECONDS = 0.6
MIN_CONFIDENCE = 0.30


def _load_tracking(job: str) -> dict[int, dict]:
    records = {}
    path = PROJECT_ROOT / "outputs" / job / "tracking.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("source_frame") is not None:
            records[int(record["source_frame"])] = record
    return records


def _point(keypoints, conf, index: int):
    if keypoints is None or index >= len(keypoints):
        return None
    if conf is not None and index < len(conf) and float(conf[index]) < MIN_CONFIDENCE:
        return None
    x, y = float(keypoints[index][0]), float(keypoints[index][1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return np.array([x, y], dtype=np.float32)


def _travel(samples, index: int, scale: float) -> float:
    """Total path length of one joint over the window, in body lengths."""
    points = [_point(kp, cf, index) for kp, cf in samples]
    points = [p for p in points if p is not None]
    if len(points) < 2:
        return 0.0
    total = sum(float(np.linalg.norm(points[i + 1] - points[i])) for i in range(len(points) - 1))
    return total / max(1.0, scale)


def evidence_for(job: str, fighter: str, peak_frame: int, fps: float = 30.0) -> dict:
    tracking = _load_tracking(job)
    span = max(2, int(fps * WINDOW_SECONDS))
    other = "B" if fighter == "A" else "A"

    samples, heights, own_boxes, opponent_boxes = [], [], [], []
    for offset in range(-span, span + 1):
        record = tracking.get(peak_frame + offset)
        if not record:
            continue
        own = (record.get("fighter_%s" % fighter) or {}).get("observation") or {}
        if own.get("keypoints"):
            samples.append((own["keypoints"], own.get("keypoint_conf")))
        if own.get("box"):
            box = own["box"]
            heights.append(max(1.0, box[3] - box[1]))
            own_boxes.append(box)
        opp = (record.get("fighter_%s" % other) or {}).get("observation") or {}
        if opp.get("box"):
            opponent_boxes.append(opp["box"])

    if not samples or not heights:
        return {"usable": False, "reason": "no pose in this window"}

    scale = float(np.median(heights))
    hand = max(_travel(samples, L_WRIST, scale), _travel(samples, R_WRIST, scale))
    foot = max(_travel(samples, L_ANKLE, scale), _travel(samples, R_ANKLE, scale))
    knee = max(_travel(samples, L_KNEE, scale), _travel(samples, R_KNEE, scale))

    gap = None
    if own_boxes and opponent_boxes:
        own_c = np.mean([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in own_boxes], axis=0)
        opp_c = np.mean([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in opponent_boxes], axis=0)
        gap = float(np.linalg.norm(own_c - opp_c) / scale)

    # What the movement alone suggests, before anybody looks at the picture.
    if max(hand, foot, knee) < 0.35:
        reading = "nothing"
    elif foot > 1.6 * hand or knee > 1.6 * hand:
        reading = "leg"
    elif hand > 1.6 * max(foot, knee):
        reading = "hand"
    else:
        reading = "unclear"

    return {
        "usable": True,
        "hand_travel": round(hand, 3),
        "foot_travel": round(foot, 3),
        "knee_travel": round(knee, 3),
        "opponent_gap": None if gap is None else round(gap, 2),
        "opponent_seen": len(opponent_boxes),
        "pose_samples": len(samples),
        "reading": reading,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Limb movement around each clip.")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    index = json.loads((PROJECT_ROOT / "labelpack" / args.job / "index.json").read_text(encoding="utf-8"))
    out = []
    for item in index["candidates"]:
        ev = evidence_for(args.job, item["fighter"], int(item["peak_frame"]))
        out.append({"id": item["id"], "proposed": item["proposed"], "source": item["source"], **ev})
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
