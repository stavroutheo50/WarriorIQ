"""Turn StrikeMetrics' kickboxing / Muay Thai fights into WarriorIQ training sequences.

StrikeMetrics (github.com/sswhitehat/StrikeMetrics---Kickboxing-AI-Tool, MIT)
publishes five professional fights with, per fight, a CVAT annotation file -
one box on the frame a strike lands, labelled Jab / Cross / Hook / Upper /
Body Kick / High Kick / Leg Kick, plus Fighter 1, Fighter 2 and referee boxes
- and MoveNet multipose keypoints for every frame (17 COCO joints, up to six
people, coordinates as fractions of a 1920x1080 frame).

    python tools/import_strikemetrics.py --root dataset/public/strikemetrics \\
        --out dataset/sequences_strikemetrics

## Training material only, never an answer key

**Not every strike is marked** (the authors say so), so an unmarked window may
well hold a strike: precision cannot be measured on it and negatives cannot be
mined from it. Every sequence is written with fight id ``strikemetrics_*``,
which core/strike_exam.py refuses as exam evidence.

## Only fights whose two files line up

The keypoint file and the annotation file are matched by frame number. That
was checked rather than assumed: on every frame people boxed a fighter, at
least 60% of some skeleton's confident joints must fall inside the box. On
2026-10-08 three fights matched on every boxed frame; Superbon v Petrosyan
matched 4 of 12 (its keypoint file ends 49 frames early) and Slugfest has only
two fighter boxes to check against. A fight under ``--min-match`` (0.9), or
with fewer than ``--min-boxes`` (10) boxes to check, is skipped and says why.

## How a window is cut

* **Who threw it:** the skeleton with most joints inside the strike box on its
  frame (at least three).
* **Side:** for hooks, uppercuts and kicks, whichever wrist or ankle travelled
  further over the window - the method tools/import_boxingvi.py uses.
  Jab and cross keep their own class. Body, high and leg kicks are round kicks.
* **Frames:** ``--window`` (12) frames sampled at ``--sample-fps`` (12) from
  the source's ``--fps`` (30, assumed: the files do not record it), ending
  ``--after`` (3) samples past the strike frame, so the clip carries the
  wind-up, the strike and the start of the return like the windows the
  analysis classifies. The striker is followed frame to frame as the nearest
  skeleton; a jump of more than ``--max-jump`` (0.15 of the frame width)
  drops the window.

Features come from ``core.action._feature_vector``, never a copy of it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.action import Sample, _feature_vector  # noqa: E402
from core.temporal_model import ACTION_CLASSES  # noqa: E402

# Annotation file -> keypoint file, as published.
FIGHTS = {
    "HaggertyvNaitoAnnotations.xml": "Trimmed_Haggerty_vs_Naito_No_CommentaryKeypoints.csv",
    "HaggertyvMongkolpetchAnnotations.xml": "Trimmed_Haggerty_vs_Mongkolpetch_No_CommentaryKeypoints.csv",
    "RodtangvGoncalvesAnnotations.xml": "Trimmed_Rodtang_vs_Goncalves_No_CommentaryKeypoints.csv",
    "SuperbonvPetrysan.xml": "SuperbonvPetrosyanKeypoints.csv",
    "slugfestannotations.xml": "SlugfestVideo.csv",
}
DIRECT = {"Jab": "jab", "Cross": "cross"}
SIDED = {"Hook": ("left_hook", "right_hook"), "Upper": ("left_uppercut", "right_uppercut"),
         "Body Kick": ("left_round_kick", "right_round_kick"),
         "High Kick": ("left_round_kick", "right_round_kick"),
         "Leg Kick": ("left_round_kick", "right_round_kick")}
L_WRIST, R_WRIST, L_ANKLE, R_ANKLE = 9, 10, 15, 16
JOINT_CONF = 0.3


def read_annotations(path: Path):
    """(width, height, fighter boxes by frame, [(frame, label, box)])."""
    text = path.read_text(encoding="utf-8")
    size = re.search(r"<original_size>\s*<width>(\d+)</width>\s*<height>(\d+)</height>", text)
    width, height = (int(size.group(1)), int(size.group(2))) if size else (1920, 1080)
    fighters: dict[int, list] = collections.defaultdict(list)
    strikes = []
    for track in re.finditer(r'<track id="\d+" label="([^"]+)"[^>]*>(.*?)</track>', text, re.S):
        label = track.group(1)
        for box in re.finditer(r'<box frame="(\d+)"[^>]*outside="0"[^>]*xtl="([\d.]+)" ytl="([\d.]+)" '
                               r'xbr="([\d.]+)" ybr="([\d.]+)"', track.group(2)):
            frame, values = int(box.group(1)), tuple(float(v) for v in box.groups()[1:])
            if label in ("F1", "F2"):
                fighters[frame].append(values)
            elif label in DIRECT or label in SIDED:
                strikes.append((frame, label, values))
    return width, height, fighters, sorted(strikes)


def read_keypoints(path: Path, width: int, height: int) -> dict[int, list[np.ndarray]]:
    """frame -> list of (17, 3) arrays in pixels, one per detected person."""
    people: dict[int, list] = collections.defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            joints = np.array([[float(row[f"keypoint_{i}_x"]) * width, float(row[f"keypoint_{i}_y"]) * height,
                                float(row[f"keypoint_{i}_confidence"])] for i in range(17)], dtype=np.float32)
            people[int(row["frame_id"])].append(joints)
    return people


def _inside(joints: np.ndarray, box) -> tuple[int, int]:
    seen = joints[joints[:, 2] > JOINT_CONF]
    hits = int(((seen[:, 0] >= box[0]) & (seen[:, 0] <= box[2]) & (seen[:, 1] >= box[1])
                & (seen[:, 1] <= box[3])).sum())
    return hits, len(seen)


def alignment(fighters, people) -> tuple[int, int]:
    """(fighter boxes containing a skeleton, fighter boxes checked)."""
    good = total = 0
    for frame, boxes in fighters.items():
        for box in boxes:
            total += 1
            if any(n and hits / n >= 0.6 for hits, n in (_inside(p, box) for p in people.get(frame, []))):
                good += 1
    return good, total


def _centre(joints: np.ndarray) -> np.ndarray | None:
    seen = joints[joints[:, 2] > JOINT_CONF]
    return seen[:, :2].mean(axis=0) if len(seen) >= 4 else None


def follow(people, frames, start: np.ndarray, start_frame: int, max_jump: float) -> list | None:
    """The striker on each of ``frames``, linked frame to frame by nearest centre."""
    chosen: dict[int, np.ndarray] = {start_frame: start}

    def walk(order):
        previous = _centre(start)
        for frame in order:
            options = [(p, _centre(p)) for p in people.get(frame, [])]
            options = [(p, c) for p, c in options if c is not None]
            if previous is None or not options:
                return False
            joints, centre = min(options, key=lambda item: float(np.linalg.norm(item[1] - previous)))
            if float(np.linalg.norm(centre - previous)) > max_jump:
                return False
            chosen[frame], previous = joints, centre
        return True

    before = sorted((f for f in frames if f < start_frame), reverse=True)
    after = sorted(f for f in frames if f > start_frame)
    if not walk(before) or not walk(after):
        return None
    return [chosen[f] for f in frames]


def _travel(track: list[np.ndarray], joint: int) -> float:
    points = np.array([j[joint, :2] for j in track if j[joint, 2] > JOINT_CONF])
    return float(np.abs(np.diff(points, axis=0)).sum()) if len(points) >= 2 else 0.0


def features(track: list[np.ndarray], step_seconds: float) -> np.ndarray:
    samples = []
    for order, joints in enumerate(track):
        seen = joints[joints[:, 2] > JOINT_CONF]
        box = (np.array([seen[:, 0].min(), seen[:, 1].min(), seen[:, 0].max(), seen[:, 1].max()], np.float32)
               if len(seen) >= 4 else np.array([0, 0, 1, 1], np.float32))
        samples.append(Sample(frame=order, time=order * step_seconds, round_number=None, box=box,
                              keypoints=joints[:, :2].copy(), conf=joints[:, 2].copy(),
                              opponent_box=None, opponent_keypoints=None, opponent_conf=None,
                              identity_confidence=1.0, opponent_identity_confidence=1.0))
    return np.stack([_feature_vector(s, samples[i - 1] if i else None)
                     for i, s in enumerate(samples)]).astype(np.float32)


def import_fight(annotation: Path, keypoints: Path, out: Path, args) -> tuple[collections.Counter, str | None]:
    width, height, fighters, strikes = read_annotations(annotation)
    people = read_keypoints(keypoints, width, height)
    good, total = alignment(fighters, people)
    if total < args.min_boxes:
        return collections.Counter(), f"only {total} fighter boxes to check the frame alignment against"
    if good / total < args.min_match:
        return collections.Counter(), f"frames do not line up: {good} of {total} fighter boxes hold a skeleton"
    name = "strikemetrics_" + re.sub(r"[^a-z0-9]+", "_", annotation.stem.lower()).strip("_")
    step = max(1, round(args.fps / args.sample_fps))
    made = collections.Counter()
    for index, (frame, label, box) in enumerate(strikes):
        candidates = [(_inside(p, box)[0], p) for p in people.get(frame, [])]
        candidates = [c for c in candidates if c[0] >= 3]
        if not candidates:
            made["skipped: no skeleton in the strike box"] += 1
            continue
        striker = max(candidates, key=lambda c: c[0])[1]
        frames = [frame + (i - (args.window - 1 - args.after)) * step for i in range(args.window)]
        track = follow(people, frames, striker, frame, args.max_jump * width)
        if track is None:
            made["skipped: striker not followed through the window"] += 1
            continue
        if label in DIRECT:
            action = DIRECT[label]
        else:
            left, right = SIDED[label]
            joints = (L_WRIST, R_WRIST) if "kick" not in left else (L_ANKLE, R_ANKLE)
            action = left if _travel(track, joints[0]) >= _travel(track, joints[1]) else right
        x = features(track, step / args.fps)
        if not np.isfinite(x).all():
            made["skipped: non-finite features"] += 1
            continue
        np.savez_compressed(out / f"{name}__{index:05d}.npz", x=x,
                            y=np.int64(ACTION_CLASSES.index(action)), fight_id=name)
        made[action] += 1
    return made, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "dataset" / "public" / "strikemetrics")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "sequences_strikemetrics")
    parser.add_argument("--fps", type=float, default=30.0, help="source frame rate (assumed; not recorded)")
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--after", type=int, default=3)
    parser.add_argument("--max-jump", type=float, default=0.15)
    parser.add_argument("--min-match", type=float, default=0.9)
    parser.add_argument("--min-boxes", type=int, default=10)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    totals = collections.Counter()
    for annotation_name, keypoint_name in FIGHTS.items():
        annotation = args.root / "Annotations" / annotation_name
        keypoints = args.root / "Keypoint Files" / keypoint_name
        if not annotation.exists() or not keypoints.exists():
            print(f"{annotation_name}: missing from {args.root}")
            continue
        made, refused = import_fight(annotation, keypoints, args.out, args)
        if refused:
            print(f"{annotation_name}: skipped - {refused}")
            continue
        print(f"{annotation_name}: {dict(made)}")
        totals.update(made)
    print(f"Total: {dict(totals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
