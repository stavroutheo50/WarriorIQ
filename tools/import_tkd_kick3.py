"""Turn TKD-Kick3's taekwondo kicks into WarriorIQ training sequences.

Every kick class in ACTION_CLASSES has zero data. BoxingVI supplied punches and
nothing else - no kicks, no knees, no elbows - which leaves eight of the
seventeen classes empty on a product whose main sport is kickboxing.

TKD-Kick3 (Zenodo 10.5281/zenodo.20390892, **CC BY 4.0**, so usable
commercially with attribution) is 765 pose sequences of front, roundhouse and
axe kicks, de-identified to keypoints only. It is the first kick data this
project has had.

## Three things that made it usable

**The keypoints map onto COCO-17 exactly.** Its `keypoint_mapping.csv` lists 13
joints - nose, shoulders, elbows, wrists, hips, knees, ankles - which are COCO
indices 0 and 5 through 16. Only the eyes and ears are missing, and they are
left invisible. That costs little here: measured on BoxingVI, this project's own
footage runs a median keypoint visibility of 0.36, so a partial upper body is
the normal case rather than a degraded one.

**It records `fps` per sequence.** The BoxingVI import had to assume 30 and say
so, because velocity is 51 of the 102 feature dimensions and nothing in that
dataset stated a frame rate. Here the timebase is measured, not guessed - 30 fps
for most, 60 for 102 of them, and a few at 29.x.

**The kicking leg is recoverable, where the punching hand was not.** This is
the interesting part, because the equivalent test failed for hooks and
uppercuts and 2,104 clips were dropped over it.

The first attempt used the same feature as the punch version, total ankle path
length, and it failed the same way - median margin 0.35, only 4% of clips above
0.6. That is not the side being unreadable; it is the feature being wrong. A
kick pivots the support leg and rotates the body, so both ankles travel.

What separates them is height. The kicking foot rises toward or above the hip;
the support foot never leaves the floor. Scoring on peak foot lift relative to
the hip line instead:

    margin   >=0.3   >=0.6   >=0.9      median
    punches   ----    ----    ----        0.44   (unusable)
    kicks     74%     71%     68%         1.00

The distribution is bimodal rather than merely better: the margin saturates at
1.0 whenever the support foot stays below the hip, which is most of the time.
So about 70% of clips name their kicking leg unambiguously, and `--min-margin`
drops the rest rather than guessing.

## What this does not fix

**Axe kicks have no class.** ACTION_CLASSES has round, front, push and knee -
no axe - so 285 of the 765 sequences are dropped. Adding a class would change
the model's output space and the release gate's required-class list, which is
not a decision an importer should make.

**It is still out-of-domain.** These are staged taekwondo kicks from smartphone
video with MediaPipe poses, not kickboxing under tournament lighting with a
YOLO pose model. The BoxingVI run proved what that costs: a model trained on
another dataset's positives learns to recognise the dataset. This is a
foundation for the day in-domain kick labels exist, not a substitute for them.

    python tools/import_tkd_kick3.py --source dataset/tkd_kick3/TKD-Kick3.zip
"""
from __future__ import annotations

import argparse
import collections
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.action import Sample, _feature_vector
from core.config import SETTINGS
from core.temporal_model import ACTION_CLASSES

# This dataset's 13-point order -> COCO-17. Eyes and ears have no source.
TO_COCO = {0: 0, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10,
           7: 11, 8: 12, 9: 13, 10: 14, 11: 15, 12: 16}
L_ANKLE, R_ANKLE, L_HIP, R_HIP = 11, 12, 7, 8

SIDED = {"front": ("left_front_kick", "right_front_kick"),
         "roundhouse": ("left_round_kick", "right_round_kick")}


def kicking_side(kpts: np.ndarray) -> tuple[str | None, float]:
    """Which leg kicked, and how confident that is, from peak foot lift.

    y grows downward in normalised image coordinates, so a raised foot has a
    smaller y than the hip line. The support foot stays below it, which is what
    makes the margin saturate rather than merely lean.
    """
    hip = np.nanmean(kpts[:, [L_HIP, R_HIP], 1], axis=1)
    lifts = {}
    for name, joint in (("left", L_ANKLE), ("right", R_ANKLE)):
        lift = hip - kpts[:, joint, 1]
        if not np.isfinite(lift).any():
            return None, 0.0
        lifts[name] = float(np.nanmax(lift))
    total = abs(lifts["left"]) + abs(lifts["right"])
    if total <= 0:
        return None, 0.0
    margin = abs(lifts["left"] - lifts["right"]) / total
    return (max(lifts, key=lifts.__getitem__), margin)


def to_samples(kpts: np.ndarray, vis: np.ndarray, fps: float,
               boxes: np.ndarray | None) -> list[Sample]:
    """COCO-17 Samples in pixels.

    The archive holds two layouts and they differ in more than field names:
    752 sequences store PIXEL coordinates with a per-frame `bbox_xyxy`, and 13
    store NORMALISED coordinates with a `size`. Callers scale the second kind
    before this point, so everything here is pixels; `boxes` is the dataset's
    own box where it has one, and the visible keypoint extent where it does not.
    """
    step = 1.0 / max(1e-6, fps)
    samples: list[Sample] = []
    for order in range(kpts.shape[0]):
        points = np.zeros((17, 2), dtype=np.float32)
        confidence = np.zeros(17, dtype=np.float32)
        for source, target in TO_COCO.items():
            x, y = kpts[order, source, 0], kpts[order, source, 1]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            points[target] = (x, y)
            value = vis[order, source] if vis is not None else 1.0
            confidence[target] = float(value) if np.isfinite(value) else 1.0
        seen = confidence > 0
        if boxes is not None and np.isfinite(boxes[order]).all():
            box = np.asarray(boxes[order], dtype=np.float32)
        elif seen.sum() < 4:
            box = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        else:
            visible = points[seen]
            box = np.asarray([visible[:, 0].min(), visible[:, 1].min(),
                              visible[:, 0].max(), visible[:, 1].max()], dtype=np.float32)
        samples.append(Sample(
            frame=order, time=order * step, round_number=None,
            box=box, keypoints=points, conf=confidence,
            opponent_box=None, opponent_keypoints=None, opponent_conf=None,
            identity_confidence=1.0, opponent_identity_confidence=1.0))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="dataset/tkd_kick3/TKD-Kick3.zip")
    parser.add_argument("--out", default="dataset/sequences_tkd_kick3")
    parser.add_argument("--min-margin", type=float, default=0.6,
                        help="how decisively one foot must out-lift the other")
    arguments = parser.parse_args()

    archive = zipfile.ZipFile(PROJECT_ROOT / arguments.source)
    destination = PROJECT_ROOT / arguments.out
    destination.mkdir(parents=True, exist_ok=True)
    window = int(SETTINGS.action_window)

    per_class: collections.Counter = collections.Counter()
    per_group: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()
    written = 0

    for name in sorted(n for n in archive.namelist() if n.endswith(".npz")):
        data = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
        label = str(data["class_label"])
        if label not in SIDED:
            skipped["axe kick has no class in ACTION_CLASSES"] += 1
            continue

        kpts = np.asarray(data["kpts"], dtype=np.float32)
        vis = np.asarray(data["vis"], dtype=np.float32)
        if bool(np.asarray(data["invalid_for_scoring"])) if "invalid_for_scoring" in data else False:
            skipped["dataset marked it invalid for scoring"] += 1
            continue
        if kpts.ndim != 3 or kpts.shape[1] != 13 or kpts.shape[0] < 2:
            skipped["odd sequence shape"] += 1
            continue

        side, margin = kicking_side(kpts)
        if side is None or margin < arguments.min_margin:
            skipped["kicking leg not decisive"] += 1
            continue

        if "bbox_xyxy" in data:
            boxes = np.asarray(data["bbox_xyxy"], dtype=np.float32)
        else:
            # Normalised layout: scale to pixels so both kinds meet here in the
            # same units, since _feature_vector divides joints by the box.
            height, width = (int(v) for v in np.asarray(data["size"], dtype=np.int32)[:2])
            kpts = kpts.copy()
            kpts[:, :, 0] *= width
            kpts[:, :, 1] *= height
            boxes = None
        fps = float(data["fps"]) or 30.0
        samples = to_samples(kpts, vis, fps, boxes)

        # Even indices across the clip, so the window covers the whole kick
        # rather than its first half second.
        picks = np.linspace(0, len(samples) - 1, window).round().astype(int)
        chosen = [samples[i] for i in picks]
        features = np.stack([
            _feature_vector(s, chosen[i - 1] if i else None)
            for i, s in enumerate(chosen)]).astype(np.float32)
        if not np.isfinite(features).all():
            skipped["non-finite features"] += 1
            continue

        left_name, right_name = SIDED[label]
        action = left_name if side == "left" else right_name
        # The split folder is the only subject grouping this dataset exposes,
        # and the trainer must not put one person's clips on both sides.
        group = "tkd_%s" % str(data["split"])
        np.savez_compressed(
            destination / ("tkd_%s__%s" % (group, Path(name).stem + ".npz")),
            x=features, y=np.int64(ACTION_CLASSES.index(action)), fight_id=group)
        per_class[action] += 1
        per_group[group] += 1
        written += 1

    print("wrote %d sequences to %s" % (written, destination))
    print("  groups: %s" % dict(per_group))
    for action, count in per_class.most_common():
        print("     %-18s %4d" % (action, count))
    if skipped:
        print("  skipped:")
        for reason, count in skipped.most_common():
            print("     %-40s %4d" % (reason, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
