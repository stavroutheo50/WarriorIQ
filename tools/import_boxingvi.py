"""Turn BoxingVI's skeletons into WarriorIQ training sequences.

BoxingVI (arXiv 2511.16524, github.com/Bikudebug/BoxingVI) publishes punch clips
cut from sparring footage with AlphaPose COCO keypoints already extracted. Nine
of its ten skeleton files are 25-frame clips aligned one-to-one with a labelled
spreadsheet row, which is 4,766 labelled punches - against the 142 sequences
this project has of its own, the reason `warrioriq_temporal_best.pt` does not
exist and `core/action.py` still classifies by hand-written geometry.

**The features are built by importing `core.action._feature_vector`, not by
reimplementing it.** A training set whose features are computed by a copy of the
inference code is a training set that silently drifts from it, and this project
has already lost an afternoon to an A/B arm that was not actually switched on.
Same function, same 102 dims, or the model learns a different space than it is
asked to predict in.

## What the data needed before it could be used

**The column order is not stable.** Several sheets put a helper formula where
the class belongs, so reading `row[4]` reports "=A2-B2" as a punch. Every row is
scanned for a cell whose text is a known class instead.

**V6 is a different layout** - 46,497 per-frame skeletons with a confidence
channel, not pre-cut clips, and its 685 labels do not align with its rows. It
is skipped, with a count, rather than silently half-imported.

**Lead and rear are stance-relative; this project's classes are not, and the
gap cannot be closed from this data. Hooks and uppercuts are excluded.**

ACTION_CLASSES has `left_hook`/`right_hook`, not lead/rear, and no southpaw flag
exists anywhere, so the side has to come off the skeleton. Reading it as "the
wrist that travelled furthest threw the punch" was tried and measured, and it
does not hold up.

The test needs no ground truth: within one video, lead and rear punches must be
thrown with OPPOSITE hands, so `|lead_left_share - rear_left_share|` should be
near 1.0. Measured over the nine usable videos, at every confidence margin:

    margin   kept of 2104   separation
    0.0          2104          0.44
    0.2          1304          0.48
    0.3           941          0.44
    0.4           594          0.51     (only 3 videos still qualify)
    0.6           161          0.66     (1 video)

0.44 is about 72/28 - so roughly a quarter to a third of sided labels would be
wrong, and gating on margin does not improve it, it only shrinks the sample. V4
is the clearest failure: it assigns lead AND rear to the same hand, which is
impossible. The likely cause is AlphaPose swapping left and right limbs when a
fighter turns away, which is a known failure of pose estimation and not
something a threshold downstream can repair.

This project's own history says what to do here: many features all scoring
around 0.70 means the labels are wrong, and a confident wrong answer is worse
than no answer. So `jab` and `cross` - which have no side variants in
ACTION_CLASSES and map straight through with no inference at all - are imported,
and 2,104 hook and uppercut clips are left on the floor until something can
resolve stance. `--sided` imports them anyway for anyone who wants to
experiment; it is off by default on purpose.

## The assumption to be aware of

The source frame rate is not published, and velocity is 51 of the 102 dims.
`--fps` (default 30) plus the resample to `action_window` frames sets the
timebase; the default makes a clip's sampling interval close to what stride 3
produces at inference, which is where this model will actually run. If the real
footage is 25 or 60 fps the velocity half is scaled wrong, consistently - which
is a thing to test for, not a thing this file can detect.

**This buys a model that works on BoxingVI. Whether it transfers to tournament
footage at 480x220 is a separate question that needs WarriorIQ's own labels, and
nothing here answers it.**

Licence: the BoxingVI repository states none, and its clips are cut from
YouTube. Fine for measuring whether the approach works; get that settled before
anything trained on it ships.

    python tools/import_boxingvi.py --source dataset/boxingvi --out dataset/sequences_boxingvi
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.action import Sample, _feature_vector
from core.config import SETTINGS
from core.temporal_model import ACTION_CLASSES

L_WRIST, R_WRIST = 9, 10

# Lead/rear resolves to a side per clip; jab and cross have no side variants.
DIRECT = {"jab": "jab", "cross": "cross"}
SIDED = {
    "lead hook": ("left_hook", "right_hook"),
    "rear hook": ("left_hook", "right_hook"),
    "lead uppercut": ("left_uppercut", "right_uppercut"),
    "rear uppercut": ("left_uppercut", "right_uppercut"),
}
KNOWN = set(DIRECT) | set(SIDED)


def read_labels(path: Path) -> list[str]:
    """Every row's class, found by matching text rather than by column index.

    **Not `read_only=True`.** That mode reports the sheet's declared dimension
    rather than its used range, which on V1 and V3 is 1,048,575 rows of nothing.
    The alignment check below then saw "1866 clips but 1048575 rows" and threw
    away the two largest files in the dataset - 2,681 of its 4,766 clips, over
    half, silently and with a message that looked like a data problem rather
    than a reader problem.

    Trailing rows are trimmed for the same reason at smaller scale: V2 carries
    seven blank rows after its last punch, which is not a misalignment.
    """
    import openpyxl

    sheet = openpyxl.load_workbook(path).active
    labels: list[str] = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if not row or all(cell is None for cell in row):
            labels.append("")
            continue
        found = None
        for cell in row:
            if isinstance(cell, str) and cell.strip().lower() in KNOWN:
                found = cell.strip().lower()
                break
        labels.append(found or "")
    while labels and not labels[-1]:
        labels.pop()
    return labels


def resample(clip: np.ndarray, frames: int) -> np.ndarray:
    """(T, 17, 2) -> (frames, 17, 2) by picking evenly spaced source frames."""
    index = np.linspace(0, len(clip) - 1, frames).round().astype(int)
    return clip[index]


def throwing_side(clip: np.ndarray) -> str | None:
    """Which wrist travelled furthest: that is the arm that threw the punch."""
    best: dict[str, float] = {}
    for name, joint in (("left", L_WRIST), ("right", R_WRIST)):
        track = clip[:, joint, :2]
        valid = (track[:, 0] > 0) & (track[:, 1] > 0)
        if valid.sum() < 2:
            continue
        points = track[valid]
        best[name] = float(np.abs(np.diff(points, axis=0)).sum())
    if len(best) < 2:
        return next(iter(best), None)
    return max(best, key=best.__getitem__)


def to_samples(clip: np.ndarray, fps: float, source_frames: int) -> list[Sample]:
    """One Sample per resampled frame, with a box derived from the keypoints.

    BoxingVI carries no boxes. `_feature_vector` normalises each joint against
    the box, so the keypoint extent is used - which is what a detector box
    approximates anyway, and being consistently derived matters more here than
    matching any particular detector's padding.
    """
    step = (source_frames / max(1, len(clip))) / max(1e-6, fps)
    samples: list[Sample] = []
    for order, frame in enumerate(clip):
        points = np.asarray(frame[:, :2], dtype=np.float32)
        valid = (points[:, 0] > 0) & (points[:, 1] > 0)
        if valid.sum() < 4:
            box = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        else:
            seen = points[valid]
            box = np.asarray([seen[:, 0].min(), seen[:, 1].min(),
                              seen[:, 0].max(), seen[:, 1].max()], dtype=np.float32)
        confidence = valid.astype(np.float32)
        samples.append(Sample(
            frame=order, time=order * step, round_number=None,
            box=box, keypoints=points, conf=confidence,
            opponent_box=None, opponent_keypoints=None, opponent_conf=None,
            identity_confidence=1.0, opponent_identity_confidence=1.0))
    return samples


def labelled_spans(path: Path) -> list[tuple[int, int]]:
    """(start, end) frame ranges that contain a punch, from a per-frame sheet."""
    import openpyxl

    sheet = openpyxl.load_workbook(path).active
    spans: list[tuple[int, int]] = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if not row:
            continue
        numbers = [int(c) for c in row if isinstance(c, (int, float)) and not isinstance(c, bool)]
        if len(numbers) >= 2:
            first, second = numbers[0], numbers[1]
            spans.append((min(first, second), max(first, second)))
    return sorted(spans)


def mine_negatives(array: np.ndarray, sheet: Path, destination: Path,
                   name: str, window: int, fps: float,
                   source_frames: int = 25, margin: int = 12, limit: int = 1200) -> int:
    """Cut `none` sequences from the footage between labelled punches.

    A margin is left either side of every labelled span, because the frames
    just before a punch are the wind-up and the ones just after are the
    retraction - both are part of the action even when the annotation's
    boundary says otherwise, and feeding them in as "nothing happening" would
    teach the model that a punch is not a punch.

    Windows are spaced a full clip apart so no two negatives overlap.
    """
    spans = labelled_spans(sheet)
    total = int(array.shape[0])
    blocked = np.zeros(total + source_frames, dtype=bool)
    for start, end in spans:
        low = max(0, start - margin)
        high = min(len(blocked), end + margin + 1)
        blocked[low:high] = True

    made = 0
    position = 0
    while position + source_frames < total and made < limit:
        if blocked[position:position + source_frames].any():
            position += 1
            continue
        clip = np.asarray(array[position:position + source_frames, 0, :, :2], dtype=np.float32)
        visible = (clip[:, :, 0] > 0) & (clip[:, :, 1] > 0)
        # A window where the pose is mostly missing is not a negative example,
        # it is an absent one, and it would teach the model that empty means no.
        if visible.mean() < 0.5:
            position += source_frames
            continue
        resampled = resample(clip, window)
        samples = to_samples(resampled, fps, source_frames)
        features = np.stack([
            _feature_vector(sample, samples[i - 1] if i else None)
            for i, sample in enumerate(samples)
        ]).astype(np.float32)
        if np.isfinite(features).all():
            np.savez_compressed(
                destination / ("boxingvi_%s__neg%05d.npz" % (name, position)),
                x=features,
                y=np.int64(ACTION_CLASSES.index("none")),
                fight_id="boxingvi_%s" % name,
            )
            made += 1
        position += source_frames
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="dataset/boxingvi")
    parser.add_argument("--out", default="dataset/sequences_boxingvi")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="assumed source frame rate; sets the velocity timebase")
    parser.add_argument("--sided", action="store_true",
                        help="also import hooks and uppercuts, whose side is inferred "
                             "and measured unreliable - see the module docstring")
    arguments = parser.parse_args()

    source = PROJECT_ROOT / arguments.source
    destination = PROJECT_ROOT / arguments.out
    destination.mkdir(parents=True, exist_ok=True)
    window = int(SETTINGS.action_window)

    written = 0
    per_class: collections.Counter = collections.Counter()
    per_fight: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()

    for npy in sorted((source / "Skeleton_data").glob("V*.npy")):
        name = npy.stem
        sheet = source / "Annotation_files" / ("%s.xlsx" % name)
        if not sheet.exists():
            skipped["no annotation sheet"] += 1
            continue

        array = np.load(npy, allow_pickle=True)
        if array.ndim != 4 or array.shape[1] < 2:
            # Per-frame file. It carries no extra punches this importer can cut
            # reliably, but the footage BETWEEN its labelled punches is the one
            # thing the clip files cannot provide: negatives. The trainer
            # refuses to run without them, and it is right to - a classifier
            # shown nothing but punches learns to call everything a punch,
            # while at inference most windows are two people circling.
            made = mine_negatives(array, sheet, destination, name, window, arguments.fps)
            per_class["none"] += made
            per_fight["boxingvi_%s" % name] += made
            written += made
            print("  %-4s %5d frames -> %5d negatives (none)" % (name, array.shape[0], made))
            continue

        labels = read_labels(sheet)
        if len(labels) != array.shape[0]:
            skipped["misaligned (%s)" % name] += 1
            print("  %-4s SKIPPED - %d clips but %d rows" % (name, array.shape[0], len(labels)))
            continue

        kept = 0
        for index, label in enumerate(labels):
            if label not in KNOWN:
                skipped["unlabelled row"] += 1
                continue
            clip = np.asarray(array[index], dtype=np.float32)
            if clip.ndim != 3 or clip.shape[1] != 17:
                skipped["odd clip shape"] += 1
                continue

            # Some clips carry no skeleton at all - AlphaPose found nobody, and
            # every coordinate is zero, while the spreadsheet still calls them a
            # jab. Two such clips exist (V3 row 295, V5 row 321) and they are
            # byte-identical, which is how they were found: the dataset audit
            # counts duplicate fingerprints and the trainer refuses to start
            # while any exist.
            #
            # Deduplicating would have satisfied the gate and kept the problem.
            # A labelled punch with no visible fighter teaches the model that
            # an empty window is a jab, which is the failure this project keeps
            # meeting from the other direction - a confident answer built on no
            # evidence. So they are dropped on visibility, not on sameness.
            #
            # **The threshold is low on purpose.** Sparse skeletons are this
            # dataset's normal state, not its failure state - measured over all
            # 4,766 clips, median visibility is 0.36 and the lower quartile is
            # 0.28, which is roughly an upper body with the legs missing:
            #
            #     >= 0.05   4757 clips   99.8%
            #     >= 0.20   4695         98.5%
            #     >= 0.30   3289         69.0%      <- a first guess at 0.3
            #     >= 0.50    641         13.4%
            #
            # A 0.3 cut threw away 1,477 perfectly ordinary clips, a third of
            # the dataset, for being typical of it. Only 5 clips are entirely
            # empty and 9 sit below 0.05, so that is where the real garbage
            # ends and this is where the line goes.
            visible = (clip[:, :, 0] > 0) & (clip[:, :, 1] > 0)
            if visible.mean() < 0.05:
                skipped["clip has almost no visible skeleton"] += 1
                continue

            if label in DIRECT:
                action = DIRECT[label]
            elif not arguments.sided:
                skipped["sided class (side is not recoverable)"] += 1
                continue
            else:
                side = throwing_side(clip)
                if side is None:
                    skipped["no wrist to read side from"] += 1
                    continue
                left_name, right_name = SIDED[label]
                action = left_name if side == "left" else right_name

            resampled = resample(clip, window)
            samples = to_samples(resampled, arguments.fps, clip.shape[0])
            features = np.stack([
                _feature_vector(sample, samples[i - 1] if i else None)
                for i, sample in enumerate(samples)
            ]).astype(np.float32)

            if not np.isfinite(features).all():
                skipped["non-finite features"] += 1
                continue

            np.savez_compressed(
                destination / ("boxingvi_%s__%05d.npz" % (name, index)),
                x=features,
                y=np.int64(ACTION_CLASSES.index(action)),
                fight_id="boxingvi_%s" % name,
            )
            per_class[action] += 1
            per_fight["boxingvi_%s" % name] += 1
            written += 1
            kept += 1
        print("  %-4s %5d clips -> %5d sequences" % (name, array.shape[0], kept))

    print()
    print("wrote %d sequences to %s" % (written, destination))
    print("  source fights: %d (the trainer splits on these, never within one)" % len(per_fight))
    for action, count in per_class.most_common():
        print("     %-16s %5d" % (action, count))
    if skipped:
        print("  skipped:")
        for reason, count in skipped.most_common():
            print("     %-34s %5d" % (reason, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
