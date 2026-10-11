"""Turn the gym part of the Combat Sports Dataset into peak-pose examples.

The Combat Sports Dataset (Evan Quinn, Zenodo doi:10.5281/zenodo.15349809,
CC BY 4.0) is about 7,700 still images with boxes on the striking limb
(cross, hook, kick), on guards (high-guard, low-guard), on each person and on
punch bags. Download "CombatSports Dataset - Greyscale 10%.zip" from the
Zenodo page, then from the repository root:

    .venv\\Scripts\\python.exe tools\\import_combat_sports.py --zip "C:\\path\\to\\CombatSports Dataset - Greyscale 10%.zip"

## Gym footage only

Many of its images are cut from television broadcasts and press photographs,
whose rights the dataset's licence cannot grant. Only the sources in
GYM_SOURCES are read: bag-work videos checked by eye, on 2026-10-10, to be
gym recordings rather than broadcast or press material. Anything else in the
archive is skipped and counted, never imported.

## What one example is

For each person box, the limb and guard boxes inside it (at least half of the
limb box's area) say what that person is doing on that frame:

* cross or hook -> ``punch``; kick -> ``kick``; a guard and no strike -> ``none``.
* a person with both a punch and a kick box, or with no labelled box, is
  skipped - there is no single true answer for them.

RTMPose - the same top-down model the live analysis refines fighters with,
through core.rtm_pose.refine - reads that person's joints from the box, and
core.peak_pose.pose_features turns them into the 51 numbers the model learns.
If RTMPose is not available nothing is written: the detector's own keypoints
do not exist here and nothing else may stand in for them.

## Duplicates and the held-out videos

The archive keeps up to twelve augmented copies of a frame (``.rf.<hash>``);
only the first of each is used. The dataset's own train/valid/test split is
per image, so near-identical frames of one video sit on both sides of it.
This splits by video instead: EXAM_SOURCES are never trained on, so the
score on them is a score on videos the model has not seen.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

NAMES = ("boxing-bag", "cross", "high-guard", "hook", "kick", "low-guard", "person")
# Bag-work videos recorded in ordinary gyms and garages, checked by eye on
# contact sheets of the 30 largest sources (2026-10-10/11). Keys are the
# "<n>_mp4" prefix of the image names. Left out: broadcast, press, the ~360
# small unreviewed sources, and four that look like commercial or stock
# productions (1, 3 green screen, 4 a branded fitness product, 6 studio white).
# Who filmed each video is not recorded in the dataset, so this is caution,
# not proof of ownership.
GYM_SOURCES = ("5", "7", "8", "9", "12", "17", "26", "27", "28", "29", "65", "66")
# Held out by video: 90 punches, 79 guards and 7 kicks after the steps below (2026-10-11).
EXAM_SOURCES = ("5", "9", "26")
MIN_INSIDE = 0.5
OUT = ROOT / "dataset" / "peak_pose" / "combat_sports_gym.npz"


def source_of(filename: str) -> str | None:
    """'28_mp4-0017_jpg.rf.<hash>.txt' -> '28'; anything not cut from a numbered video -> None."""
    match = re.match(r"^(\d+)_mp4-", Path(filename).name)
    return match.group(1) if match else None


def frame_key(filename: str) -> str:
    """The frame an augmented copy was made from."""
    return Path(filename).name.split(".rf.")[0]


def parse_labels(text: str) -> list[tuple[str, list[float]]]:
    """YOLO rows -> (class name, [x1, y1, x2, y2] as fractions of the image)."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        index, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        if 0 <= index < len(NAMES):
            rows.append((NAMES[index], [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return rows


def _inside(part: list[float], whole: list[float]) -> float:
    """How much of ``part``'s area lies inside ``whole``."""
    w = max(0.0, min(part[2], whole[2]) - max(part[0], whole[0]))
    h = max(0.0, min(part[3], whole[3]) - max(part[1], whole[1]))
    area = max(1e-9, (part[2] - part[0]) * (part[3] - part[1]))
    return w * h / area


def people_with_labels(rows: list[tuple[str, list[float]]]) -> list[tuple[list[float], str, str]]:
    """(person box, family, technique) for every person with one true answer."""
    from core.peak_pose import FAMILY

    people = [box for name, box in rows if name == "person"]
    held: list[list[str]] = [[] for _ in people]
    for name, box in rows:
        if name not in FAMILY or not people:
            continue
        overlap = [_inside(box, person) for person in people]
        best = int(np.argmax(overlap))
        if overlap[best] >= MIN_INSIDE:
            held[best].append(name)
    out = []
    for person, names in zip(people, held):
        strikes = sorted({n for n in names if FAMILY[n] != "none"})
        families = {FAMILY[n] for n in strikes}
        if len(families) > 1:
            continue
        if strikes:
            out.append((person, families.pop(), "+".join(strikes)))
        elif names:
            out.append((person, "none", "guard"))
    return out


def keypoints_for(image: np.ndarray, boxes: list[list[float]]):
    """RTMPose joints for each pixel box, or None if the model is unavailable."""
    from core import rtm_pose
    from core.types import PersonObservation

    people = [PersonObservation(track_id=None, box=np.asarray(b, np.float32), confidence=1.0) for b in boxes]
    if rtm_pose.refine(image, people) != len(people):
        return None
    return [(p.keypoints, p.keypoint_conf) for p in people]


def import_archive(path: Path, *, sources=GYM_SOURCES, exam=EXAM_SOURCES, limit: int | None = None):
    import cv2

    from core.peak_pose import pose_features

    archive = zipfile.ZipFile(path)
    labels = sorted(n for n in archive.namelist() if "/labels/" in n and n.endswith(".txt"))
    images = {n.rsplit(".", 1)[0]: n for n in archive.namelist() if "/images/" in n}
    seen_frames: set[str] = set()
    counts: collections.Counter = collections.Counter()
    rows: dict[str, list] = collections.defaultdict(list)
    for name in labels:
        source = source_of(name)
        if source not in sources:
            counts["skipped: not a gym source"] += 1
            continue
        if frame_key(name) in seen_frames:
            counts["skipped: augmented copy"] += 1
            continue
        seen_frames.add(frame_key(name))
        labelled = people_with_labels(parse_labels(archive.read(name).decode("utf-8", "replace")))
        image_name = images.get(name.replace("/labels/", "/images/").rsplit(".", 1)[0])
        if not labelled or image_name is None:
            counts["skipped: no person with one answer"] += 1
            continue
        image = cv2.imdecode(np.frombuffer(archive.read(image_name), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            counts["skipped: unreadable image"] += 1
            continue
        height, width = image.shape[:2]
        boxes = [[b[0] * width, b[1] * height, b[2] * width, b[3] * height] for b, _, _ in labelled]
        joints = keypoints_for(image, boxes)
        if joints is None:
            raise SystemExit("RTMPose is not available on this machine (see README: requirements-rtm-*.txt). "
                             "Nothing was written.")
        for (_, family, technique), box, (kp, conf) in zip(labelled, boxes, joints):
            rows["features"].append(pose_features(box, kp, conf))
            rows["family"].append(family)
            rows["technique"].append(technique)
            rows["source"].append(source)
            rows["split"].append("exam" if source in exam else "train")
            rows["image"].append(Path(image_name).name)
            counts[("exam" if source in exam else "train", family)] += 1
        if limit and len(seen_frames) >= limit:
            break
    return rows, counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True, help="the dataset's Greyscale 10%% zip, unopened")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, help="stop after this many frames (a quick trial)")
    args = parser.parse_args(argv)
    if not args.zip.exists():
        raise SystemExit(f"No file at {args.zip}.")

    rows, counts = import_archive(args.zip, limit=args.limit)
    if not rows:
        raise SystemExit("No examples found: is this the Combat Sports Dataset zip?")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=np.stack(rows["features"]).astype(np.float32),
                        **{k: np.asarray(v) for k, v in rows.items() if k != "features"})
    for key, value in sorted(counts.items(), key=str):
        print(f"  {key if isinstance(key, str) else ' '.join(key)}: {value}")
    print(f"Wrote {len(rows['family'])} examples to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
