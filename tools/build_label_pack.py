"""Turn a finished analysis into a pack of clips that can be labelled quickly.

The model cannot be measured, let alone improved, without labels. There are 27
sequences across 17 classes, which is not a small dataset but an empty one, and
the existing route to more is the web annotator: open a job, correct one event,
repeat. That is fine for spot-checks and hopeless for building a test set.

This does the slow part in advance. Every candidate action becomes a filmstrip
image on disk, and the labelling itself becomes a keypress in a browser page
with no server, no database and no network - so it can be done offline, in one
sitting, by somebody who is not this program.

Two things it deliberately does that the annotator does not:

  * It proposes the analysis's own answer, so the common case is confirming
    rather than typing. Corrections are where the signal is.
  * It includes quiet windows the analysis called nothing at all. A set built
    only from detected events teaches a model what strikes look like and never
    what they do not, and cannot measure a missed-strike rate at all.

    tools/build_label_pack.py --job <job> --video <path> [--negatives 40]
    tools/ingest_labels.py    --job <job> --labels <downloaded.json>
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DB_PATH, OUTPUTS
from app.state import completed_artifact_directory
from core.temporal_model import ACTION_CLASSES

STRIP_FRAMES = 6
THUMB_HEIGHT = 230


def _read_tracking(job: str) -> dict[int, dict]:
    directory = completed_artifact_directory(job)
    if directory is None:
        raise SystemExit("This analysis has no completed generation yet")
    path = directory / "tracking.jsonl"
    if not path.exists():
        raise SystemExit("no tracking at %s - run the analysis for this job first" % path)
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(record["source_frame"])] = record
    return out


def _read_events(job: str) -> list[dict]:
    directory = completed_artifact_directory(job)
    if directory is None:
        raise SystemExit("This analysis has no completed generation yet")
    path = directory / "events.json"
    if not path.exists():
        raise SystemExit("no events at %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("events", payload) if isinstance(payload, dict) else payload


def _other(fighter: str) -> str:
    return "B" if fighter == "A" else "A"


def _window(record: dict, fighter: str):
    """Both fighters when we have them: target and outcome need the pair in shot."""
    boxes = []
    for key in ("fighter_%s" % fighter, "fighter_%s" % _other(fighter)):
        box = ((record.get(key) or {}).get("observation") or {}).get("box")
        if box:
            boxes.append(np.asarray(box, dtype=np.float32))
    if not boxes:
        return None
    stacked = np.stack(boxes)
    return np.array([stacked[:, 0].min(), stacked[:, 1].min(),
                     stacked[:, 2].max(), stacked[:, 3].max()], dtype=np.float32)


def _boxes_at(tracking, index, fighter):
    """The attacker's and opponent's boxes at this frame, if we have them.

    Only from a record close enough to be about this frame. The nearest
    analysed record can be a second away, and a box from a second away drawn
    on this frame points at empty mat.
    """
    if not tracking:
        return None, None
    near = min(tracking, key=lambda f: abs(f - index))
    if abs(near - index) > 6:
        return None, None
    record = tracking[near]
    def box(key):
        value = ((record.get(key) or {}).get("observation") or {}).get("box")
        return None if not value else np.asarray(value, dtype=np.float32)
    return box("fighter_%s" % fighter), box("fighter_%s" % _other(fighter))


def _crop_window(tracking, peak_frame, fighter, shape):
    """One rectangle for the whole strip, chosen at the peak.

    Recomputing the crop per frame made the strip jump between shots, so eight
    tiles of the same action looked like eight different moments. The subject
    also has to be big enough to read: framing on the pair puts two people
    forty pixels tall at opposite ends of a wide box, which is what made the
    first pack unusable.
    """
    height, width = shape[:2]
    own, other = _boxes_at(tracking, peak_frame, fighter)
    if own is None:
        return None
    size = max(float(own[2] - own[0]), float(own[3] - own[1]))
    x1, y1 = float(own[0]) - 0.7 * size, float(own[1]) - 0.45 * size
    x2, y2 = float(own[2]) + 0.7 * size, float(own[3]) + 0.45 * size
    own_centre = (float(own[0] + own[2]) / 2, float(own[1] + own[3]) / 2)
    if other is not None:
        centre = (float(other[0] + other[2]) / 2, float(other[1] + other[3]) / 2)
        if abs(centre[0] - own_centre[0]) > 2.6 * size:
            other = None      # a different exchange; including them shrinks this one
    if other is not None:
        # Include the opponent when they are close enough to matter, since
        # whether a strike landed cannot be judged without them.
        x1, y1 = min(x1, float(other[0]) - 0.2 * size), min(y1, float(other[1]) - 0.2 * size)
        x2, y2 = max(x2, float(other[2]) + 0.2 * size), max(y2, float(other[3]) + 0.2 * size)
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(width, x2)), int(min(height, y2))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return x1, y1, x2, y2


# What the model gets, as opposed to what the labeller gets. The filmstrip is
# six tiles at 230px because a person has to read it; this is sixteen frames at
# 64x96 grayscale because a model does not.
#
# Measured on fight 5736, where the fighters are sixty pixels tall in
# messenger-compressed 480x220 footage: upper-body motion across a ~0.3s window
# separates flagged strike moments from the rest by 1.04 standard deviations
# for fighter A and 0.78 for B, on a 2000-permutation test, p < 0.001. That is
# a floor rather than a result - the flagged moments come from a rule-based
# detector measured at 29% precision, so most of them are not strikes and are
# dragging the separation down.
#
# It matters that this is not keypoints. core/action.py reads wrist positions
# and elbow angles, and on a sixty-pixel fighter the error on a wrist is about
# the size of the punch extension being measured, which is why thresholds were
# tested and rejected. Crop motion needs no landmark precision, so it survives
# exactly the footage this product exists to serve.
SEQ_FRAMES = 16
SEQ_SIZE = (64, 96)                     # w, h


def _crop_sequence(cap, tracking, fighter, peak_frame, span_frames):
    """The crop over time, as an array, for a model to learn motion from.

    Deliberately the same window the filmstrip uses, so the clip a person
    labels and the tensor a model trains on describe the same moment. A label
    attached to a different framing than the model sees is a label about
    something else.
    """
    first = max(0, peak_frame - span_frames // 2)
    wanted = [first + round(i * span_frames / (SEQ_FRAMES - 1)) for i in range(SEQ_FRAMES)]
    window, frames = None, []
    for index in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if window is None:
            window = _crop_window(tracking, peak_frame, fighter, frame.shape)
            if window is None:
                return None
        x1, y1, x2, y2 = window
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(grey, SEQ_SIZE, interpolation=cv2.INTER_AREA))
    if len(frames) < SEQ_FRAMES:
        return None
    return np.stack(frames).astype(np.uint8)


def _filmstrip(cap, tracking, fighter, peak_frame, span_frames):
    first = max(0, peak_frame - span_frames // 2)
    wanted = [first + round(i * span_frames / (STRIP_FRAMES - 1)) for i in range(STRIP_FRAMES)]
    # One sample must be the peak itself. Evenly spaced frames can miss it, and
    # then the frame labelled THE MOMENT is one the analysis never looked at -
    # so it carries no box, on the very frame the labeller is told to judge.
    nearest = min(range(len(wanted)), key=lambda i: abs(wanted[i] - peak_frame))
    wanted[nearest] = peak_frame
    window = None
    tiles = []
    for index in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if window is None:
            window = _crop_window(tracking, peak_frame, fighter, frame.shape)
            if window is None:
                return None
        x1, y1, x2, y2 = window
        shot = frame.copy()
        own, other = _boxes_at(tracking, index, fighter)
        # The subject is marked in every frame. Without it the labeller is
        # shown three people and asked what "the fighter" did.
        if other is not None:
            cv2.rectangle(shot, (int(other[0]), int(other[1])), (int(other[2]), int(other[3])),
                          (90, 90, 90), 1)
        if own is not None:
            cv2.rectangle(shot, (int(own[0]), int(own[1])), (int(own[2]), int(own[3])),
                          (0, 215, 255), 2)
        crop = shot[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        scale = THUMB_HEIGHT / crop.shape[0]
        tile = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), THUMB_HEIGHT),
                          interpolation=cv2.INTER_CUBIC)
        at_peak = index == peak_frame
        tile = cv2.copyMakeBorder(tile, 22, 4, 2, 2, cv2.BORDER_CONSTANT,
                                  value=(0, 140, 200) if at_peak else (24, 24, 24))
        if at_peak:
            cv2.putText(tile, "THE MOMENT", (6, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    width = min(t.shape[1] for t in tiles)
    return cv2.hconcat([t[:, :width] for t in tiles])


# Below this, over three seconds, the subject is not a fighter having a quiet
# moment - they are sitting down. Measured on real tournament footage: seated
# people reach 0.006 to 0.021 body lengths of centre spread, and the least
# mobile fighter 0.059. Anything under 0.03 is a chair.
STILL_BODY_LENGTHS = 0.03


def _is_moving(tracking: dict, frame: int, fps: float, seconds: float = 1.5) -> bool:
    """Was the tracked subject moving at all around this frame?

    A quiet window is meant to teach the model what "no strike" looks like on a
    fighter. When identity has drifted onto a spectator, it teaches it what a
    chair looks like instead - and costs the person labelling it a real answer
    to a question that was never worth asking. Measured on the three reference
    packs, this was 13 of 40 quiet clips in one of them.

    Judged on the spread of the box centre in body lengths, which is scale-free,
    rather than on pixels - a distant fighter moves few pixels and a great many
    body lengths.
    """
    window = max(1, int(fps * seconds))
    centres, heights = [], []
    for offset in range(-window, window + 1):
        record = tracking.get(frame + offset)
        if not record:
            continue
        box = ((record.get("fighter_A") or {}).get("observation") or {}).get("box")
        if not box:
            continue
        centres.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
        heights.append(max(1.0, box[3] - box[1]))
    if len(centres) < 5:
        # Too little to judge. Keep it: a missing measurement is not evidence
        # of a chair, and the labeller can always answer "nothing".
        return True
    points = np.asarray(centres, dtype=np.float32)
    spread = float(np.linalg.norm(points.std(axis=0)) / float(np.mean(heights)))
    return spread >= STILL_BODY_LENGTHS


def _video_for(job):
    """Where the fight that produced this job actually lives.

    Passing --video by hand meant knowing which of 126 analysed jobs came from
    which file. The fights table already records it, so the answer is looked up
    rather than remembered. Footage ages out after the retention window, so a
    job whose video is gone is skipped rather than guessed at.
    """
    try:
        with sqlite3.connect(str(DB_PATH)) as con:
            row = con.execute(
                "SELECT video_path FROM fights WHERE job_id=?", (job,)).fetchone()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    return row[0] if Path(row[0]).exists() else None


def _packable_jobs():
    """Every analysed job whose video is still on disk, newest first."""
    found = []
    reports = []
    for job_dir in OUTPUTS.iterdir():
        if not job_dir.is_dir():
            continue
        directory = completed_artifact_directory(job_dir.name)
        if directory is not None and (directory / "report.json").is_file():
            reports.append((job_dir.name, directory / "report.json"))
    reports.sort(key=lambda item: item[1].stat().st_mtime, reverse=True)
    for job, report in reports:
        video = _video_for(job)
        if video:
            found.append((job, video))
    return found


def _candidates_for(job, video, negatives, rng):
    """The proposed actions and quiet windows for one job."""
    tracking = _read_tracking(job)
    events = _read_events(job)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    candidates = []
    for event in events:
        candidates.append({
            "job": job,
            "fighter": event.get("fighter", "A"),
            "peak_frame": int(event.get("peak_frame", 0)),
            "peak_time": float(event.get("peak_time", 0.0)),
            "proposed": event.get("technique", "none"),
            "target": event.get("target") or "none",
            "outcome": event.get("outcome") or "uncertain",
            "source": "event",
        })

    busy = [c["peak_frame"] for c in candidates]
    quiet = [f for f in sorted(tracking)
             if all(abs(f - b) > fps for b in busy)
             and ((tracking[f].get("fighter_A") or {}).get("observation") or {}).get("keypoints")
             and _is_moving(tracking, f, fps)]
    for frame in rng.sample(quiet, min(negatives, len(quiet))):
        candidates.append({
            "job": job, "fighter": "A", "peak_frame": int(frame),
            "peak_time": float(tracking[frame].get("time_seconds", frame / fps)),
            "proposed": "none", "target": "none", "outcome": "uncertain",
            "source": "quiet",
        })
    return candidates, tracking, cap


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a clip pack for fast labelling.")
    parser.add_argument("--job", nargs="*", default=[],
                        help="job ids under outputs/; repeat for several")
    parser.add_argument("--all", action="store_true",
                        help="every analysed job whose video is still on disk")
    parser.add_argument("--video", default=None,
                        help="the fight video, when it cannot be looked up "
                             "(only meaningful with a single --job)")
    parser.add_argument("--negatives", type=int, default=40,
                        help="quiet windows per job, so the set has true negatives")
    parser.add_argument("--out", default="labelpack")
    parser.add_argument("--no-sequences", dest="sequences", action="store_false",
                        help="skip the per-clip crop tensors. They are what a "
                             "motion model trains on, and they are the reason "
                             "labelling this pack is worth doing on cheap "
                             "footage - see _crop_sequence.")
    parser.add_argument("--name", default=None,
                        help="folder name for a multi-job pack (default: mixed)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many clips in total")
    parser.add_argument("--no-inline", dest="inline", action="store_false",
                        help="reference the clips as files instead of carrying them "
                             "in the page; smaller, but the page stops working if "
                             "it is moved away from its clips folder")
    parser.set_defaults(inline=True)
    args = parser.parse_args()

    if args.all:
        jobs = _packable_jobs()
    elif args.job:
        jobs = []
        for job in args.job:
            video = args.video if (args.video and len(args.job) == 1) else _video_for(job)
            if not video:
                print("skipping %s: no video on disk" % job)
                continue
            jobs.append((job, video))
    else:
        raise SystemExit("give --job <id> [<id> ...] or --all")
    if not jobs:
        raise SystemExit("nothing to pack: no job had a video still on disk")

    single = len(jobs) == 1
    root = Path(args.out) / (jobs[0][0] if single else (args.name or "mixed"))
    clips = root / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    sequences = None
    if args.sequences:
        sequences = root / "sequences"
        sequences.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    # A budget per job rather than first-come, so a pack spread over a hundred
    # fights is not ninety percent the first one. A model trained on one bout
    # learns that bout: 142 sequences memorised, which is why the note in
    # core/model_validation.py says label broadly before training anything.
    per_job = max(1, args.limit // len(jobs)) if args.limit else 0

    index = []
    number = 0
    for job, video in jobs:
        candidates, tracking, cap = _candidates_for(job, video, args.negatives, rng)
        if candidates is None:
            print("skipping %s: could not open %s" % (job, video))
            continue
        rng.shuffle(candidates)          # order carries no hint about the answer
        if per_job:
            candidates = candidates[:per_job]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        span = max(6, int(round(0.8 * fps)))
        for candidate in candidates:
            if args.limit and len(index) >= args.limit:
                break
            strip = _filmstrip(cap, tracking, candidate["fighter"],
                               candidate["peak_frame"], span)
            if strip is None:
                continue
            name = "%05d.jpg" % number
            cv2.imwrite(str(clips / name), strip, [cv2.IMWRITE_JPEG_QUALITY, 72])
            candidate["id"] = number
            candidate["clip"] = "clips/%s" % name
            if sequences is not None:
                # Saved under the same id as the filmstrip, so ingesting a
                # label joins them without a second lookup table. Compressed:
                # 16 x 96 x 64 uint8 is 98 KB raw and about 20 KB stored, so a
                # 320-clip pack costs single-digit megabytes.
                seq = _crop_sequence(cap, tracking, candidate["fighter"],
                                     candidate["peak_frame"], span)
                if seq is not None:
                    np.savez_compressed(
                        str(sequences / ("%05d.npz" % number)), frames=seq,
                        fighter=candidate["fighter"],
                        peak_frame=int(candidate["peak_frame"]),
                        span_frames=int(span), job=job)
                    candidate["sequence"] = "sequences/%05d.npz" % number
            index.append(candidate)
            number += 1
        cap.release()
        if args.limit and len(index) >= args.limit:
            break

    if not index:
        raise SystemExit("no clips could be cut from those jobs")

    # Interleaved so a session moves between fights instead of sitting in one
    # for an hour, the same reason the order within a job is shuffled.
    rng.shuffle(index)

    payload = {"job": jobs[0][0] if single else "mixed",
               "jobs": [j for j, _ in jobs],
               "video": jobs[0][1] if single else None,
               "classes": ACTION_CLASSES, "candidates": index}
    (root / "index.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # The candidate list is injected rather than fetched, because the page is
    # opened straight off the filesystem and file:// blocks fetch and XHR.
    #
    # The clips are carried inside it too, by default. Relative <img> paths do
    # resolve from file://, but then the page only works while it sits in this
    # exact folder beside this exact directory - and a labelling pack is
    # something to copy onto a laptop and work through on a train. One file
    # that always works is worth a few megabytes.
    page_payload = dict(payload)
    if args.inline:
        carried = []
        for candidate in index:
            item = dict(candidate)
            raw = (root / candidate["clip"]).read_bytes()
            item["clip"] = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
            carried.append(item)
        page_payload["candidates"] = carried

    page = (Path(__file__).parent / "label_pack_page.html").read_text(encoding="utf-8")
    if "/*__DATA__*/ null" not in page:
        raise SystemExit("label_pack_page.html has lost its data placeholder")
    page = page.replace("/*__DATA__*/ null",
                        json.dumps(page_payload).replace("</", "<\\/"))
    (root / "label.html").write_text(page, encoding="utf-8")

    proposed = sum(1 for c in index if c["source"] == "event")
    if sequences is not None:
        print("%d crop sequences in %s" % (
            len(list(sequences.glob("*.npz"))), sequences))
    print("%d clips in %s  (%d proposed actions, %d quiet windows, %d fights)"
          % (len(index), root, proposed, len(index) - proposed, len(jobs)))
    print("open: %s" % (root / "label.html"))
    print("      answers save to disk as you give them; no button to remember.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
