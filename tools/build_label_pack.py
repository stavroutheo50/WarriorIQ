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
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import OUTPUTS
from core.temporal_model import ACTION_CLASSES

STRIP_FRAMES = 8
THUMB_HEIGHT = 190


def _read_tracking(job: str) -> dict[int, dict]:
    path = OUTPUTS / job / "tracking.jsonl"
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
    path = OUTPUTS / job / "events.json"
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


def _filmstrip(cap, tracking, fighter, peak_frame, span_frames):
    first = max(0, peak_frame - span_frames // 2)
    wanted = [first + round(i * span_frames / (STRIP_FRAMES - 1)) for i in range(STRIP_FRAMES)]
    tiles = []
    for index in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        near = min(tracking, key=lambda f: abs(f - index)) if tracking else None
        box = _window(tracking[near], fighter) if near is not None else None
        if box is None:
            continue
        pad = 0.55 * max(box[2] - box[0], box[3] - box[1])
        x1, y1 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad))
        x2 = int(min(frame.shape[1], box[2] + pad))
        y2 = int(min(frame.shape[0], box[3] + pad))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        scale = THUMB_HEIGHT / crop.shape[0]
        tile = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), THUMB_HEIGHT),
                          interpolation=cv2.INTER_CUBIC)
        if abs(index - peak_frame) <= max(1, span_frames // (STRIP_FRAMES * 2)):
            # Mark where the analysis thinks the action peaks, so a labeller
            # judging "which technique" is looking at the right moment.
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 215, 255), 3)
        tiles.append(tile)
    if not tiles:
        return None
    width = min(t.shape[1] for t in tiles)
    return cv2.hconcat([t[:, :width] for t in tiles])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a clip pack for fast labelling.")
    parser.add_argument("--job", required=True, help="job id under outputs/")
    parser.add_argument("--video", required=True, help="the fight video that job analysed")
    parser.add_argument("--negatives", type=int, default=40,
                        help="quiet windows to include, so the set has true negatives")
    parser.add_argument("--out", default="labelpack")
    parser.add_argument("--no-inline", dest="inline", action="store_false",
                        help="reference the clips as files instead of carrying them "
                             "in the page; smaller, but the page stops working if "
                             "it is moved away from its clips folder")
    parser.set_defaults(inline=True)
    args = parser.parse_args()

    tracking = _read_tracking(args.job)
    events = _read_events(args.job)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    span = max(6, int(round(0.8 * fps)))

    root = Path(args.out) / args.job
    clips = root / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    candidates = []
    for event in events:
        candidates.append({
            "fighter": event.get("fighter", "A"),
            "peak_frame": int(event.get("peak_frame", 0)),
            "peak_time": float(event.get("peak_time", 0.0)),
            "proposed": event.get("technique", "none"),
            "target": event.get("target") or "none",
            "outcome": event.get("outcome") or "uncertain",
            "source": "event",
        })

    # Quiet windows: at least a second from any proposed action, and only where
    # a fighter was actually being followed, so "none" means "nothing happened"
    # rather than "nobody was tracked".
    busy = [c["peak_frame"] for c in candidates]
    quiet = [f for f in sorted(tracking)
             if all(abs(f - b) > fps for b in busy)
             and ((tracking[f].get("fighter_A") or {}).get("observation") or {}).get("keypoints")]
    rng = random.Random(0)
    for frame in rng.sample(quiet, min(args.negatives, len(quiet))):
        candidates.append({
            "fighter": "A",
            "peak_frame": int(frame),
            "peak_time": float(tracking[frame].get("time_seconds", frame / fps)),
            "proposed": "none", "target": "none", "outcome": "uncertain",
            "source": "quiet",
        })

    # Shuffled so the order carries no hint about what the answer should be.
    rng.shuffle(candidates)
    index = []
    for number, candidate in enumerate(candidates):
        strip = _filmstrip(cap, tracking, candidate["fighter"], candidate["peak_frame"], span)
        if strip is None:
            continue
        name = "%04d.jpg" % number
        cv2.imwrite(str(clips / name), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        candidate["id"] = number
        candidate["clip"] = "clips/%s" % name
        index.append(candidate)
    cap.release()

    payload = {"job": args.job, "video": args.video, "classes": ACTION_CLASSES,
               "candidates": index}
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
    print("%d clips in %s  (%d proposed actions, %d quiet windows)"
          % (len(index), root, proposed, len(index) - proposed))
    print("open %s in a browser, label, then press Download labels" % (root / "label.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
