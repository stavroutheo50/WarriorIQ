"""Turn a downloaded label file into training sequences.

Deliberately thin. The features are built by core.annotations.export_sequence,
the same function the web annotator calls, so a sequence produced here and a
sequence produced by correcting an event in the browser are byte-identical in
shape and meaning. Any drift between the two would be invisible until a model
trained on one was evaluated on the other.

    tools/ingest_labels.py --job <job> --labels <job>-labels.json

Sequence ids start high on purpose, so a pack ingested here can never overwrite
a sequence the web annotator wrote for the same job.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.annotations import _temporal_label, export_sequence
from core.config import DATASET

ID_BASE = 500000


def main() -> int:
    parser = argparse.ArgumentParser(description="Write .npz sequences from a label pack.")
    parser.add_argument("--job", required=True)
    parser.add_argument("--labels", required=True, help="the file the page downloaded")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if payload.get("job") != args.job:
        raise SystemExit("label file is for job %r, not %r" % (payload.get("job"), args.job))
    entries = payload.get("labels", [])
    if not entries:
        raise SystemExit("no labels in that file")

    FAMILIES = {"punch", "kick", "knee"}
    kept = collections.Counter()
    coarse = []
    agreed = disagreed = failed = 0
    for entry in entries:
        technique = entry["technique"]
        if technique in FAMILIES:
            # A family is a real answer, not a missing one, and it must never
            # reach _temporal_label - that maps anything it does not recognise
            # to "none", which would file a kick as "no strike happened" and
            # quietly teach the model the opposite of what the labeller saw.
            coarse.append(entry)
            continue
        # What the trainer will actually see, after kick heights collapse.
        kept[_temporal_label(technique)] += 1
        if technique == entry.get("proposed"):
            agreed += 1
        else:
            disagreed += 1
        if args.dry_run:
            continue
        written = export_sequence(
            args.job, ID_BASE + int(entry["id"]),
            {"fighter": entry.get("fighter", "A"), "technique": technique,
             "target": entry.get("target"), "outcome": entry.get("outcome")},
            float(entry["peak_time"]),
        )
        if written is None:
            failed += 1

    if coarse and not args.dry_run:
        # Kept for a family-level model, which is the granularity this footage
        # can actually support. The 17-class trainer cannot use them.
        path = DATASET / ("%s-family-labels.json" % args.job)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(coarse, indent=1), encoding="utf-8")
        print("%d answered only as punch/kick/knee -> %s" % (len(coarse), path))
        print("   (the 17-class trainer cannot use these; they are not lost)")
    elif coarse:
        print("%d answered only as punch/kick/knee (not written in a dry run)" % len(coarse))

    print("%d labels: %d exact, %d only a family, of which %d confirmed our guess"
          % (len(entries), len(entries) - len(coarse), len(coarse), agreed))
    if payload.get("unsure"):
        print("%d marked unsure and skipped" % len(payload["unsure"]))
    print("class balance after kick heights collapse:")
    for name, n in kept.most_common():
        print("   %-22s %4d" % (name, n))
    thin = [name for name, n in kept.items() if name != "none" and n < 10]
    if thin:
        print("under ten examples each, so not yet learnable: %s" % ", ".join(sorted(thin)))
    if args.dry_run:
        print("dry run: nothing written")
    else:
        if failed:
            print("%d could not be exported (no tracking near that time)" % failed)
        total = len(list((DATASET / "sequences").glob("*.npz")))
        print("dataset/sequences now holds %d sequences" % total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
