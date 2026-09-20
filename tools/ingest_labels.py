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
from app.state import completed_artifact_directory, get_job

ID_BASE = 500000


def _agrees(technique: str, proposed: str | None) -> bool:
    """Did the labeller confirm the detector's guess?

    Compared as the trainer will see them, not as raw strings. The detector
    names a kick by where it landed - core/contact.py rewrites a round kick to
    left_low_kick, left_body_kick or left_head_kick - while the page can only
    offer the seventeen technique classes, so left_round_kick is the only way
    to agree about that kick at all. A string compare therefore recorded a
    disagreement every time the labeller agreed about one, and 104 of the
    proposals across the packs on disk carry a height.

    Neither side can collapse to "none" by accident here: families are
    diverted before this is reached, and every proposal the builder writes
    maps to a real class, so this cannot manufacture an agreement out of a
    name that failed to map.
    """
    return _temporal_label(technique) == _temporal_label(proposed or "none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write .npz sequences from a label pack.")
    parser.add_argument("--job", default=None,
                        help="the job this pack came from; not needed for a "
                             "pack built across several fights, where every "
                             "label carries its own")
    parser.add_argument("--labels", required=True, help="the file the page downloaded")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    # A pack can now span many fights - one sitting across a hundred bouts
    # instead of a hundred separate packs - so the job belongs to the label,
    # not to the file. A single-job pack still names itself at the top level
    # and is still checked against --job when one is given.
    mixed = payload.get("job") == "mixed" or bool(payload.get("jobs"))
    if not mixed and args.job and payload.get("job") != args.job:
        raise SystemExit("label file is for job %r, not %r" % (payload.get("job"), args.job))
    if not mixed and not args.job:
        args.job = payload.get("job")
        if not args.job:
            raise SystemExit("that label file names no job; pass --job")
    entries = payload.get("labels", [])
    wrong = payload.get("wrong_person") or []
    unsure = payload.get("unsure") or []
    if not entries:
        # A pack answered entirely "wrong person" produces no training data and
        # is the most informative file the labeller can hand back, so it is
        # reported rather than refused. Bailing here printed "no labels in that
        # file" over the measurement that mattered.
        if wrong:
            judged = len(wrong) + len(unsure)
            print("%d of %d judged clips were the WRONG PERSON (%.0f%%)."
                  % (len(wrong), judged, 100.0 * len(wrong) / max(1, judged)))
            print("No technique labels, and none are worth collecting from this fight")
            print("until identity holds: a technique recorded against the wrong person")
            print("is training data pointing the wrong way.")
            raise SystemExit(0)
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
        if _agrees(technique, entry.get("proposed")):
            agreed += 1
        else:
            disagreed += 1
        if args.dry_run:
            continue
        job_id = entry.get("job") or args.job
        directory = completed_artifact_directory(job_id)
        if directory is None:
            failed += 1
            continue
        written = export_sequence(
            job_id, ID_BASE + int(entry["id"]),
            {"fighter": entry.get("fighter", "A"), "technique": technique,
             "target": entry.get("target"), "outcome": entry.get("outcome")},
            float(entry["peak_time"]),
            tracking_path=directory / "tracking.jsonl",
            source_fight_id=(get_job(job_id) or {}).get("source_video_sha256"),
        )
        if written is None:
            failed += 1

    if coarse and not args.dry_run:
        # Kept for a family-level model, which is the granularity this footage
        # can actually support. The 17-class trainer cannot use them.
        path = DATASET / ("%s-family-labels.json" % (args.job or "mixed"))
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
    # Not training data, and the most important number in the file. A clip
    # whose box is on the referee or a spectator says the analysis lost the
    # fighter there - and unlike coverage, which only counts whether *something*
    # was tracked, this says whether it was the right something.
    if wrong:
        judged = len(entries) + len(unsure) + len(wrong)
        print("%d of %d clips were the WRONG PERSON (%.0f%%) - identity, not technique,"
              % (len(wrong), judged, 100.0 * len(wrong) / max(1, judged)))
        print("   is what those clips measure. Nothing trained on this fight is")
        print("   trustworthy while that share is high.")
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
