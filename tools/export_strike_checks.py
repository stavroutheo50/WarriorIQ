"""Turn fighters' one-tap answers on a fight into a benchmark fight.

On a result page every counted strike has "Right / Not a strike / It was a
punch, kick or knee". Those answers are stored as annotations with source
"strike_check" (app.main.check_counted_strike). This writes one fight's
answers, with the pose track its analysis produced, into the folder layout
tools/benchmark_labelled_fight.py reads - the same layout as the hand-labelled
Kick Light bout - so the next detector change is measured on it too.

    python tools/export_strike_checks.py --job <job_id>
    python tools/export_strike_checks.py --job <job_id> --name club_sparring_3

Run it where the database and the analysis artifacts live (the web host).
Only the pose track is copied - keypoints and boxes, never video - and the
labels carry times and verdicts, never names.

What this data can and cannot say: every answer is about a strike the
report counted, so it measures precision (how many counted strikes were
real, and of the right type). It cannot measure missed strikes, which were
never offered to anyone. A fight with few answers is exported anyway; the
benchmark reports how many moments each fight has.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.state import completed_artifact_directory
from core.db import get_annotations, is_strike_check

REGRESSION = PROJECT_ROOT / "dataset" / "regression"

# The benchmark's answer vocabulary (see benchmark_labelled_fight._family).
_ANSWER = {"not_a_strike": "none", "wrong_fighter": "__wrongperson__"}

# Only what the replay reads. identity_confidence feeds the action engine.
_OBSERVATION_KEYS = ("track_id", "box", "confidence", "keypoints", "keypoint_conf")


def labels_for(job_id: str) -> list[dict]:
    rows = []
    for item in get_annotations(job_id):
        if not is_strike_check(item):
            continue
        corrected = item["corrected"]
        verdict = corrected.get("verdict") or ""
        proposed = (item.get("predicted") or {}).get("family")
        answer = _ANSWER.get(verdict, proposed if verdict == "right" else verdict)
        rows.append({"fighter": corrected["fighter"], "peak_time": float(item["event_time"]),
                     "source": "event", "proposed": proposed, "answer": answer})
    return sorted(rows, key=lambda row: row["peak_time"])


def slim_track(source: Path) -> bytes:
    lines = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            slim = {"source_frame": record["source_frame"], "time_seconds": record["time_seconds"],
                    "round_number": record.get("round_number")}
            for key in ("fighter_A", "fighter_B"):
                fighter = record.get(key) or {}
                observation = fighter.get("observation")
                slim[key] = {
                    "identity_confidence": fighter.get("identity_confidence", 0.0),
                    "observation": None if not observation else
                    {k: observation.get(k) for k in _OBSERVATION_KEYS},
                }
            lines.append(json.dumps(slim, separators=(",", ":")))
    return gzip.compress(("\n".join(lines) + "\n").encode("utf-8"), 9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--job", required=True)
    parser.add_argument("--name", default=None, help="folder name (default: checks_<job>)")
    args = parser.parse_args()

    labels = labels_for(args.job)
    if not labels:
        raise SystemExit("no one-tap answers on %s yet" % args.job)
    directory = completed_artifact_directory(args.job)
    track = None if directory is None else directory / "tracking.jsonl"
    if track is None or not track.exists():
        raise SystemExit("the pose track for %s is gone (it ages out with the fight's files)" % args.job)

    out = REGRESSION / (args.name or "checks_%s" % args.job)
    out.mkdir(parents=True, exist_ok=True)
    (out / "track.jsonl.gz").write_bytes(slim_track(track))
    (out / "labels.json").write_text(json.dumps({
        "fight": "exported from one-tap answers on job %s" % args.job,
        "labelled_by": "the fight's owner, on the result page",
        "answers": "a family (punch/kick/knee), 'none' or '__wrongperson__'",
        "source": "every label is a strike the report counted",
        "labels": labels,
    }, indent=1) + "\n", encoding="utf-8")
    print("%d answers written to %s" % (len(labels), out))
    print("next: python tools/benchmark_labelled_fight.py --write-baseline, then commit the folder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
