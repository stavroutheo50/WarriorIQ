"""Turn owners' answers on "Moments a fighter went down" into test data.

On a result page each moment the analysis found (core/ground.py) has
"Fighter A / Fighter B / Nobody". Those answers are stored as annotations
with source "down_check" (app.main.check_went_down). This writes one fight's
moments and answers to dataset/regression/downs_<job>/labels.json, and
--summary reads every exported fight and says what the answers show:

    python tools/export_down_checks.py --job <job_id>
    python tools/export_down_checks.py --summary

Run the export where the database and the analysis artifacts live (the web
host). Only times and answers are written: no video, no names.

What the answers can and cannot say: every answer is about a moment the
analysis flagged, so they measure how often a flag was right ("nobody" is a
false alarm) and who actually went down - the attribution the analysis does
not make yet. They cannot measure downs that were never flagged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REGRESSION = PROJECT_ROOT / "dataset" / "regression"
PREFIX = "downs_"


def labels_for(job_id: str, report: dict) -> list[dict]:
    """The report's moments, each with the owner's answer (None when not answered)."""
    from app.main import _down_checks

    answers = _down_checks(job_id)
    out = []
    for moment in (report.get("went_down") or {}).get("moments") or []:
        seconds = round(float(moment["seconds"]), 3)
        out.append({"seconds": seconds, "round": moment.get("round"),
                    "down_seconds": moment.get("down_seconds"), "answer": answers.get(seconds)})
    return out


def summary(root: Path = REGRESSION) -> dict:
    """Across every exported fight: answered moments, false alarms, and who went down."""
    answers = Counter()
    fights = 0
    for path in sorted(root.glob(PREFIX + "*/labels.json")):
        fights += 1
        for moment in json.loads(path.read_text(encoding="utf-8"))["moments"]:
            if moment.get("answer"):
                answers[moment["answer"]] += 1
    answered = sum(answers.values())
    real = answers["A"] + answers["B"]
    return {"fights": fights, "answered": answered, "someone_went_down": real,
            "nobody": answers["nobody"], "fighter_A": answers["A"], "fighter_B": answers["B"],
            "flags_right": None if not answered else round(real / answered, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--job")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        print(json.dumps(summary(), indent=1))
        return 0
    if not args.job:
        parser.error("--job or --summary")
    from app.state import completed_artifact_directory

    directory = completed_artifact_directory(args.job)
    report_path = None if directory is None else directory / "report.json"
    if report_path is None or not report_path.exists():
        raise SystemExit("no report for %s" % args.job)
    moments = labels_for(args.job, json.loads(report_path.read_text(encoding="utf-8")))
    if not any(m["answer"] for m in moments):
        raise SystemExit("no answers on %s yet" % args.job)
    out = REGRESSION / f"{PREFIX}{args.job}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps({
        "fight": "exported from went-down answers on job %s" % args.job,
        "answers": "'A' or 'B' went down, 'nobody' when the moment was a false alarm, null when unanswered",
        "moments": moments,
    }, indent=1) + "\n", encoding="utf-8")
    print("%d moments (%d answered) written to %s" % (len(moments), sum(1 for m in moments if m["answer"]), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
