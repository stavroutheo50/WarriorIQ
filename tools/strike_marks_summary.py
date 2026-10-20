"""How the automatic strike count did on every fight someone marked.

The owner marks strikes on /mark/<job_id> (app.main.strike_marks_page); the
marks are stored as outputs/<job_id>/strike_marks.json. This adds every
marked fight together: of the strikes WarriorIQ counted, how many the person
also marked, and of the strikes they marked, how many WarriorIQ found.

    python tools/strike_marks_summary.py
    python tools/strike_marks_summary.py --json

Run it where the analysis artifacts live (the web host). It reads only; it
prints times and counts, never names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import STRIKE_MARKS_FILE, _counted_strikes
from app.state import completed_artifact_directory
from core import strike_marks
from core.config import OUTPUTS


def summarize() -> dict:
    fights, totals = [], {"marked": 0, "counted": 0, "matched": 0, "type_checked": 0, "type_right": 0}
    for path in sorted(OUTPUTS.glob(f"*/{STRIKE_MARKS_FILE}")):
        job_id = path.parent.name
        directory = completed_artifact_directory(job_id)
        if directory is None or not (directory / "report.json").is_file():
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        result = strike_marks.compare(stored.get("marks") or [], _counted_strikes(report, strike_marks.FAMILIES),
                                      float(stored.get("covered_seconds") or 0.0))
        fights.append({"job_id": job_id, **{key: result[key] for key in (*totals, "covered_seconds")}})
        for key in totals:
            totals[key] += result[key]
    totals["counted_real_share"] = round(totals["matched"] / totals["counted"], 3) if totals["counted"] else None
    totals["marked_found_share"] = round(totals["matched"] / totals["marked"], 3) if totals["marked"] else None
    return {"fights": fights, "totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    summary = summarize()
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    for fight in summary["fights"]:
        print(f"{fight['job_id']}: watched {fight['covered_seconds']:.0f}s, marked {fight['marked']}, "
              f"counted {fight['counted']}, found by both {fight['matched']}")
    t = summary["totals"]
    pct = lambda v: "-" if v is None else f"{v * 100:.0f}%"
    print(f"\n{len(summary['fights'])} marked fights. Counted strikes that were real: {pct(t['counted_real_share'])} "
          f"({t['matched']} of {t['counted']}). Real strikes found: {pct(t['marked_found_share'])} "
          f"({t['matched']} of {t['marked']}).")
    if t["type_checked"]:
        print(f"Type right where the person gave it: {t['type_right']} of {t['type_checked']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
