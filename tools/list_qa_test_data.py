"""List the QA pass's test fighters and analyses so a person can review them.

READ-ONLY. It opens the database in read-only mode and deletes nothing; the
people running it decide what goes, and delete through the site (Fight library
> Delete, Fighters > Archive) or the normal account-deletion flow.

    python tools/list_qa_test_data.py --email qa@example.com [--email ...]
    python tools/list_qa_test_data.py --since 2026-10-01 --until 2026-10-05
    python tools/list_qa_test_data.py --name-like test --name-like qa --csv qa_review.csv

Selection, any of which can be combined (a row is listed when it matches all
the filters given):

  --email       accounts used for QA (exact, repeatable)
  --since/--until  created between these dates (UTC, inclusive)
  --name-like   fighter name or fight file name containing this text
                (case-insensitive, repeatable; any one matches)

Prints fighters, then analyses, each with the account email, the fighter name,
the creation time and the job id - the id the Fight library and /result/<id>
use - plus the files on disk that deleting the analysis would remove. Nothing
here is fabricated: every row is read straight from the database.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _matches(row: dict, args, *names: str) -> bool:
    if args.email and (row.get("email") or "").lower() not in {e.lower() for e in args.email}:
        return False
    created = str(row.get("created_at") or "")[:10]
    if args.since and created < args.since:
        return False
    if args.until and created > args.until:
        return False
    if args.name_like:
        haystack = " ".join(str(row.get(name) or "") for name in names).lower()
        if not any(part.lower() in haystack for part in args.name_like):
            return False
    return True


def main() -> int:
    from core.config import DB_PATH

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--email", action="append", default=[])
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--name-like", action="append", default=[])
    parser.add_argument("--csv", help="also write every listed row to this CSV file")
    args = parser.parse_args()
    if not (args.email or args.since or args.until or args.name_like):
        parser.error("give at least one filter, so the whole database is never listed by accident")

    connection = _connect(Path(args.db))
    fighters = [dict(row) for row in connection.execute(
        "SELECT f.id, f.name, f.archived, f.created_at, a.email FROM fighters f "
        "LEFT JOIN accounts a ON a.profile_id = f.profile_id ORDER BY f.created_at")]
    fights = [dict(row) for row in connection.execute(
        "SELECT x.job_id, x.original_name, x.ruleset, x.created_at, x.report_path, x.video_path, "
        "x.summary_json, a.email, fr.name AS fighter_name FROM fights x "
        "LEFT JOIN accounts a ON a.profile_id = x.profile_id "
        "LEFT JOIN fighters fr ON fr.id = x.fighter_id ORDER BY x.created_at")]
    fighters = [row for row in fighters if _matches(row, args, "name")]
    fights = [row for row in fights if _matches(row, args, "fighter_name", "original_name")]

    print(f"Fighters ({len(fighters)}):")
    for row in fighters:
        print(f"  #{row['id']:<6} {row['name']!r:<40} {row['email'] or '(no account)':<32} "
              f"{row['created_at']}{'  archived' if row['archived'] else ''}")
    print(f"\nAnalyses ({len(fights)}):")
    for row in fights:
        mode = ((json.loads(row["summary_json"] or "{}").get("progress_report") or {}).get("setup") or {}).get("mode")
        print(f"  {row['job_id']:<14} {row['created_at'][:19]}  {row['email'] or '(no account)':<32} "
              f"fighter={row['fighter_name'] or '-'!s:<24} file={row['original_name']!r}"
              f"{'  [solo]' if mode == 'solo' else ''}")
        for label in ("report_path", "video_path"):
            if row[label]:
                print(f"      {label}: {row[label]}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["kind", "id", "name", "email", "created_at", "file"])
            for row in fighters:
                writer.writerow(["fighter", row["id"], row["name"], row["email"], row["created_at"], ""])
            for row in fights:
                writer.writerow(["analysis", row["job_id"], row["fighter_name"], row["email"],
                                 row["created_at"], row["original_name"]])
        print(f"\nWrote {len(fighters) + len(fights)} rows to {args.csv}")
    print("\nNothing was changed. Review the list, then delete from the site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
