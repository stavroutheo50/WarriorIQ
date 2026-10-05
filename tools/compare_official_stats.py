"""Compare one analysed UFC bout with its official statistics.

    python tools/compare_official_stats.py \\
        --result outputs/<job>/  \\
        --official dataset/public/ufc_stats/ufc_fight_stats.csv \\
        --bout "Raul Rosas Jr. vs. Raoni Barcelos" \\
        --fighter-a "Raul Rosas Jr." --fighter-b "Raoni Barcelos"

``--result`` is a WarriorIQ result directory (it reads events.json). Fighter A
and B are whoever was picked as A and B on the selection page. Prints JSON.
See core/official_stats.py for what is compared and why both official
definitions are shown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.official_stats import compare, load_official, official_rounds, warrioriq_rounds  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--bout", required=True)
    parser.add_argument("--event")
    parser.add_argument("--fighter-a", required=True)
    parser.add_argument("--fighter-b", required=True)
    args = parser.parse_args(argv)

    events_path = args.result / "events.json" if args.result.is_dir() else args.result
    payload = json.loads(events_path.read_text())
    events = payload.get("events", payload) if isinstance(payload, dict) else payload
    rows = load_official(args.official)
    report = {}
    for letter, name in (("A", args.fighter_a), ("B", args.fighter_b)):
        official = official_rounds(rows, bout=args.bout, fighter=name, event=args.event)
        report[f"fighter_{letter}"] = {"name": name, **compare(warrioriq_rounds(events, letter), official)}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
