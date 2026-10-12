"""Compare WarriorIQ's ground control and takedowns with a UFC bout's official numbers.

    python tools/measure_grappling.py \\
        --tracking outputs/<job>/tracking.jsonl \\
        --official dataset/public/ufc_stats/ufc_fight_stats.csv \\
        --bout "Max Holloway vs. Charles Oliveira" --event "UFC 326: Holloway vs. Oliveira 2" \\
        --fighter-a "Max Holloway" --fighter-b "Charles Oliveira" --official-round 1

Nobody labels anything: the answer is ufcstats.com's per-round control time
("CTRL") and takedowns landed. ``--official-round`` says which official round
the analysed video is, for a clip holding one round (the analysis numbers it
round 1). With ``--events`` it also compares strikes thrown standing with the
official distance strikes. Official control time also includes dominant positions standing
(back control, pinning against the fence), which core/grappling.py does not
read - so WarriorIQ's ground control should come out at or below it, never
above. Prints JSON. See core/grappling.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.grappling import strikes_while_standing, summary  # noqa: E402
from core.official_stats import load_official, official_rounds  # noqa: E402


STRIKE_FAMILIES = ("punch", "kick", "knee", "elbow")


def standing_strikes(events: list[dict], records: list[dict], official: dict, analysis_round: int) -> dict:
    """Strikes thrown standing, before and after the standing gate, against official distance strikes.

    Official distance strikes are significant strikes only; light strikes at
    distance are in TOTAL STR. but not split by position, so the official
    distance number is a floor, not an exact answer.
    """
    missing = [letter for letter in ("A", "B") if "distance" not in official[letter].by_position]
    if missing:
        raise SystemExit("The official statistics file has no DISTANCE column, so strikes thrown standing "
                         "cannot be compared with it.")
    strikes = [e for e in events if e.get("family") in STRIKE_FAMILIES and e.get("round_number") == analysis_round]
    kept = strikes_while_standing(strikes, records)
    return {letter: {"all_detected": sum(e.get("fighter") == letter for e in strikes),
                     "while_standing": sum(e.get("fighter") == letter for e in kept),
                     "official_distance_attempted": official[letter].by_position["distance"].attempted,
                     "official_total_attempted": official[letter].total.attempted}
            for letter in ("A", "B")}


def compare(ours: dict, official: dict, names: dict[str, str]) -> dict:
    """``ours``: one round of core.grappling.summary; ``official``: {"A": RoundLine, "B": RoundLine}."""
    out = {"read_seconds": ours["read_seconds"], "ground_seconds": ours["ground_seconds"], "fighters": {}}
    for letter in ("A", "B"):
        line = official[letter]
        out["fighters"][letter] = {
            "name": names[letter],
            "control_seconds": {"warrioriq": ours["control_seconds"][letter], "official": line.control_seconds},
            "takedowns": {"warrioriq": ours["takedowns"][letter], "official": line.takedowns.landed},
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--bout", required=True)
    parser.add_argument("--event")
    parser.add_argument("--fighter-a", required=True)
    parser.add_argument("--fighter-b", required=True)
    parser.add_argument("--official-round", type=int, required=True)
    parser.add_argument("--analysis-round", type=int, default=1)
    parser.add_argument("--events", type=Path, help="events.json from the same analysis, to measure standing strikes")
    args = parser.parse_args(argv)

    with args.tracking.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    ours = summary(records).get(args.analysis_round)
    if ours is None:
        print(json.dumps({"error": f"no readable frames in analysis round {args.analysis_round}"}))
        return 1
    rows = load_official(args.official)
    names = {"A": args.fighter_a, "B": args.fighter_b}
    official = {}
    for letter, name in names.items():
        lines = official_rounds(rows, bout=args.bout, fighter=name, event=args.event)
        if args.official_round not in lines:
            print(json.dumps({"error": f"no official round {args.official_round} for {name}"}))
            return 1
        official[letter] = lines[args.official_round]
    report = {"official_round": args.official_round, **compare(ours, official, names)}
    if args.events:
        payload = json.loads(args.events.read_text(encoding="utf-8"))
        events = payload.get("events", payload) if isinstance(payload, dict) else payload
        report["standing_strikes"] = standing_strikes(events, records, official, args.analysis_round)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
