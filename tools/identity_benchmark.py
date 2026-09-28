"""How often the analysis followed the right fighter, against frames marked by a person.

    python tools/identity_benchmark.py <tracking.jsonl> <truth.json>

tracking.jsonl is what an analysis writes (output_dir/tracking.jsonl); the
truth files are in dataset/regression/identity_pankration. For each marked
standing frame, each fighter's box is one of:

  right    on that fighter (IoU >= MATCH_IOU with the marked box)
  swapped  on the other fighter
  other    on somebody else - the referee, a spectator, another bout
  missing  no box at all

Coverage, the number the analysis reports about itself, counts "right",
"swapped" and "other" alike, which is why it looked healthy on bouts where the
box spent most of its time on the wrong person.
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

from core.identity import box_iou

MATCH_IOU = 0.5
# On the ground the marked box holds both fighters; a fighter's box inside it
# is as right as anything can be there.
PAIR_IOU = 0.3


def _nearest(records: list[dict], frame: int) -> dict | None:
    best = min(records, key=lambda r: abs(int(r["source_frame"]) - frame), default=None)
    return best if best is not None and abs(int(best["source_frame"]) - frame) <= 2 else None


def _box(record: dict, fighter: str):
    observation = (record.get(f"fighter_{fighter}") or {}).get("observation")
    return None if not observation else observation.get("box")


def score(records: list[dict], truth: dict) -> dict:
    counts = {"A": Counter(), "B": Counter()}
    ground = Counter()
    for frame in truth["frames"]:
        record = _nearest(records, int(frame["source_frame"]))
        if record is None:
            continue
        if frame["phase"] == "ground":
            for fighter in ("A", "B"):
                box = _box(record, fighter)
                ground["missing" if box is None else
                       "on_the_pair" if box_iou(box, frame["pair"]) >= PAIR_IOU else "elsewhere"] += 1
            continue
        for fighter, other in (("A", "B"), ("B", "A")):
            if frame.get(fighter) is None:
                continue
            box = _box(record, fighter)
            if box is None:
                counts[fighter]["missing"] += 1
            elif box_iou(box, frame[fighter]) >= MATCH_IOU:
                counts[fighter]["right"] += 1
            elif frame.get(other) is not None and box_iou(box, frame[other]) >= MATCH_IOU:
                counts[fighter]["swapped"] += 1
            else:
                counts[fighter]["other"] += 1
    keys = ("right", "swapped", "other", "missing")
    out = {fighter: {key: counts[fighter][key] for key in keys} for fighter in ("A", "B")}
    for fighter in ("A", "B"):
        out[fighter]["frames"] = sum(out[fighter][key] for key in keys)
    out["ground"] = {key: ground[key] for key in ("on_the_pair", "elsewhere", "missing")}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tracking")
    parser.add_argument("truth")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in open(args.tracking, encoding="utf-8") if line.strip()]
    result = score(records, json.loads(Path(args.truth).read_text(encoding="utf-8")))
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    for fighter in ("A", "B"):
        r = result[fighter]
        print(f"fighter {fighter}: right {r['right']}/{r['frames']}  swapped {r['swapped']}  "
              f"other person {r['other']}  missing {r['missing']}")
    g = result["ground"]
    print(f"ground (both fighters in one box): on the pair {g['on_the_pair']}  elsewhere {g['elsewhere']}  missing {g['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
