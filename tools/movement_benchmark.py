"""How well the analysis read where each fighter stood and which way they moved (2026-11-25).

    python tools/movement_benchmark.py <tracking.jsonl> <truth.json> [--json]

The scoreboard said "Movement: not measured yet: nothing marked to compare it
with". Something was marked: the who-is-who answer keys
(dataset/regression/identity_*/) give each fighter's box, drawn by a person,
every few seconds of each clip. The movement numbers on a report - pressure,
time in the centre, how far and fast each fighter moved - are all built from
where each fighter stood, so this scores that, two ways:

  right place      on each marked standing frame, the fighter's tracked floor
                   point (the bottom middle of their box, where they stand) is
                   within PLACE_TOLERANCE of their height from the marked one.
                   No box, or a box on someone else, is a wrong place.
  right direction  between two marked frames up to MAX_GAP_SECONDS apart, a
                   fighter who clearly moved (at least MOVE_THRESHOLD of their
                   height along the line to the opponent) was tracked moving
                   the same way, toward or away. This is what pressure reads.

It cannot score time in the centre: nothing marks where the ring is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.identity_benchmark import _box, _nearest  # noqa: E402

PLACE_TOLERANCE = 0.25      # of the fighter's marked height
MOVE_THRESHOLD = 0.25       # of their height, along the line to the opponent
MAX_GAP_SECONDS = 6.0


def _floor(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, float(box[3]))


def _height(box) -> float:
    return max(1.0, float(box[3]) - float(box[1]))


def _toward(first_self, first_other, later_self) -> float:
    """How far `later_self` moved from `first_self` along the line to the opponent, in pixels."""
    (x0, y0), (ox, oy), (x1, y1) = _floor(first_self), _floor(first_other), _floor(later_self)
    dx, dy = ox - x0, oy - y0
    norm = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    return ((x1 - x0) * dx + (y1 - y0) * dy) / norm


def score(records: list[dict], truth: dict) -> dict:
    marked = [f for f in truth.get("frames") or [] if f.get("phase", "standing") == "standing"]
    place = {"right": 0, "frames": 0}
    direction = {"right": 0, "moves": 0}
    tracked: list[tuple[dict, dict | None]] = []
    for frame in marked:
        record = _nearest(records, int(frame["source_frame"]))
        tracked.append((frame, record))
        for fighter in ("A", "B"):
            own = frame.get(fighter)
            if own is None:
                continue
            place["frames"] += 1
            box = _box(record, fighter) if record is not None else None
            if box is None:
                continue
            (tx, ty), (mx, my) = _floor(box), _floor(own)
            if ((tx - mx) ** 2 + (ty - my) ** 2) ** 0.5 <= PLACE_TOLERANCE * _height(own):
                place["right"] += 1
    for (first, rec0), (later, rec1) in zip(tracked, tracked[1:]):
        if float(later["time_seconds"]) - float(first["time_seconds"]) > MAX_GAP_SECONDS:
            continue
        for fighter, other in (("A", "B"), ("B", "A")):
            if any(f.get(k) is None for f in (first, later) for k in (fighter, other)):
                continue
            moved = _toward(first[fighter], first[other], later[fighter])
            if abs(moved) < MOVE_THRESHOLD * _height(first[fighter]):
                continue
            direction["moves"] += 1
            boxes = [_box(r, k) if r is not None else None for r, k in ((rec0, fighter), (rec0, other), (rec1, fighter))]
            if any(b is None for b in boxes):
                continue
            if _toward(*boxes) * moved > 0:
                direction["right"] += 1
    return {"place": place, "direction": direction}


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
    p, d = result["place"], result["direction"]
    print(f"right place: {p['right']}/{p['frames']}   right direction: {d['right']}/{d['moves']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
