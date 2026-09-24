"""Which proposals to label next, and why those.

131 proposals are unlabelled across the three packs. Labelling all of them is
hours of watching video, and most of that work would not change any decision
anybody is waiting on.

One decision is waiting: whether punch counting can be switched back on.
`core.report.STRIKE_COUNTS_PRECISION_VALIDATED` is False, which is what keeps
the score withheld on every analysis - scoring a round needs hands and feet,
so the punch head being off is what holds the scorecard shut.

The number that decides it is precision over the proposals the report would
actually **show**. The report publishes only strikes whose limb arrived, so a
proposal that missed changes the detector's overall precision and changes
nothing a reader sees. That cuts the queue hard:

    punch, arrived, unlabelled        30   <- decides it
    punch, missed, unlabelled         34
    kick, arrived, unlabelled         37
    kick, missed, unlabelled          30

So thirty answers settle the open question, and the other hundred are worth
having eventually rather than now.

Within those thirty, the order matters too. A pack where punches are already
well measured moves the total less than one where they are barely measured at
all, so the packs with the least evidence come first.

    python tools/label_queue.py                 # the queue, and what it buys
    python tools/label_queue.py --all           # every unlabelled proposal
    python tools/label_queue.py --write         # writes each pack's queue.json

`--write` leaves a `queue.json` beside the pack that tools/serve_label_pack.py
reads, so the labelling page offers these first instead of walking the pack in
file order.
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

from core.report import ARRIVED_OUTCOMES
from tools.evaluate_strike_counts import (
    CONFIRMED, REJECTED, _pack_path, discover, family_of,
)

# What the report would show. A proposal that missed is measured, stored, and
# never displayed, so labelling it cannot move the number that decides whether
# punches are published.
DECIDES = ("punch", "arrived")


def _answered_in_pack(pack: Path) -> set[int]:
    """Ids somebody already answered through the label page.

    There are two answer stores for the same clips and this queue read only
    one. tools/labels_*.json holds the verdicts the evaluation harness reads;
    labelpack/<pack>/<pack>-labels.json holds whatever was answered in the
    browser, written straight by the label server.

    Nothing joined them, so the queue offered ten cropmotion clips that all
    had answers already - the page opened, saw every one of them answered and
    said "that's all of them". A queue that asks for work already done is
    worse than no queue, because it is believed.

    Unsure and wrong-person count as answered here. They are not evidence,
    and the harness is right to ignore them, but somebody has already looked
    at that clip and said what they could - asking again spends the one thing
    this queue exists to save.
    """
    path = pack / f"{pack.name}-labels.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    seen: set[int] = set()
    for key in ("labels", "wrong_person"):
        for item in data.get(key) or []:
            try:
                seen.add(int((item or {}).get("id")))
            except (TypeError, ValueError):
                continue
    for item in data.get("unsure") or []:
        try:
            seen.add(int(item))
        except (TypeError, ValueError):
            continue
    return seen


def _unlabelled(labels_path: Path) -> tuple[str, list[dict], dict]:
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    pack = _pack_path(str(data.get("pack") or ""))
    index = pack / "index.json"
    if not index.exists():
        return pack.name, [], {}
    candidates = {
        int(c["id"]): c
        for c in json.loads(index.read_text(encoding="utf-8")).get("candidates", [])
        if c.get("source") == "event"
    }
    judged = {
        int(l["id"]) for l in data.get("labels", []) if "id" in l
        and str(l.get("verdict") or "").strip().lower() in {CONFIRMED, REJECTED}
    }
    judged |= _answered_in_pack(pack)
    out = []
    for identifier, candidate in candidates.items():
        if identifier in judged:
            continue
        family = family_of(str(candidate.get("proposed") or ""))
        arrived = str(candidate.get("outcome") or "") in ARRIVED_OUTCOMES
        out.append({
            "id": identifier,
            "pack": pack.name,
            "family": family,
            "arrived": arrived,
            "decides": (family, "arrived" if arrived else "missed") == DECIDES,
            "at": round(float(candidate.get("peak_time") or 0.0), 2),
            "proposed": candidate.get("proposed"),
            "fighter": candidate.get("fighter"),
        })
    # How much punch evidence this pack already has. A pack that is nearly
    # unmeasured moves the total most per answer, so it goes first.
    judged_punches = sum(
        1 for identifier in judged
        if family_of(str(candidates.get(identifier, {}).get("proposed") or "")) == "punch"
    )
    return pack.name, out, {"judged_punches": judged_punches}


def build(everything: bool = False) -> dict[str, list[dict]]:
    packs: dict[str, list[dict]] = {}
    weight: dict[str, int] = {}
    for labels_path in discover():
        name, rows, meta = _unlabelled(labels_path)
        if not rows:
            continue
        wanted = rows if everything else [r for r in rows if r["decides"]]
        if not wanted:
            continue
        # Earliest first inside a pack: labelling in time order means watching
        # the round once rather than scrubbing back and forth.
        packs[name] = sorted(wanted, key=lambda r: r["at"])
        weight[name] = meta.get("judged_punches", 0)
    return {name: packs[name] for name in sorted(packs, key=lambda n: weight.get(n, 0))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--all", action="store_true",
                        help="every unlabelled proposal, not just the deciding ones")
    parser.add_argument("--write", action="store_true",
                        help="write queue.json beside each pack")
    arguments = parser.parse_args()

    queue = build(everything=arguments.all)
    if not queue:
        print("Nothing left to label.")
        return 0

    counted = Counter()
    for labels_path in discover():
        _, rows, _ = _unlabelled(labels_path)
        for row in rows:
            counted[(row["family"], "arrived" if row["arrived"] else "missed")] += 1

    total = sum(len(rows) for rows in queue.values())
    print("Unlabelled proposals, by what an answer would move:\n")
    for key in sorted(counted):
        marker = "  <- decides whether punches can be shown" if key == DECIDES else ""
        print(f"  {key[0]:6} {key[1]:8} {counted[key]:3}{marker}")
    print()
    print(f"Queue: {total} answer{'' if total == 1 else 's'}"
          f"{'' if arguments.all else ' - the rest are worth having, but not now'}.\n")

    for name, rows in queue.items():
        print(f"  {name}  ({len(rows)})")
        for row in rows[:6]:
            print(f"    id {row['id']:>4}  {row['at']:>7.2f}s  fighter {row['fighter']}"
                  f"  proposed {row['proposed']}")
        if len(rows) > 6:
            print(f"    ... and {len(rows) - 6} more")
        if arguments.write:
            target = PROJECT_ROOT / "labelpack" / name / "queue.json"
            target.write_text(json.dumps({
                "schema": "warrioriq.label_queue.v1",
                "why": "punch proposals that arrived - the set that decides whether "
                       "punch counting can be published",
                "ids": [row["id"] for row in rows],
            }, indent=2) + "\n", encoding="utf-8")
            print(f"    wrote {target.relative_to(PROJECT_ROOT)}")
        print()

    if arguments.write:
        # A pack that has run out of deciding clips must lose its queue file,
        # or the label server keeps advertising work that is finished. This is
        # how cropmotion came to offer ten clips that all had answers.
        for stale in sorted((PROJECT_ROOT / "labelpack").glob("*/queue.json")):
            if stale.parent.name not in queue:
                stale.unlink()
                print(f"  {stale.parent.name}: nothing left to decide, "
                      f"removed {stale.relative_to(PROJECT_ROOT)}")
        print()

    print("Then re-measure:")
    print("  python tools/evaluate_strike_counts.py --baseline dataset/strike_counts_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
