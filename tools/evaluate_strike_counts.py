"""Score strike counts against hand labels, across every labelled fight at once.

Why this exists
---------------
The punch count is the number fighters most want and the one WarriorIQ does
not show. It is switched off by a single constant,
`core.report.STRIKE_COUNTS_PRECISION_VALIDATED`, because hand-checking three
bouts against the video found punches overstated by eleven in two of them.

That decision was right, and it was made from a hand count written into a
docstring. Turning it back on needs a number that can be *re-measured* after a
change, otherwise "the punch head got better" is a feeling. This is that
measurement.

`tools/measure_strike_recall.py` answers the other half - what share of real
strikes reach the published report - for one fight at a time, reading the
report's HTML. This one answers precision and count error, over every labelled
fight, reading the proposals themselves. Both halves are needed: a count that
misses half of everything can still name the right winner, and a count that
invents half of everything cannot.

What it measures, and what it cannot
------------------------------------
The labels are verdicts on candidates the detector *proposed*. So:

  * **Precision is real.** Of the proposals a human looked at, the share that
    were a genuine strike by the fighter named is measured directly.

  * **Over-count is real**, on the labelled subset: proposals minus confirmed.
    This is the "overstated by eleven" quantity.

  * **Recall is not measurable here at all.** A strike the detector never
    proposed has no candidate and therefore no label, so nothing in this file
    can see it. Use measure_strike_recall.py, and read its upper-bound
    warning.

A proposal is counted as correct only when the human confirmed a strike *and*
did not mark the fighter wrong. Getting the technique right on the wrong
fighter is not a correct count - it moves a strike from one column to the
other, which is worse for a score than dropping it.

`FAMILY WRONG` in a label's reason means the proposal named punch for a kick
or the reverse. The label is the truth, so the event counts against the family
that was proposed and for the family that was real.

Usage
-----
    python tools/evaluate_strike_counts.py
    python tools/evaluate_strike_counts.py --json outputs/strike_counts.json

Writing the JSON and committing it is what makes the next run a comparison
rather than a fresh opinion. --baseline prints the change against one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABELS_DIR = PROJECT_ROOT / "tools"
PACKS_DIR = PROJECT_ROOT / "labelpack"

# A label's verdict. "unsure" is deliberately neither right nor wrong: counting
# it either way would be inventing a human judgement that was withheld on
# purpose, and the count of them is reported so the reader can see how much was
# set aside.
CONFIRMED = "strike"
REJECTED = "none"
UNSURE = "unsure"

# The family a technique name belongs to. Knees sit with kicks because the
# report already treats them as one family: judged by eye a knee and a round
# kick are a coin flip.
KICK_WORDS = ("kick", "knee")

# What the report actually publishes. core.report.ARRIVED_OUTCOMES keeps only
# the events whose limb reached the opponent; `missed` and `uncertain` are
# measured, stored, and never shown. Precision over all proposals is therefore
# not the precision a reader sees, and comparing punches against kicks on the
# wrong one of those is how a real difference gets missed.
ARRIVED_OUTCOMES = frozenset({"clean", "likely_landed", "blocked"})


def family_of(technique: str) -> str:
    name = (technique or "").lower()
    return "kick" if any(word in name for word in KICK_WORDS) else "punch"


@dataclass
class Tally:
    """One family's count on one fight, or summed across fights."""

    proposed: int = 0        # proposals a human looked at
    confirmed: int = 0       # of those, a real strike by the named fighter
    wrong_fighter: int = 0   # real, but attributed to the other fighter
    wrong_family: int = 0    # real, but a punch called a kick or the reverse
    unsure: int = 0          # set aside by the labeller
    unlabelled: int = 0      # proposed but never looked at
    # The same two counts, restricted to proposals the report would publish.
    arrived_judged: int = 0
    arrived_confirmed: int = 0

    @property
    def judged(self) -> int:
        """Proposals carrying a usable verdict.

        Both exclusions matter and one of them was got wrong first time round.
        A proposal nobody looked at is not evidence of anything, and counting
        it as a miss put pack_1mp4's punch precision at 2% - 44 of its 49
        proposals had simply never been labelled. An "unsure" is a judgement
        deliberately withheld, so it is not evidence either. Both are reported
        separately instead of being folded into the score.
        """
        return self.proposed - self.unsure - self.unlabelled

    @property
    def precision(self) -> float | None:
        return self.confirmed / self.judged if self.judged else None

    @property
    def over_count(self) -> int:
        """How many more strikes were claimed than were there.

        The audit's "overstated by eleven", on the labelled subset.
        """
        return self.judged - self.confirmed

    @property
    def arrived_precision(self) -> float | None:
        return (self.arrived_confirmed / self.arrived_judged
                if self.arrived_judged else None)

    def merge(self, other: "Tally") -> None:
        for name in ("proposed", "confirmed", "wrong_fighter", "wrong_family",
                     "unsure", "unlabelled", "arrived_judged", "arrived_confirmed"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict:
        return {
            "proposed": self.proposed, "judged": self.judged,
            "confirmed": self.confirmed, "over_count": self.over_count,
            "precision": None if self.precision is None else round(self.precision, 4),
            "wrong_fighter": self.wrong_fighter, "wrong_family": self.wrong_family,
            "unsure": self.unsure, "unlabelled": self.unlabelled,
            "arrived_judged": self.arrived_judged,
            "arrived_confirmed": self.arrived_confirmed,
            "arrived_precision": (None if self.arrived_precision is None
                                  else round(self.arrived_precision, 4)),
        }


@dataclass
class FightResult:
    fight: str
    pack: str
    labels_file: str
    source: dict = field(default_factory=dict)
    families: dict[str, Tally] = field(default_factory=dict)
    per_fighter: dict[str, Tally] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "fight": self.fight, "pack": self.pack, "labels_file": self.labels_file,
            "source": dict(self.source),
            "families": {k: v.as_dict() for k, v in sorted(self.families.items())},
            "fighters": {k: v.as_dict() for k, v in sorted(self.per_fighter.items())},
        }


def discover(labels_dir: Path = LABELS_DIR) -> list[Path]:
    """Every labels file that names a pack and carries verdicts."""
    found = []
    for path in sorted(labels_dir.glob("labels_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("labels") and data.get("pack"):
            found.append(path)
    return found


def provenance(data: dict) -> dict:
    """Who made these labels, and whether they are ground truth.

    Every label set in this repository was written by Claude and says so in
    its own header - "MACHINE-GENERATED, by Claude... NOT human ground truth.
    Kept separate from human labels for that reason." This tool read them all
    as though a person had made them, and reported a precision figure that
    was really one model grading another model's output.

    That is not a small caveat. The question these numbers are asked to
    settle is whether a detector is right often enough to publish, and an
    answer produced by the same family of model shares its blind spots -
    particularly on the exact case in dispute, a few pixels of arm movement.

    So provenance travels with the number from here on.
    """
    described = data.get("_what_this_is")
    text = " ".join(described) if isinstance(described, list) else str(described or "")
    labeller = str(data.get("labeller") or "unknown")
    machine = "MACHINE-GENERATED" in text or "claude" in labeller.lower()
    return {"labeller": labeller, "machine_generated": machine}


def thinness(result: "FightResult") -> float:
    """What share of this pack's proposals nobody ever ruled on.

    This replaced a phrase match, which got it backwards. The first version
    looked for "could not tell" in a pack's header and flagged athens_hd -
    whose header contains that phrase while describing a *different* pack,
    to explain why that one is weak and this one is not. A heuristic that
    reads English and misattributes it is worse than no heuristic.

    The harness already counts what matters, so it counts it: a pack where
    most proposals were never judged is thin evidence whatever its header
    says, and that is arithmetic rather than reading comprehension.
    """
    proposed = sum(t.proposed for t in result.families.values())
    judged = sum(t.judged for t in result.families.values())
    return 0.0 if not proposed else 1.0 - (judged / proposed)


def _pack_path(pack: str) -> Path:
    """Packs are named either "athens_hd" or "labelpack/cropmotion"."""
    name = (pack or "").strip().replace("\\", "/")
    if name.startswith("labelpack/"):
        name = name[len("labelpack/"):]
    return PACKS_DIR / name


def evaluate(labels_path: Path) -> FightResult | None:
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    pack_dir = _pack_path(str(data.get("pack") or ""))
    index = pack_dir / "index.json"
    if not index.exists():
        return None
    candidates = {
        int(c["id"]): c
        for c in json.loads(index.read_text(encoding="utf-8")).get("candidates", [])
        # Only proposals the action engine made. A pack also carries sampled
        # frames that were never claimed to be a strike, and scoring the
        # detector against those would flatter it.
        if c.get("source") == "event"
    }
    labels = {int(l["id"]): l for l in data["labels"] if "id" in l}

    result = FightResult(
        fight=str(data.get("fight") or pack_dir.name),
        pack=pack_dir.name,
        labels_file=labels_path.name,
        source=provenance(data),
    )

    for identifier, candidate in candidates.items():
        proposed_family = family_of(str(candidate.get("proposed") or ""))
        side = str(candidate.get("fighter") or "?")
        family_tally = result.families.setdefault(proposed_family, Tally())
        fighter_tally = result.per_fighter.setdefault(side, Tally())
        for tally in (family_tally, fighter_tally):
            tally.proposed += 1

        label = labels.get(identifier)
        if label is None:
            for tally in (family_tally, fighter_tally):
                tally.unlabelled += 1
            continue

        verdict = str(label.get("verdict") or "").strip().lower()
        why = str(label.get("why") or "").upper()
        arrived = str(candidate.get("outcome") or "").strip().lower() in ARRIVED_OUTCOMES
        if verdict == UNSURE or verdict not in {CONFIRMED, REJECTED}:
            for tally in (family_tally, fighter_tally):
                tally.unsure += 1
            continue
        if arrived:
            for tally in (family_tally, fighter_tally):
                tally.arrived_judged += 1
        if verdict == REJECTED:
            continue

        # Confirmed a strike. Two ways it can still be a wrong count.
        if "WRONG FIGHTER" in why:
            for tally in (family_tally, fighter_tally):
                tally.wrong_fighter += 1
            continue
        if "FAMILY WRONG" in why:
            family_tally.wrong_family += 1
            fighter_tally.wrong_family += 1
            # Real strike, right fighter, wrong family: it counts for the
            # fighter and against the family that claimed it.
            fighter_tally.confirmed += 1
            if arrived:
                fighter_tally.arrived_confirmed += 1
            continue
        for tally in (family_tally, fighter_tally):
            tally.confirmed += 1
            if arrived:
                tally.arrived_confirmed += 1

    return result


def _rate(value: float | None) -> str:
    return "  -  " if value is None else f"{100 * value:4.0f}%"


def render(results: list[FightResult]) -> str:
    lines: list[str] = []
    lines.append("Strike counts against hand labels")
    lines.append("=" * 74)
    lines.append("")
    lines.append("Precision and over-count are measured. Recall is NOT: a strike the")
    lines.append("detector never proposed has no label. See measure_strike_recall.py.")
    lines.append("")
    lines.append("\"shown\" is precision over the subset the report actually publishes -")
    lines.append("the events whose limb arrived. That is what a reader sees.")
    lines.append("")

    totals: dict[str, Tally] = {}
    for result in results:
        marks = []
        if result.source.get("machine_generated"):
            marks.append("machine-labelled by " + str(result.source.get("labeller")))
        unjudged = thinness(result)
        if unjudged >= 0.5:
            marks.append(f"{unjudged:.0%} of its proposals were never ruled on")
        suffix = ("  [" + "; ".join(marks) + "]") if marks else ""
        lines.append(f"{result.fight}  ({result.labels_file}){suffix}")
        lines.append(f"  {'family':<8}{'proposed':>9}{'judged':>8}{'real':>6}"
                     f"{'over':>6}{'precision':>11}{'shown':>9}{'unsure':>8}{'unseen':>8}")
        for name in ("punch", "kick"):
            tally = result.families.get(name)
            if not tally:
                continue
            lines.append(
                f"  {name:<8}{tally.proposed:>9}{tally.judged:>8}{tally.confirmed:>6}"
                f"{tally.over_count:>+6}{_rate(tally.precision):>11}"
                f"{_rate(tally.arrived_precision):>9}"
                f"{tally.unsure:>8}{tally.unlabelled:>8}")
            totals.setdefault(name, Tally()).merge(tally)
        wrong = sum(t.wrong_fighter for t in result.families.values())
        if wrong:
            lines.append(f"  {wrong} confirmed strike(s) were attributed to the wrong fighter")
        lines.append("")

    lines.append("-" * 74)
    lines.append("All labelled fights")
    lines.append(f"  {'family':<8}{'proposed':>9}{'judged':>8}{'real':>6}"
                 f"{'over':>6}{'precision':>11}{'shown':>9}{'unsure':>8}{'unseen':>8}")
    for name in ("punch", "kick"):
        tally = totals.get(name)
        if not tally:
            continue
        lines.append(
            f"  {name:<8}{tally.proposed:>9}{tally.judged:>8}{tally.confirmed:>6}"
            f"{tally.over_count:>+6}{_rate(tally.precision):>11}"
            f"{_rate(tally.arrived_precision):>9}"
            f"{tally.unsure:>8}{tally.unlabelled:>8}")
    lines.append("")

    punch, kick = totals.get("punch"), totals.get("kick")
    if punch and punch.judged:
        lines.append(f"Punch precision is {_rate(punch.precision).strip()} over "
                     f"{punch.judged} judged proposals, overstating by "
                     f"{punch.over_count}.")
    if kick and kick.judged:
        lines.append(f"Kick precision is {_rate(kick.precision).strip()} over "
                     f"{kick.judged} judged proposals, overstating by "
                     f"{kick.over_count}.")
    if punch and kick and punch.judged and kick.judged:
        lines.append("")
        lines.append("Punches are withheld and kicks are published. On these labels that")
        gap = (punch.precision or 0) - (kick.precision or 0)
        lines.append(f"difference is {abs(100 * gap):.0f} points of precision "
                     f"({'punches ahead' if gap > 0 else 'kicks ahead'}), which is")
        lines.append("worth re-reading before the split is treated as settled.")
    lines.append("")
    lines.append("core.report.STRIKE_COUNTS_PRECISION_VALIDATED is the switch this")
    lines.append("number governs. Nothing here turns it on by itself - that is a")
    lines.append("decision about what is good enough to publish, not a computation.")
    if results and all(r.source.get("machine_generated") for r in results):
        lines.append("")
        lines.append("!! EVERY LABEL ABOVE WAS WRITTEN BY A MODEL, NOT A PERSON.")
        lines.append("   Each label file says so in its own header. These numbers are")
        lines.append("   one model grading another model's output, and the two share")
        lines.append("   blind spots on exactly the case in dispute - whether a few")
        lines.append("   pixels of arm movement was a punch. A signal worth following,")
        lines.append("   never the evidence that settles it.")
    return "\n".join(lines)


def summarise(results: list[FightResult]) -> dict:
    totals: dict[str, Tally] = {}
    for result in results:
        for name, tally in result.families.items():
            totals.setdefault(name, Tally()).merge(tally)
    return {
        "schema": "warrioriq.strike_counts.v1",
        "fights": [r.as_dict() for r in results],
        "totals": {k: v.as_dict() for k, v in sorted(totals.items())},
        # Travels with the numbers, so a baseline committed today cannot be
        # read next month as though a person had produced it.
        "label_sources": {r.pack: dict(r.source) for r in results},
        "human_ground_truth": not all(r.source.get("machine_generated") for r in results),
        "measures": {
            "precision": "confirmed / judged, over proposals a human ruled on",
            "over_count": "judged - confirmed, on the labelled subset",
            "recall": "not measurable from labels on proposals; see measure_strike_recall.py",
        },
    }


def _compare(current: dict, baseline: dict) -> str:
    lines = ["", "Change against baseline", "-" * 74]
    for name in ("punch", "kick"):
        now = current["totals"].get(name)
        was = baseline.get("totals", {}).get(name)
        if not now or not was:
            continue
        a, b = was.get("precision"), now.get("precision")
        moved = "" if a is None or b is None else f"{100 * (b - a):+.1f} points"
        lines.append(f"  {name:<6} precision {a} -> {b}  {moved}")
        lines.append(f"  {name:<6} over-count {was.get('over_count')} -> {now.get('over_count')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", help="write the machine-readable summary here")
    parser.add_argument("--baseline", help="an earlier --json file to compare against")
    arguments = parser.parse_args()

    results = [r for r in (evaluate(p) for p in discover()) if r is not None]
    if not results:
        print("No labelled fights found. Label packs live in labelpack/ and their "
              "verdicts in tools/labels_*.json.")
        return 1

    print(render(results))
    summary = summarise(results)

    if arguments.baseline:
        baseline_path = Path(arguments.baseline)
        if baseline_path.exists():
            print(_compare(summary, json.loads(baseline_path.read_text(encoding="utf-8"))))
        else:
            print(f"\nNo baseline at {baseline_path}; this run can become one with --json.")

    if arguments.json:
        out = Path(arguments.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
