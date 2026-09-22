"""How many real strikes reach the report, and is the loss even between fighters?

Precision says what fraction of published moments are real. Scoring needs the
other half of the question: what fraction of real strikes are published, and -
because a score is a comparison - whether the two fighters lose the same share.
A count that misses half of everything can still name the right winner. A count
that misses half of ONE fighter cannot.

Measured on athens_hd, the only fight with hand labels:

    fighter   real   published   recall
    A         19     11          58%
    B         13      5          38%

    punch A    8      5          62%
    punch B   10      5          50%
    kick  A   11      6          55%
    kick  B    3      0           0%     <- an entire category, gone

The winner survived - A led by six either way - but that is luck, not a
property. Twenty points of recall difference between the fighters, and one
whole category of B's offence missing, would turn a close round the wrong way.
This is why the scorecard stays shut while the evidence list is published: the
list only has to be true, a score has to be complete.

**This is recall against PROPOSALS.** A strike the detector never proposed is
invisible to it, so every figure here is an upper bound on the truth.

    tools/measure_strike_recall.py --labels tools/labels_athens_hd_claude.json \
        --pack labelpack/athens_hd --report outputs/athens_hd/report.html
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"<tr><td>(\d+|-)</td><td>([\d.]+)</td><td>([AB])</td><td>([^<]+)</td></tr>")


def real_strikes(labels_path: Path, pack_path: Path) -> list[dict]:
    """Every event a human confirmed as a real strike by the named fighter."""
    labels = {int(l["id"]): l for l in
              json.loads(labels_path.read_text(encoding="utf-8"))["labels"]}
    candidates = {int(c["id"]): c for c in
                  json.loads((pack_path / "index.json").read_text(encoding="utf-8"))["candidates"]}
    out = []
    for identifier, label in labels.items():
        candidate = candidates.get(identifier)
        if not candidate or candidate.get("source") != "event":
            continue
        why = label.get("why") or ""
        if label.get("verdict") != "strike" or "WRONG FIGHTER" in why:
            continue
        proposed = str(candidate["proposed"])
        family = "kick" if ("kick" in proposed or "knee" in proposed) else "punch"
        # The labeller writes FAMILY WRONG where the proposal named the other
        # one; the label is the truth and the proposal is not.
        if "FAMILY WRONG" in why:
            family = "punch" if family == "kick" else "kick"
        out.append({"time": round(float(candidate["peak_time"]), 2),
                    "side": str(candidate["fighter"]), "family": family,
                    "outcome": candidate.get("outcome")})
    return out


def published_moments(report_html: Path) -> set:
    block = report_html.read_text(encoding="utf-8")
    if "Evidence timeline" not in block:
        return set()
    block = block.split("Evidence timeline")[1].split("</section>")[0]
    return {(round(float(t), 2), side) for _round, t, side, _what in ROW.findall(block)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--report", required=True)
    arguments = parser.parse_args()

    truth = real_strikes(PROJECT_ROOT / arguments.labels, PROJECT_ROOT / arguments.pack)
    published = published_moments(PROJECT_ROOT / arguments.report)
    if not truth:
        print("no confirmed strikes in those labels")
        return 1

    def line(name: str, rows: list[dict]) -> tuple[int, int]:
        hit = [r for r in rows if (r["time"], r["side"]) in published]
        share = 100.0 * len(hit) / len(rows) if rows else 0.0
        print("  %-16s %3d real  %3d published   recall %3.0f%%" % (name, len(rows), len(hit), share))
        return len(rows), len(hit)

    print("recall against the events a human confirmed (an upper bound - see the docstring)\n")
    per_side = {}
    for side in ("A", "B"):
        per_side[side] = line("fighter %s" % side, [r for r in truth if r["side"] == side])
    print()
    for family in ("punch", "kick"):
        for side in ("A", "B"):
            line("%s %s" % (family, side),
                 [r for r in truth if r["side"] == side and r["family"] == family])

    (real_a, published_a), (real_b, published_b) = per_side["A"], per_side["B"]
    print()
    print("  truth     : A %d, B %d" % (real_a, real_b))
    print("  published : A %d, B %d" % (published_a, published_b))
    gap = abs((100.0 * published_a / max(1, real_a)) - (100.0 * published_b / max(1, real_b)))
    print("  recall differs between the fighters by %.0f points" % gap)
    if gap >= 10:
        print("  -> NOT a basis for a score: the loss is uneven, so a close round")
        print("     would be decided by which fighter the detector sees better.")
    missed = Counter(r["outcome"] for r in truth if (r["time"], r["side"]) not in published)
    print("\n  missed strikes, by the outcome the analyser gave them: %s" % dict(missed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
