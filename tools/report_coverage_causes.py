"""Why each fighter was lost, side by side, from a stored analysis.

The red corner is tracked worse than the blue one. Across four sampled
reports, fighter A came in at 64-74% coverage against B's 82-88%, and the two
were never once close. That is reproducible enough to be a bug rather than
variance, but a bug with no named cause, and picking a fix before naming one
is how the three attempts recorded in project-detection-not-identity went.

`blocked_recovery_reasons` in a stored report already counts why *somebody*
was unassigned. It pools A and B, so it cannot answer this question at all: a
cause that is most of A's losses and none of B's looks exactly like one shared
evenly between them, and those need opposite fixes. `blocked_recovery_by_
fighter` splits them, and this prints the split with each fighter's missing
frames as the denominator, so the shares are comparable.

    python tools/report_coverage_causes.py outputs/*/report.json

A caution that applies to every number this prints: runs are not reliably
repeatable unless the stride is pinned. Fighter B's coverage on fight 1 has
come out at both 0.26 and 0.73 from identical inputs, because the stride is
chosen off a wall clock. Set WARRIORIQ_FORCE_STRIDE before comparing two runs,
or the difference you read will be the clock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# How different two fighters' shares of one cause must be before it is worth
# pointing at. Below this the cause is affecting both corners about equally,
# so it is not what makes them differ - whatever else it may be doing.
NOTABLE_GAP = 0.15


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def causes(report: dict) -> dict:
    tracking = report.get("tracking") or {}
    return {
        "coverage": {
            "A": float(tracking.get("fighter_A_coverage") or 0.0),
            "B": float(tracking.get("fighter_B_coverage") or 0.0),
        },
        "missing": tracking.get("missing_frames_by_fighter") or {},
        "by_fighter": tracking.get("blocked_recovery_by_fighter") or {},
        "pooled": tracking.get("blocked_recovery_reasons") or {},
        "analyzed_frames": int(tracking.get("analyzed_frames") or 0),
    }


def render(name: str, data: dict) -> str:
    lines = [f"{name}", "=" * 72]
    coverage = data["coverage"]
    lines.append(f"  coverage      A {coverage['A']:.0%}      B {coverage['B']:.0%}"
                 f"      gap {abs(coverage['A'] - coverage['B']):.0%}")
    missing = data["missing"]
    if not data["by_fighter"] or not any(data["by_fighter"].values()):
        lines.append("")
        lines.append("  No per-fighter causes in this report. It was produced before")
        lines.append("  blocked_recovery_by_fighter existed; re-run the analysis to get")
        lines.append("  the breakdown. The pooled totals it does carry:")
        for reason, count in sorted(data["pooled"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:<34}{count:>7}")
        return "\n".join(lines)

    lines.append(f"  frames missing  A {missing.get('A', 0):<8} B {missing.get('B', 0)}")
    lines.append("")
    lines.append(f"  {'cause':<34}{'A':>7}{'share':>8}{'B':>7}{'share':>8}{'gap':>8}")

    reasons = sorted(set(data["by_fighter"].get("A", {})) | set(data["by_fighter"].get("B", {})))
    rows = []
    for reason in reasons:
        a = int(data["by_fighter"].get("A", {}).get(reason, 0))
        b = int(data["by_fighter"].get("B", {}).get(reason, 0))
        a_share = a / missing["A"] if missing.get("A") else 0.0
        b_share = b / missing["B"] if missing.get("B") else 0.0
        rows.append((abs(a_share - b_share), reason, a, a_share, b, b_share))
    # Biggest difference between the fighters first: that is the ordering the
    # question asks for, not the biggest total.
    for gap, reason, a, a_share, b, b_share in sorted(rows, reverse=True):
        lines.append(f"  {reason:<34}{a:>7}{a_share:>7.0%}{b:>7}{b_share:>7.0%}{gap:>8.0%}")

    lines.append("")
    leading = [r for r in sorted(rows, reverse=True) if r[0] >= NOTABLE_GAP]
    if not leading:
        lines.append("  No cause separates the two fighters by more than "
                     f"{NOTABLE_GAP:.0%} of their missing frames.")
        lines.append("  Whatever costs the red corner its coverage, it is not one of these")
        lines.append("  gates preferring one fighter - look upstream, at detection.")
    else:
        for gap, reason, a, a_share, b, b_share in leading:
            worse = "A" if a_share > b_share else "B"
            lines.append(f"  {reason} accounts for {max(a_share, b_share):.0%} of "
                         f"{worse}'s missing frames against "
                         f"{min(a_share, b_share):.0%} of the other's.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("reports", nargs="*", default=[],
                        help="report.json files; defaults to every one under outputs/")
    parser.add_argument("--json", help="write the collected causes here")
    arguments = parser.parse_args()

    paths = [Path(p) for p in arguments.reports] or sorted(
        (PROJECT_ROOT / "outputs").glob("*/report.json"))
    if not paths:
        print("No stored analyses found under outputs/.")
        return 1

    collected = {}
    for path in paths:
        report = _load(path)
        if report is None:
            print(f"{path}: not readable as JSON\n")
            continue
        data = causes(report)
        collected[path.parent.name] = data
        print(render(path.parent.name, data))
        print()

    print("Runs are not repeatable unless WARRIORIQ_FORCE_STRIDE is pinned; see")
    print("the module docstring before comparing two of these.")

    if arguments.json:
        out = Path(arguments.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(collected, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
