"""Score the strike count against fights where every strike is labelled.

tools/benchmark_labelled_fight.py scores the detector on moments it once
proposed, so it says how many counted strikes are real but only gives a floor
on the ones it never found. Here every strike thrown in a stretch of video is
labelled, so both are measured: of the strikes the report counts, how many
were thrown (precision), and of the strikes thrown, how many it counts
(recall).

    python tools/benchmark_dense_strikes.py
    python tools/benchmark_dense_strikes.py --ruleset BOXING --json

Each fight is a folder in dataset/regression/ with:

  track.jsonl.gz   the analyser's tracking.jsonl for the labelled stretch
  strikes.json     every strike thrown: time of fullest extension, fighter,
                   family, outcome, and whether the labeller was sure; plus
                   the spans that could not be judged

The action path is replayed from the track (as benchmark_labelled_fight does),
then the report's own attempt gate (core.analyzer._live_attempt_reliable) and
one-label-per-moment rule decide what is counted. A counted strike matches a
labelled one of the same fighter within MATCH_SECONDS, nearest first, each
used once. A counted strike near an unsure label, or inside an unreadable
span, is left out of both sides rather than guessed at.

What it cannot do: the fight-presence and kit re-attribution steps read the
video, so they are not replayed. Both only remove or move strikes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.benchmark_labelled_fight import replay  # noqa: E402

MATCH_SECONDS = 0.35
OUTCOME_OF = {"clean": "landed", "likely_landed": "landed", "blocked": "blocked", "checked": "blocked",
              "missed": "missed"}


def dense_fights() -> list[Path]:
    root = PROJECT_ROOT / "dataset" / "regression"
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "track.jsonl.gz").exists() and (p / "strikes.json").exists())


def counted_events(events: list, ruleset: str | None) -> list:
    from core.analyzer import _live_attempt_reliable
    from core.scoring import collapse_simultaneous_labels

    kept = [event for event in events if _live_attempt_reliable(event, ruleset)]
    kept, _removed = collapse_simultaneous_labels(kept)
    return kept


def _inside(t: float, spans) -> bool:
    return any(start <= t <= end for start, end in spans)


def score(events: list, truth: dict) -> dict:
    start, end = truth["covered"]
    spans = truth.get("unreadable") or []
    labels = [s for s in truth["strikes"] if start <= s["t"] <= end and not _inside(s["t"], spans)]
    counted = [e for e in events if start <= float(e.peak_time) <= end and not _inside(float(e.peak_time), spans)]
    pairs = []
    for i, event in enumerate(counted):
        for j, label in enumerate(labels):
            gap = abs(float(event.peak_time) - label["t"])
            if label["fighter"] == event.fighter and gap <= MATCH_SECONDS:
                pairs.append((gap, i, j))
    used_events, used_labels, matched = set(), set(), []
    for _, i, j in sorted(pairs):
        if i in used_events or j in used_labels:
            continue
        used_events.add(i)
        used_labels.add(j)
        matched.append((counted[i], labels[j]))
    sure_matched = [(e, s) for e, s in matched if s.get("sure", True)]
    unmatched = [e for i, e in enumerate(counted) if i not in used_events]
    # An unmatched count next to an unsure label is not called wrong.
    unsure_near = [e for e in unmatched if any(
        not s.get("sure", True) and s["fighter"] == e.fighter and abs(s["t"] - float(e.peak_time)) <= MATCH_SECONDS
        for s in labels)]
    false = len(unmatched) - len(unsure_near)
    sure_labels = [s for s in labels if s.get("sure", True)]
    found = len(sure_matched)
    outcomes = [(OUTCOME_OF.get(str(e.outcome)), s.get("outcome")) for e, s in sure_matched
                if s.get("outcome") in {"landed", "blocked", "missed"}]
    result = {
        "watched_seconds": round(end - start - sum(b - a for a, b in spans), 1),
        "labelled": len(sure_labels),
        "counted": found + false,
        "real": found,
        "false": false,
        "missed": len(sure_labels) - found,
        "family_right": sum(1 for e, s in sure_matched if e.family == s.get("family")),
        "outcome_checked": len(outcomes),
        "outcome_right": sum(1 for got, want in outcomes if got == want),
    }
    for fighter in ("A", "B"):
        mine = [s for s in sure_labels if s["fighter"] == fighter]
        result[f"labelled_{fighter}"] = len(mine)
        result[f"counted_{fighter}"] = sum(1 for e, _ in sure_matched if e.fighter == fighter) + sum(
            1 for e in unmatched if e.fighter == fighter and e not in unsure_near)
    return result


def _shares(row: dict) -> dict:
    row["precision"] = round(row["real"] / row["counted"], 3) if row["counted"] else None
    row["recall"] = round(row["real"] / row["labelled"], 3) if row["labelled"] else None
    return row


def run(ruleset: str | None) -> dict:
    results = {}
    total: dict = {}
    for fight in dense_fights():
        truth = json.loads((fight / "strikes.json").read_text(encoding="utf-8"))
        row = score(counted_events(replay(fight / "track.jsonl.gz"), ruleset), truth)
        results[fight.name] = _shares(row)
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"precision", "recall"}:
                total[key] = round(total.get(key, 0) + value, 1)
    results["TOTAL"] = _shares(total)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ruleset", default=None, help="score as the report for this ruleset counts (e.g. BOXING)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = run(args.ruleset)
    if args.json:
        print(json.dumps(results, indent=1))
        return 0
    pct = lambda v: "-" if v is None else f"{v * 100:.0f}%"  # noqa: E731
    print(f"{'fight':34} {'labelled':>8} {'counted':>8} {'real':>5} {'false':>6} {'missed':>7} "
          f"{'precision':>9} {'recall':>7} {'outcome':>8}")
    for name, row in results.items():
        outcome = f"{row['outcome_right']}/{row['outcome_checked']}"
        print(f"{name:34} {row['labelled']:>8} {row['counted']:>8} {row['real']:>5} {row['false']:>6} "
              f"{row['missed']:>7} {pct(row['precision']):>9} {pct(row['recall']):>7} {outcome:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
