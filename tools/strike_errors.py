"""Where the strike count goes wrong, stage by stage (2026-11-24).

    python tools/strike_errors.py            # a table to paste back
    python tools/strike_errors.py --list     # also every miss and false count, one line each
    python tools/strike_errors.py --json

tools/benchmark_dense_strikes.py says how many strikes were missed or counted
falsely; this says why. It replays the same fights the same way (the stored
pose track, no video, as tools/scoreboard.py does) but keeps every stage of
the action path apart:

  raw      every action the engine proposed (core/action.py), contact classified
  aimed    ... thrown at the opponent (core.contact.thrown_at_opponent)
  paired   ... after the pair rule drops the defender's echo
           (core.contact.resolve_simultaneous_attribution)
  gated    ... passed by the report's attempt gate (core.analyzer._attempt_drop_reason)
  counted  ... one label per fighter per moment (core.scoring.collapse_simultaneous_labels)

Each sure label that no counted strike matched is put at the last stage where
a strike of the same fighter was still within the benchmark's window, or
called "never proposed", "proposed for the other fighter" or "proposed, but
too early or late". Each false count is called "the other fighter's strike",
"a second count of one strike", "near a strike, too early or late" or
"nothing labelled there".

It prints numbers only: times, fighters, families and gate values, no
pictures and no names, so its output can be pasted anywhere. It changes
nothing about how anything is counted.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import benchmark_dense_strikes as dense  # noqa: E402

MATCH = dense.MATCH_SECONDS
NEAR = 1.0          # "too early or late": a strike of the right fighter within a second


def stages(track: Path, ruleset: str | None) -> dict[str, list]:
    """Every event at every stage of the replayed action path."""
    from core.action import ActionEngine
    from core.analyzer import _attempt_drop_reason
    from core.contact import classify_contact, resolve_simultaneous_attribution, thrown_at_opponent
    from core.scoring import collapse_simultaneous_labels
    from tools.benchmark_labelled_fight import _observation

    engine, raw = ActionEngine(), []
    with gzip.open(track, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            a = _observation(record["fighter_A"]["observation"])
            b = _observation(record["fighter_B"]["observation"])
            conf_a = float(record["fighter_A"]["identity_confidence"])
            conf_b = float(record["fighter_B"]["identity_confidence"])
            args = (record["source_frame"], record["time_seconds"], record["round_number"])
            for event in engine.update("A", *args, a, b, conf_a, conf_b) + engine.update("B", *args, b, a, conf_b, conf_a):
                raw.append(classify_contact(event))
    aimed = [event for event in raw if thrown_at_opponent(event)]
    paired, _ = resolve_simultaneous_attribution(list(aimed))
    reasons = {id(event): _attempt_drop_reason(event, ruleset) for event in paired}
    gated = [event for event in paired if reasons[id(event)] is None]
    counted, _ = collapse_simultaneous_labels(list(gated))
    return {"raw": raw, "aimed": aimed, "paired": paired, "gated": gated, "counted": counted, "reasons": reasons}


def _near(events, fighter: str, t: float, window: float) -> list:
    return [event for event in events if event.fighter == fighter and abs(float(event.peak_time) - t) <= window]


def _other(fighter: str) -> str:
    return "B" if fighter == "A" else "A"


def diagnose(fight: Path) -> dict:
    truth = dense.load_truth(fight)
    ruleset = dense._fight_ruleset(fight)
    path = stages(fight / "track.jsonl.gz", ruleset)
    start, end = truth["covered"]
    spans = truth.get("unreadable") or []

    def watched(t: float) -> bool:
        return start <= t <= end and not dense._inside(t, spans)

    labels = [s for s in truth["strikes"] if watched(float(s["t"]))]
    counted = [e for e in path["counted"] if watched(float(e.peak_time))]

    # The benchmark's own matching, so the totals here are its totals.
    pairs = sorted((abs(float(e.peak_time) - s["t"]), i, j) for i, e in enumerate(counted)
                   for j, s in enumerate(labels) if s["fighter"] == e.fighter and abs(float(e.peak_time) - s["t"]) <= MATCH)
    used_e, used_l = set(), set()
    for _, i, j in pairs:
        if i not in used_e and j not in used_l:
            used_e.add(i)
            used_l.add(j)

    misses, falses = [], []
    for j, label in enumerate(labels):
        if j in used_l or not label.get("sure", True):
            continue
        t, fighter = float(label["t"]), label["fighter"]
        if _near(path["counted"], fighter, t, MATCH):
            why = "a counted strike here went to another label"
        elif _near(path["gated"], fighter, t, MATCH):
            why = "one label per moment (collapsed)"
        elif _near(path["paired"], fighter, t, MATCH):
            reason = Counter(path["reasons"][id(e)] for e in _near(path["paired"], fighter, t, MATCH)).most_common(1)[0][0]
            why = f"attempt gate: {reason}"
        elif _near(path["aimed"], fighter, t, MATCH):
            why = "pair rule (taken as the defender's echo)"
        elif _near(path["raw"], fighter, t, MATCH):
            why = "not aimed at the opponent"
        elif _near(path["raw"], _other(fighter), t, MATCH):
            why = "proposed for the other fighter"
        elif _near(path["raw"], fighter, t, NEAR):
            why = "proposed, but too early or late"
        else:
            why = "never proposed"
        closest = min(path["raw"], key=lambda e: abs(float(e.peak_time) - t), default=None)
        misses.append({"t": round(t, 2), "fighter": fighter, "family": label.get("family"), "why": why,
                       "nearest_raw": None if closest is None else _brief(closest, t)})

    unsure = [s for s in labels if not s.get("sure", True)]
    for i, event in enumerate(counted):
        if i in used_e:
            continue
        t, fighter = float(event.peak_time), event.fighter
        if any(s["fighter"] == fighter and abs(s["t"] - t) <= MATCH for s in unsure):
            continue        # the benchmark does not call these wrong either
        if any(s["fighter"] == fighter and abs(s["t"] - t) <= MATCH for s in labels):
            why = "a second count of one strike"
        elif any(s["fighter"] == _other(fighter) and abs(s["t"] - t) <= MATCH for s in labels):
            why = "the other fighter's strike"
        elif any(s["fighter"] == fighter and abs(s["t"] - t) <= NEAR for s in labels):
            why = "near a strike, too early or late"
        else:
            why = "nothing labelled there"
        falses.append({"t": round(t, 2), "fighter": fighter, "why": why, **_brief(event, t)})

    return {"ruleset": ruleset, "labelled": sum(1 for s in labels if s.get("sure", True)),
            "counted": len(counted) - sum(1 for i, e in enumerate(counted) if i not in used_e and any(
                s["fighter"] == e.fighter and abs(s["t"] - float(e.peak_time)) <= MATCH for s in unsure)),
            "real": len(used_e), "stage_sizes": {k: len(v) for k, v in path.items() if k != "reasons"},
            "misses": misses, "false": falses}


def _brief(event, t: float) -> dict:
    """The numbers a gate looks at, rounded: enough to see why, nothing identifying."""
    evidence = getattr(event, "evidence", {}) or {}
    metadata = getattr(event, "metadata", {}) or {}
    keep = {}
    for key in ("action_peak_time", "extension", "peak_speed", "reach_ratio", "min_distance_ratio",
                "attacker_identity_confidence", "opponent_identity_confidence", "video_punch_probability"):
        value = evidence.get(key, metadata.get(key))
        if isinstance(value, (int, float)):
            keep[key] = round(float(value), 3)
    return {"family": getattr(event, "family", None), "limb": getattr(event, "limb", None),
            "dt": round(float(event.peak_time) - t, 2), "confidence": round(float(getattr(event, "confidence", 0.0)), 3),
            "contact": round(float(getattr(event, "contact_confidence", 0.0) or 0.0), 3), **keep}


def run(person_only: bool = True) -> dict:
    fights = [f for f in dense.dense_fights() if dense.labelled_by_person(f) or not person_only]
    return {fight.name: diagnose(fight) for fight in fights}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="also list every miss and false count")
    parser.add_argument("--all", action="store_true", help="include fights only a machine labelled")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = run(person_only=not args.all)
    if args.json:
        print(json.dumps(results, indent=1))
        return 0
    if not results:
        print("No fight labelled by a person with its pose track on this machine "
              "(dataset/public/own_fights/, tools/build_own_fight_keys.py).")
        return 1
    misses = Counter(m["why"] for r in results.values() for m in r["misses"])
    falses = Counter(f["why"] for r in results.values() for f in r["false"])
    labelled = sum(r["labelled"] for r in results.values())
    counted = sum(r["counted"] for r in results.values())
    real = sum(r["real"] for r in results.values())
    print(f"{len(results)} fight(s): {labelled} labelled, {counted} counted, {real} right")
    print(f"\nMISSED ({labelled - real}) - where each labelled strike was lost")
    for why, n in misses.most_common():
        print(f"  {n:4}  {why}")
    print(f"\nFALSE ({counted - real}) - what each wrong count was")
    for why, n in falses.most_common():
        print(f"  {n:4}  {why}")
    print("\nPer fight: labelled / counted / right; events at each stage (raw, aimed, paired, gated, counted)")
    for name, r in results.items():
        sizes = r["stage_sizes"]
        print(f"  {name[:28]:28} {r['ruleset'] or '-':10} {r['labelled']:4} {r['counted']:4} {r['real']:4}   "
              + " ".join(str(sizes[k]) for k in ("raw", "aimed", "paired", "gated", "counted")))
    if args.list:
        for name, r in results.items():
            print(f"\n{name}")
            for item in sorted(r["misses"], key=lambda m: m["t"]):
                near = item["nearest_raw"] or {}
                print(f"  miss  {item['t']:8.2f} {item['fighter']} {item['family'] or '-':6} {item['why']:42} "
                      f"nearest {near.get('family', '-')} dt={near.get('dt', '-')} conf={near.get('confidence', '-')}")
            for item in sorted(r["false"], key=lambda f: f["t"]):
                extra = " ".join(f"{k}={v}" for k, v in item.items() if k not in {"t", "fighter", "why", "family", "limb", "dt"})
                print(f"  false {item['t']:8.2f} {item['fighter']} {item['family'] or '-':6} {item['why']:42} {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
