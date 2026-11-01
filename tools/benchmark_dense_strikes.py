"""Score the strike count against fights where every strike is labelled.

tools/benchmark_labelled_fight.py scores the detector on moments it once
proposed, so it says how many counted strikes are real but only gives a floor
on the ones it never found. Here every strike thrown in a stretch of video is
labelled, so both are measured: of the strikes the report counts, how many
were thrown (precision), and of the strikes thrown, how many it counts
(recall).

    python tools/benchmark_dense_strikes.py
    python tools/benchmark_dense_strikes.py --ruleset BOXING --json

Each fight is a folder in dataset/regression/ or dataset/public/phone_strikes/ with:

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

    python tools/benchmark_dense_strikes.py --ruleset BOXING --video

--video also runs the video punch check (core/strike_verifier.py) as the
analysis does, for fights with a source.json naming their video. Its features
are cached beside each fight (features_check.npz), keyed by fighter and time.
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
    # dataset/public/ is not in Git: pose tracks of identifiable athletes stay
    # out of a public repository (.gitignore).
    # dataset/stock_strikes/ holds free-licence stock clips labelled strike by
    # strike; their labels are committed, their pose tracks are rebuilt with
    # tools/make_stock_strikes.py and stay out of Git like the others.
    roots = [PROJECT_ROOT / "dataset" / "regression", PROJECT_ROOT / "dataset" / "public" / "phone_strikes",
             PROJECT_ROOT / "dataset" / "stock_strikes"]
    return sorted(p for root in roots if root.is_dir() for p in root.iterdir()
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


def _thrown(track: Path) -> list:
    """The action path up to the pair rule: what the video check sees."""
    import gzip

    from core.action import ActionEngine
    from core.contact import classify_contact, thrown_at_opponent
    from tools.benchmark_labelled_fight import _observation

    engine, events = ActionEngine(), []
    with gzip.open(track, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            a, b = _observation(record["fighter_A"]["observation"]), _observation(record["fighter_B"]["observation"])
            conf_a = float(record["fighter_A"]["identity_confidence"])
            conf_b = float(record["fighter_B"]["identity_confidence"])
            args = (record["source_frame"], record["time_seconds"], record["round_number"])
            for event in engine.update("A", *args, a, b, conf_a, conf_b) + engine.update("B", *args, b, a, conf_b, conf_a):
                event = classify_contact(event)
                if thrown_at_opponent(event):
                    events.append(event)
    return events


HELD_OUT = False


def _held_out_head(fight_name: str):
    """A head fitted without this fight's judged punches (tools/fit_strike_check.py),
    so a fight the shipped head was fitted on is not scored by it."""
    import numpy as np

    from tools import fit_strike_check

    folders = [p for p in fit_strike_check.ROOT.iterdir() if (p / "features_r3d18.npz").exists()
               and not p.name.startswith(fight_name)]
    X = np.concatenate([np.load(p / "features_r3d18.npz")["features"] for p in folders])
    y = np.concatenate([np.load(p / "features_r3d18.npz")["labels"] for p in folders])
    head = fit_strike_check.fit(X, y)
    return lambda features: fit_strike_check.predict(head, np.atleast_2d(features))[0]


def _local_video(fight: Path) -> Path:
    """The fight's video on this machine.

    A regression fight names its file; a stock clip (dataset/stock_strikes/)
    only names where to download it, so it is fetched once into
    dataset/public/, which stays out of Git like every other fight video.
    """
    source = json.loads((fight / "source.json").read_text(encoding="utf-8"))
    if "video" in source:
        return PROJECT_ROOT / source["video"]
    local = PROJECT_ROOT / "dataset" / "public" / "stock_strikes_video" / f"{fight.name}.mp4"
    if not local.exists():
        import urllib.request

        local.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(source["download_url"], local)
    return local


def _video_checked(fight: Path, truth: dict) -> list:
    """Replay with the video punch check scoring punches before the pair rule."""
    import gzip

    import numpy as np

    from core import strike_verifier
    from core.contact import resolve_simultaneous_attribution

    start, end = truth["covered"]
    events = [e for e in _thrown(fight / "track.jsonl.gz") if start - 1.0 <= float(e.peak_time) <= end + 1.0]
    punches = [e for e in events if e.family == "punch"]
    keys = [f"{e.fighter}:{float(e.evidence.get('action_peak_time', e.peak_time)):.3f}" for e in punches]
    cache_path = fight / "features_check.npz"
    cache = dict(np.load(cache_path, allow_pickle=False)) if cache_path.exists() else {}
    missing = [i for i, key in enumerate(keys) if key not in cache]
    if missing:
        rows = []
        with gzip.open(fight / "track.jsonl.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                boxes = [((record.get(f"fighter_{f}") or {}).get("observation") or {}).get("box") for f in "AB"]
                rows.append((float(record["time_seconds"]), boxes[0], boxes[1]))
        lookup = strike_verifier.boxes_from_track(rows)
        values = strike_verifier.clip_features(
            str(_local_video(fight)),
            [float(punches[i].evidence.get("action_peak_time", punches[i].peak_time)) for i in missing],
            [lookup(punches[i].fighter) for i in missing])
        for i, value in zip(missing, values):
            if value is not None:
                cache[keys[i]] = value
        np.savez(cache_path, **cache)
    probability = _held_out_head(fight.name) if HELD_OUT else strike_verifier.Head().probability
    for event, key in zip(punches, keys):
        if key in cache:
            event.evidence["video_punch_probability"] = float(probability(cache[key]))
    kept, _ = resolve_simultaneous_attribution(events)
    return kept


def run(ruleset: str | None, video: bool = False) -> dict:
    results = {}
    total: dict = {}
    for fight in dense_fights():
        truth = json.loads((fight / "strikes.json").read_text(encoding="utf-8"))
        if video and (fight / "source.json").exists():
            events = _video_checked(fight, truth)
        else:
            events = replay(fight / "track.jsonl.gz")
        row = score(counted_events(events, ruleset), truth)
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
    parser.add_argument("--video", action="store_true", help="also run the video punch check, as the analysis does")
    parser.add_argument("--held-out", action="store_true",
                        help="with --video: score each fight with a head fitted without its judged punches")
    args = parser.parse_args()
    global HELD_OUT
    HELD_OUT = args.held_out
    results = run(args.ruleset, video=args.video)
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
