"""Score the strike detector against a hand-labelled fight, with no video.

The labels answer "what really happened at this moment" for moments the
detector once proposed (and for quiet moments it did not). The pose track is
what the analysis saw of both fighters, frame by frame. Replaying the action
engine over that track is fast and needs neither the video nor a GPU, so any
change to core/action.py or core/contact.py can be measured against a
person's answers in a couple of seconds:

    python tools/benchmark_labelled_fight.py
    python tools/benchmark_labelled_fight.py --json          # machine-readable
    python tools/benchmark_labelled_fight.py --write-baseline

The fight lives in dataset/regression/kicklight_stavrou_ceschia/:

  track.jsonl.gz  one record per analysed frame, as the analyser writes
                  tracking.jsonl (keypoints after the joint gate)
  labels.json     72 answers from a competitor, via the labelling page
  baseline.json   the numbers the detector produced when they were stored

What it measures, and what it cannot
------------------------------------
A proposal is matched to a labelled moment of the same fighter within
MATCH_SECONDS. Matched proposals are real strikes, the right type, or nothing
at all. Proposals that match no label are reported as unlabelled rather than
guessed at: a changed detector can propose moments nobody looked at, and those
are unknown, not wrong.

A labelled strike with no proposal near it is a miss. That is only a floor on
misses - the person labelled 72 moments, not the whole fight.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.action import ActionEngine
from core.contact import classify_contact, resolve_simultaneous_attribution, thrown_at_opponent
from core.scoring import collapse_simultaneous_labels
from core.types import PersonObservation

FIGHT = PROJECT_ROOT / "dataset" / "regression" / "kicklight_stavrou_ceschia"

# Contact classification moves a proposal onto its impact frame, a few frames
# from where the labelling page cut the clip, so an exact time never matches.
# 0.35 s is under the gap between two real strikes by one fighter in this bout.
MATCH_SECONDS = 0.35

FAMILIES = ("punch", "kick", "knee")


def _family(answer: str | None) -> str | None:
    """"left_round_kick" -> "kick"; "none" -> None; unsure answers -> "?"."""
    if not answer or answer == "none":
        return None
    if answer.startswith("__"):
        return "?"
    if answer in FAMILIES:
        return answer
    return "kick" if "kick" in answer else "knee" if "knee" in answer else "punch"


def load_moments(path: Path = FIGHT / "labels.json") -> list[dict]:
    """One truth per fighter-moment.

    The same moment was sometimes offered more than once - the detector filed
    it under several limbs - and the answers did not always agree (87.67 s for
    fighter A was answered cross, nothing, wrong person, nothing). The majority
    answer stands; a tie is set aside as unsure rather than decided here.
    """
    labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
    grouped: dict[tuple[str, float], list[str | None]] = {}
    for item in labels:
        key = (item["fighter"], round(float(item["peak_time"]), 2))
        grouped.setdefault(key, []).append(_family(item["answer"]))
    moments = []
    for (fighter, time), answers in sorted(grouped.items()):
        votes = Counter(answers).most_common()
        truth = votes[0][0] if len(votes) == 1 or votes[0][1] > votes[1][1] else "?"
        moments.append({"fighter": fighter, "time": time, "truth": truth, "answers": len(answers)})
    return moments


def _observation(item: dict | None) -> PersonObservation | None:
    if not item:
        return None
    return PersonObservation(
        track_id=item.get("track_id"), box=np.asarray(item["box"], dtype=np.float32),
        confidence=float(item.get("confidence") or 0.0),
        keypoints=None if item.get("keypoints") is None else np.asarray(item["keypoints"], dtype=np.float32),
        keypoint_conf=None if item.get("keypoint_conf") is None else np.asarray(item["keypoint_conf"], dtype=np.float32),
    )


def replay(path: Path = FIGHT / "track.jsonl.gz") -> list:
    """The analyser's action path, fed from the stored track instead of video."""
    engine = ActionEngine()
    events = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            a = _observation(record["fighter_A"]["observation"])
            b = _observation(record["fighter_B"]["observation"])
            conf_a = float(record["fighter_A"]["identity_confidence"])
            conf_b = float(record["fighter_B"]["identity_confidence"])
            args = (record["source_frame"], record["time_seconds"], record["round_number"])
            new = engine.update("A", *args, a, b, conf_a, conf_b)
            new += engine.update("B", *args, b, a, conf_b, conf_a)
            for event in new:
                event = classify_contact(event)
                if thrown_at_opponent(event):
                    events.append(event)
    events, _dropped = resolve_simultaneous_attribution(events)
    return events


def score(events: list, moments: list[dict]) -> dict:
    """Match proposals to labelled moments and count what they turned out to be."""
    counted, _removed = collapse_simultaneous_labels(events)
    result = Counter()
    hit = set()
    for event in counted:
        near = [m for m in moments
                if m["fighter"] == event.fighter and abs(m["time"] - event.peak_time) <= MATCH_SECONDS]
        if not near:
            result["unlabelled"] += 1
            continue
        moment = min(near, key=lambda m: abs(m["time"] - event.peak_time))
        hit.add((moment["fighter"], moment["time"]))
        if moment["truth"] == "?":
            result["unsure"] += 1
        elif moment["truth"] is None:
            result["nothing_happened"] += 1
        else:
            result["real_strike"] += 1
            result["right_type"] += int(moment["truth"] == event.family)
    real_moments = [m for m in moments if m["truth"] in FAMILIES]
    result["labelled_strikes"] = len(real_moments)
    result["labelled_strikes_missed"] = sum(
        1 for m in real_moments if (m["fighter"], m["time"]) not in hit)
    result["proposals_counted"] = len(counted)
    result["proposals_raw"] = len(events)
    return {key: int(result[key]) for key in (
        "proposals_raw", "proposals_counted", "real_strike", "right_type", "nothing_happened",
        "unsure", "unlabelled", "labelled_strikes", "labelled_strikes_missed")}


def run() -> dict:
    return score(replay(), load_moments())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true", help="print the numbers as JSON")
    parser.add_argument("--write-baseline", action="store_true",
                        help="store these numbers as the comparison for the next run")
    args = parser.parse_args()
    now = run()
    baseline_path = FIGHT / "baseline.json"
    if args.write_baseline:
        baseline_path.write_text(json.dumps(now, indent=1) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(now, indent=1))
        return 0
    before = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    for key, value in now.items():
        change = "" if key not in before or before[key] == value else "  (was %d)" % before[key]
        print("%-26s %4d%s" % (key, value, change))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
