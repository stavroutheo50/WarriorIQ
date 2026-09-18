"""How often does the travel-ratio rule change a strike's family, and at what cost?

`core/action.py` proposes a family from the keypoint pattern, then lets a single
number overrule it: if the hands travelled more than FAMILY_MARGIN times as far
as the feet the action becomes a punch, and if the feet won by the same margin
it becomes a kick. Both directions matter because they are not symmetric in
what the reader sees:

  * kick/knee -> punch  removes an action from the ONLY strike number the report
    publishes. `observed_summary` reports kick_attempts + knee_attempts and
    withholds punches entirely, so a flip in this direction deletes a leg strike
    from the count with no trace.
  * punch -> kick       does the reverse: it promotes an action out of the
    punch bucket, which hand-checking measured as noise-dominated (fight 1
    reported 11 punches against 0 actually thrown), into the published number.

This script answers the question the reported bug needs answering - how often,
and which way - WITHOUT needing punch-vs-kick ground truth, which this project
does not have. It counts flips and shows what the published kick total would be
if each direction of the rule were removed.

It deliberately does NOT recommend a threshold. The two distributions were
measured overlapping (kick median ratio 0.66, punch median 1.43, margin 1.60),
so no cut separates them, and any cut fitted to action.py's own labels would be
circular.

    python tools/measure_family_flip.py               # every verified fight
    python tools/measure_family_flip.py 5736 1mp4     # named fights only

Pin WARRIORIQ_FORCE_STRIDE first if you want to compare two runs; the planner
reads a wall clock otherwise and the frame set changes between runs.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = ROOT / "tools" / "verified_seeds.json"


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarise(events: list[dict]) -> dict:
    """Flip counts and the published total under each variant of the rule."""
    flips = Counter()
    ratios: dict[str, list[float]] = {"kick": [], "knee": [], "punch": []}
    published_now = 0
    published_without_demotion = 0   # kick/knee -> punch removed
    published_without_promotion = 0  # punch -> kick removed
    published_seed_only = 0          # rule removed entirely

    for event in events:
        evidence = event.get("evidence") or {}
        seed = evidence.get("family_seed")
        final = event.get("family")
        ratio = evidence.get("family_travel_ratio")
        if seed is None:
            continue
        if isinstance(ratio, (int, float)) and seed in ratios:
            ratios[seed].append(float(ratio))
        if seed != final:
            flips[f"{seed}->{final}"] += 1

        leg_final = final in {"kick", "knee"}
        leg_seed = seed in {"kick", "knee"}
        published_now += leg_final
        published_seed_only += leg_seed
        # Without the demotion, a leg action the rule turned into a punch stays.
        published_without_demotion += leg_final or (leg_seed and final == "punch")
        # Without the promotion, a punch the rule turned into a kick does not count.
        published_without_promotion += leg_final and not (seed == "punch" and leg_final)

    return {
        "events": len(events),
        "flips": dict(flips),
        "flip_rate": round(sum(flips.values()) / len(events), 3) if events else None,
        "published_kick_total": {
            "as_shipped": published_now,
            "without_kick_to_punch": published_without_demotion,
            "without_punch_to_kick": published_without_promotion,
            "rule_removed_entirely": published_seed_only,
        },
        "travel_ratio_median": {
            family: (None if not values else round(_percentile(values, 0.5), 3))
            for family, values in ratios.items()
        },
        "travel_ratio_n": {family: len(values) for family, values in ratios.items()},
    }


def main() -> int:
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))
    wanted = sys.argv[1:] or list(seeds["fights"])

    import cv2

    from core import analyzer
    from core.types import AnalysisRequest

    report_lines = []
    for name in wanted:
        fight = seeds["fights"].get(name)
        if not fight:
            print(f"{name}: not in verified_seeds.json", file=sys.stderr)
            continue
        video = ROOT / fight["video"]
        if not video.exists():
            print(f"{name}: {video} is missing", file=sys.stderr)
            continue

        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        capture.release()
        duration = float(frames) / float(fps or 25.0)
        start_seconds = float(fight.get("start_seconds") or 0.0)

        print(f"--- {name}: analysing {video.name} "
              f"({duration:.0f}s from {start_seconds:.0f}s) ---", flush=True)
        report = analyzer.analyze(AnalysisRequest(
            video_path=str(video),
            fighter_a_box=[float(v) for v in fight["fighter_a"]],
            fighter_b_box=[float(v) for v in fight["fighter_b"]],
            ruleset=fight.get("ruleset", "K1"), fight_type="competition",
            round_count=1, start_seconds=start_seconds,
            round_duration_seconds=max(1.0, duration - start_seconds),
            job_id=f"flipmeasure_{name}", profile_id=1, persist_result=False))
        events = (report.get("events") or []) if isinstance(report, dict) else []
        result = summarise(events)
        result["fight"] = name
        report_lines.append(result)
        print(json.dumps(result, indent=1), flush=True)

    print("\n=== all fights ===")
    print(json.dumps(report_lines, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
