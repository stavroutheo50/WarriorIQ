"""Build the public sample report's data from a real analysis (2026-11-01).

The sample report used invented fighters and numbers. It now shows a real
WarriorIQ analysis of a free-licence stock clip - the one on the homepage,
Mixkit 40979 (Mixkit Stock Video Free License) - so a visitor sees what the
product actually produces, not a mock-up.

    python tools/make_sample_report.py path/to/report.json

The analysis's report.json holds per-frame evidence (boxes, keypoints) that
stays out of this public repository with every other pose measurement. This
keeps only what the sample page draws: the per-fighter statistics and
movement numbers, each detected strike's time, fighter and family, the
defences, the round and the identity check. Written to app/sample_fight.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "sample_fight.json"

SOURCE = {
    "title": "Two women fighting mixed martial arts",
    "page_url": "https://mixkit.co/free-stock-video/two-women-fighting-mixed-martial-arts-40979/",
    "licence": "Mixkit Stock Video Free License",
    "licence_url": "https://mixkit.co/license/#videoFree",
}
STAT_KEYS = ("punch_attempts", "kick_attempts", "knee_attempts", "activity_rate", "observation_coverage")
METRIC_KEYS = ("footwork_body_lengths_per_second", "pressure_index", "ring_center_control", "balance_index",
               "pose_coverage", "guard_index")
NUMBER_KEYS = ("distance_body_lengths", "forward_seconds", "backward_seconds", "hands_up_share",
               "off_balance_count", "seen_seconds")


def trim(report: dict) -> dict:
    fighters = (report.get("statistics") or {}).get("fighters") or {}
    metrics = report.get("metrics") or {}
    out_metrics = {}
    for side in ("A", "B"):
        m = metrics.get(side) or {}
        numbers = m.get("numbers") or {}
        series = ((numbers.get("series") or {}).get("points") or [])
        out_metrics[side] = {**{k: m.get(k) for k in METRIC_KEYS},
                             "numbers": {**{k: numbers.get(k) for k in NUMBER_KEYS},
                                         "series": {"points": [{"t": p.get("t"), "speed": p.get("speed")} for p in series]}}}
    return {
        "source": SOURCE,
        "scorecard": {k: (report.get("scorecard") or {}).get(k) for k in ("sport", "sport_label", "ruleset", "available")},
        "statistics": {"fighters": {side: {k: (fighters.get(side) or {}).get(k) for k in STAT_KEYS} for side in ("A", "B")}},
        "metrics": out_metrics,
        "events": [{"fighter": e.get("fighter"), "peak_time": e.get("peak_time"), "family": e.get("family"),
                    "attempted": e.get("attempted")} for e in report.get("events") or [] if e.get("attempted")],
        # The feed the totals are counted from (core/stat_sheet._counted_strikes).
        "event_feed": [{k: item.get(k) for k in ("kind", "fighter", "family", "limb", "technique", "time_seconds", "confidence")}
                       for item in report.get("event_feed") or []],
        "defenses": [{"fighter": d.get("fighter")} for d in report.get("defenses") or []],
        "rounds": [{"number": r.get("number"), "start_seconds": r.get("start_seconds"), "end_seconds": r.get("end_seconds")}
                   for r in report.get("rounds") or []],
        "integrity": {"identity_evidence_trusted": (report.get("integrity") or {}).get("identity_evidence_trusted", True)},
        "tracking": {k: (report.get("tracking") or {}).get(k) for k in ("fighter_A_coverage", "fighter_B_coverage")},
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    OUT.write_text(json.dumps(trim(json.loads(Path(argv[1]).read_text(encoding="utf-8"))), indent=1), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
