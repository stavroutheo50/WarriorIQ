from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.annotations import accuracy_summary
from core.config import OUTPUTS, UPLOADS
from core.db import get_fight, list_annotations
from core.regression_manifest import build_regression_manifest, file_sha256
from core.release_validation import assess_end_to_end_validation, end_to_end_metadata


def _video_path(job_id: str) -> Path | None:
    fight = get_fight(job_id)
    if fight and fight.get("video_path"):
        candidate = Path(fight["video_path"])
        if candidate.is_file():
            return candidate
    return next((path for path in sorted(UPLOADS.glob(f"{job_id}.*")) if path.is_file()), None)


def _report_path(job_id: str) -> Path | None:
    fight = get_fight(job_id)
    if fight and fight.get("report_path"):
        candidate = Path(fight["report_path"])
        if candidate.is_file():
            return candidate
    candidate = OUTPUTS / job_id / "report.json"
    return candidate if candidate.is_file() else None


def _public_annotation(annotation: dict) -> dict:
    return {
        "event_time": float(annotation["event_time"]),
        "ruleset": annotation["ruleset"],
        "predicted": annotation["predicted"],
        "corrected": annotation["corrected"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze private, human-labelled fight evidence for regression testing.")
    parser.add_argument(
        "--output",
        default="dataset/regression/private/manifest.json",
        help="Private output path; the default directory is ignored by Git.",
    )
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for annotation in list_annotations():
        grouped[str(annotation["job_id"])].append(_public_annotation(annotation))
    if not grouped:
        raise RuntimeError("No human-reviewed annotations exist. Review real fight events before building a regression manifest.")

    fights = []
    for job_id, annotations in grouped.items():
        video = _video_path(job_id)
        report = _report_path(job_id)
        fights.append({
            "fight_id": job_id,
            "video_sha256": file_sha256(video) if video else None,
            "report_sha256": file_sha256(report) if report else None,
            "annotations": sorted(annotations, key=lambda item: item["event_time"]),
        })

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = build_regression_manifest(fights, created_at=created_at)
    output = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    annotations = [item for fight in fights for item in fight["annotations"]]
    gate = assess_end_to_end_validation(end_to_end_metadata(accuracy_summary([
        {**item, "job_id": fight["fight_id"]}
        for fight in fights for item in fight["annotations"]
    ])))
    print(json.dumps({
        "manifest": str(output),
        "fights": len(fights),
        "annotations": len(annotations),
        "asset_hashes_complete": all(fight["video_sha256"] and fight["report_sha256"] for fight in fights),
        "release_gate_passed": gate["passed"],
        "release_gate_failures": gate["failures"],
    }, indent=2))


if __name__ == "__main__":
    main()
