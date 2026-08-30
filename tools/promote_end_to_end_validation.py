from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.annotations import accuracy_summary
from core.regression_manifest import REGRESSION_MANIFEST_SCHEMA, validate_regression_manifest
from core.release_validation import assess_end_to_end_validation, end_to_end_metadata


def _load_annotations(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict) and payload.get("schema") == REGRESSION_MANIFEST_SCHEMA:
        return validate_regression_manifest(payload)
    if isinstance(payload, dict):
        payload = payload.get("annotations")
    if not isinstance(payload, list):
        raise RuntimeError("Annotation input must be a JSON list, an annotations object, or JSONL")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure identity, target, outcome, legality and contact timing before model promotion."
    )
    parser.add_argument("--annotations", required=True, help="Private ground-truth JSON or JSONL file")
    parser.add_argument("--checkpoint", default="models/warrioriq_temporal_best.pt")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    annotations = _load_annotations(Path(args.annotations))
    metadata = end_to_end_metadata(accuracy_summary(annotations))
    gate = assess_end_to_end_validation(metadata)
    result = {"end_to_end_validation": metadata, "gate": gate}
    print(json.dumps(result, indent=2))

    if not args.promote:
        return
    if not gate["passed"]:
        raise RuntimeError("Checkpoint was not promoted because the end-to-end evidence gate failed")

    import torch

    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("Checkpoint must contain WarriorIQ training metadata and a state_dict")
    payload["end_to_end_validation"] = metadata
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".end-to-end-promoting")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    print("Promoted checkpoint with end-to-end validation evidence:", checkpoint_path)


if __name__ == "__main__":
    main()
