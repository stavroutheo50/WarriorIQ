from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evidence_trust import (
    MIN_ACTION_TEST_ACCURACY,
    MIN_HELD_OUT_FIGHTS,
    MIN_PER_CLASS_TEST_ACCURACY,
    MIN_PER_CLASS_TEST_F1,
    MIN_TESTED_ACTION_CLASSES,
)
from core.temporal_model import ACTION_CLASSES, build_temporal_network
from core.model_validation import audit_sequence_directory
from train_temporal_model import SequenceDataset, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a WarriorIQ checkpoint on untouched complete-fight labels.")
    parser.add_argument("--checkpoint", default="models/warrioriq_temporal_best.pt")
    parser.add_argument("--test-data", required=True, help="Separate directory that was never used for training or validation")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--promote", action="store_true", help="Write passing untouched-test metadata into the checkpoint")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("Checkpoint must contain WarriorIQ training metadata and a state_dict")
    if list(payload.get("classes", [])) != ACTION_CLASSES:
        raise RuntimeError("Checkpoint action classes do not match this WarriorIQ build")

    test_root = Path(args.test_data)
    audit = audit_sequence_directory(test_root)
    if audit["invalid_sequences"]:
        first = audit["issues"][0]
        raise RuntimeError(
            f"Untouched-test audit rejected {audit['invalid_sequences']} invalid sequence(s). "
            f"First problem: {first['file']}: {first['reason']}"
        )
    if not audit["all_classes_covered"]:
        raise RuntimeError(
            "Untouched-test data must cover every supported action class. Missing: "
            + ", ".join(audit["missing_classes"])
        )
    dataset = SequenceDataset(test_root)
    test_fights = sorted(set(dataset.fight_ids))
    development_fights = set(payload.get("training_fights") or []) | set(payload.get("held_out_fights") or [])
    overlap = sorted(development_fights & set(test_fights))
    if overlap:
        raise RuntimeError("Untouched-test leakage detected in fights: " + ", ".join(overlap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_x, _ = dataset[0]
    model = build_temporal_network(
        payload.get("architecture", "pose_transformer_v2"),
        int(payload.get("input_dim", sample_x.shape[-1])),
        len(ACTION_CLASSES),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, device)
    accuracy = float(metrics["accuracy"] or 0.0)
    per_class = {name: item["recall"] for name, item in metrics["per_class"].items()}
    per_class_precision = {name: item["precision"] for name, item in metrics["per_class"].items()}
    per_class_f1 = {name: item["f1"] for name, item in metrics["per_class"].items()}
    valid_recall = [value for value in per_class.values() if value is not None]
    valid_f1 = [value for value in per_class_f1.values() if value is not None]
    worst_class = min(valid_recall) if valid_recall else 0.0
    worst_class_f1 = min(valid_f1) if valid_f1 else 0.0
    all_classes_measured = all(per_class.get(name) is not None and per_class_f1.get(name) is not None for name in ACTION_CLASSES)
    passed = bool(
        len(test_fights) >= MIN_HELD_OUT_FIGHTS
        and all_classes_measured
        and len(valid_recall) >= MIN_TESTED_ACTION_CLASSES
        and accuracy >= MIN_ACTION_TEST_ACCURACY
        and worst_class >= MIN_PER_CLASS_TEST_ACCURACY
        and worst_class_f1 >= MIN_PER_CLASS_TEST_F1
    )
    result = {
        "test_accuracy": accuracy,
        "held_out_test_fights": test_fights,
        "per_class_test_accuracy": per_class,
        "per_class_test_precision": per_class_precision,
        "per_class_test_f1": per_class_f1,
        "macro_test_f1": metrics["macro_f1"],
        "macro_action_test_f1": metrics["macro_action_f1"],
        "worst_class_test_accuracy": worst_class,
        "worst_class_test_f1": worst_class_f1,
        "release_gate_passed": passed,
    }
    print(json.dumps(result, indent=2))
    checkpoint_path.with_suffix(".untouched-test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.promote:
        if not passed:
            raise RuntimeError("Checkpoint was not promoted because the untouched-test release gate failed")
        payload.update({
            "test_accuracy": accuracy,
            "held_out_test_fights": test_fights,
            "per_class_test_accuracy": per_class,
            "per_class_test_precision": per_class_precision,
            "per_class_test_f1": per_class_f1,
            "macro_test_f1": metrics["macro_f1"],
            "macro_action_test_f1": metrics["macro_action_f1"],
        })
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".promoting")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
        print("Promoted checkpoint with untouched-test evidence:", checkpoint_path)


if __name__ == "__main__":
    main()
