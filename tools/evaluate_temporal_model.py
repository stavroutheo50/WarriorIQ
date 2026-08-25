from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.evidence_trust import (
    MIN_ACTION_TEST_ACCURACY,
    MIN_HELD_OUT_FIGHTS,
    MIN_PER_CLASS_TEST_ACCURACY,
    MIN_TESTED_ACTION_CLASSES,
)
from core.temporal_model import ACTION_CLASSES, build_temporal_network
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

    dataset = SequenceDataset(Path(args.test_data))
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
    accuracy, per_class = evaluate(model, loader, device)
    worst_class = min(per_class.values()) if per_class else 0.0
    passed = bool(
        len(test_fights) >= MIN_HELD_OUT_FIGHTS
        and len(per_class) >= MIN_TESTED_ACTION_CLASSES
        and accuracy >= MIN_ACTION_TEST_ACCURACY
        and worst_class >= MIN_PER_CLASS_TEST_ACCURACY
    )
    result = {
        "test_accuracy": accuracy,
        "held_out_test_fights": test_fights,
        "per_class_test_accuracy": per_class,
        "worst_class_test_accuracy": worst_class,
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
        })
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".promoting")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
        print("Promoted checkpoint with untouched-test evidence:", checkpoint_path)


if __name__ == "__main__":
    main()
