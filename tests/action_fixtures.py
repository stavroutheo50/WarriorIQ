"""Synthetic validation metadata for positive controls, never real model evidence."""

from core.temporal_model import ACTION_CLASSES


def validated_classifier():
    return {"custom_temporal_checkpoint_loaded": True, "temporal_validation": {
        "val_accuracy": .96, "held_out_fights": ["v1", "v2", "v3"], "dataset_version": "test-only",
        "test_accuracy": .94, "held_out_test_fights": ["t1", "t2", "t3"],
        "per_class_test_accuracy": {name: .85 for name in ACTION_CLASSES},
        "per_class_test_f1": {name: .82 for name in ACTION_CLASSES},
        "end_to_end_validation": {"fights": 5, "action_labels": 120, "timing_samples": 80,
            "fighter_identity_accuracy": .97, "target_accuracy": .93, "outcome_accuracy": .88,
            "legality_accuracy": .97, "timing_mae_seconds": .16},
    }}


def supported_prediction(event):
    label = event["technique"]
    if label not in ACTION_CLASSES and "kick" in label:
        label = "right_round_kick" if label.startswith("right") else "left_round_kick"
    event["model_source"] = "warrioriq_temporal_model"
    event.setdefault("evidence", {})["temporal_decision"] = {
        "status": "strike", "label": label, "confidence": .95,
    }
    return event
