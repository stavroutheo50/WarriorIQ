from __future__ import annotations

from typing import Any


# These are release gates, not claims about the current build.  A checkpoint
# must carry its held-out validation metadata before its action labels can be
# shown to customers as evidence.
MIN_ACTION_VALIDATION_ACCURACY = 0.90
MIN_HELD_OUT_FIGHTS = 3
MIN_ACTION_TEST_ACCURACY = 0.90
MIN_PER_CLASS_TEST_ACCURACY = 0.80
MIN_TESTED_ACTION_CLASSES = 18


def automated_evidence_trust(classifier: dict[str, Any] | None) -> dict[str, Any]:
    classifier = classifier or {}
    validation = classifier.get("temporal_validation") or {}
    checkpoint_loaded = bool(classifier.get("custom_temporal_checkpoint_loaded"))
    accuracy = validation.get("val_accuracy")
    held_out_fights = validation.get("held_out_fights")
    test_accuracy = validation.get("test_accuracy")
    held_out_test_fights = validation.get("held_out_test_fights")
    per_class_test = validation.get("per_class_test_accuracy") or {}
    dataset_version = str(validation.get("dataset_version") or "").strip()

    try:
        accuracy_value = float(accuracy)
    except (TypeError, ValueError):
        accuracy_value = None
    held_out_value = len(held_out_fights) if isinstance(held_out_fights, (list, tuple, set)) else None
    if held_out_value is None:
        try:
            held_out_value = int(held_out_fights)
        except (TypeError, ValueError):
            held_out_value = None
    try:
        test_accuracy_value = float(test_accuracy)
    except (TypeError, ValueError):
        test_accuracy_value = None
    test_fight_count = len(held_out_test_fights) if isinstance(held_out_test_fights, (list, tuple, set)) else None
    if test_fight_count is None:
        try:
            test_fight_count = int(held_out_test_fights)
        except (TypeError, ValueError):
            test_fight_count = None
    tested_class_values = []
    for value in per_class_test.values() if isinstance(per_class_test, dict) else []:
        try:
            tested_class_values.append(float(value))
        except (TypeError, ValueError):
            pass
    worst_test_class = min(tested_class_values) if tested_class_values else None

    trusted = bool(
        checkpoint_loaded
        and accuracy_value is not None
        and accuracy_value >= MIN_ACTION_VALIDATION_ACCURACY
        and held_out_value is not None
        and held_out_value >= MIN_HELD_OUT_FIGHTS
        and dataset_version
        and test_accuracy_value is not None
        and test_accuracy_value >= MIN_ACTION_TEST_ACCURACY
        and test_fight_count is not None
        and test_fight_count >= MIN_HELD_OUT_FIGHTS
        and worst_test_class is not None
        and worst_test_class >= MIN_PER_CLASS_TEST_ACCURACY
        and len(tested_class_values) >= MIN_TESTED_ACTION_CLASSES
    )
    if trusted:
        reason = (
            f"Validated action checkpoint {dataset_version} passed the release gate "
            f"({accuracy_value * 100:.1f}% validation and {test_accuracy_value * 100:.1f}% untouched-test accuracy)."
        )
        status = "validated_model"
    elif not checkpoint_loaded:
        reason = (
            "The current action model has not yet passed release validation. WarriorIQ shows its score only as a "
            "preliminary estimate and provides a video-check workflow; unreviewed action labels are never presented as facts."
        )
        status = "candidate_only"
    else:
        reason = (
            "The action checkpoint has not supplied enough held-out-fight validation to "
            "pass WarriorIQ's release gate. Its detections remain review candidates."
        )
        status = "checkpoint_not_release_ready"

    return {
        "automated_evidence_trusted": trusted,
        "action_evidence_status": status,
        "action_evidence_reason": reason,
        "release_gate": {
            "minimum_validation_accuracy": MIN_ACTION_VALIDATION_ACCURACY,
            "minimum_held_out_fights": MIN_HELD_OUT_FIGHTS,
            "minimum_untouched_test_accuracy": MIN_ACTION_TEST_ACCURACY,
            "minimum_per_class_test_accuracy": MIN_PER_CLASS_TEST_ACCURACY,
            "minimum_tested_action_classes": MIN_TESTED_ACTION_CLASSES,
            "dataset_version_required": True,
        },
    }


def report_evidence_trust(report: dict[str, Any]) -> dict[str, Any]:
    """Return a fresh trust decision so older saved reports are also safe."""
    return automated_evidence_trust(report.get("classifier"))
