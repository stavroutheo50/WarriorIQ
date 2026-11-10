from __future__ import annotations

from typing import Any


# Minimum engineering gates. Passing them permits a checkpoint to be tested in
# the product; it is not, by itself, a claim of official judging accuracy.
MIN_END_TO_END_FIGHTS = 5
MIN_END_TO_END_ACTION_LABELS = 100
MIN_END_TO_END_TIMING_SAMPLES = 50
MIN_FIGHTER_IDENTITY_ACCURACY = 0.95
MIN_TARGET_ACCURACY = 0.90
MIN_OUTCOME_ACCURACY = 0.85
MIN_LEGALITY_ACCURACY = 0.95
MAX_TIMING_MAE_SECONDS = 0.25


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in {float("inf"), float("-inf")} else None


def assess_end_to_end_validation(validation: dict[str, Any] | None) -> dict:
    validation = validation or {}
    requirements = {
        "fights": (MIN_END_TO_END_FIGHTS, "minimum"),
        "action_labels": (MIN_END_TO_END_ACTION_LABELS, "minimum"),
        "timing_samples": (MIN_END_TO_END_TIMING_SAMPLES, "minimum"),
        "fighter_identity_accuracy": (MIN_FIGHTER_IDENTITY_ACCURACY, "minimum"),
        "target_accuracy": (MIN_TARGET_ACCURACY, "minimum"),
        "outcome_accuracy": (MIN_OUTCOME_ACCURACY, "minimum"),
        "legality_accuracy": (MIN_LEGALITY_ACCURACY, "minimum"),
        "timing_mae_seconds": (MAX_TIMING_MAE_SECONDS, "maximum"),
    }
    failures = []
    measured = {}
    for name, (threshold, direction) in requirements.items():
        value = _number(validation.get(name))
        measured[name] = value
        if value is None:
            failures.append(f"{name} is missing")
        elif direction == "minimum" and value < threshold:
            failures.append(f"{name} {value:.3f} is below {threshold:.3f}")
        elif direction == "maximum" and value > threshold:
            failures.append(f"{name} {value:.3f} exceeds {threshold:.3f}")
    return {
        "passed": not failures,
        "failures": failures,
        "measured": measured,
        "requirements": {
            name: {"value": threshold, "direction": direction}
            for name, (threshold, direction) in requirements.items()
        },
    }


def end_to_end_metadata(summary: dict) -> dict:
    metrics = summary.get("metrics", {})
    timing = summary.get("timing", {})

    def accuracy(name: str) -> float | None:
        return metrics.get(name, {}).get("accuracy")

    return {
        "fights": summary.get("fights", 0),
        "action_labels": summary.get("positive_labels", 0),
        "timing_samples": timing.get("samples", 0),
        "fighter_identity_accuracy": accuracy("fighter_identity"),
        "target_accuracy": accuracy("target"),
        "outcome_accuracy": accuracy("outcome"),
        "legality_accuracy": accuracy("legality"),
        "timing_mae_seconds": timing.get("mean_absolute_error_seconds"),
    }
