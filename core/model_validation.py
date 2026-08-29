from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from core.config import SETTINGS
from core.temporal_model import ACTION_CLASSES


def classification_metrics(
    expected: Iterable[str],
    predicted: Iterable[str],
    *,
    classes: list[str] | tuple[str, ...] = ACTION_CLASSES,
) -> dict:
    """Calculate transparent class metrics without filling missing evidence with zeroes.

    Overall accuracy alone is unsafe for an imbalanced action dataset: a model
    can look strong by predicting common punches while failing rare kicks.  The
    release workflow therefore keeps precision, recall and F1 per class.
    """
    expected_values = list(expected)
    predicted_values = list(predicted)
    if len(expected_values) != len(predicted_values):
        raise ValueError("Expected and predicted labels must have the same length")
    supported = set(classes)
    unknown = sorted((set(expected_values) | set(predicted_values)) - supported)
    if unknown:
        raise ValueError("Unknown action labels: " + ", ".join(unknown))

    confusion = {
        actual: {guess: 0 for guess in classes}
        for actual in classes
    }
    for actual, guess in zip(expected_values, predicted_values):
        confusion[actual][guess] += 1

    per_class = {}
    f1_values = []
    action_f1_values = []
    for label in classes:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[actual][label] for actual in classes if actual != label)
        false_negative = sum(confusion[label][guess] for guess in classes if guess != label)
        support = sum(confusion[label].values())
        predicted_count = true_positive + false_positive
        precision = None if predicted_count == 0 else true_positive / predicted_count
        recall = None if support == 0 else true_positive / support
        f1 = None
        if precision is not None and recall is not None:
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            f1_values.append(f1)
            if label != "none":
                action_f1_values.append(f1)
        per_class[label] = {
            "support": support,
            "predicted": predicted_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    samples = len(expected_values)
    exact = sum(actual == guess for actual, guess in zip(expected_values, predicted_values))
    return {
        "samples": samples,
        "exact": exact,
        "accuracy": None if not samples else exact / samples,
        "macro_f1": None if not f1_values else sum(f1_values) / len(f1_values),
        "macro_action_f1": None if not action_f1_values else sum(action_f1_values) / len(action_f1_values),
        "false_alarms": sum(actual == "none" and guess != "none" for actual, guess in zip(expected_values, predicted_values)),
        "missed_actions": sum(actual != "none" and guess == "none" for actual, guess in zip(expected_values, predicted_values)),
        "per_class": per_class,
        "confusion": confusion,
    }


def _empty_audit(root: Path) -> dict:
    return {
        "path": str(root),
        "exists": root.exists(),
        "files": 0,
        "valid_sequences": 0,
        "invalid_sequences": 0,
        "fights": 0,
        "fight_ids": [],
        "positive_sequences": 0,
        "negative_sequences": 0,
        "covered_classes": 0,
        "total_classes": len(ACTION_CLASSES),
        "class_support": {name: 0 for name in ACTION_CLASSES},
        "missing_classes": list(ACTION_CLASSES),
        "duplicate_sequences": 0,
        "sequence_fingerprints": [],
        "issues": [],
        "experimental_train_ready": False,
        "all_classes_covered": False,
    }


def audit_sequence_directory(root: Path) -> dict:
    """Inspect the exact NPZ contract consumed by the temporal trainer.

    Corrupt, non-finite, wrongly shaped and out-of-range sequences are excluded
    from every readiness count.  This keeps the Accuracy Lab from reporting a
    healthy dataset that the trainer cannot safely consume.
    """
    root = Path(root)
    result = _empty_audit(root)
    paths = sorted(root.glob("*.npz")) if root.exists() else []
    result["files"] = len(paths)
    fights: set[str] = set()
    fingerprints: set[str] = set()

    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "x" not in data or "y" not in data:
                    raise ValueError("missing required x or y array")
                sequence = np.asarray(data["x"], dtype=np.float32)
                label_array = np.asarray(data["y"])
                if sequence.ndim != 2:
                    raise ValueError(f"x must be 2D, got shape {sequence.shape}")
                if sequence.shape[0] != SETTINGS.action_window:
                    raise ValueError(
                        f"x must contain {SETTINGS.action_window} frames, got {sequence.shape[0]}"
                    )
                if sequence.shape[1] != 102:
                    raise ValueError(f"x must contain 102 features, got {sequence.shape[1]}")
                if not np.isfinite(sequence).all():
                    raise ValueError("x contains NaN or infinite values")
                if label_array.size != 1:
                    raise ValueError("y must be one scalar class index")
                label_index = int(label_array.item())
                if label_index < 0 or label_index >= len(ACTION_CLASSES):
                    raise ValueError(f"y class index {label_index} is outside the supported range")
                if "fight_id" in data:
                    fight_id = str(np.asarray(data["fight_id"]).item()).strip()
                else:
                    fight_id = path.stem.split("__", 1)[0].strip()
                if not fight_id:
                    raise ValueError("fight_id is empty")

                digest = hashlib.sha256()
                digest.update(sequence.tobytes(order="C"))
                digest.update(str(label_index).encode("ascii"))
                fingerprint = digest.hexdigest()
        except Exception as exc:
            result["issues"].append({"file": path.name, "reason": str(exc)})
            continue

        result["valid_sequences"] += 1
        fights.add(fight_id)
        label = ACTION_CLASSES[label_index]
        result["class_support"][label] += 1
        if fingerprint in fingerprints:
            result["duplicate_sequences"] += 1
        fingerprints.add(fingerprint)

    result["invalid_sequences"] = result["files"] - result["valid_sequences"]
    result["fight_ids"] = sorted(fights)
    result["fights"] = len(fights)
    result["sequence_fingerprints"] = sorted(fingerprints)
    result["negative_sequences"] = result["class_support"]["none"]
    result["positive_sequences"] = result["valid_sequences"] - result["negative_sequences"]
    result["covered_classes"] = sum(value > 0 for value in result["class_support"].values())
    result["missing_classes"] = [name for name, value in result["class_support"].items() if value == 0]
    result["all_classes_covered"] = not result["missing_classes"]
    result["experimental_train_ready"] = bool(
        result["invalid_sequences"] == 0
        and result["duplicate_sequences"] == 0
        and result["fights"] >= 2
        and result["positive_sequences"] >= 20
        and result["negative_sequences"] >= 20
    )
    return result


def audit_dataset_split(development_root: Path, untouched_test_root: Path) -> dict:
    development = audit_sequence_directory(development_root)
    untouched = audit_sequence_directory(untouched_test_root)
    fight_overlap = sorted(set(development["fight_ids"]) & set(untouched["fight_ids"]))
    content_overlap = len(
        set(development["sequence_fingerprints"]) & set(untouched["sequence_fingerprints"])
    )
    test_ready = bool(
        untouched["invalid_sequences"] == 0
        and untouched["fights"] >= 3
        and untouched["all_classes_covered"]
        and not fight_overlap
        and not content_overlap
    )
    return {
        "development": development,
        "untouched_test": untouched,
        "fight_overlap": fight_overlap,
        "content_overlap": content_overlap,
        "leakage_safe": not fight_overlap and not content_overlap,
        "untouched_test_ready": test_ready,
        "release_data_ready": development["experimental_train_ready"] and test_ready,
    }
