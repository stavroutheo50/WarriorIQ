from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.action import Sample, _feature_vector
from core.config import DATASET, OUTPUTS, SETTINGS
from core.fight_stats import normalize_outcome
from core.model_validation import classification_metrics
from core.scoring import is_legal_event
from core.temporal_model import ACTION_CLASSES
from core.types import StrikeEvent


def _temporal_label(technique: str) -> str:
    value = (technique or "none").lower()
    for height in ("low", "body", "head"):
        value = value.replace(f"_{height}_kick", "_round_kick")
    return value if value in ACTION_CLASSES else "none"


def _reviewed_label(value: object) -> str | None:
    """Keep an explicit negative distinct from an absent or coarse review."""
    if not isinstance(value, str):
        return None
    technique = value.strip().lower()
    if technique == "none":
        return "none"
    label = _temporal_label(technique)
    return label if label != "none" else None


def _sample(record: dict, fighter: str) -> Sample | None:
    own = record.get(f"fighter_{fighter}", {}).get("observation")
    other = record.get(f"fighter_{'B' if fighter == 'A' else 'A'}", {}).get("observation")
    if not own or not own.get("box") or not own.get("keypoints"):
        return None
    return Sample(
        frame=int(record["source_frame"]), time=float(record["time_seconds"]),
        round_number=record.get("round_number"), box=np.asarray(own["box"], dtype=np.float32),
        keypoints=np.asarray(own["keypoints"], dtype=np.float32),
        conf=None if own.get("keypoint_conf") is None else np.asarray(own["keypoint_conf"], dtype=np.float32),
        opponent_box=None if not other or not other.get("box") else np.asarray(other["box"], dtype=np.float32),
        opponent_keypoints=None if not other or not other.get("keypoints") else np.asarray(other["keypoints"], dtype=np.float32),
        opponent_conf=None if not other or other.get("keypoint_conf") is None else np.asarray(other["keypoint_conf"], dtype=np.float32),
        identity_confidence=float(record.get(f"fighter_{fighter}", {}).get("identity_confidence", 0.0)),
        opponent_identity_confidence=float(
            record.get(f"fighter_{'B' if fighter == 'A' else 'A'}", {}).get("identity_confidence", 0.0)
        ),
    )


def export_sequence(
    job_id: str, annotation_id: int, corrected: dict, event_time: float,
    *, tracking_path: Path | None = None, source_fight_id: str | None = None,
) -> str | None:
    tracking = tracking_path if tracking_path is not None else OUTPUTS / job_id / "tracking.jsonl"
    if not tracking.exists():
        return None
    records = []
    with tracking.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if abs(float(item.get("time_seconds", -999)) - float(event_time)) <= 1.25:
                records.append(item)
    fighter = corrected.get("fighter", "A")
    samples = [sample for record in records if (sample := _sample(record, fighter)) is not None]
    if not samples:
        return None
    samples.sort(key=lambda item: item.time)
    features = []
    previous = None
    for sample in samples:
        features.append(_feature_vector(sample, previous))
        previous = sample
    needed = SETTINGS.action_window
    if len(features) >= needed:
        center = min(range(len(samples)), key=lambda i: abs(samples[i].time - event_time))
        start = max(0, min(len(features) - needed, center - needed // 2))
        features = features[start:start + needed]
    else:
        features += [features[-1].copy() for _ in range(needed - len(features))]
    label = _temporal_label(corrected.get("technique", "none"))
    folder = DATASET / "sequences"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{job_id}__annotation_{annotation_id:06d}.npz"
    np.savez_compressed(
        path, x=np.asarray(features, dtype=np.float32), y=np.int64(ACTION_CLASSES.index(label)),
        fight_id=np.asarray(source_fight_id or job_id), fighter=np.asarray(fighter),
        technique=np.asarray(corrected.get("technique", "none")),
        target=np.asarray(corrected.get("target") or "none"),
        outcome=np.asarray(corrected.get("outcome") or "uncertain"),
    )
    return str(path)


def accuracy_summary(annotations: list[dict]) -> dict:
    fields = {
        "fighter_identity": ("fighter", 0.95), "technique": ("technique", 0.90),
        "limb_side": ("technique", 0.85), "target": ("target", 0.90),
        "outcome": ("outcome", 0.85), "legality": ("legality", 0.95),
    }
    counts = {name: {"correct": 0, "total": 0, "threshold": threshold} for name, (_, threshold) in fields.items()}

    def side(value):
        if not isinstance(value, str):
            return None
        text = value.lower()
        return "left" if "left" in text else "right" if "right" in text else None

    classifiable = []
    timing_errors = []
    source_hashes = set()
    positive_labels = negative_labels = 0

    for item in annotations:
        predicted, corrected = item["predicted"], item["corrected"]
        truth = _reviewed_label(corrected.get("technique"))
        guess = _reviewed_label(predicted.get("technique"))
        if truth == "none":
            negative_labels += 1
        elif truth is not None:
            positive_labels += 1
        if truth is not None and guess is not None:
            classifiable.append((truth, guess))
        source_hash = item.get("video_sha256")
        if (truth is not None and isinstance(source_hash, str) and len(source_hash) == 64
                and all(character in "0123456789abcdefABCDEF" for character in source_hash)):
            source_hashes.add(source_hash.lower())
        for name, (field, _) in fields.items():
            if name == "limb_side":
                if truth in {None, "none"}:
                    continue
                b = side(corrected.get("limb"))
                if b is None:
                    continue
                a = side(predicted.get("limb"))
            elif name == "legality":
                if (truth in {None, "none"} or corrected.get("family") not in {"punch", "kick", "knee"}
                        or corrected.get("target") not in {"head", "body", "leg"}
                        or not item.get("ruleset")):
                    continue
                def event(data):
                    if (_reviewed_label(data.get("technique")) in {None, "none"}
                            or data.get("family") not in {"punch", "kick", "knee"}
                            or data.get("target") not in {"head", "body", "leg"}):
                        return None
                    return StrikeEvent(data.get("fighter", "A"), "B", 1, 0, 0, 0, 0, 0, 0,
                                       data["technique"], data["family"], data.get("limb") or "",
                                       outcome=data.get("outcome", "uncertain"), target=data.get("target"))
                predicted_event = event(predicted)
                a = None if predicted_event is None else is_legal_event(predicted_event, item["ruleset"])
                b = is_legal_event(event(corrected), item["ruleset"])
            elif name == "fighter_identity":
                b = corrected.get(field)
                if truth in {None, "none"} or b not in {"A", "B"}:
                    continue
                a = predicted.get(field)
            elif name == "technique":
                if truth is None:
                    continue
                a, b = predicted.get(field), corrected.get(field)
            elif name == "target":
                b = corrected.get(field)
                if truth in {None, "none"} or b not in {"head", "body", "leg"}:
                    continue
                a = predicted.get(field)
            elif name == "outcome":
                b = normalize_outcome(corrected.get(field))
                if truth in {None, "none"} or b == "uncertain":
                    continue
                a = normalize_outcome(predicted.get(field))
            else:
                a, b = predicted.get(field), corrected.get(field)
            counts[name]["total"] += 1
            counts[name]["correct"] += int(a == b)
        # Timing needs an independently supplied contact timestamp. A missing
        # timestamp is not proof that the proposal landed at exactly that time.
        if truth in {None, "none"} or guess in {None, "none"}:
            continue
        try:
            predicted_time = float(item["event_time"])
            corrected_time = float(corrected["contact_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(predicted_time) and np.isfinite(corrected_time) and predicted_time >= 0 and corrected_time >= 0:
            timing_errors.append(abs(predicted_time - corrected_time))
    for value in counts.values():
        value["accuracy"] = None if not value["total"] else value["correct"] / value["total"]
        value["passed"] = value["accuracy"] is not None and value["accuracy"] >= value["threshold"]
    fights = len(source_hashes)
    timing_samples = len(timing_errors)
    timing = {
        "samples": timing_samples,
        "mean_absolute_error_seconds": None if not timing_samples else float(np.mean(timing_errors)),
        "median_absolute_error_seconds": None if not timing_samples else float(np.median(timing_errors)),
        "p95_absolute_error_seconds": None if not timing_samples else float(np.percentile(timing_errors, 95)),
        "within_250ms": sum(value <= 0.2500001 for value in timing_errors),
        "within_500ms": sum(value <= 0.5000001 for value in timing_errors),
    }
    timing["within_250ms_rate"] = None if not timing_samples else timing["within_250ms"] / timing_samples
    timing["within_500ms_rate"] = None if not timing_samples else timing["within_500ms"] / timing_samples
    technique_validation = classification_metrics(
        [truth for truth, _ in classifiable],
        [guess for _, guess in classifiable],
        classes=ACTION_CLASSES,
    )
    return {
        "metrics": counts,
        "annotations": len(annotations),
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "fights": fights,
        "verified_source_fights": fights,
        "train_ready": fights >= 2 and positive_labels >= 20 and negative_labels >= 20,
        "technique_validation": technique_validation,
        "timing": timing,
    }
