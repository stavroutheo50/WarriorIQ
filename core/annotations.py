from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.action import Sample, _feature_vector
from core.config import DATASET, OUTPUTS, SETTINGS
from core.scoring import is_legal_event
from core.temporal_model import ACTION_CLASSES
from core.types import StrikeEvent


def _temporal_label(technique: str) -> str:
    value = (technique or "none").lower()
    for height in ("low", "body", "head"):
        value = value.replace(f"_{height}_kick", "_round_kick")
    return value if value in ACTION_CLASSES else "none"


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


def export_sequence(job_id: str, annotation_id: int, corrected: dict, event_time: float) -> str | None:
    tracking = OUTPUTS / job_id / "tracking.jsonl"
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
        fight_id=np.asarray(job_id), fighter=np.asarray(fighter),
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
        text = (value or "").lower()
        return "left" if "left" in text else "right" if "right" in text else "unspecified"

    for item in annotations:
        predicted, corrected = item["predicted"], item["corrected"]
        for name, (field, _) in fields.items():
            if name == "limb_side":
                a, b = side(predicted.get("technique")), side(corrected.get("technique"))
            elif name == "legality":
                # A negative label means the candidate was not an action, so
                # legality is not a meaningful comparison for that sample.
                if corrected.get("technique") == "none":
                    continue
                def event(data):
                    return StrikeEvent(data.get("fighter", "A"), "B", 1, 0, 0, 0, 0, 0, 0,
                                       data.get("technique", "none"), data.get("family", "punch"), data.get("limb", "right_hand"),
                                       outcome=data.get("outcome", "uncertain"), target=data.get("target"))
                a = is_legal_event(event(predicted), item["ruleset"])
                b = is_legal_event(event(corrected), item["ruleset"])
            else:
                a, b = predicted.get(field), corrected.get(field)
            counts[name]["total"] += 1
            counts[name]["correct"] += int(a == b)
    for value in counts.values():
        value["accuracy"] = None if not value["total"] else value["correct"] / value["total"]
        value["passed"] = value["accuracy"] is not None and value["accuracy"] >= value["threshold"]
    fights = len({item["job_id"] for item in annotations})
    negative_labels = sum(item.get("corrected", {}).get("technique") == "none" for item in annotations)
    positive_labels = len(annotations) - negative_labels
    return {
        "metrics": counts,
        "annotations": len(annotations),
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "fights": fights,
        "train_ready": fights >= 2 and positive_labels >= 20 and negative_labels >= 20,
    }
