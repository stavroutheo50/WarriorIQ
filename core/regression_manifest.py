from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REGRESSION_MANIFEST_SCHEMA = "warrioriq.end_to_end_regression.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(fights: list[dict[str, Any]]) -> str:
    payload = {"schema": REGRESSION_MANIFEST_SCHEMA, "fights": fights}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_regression_manifest(fights: list[dict[str, Any]], *, created_at: str) -> dict[str, Any]:
    ordered = sorted(fights, key=lambda item: str(item.get("fight_id", "")))
    manifest = {
        "schema": REGRESSION_MANIFEST_SCHEMA,
        "created_at": created_at,
        "fight_count": len(ordered),
        "annotation_count": sum(len(item.get("annotations", [])) for item in ordered),
        "fights": ordered,
    }
    manifest["content_sha256"] = _content_digest(ordered)
    validate_regression_manifest(manifest)
    return manifest


def validate_regression_manifest(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema") != REGRESSION_MANIFEST_SCHEMA:
        raise RuntimeError("Unsupported end-to-end regression manifest schema")
    fights = payload.get("fights")
    if not isinstance(fights, list):
        raise RuntimeError("Regression manifest fights must be a list")
    if payload.get("content_sha256") != _content_digest(fights):
        raise RuntimeError("Regression manifest content hash does not match; rebuild the frozen manifest")
    if payload.get("fight_count") != len(fights):
        raise RuntimeError("Regression manifest fight count does not match its contents")

    flattened: list[dict[str, Any]] = []
    seen_fights: set[str] = set()
    for fight in fights:
        if not isinstance(fight, dict):
            raise RuntimeError("Every regression fight must be an object")
        fight_id = str(fight.get("fight_id", "")).strip()
        if not fight_id or fight_id in seen_fights:
            raise RuntimeError("Regression fight IDs must be non-empty and unique")
        seen_fights.add(fight_id)
        annotations = fight.get("annotations")
        if not isinstance(annotations, list):
            raise RuntimeError(f"Regression annotations for {fight_id} must be a list")
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise RuntimeError(f"Regression annotation for {fight_id} must be an object")
            if not isinstance(annotation.get("predicted"), dict) or not isinstance(annotation.get("corrected"), dict):
                raise RuntimeError(f"Regression annotation for {fight_id} is missing predicted/corrected labels")
            try:
                event_time = float(annotation["event_time"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Regression annotation for {fight_id} has an invalid event time") from error
            flattened.append({**annotation, "job_id": fight_id, "event_time": event_time})

    if payload.get("annotation_count") != len(flattened):
        raise RuntimeError("Regression manifest annotation count does not match its contents")
    return flattened
