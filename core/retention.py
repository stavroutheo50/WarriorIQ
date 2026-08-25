from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.config import OUTPUTS, UPLOADS


GUEST_RETENTION_HOURS = 2


def mark_guest_job(job_id: str, guest_id: str, video_path: str) -> dict:
    created = datetime.now(timezone.utc)
    metadata = {
        "job_id": job_id,
        "guest_id": guest_id,
        "video_path": video_path,
        "created_at": created.isoformat(),
        "expires_at": (created + timedelta(hours=GUEST_RETENTION_HOURS)).isoformat(),
    }
    folder = OUTPUTS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "guest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def guest_job(job_id: str) -> dict | None:
    path = OUTPUTS / job_id / "guest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def guest_job_valid(job_id: str, guest_id: str) -> bool:
    metadata = guest_job(job_id)
    if not metadata or metadata.get("guest_id") != guest_id:
        return False
    try:
        return datetime.fromisoformat(metadata["expires_at"]) > datetime.now(timezone.utc)
    except (KeyError, ValueError, TypeError):
        return False


def cleanup_expired_guest_jobs(protected_job_ids: set[str] | None = None) -> list[str]:
    protected_job_ids = protected_job_ids or set()
    removed: list[str] = []
    now = datetime.now(timezone.utc)
    outputs_root = OUTPUTS.resolve()
    uploads_root = UPLOADS.resolve()
    for marker in OUTPUTS.glob("*/guest.json"):
        if marker.parent.name in protected_job_ids:
            continue
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            expired = datetime.fromisoformat(metadata["expires_at"]) <= now
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            expired = True
            metadata = {}
        if not expired:
            continue
        job_dir = marker.parent.resolve()
        video_text = metadata.get("video_path")
        if video_text:
            video = Path(video_text).resolve()
            if video.parent == uploads_root:
                try:
                    video.unlink(missing_ok=True)
                except OSError:
                    # A browser or decoder may still hold the file on Windows.
                    # Keep the marker so the next cleanup can retry safely.
                    continue
        if job_dir.parent == outputs_root:
            shutil.rmtree(job_dir, ignore_errors=True)
        removed.append(marker.parent.name)
    return removed
