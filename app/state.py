from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from core.config import OUTPUTS


_jobs: dict[str, dict] = {}
_lock = Lock()
_SESSION_FILE = "analysis-session.json"
_TRANSIENT_KEYS = {"report"}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _session_path(job_id: str) -> Path:
    return OUTPUTS / job_id / _SESSION_FILE


def _write_session(job_id: str, job: dict) -> None:
    path = _session_path(job_id)
    temporary = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in job.items() if key not in _TRANSIENT_KEYS}
        temporary.write_text(json.dumps(_json_safe(payload), separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        # Persistence failure must not crash an otherwise healthy analysis.
        # The current process still retains the authoritative in-memory job.
        temporary.unlink(missing_ok=True)


def _read_session(job_id: str) -> dict | None:
    path = _session_path(job_id)
    if not path.exists():
        return None
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(job, dict):
        return None
    # A worker cannot survive a server-process restart. Preserve the upload and
    # selections, but describe the interruption honestly so it can be restarted
    # without uploading the video or selecting the fighters again.
    if job.get("status") in {"queued", "running"}:
        job.update({
            "status": "interrupted",
            "message": "The analysis server restarted. Your video and fighter selections are safe; restart the analysis to continue.",
            "eta_seconds": None,
        })
        _write_session(job_id, job)
    return job


def create_job(job_id: str, data: dict) -> None:
    with _lock:
        now = time.time()
        job = {
            "job_id": job_id,
            "status": "selecting",
            "percent": 0.0,
            "message": "Choose a clear fighter-selection frame",
            "created_at_epoch": now,
            "updated_at_epoch": now,
            **data,
        }
        _jobs[job_id] = job
        _write_session(job_id, job)


def update_job(job_id: str, patch: dict) -> None:
    with _lock:
        job = _jobs.get(job_id) or _read_session(job_id)
        if job is None:
            return
        job.update(patch)
        job["updated_at_epoch"] = time.time()
        _jobs[job_id] = job
        _write_session(job_id, job)


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            job = _read_session(job_id)
            if job is not None:
                _jobs[job_id] = job
        return dict(job) if job is not None else None


def list_jobs() -> list[tuple[str, dict]]:
    with _lock:
        for path in OUTPUTS.glob(f"*/{_SESSION_FILE}"):
            job_id = path.parent.name
            if job_id not in _jobs:
                job = _read_session(job_id)
                if job is not None:
                    _jobs[job_id] = job
        return [(job_id, dict(job)) for job_id, job in _jobs.items()]


def delete_job(job_id: str) -> None:
    with _lock:
        _jobs.pop(job_id, None)
        _session_path(job_id).unlink(missing_ok=True)
