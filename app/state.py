from __future__ import annotations

import json
import importlib.util
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from core.config import OUTPUTS, SETTINGS


_jobs: dict[str, dict] = {}
_lock = Lock()
_SESSION_FILE = "analysis-session.json"
_WORKER_HEARTBEAT_FILE = "worker-heartbeat.json"
_CLAIM_DIRECTORY = ".claim"
_TRANSIENT_KEYS = {"report"}
LOGGER = logging.getLogger("warrioriq.state")


class AnalysisRunLost(RuntimeError):
    """Raised when an older worker no longer owns an analysis run."""


class AnalysisStateNotPersisted(RuntimeError):
    """Raised when a queued run cannot be written for a detached worker to claim."""


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


def _write_session(job_id: str, job: dict) -> bool:
    path = _session_path(job_id)
    # Keep the staging name close to the final name. A long suffix pushed the
    # temporary file past the Windows 260-character path limit on deep project
    # directories, so the session silently failed to persist.
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in job.items() if key not in _TRANSIENT_KEYS}
        temporary.write_text(json.dumps(_json_safe(payload), separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
        return True
    except OSError as exc:
        # Persistence failure must not crash an otherwise healthy analysis.
        # The current process still retains the authoritative in-memory job.
        temporary.unlink(missing_ok=True)
        LOGGER.error("analysis_state_write_failed job_id=%s error=%s", job_id, type(exc).__name__)
        return False


def _read_session(job_id: str, *, recover_orphan: bool = False) -> dict | None:
    path = _session_path(job_id)
    if not path.exists():
        return None
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(job, dict):
        return None
    # Queued work is durable and may be claimed by a separate GPU process.
    # Running work is interrupted only after its renewable worker lease expires.
    lease_expires = float(job.get("worker_lease_expires_epoch", 0.0) or 0.0)
    has_worker_lease = bool(job.get("worker_id"))
    if job.get("status") == "running" and (
        (has_worker_lease and lease_expires <= time.time())
        or (recover_orphan and not has_worker_lease)
    ):
        job.update({
            "status": "interrupted",
            "message": "The analysis server restarted. Your video and fighter selections are safe; restart the analysis to continue.",
            "eta_seconds": None,
            "worker_id": None,
            "worker_lease_expires_epoch": None,
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


def update_job(job_id: str, patch: dict) -> bool:
    with _lock:
        # Always re-read persisted state. In external-worker mode the web and
        # GPU processes have separate memories and the file is their contract.
        job = _read_session(job_id) or _jobs.get(job_id)
        if job is None:
            return False
        job.update(patch)
        job["updated_at_epoch"] = time.time()
        _jobs[job_id] = job
        return _write_session(job_id, job)


def get_job(job_id: str) -> dict | None:
    with _lock:
        persisted = _read_session(job_id, recover_orphan=job_id not in _jobs)
        job = persisted or _jobs.get(job_id)
        if job is not None:
            _jobs[job_id] = job
        return dict(job) if job is not None else None


def list_jobs() -> list[tuple[str, dict]]:
    with _lock:
        for path in OUTPUTS.glob(f"*/{_SESSION_FILE}"):
            job_id = path.parent.name
            # A web process and an external worker have separate memories.
            # Refresh known jobs too, otherwise navigation and cleanup can act
            # on an old queued/running status after the worker has advanced it.
            job = _read_session(job_id, recover_orphan=job_id not in _jobs)
            if job is not None:
                _jobs[job_id] = job
        return [(job_id, dict(job)) for job_id, job in _jobs.items()]


def prepare_job_run(job_id: str, patch: dict) -> str:
    """Queue a clean analysis generation without retaining prior run output."""
    analysis_run_id = uuid.uuid4().hex
    reset = {
        "status": "queued",
        "percent": 0.0,
        "message": "Queued for fight analysis",
        "stage": "queued",
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "processed_video_seconds": 0.0,
        "fighter_a_confidence": 0.0,
        "fighter_b_confidence": 0.0,
        "current_round": None,
        "live_event_mode": "withheld",
        "live_events": [],
        "provisional_stats": {},
        "latest_observation": None,
        "report": None,
        "worker_id": None,
        "worker_started_at_epoch": None,
        "worker_heartbeat_epoch": None,
        "worker_lease_expires_epoch": None,
        "analysis_run_id": analysis_run_id,
        **patch,
    }
    # A detached worker discovers queued work only through this file. If it
    # cannot be written the analysis would sit at "Queued" forever, so fail the
    # request instead of stranding the fight silently.
    if not update_job(job_id, reset) and SETTINGS.analysis_worker_mode != "inprocess":
        raise AnalysisStateNotPersisted(f"Queued analysis {job_id} could not be persisted for a worker to claim")
    return analysis_run_id


def start_job_run(job_id: str, worker_id: str, analysis_run_id: str) -> bool:
    """Move the exact queued generation to running."""
    with _lock:
        job = _read_session(job_id) or _jobs.get(job_id)
        if (
            not job
            or job.get("status") != "queued"
            or job.get("analysis_run_id") != analysis_run_id
        ):
            return False
        now = time.time()
        job.update({
            "status": "running",
            "message": "Starting fight analysis",
            "worker_id": worker_id,
            "worker_started_at_epoch": now,
            "worker_heartbeat_epoch": now,
            "worker_lease_expires_epoch": now + SETTINGS.worker_lease_seconds,
            "updated_at_epoch": now,
        })
        _jobs[job_id] = job
        return _write_session(job_id, job)


def delete_job(job_id: str) -> None:
    with _lock:
        _jobs.pop(job_id, None)
        _session_path(job_id).unlink(missing_ok=True)


def _claim_path(job_id: str) -> Path:
    return _session_path(job_id).parent / _CLAIM_DIRECTORY


def claim_next_job(worker_id: str) -> tuple[str, dict] | None:
    """Atomically claim the oldest queued job from a shared runtime directory."""
    candidates = []
    for path in OUTPUTS.glob(f"*/{_SESSION_FILE}"):
        try:
            candidates.append((path.stat().st_mtime, path.parent.name))
        except OSError:
            continue
    for _, job_id in sorted(candidates):
        claim_path = _claim_path(job_id)
        try:
            claim_path.mkdir()
        except FileExistsError:
            continue
        try:
            job = _read_session(job_id)
            if not job or job.get("status") != "queued":
                continue
            now = time.time()
            analysis_run_id = str(job.get("analysis_run_id") or uuid.uuid4().hex)
            job.update({
                "status": "running",
                "message": "GPU worker accepted the fight",
                "worker_id": worker_id,
                "analysis_run_id": analysis_run_id,
                "worker_started_at_epoch": now,
                "worker_heartbeat_epoch": now,
                "worker_lease_expires_epoch": now + SETTINGS.worker_lease_seconds,
                "updated_at_epoch": now,
            })
            if not _write_session(job_id, job):
                continue
            with _lock:
                _jobs[job_id] = job
            return job_id, dict(job)
        finally:
            shutil.rmtree(claim_path, ignore_errors=True)
    return None


def update_job_for_worker(
    job_id: str,
    worker_id: str,
    analysis_run_id: str,
    patch: dict,
    *,
    renew_lease: bool = True,
) -> bool:
    """Update only while this worker owns the exact live analysis generation."""
    with _lock:
        job = _read_session(job_id)
        if (
            not job
            or job.get("status") != "running"
            or job.get("worker_id") != worker_id
            or job.get("analysis_run_id") != analysis_run_id
        ):
            return False
        now = time.time()
        job.update(patch)
        job["updated_at_epoch"] = now
        if renew_lease and job.get("status") == "running":
            job["worker_heartbeat_epoch"] = now
            job["worker_lease_expires_epoch"] = now + SETTINGS.worker_lease_seconds
        _jobs[job_id] = job
        return _write_session(job_id, job)


def finalize_job_from_worker(
    job_id: str,
    worker_id: str,
    analysis_run_id: str,
    report: dict,
    artifacts: dict[str, Path],
) -> bool:
    """Publish one remote worker generation atomically enough for readers.

    Remote uploads are first written to run-specific staging files by the web
    process. Only the worker that still owns the live generation may replace
    the canonical report/tracking artifacts and mark the job complete.
    """
    with _lock:
        job = _read_session(job_id)
        if (
            not job
            or job.get("status") != "running"
            or job.get("worker_id") != worker_id
            or job.get("analysis_run_id") != analysis_run_id
        ):
            return False
        job_dir = _session_path(job_id).parent
        temporary_report = job_dir / f"report.json.{analysis_run_id}.tmp"
        try:
            temporary_report.write_text(
                json.dumps(_json_safe(report), separators=(",", ":")),
                encoding="utf-8",
            )
            for name, source in artifacts.items():
                os.replace(source, job_dir / name)
            os.replace(temporary_report, job_dir / "report.json")
            now = time.time()
            job.update({
                "status": "complete",
                "report": report,
                "percent": 100.0,
                "message": "Complete",
                "stage": "complete",
                "worker_heartbeat_epoch": now,
                "worker_lease_expires_epoch": None,
                "updated_at_epoch": now,
            })
            _jobs[job_id] = job
            return _write_session(job_id, job)
        except OSError as exc:
            temporary_report.unlink(missing_ok=True)
            LOGGER.error(
                "remote_worker_finalize_failed job_id=%s worker_id=%s error=%s",
                job_id, worker_id, type(exc).__name__,
            )
            return False


def renew_job_lease(job_id: str, worker_id: str, analysis_run_id: str | None = None) -> bool:
    job = _read_session(job_id)
    if not job:
        return False
    expected_run_id = analysis_run_id or str(job.get("analysis_run_id") or "")
    return update_job_for_worker(job_id, worker_id, expected_run_id, {})


def record_worker_heartbeat(worker_id: str, current_job_id: str | None = None) -> None:
    path = OUTPUTS / _WORKER_HEARTBEAT_FILE
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = {
        "worker_id": worker_id,
        "heartbeat_epoch": time.time(),
        "current_job_id": current_job_id,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        LOGGER.error("worker_heartbeat_write_failed worker_id=%s error=%s", worker_id, type(exc).__name__)


def worker_status() -> dict:
    path = OUTPUTS / _WORKER_HEARTBEAT_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    heartbeat = float(payload.get("heartbeat_epoch", 0.0) or 0.0)
    age = max(0.0, time.time() - heartbeat) if heartbeat else None
    missing_dependencies: list[str] = []
    if SETTINGS.analysis_worker_mode == "inprocess":
        required = ["torch", "ultralytics"]
        if SETTINGS.sam_recovery_enabled or SETTINGS.sam_continuous_enabled:
            required.append("sam2")
        missing_dependencies = [name for name in required if importlib.util.find_spec(name) is None]
    mode = SETTINGS.analysis_worker_mode
    remote_configured = bool(SETTINGS.worker_token) if mode == "remote" else True
    known_mode = mode in {"inprocess", "external", "remote"}
    available = bool(
        known_mode
        and remote_configured
        and (
            not missing_dependencies
            if mode == "inprocess"
            else age is not None and age <= SETTINGS.worker_stale_seconds
        )
    )
    if available:
        reason = None
    elif not known_mode:
        reason = "worker_mode_invalid"
    elif mode == "remote" and not remote_configured:
        reason = "worker_token_missing"
    elif missing_dependencies:
        reason = "analysis_dependencies_missing"
    else:
        reason = "worker_heartbeat_missing"
    return {
        "mode": mode,
        "available": available,
        "reason": reason,
        "missing_dependencies": missing_dependencies,
        "heartbeat_age_seconds": age,
        "current_job_id": payload.get("current_job_id"),
        "queued_jobs": sum(job.get("status") == "queued" for _, job in list_jobs()),
        "running_jobs": sum(job.get("status") == "running" for _, job in list_jobs()),
    }
