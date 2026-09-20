from __future__ import annotations

import json
import importlib.util
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from core.config import OUTPUTS, SETTINGS


_jobs: dict[str, dict] = {}
# Guards the in-memory mirror and the per-job lock table only. It is never
# held while waiting on a per-job lock or on disk, so one stalled job cannot
# freeze state access for every other job in this process.
_lock = RLock()
_job_locks: dict[str, "_JobLock"] = {}
_SESSION_FILE = "analysis-session.json"
_WORKER_HEARTBEAT_FILE = "worker-heartbeat.json"
_CLAIM_DIRECTORY = ".claim"
_TRANSIENT_KEYS = {"report"}
LOGGER = logging.getLogger("warrioriq.state")
_ARTIFACT_NAMES = {"report.json", "report.html", "tracking.jsonl", "events.json"}


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


class _JobLock:
    """One job's lock: reentrant in this process, exclusive across processes."""

    __slots__ = ("guard", "depth")

    def __init__(self) -> None:
        self.guard = RLock()
        self.depth = 0


def _remember(job_id: str, job: dict) -> None:
    with _lock:
        _jobs[job_id] = job


def _forget(job_id: str) -> None:
    # The lock object itself is deliberately kept. Dropping it while another
    # thread waits on it would hand the two of them different guards, and both
    # would then try to take the same file lock on separate descriptors.
    with _lock:
        _jobs.pop(job_id, None)


def _hold_file_lock(handle, acquire: bool) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if acquire else fcntl.LOCK_UN)


@contextmanager
def _job_lock(job_id: str, *, create: bool = False):
    """Serialize session commits across the web process and local workers.

    The process-wide lock is taken only to find this job's lock, and released
    before anything can block, so waiting on one job never stalls state access
    for another. The file lock is taken once per outermost entry: the guard is
    reentrant, and flock() on a second descriptor for the same file in one
    process deadlocks rather than nesting.
    """
    with _lock:
        job_lock = _job_locks.get(job_id)
        if job_lock is None:
            job_lock = _job_locks[job_id] = _JobLock()
    with job_lock.guard:
        directory = _session_path(job_id).parent
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir() or job_lock.depth:
            # Already inside this job's file lock, or there is no directory to
            # put one in. The reentrant guard is the whole exclusion here.
            job_lock.depth += 1
            try:
                yield
            finally:
                job_lock.depth -= 1
            return
        with (directory / ".state.lock").open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            _hold_file_lock(handle, True)
            job_lock.depth = 1
            try:
                yield
            finally:
                job_lock.depth = 0
                _hold_file_lock(handle, False)


def analysis_run_directory(job_id: str, analysis_run_id: str) -> Path:
    """Keep each generation's files separate, including failed partial output."""
    if not analysis_run_id or any(c not in "0123456789abcdef" for c in analysis_run_id):
        raise ValueError("Invalid analysis generation")
    return OUTPUTS / job_id / ".runs" / analysis_run_id


def completed_artifact_directory(job_id: str, job: dict | None = None) -> Path | None:
    """Return only the current committed result, with pre-upgrade compatibility."""
    if _session_path(job_id).exists():
        with _job_lock(job_id):
            persisted = _read_session(job_id)
        if persisted is None:
            return None
        job = persisted
    if job is None:
        job = get_job(job_id)
    if job is None:
        return OUTPUTS / job_id
    if job.get("status") != "complete":
        return None
    if not job.get("artifact_isolation_version"):
        return OUTPUTS / job_id
    run_id = str(job.get("analysis_run_id") or "")
    if not run_id or job.get("artifacts_run_id") != run_id:
        return None
    return analysis_run_directory(job_id, run_id)


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
    with _job_lock(job_id, create=True):
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
        _remember(job_id, job)
        _write_session(job_id, job)


def update_job(job_id: str, patch: dict) -> bool:
    with _job_lock(job_id):
        # Always re-read persisted state. In external-worker mode the web and
        # GPU processes have separate memories and the file is their contract.
        job = _read_session(job_id) or _jobs.get(job_id)
        if job is None:
            return False
        job.update(patch)
        job["updated_at_epoch"] = time.time()
        _remember(job_id, job)
        return _write_session(job_id, job)


def get_job(job_id: str) -> dict | None:
    with _job_lock(job_id):
        persisted = _read_session(job_id, recover_orphan=job_id not in _jobs)
        job = persisted or _jobs.get(job_id)
        if job is not None:
            _remember(job_id, job)
        return dict(job) if job is not None else None


def list_jobs() -> list[tuple[str, dict]]:
    # Never hold the process-wide lock across a per-job lock. Acquiring them in
    # that order here, while every job-level write takes them in the opposite
    # order, is what turns one busy worker into a stalled web process.
    for path in OUTPUTS.glob(f"*/{_SESSION_FILE}"):
        job_id = path.parent.name
        # A web process and an external worker have separate memories.
        # Refresh known jobs too, otherwise navigation and cleanup can act
        # on an old queued/running status after the worker has advanced it.
        with _job_lock(job_id):
            job = _read_session(job_id, recover_orphan=job_id not in _jobs)
        if job is not None:
            _remember(job_id, job)
    with _lock:
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
        "artifacts_run_id": None,
        "history_saved": None,
        "artifact_isolation_version": 1,
        **patch,
    }
    # A detached worker discovers queued work only through this file. If it
    # cannot be written the analysis would sit at "Queued" forever, so fail the
    # request instead of stranding the fight silently.
    if not update_job(job_id, reset):
        raise AnalysisStateNotPersisted(f"Queued analysis {job_id} could not be persisted for a worker to claim")
    return analysis_run_id


def start_job_run(job_id: str, worker_id: str, analysis_run_id: str) -> bool:
    """Move the exact queued generation to running."""
    with _job_lock(job_id):
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
        _remember(job_id, job)
        return _write_session(job_id, job)


def delete_job(job_id: str) -> None:
    with _job_lock(job_id):
        _session_path(job_id).unlink(missing_ok=True)
    _forget(job_id)


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
            with _job_lock(job_id):
                job = _read_session(job_id)
                if not job or job.get("status") != "queued":
                    continue
                now = time.time()
                requested = job.get("wake_requested_at_epoch")
                wake_latency = round(now - float(requested), 2) if requested else None
                analysis_run_id = str(job.get("analysis_run_id") or uuid.uuid4().hex)
                job.update({
                    "status": "running",
                    "message": "GPU worker accepted the fight",
                    "worker_id": worker_id,
                    "analysis_run_id": analysis_run_id,
                    "worker_started_at_epoch": now,
                    "wake_latency_seconds": wake_latency,
                    "worker_heartbeat_epoch": now,
                    "worker_lease_expires_epoch": now + SETTINGS.worker_lease_seconds,
                    "updated_at_epoch": now,
                })
                if not _write_session(job_id, job):
                    continue
                _remember(job_id, job)
                return job_id, dict(job)
        finally:
            shutil.rmtree(claim_path, ignore_errors=True)
    return None


def wake_observations(limit: int = 40) -> list[float]:
    """Observed seconds between asking the machine to wake and it claiming work.

    Read from the jobs themselves rather than a separate counter, so the record
    cannot drift from what actually happened.
    """
    seen: list[tuple[float, float]] = []
    for path in OUTPUTS.glob(f"*/{_SESSION_FILE}"):
        try:
            job = get_job(path.parent.name)
        except Exception:
            continue
        if not job:
            continue
        latency = job.get("wake_latency_seconds")
        started = job.get("worker_started_at_epoch")
        if latency is None or started is None:
            continue
        try:
            seen.append((float(started), float(latency)))
        except (TypeError, ValueError):
            continue
    seen.sort(reverse=True)
    return [latency for _, latency in seen[:limit]]


def wake_status() -> dict:
    """What the wake path is actually achieving, as a number rather than a hope."""
    latencies = wake_observations()
    if not latencies:
        return {"observations": 0, "median_seconds": None, "fastest_seconds": None,
                "slowest_seconds": None}
    ordered = sorted(latencies)
    middle = ordered[len(ordered) // 2]
    return {
        "observations": len(ordered),
        "median_seconds": round(middle, 1),
        "fastest_seconds": round(ordered[0], 1),
        "slowest_seconds": round(ordered[-1], 1),
    }


def update_job_for_worker(
    job_id: str,
    worker_id: str,
    analysis_run_id: str,
    patch: dict,
    *,
    renew_lease: bool = True,
) -> bool:
    """Update only while this worker owns the exact live analysis generation."""
    with _job_lock(job_id):
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
        _remember(job_id, job)
        return _write_session(job_id, job)


def finalize_job_from_worker(
    job_id: str,
    worker_id: str,
    analysis_run_id: str,
    report: dict,
    artifacts: dict[str, Path],
) -> bool:
    """Commit one complete generation after all its files have been written.

    Readers resolve the directory through the atomic session file. Failed or
    superseded generations never become current, and older files are retained.
    """
    if set(artifacts) - _ARTIFACT_NAMES:
        raise ValueError("Unsupported analysis artifact")
    with _job_lock(job_id):
        job = _read_session(job_id)
        if (
            not job
            or job.get("status") != "running"
            or job.get("worker_id") != worker_id
            or job.get("analysis_run_id") != analysis_run_id
        ):
            return False
        job_dir = analysis_run_directory(job_id, analysis_run_id)
        temporary_report = job_dir / "report.json.tmp"
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            temporary_report.write_text(
                json.dumps(_json_safe({**report, "analysis_run_id": analysis_run_id}), separators=(",", ":")),
                encoding="utf-8",
            )
            for name, source in artifacts.items():
                if name != "report.json" and source.resolve() != (job_dir / name).resolve():
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
                "artifacts_run_id": analysis_run_id,
                "artifact_isolation_version": 1,
            })
            if not _write_session(job_id, job):
                return False
            _remember(job_id, job)
            return True
        except OSError as exc:
            temporary_report.unlink(missing_ok=True)
            LOGGER.error(
                "remote_worker_finalize_failed job_id=%s worker_id=%s error=%s",
                job_id, worker_id, type(exc).__name__,
            )
            return False


def persist_completed_job(job_id: str, analysis_run_id: str) -> bool:
    """Save history for the current generation without invalidating its report."""
    from core.db import save_completed_analysis

    with _job_lock(job_id):
        job = _read_session(job_id)
        if not job or job.get("status") != "complete" or job.get("artifacts_run_id") != analysis_run_id:
            return False
        if not job.get("persist_result") or job.get("history_saved"):
            return True
        report_path = analysis_run_directory(job_id, analysis_run_id) / "report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            save_completed_analysis(job_id, job, report, str(report_path))
        except Exception as exc:
            LOGGER.error("analysis_history_save_failed job_id=%s error_type=%s", job_id, type(exc).__name__)
            job["history_saved"] = False
            job["message"] = "Analysis complete. Saving to fight history failed; your report is still available. Open it to retry saving."
        else:
            job["history_saved"] = True
            job["message"] = "Complete"
        if _write_session(job_id, job):
            _remember(job_id, job)
        return job["history_saved"]


def renew_job_lease(job_id: str, worker_id: str, analysis_run_id: str | None = None) -> bool:
    job = get_job(job_id)
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
