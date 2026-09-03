from __future__ import annotations

import argparse
import os
import hashlib
import logging
import shutil
import socket
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import contextlib
import ctypes
import sys

from dotenv import load_dotenv

load_dotenv()

from app.state import (
    AnalysisRunLost, claim_next_job, get_job, record_worker_heartbeat,
    update_job, update_job_for_worker,
)
from core.config import OUTPUTS, SETTINGS
from core.db import release_analysis
from core.types import AnalysisRequest
from core.worker_client import RemoteWorkerClient, RemoteWorkerError, retry_heartbeat


LOGGER = logging.getLogger("warrioriq.worker")


def _request_from_job(job_id: str, job: dict) -> AnalysisRequest:
    return AnalysisRequest(
        video_path=job["video_path"],
        fighter_a_box=list(job["fighter_a_box"]),
        fighter_b_box=list(job["fighter_b_box"]),
        original_name=job.get("original_name"),
        analysis_target="BOTH",
        focus_fighter=job.get("focus_fighter") or "A",
        fight_type=job["fight_type"],
        ruleset=job["ruleset"],
        start_seconds=float(job.get("start_seconds", 0.0)),
        round_count=int(job.get("round_count", 1)),
        round_duration_seconds=float(job.get("round_duration_seconds", 120.0)),
        break_duration_seconds=float(job.get("break_duration_seconds", 60.0)),
        selected_rounds=job.get("selected_rounds"),
        end_seconds=job.get("end_seconds"),
        job_id=job_id,
        profile_id=int(job.get("profile_id", 0)),
        persist_result=bool(job.get("persist_result", False)),
        openai_identity_recovery=bool(job.get("openai_identity_recovery", False)),
        fighter_id=job.get("fighter_id"),
    )


def run_claimed_job(worker_id: str, job_id: str, job: dict) -> None:
    from core.analyzer import analyze

    analysis_run_id = str(job.get("analysis_run_id") or "")

    def progress(patch: dict) -> None:
        if not update_job_for_worker(job_id, worker_id, analysis_run_id, patch):
            raise AnalysisRunLost(f"Analysis run {analysis_run_id} no longer owns {job_id}")
        record_worker_heartbeat(worker_id, job_id)

    try:
        report = analyze(_request_from_job(job_id, job), progress)
        if not update_job_for_worker(job_id, worker_id, analysis_run_id, {
            "status": "complete", "report": report, "percent": 100.0,
            "message": "Complete", "worker_lease_expires_epoch": None,
        }, renew_lease=False):
            LOGGER.warning("Analysis completion discarded for superseded job %s", job_id)
    except AnalysisRunLost:
        LOGGER.warning("Analysis worker lost ownership of job %s; stale output was discarded", job_id)
    except Exception as exc:
        current = get_job(job_id) or job
        still_owns_run = current.get("analysis_run_id") == analysis_run_id
        if still_owns_run and current.get("usage_reserved") and current.get("account_id"):
            release_analysis(int(current["account_id"]), job_id)
            update_job(job_id, {"usage_reserved": False})
        LOGGER.exception("Analysis job %s failed", job_id, exc_info=exc)
        if still_owns_run:
            update_job(job_id, {
                "status": "error",
                "message": "WarriorIQ could not finish this analysis. Your upload and fighter selections are preserved so you can try again.",
                "worker_lease_expires_epoch": None,
            })


@contextlib.contextmanager
def _keep_machine_awake():
    """Stop Windows sleeping mid-analysis, and only mid-analysis.

    A worker that runs all the time must not also keep the machine awake all
    the time - the whole design depends on the PC sleeping between fights and
    being woken by a magic packet. So the hold is taken when a job starts and
    released the moment it finishes: the analysis can never be suspended
    halfway, and an idle worker never blocks sleep.

    Sleeping does not kill the process. The worker is suspended with the
    machine and resumes polling the instant it wakes, which is why a queued
    fight starts seconds after the wake rather than waiting for a timer.
    """
    if sys.platform != "win32":
        yield
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:
        LOGGER.debug("Could not take a sleep hold; continuing without one")
        yield
        return
    try:
        yield
    finally:
        try:
            kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            LOGGER.debug("Could not release the sleep hold")


def run_worker(*, once: bool = False) -> int:
    worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
    if SETTINGS.worker_remote_url:
        if not SETTINGS.worker_token:
            LOGGER.error("WARRIORIQ_WORKER_TOKEN is required with WARRIORIQ_WORKER_REMOTE_URL")
            return 2
        return run_remote_worker(worker_id, once=once)
    while True:
        record_worker_heartbeat(worker_id)
        claimed = claim_next_job(worker_id)
        if claimed:
            job_id, job = claimed
            record_worker_heartbeat(worker_id, job_id)
            run_claimed_job(worker_id, job_id, job)
            record_worker_heartbeat(worker_id)
        elif once:
            return 0
        else:
            time.sleep(SETTINGS.worker_poll_seconds)


def _remote_request(job: dict, video_path: Path) -> AnalysisRequest:
    payload = dict(job)
    payload.update({
        "video_path": str(video_path),
        "original_name": "Fight video",
        "profile_id": 0,
        "persist_result": False,
    })
    return _request_from_job(str(job["job_id"]), payload)


def _worker_result_archive(job_id: str, destination: Path) -> None:
    job_dir = OUTPUTS / job_id
    required = (job_dir / "report.json", job_dir / "tracking.jsonl")
    if any(not path.is_file() for path in required):
        raise RuntimeError("Analysis completed without the report or skeleton tracking artifact")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in ("report.json", "tracking.jsonl", "events.json"):
            path = job_dir / name
            if path.is_file():
                bundle.write(path, arcname=name)


def _keep_remote_lease(
    client: RemoteWorkerClient,
    job_id: str,
    analysis_run_id: str,
    stop: threading.Event,
    ownership_lost: threading.Event,
) -> None:
    """Renew ownership while downloads or model startup have no progress events."""
    interval = max(10.0, min(30.0, SETTINGS.worker_lease_seconds / 3.0))
    while not stop.wait(interval):
        try:
            client.progress(job_id, analysis_run_id, {})
        except AnalysisRunLost:
            ownership_lost.set()
            return
        except RemoteWorkerError as exc:
            # A brief network problem should not stop local inference. The next
            # heartbeat or normal progress update can still renew the lease.
            LOGGER.warning("Remote lease renewal failed for job %s: %s", job_id, exc)


def run_remote_claimed_job(client: RemoteWorkerClient, job: dict) -> None:
    from core.analyzer import analyze

    job_id = str(job["job_id"])
    analysis_run_id = str(job["analysis_run_id"])
    output_dir = OUTPUTS / job_id
    stop_keepalive = threading.Event()
    ownership_lost = threading.Event()
    keepalive = threading.Thread(
        target=_keep_remote_lease,
        args=(client, job_id, analysis_run_id, stop_keepalive, ownership_lost),
        name=f"warrioriq-lease-{job_id}",
        daemon=True,
    )
    keepalive.start()
    try:
        with tempfile.TemporaryDirectory(prefix=f"warrioriq-{job_id}-") as temporary:
            video_path = Path(temporary) / f"fight{job.get('video_extension') or '.mp4'}"
            archive_path = Path(temporary) / "worker-result.zip"
            client.download_video(job, video_path)
            if ownership_lost.is_set():
                raise AnalysisRunLost(f"Analysis run {analysis_run_id} no longer owns {job_id}")
            if output_dir.parent.resolve() != OUTPUTS.resolve():
                raise RuntimeError("Unsafe worker output path")
            shutil.rmtree(output_dir, ignore_errors=True)

            def progress(patch: dict) -> None:
                if ownership_lost.is_set():
                    raise AnalysisRunLost(f"Analysis run {analysis_run_id} no longer owns {job_id}")
                client.progress(job_id, analysis_run_id, patch)

            analyze(_remote_request(job, video_path), progress)
            if ownership_lost.is_set():
                raise AnalysisRunLost(f"Analysis run {analysis_run_id} no longer owns {job_id}")
            _worker_result_archive(job_id, archive_path)
            client.complete(job_id, analysis_run_id, archive_path)
    except AnalysisRunLost:
        LOGGER.warning("Remote worker lost ownership of job %s; output was discarded", job_id)
    except Exception as exc:
        LOGGER.exception("Remote analysis job %s failed", job_id, exc_info=exc)
        try:
            client.failed(job_id, analysis_run_id, type(exc).__name__)
        except (AnalysisRunLost, RemoteWorkerError):
            LOGGER.warning("Could not report remote failure for job %s", job_id)
    finally:
        stop_keepalive.set()
        keepalive.join(timeout=2.0)
        if output_dir.parent.resolve() == OUTPUTS.resolve():
            shutil.rmtree(output_dir, ignore_errors=True)


def _source_fingerprint() -> str:
    """A fingerprint of the code this worker is running.

    The analysis runs here, on the GPU machine, not on the web server. So
    deploying the website changes nothing about how a fight is analysed, and a
    worker started before a fix keeps running the old code with no sign that it
    is doing so. One worker ran for five hours across fourteen commits - every
    analysis fix of the day sat on disk, loaded by nothing, while the fights it
    was supposed to fix came back unchanged.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("core/*.py")) + [root / "worker.py"]:
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(path.name.encode())
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(stat.st_size).encode())
    return digest.hexdigest()[:12]


def _code_changed_since(fingerprint: str) -> bool:
    return _source_fingerprint() != fingerprint


def run_remote_worker(worker_id: str, *, once: bool = False) -> int:
    client = RemoteWorkerClient(SETTINGS.worker_remote_url, SETTINGS.worker_token, worker_id)
    running_code = _source_fingerprint()
    LOGGER.info("Worker running analysis code %s", running_code)
    while True:
        # Checked between jobs, never during one. Exiting hands the queue back
        # cleanly and whatever supervises this restarts it on the new code.
        if _code_changed_since(running_code):
            # Restart in place rather than exiting. Exiting assumed something
            # was supervising this process; when nothing was, the worker simply
            # vanished the moment the code changed and a queued fight sat there
            # with no one to claim it. Re-exec needs no supervisor and cannot
            # leave a hole.
            LOGGER.warning(
                "Analysis code changed on disk (was %s, now %s); restarting on it",
                running_code, _source_fingerprint(),
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, [sys.executable, *sys.argv])
        try:
            retry_heartbeat(client)
            claimed = client.claim()
        except RemoteWorkerError as exc:
            LOGGER.warning("Remote worker connection unavailable: %s", exc)
            if once:
                return 1
            time.sleep(max(2.0, SETTINGS.worker_poll_seconds))
            continue
        except Exception:
            # A long-lived worker must outlive surprises. Only connection
            # faults were handled above, so anything else -- a decode error, a
            # transient filesystem failure, a bug in one job's payload -- ended
            # the process and left the queue unattended until someone noticed
            # and restarted it by hand. Log it and carry on; the job's own
            # lease expires and the fight becomes claimable again.
            LOGGER.exception("Unexpected worker failure; continuing to poll")
            if once:
                return 1
            time.sleep(max(2.0, SETTINGS.worker_poll_seconds))
            continue
        try:
            if claimed:
                with _keep_machine_awake():
                    run_remote_claimed_job(client, claimed)
            elif once:
                return 0
            else:
                time.sleep(SETTINGS.worker_poll_seconds)
        except AnalysisRunLost:
            # A newer run or a recovery already owns this fight.
            LOGGER.warning("Analysis run ownership lost; returning to the queue")
        except Exception:
            LOGGER.exception("Analysis failed unexpectedly; continuing to poll")
            if once:
                return 1
            time.sleep(max(2.0, SETTINGS.worker_poll_seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WarriorIQ durable GPU analysis worker")
    parser.add_argument("--once", action="store_true", help="Claim at most one queued job and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(run_worker(once=args.once))
