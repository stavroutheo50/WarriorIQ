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

ANALYSIS_REQUIREMENTS = ("dotenv", "numpy", "cv2", "torch", "ultralytics")


def _missing_requirements() -> list[str]:
    """Which analysis dependencies this interpreter lacks.

    Asked before the imports that would need them, and before claiming
    anything. A worker that cannot import the models can still poll, claim a
    job and fail it, and - now that there is a single-worker lock - can hold
    that lock against the one interpreter that could have done the work.

    Seen on the machine that runs this: worker.py was being launched with an
    unrelated Python that had none of torch, OpenCV or ultralytics. It died on
    the first third-party import with a bare traceback, over and over, which
    reads as a broken worker rather than as the wrong Python.
    """
    import importlib.util

    return [name for name in ANALYSIS_REQUIREMENTS
            if importlib.util.find_spec(name) is None]


if __name__ == "__main__":
    _lacking = _missing_requirements()
    if _lacking:
        print("\n".join([
            "This Python cannot run the WarriorIQ analysis.",
            "  missing:     " + ", ".join(_lacking),
            "  interpreter: " + sys.executable,
            "",
            "Start the worker with the project's own environment instead:",
            "    .venv/Scripts/python.exe worker.py     (or start-worker.bat)",
            "and close whatever launched this one.",
        ]), file=sys.stderr)
        raise SystemExit(2)

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
        # Says "still here" to the single-worker lock as well as to the queue.
        refresh_worker_lock()
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


# Distinct from a crash, so a supervisor can tell a planned restart apart
# from a failing worker if it ever wants to.
EXIT_CODE_CHANGED = 75


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


LOCK_STALE_SECONDS = 90


def _lock_path() -> Path:
    return Path(__file__).resolve().parent / "worker.lock"


def _read_lock() -> "tuple[int, float] | None":
    try:
        parts = _lock_path().read_text(encoding="utf-8").split()
        return int(parts[0]), float(parts[1])
    except (OSError, ValueError, IndexError):
        return None


def refresh_worker_lock() -> None:
    """Say the holder is still here."""
    try:
        _lock_path().write_text(f"{os.getpid()} {time.time():.0f}", encoding="utf-8")
    except OSError:
        pass


# A third of the staleness limit, so two beats can be missed before anyone
# concludes this worker is gone.
LOCK_HEARTBEAT_SECONDS = LOCK_STALE_SECONDS / 3.0
_heartbeat_started = False


def start_lock_heartbeat() -> None:
    """Keep the lock warm from a thread, not from the poll loop.

    This was the poll loop's job, and the poll loop is the one place that
    cannot do it. `run_claimed_job` runs the entire analysis inside a single
    iteration, and an analysis takes minutes - so between claiming a job and
    finishing it, nothing wrote the lock. After ninety seconds a working
    worker read as a dead one, and the next worker to start would announce
    "taking over the worker lock" and begin a second analysis on the same 8 GB
    card. That is the exact failure the lock was built to prevent, and it fired
    on every job longer than a minute and a half, which is all of them.

    A thread makes the heartbeat mean what `_claim_sole_worker` already says it
    means: this process is alive. Not "this process is between jobs".

    Daemon, so it never holds the process open - a worker exiting to pick up
    new code (EXIT_CODE_CHANGED) must still exit promptly, and a lock left
    behind by a process that has genuinely gone goes stale on its own.
    """
    global _heartbeat_started
    if _heartbeat_started:
        return
    _heartbeat_started = True

    def beat() -> None:
        while True:
            time.sleep(LOCK_HEARTBEAT_SECONDS)
            refresh_worker_lock()

    threading.Thread(target=beat, name="warrioriq-lock-heartbeat", daemon=True).start()


def _claim_sole_worker() -> bool:
    """Refuse to start if a live worker on this machine already holds the lock.

    The analysis is not safe to run many times over: every worker polls the
    same queue and loads its own copy of the models onto one 8 GB card. Two of
    them race for jobs; forty make the machine useless, which is exactly what a
    re-exec bug produced here once.

    The claim is a heartbeat, not a process id, because a process id alone is
    not enough on this machine. Something outside the project was launching
    worker.py with an unrelated Python that could not import the models; it
    took the lock and would not let go of it. A holder that stops refreshing -
    because it crashed, was killed, or never got as far as working - loses the
    lock after LOCK_STALE_SECONDS and the next capable worker takes it. That
    needs no knowledge of who the other process is, which is the only thing
    that reliably works here.
    """
    held = _read_lock()
    if held:
        pid, beat = held
        age = time.time() - beat
        if pid != os.getpid() and age < LOCK_STALE_SECONDS and _worker_is_alive(pid):
            LOGGER.error(
                "Another WarriorIQ worker holds the lock (process %s, last seen "
                "%.0fs ago). Not starting a second one: they would race for the "
                "same jobs and share one GPU.", pid, age)
            return False
        if pid != os.getpid():
            LOGGER.warning(
                "Taking over the worker lock from process %s (last seen %.0fs ago).",
                pid, age)
    refresh_worker_lock()
    start_lock_heartbeat()
    return True


def _worker_is_alive(pid: int) -> bool:
    """Is this process id a live Python process, rather than a recycled number?"""
    try:
        import subprocess

        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
        ).stdout if os.name == "nt" else ""
        if os.name == "nt":
            return "python" in out.lower()
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def run_remote_worker(worker_id: str, *, once: bool = False) -> int:
    client = RemoteWorkerClient(SETTINGS.worker_remote_url, SETTINGS.worker_token, worker_id)
    running_code = _source_fingerprint()
    LOGGER.info("Worker running analysis code %s", running_code)
    while True:
        # Checked between jobs, never during one. Exiting hands the queue back
        # cleanly and whatever supervises this restarts it on the new code.
        if _code_changed_since(running_code):
            # Exit and let the launcher start the next one. This used to
            # re-exec in place, on the reasoning that re-exec needs no
            # supervisor and cannot leave a hole. That is true on POSIX and
            # false on Windows, where os.execv starts a new process and the
            # old one carries on: every save while editing core/ doubled the
            # workers instead of replacing them. Measured on the machine that
            # actually runs the analysis - 177 restarts logged, 40 live
            # workers, all polling the same queue and sharing one 8 GB GPU.
            #
            # start-worker.bat loops, and the Startup entry launches it, so
            # there is a supervisor on the machine this runs on. Where there is
            # not, the message below says plainly what happened and what to do,
            # which beats a process table filling up silently.
            LOGGER.warning(
                "Analysis code changed on disk (was %s, now %s); exiting so the "
                "launcher can start the new code. If nothing restarts this, run "
                "start-worker.bat.",
                running_code, _source_fingerprint(),
            )
            sys.stdout.flush()
            sys.stderr.flush()
            return EXIT_CODE_CHANGED
        refresh_worker_lock()
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
    if not _claim_sole_worker() and not args.once:
        raise SystemExit(1)
    try:
        raise SystemExit(run_worker(once=args.once))
    finally:
        held = _read_lock()
        if held and held[0] == os.getpid():
            try:
                _lock_path().unlink()
            except OSError:
                pass
