from __future__ import annotations

import argparse
import logging
import socket
import time
import uuid

from app.state import (
    AnalysisRunLost, claim_next_job, get_job, record_worker_heartbeat,
    update_job, update_job_for_worker,
)
from core.config import SETTINGS
from core.db import release_analysis
from core.types import AnalysisRequest


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


def run_worker(*, once: bool = False) -> int:
    worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WarriorIQ durable GPU analysis worker")
    parser.add_argument("--once", action="store_true", help="Claim at most one queued job and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(run_worker(once=args.once))
