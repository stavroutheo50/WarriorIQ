from __future__ import annotations

import os
import shutil
from pathlib import Path

from core.config import DB_PATH, SETTINGS, UPLOADS
from core.db import connection
from core.legal import launch_readiness


def _database_check() -> dict:
    try:
        with connection() as con:
            con.execute("SELECT 1").fetchone()
        return {"ready": True, "backend": "sqlite"}
    except Exception:
        return {"ready": False, "backend": "sqlite", "reason": "database_unavailable"}


def _storage_check(path: Path) -> dict:
    try:
        path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1024**3
        writable = os.access(path, os.W_OK)
        ready = writable and free_gb >= SETTINGS.minimum_free_storage_gb
        return {
            "ready": ready,
            "backend": "private_local",
            "free_gb": round(free_gb, 2),
            "minimum_free_gb": SETTINGS.minimum_free_storage_gb,
            "reason": None if ready else "storage_capacity_or_permissions",
        }
    except OSError:
        return {"ready": False, "backend": "private_local", "reason": "storage_unavailable"}


def operational_readiness(worker: dict) -> dict:
    database = _database_check()
    storage = _storage_check(UPLOADS)
    worker_public = {
        "ready": bool(worker.get("available")),
        "mode": worker.get("mode"),
        "queued_jobs": int(worker.get("queued_jobs", 0)),
        "running_jobs": int(worker.get("running_jobs", 0)),
    }
    components = {"database": database, "storage": storage, "analysis_worker": worker_public}
    return {
        "status": "ready" if all(item["ready"] for item in components.values()) else "not_ready",
        "ready": all(item["ready"] for item in components.values()),
        "service": "WarriorIQ",
        "components": components,
    }


def release_readiness(worker: dict) -> dict:
    operational = operational_readiness(worker)
    legal = launch_readiness()
    email_ready = SETTINGS.email_provider.lower() == "smtp" and all((
        os.getenv("WARRIORIQ_SMTP_HOST", "").strip(),
        os.getenv("WARRIORIQ_EMAIL_FROM", "").strip(),
        SETTINGS.support_email,
        SETTINGS.privacy_email,
    ))
    scanner_ready = bool(SETTINGS.malware_scan_command) if SETTINGS.malware_scan_required else True
    email_verification_ready = bool(SETTINGS.require_email_verification and email_ready)
    return {
        **operational,
        "release_ready": bool(
            operational["ready"] and legal["ready"] and email_verification_ready and scanner_ready
        ),
        "launch_identity_ready": bool(legal["ready"]),
        "transactional_email_ready": bool(email_ready),
        "email_verification_ready": email_verification_ready,
        "malware_scanner_ready": scanner_ready,
        "payments_enabled": bool(SETTINGS.payments_enabled),
        "database_path_configured": bool(DB_PATH),
    }
