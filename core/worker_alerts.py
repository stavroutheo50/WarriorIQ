"""Tell the owner when fights are waiting and the analysis PC is not there.

Analysis runs on one PC. When it is off, asleep past waking, offline or
crashed, the website keeps uploaded fights queued and nothing else happens:
customers wait and nobody knows. One measured fight waited 17 hours.

Checked when a customer's progress page polls a fight that is still queued -
the moment someone is actually waiting - so it needs no scheduler on the host.
The administrators (WARRIORIQ_ADMIN_EMAILS) get one email when a fight has
waited longer than WARRIORIQ_WORKER_ALERT_SECONDS with no worker connected,
at most one every REPEAT_SECONDS while it stays down, and one more when the
worker connects again. 0 switches the alert off.

An alert is "open" while its marker file exists, so the worker's frequent
claim calls cost one existence check and nothing else.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger("warrioriq.worker_alerts")

REPEAT_SECONDS = 3 * 60 * 60
MARKER = "worker-offline-alert.json"
_lock = threading.Lock()


def _send_in_background(send: Callable[[str, str, str], bool], recipients, subject: str, body: str) -> None:
    def run() -> None:
        for recipient in recipients:
            try:
                send(recipient, subject, body)
            except Exception as exc:                                # noqa: BLE001
                LOGGER.error("worker_alert_email_failed error=%s", type(exc).__name__)

    threading.Thread(target=run, name="wiq-worker-alert", daemon=True).start()


def check_waiting_fight(job: dict, *, worker_online: bool, outputs: Path, recipients,
                        threshold_seconds: float, send, now: float | None = None) -> bool:
    """Alert if this queued fight has waited too long with no worker. True when an email was sent."""
    if threshold_seconds <= 0 or worker_online or not recipients or job.get("status") != "queued":
        return False
    queued_at = float(job.get("wake_requested_at_epoch") or 0.0)
    now = time.time() if now is None else now
    if not queued_at or now - queued_at < threshold_seconds:
        return False
    marker = outputs / MARKER
    with _lock:
        try:
            last = float(json.loads(marker.read_text(encoding="utf-8"))["sent_at_epoch"])
        except (OSError, ValueError, KeyError, TypeError):
            last = None                                             # no alert open
        if last is not None and now - last < REPEAT_SECONDS:
            return False
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"sent_at_epoch": now}), encoding="utf-8")
    minutes = int((now - queued_at) // 60)
    LOGGER.warning("worker_offline_alert waited_minutes=%s", minutes)
    _send_in_background(
        send, recipients, "WarriorIQ: fights are waiting - the analysis PC is not connected",
        f"A fight has been waiting {minutes} minutes and no analysis worker is connected.\n\n"
        "Check that the analysis PC is switched on, awake and online, and that the WarriorIQ "
        "worker is running on it. Waiting fights are kept and will be analysed as soon as it "
        "connects.\n\nYou will get one more email when it is back.")
    return True


def note_worker_connected(*, outputs: Path, recipients, send) -> bool:
    """After an alert, say once that the worker is back. True when an email was sent."""
    marker = outputs / MARKER
    if not marker.exists():
        return False
    with _lock:
        if not marker.exists():
            return False
        marker.unlink(missing_ok=True)
    LOGGER.info("worker_back_online")
    if recipients:
        _send_in_background(send, recipients, "WarriorIQ: the analysis PC is back",
                            "The analysis worker is connected again. Waiting fights are being analysed.")
        return True
    return False
