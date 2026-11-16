"""What the admin page shows about the whole service, in one place (owner, 2026-11-15).

"Organise the admin panel and make it have the most information it can." Every
number here is a count read from the database or the data folders at the time
of the request; nothing is estimated or sampled. Counts only, plus the email
address of an account where a row needs one - the admin page is already
restricted to the configured administrator emails, and every visit is logged.

Kept apart from app/main.py so the page template stays presentation only.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.config import DB_PATH, OUTPUTS, UPLOADS
from core.db import connection, init_db

# Walking the report folders reads every file's size. Fine once; not on every
# refresh of the admin page, so the result is kept for a few minutes.
_STORAGE_TTL_SECONDS = 300.0
_storage_cache: dict = {}
_storage_lock = threading.Lock()


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _folder_bytes(root: Path) -> tuple[int, int]:
    """Total size and number of files under `root`, without following links."""
    total, count = 0, 0
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            count += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, count


def storage(now: float | None = None) -> dict:
    """Bytes used by uploads, reports and the database, and the disk's free space."""
    now = time.monotonic() if now is None else now
    with _storage_lock:
        if _storage_cache and now - _storage_cache["at"] < _STORAGE_TTL_SECONDS:
            return _storage_cache["value"]
    uploads_bytes, uploads_files = _folder_bytes(Path(UPLOADS))
    outputs_bytes, outputs_files = _folder_bytes(Path(OUTPUTS))
    database_bytes = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            database_bytes += Path(f"{DB_PATH}{suffix}").stat().st_size
        except OSError:
            pass
    try:
        disk = shutil.disk_usage(Path(UPLOADS) if Path(UPLOADS).exists() else Path(DB_PATH).parent)
        disk_free, disk_total = disk.free, disk.total
    except OSError:
        disk_free = disk_total = None
    value = {"uploads_bytes": uploads_bytes, "uploads_files": uploads_files,
             "outputs_bytes": outputs_bytes, "outputs_files": outputs_files,
             "database_bytes": database_bytes, "disk_free_bytes": disk_free, "disk_total_bytes": disk_total,
             "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with _storage_lock:
        _storage_cache.update({"at": now, "value": value})
    return value


def accounts(now: datetime | None = None) -> dict:
    """How many accounts, how new, and how they are set up."""
    now = now or datetime.now(timezone.utc)
    day, week, month = (_iso(now - timedelta(days=n)) for n in (1, 7, 30))
    init_db()
    with connection() as con:
        row = con.execute(
            """SELECT COUNT(*) AS total,
                      COALESCE(SUM(a.created_at >= ?),0) AS new_day,
                      COALESCE(SUM(a.created_at >= ?),0) AS new_week,
                      COALESCE(SUM(a.created_at >= ?),0) AS new_month,
                      COALESCE(SUM(a.email_verified_at IS NOT NULL AND a.email_verified_at <> ''),0) AS verified,
                      COALESCE(SUM(a.totp_enabled_at IS NOT NULL AND a.totp_enabled_at <> ''),0) AS two_factor,
                      COALESCE(SUM(a.account_status <> 'active'),0) AS suspended,
                      COALESCE(SUM(a.guardian_approval_status = 'pending_guardian'),0) AS guardian_pending,
                      COALESCE(SUM(a.marketing_consent = 1),0) AS marketing,
                      COALESCE(SUM(p.account_type = 'coach'),0) AS coaches
               FROM accounts a LEFT JOIN profiles p ON p.id = a.profile_id""",
            (day, week, month),
        ).fetchone()
        plans = con.execute(
            """SELECT COALESCE(NULLIF(plan_override,''), plan) AS plan, COUNT(*) AS n
               FROM accounts GROUP BY 1 ORDER BY n DESC""").fetchall()
        signed_in = con.execute(
            "SELECT COUNT(DISTINCT account_id) FROM sessions WHERE expires_at > ?", (_iso(now),)).fetchone()[0]
        recent = con.execute(
            """SELECT a.id, a.email, a.created_at, a.account_status,
                      COALESCE(NULLIF(a.plan_override,''), a.plan) AS plan,
                      a.email_verified_at IS NOT NULL AND a.email_verified_at <> '' AS verified,
                      COALESCE(p.account_type,'athlete') AS account_type,
                      (SELECT COUNT(*) FROM fights f WHERE f.profile_id = a.profile_id) AS fights
               FROM accounts a LEFT JOIN profiles p ON p.id = a.profile_id
               ORDER BY a.id DESC LIMIT 12""").fetchall()
    result = dict(row)
    result["athletes"] = result["total"] - result["coaches"]
    result["plans"] = [dict(item) for item in plans]
    result["signed_in"] = int(signed_in)
    result["recent"] = [dict(item) for item in recent]
    return result


def fights(now: datetime | None = None) -> dict:
    """Saved analyses: how many, how recent, which rules, and whether videos are still held."""
    now = now or datetime.now(timezone.utc)
    day, week, month = (_iso(now - timedelta(days=n)) for n in (1, 7, 30))
    init_db()
    with connection() as con:
        row = con.execute(
            """SELECT COUNT(*) AS total,
                      COALESCE(SUM(created_at >= ?),0) AS new_day,
                      COALESCE(SUM(created_at >= ?),0) AS new_week,
                      COALESCE(SUM(created_at >= ?),0) AS new_month,
                      COALESCE(SUM(video_path <> ''),0) AS videos_held,
                      COUNT(DISTINCT profile_id) AS profiles
               FROM fights""",
            (day, week, month),
        ).fetchone()
        rulesets = con.execute(
            "SELECT ruleset, COUNT(*) AS n FROM fights GROUP BY ruleset ORDER BY n DESC").fetchall()
        types = con.execute(
            "SELECT fight_type, COUNT(*) AS n FROM fights GROUP BY fight_type ORDER BY n DESC").fetchall()
        recent = con.execute(
            """SELECT f.job_id, f.created_at, f.ruleset, f.fight_type, f.video_path <> '' AS video_held,
                      a.email
               FROM fights f LEFT JOIN accounts a ON a.profile_id = f.profile_id
               ORDER BY f.id DESC LIMIT 15""").fetchall()
        interest = con.execute(
            "SELECT plan_key, COUNT(DISTINCT account_id) AS n FROM plan_interest GROUP BY plan_key ORDER BY n DESC"
        ).fetchall()
    result = dict(row)
    result["rulesets"] = [dict(item) for item in rulesets]
    result["types"] = [dict(item) for item in types]
    result["recent"] = [dict(item) for item in recent]
    result["plan_interest"] = [dict(item) for item in interest]
    return result


def failures(now: datetime | None = None) -> dict:
    """Analyses that failed in the last 30 days: by reason, and the latest few."""
    now = now or datetime.now(timezone.utc)
    month = _iso(now - timedelta(days=30))
    init_db()
    with connection() as con:
        reasons = con.execute(
            """SELECT reason, COUNT(*) AS n FROM analysis_failures WHERE occurred_at >= ?
               GROUP BY reason ORDER BY n DESC""", (month,)).fetchall()
        recent = con.execute(
            """SELECT f.job_id, f.stage, f.reason, f.detail, f.attempts, f.occurred_at, a.email
               FROM analysis_failures f LEFT JOIN accounts a ON a.id = f.account_id
               ORDER BY f.id DESC LIMIT 10""").fetchall()
    return {"total_month": sum(int(item["n"]) for item in reasons),
            "reasons": [dict(item) for item in reasons], "recent": [dict(item) for item in recent]}


def live_jobs(jobs: list[tuple[str, dict]]) -> dict:
    """The analyses the site is holding right now, by status, and the active ones."""
    counts: dict[str, int] = {}
    active = []
    for job_id, job in jobs:
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status in {"queued", "running"}:
            active.append({"job_id": job_id, "status": status, "stage": job.get("stage") or "",
                           "percent": job.get("percent"), "owner_key": job.get("owner_key") or "",
                           "ruleset": job.get("ruleset") or ""})
    owners = {}
    account_ids = {int(item["owner_key"].split(":", 1)[1]) for item in active
                   if item["owner_key"].startswith("account:") and item["owner_key"].split(":", 1)[1].isdigit()}
    if account_ids:
        init_db()
        with connection() as con:
            marks = ",".join("?" * len(account_ids))
            owners = {int(row["id"]): row["email"] for row in con.execute(
                f"SELECT id, email FROM accounts WHERE id IN ({marks})", tuple(account_ids))}
    for item in active:
        owner = item.pop("owner_key")
        account = owner.split(":", 1)[1] if owner.startswith("account:") else ""
        item["email"] = owners.get(int(account)) if account.isdigit() else ("guest" if owner else "")
    return {"counts": dict(sorted(counts.items())), "active": active}


# --- Email (owner, 2026-11-16) -------------------------------------------------
# A friend's password reset never reached his iCloud inbox. The admin page now
# shows how mail is sent and every account email the site tried to send, so
# "was it sent?" has an answer. "Sent" means the mail server accepted it;
# whether the receiving inbox then delivers it, or files it as junk, the site
# cannot see.

EMAIL_EVENTS = {
    "password_reset_requested": "Password reset link",
    "email_verification_requested": "Email verification",
    "email_verification_resent": "Email verification (resent)",
    "guardian_approval_requested": "Guardian approval request",
    "email_change_requested": "Email change confirmation",
    "coach_invite_created": "Coaching invitation",
    "gym_invite_created": "Gym invitation",
    "admin_plan_granted": "Plan approved",
}
# Sent to someone other than the account the event belongs to.
_OTHER_RECIPIENT = {"guardian_approval_requested": "the guardian named at sign-up",
                    "coach_invite_created": "the person invited", "gym_invite_created": "the person invited"}


def _was_sent(metadata: dict) -> bool:
    # Invitations record {"emailed": bool}; the rest {"email_delivery": "sent" | ...}.
    return metadata.get("email_delivery") == "sent" or metadata.get("emailed") is True


def _domain(address: str) -> str:
    return address.rpartition("@")[2].strip().lower() if "@" in (address or "") else ""


def email_setup() -> dict:
    """How mail is sent, from the website's settings. Never the password."""
    from email.utils import parseaddr

    from core.config import SETTINGS
    from core.notifications import _from_header, email_settings_problem

    username = os.getenv("WARRIORIQ_SMTP_USERNAME", "").strip()
    sender = parseaddr(_from_header(os.getenv("WARRIORIQ_EMAIL_FROM", "").strip(), username))[1]
    host = os.getenv("WARRIORIQ_SMTP_HOST", "").strip()
    warnings = []
    if sender and username and _domain(sender) != _domain(username):
        warnings.append(
            f"The From address ({sender}) is on a different domain from the account that sends the mail "
            f"({username}). iCloud, Yahoo and Outlook often junk or silently drop such mail. Use the same "
            "address for both.")
    if _domain(sender) in _PERSONAL_MAIL:
        warnings.append(
            f"Mail is sent from a personal {_domain(sender)} mailbox. Gmail inboxes take it, but iCloud, Yahoo "
            "and Outlook/Hotmail often junk it or drop it without telling anyone.")
    return {"provider": SETTINGS.email_provider, "host": host, "port": os.getenv("WARRIORIQ_SMTP_PORT", "587"),
            "username": username, "sender": sender, "problem": email_settings_problem(), "warnings": warnings,
            # The setup that reaches every inbox: a mailbox on the site's own
            # domain, sent through the domain's own mail server, which the
            # domain's SPF and DKIM records vouch for.
            "own_domain": bool(sender) and _domain(sender) == SETTINGS.public_email_domain
                          and _domain(username or sender) == SETTINGS.public_email_domain,
            "domain": SETTINGS.public_email_domain}


_PERSONAL_MAIL = {"gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "icloud.com",
                  "me.com", "yahoo.com"}


def email_log(limit: int = 60, now: datetime | None = None) -> dict:
    """Account emails the site tried to send, newest first, and totals by inbox provider."""
    import json

    init_db()
    now = now or datetime.now(timezone.utc)
    kinds = tuple(EMAIL_EVENTS)
    marks = ",".join("?" * len(kinds))
    with connection() as con:
        rows = con.execute(
            f"""SELECT e.event_type, e.occurred_at, e.metadata_json, a.email
                FROM security_events e LEFT JOIN accounts a ON a.id = e.account_id
                WHERE e.event_type IN ({marks}) ORDER BY e.id DESC LIMIT ?""",
            (*kinds, max(1, min(int(limit), 300))),
        ).fetchall()
        month = con.execute(
            f"""SELECT e.metadata_json, a.email FROM security_events e LEFT JOIN accounts a ON a.id = e.account_id
                WHERE e.event_type IN ({marks}) AND e.event_type NOT IN ({",".join("?" * len(_OTHER_RECIPIENT))})
                AND e.occurred_at >= ?""",
            (*kinds, *_OTHER_RECIPIENT, _iso(now - timedelta(days=30))),
        ).fetchall()
    recent = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        sent = _was_sent(metadata)
        recent.append({
            "what": EMAIL_EVENTS[row["event_type"]], "at": row["occurred_at"],
            "to": _OTHER_RECIPIENT.get(row["event_type"]) or row["email"] or "deleted account",
            "sent": sent, "reason": "" if sent else (
                metadata.get("reason") or "Email is switched off or failed; the website log has the reason."),
        })
    providers: dict[str, dict] = {}
    for row in month:
        if not row["email"]:
            continue
        line = providers.setdefault(_domain(row["email"]), {"sent": 0, "not_sent": 0})
        line["sent" if _was_sent(json.loads(row["metadata_json"] or "{}")) else "not_sent"] += 1
    return {"recent": recent, "providers": dict(sorted(providers.items(), key=lambda kv: -sum(kv[1].values())))}


# --- One person, for support (owner, 2026-11-16) -------------------------------

def person(query: str, limit: int = 5) -> list[dict]:
    """Everything the admin needs to help one person, for accounts whose email matches."""
    import json

    from core.payments import effective_plan_key, override_in_force

    query = (query or "").strip().lower()
    if not query:
        return []
    init_db()
    now = _iso(datetime.now(timezone.utc))
    with connection() as con:
        accounts_found = con.execute(
            """SELECT * FROM accounts WHERE lower(email) LIKE ? ORDER BY (lower(email)=?) DESC, id DESC LIMIT ?""",
            (f"%{query}%", query, max(1, min(int(limit), 10))),
        ).fetchall()
        cards = []
        for account in accounts_found:
            account = dict(account)
            account_id = int(account["id"])
            sessions = con.execute("SELECT COUNT(*) FROM sessions WHERE account_id=? AND expires_at>?",
                                   (account_id, now)).fetchone()[0]
            fights_saved = con.execute("SELECT COUNT(*) FROM fights WHERE profile_id=?",
                                       (int(account["profile_id"]),)).fetchone()[0]
            last_login = con.execute(
                "SELECT occurred_at FROM security_events WHERE account_id=? AND event_type='login_succeeded' "
                "ORDER BY id DESC LIMIT 1", (account_id,)).fetchone()
            events = con.execute(
                "SELECT event_type, severity, occurred_at, metadata_json FROM security_events WHERE account_id=? "
                "ORDER BY id DESC LIMIT 15", (account_id,)).fetchall()
            cards.append({
                "id": account_id, "email": account["email"], "status": account.get("account_status") or "active",
                # The plan the person actually has now, an ended grant ignored.
                "plan": effective_plan_key(account.get("plan"), account.get("plan_override"), account.get("email"),
                                           account.get("plan_override_until")),
                "given_plan": account.get("plan_override") if override_in_force(
                    account.get("plan_override"), account.get("plan_override_until")) else None,
                "given_until": account.get("plan_override_until"),
                "created_at": account.get("created_at") or "",
                "email_verified_at": account.get("email_verified_at"),
                "guardian_status": account.get("guardian_approval_status") or "not_applicable",
                "signed_in_devices": int(sessions), "fights": int(fights_saved),
                "last_sign_in": last_login[0] if last_login else None,
                "two_factor": bool(account.get("totp_enabled_at")),
                "events": [{"type": event["event_type"].replace("_", " "), "severity": event["severity"],
                            "at": event["occurred_at"],
                            "email": _email_note(event["event_type"], json.loads(event["metadata_json"] or "{}"))}
                           for event in events],
            })
    return cards


def _email_note(event_type: str, metadata: dict) -> str:
    if event_type not in EMAIL_EVENTS:
        return ""
    return "email sent" if _was_sent(metadata) else "email NOT sent"
