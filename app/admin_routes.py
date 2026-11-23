"""The admin control room: /admin and its tools, for configured administrators only.

Moved out of app/main.py unchanged (2026-11-23). Names shared with the rest of
the application are read from app.main when a request arrives (``m.name``), not
copied in at import, so anything that replaces one of them on app.main - the
tests do this - reaches this code too.

The routes are registered straight onto app.main's application, when app.main
imports this module at the spot they used to occupy. Not an APIRouter: newer
FastAPI keeps an included router as one nested entry, which code and tests that
walk app.routes (the CSRF guard test does) would no longer see into. This way
the route table is exactly what it was, in the same order, on any version.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import app.main as m

app = m.app


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, q: str = ""):
    if not m._is_admin(request):
        raise HTTPException(404)
    account = m._account(request)
    m.record_security_event("admin_area_viewed", account_id=int(account["id"]), resource_type="admin")
    return m.templates.TemplateResponse(
        request=request, name="admin.html",
        context={
            "request": request, "query": q[:200], "users": m.list_accounts(q),
            "reports": m.list_moderation_reports(), "security_events": m.list_security_events(),
            "traffic": m.page_view_summary(30),
            "email_problem": m.email_settings_problem(),
            "email_tests": _email_tests(),
            # Set while a worker is being refused for running other code.
            "worker_version_flag": m.worker_alerts.version_mismatch_flag(m.OUTPUTS),
            "release": m.release_version(),
            # The whole service at a glance (app/admin_overview.py): counts
            # read from the database and data folders on this request.
            "overview": {
                "accounts": m.admin_overview.accounts(),
                "fights": m.admin_overview.fights(),
                "failures": m.admin_overview.failures(),
                "storage": m.admin_overview.storage(),
                "live": m.admin_overview.live_jobs(m.list_jobs()),
                # Where new accounts stop between sign-up and a report (2026-11-22).
                "onboarding": m.admin_overview.onboarding(m.list_jobs()),
                "worker": m.worker_status(),
                "worker_refusal": m.recent_worker_refusal(),
                "payments_enabled": m.SETTINGS.payments_enabled,
                "worker_mode": m.SETTINGS.analysis_worker_mode,
            },
            # Backups and server checks without a shell (app/site_backup.py).
            "backups": {
                "enabled": m.site_backup.enabled(),
                "keep": m.site_backup.keep(),
                "copies": m.site_backup.backups(),
                "stale": _backup_is_stale(),
            },
            # How often the analysis was right, from people's answers (app/accuracy.py).
            "accuracy": _accuracy_summary(),
            "server_checks": m.site_backup.server_checks(),
            # How mail is sent and every account email tried, and one person's
            # whole picture for a support question (app/admin_overview.py).
            "email": {"setup": m.admin_overview.email_setup(), "log": m.admin_overview.email_log()},
            "people": m.admin_overview.person(q[:200]) if q.strip() else [],
            "leftovers": m.site_backup.leftover_copies(),
            # The QA pass's test fighters still on the admin's own roster (app/qa_fighters.py).
            "qa_fighters": m.qa_fighters.on_roster(int(m._account(request)["profile_id"])),
            "tool_result": _ADMIN_TOOL_RESULTS.get(request.query_params.get("tool", ""), ""),
            # Plan requests from the pricing page, for a yes with an end date
            # or a no (owner, 2026-11-16).
            "plan_requests": m.pending_plan_requests(),
            "plan_choices": [(key, plan["label"]) for key, plan in m.PLANS.items() if key != "free"],
            "plan_durations": _PLAN_DURATIONS,
        },
    )


# What the backup and clean-up buttons report, by a fixed code in the address
# (never free text from the address, which anyone could put in a link).
_ADMIN_TOOL_RESULTS = {
    "backup_done": "Backup made and checked.",
    "backup_recent": "A backup was made less than 5 minutes ago, so no new one was made.",
    "backup_busy": "A backup is already being made. Reload in a minute.",
    "backup_failed": "The backup failed. The security log has the reason.",
    "backup_no_database": "No database was found where the website expects it, so nothing was backed up.",
    "leftovers_deleted": "The selected leftover copies were deleted.",
    "leftovers_none": "Nothing was deleted: tick the confirmation and at least one copy.",
    "qa_archived": "The QA test fighters were archived. Their fights were kept; adding a fighter with the same name brings one back.",
    "qa_none": "Nothing was archived: tick the confirmation first.",
    "plan_granted": "Plan given. The person was emailed if email is working; the Email section shows whether it was sent.",
    "plan_declined": "Request declined. The person's plan was not changed.",
    "plan_ended": "The plan was taken away. The person is back on the plan they pay for, or Starter.",
    "plan_invalid": "Nothing changed: choose a plan and how long it lasts.",
}


def _accuracy_summary() -> dict | None:
    def anchors_for(job_id: str, job: dict) -> list[dict]:
        return m._identity_anchors(m._crossing_questions(job_id, job), dict(job.get("identity_answers") or {}))
    try:
        return m.accuracy_answers.summary(m.list_jobs(), anchors_for)
    except Exception:                                               # noqa: BLE001
        # A figure that cannot be counted is left out; the admin page still opens.
        m.LOGGER.exception("admin_accuracy_failed")
        return None


def _backup_is_stale() -> bool:
    """True when daily backups are on but the newest is over two days old, or missing."""
    if not m.site_backup.enabled():
        return False
    last = m.site_backup.latest()
    return last is None or datetime.now(timezone.utc) - last["created_at"] > timedelta(hours=48)


@app.post("/admin/backup", dependencies=[Depends(m.require_csrf)])
def admin_backup_now(request: Request):
    if not m._is_admin(request):
        raise HTTPException(404)
    actor = m._account(request)
    result = m.site_backup.make_backup(scheduled=False)
    if result["outcome"] in {"done", "failed"}:
        m.record_security_event(
            "database_backup_created" if result["outcome"] == "done" else "database_backup_failed",
            account_id=int(actor["id"]), severity="info" if result["outcome"] == "done" else "warning",
            resource_type="backup", resource_id=result.get("name", ""),
            metadata={"scheduled": False, **{key: result[key] for key in ("bytes", "error", "removed") if key in result}},
        )
    return RedirectResponse(f"/admin?tool=backup_{result['outcome']}#adminTools", status_code=303)


@app.post("/admin/leftovers/delete", dependencies=[Depends(m.require_csrf)])
async def admin_delete_leftovers(request: Request):
    if not m._is_admin(request):
        raise HTTPException(404)
    form = await request.form()
    names = [str(value) for value in form.getlist("name")][:500]
    if form.get("confirm") != "yes" or not names:
        return RedirectResponse("/admin?tool=leftovers_none#adminTools", status_code=303)
    actor = m._account(request)
    for name in m.site_backup.delete_leftover_copies(names):
        m.record_security_event("orphan_web_video_deleted", account_id=int(actor["id"]), resource_type="fight",
                              resource_id=name[: -len("_web.mp4")], metadata={"tool": "admin_page"})
    return RedirectResponse("/admin?tool=leftovers_deleted#adminTools", status_code=303)


@app.post("/admin/qa-fighters/archive", dependencies=[Depends(m.require_csrf)])
async def admin_archive_qa_fighters(request: Request):
    """Archive the QA pass's test fighters on the admin's own roster. Reversible; no fight is deleted."""
    if not m._is_admin(request):
        raise HTTPException(404)
    form = await request.form()
    if form.get("confirm") != "yes":
        return RedirectResponse("/admin?tool=qa_none#adminTools", status_code=303)
    actor = m._account(request)
    m.qa_fighters.archive_all(int(actor["profile_id"]), int(actor["id"]))
    return RedirectResponse("/admin?tool=qa_archived#adminTools", status_code=303)


@app.post("/admin/email/test", dependencies=[Depends(m.require_csrf)])
def admin_send_test_email(request: Request, to: str = Form("")):
    """Send a test message and show exactly what the mail server said.

    Password resets, verification and guardian emails all go through the same
    sender, so this is the one place an administrator can see why they fail.
    It can go to any address, not only the admin's own: mail was reaching
    Gmail but not iCloud, Yahoo or Outlook (owner, 2026-11-16), and the only
    way to check one of those is to send to an address there.

    The result is kept as a security event and the page shows it from there,
    so nothing in the page's address is ever shown back.
    """
    if not m._is_admin(request):
        raise HTTPException(404)
    m._enforce_rate_limit(request, "admin-email-test", 20, 3600)
    actor = m._account(request)
    recipient = m.normalize_email(to) if to.strip() else actor["email"]
    if not m.valid_email(recipient):
        outcome = "Not sent: that is not an email address."
    else:
        try:
            m.deliver_email(recipient, "WarriorIQ test email",
                          "This is a test from the WarriorIQ admin page. Email is working.")
            outcome = (f"Sent to {recipient}. The mail server accepted it. If it has not arrived within a few "
                       "minutes, look in Junk; if it is not there either, that inbox provider dropped it.")
        except m.EmailNotSent as reason:
            outcome = f"Not sent to {recipient}: {reason}"
    m.record_security_event("admin_email_test", account_id=int(actor["id"]), resource_type="email",
                          metadata={"outcome": outcome[:400], "sent": outcome.startswith("Sent")})
    return RedirectResponse("/admin#adminEmail", status_code=303)


def _email_tests(limit: int = 6) -> list[dict]:
    """The latest test emails and what happened, newest first."""
    tests = [event for event in m.list_security_events(300) if event["event_type"] == "admin_email_test"]
    return [{"at": event["occurred_at"], "outcome": event["metadata"].get("outcome", ""),
             "sent": bool(event["metadata"].get("sent", str(event["metadata"].get("outcome", "")).startswith("Sent")))}
            for event in tests[:limit]]


@app.post("/admin/accounts/{account_id}/status", dependencies=[Depends(m.require_csrf)])
def admin_account_status(request: Request, account_id: int, status: str = Form(...)):
    if not m._is_admin(request):
        raise HTTPException(404)
    actor = m._account(request)
    if int(actor["id"]) == int(account_id) and status != "active":
        raise HTTPException(400, "An administrator cannot suspend their current account.")
    if not m.set_account_status(account_id, status):
        raise HTTPException(404)
    m.record_security_event(
        "admin_account_status_changed", account_id=int(actor["id"]), severity="warning",
        resource_type="account", resource_id=str(account_id), metadata={"status": status},
    )
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/accounts/{account_id}/test-account", dependencies=[Depends(m.require_csrf)])
def admin_mark_test_account(request: Request, account_id: int, test: str = Form("1")):
    """Leave an account out of "Where new accounts stop", or count it again (2026-11-22).

    The owner's own trial sign-ups were a sixth of the report. Nothing about
    the account changes; the latest mark wins (app/admin_overview.py).
    """
    if not m._is_admin(request):
        raise HTTPException(404)
    if m.get_account(int(account_id)) is None:
        raise HTTPException(404)
    actor = m._account(request)
    m.record_security_event("admin_marked_test_account", account_id=int(actor["id"]), resource_type="account",
                          resource_id=str(int(account_id)), metadata={"test": test == "1"})
    return RedirectResponse("/admin#adminOnboarding", status_code=303)


# How long a plan given from the admin page lasts; "" is no end date.
_PLAN_DURATIONS = [("7", "1 week"), ("14", "2 weeks"), ("30", "1 month"), ("90", "3 months"),
                   ("180", "6 months"), ("365", "1 year"), ("", "No end date")]


@app.post("/admin/plans/grant", dependencies=[Depends(m.require_csrf)])
def admin_grant_plan(request: Request, account_id: int = Form(...), plan_key: str = Form(...),
                     days: str = Form("30")):
    """Give a person a plan for a set time, from their request or their support card.

    Billing is not touched: this is the same local entitlement as
    plan_override, now with an end date, after which the person is back on
    the plan they pay for (or Starter) without anyone having to remember.
    """
    if not m._is_admin(request):
        raise HTTPException(404)
    if days not in {value for value, _ in _PLAN_DURATIONS} or plan_key not in m.PLANS or plan_key == "free":
        return RedirectResponse("/admin?tool=plan_invalid#adminPlans", status_code=303)
    target = m.get_account(int(account_id))
    if not target:
        raise HTTPException(404)
    until = m.grant_plan(int(account_id), plan_key, int(days) if days else None)
    label = m.PLANS[plan_key]["label"]
    ends = f"until {datetime.fromisoformat(until).strftime('%d %B %Y')}" if until else "with no end date"
    emailed = m.send_transactional_email(
        target["email"], f"Your WarriorIQ plan: {label}",
        f"Your request was approved. Your account now has the {label} plan {ends}.\n\n"
        "Nothing is charged for it. When it ends, your account goes back to the plan it had before.\n\n"
        f"Open WarriorIQ: {m._public_base(request)}/camp")
    actor = m._account(request)
    m.record_security_event(
        "admin_plan_granted", account_id=int(target["id"]), severity="warning", resource_type="account",
        resource_id=str(account_id),
        metadata={"plan": plan_key, "until": until, "by": int(actor["id"]),
                  "email_delivery": "sent" if emailed else "failed"})
    return RedirectResponse("/admin?tool=plan_granted#adminPlans", status_code=303)


@app.post("/admin/plans/decline", dependencies=[Depends(m.require_csrf)])
def admin_decline_plan(request: Request, account_id: int = Form(...), plan_key: str = Form(...)):
    if not m._is_admin(request):
        raise HTTPException(404)
    if m.decline_plan_request(int(account_id), plan_key):
        m.record_security_event("admin_plan_request_declined", account_id=int(account_id), resource_type="account",
                              resource_id=str(account_id),
                              metadata={"plan": plan_key, "by": int(m._account(request)["id"])})
    return RedirectResponse("/admin?tool=plan_declined#adminPlans", status_code=303)


@app.post("/admin/plans/end", dependencies=[Depends(m.require_csrf)])
def admin_end_plan(request: Request, account_id: int = Form(...)):
    """Take away a plan given from the admin page, now."""
    if not m._is_admin(request):
        raise HTTPException(404)
    if not m.set_plan_override(int(account_id), None):
        raise HTTPException(404)
    m.record_security_event("admin_plan_ended", account_id=int(account_id), severity="warning",
                          resource_type="account", resource_id=str(account_id),
                          metadata={"by": int(m._account(request)["id"])})
    return RedirectResponse("/admin?tool=plan_ended#adminPlans", status_code=303)


@app.post("/admin/reports/{report_id}/resolve", dependencies=[Depends(m.require_csrf)])
def admin_resolve_report(request: Request, report_id: int):
    if not m._is_admin(request):
        raise HTTPException(404)
    if not m.resolve_moderation_report(report_id):
        raise HTTPException(404)
    actor = m._account(request)
    m.record_security_event(
        "admin_moderation_report_resolved", account_id=int(actor["id"]),
        resource_type="moderation_report", resource_id=str(report_id),
    )
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/videos/delete", dependencies=[Depends(m.require_csrf)])
def admin_delete_prohibited_video(request: Request, job_id: str = Form(...)):
    if not m._is_admin(request):
        raise HTTPException(404)
    fight = m.delete_fight(job_id.strip())
    if not fight:
        raise HTTPException(404, "No saved analysis matches that ID.")
    m._remove_fight_files(fight)
    actor = m._account(request)
    m.record_security_event(
        "admin_prohibited_content_deleted", account_id=int(actor["id"]), severity="warning",
        resource_type="fight", resource_id=job_id.strip(),
    )
    return RedirectResponse("/admin", status_code=303)


@app.get("/admin/openapi.json", include_in_schema=False)
def admin_openapi(request: Request):
    """The API schema, for a signed-in administrator only."""
    if not m._is_admin(request):
        raise HTTPException(404)
    return JSONResponse(app.openapi())

