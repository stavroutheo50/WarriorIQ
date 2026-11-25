"""Settings: the privacy and billing pages, and every change made from them.

Moved out of app/main.py unchanged (2026-11-23). Names shared with the rest of
the application are read from app.main when a request arrives (``m.name``), not
copied in at import, so anything that replaces one of them on app.main - the
tests do this - reaches this code too.

The routes are registered straight onto app.main's application when app.main
runs this module, at the spot they used to occupy, so the route table is
exactly what it was (see app/admin_routes.py for why not an APIRouter).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

import app.main as m

app = m.app


@app.get("/settings")
def settings_root(request: Request):
    return RedirectResponse("/settings/privacy", status_code=303)


@app.get("/settings/{section}", response_class=HTMLResponse)
def settings_page(request: Request, section: str, notice: str = "", error: str = "", area: str = ""):
    account = m._account(request)
    if not account:
        return RedirectResponse(f"/login?next=/settings/{section}", status_code=303)
    if section not in {"privacy", "billing"}:
        raise HTTPException(404)
    return m._settings_response(request, account, section, notice=notice, error=error, area=area)


@app.post("/settings/marketing", dependencies=[Depends(m.require_csrf)])
def save_marketing_preference(request: Request, enabled: bool = Form(False)):
    account = m._account(request)
    if not account:
        raise HTTPException(403)
    m.update_marketing_consent(int(account["id"]), bool(enabled))
    m.record_legal_acceptance(
        "marketing_consent", m.SETTINGS.policy_version, profile_id=int(account["profile_id"]),
        metadata={"enabled": bool(enabled), "source": "privacy_settings"},
        current_status="accepted" if enabled else "withdrawn",
    )
    return RedirectResponse("/settings/privacy?notice=Marketing+preference+saved", status_code=303)


@app.post("/settings/quick-check-training", dependencies=[Depends(m.require_csrf)])
def save_quick_check_training(request: Request, enabled: bool = Form(False)):
    """Whether this account's quick-check answers may train strike counting (2026-11-25).

    Off also takes out answers already given (app/quick_check_training.py).
    """
    account = m._account(request)
    if not account:
        raise HTTPException(403)
    from app import quick_check_training

    quick_check_training.set_choice(int(account["profile_id"]), bool(enabled), m.SETTINGS.policy_version)
    return RedirectResponse("/settings/privacy?notice=Quick-check+choice+saved", status_code=303)


@app.post("/settings/sessions/revoke", dependencies=[Depends(m.require_csrf)])
def revoke_other_sessions(request: Request):
    account = m._account(request)
    if not account:
        raise HTTPException(403)
    current = request.cookies.get(m.SESSION_COOKIE)
    m.revoke_account_sessions(int(account["id"]), m.token_digest(current) if current else None)
    m.record_security_event("other_sessions_revoked", account_id=int(account["id"]))
    return RedirectResponse("/settings/privacy?notice=Other+sessions+signed+out", status_code=303)


@app.post("/settings/sessions/{session_id}/revoke", dependencies=[Depends(m.require_csrf)])
def revoke_one_session(request: Request, session_id: int):
    """Sign one other device out. Scoped to this account in the query itself."""
    account = m._account(request)
    if not account:
        raise HTTPException(403)
    current = request.cookies.get(m.SESSION_COOKIE)
    mine = {row["id"]: row for row in m.list_account_sessions(int(account["id"]), m.token_digest(current) if current else None)}
    if session_id in mine and mine[session_id]["current"]:
        return _settings_redirect(area="devices", error="That is this browser. Use Sign out in the menu.")
    if not m.revoke_session_by_id(int(account["id"]), session_id):
        return _settings_redirect(area="devices", error="That device was already signed out.")
    m.record_security_event("session_revoked", account_id=int(account["id"]))
    return _settings_redirect(area="devices", notice="That device was signed out.")


def _settings_redirect(notice: str = "", error: str = "", area: str = "sign-in") -> RedirectResponse:
    """Back to Settings, at the card the message is about, with the message beside it."""
    area = area if area in m.SETTINGS_AREAS else "sign-in"
    query = urlencode({key: value for key, value in (("notice", notice), ("error", error),
                                                       ("area", area if notice or error else "")) if value})
    anchor = {"sign-in": "security", "devices": "devices",
              "export": "account-overview", "delete": "delete-account"}[area]
    return RedirectResponse(f"/settings/privacy{'?' + query if query else ''}#{anchor}", status_code=303)


@app.post("/settings/password", dependencies=[Depends(m.require_csrf)])
def change_password(request: Request, current_password: str = Form(""), new_password: str = Form(""),
                    confirm_password: str = Form("")):
    """Change the password while signed in (QA, 2026-10-11: only
    /forgot-password existed). Needs the current password; every other session
    is signed out and this browser gets a fresh one."""
    account = m._account(request)
    if not account:
        raise HTTPException(403)
    m._enforce_rate_limit(request, "password-change", 10, 3600)
    if not m._current_password_ok(int(account["id"]), current_password):
        m.record_security_event("password_change_refused", account_id=int(account["id"]))
        return _settings_redirect(error="Your current password was not right. Nothing was changed.")
    if not m.valid_password(new_password):
        return _settings_redirect(error="The new password needs between 10 and 1,024 characters.")
    if m.new_password_problem(new_password):
        return _settings_redirect(error=m.new_password_problem(new_password) + " Nothing was changed.")
    if new_password != confirm_password:
        return _settings_redirect(error="The two new passwords do not match. Nothing was changed.")
    m.update_password_hash(int(account["id"]), m.hash_password(new_password))   # ends every session
    m.record_security_event("password_changed", account_id=int(account["id"]))
    response = _settings_redirect(notice="Password changed. Every other device has been signed out.")
    response.set_cookie(m.SESSION_COOKIE, m.issue_session(int(account["id"]), device_label=m._device_label(request)),
                        max_age=60 * 60 * 24 * 30, httponly=True, samesite="lax", secure=m._request_is_secure(request))
    return response


@app.post("/settings/email", dependencies=[Depends(m.require_csrf)])
def request_email_change(request: Request, current_password: str = Form(""), new_email: str = Form(""),
                         reauth_password: str = Form("")):
    """Start an email change: nothing changes until the link sent to the new
    address is opened. The current address is told a change was asked for.

    reauth_password since 2026-11-16 (a name Chrome does not autofill);
    current_password is the earlier name."""
    current_password = reauth_password or current_password
    account = m._account(request)
    if not account:
        raise HTTPException(403)
    m._enforce_rate_limit(request, "email-change", 5, 3600)
    if not m._current_password_ok(int(account["id"]), current_password):
        m.record_security_event("email_change_refused", account_id=int(account["id"]))
        return _settings_redirect(error="Your current password was not right. Nothing was changed.")
    email = m.normalize_email(new_email)
    if not m.valid_email(email):
        return _settings_redirect(error="Enter a valid new email address.")
    if email == m.normalize_email(account.get("email") or ""):
        return _settings_redirect(error="That is already your email address.")
    if m.get_account_by_email(email):
        return _settings_redirect(error="That email address cannot be used. Choose another one.")
    token = m.session_token()
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    m.save_email_change_token(int(account["id"]), email, m.token_digest(token), expires)
    confirm_url = f"{m._public_base(request)}/settings/email/confirm/{token}"
    delivered = False
    try:
        delivered = m.send_transactional_email(
            email, "Confirm your new WarriorIQ email",
            f"Open this link within 24 hours to make this your WarriorIQ sign-in email:\n\n{confirm_url}\n\n"
            "If you did not ask for this, ignore this message and nothing changes.")
        m.send_transactional_email(
            account["email"], "Your WarriorIQ email change",
            "Someone signed in to your WarriorIQ account asked to change its email address. It changes only "
            "when the link sent to the new address is opened. If this was not you, change your password in "
            "Settings now.")
    except Exception:                                                # noqa: BLE001
        m.LOGGER.warning("email_change_delivery_failed account=%s", account["id"])
    m.record_security_event("email_change_requested", account_id=int(account["id"]),
                          metadata={"email_delivery": "sent" if delivered else "unavailable"})
    if not delivered:
        return _settings_redirect(error="The confirmation email could not be sent right now. Try again later.")
    return _settings_redirect(notice=f"Check {email}: open the link in it to finish the change.")


@app.get("/settings/email/confirm/{token}")
def confirm_email_change(request: Request, token: str):
    changed = m.consume_email_change_token(m.token_digest(token))
    if changed is None:
        raise HTTPException(410, "This email-change link is invalid, used or expired.")
    account_id, old_email, new_email = changed
    m.record_security_event("email_changed", account_id=account_id)
    try:
        m.send_transactional_email(
            old_email, "Your WarriorIQ email was changed",
            "The sign-in email for your WarriorIQ account was changed. If this was not you, reset your "
            "password from the sign-in page and contact support.")
    except Exception:                                                # noqa: BLE001
        m.LOGGER.warning("email_change_notice_failed account=%s", account_id)
    current = m._account(request)
    if current and int(current["id"]) == account_id:
        return _settings_redirect(notice=f"Your email is now {new_email}.")
    return RedirectResponse("/login?verified=1", status_code=303)


@app.post("/settings/videos/{job_id}/delete", dependencies=[Depends(m.require_csrf)])
def delete_original_video(request: Request, job_id: str):
    account = m._account(request)
    fight = m.get_fight(job_id)
    if not account or not fight or int(fight["profile_id"]) != int(account["profile_id"]):
        raise HTTPException(404)
    video = Path(fight.get("video_path") or "missing").resolve()
    if video.parent == m.UPLOADS.resolve():
        video.unlink(missing_ok=True)
        m.remove_derivative(video)
    m.training_consent.forget(job_id)
    m.mark_fight_video_deleted(job_id, int(account["profile_id"]))
    m.record_security_event(
        "fight_video_deleted", account_id=int(account["id"]), resource_type="fight", resource_id=job_id,
    )
    return RedirectResponse("/settings/privacy?notice=Original+video+deleted", status_code=303)


@app.post("/settings/billing/cancel", dependencies=[Depends(m.require_csrf)])
def cancel_subscription(request: Request):
    session_account = m._account(request)
    if not session_account:
        raise HTTPException(403)
    account = m.get_account(int(session_account["id"])) or session_account
    subscription_id = account.get("stripe_subscription_id")
    if not subscription_id:
        raise HTTPException(400, "No connected paid subscription can be cancelled.")
    try:
        provider = m.cancel_subscription_at_period_end(str(subscription_id))
    except Exception as exc:
        raise HTTPException(503, f"Cancellation was not confirmed by the payment provider: {exc}")
    if not provider.get("cancel_at_period_end"):
        raise HTTPException(503, "The payment provider did not confirm cancellation.")
    action = m.record_subscription_action(
        int(account["id"]), "cancel", "scheduled",
        effective_at=account.get("subscription_period_end"), provider_reference=str(subscription_id),
        metadata={"provider_status": provider.get("status")},
    )
    m._queue_transactional_notice(
        int(account["id"]), "subscription_cancellation_confirmation", account["email"],
        "Your WarriorIQ subscription cancellation",
        f"Cancellation was scheduled on {action['requested_at']}. Access ends at {account.get('subscription_period_end') or 'the confirmed billing-period end'}. No further renewal should be charged after that date.",
        {"requested_at": action["requested_at"], "access_ends": account.get("subscription_period_end")},
    )
    return RedirectResponse("/settings/billing?notice=Cancellation+scheduled", status_code=303)


@app.post("/settings/billing/withdraw", dependencies=[Depends(m.require_csrf)])
def request_contract_withdrawal(request: Request, confirm: bool = Form(False)):
    account = m._account(request)
    if not account or not confirm:
        return m._form_refusal("Confirm the withdrawal request.",
                             "/settings/billing", "/settings/billing")
    full_account = m.get_account(int(account["id"])) or account
    if not full_account.get("stripe_subscription_id"):
        return m._form_refusal("No connected purchase is available for withdrawal review.",
                             "/settings/billing", "/settings/billing")
    action = m.record_subscription_action(
        int(account["id"]), "eu_withdrawal", "pending_review",
        provider_reference=full_account.get("stripe_subscription_id"),
        metadata={"eligibility_not_determined": True, "policy_version": m.SETTINGS.policy_version},
    )
    m._queue_transactional_notice(
        int(account["id"]), "withdrawal_request_confirmation", account["email"],
        "WarriorIQ withdrawal request received",
        f"Your withdrawal request was received on {action['requested_at']} and is pending eligibility and payment review. This is separate from normal subscription cancellation.",
        {"requested_at": action["requested_at"], "status": "pending_review"},
    )
    return RedirectResponse("/settings/billing?notice=Withdrawal+request+recorded+for+review", status_code=303)
