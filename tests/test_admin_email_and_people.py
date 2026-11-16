"""Admin: account emails, the sending setup, and one person's picture (owner, 2026-11-16).

A friend's password reset never reached an iCloud inbox. The admin page now
says whether each account email was sent, warns about a sending setup inboxes
distrust, and shows one person's whole picture for a support question.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import app.main as web
from app import admin_overview
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session
from core.config import SETTINGS


def test_a_from_address_on_another_domain_is_flagged(monkeypatch):
    monkeypatch.setenv("WARRIORIQ_SMTP_USERNAME", "warrioriqai@gmail.com")
    monkeypatch.setenv("WARRIORIQ_EMAIL_FROM", "WarriorIQ <hello@warrioriq.eu>")
    warnings = admin_overview.email_setup()["warnings"]
    assert any("different domain" in w for w in warnings)
    monkeypatch.setenv("WARRIORIQ_EMAIL_FROM", "")
    setup = admin_overview.email_setup()
    assert setup["sender"] == "warrioriqai@gmail.com"
    assert not any("different domain" in w for w in setup["warnings"])
    assert any("personal gmail.com mailbox" in w for w in setup["warnings"])
    assert setup["own_domain"] is False                 # so the page shows how to reach every inbox
    monkeypatch.setenv("WARRIORIQ_SMTP_USERNAME", "noreply@warrioriq.eu")
    monkeypatch.setenv("WARRIORIQ_EMAIL_FROM", "noreply@warrioriq.eu")
    setup = admin_overview.email_setup()
    assert setup["own_domain"] is True and setup["warnings"] == []
    monkeypatch.setenv("WARRIORIQ_SMTP_USERNAME", "warrioriqai@gmail.com")
    monkeypatch.setenv("WARRIORIQ_SMTP_PASSWORD", "app-password-value-123")
    assert "app-password-value-123" not in repr(admin_overview.email_setup())     # never the password


def test_every_account_email_is_listed_with_its_result():
    email = f"icloud-friend-{uuid.uuid4().hex[:6]}@icloud.com"
    account = db.create_account(email, hash_password("Strong-Local-Password"))
    db.record_security_event("password_reset_requested", account_id=int(account["id"]),
                             metadata={"email_delivery": "failed", "reason": "the mail server refused the recipient"})
    db.record_security_event("coach_invite_created", account_id=int(account["id"]), metadata={"emailed": True})
    log = admin_overview.email_log()
    reset = next(item for item in log["recent"] if item["to"] == email)
    assert reset["what"] == "Password reset link" and not reset["sent"]
    assert reset["reason"] == "the mail server refused the recipient"
    invite = next(item for item in log["recent"] if item["what"] == "Coaching invitation")
    assert invite["sent"] and invite["to"] == "the person invited"
    assert log["providers"]["icloud.com"]["not_sent"] >= 1


def test_one_person_shows_what_support_needs():
    email = f"person-{uuid.uuid4().hex[:6]}@icloud.com"
    account = db.create_account(email, hash_password("Strong-Local-Password"))
    db.record_security_event("password_reset_requested", account_id=int(account["id"]),
                             metadata={"email_delivery": "sent"})
    card = admin_overview.person(email)[0]
    assert card["email"] == email and card["email_verified_at"] is None and card["last_sign_in"] is None
    assert card["events"][0]["type"] == "password reset requested" and card["events"][0]["email"] == "email sent"
    assert admin_overview.person("") == []


def test_the_admin_page_shows_the_email_section_and_the_person():
    email = f"admin-email-{uuid.uuid4().hex[:6]}@example.test"
    created = db.create_account(email, hash_password("Strong-Local-Password"))
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    previous = SETTINGS.admin_emails
    object.__setattr__(SETTINGS, "admin_emails", (email.lower(),))
    try:
        page = client.get("/admin", params={"q": email}).text
    finally:
        object.__setattr__(SETTINGS, "admin_emails", previous)
    assert 'id="adminEmail"' in page and 'href="#adminEmail"' in page
    assert 'class="admin-person"' in page and email in page and "Email confirmed" in page
    assert f"/admin/accounts/{created['id']}/status" in page
