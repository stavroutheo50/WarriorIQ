"""Finishing an email link on another device (owner, 2026-11-16).

"If I am on my PC and I accept it on my phone, I want to continue on my PC
where I left off." The browser that asked waits; once the link is used on any
device it moves on: after email verification it is signed in and sent where it
was heading; after a password reset it is sent to sign-in (never signed in by
itself), still heading there.
"""

from __future__ import annotations

import re
import uuid
from unittest.mock import patch

import pytest

import app.main as web
from browser_client import BrowserClient
from core.config import SETTINGS

SIGNUP = {"password": "Strong-Local-Password", "accept_terms": "true", "age_group": "adult"}


@pytest.fixture
def verification_on():
    previous = SETTINGS.require_email_verification
    object.__setattr__(SETTINGS, "require_email_verification", True)
    web._rate_windows.clear()
    yield
    object.__setattr__(SETTINGS, "require_email_verification", previous)


def _link(body: str, path: str) -> str:
    return re.search(rf"https?://[^/\s]+(/{path}/[^\s]+)", body).group(1)


def _sign_up(pc: BrowserClient, email: str, next_path: str = "/history") -> str:
    sent = []
    with patch.object(web, "send_transactional_email", lambda to, subject, body: sent.append(body) or True):
        response = pc.post("/signup", data={"email": email, "next_path": next_path, **SIGNUP},
                           follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/verify-email"
    return _link(sent[0], "verify-email")


def test_verifying_on_the_phone_signs_the_computer_in_where_it_was(verification_on):
    pc, phone = BrowserClient(web.app), BrowserClient(web.app)
    link = _sign_up(pc, f"cross-{uuid.uuid4().hex[:8]}@example.com")
    page = pc.get("/verify-email").text
    assert "data-auth-wait" in page and "/static/auth-wait.js" in page
    assert pc.get("/auth/wait").json() == {"state": "waiting"}

    done = phone.get(link)
    assert done.status_code == 200 and "Your email is confirmed." in done.text
    assert web.SESSION_COOKIE not in phone.cookies                 # the phone is not signed in by this

    assert pc.get("/auth/wait").json()["state"] == "done"
    carry_on = pc.get("/auth/continue", follow_redirects=False)
    assert carry_on.status_code == 303 and carry_on.headers["location"] == "/history"
    assert web.SESSION_COOKIE in carry_on.headers.get("set-cookie", "")
    # Used once.
    assert pc.get("/auth/continue", follow_redirects=False).headers["location"] == "/login"


def test_verifying_in_the_same_browser_goes_straight_on(verification_on):
    pc = BrowserClient(web.app)
    link = _sign_up(pc, f"same-{uuid.uuid4().hex[:8]}@example.com", next_path="/camp")
    response = pc.get(link, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/camp"
    assert web.SESSION_COOKIE in response.headers.get("set-cookie", "")


def test_an_old_verification_link_explains_itself(verification_on):
    response = BrowserClient(web.app).get("/verify-email/not-a-real-token")
    assert response.status_code == 410
    assert "This link has expired or a newer one was sent." in response.text and "Send a new link" in response.text


def _account(email: str):
    web._rate_windows.clear()
    BrowserClient(web.app).post("/signup", data={"email": email, **SIGNUP}, follow_redirects=False)


def test_resetting_on_the_phone_sends_the_computer_to_sign_in_not_signed_in():
    email = f"reset-{uuid.uuid4().hex[:8]}@example.com"
    _account(email)
    pc, phone = BrowserClient(web.app), BrowserClient(web.app)
    sent = []
    with patch.object(web, "deliver_email", lambda to, subject, body: sent.append(body)):
        page = pc.post("/forgot-password", data={"email": email, "next_path": "/history"})
    assert "Check your email" in page.text and email in page.text and "Junk" in page.text
    assert pc.get("/auth/wait").json() == {"state": "waiting"}

    link = _link(sent[0], "reset-password")
    done = phone.post(link, data={"password": "A-Brand-New-Password-7"}, follow_redirects=False)
    assert done.status_code == 200 and "Your password has been changed." in done.text

    assert pc.get("/auth/wait").json()["state"] == "done"
    carry_on = pc.get("/auth/continue", follow_redirects=False)
    assert carry_on.headers["location"] == "/login?reset=1&next=%2Fhistory"
    assert web.SESSION_COOKIE not in carry_on.headers.get("set-cookie", "")   # never signed in by a reset
    login = pc.get(carry_on.headers["location"])
    assert "Your password has been changed. Sign in with the new one." in login.text
    assert 'name="next_path" value="/history"' in login.text


def test_an_address_without_an_account_waits_the_same_and_never_finishes():
    pc = BrowserClient(web.app)
    web._rate_windows.clear()
    page = pc.post("/forgot-password", data={"email": f"nobody-{uuid.uuid4().hex[:8]}@example.com"})
    assert "Check your email" in page.text
    assert pc.get("/auth/wait").json() == {"state": "waiting"}


def test_a_used_reset_link_explains_itself():
    web._rate_windows.clear()
    response = BrowserClient(web.app).post("/reset-password/not-a-real-token",
                                           data={"password": "A-Brand-New-Password-7"}, follow_redirects=False)
    assert response.status_code == 410
    assert "This link has expired or was already used." in response.text and "Send reset link" in response.text


def test_the_duplicate_sign_up_leads_to_sign_in_with_the_address_filled():
    email = f"dupe-{uuid.uuid4().hex[:8]}@example.com"
    _account(email)
    web._rate_windows.clear()
    client = BrowserClient(web.app)
    page = client.post("/signup", data={"email": email, "next_path": "/history", **SIGNUP}).text
    box = page[page.index('<div class="account-exists"'):]
    box = box[:box.index("</div>")]
    assert 'href="/login?next=/history"' in box and ">Sign in</a>" in box
    login = client.get("/login?next=%2Fhistory").text
    assert f'value="{email}"' in login and 'name="next_path" value="/history"' in login
