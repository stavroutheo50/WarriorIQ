"""Signing up with an email that already has an account (2026-10-28).

It re-showed the sign-up form, emptied, with the reason below the fold on a
phone, so it looked as if nothing had happened. It now shows the sign-in form
with the address already filled in and the reason at the top.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from browser_client import BrowserClient

import app.main as web

ROOT = Path(__file__).resolve().parents[1]
SIGNUP = {"password": "Strong-Local-Password", "accept_terms": "true", "age_group": "adult"}


def test_an_existing_email_is_sent_to_sign_in_with_the_address_kept():
    web._rate_windows.clear()
    email = f"exists-{uuid.uuid4().hex[:8]}@example.com"
    first = BrowserClient(web.app).post("/signup", data={"email": email, **SIGNUP}, follow_redirects=False)
    assert first.status_code == 303
    again = BrowserClient(web.app).post("/signup", data={"email": email.upper(), **SIGNUP}, follow_redirects=False)
    assert again.status_code == 400
    page = again.text
    assert "You already have an account with this email." in page
    assert 'action="/login"' in page and 'action="/signup"' not in page
    assert f'value="{email.upper()}"' in page
    assert web.SESSION_COOKIE not in again.headers.get("set-cookie", "")


def test_other_refusals_keep_the_address_too():
    web._rate_windows.clear()
    bad = BrowserClient(web.app).post("/signup", data={"email": "not-an-email", **SIGNUP}, follow_redirects=False)
    assert bad.status_code == 400 and 'action="/signup"' in bad.text and 'value="not-an-email"' in bad.text
    wrong = BrowserClient(web.app).post("/login", data={"email": "nobody@example.com", "password": "x" * 12},
                                        follow_redirects=False)
    assert 'value="nobody@example.com"' in wrong.text


def test_the_typed_address_is_escaped():
    web._rate_windows.clear()
    response = BrowserClient(web.app).post(
        "/login", data={"email": '"><script>alert(1)</script>', "password": "x" * 12}, follow_redirects=False)
    assert "<script>alert(1)</script>" not in response.text


def test_a_refusal_is_scrolled_into_view():
    page = (ROOT / "app" / "templates" / "auth.html").read_text(encoding="utf-8")
    assert "refusal.scrollIntoView({block:'center'})" in page
