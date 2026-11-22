"""Mistyped email providers are caught at sign-up (owner, 2026-11-22: one account typed "gmaill.com")."""

from __future__ import annotations

import uuid

import app.main as web
from browser_client import BrowserClient
from core import db
from core.email_typos import TYPO_DOMAINS, suggestion


def test_common_misspellings_get_the_intended_address():
    assert suggestion("dawa@gmaill.com") == "dawa@gmail.com"
    assert suggestion("Someone@HotMial.com") == "Someone@hotmail.com"
    assert suggestion("me@icloud.con") == "me@icloud.com"


def test_real_addresses_are_never_touched():
    for email in ("me@gmail.com", "me@hotmail.co.uk", "me@yahoo.co.uk", "coach@gym-club.eu", "x@icloud.com",
                  "not-an-email", "@gmaill.com", ""):
        assert suggestion(email) is None, email
    assert not set(TYPO_DOMAINS) & {"gmail.com", "hotmail.com", "yahoo.com", "outlook.com", "icloud.com",
                                    "hotmail.co.uk", "yahoo.co.uk", "live.com", "aol.com"}


def test_sign_up_with_a_typo_is_refused_with_the_fix_and_no_account_is_made():
    email = f"typo{uuid.uuid4().hex[:6]}@gmaill.com"
    web._rate_windows.clear()
    page = BrowserClient(web.app).post("/signup", data={"email": email, "password": "Strong-Local-Password",
                                                        "accept_terms": "true", "age_group": "adult"})
    assert page.status_code == 400
    assert f"Did you mean {email.replace('gmaill', 'gmail')}?" in page.text
    assert db.get_account_by_email(email) is None
    assert 'id="emailTypoHint"' in page.text and "gmaill.com" in page.text
