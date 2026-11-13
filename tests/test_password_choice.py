"""Choosing a password, and the sign-up / sign-in form (owner, 2026-11-13).

The length rule alone accepted `password123456`. A new password that is one
of the most used ones, or one of them padded with digits and symbols, is now
refused at sign-up, reset and change; signing in with an older one still
works. The form asks for email and password first, has a show/hide button,
and gives one quiet hint line built from the server's own lists.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from browser_client import BrowserClient

import app.main as web
from core import common_passwords
from core.auth import COMMON_PASSWORD_MESSAGE, hash_password, new_password_problem
from core.db import create_account

ADULT = {"accept_terms": "true", "age_group": "adult"}


@pytest.mark.parametrize("password", [
    "password123456", "Password2024!", "qwertyuiop", "qwerty1234567", "1234567890", "!!dragon!!2026",
    "aaaaaaaaaaaa", "abababababab", "1q2w3e4r5t", "WarriorIQ-2026", "letmein!!!!!",
])
def test_common_passwords_are_refused(password):
    assert common_passwords.is_common(password)
    assert new_password_problem(password) == COMMON_PASSWORD_MESSAGE


@pytest.mark.parametrize("password", [
    "Strong-Local-Password", "velvet-moth-arcade", "Testpass-12345", "correct horse battery", "k9#Lm2vQx!7r",
])
def test_ordinary_passwords_are_accepted(password):
    assert not common_passwords.is_common(password)
    assert new_password_problem(password) == ""


def test_short_passwords_still_get_the_length_reason():
    assert "between 10 and 1,024" in new_password_problem("abc")


def test_signup_refuses_a_common_password_and_creates_nothing():
    web._rate_windows.clear()
    email = f"common-{uuid.uuid4().hex[:8]}@example.com"
    response = BrowserClient(web.app).post(
        "/signup", data={"email": email, "password": "Password2024!", **ADULT}, follow_redirects=False)
    assert response.status_code == 400
    assert "too common" in response.text
    assert f'value="{email}"' in response.text
    assert web.get_account_by_email(email) is None


def test_an_existing_common_password_still_signs_in():
    web._rate_windows.clear()
    email = f"legacy-{uuid.uuid4().hex[:8]}@example.com"
    create_account(email, hash_password("password123456"))
    response = BrowserClient(web.app).post(
        "/login", data={"email": email, "password": "password123456"}, follow_redirects=False)
    assert response.status_code == 303
    assert web.SESSION_COOKIE in response.headers.get("set-cookie", "")


def test_changing_to_a_common_password_is_refused():
    web._rate_windows.clear()
    email = f"change-{uuid.uuid4().hex[:8]}@example.com"
    client = BrowserClient(web.app)
    client.post("/signup", data={"email": email, "password": "Strong-Local-Password", **ADULT})
    response = client.post("/settings/password", data={
        "current_password": "Strong-Local-Password", "new_password": "qwerty1234567",
        "confirm_password": "qwerty1234567"}, follow_redirects=False)
    assert response.status_code == 303
    assert "too+common" in response.headers["location"] or "too%20common" in response.headers["location"]
    signed_in = BrowserClient(web.app).post(
        "/login", data={"email": email, "password": "Strong-Local-Password"}, follow_redirects=False)
    assert signed_in.status_code == 303


def test_signup_form_order_and_hint_rules():
    web._rate_windows.clear()
    page = BrowserClient(web.app).get("/signup").text
    assert page.index('id="authEmail"') < page.index('id="authPassword"') < page.index('name="age_group"') \
        < page.index('name="accept_terms"') < page.index('class="btn auth-submit"')
    assert 'class="password-toggle"' in page and 'autocomplete="new-password"' in page
    rules = json.loads(re.search(r"data-rules='([^']*)'", page).group(1))
    assert rules == common_passwords.page_rules()
    assert 'data-loading-label="Creating account…"' in page


def test_login_form_has_no_signup_rules():
    web._rate_windows.clear()
    page = BrowserClient(web.app).get("/login").text
    assert "data-rules='" not in page and 'minlength="' not in page
    assert 'class="password-toggle"' in page and 'autocomplete="current-password"' in page
    assert page.index('id="authPassword"') > page.index('href="/forgot-password"')
    assert 'data-loading-label="Signing in…"' in page
    assert "auth-required-choices" not in page
