"""The authenticator-code module and the signed-in devices list (2026-11-12).

Two-step sign-in itself was removed from the site on 2026-11-23 (owner: "we're
not a banking app"); tests/test_simple_sign_in.py covers that. What stays here:
core/two_factor.py still computes codes correctly (nothing on the site calls it
now), and one person can never sign out another's device.
"""

from __future__ import annotations

import base64
import re
import uuid


import app.main as web
from browser_client import BrowserClient
from core import db, two_factor
from core.auth import hash_password, issue_session, token_digest

PASSWORD = "correct-horse-battery"


# --- the code itself --------------------------------------------------------

def test_codes_match_the_rfc_6238_test_vectors():
    secret = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
    for moment, expected in ((59, "287082"), (1111111109, "081804"), (1234567890, "005924"), (2000000000, "279037")):
        assert two_factor.code_at(secret, moment // 30) == expected


def test_a_code_is_taken_once_and_only_inside_the_window():
    secret = two_factor.new_secret()
    now = 1_700_000_000.0
    step = two_factor.current_step(now)
    good = two_factor.code_at(secret, step)
    assert two_factor.verify(secret, good, None, now) == step
    assert two_factor.verify(secret, good, step, now) is None                       # already used
    assert two_factor.verify(secret, two_factor.code_at(secret, step - 1), None, now) == step - 1   # a slow phone
    assert two_factor.verify(secret, two_factor.code_at(secret, step - 3), None, now) is None       # too old
    assert two_factor.verify(secret, "12345", None, now) is None and two_factor.verify(secret, "", None, now) is None
    assert two_factor.verify(secret, good[:3] + " " + good[3:], None, now) == step    # spaces are fine


def test_recovery_codes_are_readable_and_only_their_hashes_are_kept():
    codes = two_factor.new_recovery_codes()
    assert len(codes) == len(set(codes)) == two_factor.RECOVERY_CODE_COUNT
    assert all(len(code) == 10 and not set(code) & set("01oilOIL") for code in codes)
    shown = two_factor.display_code(codes[0])
    assert two_factor.recovery_hash(shown.upper()) == two_factor.recovery_hash(codes[0])
    assert two_factor.looks_like_recovery(shown) and not two_factor.looks_like_recovery("123 456")


def test_the_secret_is_sealed_when_a_key_is_set_and_fails_closed_without_it(monkeypatch):
    monkeypatch.delenv("WARRIORIQ_TWO_FACTOR_KEY", raising=False)
    assert two_factor.seal("ABC").startswith("plain:") and two_factor.unseal(two_factor.seal("ABC")) == "ABC"
    monkeypatch.setenv("WARRIORIQ_TWO_FACTOR_KEY", "a long private passphrase")
    sealed = two_factor.seal("ABC")
    assert sealed.startswith("fernet:") and "ABC" not in sealed and two_factor.unseal(sealed) == "ABC"
    monkeypatch.setenv("WARRIORIQ_TWO_FACTOR_KEY", "a different passphrase")
    assert two_factor.unseal(sealed) is None
    monkeypatch.delenv("WARRIORIQ_TWO_FACTOR_KEY")
    assert two_factor.unseal(sealed) is None


# --- the web flow ----------------------------------------------------------

def _account():
    email = f"2fa-{uuid.uuid4().hex}@example.test"
    created = db.create_account(email, hash_password(PASSWORD))
    return created, email


def _signed_in_client(created):
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    return client


def _turn_on(created, client) -> tuple[str, list[str]]:
    """Walk the real settings flow; returns the secret and the recovery codes shown."""
    assert client.post("/settings/2fa/start", data={"current_password": PASSWORD}, follow_redirects=False).status_code == 303
    pending = two_factor.unseal(db.two_factor_row(created["id"])["secret"])
    assert pending and not db.two_factor_row(created["id"])["enabled"]       # a secret alone is not "on"
    page = client.get("/settings/privacy").text
    assert two_factor.grouped(pending) in page
    done = client.post("/settings/2fa/confirm", data={"code": two_factor.code_at(pending, two_factor.current_step())})
    assert done.status_code == 200 and "no-store" in done.headers["cache-control"]
    codes = re.findall(r"<li>([a-z0-9]{5}-[a-z0-9]{5})</li>", done.text)
    assert len(codes) == two_factor.RECOVERY_CODE_COUNT and db.two_factor_row(created["id"])["enabled"]
    return pending, [code.replace("-", "") for code in codes]


def _login(email):
    client = BrowserClient(web.app)
    web._rate_windows.clear()
    return client, client.post("/login", data={"email": email, "password": PASSWORD}, follow_redirects=False)


def _next_code(secret, last=0):
    """A valid code for a step after the last one used: the test stands in for a person waiting for the next code."""
    state = {"step": max(two_factor.current_step(), last + 1)}
    return two_factor.code_at(secret, state["step"]), state["step"]


# --- the signed-in devices list ---------------------------------------------

def test_the_device_list_names_devices_and_signs_out_only_your_own():
    created, email = _account()
    client = BrowserClient(web.app)
    web._rate_windows.clear()
    chrome = {"user-agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/120.0 Safari/537.36"}
    iphone = {"user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Safari/604.1"}
    client.post("/login", data={"email": email, "password": PASSWORD}, headers=chrome, follow_redirects=False)
    phone = BrowserClient(web.app)
    phone.post("/login", data={"email": email, "password": PASSWORD}, headers=iphone, follow_redirects=False)
    page = client.get("/settings/privacy").text
    assert "Chrome on Windows" in page and "Safari on iPhone" in page and "This browser" in page
    sessions = db.list_account_sessions(created["id"])
    other_id = next(item["id"] for item in sessions if "iPhone" in item["device"])
    # Someone else's account cannot reach it.
    stranger, _ = _account()
    assert not db.revoke_session_by_id(stranger["id"], other_id)
    assert db.account_for_session(token_digest(phone.cookies[web.SESSION_COOKIE])) is not None
    # The owner can.
    done = client.post(f"/settings/sessions/{other_id}/revoke", follow_redirects=False)
    assert "notice=" in done.headers["location"]
    assert db.account_for_session(token_digest(phone.cookies[web.SESSION_COOKIE])) is None
    # And not the browser they are using.
    mine = next(item["id"] for item in db.list_account_sessions(
        created["id"], token_digest(client.cookies[web.SESSION_COOKIE])) if item["current"])
    assert "error=" in client.post(f"/settings/sessions/{mine}/revoke", follow_redirects=False).headers["location"]


def test_no_raw_user_agent_or_address_is_stored_with_a_session():
    created, email = _account()
    BrowserClient(web.app).post("/login", data={"email": email, "password": PASSWORD},
                                headers={"user-agent": "Mozilla/5.0 SecretBuild/9.9 Chrome/120 (Windows NT 10)"}, follow_redirects=False)
    with db.connection() as con:
        labels = [row["device_label"] for row in con.execute("SELECT device_label FROM sessions WHERE account_id=?", (created["id"],))]
    assert labels == ["Chrome on Windows"]


# --- the admin page says whether secrets are encrypted ------------------------

def _admin_client(monkeypatch):
    from core.config import SETTINGS

    created, email = _account()
    object.__setattr__(SETTINGS, "admin_emails", (email.lower(),))     # SETTINGS is frozen, so as the other tests do
    return created, _signed_in_client(created)

