"""The next action is on the first screen of a phone (walkthrough, 2026-11-22).

On a 390px screen the upload button sat under the cookie notice and two
headings, and the frame to tap was under the cookie notice too.
"""

from __future__ import annotations

import uuid

import app.main as web
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session


def _client():
    created = db.create_account(f"phone-{uuid.uuid4().hex[:8]}@example.test", hash_password("Strong-Local-Password"))
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    return client


def test_the_cookie_notice_stays_off_the_analysis_flow_but_shows_elsewhere():
    client = _client()
    for path in ("/analyze", "/analyze/boxing"):
        assert 'id="cookieNotice"' not in client.get(path).text, path
    assert 'id="cookieNotice"' in client.get("/").text
    assert 'id="cookieNotice"' in BrowserClient(web.app).get("/pricing").text


def test_on_a_phone_the_upload_card_comes_first():
    css = (web.ROOT / "app" / "static" / "walkthrough.css").read_text(encoding="utf-8")
    phone = css[css.index("Phones: the next action on the first screen"):]
    assert ".ux-setup .setup-card { order: -1; }" in phone
    assert ".confirm-email-banner .confirm-email-more { display: none; }" in phone
