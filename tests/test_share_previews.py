"""Fight links (/f/<token>) and what they show (2026-11-23).

Found while reviewing share previews: base.html set a template variable named
`who` (the signed-in account's header data) and story.html uses its own `who`
for the fighter's label. For a signed-in viewer the header's won, so the page
printed the raw account dict - email included - where "Fighter A" belonged.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_client import BrowserClient  # noqa: E402

from app import main  # noqa: E402
from core import db  # noqa: E402
from core.auth import issue_session  # noqa: E402
from test_social_profiles import _shareable_fight  # noqa: E402


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *args, **kwargs: None)


def _owner():
    email = f"share-{uuid.uuid4().hex[:10]}@example.test"
    account = db.create_account(email, "x")
    client = BrowserClient(main.app)
    client.cookies.set(main.SESSION_COOKIE, issue_session(account["id"]))
    return client, account, email


def test_a_signed_in_viewer_never_sees_account_data_on_a_fight_link():
    owner, account, email = _owner()
    job_id = _shareable_fight(int(account["profile_id"]))
    url = owner.post(f"/story/{job_id}", data={"side": "A"}).json()["url"]
    path = "/f/" + url.rsplit("/f/", 1)[1]
    for viewer in (owner, BrowserClient(main.app)):           # the owner signed in, and a stranger
        page = viewer.get(path).text
        assert email not in page and email.upper() not in page
        assert "'initial'" not in page.lower() and "has_username" not in page.lower()
        assert '<div class="story-compare-head"><span>Fighter A</span>' in page


def test_the_header_data_carries_no_email():
    assert "email" not in main.account_display({"email": "someone@example.test", "display_name": "Theo"})
