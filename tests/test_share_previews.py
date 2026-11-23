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


def test_the_link_preview_leads_with_the_numbers_and_invites_a_tap():
    """What WhatsApp and the rest show above the picture (2026-11-23)."""
    import re

    owner, account, _ = _owner()
    job_id = _shareable_fight(int(account["profile_id"]))
    anonymous = owner.post(f"/story/{job_id}", data={"side": "A"}).json()["url"]
    named = owner.post(f"/story/{job_id}", data={"side": "B", "name": "Nikos"}).json()["url"]
    stranger = BrowserClient(main.app)
    page = stranger.get("/f/" + anonymous.rsplit("/f/", 1)[1]).text
    title = re.search(r'og:title" content="([^"]*)"', page).group(1)
    description = re.search(r'og:description" content="([^"]*)"', page).group(1)
    assert "Fighter A" not in title and "thrown" in title
    assert "not checked by a person" in description and "Analyse your own fight free" in description
    page = stranger.get("/f/" + named.rsplit("/f/", 1)[1]).text
    assert re.search(r'og:title" content="([^"]*)"', page).group(1).startswith("Nikos · ")


def test_the_preview_picture_still_draws():
    from core.share_image import preview_png

    card = {"sport": "Kickboxing", "fighters": {"A": {"total": 9, "strikes": {"punch": 9}},
                                                 "B": {"total": 3, "strikes": {"punch": 3}}}}
    png = preview_png(card, "A", None)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000
