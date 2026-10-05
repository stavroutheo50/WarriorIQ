"""Athlete profiles and follows: private by default, minors always private."""

import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_client import BrowserClient  # noqa: E402

from app import main  # noqa: E402
from core import db, social  # noqa: E402
from core.auth import register  # noqa: E402

PASSWORD = "Strong-Local-Password"


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    # Many sign-ins from one test address would trip the login limit, as the
    # other account tests also find (tests/test_product.py patches it the same way).
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *args, **kwargs: None)


# --- the rules -----------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [("Alex.Kicks", "alex.kicks"), ("@fighter_1", "fighter_1"), ("", None)])
def test_handles_are_normalised(raw, expected):
    assert social.normalize_handle(raw) == expected


@pytest.mark.parametrize("raw", ["ab", "-bad", "admin", "ends_", "has space", "x" * 31])
def test_bad_or_reserved_handles_are_refused(raw):
    with pytest.raises(ValueError):
        social.normalize_handle(raw)


def test_minors_are_private_and_unfollowable_whatever_the_setting():
    profile = {"id": 2, "handle": "kid", "visibility": "public"}
    assert social.effective_visibility(profile, minor=True) == "private"
    assert not social.can_view(viewer_profile_id=9, profile=profile, minor=True, follow="accepted")
    assert not social.can_follow(viewer_profile_id=9, profile=profile, minor=True)
    assert not social.is_discoverable(viewer_profile_id=9, profile=profile, minor=True)
    assert social.can_view(viewer_profile_id=2, profile=profile, minor=True, follow=None)


def test_private_profiles_show_details_only_to_accepted_followers():
    profile = {"id": 2, "handle": "alex", "visibility": "private"}
    assert not social.can_view(viewer_profile_id=9, profile=profile, minor=False, follow=None)
    assert not social.can_view(viewer_profile_id=9, profile=profile, minor=False, follow="pending")
    assert social.can_view(viewer_profile_id=9, profile=profile, minor=False, follow="accepted")


def test_the_guardian_marker_identifies_a_minor():
    assert social.is_minor_account({"guardian_approval_status": "pending_guardian"})
    assert not social.is_minor_account({"guardian_approval_status": "not_applicable"})
    assert not social.is_minor_account(None)


# --- the pages -------------------------------------------------------------------

def _athlete(handle=None, visibility="private", minor=False):
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    account = register(email, PASSWORD)
    if minor:
        with db.connection() as con:
            con.execute("UPDATE accounts SET guardian_approval_status='approved' WHERE id=?", (account["id"],))
    client = BrowserClient(main.app)
    client.post("/login", data={"email": email, "password": PASSWORD})
    if handle:
        client.post("/profile/public", data={"handle": handle, "visibility": visibility, "bio": "Kickboxer", "gym": "Team X"})
    return client, int(account["profile_id"])


def _handle():
    return "a" + uuid.uuid4().hex[:10]


def test_a_new_profile_is_private():
    client, profile_id = _athlete()
    assert (db.get_profile(profile_id) or {}).get("visibility") == "private"


def test_following_a_public_athlete_is_immediate():
    handle = _handle()
    _, star = _athlete(handle, "public")
    fan, fan_id = _athlete()
    page = fan.get(f"/athlete/{handle}")
    assert page.status_code == 200 and "training streak" in page.text
    fan.post(f"/athlete/{handle}/follow")
    assert db.follow_status(fan_id, star) == "accepted"


def test_a_private_athlete_approves_followers_before_details_show():
    handle = _handle()
    owner, owner_id = _athlete(handle, "private")
    fan, fan_id = _athlete()
    page = fan.get(f"/athlete/{handle}")
    assert "This profile is private" in page.text and "training streak" not in page.text and "Team X" not in page.text
    fan.post(f"/athlete/{handle}/follow")
    assert db.follow_status(fan_id, owner_id) == "pending"
    assert "Waiting for your approval" in owner.get("/profile").text
    owner.post(f"/profile/followers/{fan_id}/approve")
    assert db.follow_status(fan_id, owner_id) == "accepted"
    assert "training streak" in fan.get(f"/athlete/{handle}").text


def test_a_minor_cannot_go_public_and_is_invisible_to_others():
    handle = _handle()
    kid, kid_id = _athlete(handle, "public", minor=True)
    assert (db.get_profile(kid_id) or {}).get("visibility") == "private"
    stranger, _ = _athlete()
    assert stranger.get(f"/athlete/{handle}").status_code == 404
    assert stranger.post(f"/athlete/{handle}/follow").status_code == 404
    assert kid.get(f"/athlete/{handle}").status_code == 200


def test_a_taken_username_is_refused():
    handle = _handle()
    _athlete(handle, "public")
    other, other_id = _athlete()
    response = other.post("/profile/public", data={"handle": handle, "visibility": "public"})
    assert "taken" in response.text
    assert (db.get_profile(other_id) or {}).get("handle") is None


def test_a_profile_report_reaches_the_moderation_queue():
    handle = _handle()
    _athlete(handle, "public")
    reporter, _ = _athlete()
    reporter.post(f"/athlete/{handle}/report", data={"reason": "Not a real athlete"})
    assert any(r["report_type"] == "profile" and handle in (r["resource_id"] or "")
               for r in db.list_moderation_reports())


def test_deleting_an_account_removes_its_follows():
    handle = _handle()
    _, star = _athlete(handle, "public")
    fan, fan_id = _athlete()
    fan.post(f"/athlete/{handle}/follow")
    account = db.get_account_by_profile(fan_id)
    db.delete_account(int(account["id"]))
    assert db.count_follows(star)["followers"] == 0
