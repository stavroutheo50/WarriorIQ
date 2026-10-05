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


# --- fights posted to a profile -----------------------------------------------------

def _shareable_fight(profile_id):
    """A completed fight whose stats can go on a card (core.report.share_card)."""
    import json

    job_id = "post" + uuid.uuid4().hex[:10]
    job_dir = main.OUTPUTS / job_id
    job_dir.mkdir(parents=True)
    video = main.UPLOADS / f"{job_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video-placeholder")
    report = {
        "setup": {"ruleset": "K1", "fighter_name": "Nikos"},
        "video": {"focus_fighter": "A", "analysis_target": "BOTH", "original_name": "Nikos vs Giorgos.mp4"},
        "integrity": {"identity_evidence_trusted": True},
        "tracking": {"fighter_A_seed_source": "pose_detector", "fighter_B_seed_source": "pose_detector",
                     "initial_iou_A": 0.72, "initial_iou_B": 0.93, "fighter_A_coverage": 0.9,
                     "fighter_B_coverage": 0.9, "fighters_separable": True, "identity_confusions": 0},
        "metrics": {"A": {"guard_index": 0.6}, "B": {}},
        "rounds": [{"number": 1, "selected": True}],
        "events": [],
        "statistics": {"fighters": {"A": {"punch_attempts": 41, "kick_attempts": 28, "knee_attempts": 7},
                                    "B": {"punch_attempts": 12, "kick_attempts": 30, "knee_attempts": 0}}},
        "scorecard": {"available": True, "sport": "kickboxing", "sport_label": "Kickboxing", "totals": {"A": 29, "B": 28}},
    }
    (job_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    db.save_fight(job_id, profile_id, "f.mp4", str(video), str(job_dir / "report.json"),
                  "competition", "K1", "BOTH", {})
    return job_id


def test_a_posted_fight_shows_on_the_athletes_page_without_names():
    handle = _handle()
    owner, owner_id = _athlete(handle, "public")
    job_id = _shareable_fight(owner_id)
    made = owner.post(f"/story/{job_id}/profile", data={"side": "A", "posted": "1"})
    assert made.status_code == 200, made.text
    assert made.json()["posted"] and made.json()["profile_url"] == f"/athlete/{handle}"
    token = made.json()["url"].rsplit("/", 1)[1]

    stranger, _ = _athlete()
    page = stranger.get(f"/athlete/{handle}").text
    assert f"/f/{token}/card.png" in page and "Shared fights" in page
    assert "Giorgos" not in page and "Nikos" not in page

    # Taken off again by its owner, from the profile page.
    owner.post(f"/story/{job_id}/profile", data={"side": "A", "posted": "0", "next_path": "athlete"})
    assert f"/f/{token}/card.png" not in stranger.get(f"/athlete/{handle}").text


def test_a_private_athletes_posts_reach_approved_followers_only():
    handle = _handle()
    owner, owner_id = _athlete(handle, "private")
    job_id = _shareable_fight(owner_id)
    token = owner.post(f"/story/{job_id}/profile", data={"side": "A"}).json()["url"].rsplit("/", 1)[1]

    fan, fan_id = _athlete()
    assert f"/f/{token}" not in fan.get(f"/athlete/{handle}").text
    fan.post(f"/athlete/{handle}/follow")
    owner.post(f"/profile/followers/{fan_id}/approve")
    assert f"/f/{token}/card.png" in fan.get(f"/athlete/{handle}").text


def test_turning_off_the_fight_links_takes_the_post_down():
    handle = _handle()
    owner, owner_id = _athlete(handle, "public")
    job_id = _shareable_fight(owner_id)
    owner.post(f"/story/{job_id}/profile", data={"side": "A"})
    assert len(db.list_profile_posts(owner_id)) == 1
    owner.post(f"/story/{job_id}/revoke")
    assert db.list_profile_posts(owner_id) == []


def test_nobody_posts_someone_elses_fight():
    _, owner_id = _athlete(_handle(), "public")
    job_id = _shareable_fight(owner_id)
    other, _ = _athlete()
    assert other.post(f"/story/{job_id}/profile", data={"side": "A"}).status_code == 404
    assert db.list_profile_posts(owner_id) == []
