"""A coach sees an athlete's fights only with the athlete's consent (owner's brief, 2026-11-14)."""

from __future__ import annotations

import json
import re
import uuid
from unittest.mock import patch

import pytest

import app.main as web
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session


def _person(coach: bool = False, minor: bool = False):
    email = f"coaching-{uuid.uuid4().hex[:10]}@example.test"
    created = db.create_account(email, hash_password("Strong-Local-Password"))
    if coach:
        db.set_account_type(int(created["profile_id"]), "coach")
    if minor:
        db.set_guardian_approval_status(int(created["id"]), "accepted")
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    client.get("/coaching")                                   # the form token cookie
    return created, email, client


def _post(client, url, data=None):
    web._rate_windows.clear()
    return client.post(url, data=data or {}, headers={"X-CSRF-Token": client.cookies.get("warrioriq_csrf")},
                       follow_redirects=False)


def _invite(coach_client, email) -> str:
    with patch.object(web, "deliver_email", side_effect=web.EmailNotSent("off")):
        answer = _post(coach_client, "/coaching/invite", {"email": email})
    assert answer.status_code == 303, answer.text
    return re.search(r"invite=([A-Za-z0-9_-]+)", answer.headers["location"]).group(1)


def _fight(athlete, tmp_path_factory=None):
    job_id = f"coach{uuid.uuid4().hex[:8]}"
    folder = web.OUTPUTS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    report = {"setup": {"ruleset": "K1"}, "video": {"analysis_target": "BOTH"}, "metrics": {}, "integrity": {},
              "scorecard": {"ruleset_label": "K-1"}, "tracking": {}, "coaching": {}}
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    db.save_fight(job_id, int(athlete["profile_id"]), "fight.mp4", "", str(folder / "report.json"),
                  "sparring", "K1", "BOTH", {})
    return job_id


@pytest.fixture
def linked():
    coach, _, coach_client = _person(coach=True)
    athlete, athlete_email, athlete_client = _person()
    with patch.object(web, "_request_plan", return_value={"roster_limit": 5}):
        token = _invite(coach_client, athlete_email)
    page = athlete_client.get(f"/coaching/invite/{token}")
    assert page.status_code == 200 and "wants to coach you" in page.text
    assert _post(athlete_client, f"/coaching/invite/{token}/answer", {"answer": "accept"}).status_code == 303
    return coach, coach_client, athlete, athlete_client


def test_the_coach_sees_only_after_accepting_and_until_it_ends(linked):
    coach, coach_client, athlete, athlete_client = linked
    job_id = _fight(athlete)
    athlete_id = int(athlete["profile_id"])
    assert coach_client.get(f"/coaching/athletes/{athlete_id}").status_code == 200
    report = coach_client.get(f"/coaching/reports/{job_id}")
    assert report.status_code == 200 and "shared their fights with you" in report.text
    # Notes and objectives reach the athlete.
    assert _post(coach_client, f"/coaching/athletes/{athlete_id}/notes", {"body": "Keep the chin down"}).status_code == 303
    assert _post(coach_client, f"/coaching/athletes/{athlete_id}/objectives", {"title": "Jab and step out"}).status_code == 303
    assert "Keep the chin down" in athlete_client.get("/coaching").text
    assert any(item["title"] == "Jab and step out" and item["assigned_by_profile_id"] == int(coach["profile_id"])
               for item in db.list_assignments(athlete_id))
    # The athlete ends it: the coach sees nothing any more.
    (link,) = db.coach_links_for_athlete(athlete_id)
    assert _post(athlete_client, f"/coaching/links/{link['id']}/end").status_code == 303
    assert coach_client.get(f"/coaching/athletes/{athlete_id}").status_code == 404
    assert coach_client.get(f"/coaching/reports/{job_id}").status_code == 404
    assert _post(coach_client, f"/coaching/athletes/{athlete_id}/notes", {"body": "x"}).status_code == 404


def test_nobody_else_sees_anything(linked):
    coach, coach_client, athlete, _ = linked
    job_id = _fight(athlete)
    other_coach, _, other_client = _person(coach=True)
    athlete_id = int(athlete["profile_id"])
    assert other_client.get(f"/coaching/athletes/{athlete_id}").status_code == 404
    assert other_client.get(f"/coaching/reports/{job_id}").status_code == 404
    assert _post(other_client, f"/coaching/athletes/{athlete_id}/objectives", {"title": "x"}).status_code == 404
    assert BrowserClient(web.app).get(f"/coaching/reports/{job_id}").status_code == 404
    # A linked coach cannot attach a note to someone else's fight.
    stranger, _, _ = _person()
    foreign = _fight(stranger)
    assert _post(coach_client, f"/coaching/athletes/{athlete_id}/notes",
                 {"body": "x", "job_id": foreign}).status_code == 404


def test_an_invitation_is_refused_when_it_should_be():
    athlete_only, _, plain_client = _person()
    coach, _, coach_client = _person(coach=True)
    target, target_email, target_client = _person()
    # Not a coach account.
    with patch.object(web, "_request_plan", return_value={"roster_limit": 5}):
        answer = _post(plain_client, "/coaching/invite", {"email": target_email})
    assert "error=" in answer.headers["location"] and not db.coach_links_for_coach(int(athlete_only["profile_id"]))
    # Seats full.
    with patch.object(web, "_request_plan", return_value={"roster_limit": 1}):
        _invite(coach_client, target_email)
        full = _post(coach_client, "/coaching/invite", {"email": f"x{uuid.uuid4().hex[:6]}@example.test"})
    assert "error=" in full.headers["location"] and len(db.coach_links_for_coach(int(coach["profile_id"]))) == 1
    # A declined invitation cannot be accepted afterwards.
    (pending,) = db.coach_links_for_coach(int(coach["profile_id"]))
    with patch.object(web, "_request_plan", return_value={"roster_limit": 5}):
        token = _invite(coach_client, target_email)
    assert _post(target_client, f"/coaching/invite/{token}/answer", {"answer": "decline"}).status_code == 303
    assert _post(target_client, f"/coaching/invite/{token}/answer", {"answer": "accept"}).status_code == 404
    assert not db.coach_links_for_athlete(int(target["profile_id"]))


def test_an_under_18_account_cannot_accept():
    coach, _, coach_client = _person(coach=True)
    minor, minor_email, minor_client = _person(minor=True)
    with patch.object(web, "_request_plan", return_value={"roster_limit": 5}):
        token = _invite(coach_client, minor_email)
    assert "cannot share their fights with a coach yet" in minor_client.get(f"/coaching/invite/{token}").text
    assert _post(minor_client, f"/coaching/invite/{token}/answer", {"answer": "accept"}).status_code == 403
    assert not db.coach_links_for_athlete(int(minor["profile_id"]))


def test_deleting_an_account_removes_its_links_and_notes(linked):
    coach, coach_client, athlete, _ = linked
    athlete_id = int(athlete["profile_id"])
    _post(coach_client, f"/coaching/athletes/{athlete_id}/notes", {"body": "note"})
    db.delete_account(int(athlete["id"]))
    assert not db.coach_links_for_coach(int(coach["profile_id"]))
    assert not db.list_coach_notes(athlete_id)


def test_tokens_in_addresses_are_never_logged(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="warrioriq")
    BrowserClient(web.app).get("/coaching/invite/secret-invite-token-123456")
    BrowserClient(web.app).get("/reset-password/secret-reset-token-123456")
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "secret-invite-token" not in logged and "secret-reset-token" not in logged
    assert "/coaching/invite/[token]" in logged


def test_a_rejected_form_on_an_invitation_address_stores_no_token():
    from fastapi.testclient import TestClient

    client = TestClient(web.app)                  # a plain client: it sends no form token
    web._rate_windows.clear()
    refused = client.post("/coaching/invite/secret-invite-token-abcdef/answer", data={"answer": "accept"})
    assert refused.status_code == 403                                  # no CSRF token sent
    stored = [e for e in db.list_security_events(50) if e["event_type"] == "csrf_token_rejected"]
    assert stored and stored[0]["resource_id"] == "/coaching/invite/[token]/answer"
    assert not any("secret-invite-token" in str(e.get("resource_id")) for e in db.list_security_events(200))
