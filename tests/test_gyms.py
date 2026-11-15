"""Gyms: coaches and athletes under one organisation, joined only by consent (owner's brief, 2026-11-14)."""

from __future__ import annotations

import json
import re
import uuid
from unittest.mock import patch

import app.main as web
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session


def _person(minor: bool = False):
    email = f"gym-{uuid.uuid4().hex[:10]}@example.test"
    created = db.create_account(email, hash_password("Strong-Local-Password"))
    if minor:
        db.set_guardian_approval_status(int(created["id"]), "accepted")
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    client.get("/gym")
    return created, email, client


def _post(client, url, data=None):
    web._rate_windows.clear()
    return client.post(url, data=data or {}, headers={"X-CSRF-Token": client.cookies.get("warrioriq_csrf")},
                       follow_redirects=False)


def _invite(owner_client, email, role):
    with patch.object(web, "deliver_email", side_effect=web.EmailNotSent("off")), \
         patch.object(web, "_request_plan", return_value={"roster_limit": 3}):
        answer = _post(owner_client, "/gym/invite", {"email": email, "role": role})
    assert answer.status_code == 303 and "invite=" in answer.headers["location"], answer.headers.get("location")
    return re.search(r"invite=([A-Za-z0-9_-]+)", answer.headers["location"]).group(1)


def _fight(person):
    job_id = f"gym{uuid.uuid4().hex[:8]}"
    folder = web.OUTPUTS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(
        {"setup": {"ruleset": "K1"}, "video": {"analysis_target": "BOTH"}, "metrics": {}, "integrity": {},
         "scorecard": {"ruleset_label": "K-1"}, "tracking": {}, "coaching": {}}), encoding="utf-8")
    db.save_fight(job_id, int(person["profile_id"]), "f.mp4", "", str(folder / "report.json"), "sparring", "K1", "BOTH", {})
    return job_id


def test_a_gym_coach_sees_a_gym_athlete_only_while_both_belong():
    owner, _, owner_client = _person()
    coach, coach_email, coach_client = _person()
    athlete, athlete_email, athlete_client = _person()
    assert _post(owner_client, "/gym/create", {"name": "Athens Muay Thai"}).status_code == 303
    coach_token = _invite(owner_client, coach_email, "coach")
    athlete_token = _invite(owner_client, athlete_email, "athlete")
    job_id = _fight(athlete)
    athlete_id = int(athlete["profile_id"])
    # Nobody sees anything before the athlete joins.
    assert _post(coach_client, f"/gym/invite/{coach_token}/answer", {"answer": "accept"}).status_code == 303
    assert coach_client.get(f"/coaching/reports/{job_id}").status_code == 404
    assert "the coaches of Athens Muay Thai can read your fight reports" in athlete_client.get(f"/gym/invite/{athlete_token}").text
    assert _post(athlete_client, f"/gym/invite/{athlete_token}/answer", {"answer": "accept"}).status_code == 303
    # Now the gym's coach and its owner see the athlete; an outsider does not.
    assert coach_client.get(f"/coaching/reports/{job_id}").status_code == 200
    assert owner_client.get(f"/coaching/athletes/{athlete_id}").status_code == 200
    assert "Athens Muay Thai" in coach_client.get("/coaching").text
    _, _, outsider = _person()
    assert outsider.get(f"/coaching/reports/{job_id}").status_code == 404
    # The athlete cannot see the coach's fights: access runs one way.
    coach_fight = _fight(coach)
    assert athlete_client.get(f"/coaching/reports/{coach_fight}").status_code == 404
    # The athlete leaves: access ends.
    (membership,) = db.memberships_of(athlete_id)
    assert _post(athlete_client, f"/gym/members/{membership['id']}/end").status_code == 303
    assert coach_client.get(f"/coaching/reports/{job_id}").status_code == 404


def test_only_the_owner_runs_the_gym_and_nobody_removes_the_owner():
    owner, _, owner_client = _person()
    member, member_email, member_client = _person()
    _post(owner_client, "/gym/create", {"name": "Gym"})
    token = _invite(owner_client, member_email, "coach")
    _post(member_client, f"/gym/invite/{token}/answer", {"answer": "accept"})
    org = db.organisation_owned_by(int(owner["profile_id"]))
    with patch.object(web, "_request_plan", return_value={"roster_limit": 3}):
        assert _post(member_client, "/gym/invite", {"email": "x@example.test"}).status_code == 404
    owner_row = next(m for m in db.organisation_members(int(org["id"])) if m["role"] == "owner")
    assert _post(member_client, f"/gym/members/{owner_row['id']}/end").status_code == 404
    assert _post(owner_client, f"/gym/members/{owner_row['id']}/end").status_code == 404


def test_seats_minors_and_declines():
    owner, _, owner_client = _person()
    _post(owner_client, "/gym/create", {"name": "Gym"})
    minor, minor_email, minor_client = _person(minor=True)
    token = _invite(owner_client, minor_email, "athlete")
    assert "cannot join a gym as an athlete yet" in minor_client.get(f"/gym/invite/{token}").text
    assert _post(minor_client, f"/gym/invite/{token}/answer", {"answer": "accept"}).status_code == 403
    # Seats: three athlete invitations fill a plan of three.
    for _ in range(2):
        _invite(owner_client, f"a{uuid.uuid4().hex[:6]}@example.test", "athlete")
    with patch.object(web, "_request_plan", return_value={"roster_limit": 3}):
        full = _post(owner_client, "/gym/invite", {"email": "more@example.test", "role": "athlete"})
    assert "error=" in full.headers["location"]
    # A declined invitation cannot be used again.
    other, other_email, other_client = _person()
    token = _invite(owner_client, other_email, "coach")
    _post(other_client, f"/gym/invite/{token}/answer", {"answer": "decline"})
    assert _post(other_client, f"/gym/invite/{token}/answer", {"answer": "accept"}).status_code == 404


def test_deleting_the_owner_removes_the_gym():
    owner, _, owner_client = _person()
    member, member_email, member_client = _person()
    _post(owner_client, "/gym/create", {"name": "Gym"})
    token = _invite(owner_client, member_email, "athlete")
    _post(member_client, f"/gym/invite/{token}/answer", {"answer": "accept"})
    db.delete_account(int(owner["id"]))
    assert db.memberships_of(int(member["profile_id"])) == []
