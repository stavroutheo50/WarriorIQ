"""Plan requests the administrator approves for a set time (owner, 2026-11-16).

"When someone requests to change their plan, I can accept it, and say for how
long they are on this paid plan." A request comes from "Tell us you want" on
the pricing page; a yes gives the plan with an end date, after which the
person is back on what they had, without anyone remembering to take it away.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

import app.main as web
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session
from core.config import SETTINGS
from core.payments import effective_plan_key, override_in_force


def _person(prefix="asker"):
    email = f"{prefix}-{uuid.uuid4().hex[:10]}@example.test"
    account = db.create_account(email, hash_password("Strong-Local-Password"))
    return int(account["id"]), email


def _client(account_id):
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(account_id))
    web._rate_windows.clear()
    return client


@pytest.fixture
def admin():
    account_id, email = _person("plans-admin")
    previous = SETTINGS.admin_emails
    object.__setattr__(SETTINGS, "admin_emails", (email.lower(),))
    try:
        yield _client(account_id)
    finally:
        object.__setattr__(SETTINGS, "admin_emails", previous)


def test_a_given_plan_ends_by_itself():
    now = datetime.now(timezone.utc)
    assert effective_plan_key("free", "athlete_pro", None, (now + timedelta(days=1)).isoformat()) == "athlete_pro"
    assert effective_plan_key("free", "athlete_pro", None, (now - timedelta(seconds=1)).isoformat()) == "free"
    assert effective_plan_key("athlete", "gym", None, (now - timedelta(days=1)).isoformat()) == "athlete"
    # No end date, as on every grant made before end dates existed.
    assert effective_plan_key("free", "gym", None, None) == "gym"
    assert not override_in_force("gym", "not a date")


def test_a_request_waits_then_a_yes_gives_the_plan_for_that_long(admin, monkeypatch):
    account_id, email = _person()
    _client(account_id).post("/plan-interest/athlete_pro", data={"next_path": "/pricing"})
    assert [r["email"] for r in db.pending_plan_requests() if r["email"] == email] == [email]
    page = admin.get("/admin").text
    assert email in page and "plan request" in page and 'id="adminPlans"' in page

    sent = []
    monkeypatch.setattr(web, "send_transactional_email", lambda to, subject, body: sent.append((to, body)) or True)
    done = admin.post("/admin/plans/grant", data={"account_id": account_id, "plan_key": "athlete_pro", "days": "30"},
                      follow_redirects=False)
    assert done.status_code == 303 and "tool=plan_granted" in done.headers["location"]
    row = db.get_account(account_id)
    assert row["plan_override"] == "athlete_pro"
    ends = datetime.fromisoformat(row["plan_override_until"])
    assert timedelta(days=29, hours=23) < ends - datetime.now(timezone.utc) <= timedelta(days=30)
    assert db._effective_plan_key(row) == "athlete_pro"
    assert sent and sent[0][0] == email and "Athlete Pro" in sent[0][1] and "until" in sent[0][1]
    # Decided, so no longer waiting, and the pricing page no longer says "Noted".
    assert all(r["email"] != email for r in db.pending_plan_requests())
    assert "athlete_pro" not in db.plans_wanted_by(account_id)
    # The person card shows it, with its end date, and it can be ended early.
    card = admin.get(f"/admin?q={email}").text
    assert "given here, until" in card
    admin.post("/admin/plans/end", data={"account_id": account_id})
    assert db._effective_plan_key(db.get_account(account_id)) == "free"


def test_a_no_leaves_the_plan_alone_and_asking_again_reopens_it(admin):
    account_id, email = _person()
    _client(account_id).post("/plan-interest/coach_5", data={"next_path": "/pricing"})
    admin.post("/admin/plans/decline", data={"account_id": account_id, "plan_key": "coach_5"})
    assert db._effective_plan_key(db.get_account(account_id)) == "free"
    assert all(r["email"] != email for r in db.pending_plan_requests())
    _client(account_id).post("/plan-interest/coach_5", data={"next_path": "/pricing"})
    assert any(r["email"] == email for r in db.pending_plan_requests())


def test_an_ended_plan_is_tidied_away_and_no_end_date_stays():
    ended, _ = _person()
    kept, _ = _person()
    past = datetime.now(timezone.utc) - timedelta(days=2)
    db.grant_plan(ended, "gym", 1, now=past)
    db.grant_plan(kept, "gym", None)
    assert db._effective_plan_key(db.get_account(ended)) == "free"
    assert db.clear_expired_plan_overrides() >= 1
    assert db.get_account(ended)["plan_override"] is None
    assert db.get_account(kept)["plan_override"] == "gym" and db.get_account(kept)["plan_override_until"] is None


def test_only_an_admin_can_give_and_bad_choices_change_nothing(admin):
    account_id, _ = _person()
    stranger = _client(_person("stranger")[0])
    for path, data in (("/admin/plans/grant", {"account_id": account_id, "plan_key": "gym", "days": ""}),
                       ("/admin/plans/decline", {"account_id": account_id, "plan_key": "gym"}),
                       ("/admin/plans/end", {"account_id": account_id})):
        assert stranger.post(path, data=data).status_code == 404
    assert db.get_account(account_id)["plan_override"] is None
    for data in ({"plan_key": "free", "days": "30"}, {"plan_key": "nope", "days": "30"},
                 {"plan_key": "gym", "days": "9999"}):
        refused = admin.post("/admin/plans/grant", data={"account_id": account_id, **data}, follow_redirects=False)
        assert "tool=plan_invalid" in refused.headers["location"]
    assert db.get_account(account_id)["plan_override"] is None
