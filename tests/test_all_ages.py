"""Accounts for every age: under the minimum age a guardian approves by email.

Requested 2026-10: WarriorIQ was adults only. A young fighter can now sign up;
a parent or guardian named at sign-up is emailed a signed link, and until they
approve, the account cannot upload a fight.
"""

from __future__ import annotations

import re
import uuid
from unittest.mock import patch

import pytest

import app.main as web
from browser_client import BrowserClient
from core import db


@pytest.fixture
def client():
    web._rate_windows.clear()
    with BrowserClient(web.app) as browser:
        yield browser
    web._rate_windows.clear()


def _csrf(client, path="/signup"):
    page = client.get(path).text
    return re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)


def _signup(client, email, **extra):
    data = {"email": email, "password": "Strong-Local-Password", "accept_terms": "true",
            "csrf_token": _csrf(client)}
    data.update(extra)
    return client.post("/signup", data=data, follow_redirects=False)


def test_an_adult_signs_up_as_before(client):
    email = f"adult-{uuid.uuid4().hex}@example.test"
    response = _signup(client, email, age_group="adult")
    assert response.status_code == 303 and response.headers["location"] == "/dashboard"
    assert db.get_account_by_email(email)["guardian_approval_status"] == "not_applicable"
    # The old checkbox still means an adult.
    legacy = f"legacy-{uuid.uuid4().hex}@example.test"
    with BrowserClient(web.app) as other:
        assert _signup(other, legacy, age_confirmed="true").status_code == 303
    assert db.get_account_by_email(legacy)["guardian_approval_status"] == "not_applicable"


def test_under_the_minimum_age_needs_a_guardian_who_is_somebody_else(client):
    email = f"young-{uuid.uuid4().hex}@example.test"
    missing = _signup(client, email, age_group="minor")
    assert missing.status_code == 400 and "parent or guardian has to approve" in missing.text
    same = _signup(client, email, age_group="minor", guardian_name="Mum", guardian_email=email)
    assert "not yours" in same.text
    assert db.get_account_by_email(email) is None


def test_a_young_fighter_waits_for_the_guardian_then_can_upload(client):
    email = f"young-{uuid.uuid4().hex}@example.test"
    sent = {}

    def capture(recipient, subject, body):
        sent.update(recipient=recipient, body=body)
        return True

    with patch.object(web, "send_transactional_email", side_effect=capture):
        response = _signup(client, email, age_group="minor", guardian_name="Alex Parent",
                           guardian_email="parent@example.test")
    assert response.headers["location"] == "/guardian"
    account = db.get_account_by_email(email)
    assert account["guardian_approval_status"] == "pending_guardian"
    assert sent["recipient"] == "parent@example.test"
    link = re.search(r"/guardian/approve/(\S+)", sent["body"]).group(1)

    status = client.get("/guardian").text
    assert "Waiting for approval" in status and "p" in status and "@example.test" in status
    upload = client.get("/analyze/kickboxing").text
    assert "has not approved your account yet" in upload
    refused = client.post("/api/upload/begin", json={"filename": "f.mp4", "size": 2_000_000},
                          headers={"X-CSRF-Token": _csrf(client, "/analyze/kickboxing")})
    assert refused.status_code in {403, 404}  # 404 when chunked upload is switched off
    if refused.status_code == 403:
        assert "parent or guardian" in refused.json()["detail"]
    direct = client.post("/upload", headers={"Accept": "application/json",
                                             "X-CSRF-Token": _csrf(client, "/analyze/kickboxing")},
                         files={"video": ("fight.mp4", b"\x00" * 64, "video/mp4")},
                         data={"rights_confirmed": "true", "people_permissions_confirmed": "true",
                               "minor_permission_status": "no_minors"})
    assert direct.status_code == 403 and "parent or guardian" in direct.json()["detail"]

    with BrowserClient(web.app) as guardian:
        page = guardian.get(f"/guardian/approve/{link}")
        assert page.status_code == 200 and email in page.text
        token = _csrf(guardian, f"/guardian/approve/{link}")
        unticked = guardian.post(f"/guardian/approve/{link}", data={"decision": "approve", "csrf_token": token})
        assert unticked.status_code == 400
        approved = guardian.post(f"/guardian/approve/{link}", data={
            "decision": "approve", "guardian_confirmed": "true", "csrf_token": token})
        assert approved.status_code == 200 and "Account approved." in approved.text
        assert guardian.get("/guardian/approve/not-a-real-token").status_code == 404
    assert db.get_account_by_email(email)["guardian_approval_status"] == "guardian_approved"
    assert web._guardian_hold(db.get_account_by_email(email)) is None
    assert client.get("/guardian", follow_redirects=False).headers["location"] == "/dashboard"


def test_a_declined_account_cannot_upload():
    account = {"guardian_approval_status": "guardian_declined"}
    assert "did not approve" in web._guardian_hold(account)
    assert web._guardian_hold({"guardian_approval_status": "not_applicable"}) is None


def test_social_sign_up_sends_young_fighters_to_the_email_form(client):
    with patch.object(web.SOCIAL_AUTH, "client", return_value=object()):
        page = client.post("/auth/google/start", data={
            "mode": "signup", "accept_terms": "true", "age_group": "minor", "csrf_token": _csrf(client)})
    assert "Create your account with your email below" in page.text


def test_the_terms_and_privacy_policy_say_every_age(client):
    terms = client.get("/terms").text
    assert "People of any age may create an account" in terms
    assert "only for people aged 18 or older" not in terms
    assert "Accounts are open to every age" in client.get("/privacy").text
    signup = client.get("/signup").text
    assert 'name="age_group" value="minor"' in signup and 'id="guardianEmail"' in signup
