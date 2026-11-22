"""Where new accounts stop (owner, 2026-11-22: 8 sign-ups in a week, 2 analysed a fight).

Each step on the way to a first report is recorded once per account, every
refusal with the reason the person was shown, and the admin page reads the
furthest step back for each new account.
"""

from __future__ import annotations

import json
import uuid

import app.main as web
from app import admin_overview
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session
from core.config import SETTINGS


def _account():
    email = f"journey-{uuid.uuid4().hex[:10]}@example.test"
    created = db.create_account(email, hash_password("Strong-Local-Password"))
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    return created, email, client


def _events(account_id):
    with db.connection() as con:
        return [(row[0], json.loads(row[1])) for row in con.execute(
            "SELECT event_type, metadata_json FROM security_events WHERE account_id=? AND event_type LIKE 'journey_%' "
            "ORDER BY id", (account_id,))]


def test_opening_the_upload_page_is_recorded_once():
    created, _, client = _account()
    for _ in range(3):
        assert client.get("/analyze").status_code == 200
    assert [e[0] for e in _events(created["id"])] == ["journey_upload_page"]


def test_a_refused_upload_is_recorded_with_the_reason_shown():
    created, _, client = _account()
    refused = client.post("/upload", files={"video": ("notes.txt", b"not a video", "text/plain")},
                          data={"ruleset": "K1"})
    assert refused.status_code >= 400
    events = _events(created["id"])
    assert events and events[-1][0] == "journey_upload_refused"
    assert events[-1][1]["status"] == refused.status_code and events[-1][1].get("reason")


def test_a_guest_is_never_recorded():
    BrowserClient(web.app).get("/analyze")
    with db.connection() as con:
        assert con.execute("SELECT COUNT(*) FROM security_events WHERE account_id IS NULL "
                           "AND event_type LIKE 'journey_%'").fetchone()[0] == 0


def test_the_journey_steps_are_classified_by_route_and_outcome():
    step = web._journey_event
    assert step("GET", "/analyze/boxing", 200) == "journey_upload_page"
    assert step("GET", "/select/abc", 200) == "journey_fighter_picker"
    assert step("POST", "/upload", 303) == "journey_video_uploaded"
    assert step("POST", "/api/upload/abc/finish", 200) == "journey_video_uploaded"
    assert step("POST", "/api/upload/begin", 200) is None
    assert step("POST", "/api/upload/begin", 413) == "journey_upload_refused"
    assert step("POST", "/api/start/abc", 202) == "journey_fighters_picked"
    assert step("POST", "/api/start/abc", 400) == "journey_start_refused"
    assert step("GET", "/history", 200) is None


def test_the_admin_report_shows_each_new_account_and_where_it_stopped():
    stuck, stuck_email, client = _account()
    client.get("/analyze")
    web._record_journey(int(stuck["id"]), "journey_upload_refused", 413, "That file is too large.")
    done, done_email, _ = _account()
    db.save_fight(f"jr{uuid.uuid4().hex[:8]}", int(done["profile_id"]), "f.mp4", "", "", "sparring", "K1", "A", {})
    waiting, waiting_email, _ = _account()
    jobs = [("jobw", {"owner_key": f"account:{waiting['id']}", "status": "queued"})]

    report = admin_overview.onboarding(jobs)
    people = {p["email"]: p for p in report["people"]}
    assert people[stuck_email]["furthest_key"] == "upload_page"
    assert people[stuck_email]["next_step"] == "Uploaded a video"
    assert people[stuck_email]["problem"]["reason"] == "That file is too large."
    assert people[done_email]["furthest_key"] == "finished"
    assert people[waiting_email]["furthest_key"] == "fighters_picked"
    assert people[waiting_email]["now"] == "waiting for the analysis computer"
    funnel = {step["key"]: step["count"] for step in report["funnel"]}
    assert funnel["signed_up"] >= 3 and funnel["upload_page"] >= 3 and funnel["finished"] >= 1
    assert funnel["signed_up"] >= funnel["upload_page"] >= funnel["finished"]

    # Shown to an administrator on the admin page.
    previous = SETTINGS.admin_emails
    object.__setattr__(SETTINGS, "admin_emails", (stuck_email.lower(),))
    try:
        page = client.get("/admin").text
    finally:
        object.__setattr__(SETTINGS, "admin_emails", previous)
    assert "Where new accounts stop" in page and "That file is too large." in page


def test_a_failed_analysis_is_put_against_its_account():
    created, _, _ = _account()
    web._journey_failed({"owner_key": f"account:{created['id']}"}, "decode_failed", "jobx")
    web._journey_failed({"owner_key": "guest:abc"}, "decode_failed", "joby")
    assert _events(created["id"]) == [("journey_analysis_failed", {"reason": "decode_failed"})]
