"""The admin control room (owner, 2026-11-15).

"Organise the admin panel and make it have the most information it can." The
numbers are counted from the database on the request, so these tests check
that a new account, fight and failure move them, and that the page shows them
to an administrator only.
"""

from __future__ import annotations

import uuid

import app.main as web
from browser_client import BrowserClient
from app import admin_overview
from core import db
from core.auth import hash_password, issue_session
from core.config import SETTINGS


def _new_account():
    email = f"admin-overview-{uuid.uuid4().hex[:10]}@example.test"
    return db.create_account(email, hash_password("Strong-Local-Password")), email


def test_counts_move_with_a_new_account_fight_and_failure():
    before_accounts = admin_overview.accounts()
    before_fights = admin_overview.fights()
    before_failures = admin_overview.failures()

    created, email = _new_account()
    job_id = f"ov{uuid.uuid4().hex[:10]}"
    db.save_fight(job_id, int(created["profile_id"]), "fight.mp4", "", "", "competition", "BOXING", "A", {})
    db.record_analysis_failure(f"ovf{uuid.uuid4().hex[:8]}", "decode_failed", detail="test detail",
                               account_id=int(created["id"]))

    accounts = admin_overview.accounts()
    fights = admin_overview.fights()
    failures = admin_overview.failures()
    assert accounts["total"] == before_accounts["total"] + 1
    assert accounts["new_day"] == before_accounts["new_day"] + 1
    assert accounts["recent"][0]["email"] == email and accounts["recent"][0]["fights"] == 1
    assert fights["total"] == before_fights["total"] + 1
    assert fights["recent"][0]["job_id"] == job_id and fights["recent"][0]["email"] == email
    assert not fights["recent"][0]["video_held"]                      # saved without a video
    assert failures["total_month"] == before_failures["total_month"] + 1
    assert failures["recent"][0]["email"] == email and failures["recent"][0]["reason"] == "decode_failed"


def test_live_jobs_name_the_account_and_never_a_guest_key():
    created, email = _new_account()
    jobs = [("a1", {"status": "running", "owner_key": f"account:{created['id']}", "stage": "analysis",
                    "percent": 42.0, "ruleset": "K1"}),
            ("a2", {"status": "queued", "owner_key": "guest:secret-guest-id"}),
            ("a3", {"status": "complete"})]
    live = admin_overview.live_jobs(jobs)
    assert live["counts"] == {"complete": 1, "queued": 1, "running": 1}
    assert [(job["job_id"], job["email"]) for job in live["active"]] == [("a1", email), ("a2", "guest")]
    assert "secret-guest-id" not in repr(live)


def test_storage_counts_files_and_bytes(tmp_path, monkeypatch):
    uploads, outputs = tmp_path / "uploads", tmp_path / "outputs"
    (outputs / "job1").mkdir(parents=True)
    uploads.mkdir()
    (uploads / "job1.mp4").write_bytes(b"x" * 1000)
    (outputs / "job1" / "report.json").write_bytes(b"y" * 250)
    monkeypatch.setattr(admin_overview, "UPLOADS", uploads)
    monkeypatch.setattr(admin_overview, "OUTPUTS", outputs)
    monkeypatch.setattr(admin_overview, "_storage_cache", {})
    value = admin_overview.storage()
    assert (value["uploads_bytes"], value["uploads_files"]) == (1000, 1)
    assert (value["outputs_bytes"], value["outputs_files"]) == (250, 1)
    assert value["disk_free_bytes"] is not None


def test_the_page_shows_every_section_to_an_admin_only():
    created, email = _new_account()
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    assert client.get("/admin").status_code == 404                    # not an administrator yet
    previous = SETTINGS.admin_emails
    object.__setattr__(SETTINGS, "admin_emails", (email.lower(),))
    try:
        page = client.get("/admin")
        assert page.status_code == 200
        for section in ("adminOverview", "adminAnalysis", "adminPeople", "adminFights", "adminTraffic",
                        "adminSafety", "adminTools"):
            assert f'id="{section}"' in page.text and f'href="#{section}"' in page.text
        assert email in page.text                                       # newest accounts
        assert "Search to suspend or restore an account." in page.text  # no list until searched
        searched = client.get("/admin", params={"q": email}).text
        assert f"/admin/accounts/{created['id']}/status" in searched
    finally:
        object.__setattr__(SETTINGS, "admin_emails", previous)
