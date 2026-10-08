"""Abandoned uploads must never lock an account out of uploading.

QA, 2026-09: uploads that were begun and never finished counted as "pending",
so after a couple of them every new upload answered 429 "Finish your pending
fight selections or analyses" - with nothing pending anywhere on screen - until
the leases expired half an hour later.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

import app.main as web
from app import state
from browser_client import BrowserClient
from core import chunked_upload, db, upload_security
from core.auth import issue_session
from core.config import SETTINGS


@pytest.fixture
def settings():
    previous = {}

    def set_values(**values):
        for key, value in values.items():
            previous.setdefault(key, getattr(SETTINGS, key))
            object.__setattr__(SETTINGS, key, value)

    yield set_values
    for key, value in previous.items():
        object.__setattr__(SETTINGS, key, value)


@pytest.fixture
def account(settings):
    settings(chunked_upload_enabled=True, public_base_url="", max_pending_uploads=2)
    created = db.create_account(f"pending-{uuid.uuid4().hex}@example.test", "not-a-login-hash")
    # Unlimited analyses, so the allowance is not what refuses a third upload.
    db.set_plan_override(int(created["id"]), "gym")
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    yield created, client
    web._rate_windows.clear()
    client.close()


def _begin(client) -> object:
    return client.post("/api/upload/begin", json={
        "filename": "fight.mp4", "size": 2_000_000,
        "rights_confirmed": True, "people_permissions_confirmed": True,
        "minor_permission_status": "no_minors",
    })


def _selecting_job(created) -> str:
    job_id = uuid.uuid4().hex[:12]
    state.create_job(job_id, {
        "status": "selecting", "account_id": int(created["id"]),
        "owner_key": f"account:{created['id']}", "original_name": "round-one.mp4",
        "ruleset": "K1", "usage_reserved": False,
    })
    return job_id


def test_unfinished_uploads_do_not_count_as_pending(account):
    created, client = account
    opened = [_begin(client) for _ in range(4)]
    try:
        assert [r.status_code for r in opened] == [201, 201, 201, 201], [r.text for r in opened]
    finally:
        for response in opened:
            if response.status_code == 201:
                client.post(f"/api/upload/{response.json()['job_id']}/abort")


def test_an_idle_session_gives_back_its_lease_reservation_and_file(account):
    created, client = account
    job = _begin(client).json()["job_id"]
    assert db.analysis_allowance(int(created["id"]))["used"] == 1
    old = time.time() - 120
    for path in (chunked_upload.part_path(job), chunked_upload.meta_path(job)):
        os.utime(path, (old, old))
    released = chunked_upload.release_idle_sessions(int(created["id"]))
    assert released == [job]
    assert not chunked_upload.part_path(job).exists()
    assert not chunked_upload.meta_path(job).exists()
    assert db.analysis_allowance(int(created["id"]))["used"] == 0
    with db.connection() as con:
        assert con.execute("SELECT 1 FROM upload_leases WHERE job_id=?", (job,)).fetchone() is None


def test_a_session_still_receiving_bytes_is_left_alone(account):
    created, client = account
    job = _begin(client).json()["job_id"]
    try:
        chunked_upload.part_path(job).write_bytes(b"x" * 1024)
        assert chunked_upload.release_idle_sessions(int(created["id"])) == []
        assert chunked_upload.meta_path(job).exists()
    finally:
        client.post(f"/api/upload/{job}/abort")


def test_real_waiting_fights_count_and_cancelling_one_frees_the_slot(account):
    created, client = account
    jobs = [_selecting_job(created) for _ in range(2)]
    try:
        with pytest.raises(upload_security.UploadCapacityError) as refused:
            upload_security.reserve_upload_storage(int(created["id"]), uuid.uuid4().hex[:12], 64)
        assert refused.value.status == 429
        assert "Pending" in str(refused.value)

        library = client.get("/history")
        assert library.status_code == 200
        assert 'id="pending"' in library.text
        for job_id in jobs:
            assert f"/pending/{job_id}/cancel" in library.text
            assert f"/select/{job_id}" in library.text
        assert "Waiting for you to pick the fighters" in library.text

        cancelled = client.post(f"/pending/{jobs[0]}/cancel", follow_redirects=False)
        assert cancelled.status_code == 303
        assert cancelled.headers["location"] == "/history?cancelled=1#library-notice"
        assert state.get_job(jobs[0]) is None

        probe = uuid.uuid4().hex[:12]
        upload_security.reserve_upload_storage(int(created["id"]), probe, 64)
        upload_security.release_upload_storage(probe)
    finally:
        for job_id in jobs:
            state.delete_job(job_id)


def test_someone_else_cannot_cancel_your_fight(account):
    created, _ = account
    job_id = _selecting_job(created)
    stranger = db.create_account(f"stranger-{uuid.uuid4().hex}@example.test", "not-a-login-hash")
    try:
        with BrowserClient(web.app) as other:
            other.cookies.set(web.SESSION_COOKIE, issue_session(stranger["id"]))
            assert other.post(f"/pending/{job_id}/cancel").status_code == 404
        assert state.get_job(job_id) is not None
    finally:
        state.delete_job(job_id)


def test_cancelling_a_running_analysis_stops_the_worker(account):
    created, client = account
    job_id = _selecting_job(created)
    run = state.prepare_job_run(job_id, {})
    assert state.start_job_run(job_id, "worker-1", run)
    library = client.get("/history").text
    assert "Being analysed" in library
    assert client.post(f"/pending/{job_id}/cancel", follow_redirects=False).status_code == 303
    # The worker's next progress report is refused, which is what ends its run.
    assert state.update_job_for_worker(job_id, "worker-1", run, {"percent": 50.0}) is False


def test_the_library_does_not_label_every_fight_competition(account):
    """QA 2026-09: the question is no longer asked, but every fight said "Competition"."""
    created, client = account
    profile_id = int(created["profile_id"])
    stamp = "2999-01-01T00:00:00+00:00"
    db.save_fight(f"labelcomp{uuid.uuid4().hex[:6]}", profile_id, "a.mp4", "", "", "competition",
                  "K1", "A", {}, stamp)
    db.save_fight(f"labelspar{uuid.uuid4().hex[:6]}", profile_id, "b.mp4", "", "", "sparring",
                  "K1", "A", {}, stamp)
    library = client.get("/history").text
    assert "<span>Competition</span>" not in library
    assert library.count("<span>Sparring</span>") == 1
