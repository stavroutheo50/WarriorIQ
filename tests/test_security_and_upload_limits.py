import asyncio
import inspect
import json
import uuid
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from browser_client import BrowserClient
from starlette.requests import Request

import app.main as web
from app import state
from core import db, upload_security
from core.auth import issue_session
from core.config import SETTINGS, OUTPUTS
from core.types import VideoInfo


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
def signed_in(settings):
    settings(public_base_url="")
    account = db.create_account(f"audit-{uuid.uuid4().hex}@example.test", "not-a-login-hash")
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(account["id"]))
    yield account, client
    client.close()


def test_poisoned_hosts_are_rejected_before_reset_email(monkeypatch, settings):
    settings(public_base_url="https://warrioriq.eu")
    sent = []
    monkeypatch.setattr(web, "send_transactional_email", lambda *args: sent.append(args))
    with BrowserClient(web.app, base_url="https://warrioriq.eu") as client:
        for header in ("host", "x-forwarded-host"):
            response = client.post("/forgot-password", data={"email": "victim@example.test"},
                                   headers={header: "attacker.example"}, follow_redirects=False)
            assert response.status_code == 400
    assert sent == []


def test_reset_email_uses_configured_origin_even_on_allowed_alias(monkeypatch, settings):
    settings(public_base_url="https://warrioriq.eu", allowed_hosts=("testserver", "internal.example"))
    account = db.create_account(f"reset-{uuid.uuid4().hex}@example.test", "not-a-login-hash")
    sent = []
    monkeypatch.setattr(web, "send_transactional_email", lambda *args: sent.append(args) or True)
    with BrowserClient(web.app, base_url="https://internal.example") as client:
        response = client.post("/forgot-password", data={"email": account["email"]})
        assert response.status_code == 200
    assert "https://warrioriq.eu/reset-password/" in sent[0][2]
    assert "internal.example" not in sent[0][2]


def _stub_video_probe(monkeypatch):
    monkeypatch.setattr(web, "looks_like_video", lambda path: True)
    monkeypatch.setattr(web, "scan_upload", lambda path: {"clean": True, "status": "skipped"})
    monkeypatch.setattr(web, "normalize_container", lambda path: None)
    monkeypatch.setattr(web, "get_video_info", lambda path: VideoInfo(str(path), 25, 250, 64, 64, 10))
    monkeypatch.setattr(web, "probe_upload", lambda *args: (0, []))
    monkeypatch.setattr(web, "inspect_video_quality", lambda *args: {})
    monkeypatch.setattr(web, "read_frame", lambda *args: np.zeros((64, 64, 3), dtype=np.uint8))
    monkeypatch.setattr(web, "_record_shot_profile", lambda *args: None)


def _upload(client, **extra):
    return client.post("/upload", files={"video": ("private.mp4", b"small test video", "video/mp4")},
                       data={"rights_confirmed": "true", "people_permissions_confirmed": "true",
                             "minor_permission_status": "no_minors", **extra},
                       headers={"accept": "application/json"}, follow_redirects=False)


def test_upload_is_local_by_default_and_reserves_allowance_before_selection(signed_in, monkeypatch):
    account, client = signed_in
    _stub_video_probe(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-configuration-not-a-real-key")
    response = _upload(client)
    assert response.status_code == 201, response.text
    job = state.get_job(response.json()["job_id"])
    assert job["usage_reserved"]
    assert not job["openai_identity_recovery"]
    assert not job["external_ai_opted_in"]
    assert db.analysis_allowance(account["id"])["remaining"] == 0
    assert _upload(client).status_code == 429


def test_minors_external_processing_requires_separate_consent(signed_in, monkeypatch):
    account, client = signed_in
    _stub_video_probe(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-configuration-not-a-real-key")
    response = _upload(client, minor_permission_status="guardian_authorized", openai_identity_recovery="true")
    assert response.status_code == 400
    assert db.analysis_allowance(account["id"])["remaining"] == 1
    accepted = _upload(client, minor_permission_status="guardian_authorized",
                       openai_identity_recovery="true", external_ai_guardian_permission="true")
    assert accepted.status_code == 201, accepted.text
    assert state.get_job(accepted.json()["job_id"])["external_ai_opted_in"]
    records = db.list_legal_acceptances(profile_id=account["profile_id"])
    consent = next(row for row in records if row["kind"] == "external_ai_frame_processing")
    assert consent["metadata"]["explicit_opt_in"]
    assert consent["metadata"]["guardian_permission"]


def test_failed_upload_releases_allowance_and_files(signed_in):
    account, client = signed_in
    before = set(web.UPLOADS.iterdir())
    response = _upload(client)
    assert response.status_code == 400
    assert db.analysis_allowance(account["id"])["remaining"] == 1
    assert set(web.UPLOADS.iterdir()) == before


def test_storage_admission_serializes_concurrent_requests(signed_in, settings):
    account, _ = signed_in
    settings(max_pending_uploads=1)
    ids = [uuid.uuid4().hex[:12] for _ in range(2)]

    def reserve(job_id):
        try:
            upload_security.reserve_upload_storage(account["id"], job_id, 64)
            return True
        except upload_security.UploadCapacityError:
            return False

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(reserve, ids)) == [False, True]
    finally:
        for job_id in ids:
            upload_security.release_upload_storage(job_id)


def test_storage_admission_checks_account_and_disk_limits(signed_in, settings, monkeypatch):
    account, _ = signed_in
    settings(account_storage_bytes=100)
    with pytest.raises(upload_security.UploadCapacityError) as caught:
        upload_security.reserve_upload_storage(account["id"], "oversize", 51)
    assert caught.value.status == 413
    settings(account_storage_bytes=10**9)
    monkeypatch.setattr(upload_security.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(upload_security.UploadCapacityError) as caught:
        upload_security.reserve_upload_storage(account["id"], "no-space", 1)
    assert caught.value.status == 507


def test_chunked_multipart_body_limit_closes_parser_and_returns_413(settings):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    settings(max_upload_bytes=32, max_fight_bytes=32)

    async def endpoint(request):
        async with request.form() as form:
            return JSONResponse({"received": len(form)})

    app = upload_security.UploadBodyLimitMiddleware(Starlette(routes=[Route("/upload", endpoint, methods=["POST"])]))

    async def exercise():
        chunks = iter([b'--boundary\r\nContent-Disposition: form-data; name="video"; filename="v.mp4"\r\n\r\n', b"x" * (1024 * 1024 + 64)])
        messages = []

        async def receive():
            return {"type": "http.request", "body": next(chunks, b""), "more_body": True}

        async def send(message):
            messages.append(message)

        await app({"type": "http", "path": "/upload", "method": "POST", "query_string": b"",
                   "headers": [(b"content-type", b"multipart/form-data; boundary=boundary")]}, receive, send)
        assert messages[0]["status"] == 413

    asyncio.run(exercise())


def test_reanalysis_hides_canonical_report_and_tracking_routes(signed_in):
    account, client = signed_in
    job_id = uuid.uuid4().hex[:12]
    state.create_job(job_id, {"account_id": account["id"], "profile_id": account["profile_id"],
                              "owner_key": f"account:{account['id']}", "status": "complete"})
    folder = OUTPUTS / job_id
    (folder / "report.json").write_text(json.dumps({"old": True}))
    (folder / "tracking.jsonl").write_text('{"time_seconds": 0}\n')
    state.prepare_job_run(job_id, {})
    for url in (f"/result/{job_id}", f"/replay/{job_id}", f"/api/tracking/{job_id}"):
        assert client.get(url).status_code == 404
    assert (folder / "report.json").exists(), "Previous artifacts should be preserved, not destroyed."


def test_tracking_stream_keeps_same_frames_with_bounded_memory(tmp_path):
    path = tmp_path / "tracking.jsonl"
    row = json.dumps({"time_seconds": 1, "keypoints": [[0.1, 0.2]] * 170}).encode()
    with path.open("wb") as handle:
        for _ in range(5000):
            handle.write(row + b"\n")
    tracemalloc.start()
    total = sum(len(chunk) for chunk in web._tracking_response_chunks(path))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert total > 10_000_000
    assert peak < 1024 * 1024
    path.write_bytes(b'{"time_seconds":1}\nbroken\n{"time_seconds":2}\n')
    assert json.loads(b"".join(web._tracking_response_chunks(path))) == {
        "frames": [{"time_seconds": 1}, {"time_seconds": 2}],
    }


def test_replay_rejects_tracking_from_a_different_run(signed_in):
    account, client = signed_in
    job_id = uuid.uuid4().hex[:12]
    state.create_job(job_id, {"owner_key": f"account:{account['id']}"})
    run_id = state.prepare_job_run(job_id, {})
    assert state.start_job_run(job_id, "test-worker", run_id)
    directory = state.analysis_run_directory(job_id, run_id)
    directory.mkdir(parents=True)
    tracking = directory / "tracking.jsonl"
    tracking.write_text('{"time_seconds":1}\n')
    assert state.finalize_job_from_worker(job_id, "test-worker", run_id, {}, {"tracking.jsonl": tracking})
    assert client.get(f"/api/tracking/{job_id}?run=previous").status_code == 409
    assert client.get(f"/api/tracking/{job_id}?run={run_id}").json()["frames"] == [{"time_seconds": 1}]


def test_infrastructure_reaches_the_app_by_address_but_not_by_a_foreign_name(settings):
    settings(public_base_url="https://warrioriq.eu", allowed_hosts=())
    # A shared host's health check arrives with an address in Host. Refusing it
    # takes the whole site down for a header nobody chose.
    assert web._trusted_request_host("203.0.113.10")
    assert web._trusted_request_host("203.0.113.10:8000")
    assert web._trusted_request_host("[::1]:8000")
    assert web._trusted_request_host("warrioriq.eu")
    assert not web._trusted_request_host("evil.example.com")
    assert not web._trusted_request_host("warrioriq.eu.evil.example.com")
    assert not web._trusted_request_host("")


def test_a_refused_host_is_named_in_the_log_rather_than_a_blank_400(settings, caplog):
    settings(public_base_url="https://warrioriq.eu", allowed_hosts=())
    with BrowserClient(web.app) as client, caplog.at_level("WARNING", logger="warrioriq"):
        response = client.get("/", headers={"host": "evil.example.com"})
    assert response.status_code == 400
    assert any("evil.example.com" in record.getMessage() for record in caplog.records)


def test_job_lock_is_reentrant_within_one_thread():
    # The guard is an RLock, so a nested entry passes it; taking the file lock
    # a second time on a fresh descriptor would deadlock instead of nesting.
    job_id = uuid.uuid4().hex[:12]
    state.create_job(job_id, {})
    with ThreadPoolExecutor(max_workers=1) as pool:
        def nested():
            with state._job_lock(job_id):
                with state._job_lock(job_id):
                    return state.get_job(job_id) is not None
        assert pool.submit(nested).result(timeout=20)


def test_one_busy_job_does_not_block_state_access_to_another():
    # The process-wide lock guards the in-memory mirror only. Holding it across
    # the per-job file lock is what turns one stalled worker into a stalled site.
    busy, other = uuid.uuid4().hex[:12], uuid.uuid4().hex[:12]
    state.create_job(busy, {})
    state.create_job(other, {"status": "selecting"})
    held = threading.Event()
    release = threading.Event()

    def hold():
        with state._job_lock(busy):
            held.set()
            release.wait(20)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    try:
        assert held.wait(20)
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lambda: state.get_job(other)).result(timeout=10)["status"] == "selecting"
    finally:
        release.set()
        holder.join(timeout=20)


def test_upload_admission_follows_the_route_under_a_mount_prefix():
    # The body-limit middleware, the admission middleware and the router have
    # to agree on which request this is, or the handler is reached with nothing
    # reserved. Matching happens after the mount prefix, not on a literal path.
    under_prefix = {"type": "http", "method": "POST", "path": "/app/upload", "root_path": "/app"}
    assert upload_security.is_fight_upload(under_prefix)
    assert upload_security.is_fight_upload({"type": "http", "method": "POST", "path": "/upload"})
    assert not upload_security.is_fight_upload({"type": "http", "method": "GET", "path": "/upload"})
    assert not upload_security.is_fight_upload({"type": "http", "method": "POST", "path": "/uploads"})
    assert not upload_security.is_fight_upload({"type": "websocket", "path": "/upload"})


def test_upload_refuses_rather_than_crashing_when_admission_did_not_run(signed_in, monkeypatch, caplog):
    # Without admission there is no reserved allowance and no storage lease, so
    # accepting the footage is worse than refusing it. It used to raise
    # AttributeError reaching for the job id admission never set.
    account, client = signed_in
    monkeypatch.setattr(web, "is_fight_upload", lambda scope: False)
    with caplog.at_level("ERROR", logger="warrioriq"):
        response = client.post(
            "/upload",
            data={"ruleset": "K1", "fight_type": "kickboxing", "rights_confirmed": "true",
                  "people_permissions_confirmed": "true", "minor_permission_status": "no_minors"},
            files={"video": ("fight.mp4", b"not-a-real-video", "video/mp4")},
        )
    assert response.status_code == 500
    assert any("fight_upload_admission_skipped" in r.getMessage() for r in caplog.records)
    # Nothing was taken on the way out.
    with db.connection() as con:
        assert con.execute("SELECT COUNT(*) FROM upload_leases").fetchone()[0] == 0


def test_upload_admission_releases_both_reservations_without_awaiting(signed_in):
    # The releases sit in the finally of an async function. Anything awaited
    # there is skipped when the request is cancelled, and a surviving lease
    # locks the account out of uploading until it expires.
    source = inspect.getsource(web._admit_fight_upload)
    cleanup = "".join(
        line for line in source.split("finally:", 1)[1].splitlines(keepends=True)
        if not line.lstrip().startswith("#")
    )
    assert "release_upload_storage(job_id)" in cleanup
    assert "release_analysis(account_id, job_id)" in cleanup
    assert "await" not in cleanup


def test_abandoned_upload_lease_does_not_outlive_its_request(signed_in):
    account, client = signed_in
    with db.connection() as con:
        before = con.execute("SELECT COUNT(*) FROM upload_leases WHERE account_id=?", (account["id"],)).fetchone()[0]
    # No video, so the handler rejects the request after admission reserved.
    assert client.post("/upload", data={"ruleset": "K1"}).status_code >= 400
    with db.connection() as con:
        after = con.execute("SELECT COUNT(*) FROM upload_leases WHERE account_id=?", (account["id"],)).fetchone()[0]
        usage = con.execute("SELECT COUNT(*) FROM analysis_usage WHERE account_id=?", (account["id"],)).fetchone()[0]
    assert after == before
    assert usage == 0
