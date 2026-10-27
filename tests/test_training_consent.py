"""The optional "Help WarriorIQ get better" box (2026-10-27).

Off unless ticked, never for footage with minors, kept apart from the 30-day
video retention, deleted with the fight, its video or the account, and only
readable with the worker token.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import training_consent

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(training_consent, "FOLDER", tmp_path / "consented")
    return tmp_path


def _clip(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    for index in range(60):
        image = np.full((240, 320, 3), (60, 120, 90), np.uint8)
        cv2.rectangle(image, (40 + index, 60), (100 + index, 200), (200, 80, 40), -1)
        writer.write(image)
    writer.release()
    return path


def test_a_kept_fight_survives_the_upload_being_deleted_and_goes_with_forget(folder):
    video = folder / "upload.mp4"
    video.write_bytes(b"video bytes")
    run = folder / "run"
    run.mkdir()
    (run / "report.json").write_text("{}")
    (run / "tracking.jsonl").write_text("{}\n")
    (run / "overlay.mp4").write_text("not kept")
    assert training_consent.keep("abc123def456", video, run, {"ruleset": "KICK_LIGHT"})
    video.unlink()                                     # the 30-day retention
    kept = training_consent.kept()
    assert len(kept) == 1 and kept[0]["ruleset"] == "KICK_LIGHT"
    assert set(kept[0]["files"]) == {"video.mp4", "report.json", "tracking.jsonl"}
    assert training_consent.kept_file("abc123def456", "video.mp4").read_bytes() == b"video bytes"
    training_consent.forget("abc123def456")
    assert training_consent.kept() == []


def test_only_its_own_files_are_ever_served(folder):
    assert training_consent.kept_file("../../etc", "passwd") is None
    assert training_consent.kept_file("abc123def456", "../report.json") is None
    assert training_consent.kept_file("abc123def456", "consent.json") is None


def test_the_box_is_on_the_upload_page_unticked_and_sent_by_the_chunked_upload():
    page = (ROOT / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
    assert '<input type="checkbox" name="training_consent" value="true">' in page
    assert "'external_ai_guardian_permission', 'training_consent'" in page
    assert "Optional \"Help WarriorIQ get better\"" in (ROOT / "app" / "templates" / "privacy.html").read_text(encoding="utf-8")


@pytest.fixture
def signed_in():
    import app.main as web
    from browser_client import BrowserClient
    from core import db
    from core.auth import issue_session

    account = db.create_account(f"consent-{uuid.uuid4().hex}@example.test", "not-a-login-hash")
    db.set_plan_override(int(account["id"]), "gym")
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(account["id"]))
    web._rate_windows.clear()
    yield web, client
    client.close()


@pytest.mark.parametrize("ticked,minors,kept", [
    (True, "no_minors", True),
    (False, "no_minors", False),
    (True, "guardian_authorized", False),      # never for footage with minors
])
def test_the_upload_records_the_choice(signed_in, tmp_path, ticked, minors, kept):
    web, client = signed_in
    data = {"rights_confirmed": "true", "people_permissions_confirmed": "true",
            "minor_permission_status": minors}
    if ticked:
        data["training_consent"] = "true"
    with _clip(tmp_path / "fight.mp4").open("rb") as handle:
        response = client.post("/upload", headers={"Accept": "application/json"},
                               files={"video": ("fight.mp4", handle, "video/mp4")}, data=data)
    assert response.status_code == 201, response.text
    job_id = response.json()["job_id"]
    assert web.get_job(job_id)["training_consent"] is kept


def test_completing_a_consented_analysis_keeps_it(folder, monkeypatch):
    from app import state

    job_id = "c0" + uuid.uuid4().hex[:10]
    video = folder / "fight.mp4"
    video.write_bytes(b"v")
    run_id = uuid.uuid4().hex
    run_dir = state.analysis_run_directory(job_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps({"ok": True}))
    state.create_job(job_id, {"status": "complete", "artifacts_run_id": run_id, "persist_result": True,
                              "history_saved": False, "training_consent": True, "video_path": str(video),
                              "ruleset": "K1"})
    monkeypatch.setattr("core.db.save_completed_analysis", lambda *args: None)
    try:
        assert state.persist_completed_job(job_id, run_id)
        assert training_consent.kept_file(job_id, "video.mp4") is not None
    finally:
        state.delete_job(job_id)


def test_deleting_the_fight_files_deletes_the_kept_copy(monkeypatch):
    import app.main as web

    forgotten = []
    monkeypatch.setattr(training_consent, "forget", forgotten.append)
    web._remove_fight_files({"job_id": "abc123def456", "video_path": "/nowhere/fight.mp4"})
    assert forgotten == ["abc123def456"]


def test_the_kept_fights_need_the_worker_token():
    import app.main as web
    from browser_client import BrowserClient

    response = BrowserClient(web.app).get("/api/worker/consented")
    assert response.status_code in {401, 503}
    assert "fights" not in response.text
