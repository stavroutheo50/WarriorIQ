import asyncio
import os
import time
from unittest.mock import Mock

import app.main as web
from app import state
from browser_client import BrowserClient
from core import db, retention


def _completed_job(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    uploads = tmp_path / "uploads"
    outputs.mkdir()
    uploads.mkdir()
    for module in (state, web, retention):
        monkeypatch.setattr(module, "OUTPUTS", outputs)
    for module in (web, retention):
        monkeypatch.setattr(module, "UPLOADS", uploads)
    monkeypatch.setattr(state, "_jobs", {})
    video = uploads / "history-test.mp4"
    video.write_bytes(b"private test footage")
    state.create_job("history-test", {"persist_result": True, "video_path": str(video)})
    run = state.prepare_job_run("history-test", {})
    assert state.start_job_run("history-test", "test-worker", run)
    assert state.finalize_job_from_worker("history-test", "test-worker", run, {}, {})
    return run, video


def test_completion_is_marked_pending_before_history_save(monkeypatch, tmp_path):
    _completed_job(monkeypatch, tmp_path)
    # The process can stop immediately after publishing; cleanup must still
    # know the report has not reached the history database.
    assert state.get_job("history-test")["history_saved"] is False


def test_cleanup_preserves_completed_report_waiting_for_history(monkeypatch, tmp_path):
    run, video = _completed_job(monkeypatch, tmp_path)
    folder = state.OUTPUTS / "history-test"
    old = time.time() - 24 * 3600
    os.utime(video, (old, old))
    os.utime(folder, (old, old))
    monkeypatch.setattr(web, "_last_guest_cleanup", 0)
    monkeypatch.setattr(web, "_last_saved_video_cleanup", 0)
    with BrowserClient(web.app) as client:
        assert client.get("/health").status_code == 200
    assert video.is_file()
    assert (state.analysis_run_directory("history-test", run) / "report.json").is_file()
    assert state.get_job("history-test")["history_saved"] is False


def test_repeated_worker_completion_retries_history_once(monkeypatch, tmp_path):
    run, _ = _completed_job(monkeypatch, tmp_path)
    monkeypatch.setattr(web, "_require_remote_worker", lambda request: None)
    save = Mock()
    monkeypatch.setattr(db, "save_completed_analysis", save)
    for _ in range(2):
        response = asyncio.run(web.remote_worker_complete(
            None, "history-test", None, "test-worker", run,
        ))
        assert response["already_complete"]
    save.assert_called_once()
    assert state.get_job("history-test")["history_saved"] is True
