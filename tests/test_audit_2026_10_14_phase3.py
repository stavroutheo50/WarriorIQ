"""Audit of 14 Oct 2026, phase 3: server performance.

#3  several server processes, sharing one session secret and a WAL database
#4  static public pages cached for signed-out visitors; no job scan and no
    schema script per request
#25 replay video: range requests, faststart remux, preload, optional CDN
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------- #3
def test_hosted_runs_four_processes_when_analysis_is_remote(monkeypatch):
    import run

    for name in ("RENDER", "PORT", "SERVER_PORT", "WEB_CONCURRENCY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WARRIORIQ_WORKER_MODE", "remote")
    assert run.worker_count() == 1                      # local
    monkeypatch.setenv("RENDER", "true")
    assert run.worker_count() == 4                      # hosted
    monkeypatch.setenv("WEB_CONCURRENCY", "6")
    assert run.worker_count() == 6


def test_in_process_analysis_keeps_one_process(monkeypatch):
    import run

    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    monkeypatch.setenv("WARRIORIQ_WORKER_MODE", "inprocess")
    assert run.worker_count() == 1


def test_render_blueprint_sets_web_concurrency():
    blueprint = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "key: WEB_CONCURRENCY" in blueprint


def test_processes_starting_together_share_one_session_secret(tmp_path, monkeypatch):
    import app.main as web

    monkeypatch.setattr(web, "DATA_ROOT", tmp_path)
    previous = web.SETTINGS.oauth_state_secret
    object.__setattr__(web.SETTINGS, "oauth_state_secret", "")
    try:
        first = web._session_secret()
    finally:
        object.__setattr__(web.SETTINGS, "oauth_state_secret", previous)
    object.__setattr__(web.SETTINGS, "oauth_state_secret", "")
    # A second process arriving after the first wrote it reads, never rewrites.
    try:
        assert web._session_secret() == first
    finally:
        object.__setattr__(web.SETTINGS, "oauth_state_secret", previous)
    assert (tmp_path / "session-secret.txt").read_text(encoding="utf-8").strip() == first


def test_sqlite_runs_in_wal_mode_with_a_busy_timeout():
    from core import db

    db.init_db()
    with db.connection() as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert int(con.execute("PRAGMA busy_timeout").fetchone()[0]) > 0


# ---------------------------------------------------------------------- #4
def test_the_schema_script_runs_once_per_process():
    from core import db

    db.init_db()
    with mock.patch.object(db, "_init_db_schema") as schema:
        db.init_db()
        db.list_fights(1)
    assert schema.call_count == 0


def test_hot_lookups_are_indexed():
    from core import db

    db.init_db()
    with db.connection() as con:
        names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_fights_profile", "idx_sessions_account", "idx_sessions_expires",
            "idx_points_ledger_profile", "idx_camp_missions_profile"} <= names


def test_a_signed_out_visitor_without_an_analysis_cookie_never_scans_jobs():
    import app.main as web

    with mock.patch.object(web, "_active_job_for_owner") as scan:
        web._analysis_navigation_state("guest:abc", None, None)
        assert scan.call_count == 0
        web._analysis_navigation_state("account:1", None, None)
        assert scan.call_count == 1


def test_static_public_pages_are_cached_for_signed_out_visitors():
    from browser_client import BrowserClient

    import app.main as web

    client = BrowserClient(web.app, base_url="https://testserver")
    for path in ("/terms", "/subprocessors", "/"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "private, max-age=300", path
        assert "Cookie" in response.headers.get("vary", "")
    # Never a private page, never shared caches.
    assert "max-age=300" not in client.get("/login").headers.get("cache-control", "")


def test_a_signed_in_visitor_is_not_served_a_cached_public_page(monkeypatch):
    import app.main as web
    from fastapi import Response
    from starlette.requests import Request

    request = Request({"type": "http", "method": "GET", "path": "/terms", "query_string": b"",
                       "headers": [], "scheme": "https", "server": ("testserver", 443)})
    request.state.account = {"id": 1}
    response = Response("<p>x</p>", media_type="text/html")
    assert web._cacheable_public_page(request, response) is False
    request.state.account = None
    assert web._cacheable_public_page(request, response) is True


# --------------------------------------------------------------------- #25
def _clip(path: Path, faststart: bool) -> Path:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=1:size=160x120:rate=10", "-c:v", "libx264",
                    "-movflags", "+faststart" if faststart else "-faststart", str(path)], check=True)
    return path


@pytest.mark.skipif(not Path("/usr/bin/ffmpeg").exists() and not os.environ.get("FFMPEG_BINARY"),
                    reason="needs ffmpeg")
def test_an_mp4_with_its_index_at_the_end_gets_a_faststart_copy(tmp_path):
    from core.video import moov_after_mdat, normalize_container, playback_file

    late = _clip(tmp_path / "late.mp4", faststart=False)
    early = _clip(tmp_path / "early.mp4", faststart=True)
    assert moov_after_mdat(late) is True and moov_after_mdat(early) is False
    derivative = normalize_container(late)
    assert derivative is not None and moov_after_mdat(derivative) is False
    assert playback_file(late) == derivative
    assert normalize_container(early) is None


def test_media_supports_range_requests(tmp_path):
    from fastapi.testclient import TestClient

    import app.main as web

    video = tmp_path / "fight.mp4"
    video.write_bytes(bytes(range(256)) * 40)
    with mock.patch.object(web, "_authorized_job", return_value={"job_id": "j"}), \
            mock.patch.object(web, "get_job", return_value={"video_path": str(video)}):
        response = TestClient(web.app, base_url="https://testserver").get(
            "/media/j", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 0-99/{256 * 40}"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["cache-control"] == "private, max-age=3600"
    assert response.headers.get("etag")


def test_signed_cdn_urls_are_off_by_default_and_verified_when_on(tmp_path):
    from fastapi.testclient import TestClient

    import app.main as web

    client = TestClient(web.app, base_url="https://testserver")
    assert web._signed_media_url("j") is None
    assert client.get("/media-signed/j?exp=9999999999&sig=x").status_code == 404
    video = tmp_path / "fight.mp4"
    video.write_bytes(b"0" * 1000)
    previous = (web.SETTINGS.media_cdn_base_url, web.SETTINGS.media_signing_key)
    object.__setattr__(web.SETTINGS, "media_cdn_base_url", "https://media.warrioriq.eu")
    object.__setattr__(web.SETTINGS, "media_signing_key", "test-key")
    try:
        url = web._signed_media_url("j", now=1_000_000)
        assert url.startswith("https://media.warrioriq.eu/media-signed/j?exp=")
        # Stable within a period, so a CDN copy is reused.
        assert url == web._signed_media_url("j", now=1_000_030)
        path_and_query = "/media-signed/j?" + url.split("?", 1)[1]
        with mock.patch.object(web, "get_job", return_value={"video_path": str(video)}), \
                mock.patch.object(web.time, "time", return_value=1_000_000):
            good = client.get(path_and_query)
            forged = client.get(path_and_query.replace("sig=", "sig=0"))
        assert good.status_code == 200 and good.headers["cache-control"].startswith("public, max-age=")
        assert forged.status_code == 404
        with mock.patch.object(web, "_authorized_job", return_value={"job_id": "j"}):
            redirected = client.get("/media/j", follow_redirects=False)
        assert redirected.status_code == 302 and redirected.headers["location"].startswith(
            "https://media.warrioriq.eu/media-signed/j")
    finally:
        object.__setattr__(web.SETTINGS, "media_cdn_base_url", previous[0])
        object.__setattr__(web.SETTINGS, "media_signing_key", previous[1])


def test_the_replay_buffers_the_start_of_the_video():
    page = (ROOT / "app" / "templates" / "replay.html").read_text(encoding="utf-8")
    video = page[page.index('<video id="video"'):page.index("</video>")]
    assert 'preload="auto"' in video and 'src="/media/{{job_id}}"' in video
