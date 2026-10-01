"""The site must survive a person browsing quickly, and say so when it limits.

QA, 2026-09: about fifty quick GETs from one signed-in browser were followed
by warrioriq.eu timing out for fifteen minutes. The application cannot produce
a TCP connection timeout - it answers or it errors - so the drop itself came
from in front of it (the host's firewall or bot-challenge layer; see README
"Uptime monitoring"). What the application did contribute was cost: every
request, a stylesheet included, re-read and file-locked every job session on
disk, on the event loop. These tests pin the fixes.
"""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import patch

import pytest

import app.main as web
from browser_client import BrowserClient


@pytest.fixture()
def client():
    web._rate_windows.clear()
    with BrowserClient(web.app) as c:
        yield c
    web._rate_windows.clear()


def test_healthz_answers_without_database_session_or_disk(client):
    def explode(*_args, **_kwargs):
        raise AssertionError("the liveness probe must not touch this")

    with patch.object(web, "resolve_session", explode), patch.object(web, "list_jobs", explode):
        response = client.get("/healthz", cookies={"warrioriq_session": "x" * 40})
        head = client.head("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers
    assert head.status_code == 200


def test_assets_and_probes_skip_the_visitor_context(client):
    calls = []

    def counting_list_jobs():
        calls.append(1)
        return []

    with patch.object(web, "list_jobs", counting_list_jobs):
        for path in ("/static/style.css", "/assets/base.css", "/favicon.ico",
                     "/robots.txt", "/sitemap.xml", "/health", "/healthz"):
            assert client.get(path).status_code == 200, path
        # A missing asset still answers - in plain text, since there is no
        # visitor context to lay a page out with.
        missing = client.get("/static/does-not-exist.css")
    assert missing.status_code == 404
    assert calls == []


def test_fifty_quick_requests_all_answer_and_scan_jobs_rarely(client):
    """The QA sequence: pages, the API schema, legal pages and some 404s."""
    paths = ["/", "/analyze", "/history", "/camp", "/pricing", "/terms", "/privacy",
             "/security", "/subprocessors", "/no-such-page", "/analyze/boxing",
             "/static/style.css", "/legal", "/nope/again"]
    scans = []
    real_list_jobs = web.list_jobs

    def counting_list_jobs():
        scans.append(1)
        return real_list_jobs()

    statuses = []
    with patch.object(web, "list_jobs", counting_list_jobs):
        web._navigation_snapshot = None
        for index in range(50):
            statuses.append(client.get(paths[index % len(paths)]).status_code)
    assert all(code < 500 for code in statuses), statuses
    assert 429 not in statuses
    # Without the snapshot this was one full scan per page request (~45).
    assert len(scans) <= 3


def test_navigation_snapshot_sees_this_processes_writes_at_once():
    web._navigation_snapshot = None
    before = web._navigation_jobs()
    web.create_job("snapshotcheck1", {"owner_key": "account:999999", "status": "selecting"})
    try:
        after = dict(web._navigation_jobs())
        assert "snapshotcheck1" in after
        assert "snapshotcheck1" not in dict(before)
    finally:
        web.delete_job("snapshotcheck1")
    assert "snapshotcheck1" not in dict(web._navigation_jobs())


def test_site_wide_limit_answers_429_with_a_readable_page(client):
    limited = dataclasses.replace(web.SETTINGS, request_rate_limit_per_minute=3)
    with patch.object(web, "SETTINGS", limited):
        codes = [client.get("/pricing").status_code for _ in range(3)]
        page = client.get("/pricing")
        api = client.get("/api/active-analysis")
        # Probes and assets are never counted, so monitoring keeps working.
        probe = client.get("/healthz")
        asset = client.get("/static/style.css")
    assert codes == [200, 200, 200]
    assert page.status_code == 429
    assert int(page.headers["retry-after"]) >= 1
    assert "text/html" in page.headers["content-type"]
    assert "Slow down for a moment" in page.text
    assert "Wait about" in page.text
    assert api.status_code == 429
    assert "Wait about" in api.json()["detail"]
    assert probe.status_code == 200
    assert asset.status_code == 200


def test_route_limit_is_a_friendly_page_with_retry_after(client):
    with patch.object(web, "record_security_event") as recorded:
        for _ in range(40):
            response = client.post("/login", data={"email": "nobody@example.com", "password": "wrong-password"},
                                   follow_redirects=False)
            if response.status_code == 429:
                break
        again = client.post("/login", data={"email": "nobody@example.com", "password": "wrong-password"},
                            follow_redirects=False)
    assert response.status_code == 429
    assert "Slow down for a moment" in response.text
    assert "Analysis limit reached" not in response.text
    assert int(response.headers["retry-after"]) >= 1
    assert again.status_code == 429
    # One security event per tripped window, not one per refused request.
    exceeded = [call for call in recorded.call_args_list if call.args and call.args[0] == "rate_limit_exceeded"]
    assert len(exceeded) == 1


def test_api_documentation_is_not_public(client):
    for path in ("/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"):
        assert client.get(path).status_code == 404, path
    assert client.get("/admin/openapi.json").status_code == 404


def test_reports_do_not_name_the_graphics_card():
    import app.main as web

    report = {"performance": {"gpu": "NVIDIA GeForce RTX 5060", "pose_model": "C:/Users/x/models/yolo26m-pose.engine",
                              "vram_free_at_start": {"free_gb": 6.1}}}
    cleaned = web._without_hardware(report)
    assert "RTX" not in json.dumps(cleaned)
    assert cleaned["performance"]["compute"] == "graphics card"
    assert cleaned["performance"]["pose_model"] == "yolo26m-pose.engine"
    template = (web.ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
    assert "'the graphics card (' ~ perf.gpu" not in template
