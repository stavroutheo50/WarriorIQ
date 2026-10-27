"""Pick the fighters while the video uploads (2026-10-27).

A 3-minute phone video took 11 minutes to upload at 0.5 MB/s and the fighters
could only be picked after it. The upload page now takes frames from the file
on the phone and asks /api/upload/<id>/people who is in them, so the fighters
are tapped during the upload; afterwards it uses the fighter page's own
/api/selection-frame and /api/start, and falls back to the fighter page.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from browser_client import BrowserClient

import app.main as web
from core.config import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def _signed_in():
    from core.auth import create_account, issue_session

    client = BrowserClient(web.app)
    account = create_account(f"early-{uuid.uuid4().hex[:10]}@example.com", "a-long-enough-password")
    client.cookies.set("warrioriq_session", issue_session(int(account["id"])))
    client.get("/analyze/kickboxing")
    return client, account, {"X-CSRF-Token": client.cookies.get("warrioriq_csrf")}


def _begin(client, headers):
    response = client.post("/api/upload/begin", headers=headers, json={
        "filename": "fight.mp4", "size": 64, "rights_confirmed": True, "people_permissions_confirmed": True,
        "minor_permission_status": "no_minors", "ruleset": "K1"})
    assert response.status_code == 201, response.text
    return response.json()["job_id"]


def _jpeg():
    image = np.full((360, 640, 3), 90, np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


def test_the_people_in_a_frame_are_found_for_the_uploads_owner_only():
    previous = SETTINGS.chunked_upload_enabled
    object.__setattr__(SETTINGS, "chunked_upload_enabled", True)
    web._rate_windows.clear()
    try:
        client, account, headers = _signed_in()
        job = _begin(client, headers)
        people = [{"box": [100.0, 60.0, 200.0, 340.0], "confidence": 0.8},
                  {"box": [330.0, 70.0, 430.0, 340.0], "confidence": 0.8},
                  {"box": [10.0, 10.0, 20.0, 20.0], "confidence": 0.9}]      # crowd: too small to tap
        with patch.object(web, "detect_people_in_frame", return_value=people):
            found = client.post(f"/api/upload/{job}/people", content=_jpeg(),
                                headers={**headers, "Content-Type": "image/jpeg"})
        assert found.status_code == 200, found.text
        body = found.json()
        assert len(body["people"]) == 2 and body["pair_found"] is True and body["score"] > 0
        stranger, _, stranger_headers = _signed_in()
        refused = stranger.post(f"/api/upload/{job}/people", content=_jpeg(),
                                headers={**stranger_headers, "Content-Type": "image/jpeg"})
        assert refused.status_code in {403, 404}
        assert client.post(f"/api/upload/{job}/people", content=b"not a picture",
                           headers={**headers, "Content-Type": "image/jpeg"}).status_code == 400
        client.post(f"/api/upload/{job}/abort", headers=headers)
    finally:
        object.__setattr__(SETTINGS, "chunked_upload_enabled", previous)
        web._rate_windows.clear()


def test_the_upload_page_picks_during_the_upload_and_falls_back_to_the_fighter_page():
    page = (ROOT / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
    script = (ROOT / "app" / "static" / "early-pick.js").read_text(encoding="utf-8")
    assert 'id="earlyPick"' in page and "/static/early-pick.js" in page
    assert "window.wiqEarlyPick.begin(file, job" in page
    assert "window.wiqEarlyPick.wants()" in page and "window.wiqEarlyPick.uploaded(payload)" in page
    # The fighter page's own calls, then the fighter page if either refuses.
    assert "/api/selection-frame/${id}/image" in script and "/api/start/${id}" in script
    assert "location.assign(upload.next_url)" in script
