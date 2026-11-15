"""A small video is flagged on the selection page (owner, 2026-11-14).

A 480x220 fight lost a fighter among the people at the ringside and the
report came back with who-is-who unverified. The selection page now says so
before the analysis, while the person can still upload the original file.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np
from browser_client import BrowserClient

import app.main as web
from app.state import create_job, delete_job


def _page(width: int, height: int) -> str:
    client = BrowserClient(web.app)
    client.get("/")
    guest = client.cookies.get(web.GUEST_COOKIE)
    job_id = f"small{uuid.uuid4().hex[:10]}"
    job_dir = web.OUTPUTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(job_dir / "selection.jpg"), np.zeros((48, 64, 3), np.uint8))
    create_job(job_id, {"owner_key": f"guest:{guest}", "status": "selection", "auto_frame_done": True,
                        "video_width": width, "video_height": height, "video_duration": 3.0,
                        "start_seconds": 0.0, "selection_frame": 0, "selection_seconds": 0.0,
                        "fight_type": "sparring", "ruleset": "K1", "round_count": 1,
                        "round_duration_seconds": 3.0, "break_duration_seconds": 0.0})
    try:
        return client.get(f"/select/{job_id}").text
    finally:
        delete_job(job_id)


def test_a_small_video_is_flagged_with_its_size():
    page = _page(480, 220)
    assert 'id="smallVideoWarning"' in page and "480&times;220" in page


def test_a_normal_video_is_not():
    assert 'id="smallVideoWarning"' not in _page(1280, 720)
    assert 'id="smallVideoWarning"' not in _page(720, 1280)        # a phone filmed upright
