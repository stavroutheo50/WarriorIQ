"""Check who is who on the report (owner, 2026-11-13).

The report shows the crossings the analysis was least sure about as stills
with A and B boxed; the person taps which box is their fighter, and "Fix who
is who" analyses the fight again with those answers pinned in.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import cv2
import numpy as np
from browser_client import BrowserClient

import app.main as web
from app.state import create_job, delete_job, get_job

QUESTIONS = [
    {"entered_seconds": 1.0, "left_seconds": 2.0, "ask_seconds": 2.4, "ask_frame": 60,
     "a_box": [20.0, 20.0, 80.0, 150.0], "b_box": [200.0, 20.0, 260.0, 150.0], "lowest_confidence": 0.4},
    {"entered_seconds": 5.0, "left_seconds": 6.0, "ask_seconds": 6.4, "ask_frame": 160,
     "a_box": [40.0, 20.0, 100.0, 150.0], "b_box": [220.0, 20.0, 280.0, 150.0], "lowest_confidence": 0.5},
]


def _fight(tmp_path):
    client = BrowserClient(web.app)
    client.get("/")
    guest = client.cookies.get(web.GUEST_COOKIE)
    job_id = f"who{uuid.uuid4().hex[:10]}"
    video = tmp_path / "fight.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (320, 180))
    for index in range(200):
        writer.write(np.full((180, 320, 3), 60 + index % 50, np.uint8))
    writer.release()
    job_dir = web.OUTPUTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "report.json").write_text(json.dumps({"tracking": {"crossings": {"count": 2, "questions": QUESTIONS}}}),
                                         encoding="utf-8")
    create_job(job_id, {"owner_key": f"guest:{guest}", "status": "complete", "video_path": str(video),
                        "fighter_a_box": [20.0, 20.0, 80.0, 150.0], "fighter_b_box": [200.0, 20.0, 260.0, 150.0],
                        "fight_type": "competition", "ruleset": "K1", "round_count": 1,
                        "round_duration_seconds": 8.0, "break_duration_seconds": 0.0})
    headers = {"X-CSRF-Token": client.cookies.get("warrioriq_csrf")}
    return client, job_id, headers


def test_the_still_of_a_question_is_served_to_its_owner_only(tmp_path):
    client, job_id, _ = _fight(tmp_path)
    try:
        still = client.get(f"/result/{job_id}/crossing/60.jpg")
        assert still.status_code == 200 and still.headers["content-type"] == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(still.content, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (180, 320)
        assert client.get(f"/result/{job_id}/crossing/61.jpg").status_code == 404     # not a question
        assert BrowserClient(web.app).get(f"/result/{job_id}/crossing/60.jpg").status_code == 404
    finally:
        delete_job(job_id)


def test_answers_are_kept_only_for_frames_that_were_asked(tmp_path):
    client, job_id, headers = _fight(tmp_path)
    web._rate_windows.clear()
    try:
        saved = client.post(f"/api/identity-answers/{job_id}", headers=headers,
                            json={"answers": {"60": "b", "999": "A", "160": "maybe"}})
        assert saved.status_code == 200, saved.text
        assert saved.json()["answers"] == {"60": "B"} and saved.json()["swapped"] == 1
        assert get_job(job_id)["identity_answers"] == {"60": "B"}
    finally:
        delete_job(job_id)


def test_fix_reruns_with_the_answers_pinned_and_only_when_something_was_swapped(tmp_path):
    client, job_id, headers = _fight(tmp_path)
    web._rate_windows.clear()
    try:
        client.post(f"/api/identity-answers/{job_id}", headers=headers, json={"answers": {"60": "A"}})
        nothing = client.post(f"/api/identity-fix/{job_id}", headers=headers)
        assert nothing.status_code == 200 and nothing.json()["rerun"] is False

        client.post(f"/api/identity-answers/{job_id}", headers=headers, json={"answers": {"160": "B"}})
        submitted = []
        previous_mode = web.SETTINGS.analysis_worker_mode
        object.__setattr__(web.SETTINGS, "analysis_worker_mode", "inprocess")
        try:
            with patch.object(web, "_require_analysis_capacity", return_value={"deferred": False}), \
                 patch.object(web.executor, "submit", side_effect=lambda fn, *args: submitted.append(args)):
                fixed = client.post(f"/api/identity-fix/{job_id}", headers=headers)
        finally:
            object.__setattr__(web.SETTINGS, "analysis_worker_mode", previous_mode)
        assert fixed.status_code == 200, fixed.text
        assert fixed.json() == {"rerun": True, "progress_url": f"/progress/{job_id}"}
        job = get_job(job_id)
        assert job["status"] == "queued" and job["identity_answers"] == {}
        anchors = job["identity_anchors"]
        assert [a["frame"] for a in anchors] == [60, 160]
        assert anchors[1]["swapped"] and anchors[1]["a_box"] == QUESTIONS[1]["b_box"]
        (req,) = [args[1] for args in submitted]
        assert req.identity_anchors == anchors
    finally:
        delete_job(job_id)


def test_the_remote_worker_is_sent_the_answers():
    job = {"fighter_a_box": [1, 2, 3, 4], "fight_type": "competition", "ruleset": "K1",
           "identity_anchors": [{"frame": 60, "seconds": 2.4, "a_box": [1, 2, 3, 4], "b_box": [5, 6, 7, 8]}]}
    assert web._remote_job_payload("x", job)["identity_anchors"] == job["identity_anchors"]


def test_the_live_page_is_sent_the_questions_and_the_answers_so_far():
    job = {"status": "running", "crossing_questions": QUESTIONS[:1], "identity_answers": {"60": "B"},
           "fighter_a_box": [1, 2, 3, 4]}
    status = web._public_job_status("x", job)
    assert status["crossing_questions"] == QUESTIONS[:1] and status["identity_answers"] == {"60": "B"}
    assert "fighter_a_box" not in status
