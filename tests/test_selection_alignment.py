"""A browser-captured selection lands on the frame the analysis reads (2026-11-13).

The upload page's "tap your fighters while it uploads" sends a picture taken
by the browser, with the browser's clock. The analysis finds the chosen moment
as `seconds * average fps`, which in variable-frame-rate phone footage is a
different frame; the boxes then sat on people who had moved, the identity
check at the chosen frame failed, and two of the owner's fights were withheld
as "not sure which fighter was which". Re-tapping the same picture failed
again. The server now finds the frame that shows the picture by looking.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np

import app.main as web
from app.state import create_job, get_job
from core.config import OUTPUTS
from core.video import is_same_moment, match_frame_index

FPS = 25.0


def _frame(index: int):
    """A picture that is unmistakably frame `index`: a bar that moves and a number."""
    image = np.full((180, 320, 3), 40, np.uint8)
    x = 10 + (index * 7) % 280
    cv2.rectangle(image, (x, 30), (x + 24, 160), (220, 200, 60), -1)
    cv2.putText(image, str(index), (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return image


def _video(tmp_path, frames: int = 120):
    path = tmp_path / "fight.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 180))
    for index in range(frames):
        writer.write(_frame(index))
    writer.release()
    return path


def _browser_jpeg(index: int):
    ok, data = cv2.imencode(".jpg", _frame(index), [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def test_the_frame_showing_the_picture_is_found_near_a_wrong_guess(tmp_path):
    video = _video(tmp_path)
    match = match_frame_index(video, _browser_jpeg(50), guess_index=44, fps=FPS, window_seconds=1.0)
    assert match is not None and match["index"] == 50
    assert is_same_moment(match)


def test_a_picture_from_elsewhere_is_not_taken_for_a_match(tmp_path):
    video = _video(tmp_path)
    stranger = np.full((180, 320, 3), 200, np.uint8)
    cv2.circle(stranger, (60, 90), 50, (0, 0, 255), -1)
    match = match_frame_index(video, stranger, guess_index=44, fps=FPS, window_seconds=1.0)
    assert match is not None and not is_same_moment(match)


def _browser_job(tmp_path, picture_index: int, browser_seconds: float):
    video = _video(tmp_path)
    job_id = f"align{uuid.uuid4().hex[:10]}"
    (OUTPUTS / job_id).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUTS / job_id / "selection.jpg"), _browser_jpeg(picture_index))
    create_job(job_id, {"status": "selecting", "video_path": str(video), "video_duration": 120 / FPS,
                        "video_width": 320, "video_height": 180, "selection_from_browser": True,
                        "selection_seconds": browser_seconds, "start_seconds": browser_seconds,
                        "selection_frame": int(round(browser_seconds * FPS))})
    return job_id


def test_a_browser_selection_is_moved_to_the_frame_it_shows(tmp_path):
    # The browser said 1.76 s (frame 44); its picture is frame 50.
    job_id = _browser_job(tmp_path, picture_index=50, browser_seconds=44 / FPS)
    assert web._align_browser_selection(job_id, get_job(job_id)) == {"aligned": True}
    job = get_job(job_id)
    assert job["selection_frame"] == 50 and abs(job["selection_seconds"] - 50 / FPS) < 1e-6
    assert job["selection_aligned"] is True
    # The stored picture is now the video's own frame 50, so re-tapping it is safe.
    stored = cv2.imread(str(OUTPUTS / job_id / "selection.jpg"))
    assert is_same_moment(match_frame_index(get_job(job_id)["video_path"], stored, 50, FPS, window_seconds=0.2))
    # Done once: a second start does not look again.
    assert web._align_browser_selection(job_id, get_job(job_id)) == {}


def test_a_picture_that_matches_nothing_is_replaced_by_the_videos_own_frame(tmp_path):
    job_id = _browser_job(tmp_path, picture_index=50, browser_seconds=44 / FPS)
    stranger = np.full((180, 320, 3), 200, np.uint8)
    cv2.imwrite(str(OUTPUTS / job_id / "selection.jpg"), stranger)
    assert web._align_browser_selection(job_id, get_job(job_id)) == {"aligned": False}
    job = get_job(job_id)
    assert job["selection_from_browser"] is False and job["selection_frame"] == 44
    replaced = cv2.imread(str(OUTPUTS / job_id / "selection.jpg"))
    assert float(np.mean(np.abs(replaced.astype(float) - stranger.astype(float)))) > 20


def test_a_server_chosen_frame_is_left_alone(tmp_path):
    job_id = _browser_job(tmp_path, picture_index=50, browser_seconds=44 / FPS)
    from app.state import update_job
    update_job(job_id, {"selection_from_browser": False})
    assert web._align_browser_selection(job_id, get_job(job_id)) == {}
    assert get_job(job_id)["selection_frame"] == 44


def test_the_start_endpoint_aligns_before_anything_is_started():
    source = (web.__file__ and open(web.__file__, encoding="utf-8").read())
    start = source.index("def start(request: Request, job_id: str, payload: StartPayload):")
    body = source[start:start + 3000]
    assert "alignment = _align_browser_selection(job_id, job)" in body
    assert body.index("_align_browser_selection") < body.index("_validated_fighter_box")
    assert 'raise HTTPException(409, (\n            "WarriorIQ could not find that exact moment' in body


def test_a_failed_check_at_the_chosen_moment_says_so():
    from core.report import identity_failure
    tracking = {"fighter_A_coverage": 0.9, "fighter_B_coverage": 0.9, "identity_seed_confirmed": False}
    failure = identity_failure(tracking)
    assert failure is not None and failure["cause"] == "seed_moved" and failure["repick"] is True
    assert "where you tapped them" in failure["headline"]
