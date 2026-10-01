"""Real videos are never refused for the frame they open on or their codec.

QA 2026-09: a 1920x1080 60 s WebM failed at /finish with 422 "could not
prepare this video's fighter-selection frame" while another 1080p WebM worked.
The difference is the codec: this build of OpenCV opens an AV1 WebM, reports
its frame count and decodes nothing. And .ogv was refused as "not a video".
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core import video
from core.types import VideoInfo

FFMPEG = shutil.which("ffmpeg")


def _write_clip(path: Path, frames: int = 60, black_first: int = 0, size=(320, 240), fps=25.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(frames):
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if index >= black_first:
            image[:] = (60, 120, 90)
            cv2.rectangle(image, (40 + index, 60), (100 + index, 200), (200, 80, 40), -1)
            cv2.rectangle(image, (200 - index, 60), (260 - index, 200), (40, 80, 200), -1)
        writer.write(image)
    writer.release()


def test_a_black_opening_is_skipped(tmp_path):
    clip = tmp_path / "fade.mp4"
    _write_clip(clip, frames=200, black_first=150)
    info = video.get_video_info(clip)
    index, frame = video.selection_frame(clip, info, preferred_frame=0)
    assert frame is not None
    assert index >= 150
    assert video._usable_picture(frame)


@pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg with an AV1 encoder")
def test_an_av1_webm_still_gives_a_frame_and_a_readable_copy(tmp_path):
    source = tmp_path / "source.mp4"
    _write_clip(source, frames=50)
    av1 = tmp_path / "fight.webm"
    made = subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", str(source), "-c:v", "libaom-av1",
                           "-crf", "45", "-cpu-used", "8", str(av1)], capture_output=True)
    if made.returncode != 0:
        pytest.skip("this ffmpeg has no AV1 encoder")
    assert not video.opencv_decodes(av1), "OpenCV gained AV1 support; the fallback is now belt and braces"
    info = video.get_video_info(av1)
    index, frame = video.selection_frame(av1, info, preferred_frame=10)
    assert frame is not None and frame.shape[:2] == (240, 320)
    copy = video.decodable_copy(av1, tmp_path / "copy.mp4")
    assert copy is not None and video.opencv_decodes(copy)
    assert video.get_video_info(copy).frame_count == info.frame_count


def test_ogg_video_is_accepted_by_signature(tmp_path):
    from core.upload_security import FIGHT_VIDEO_EXTENSIONS, looks_like_video

    assert ".ogv" in FIGHT_VIDEO_EXTENSIONS
    clip = tmp_path / "fight.ogv"
    clip.write_bytes(b"OggS\x00\x02" + b"\x00" * 64)
    assert looks_like_video(clip)


# --- the upload keeps a file nothing on the host can decode ------------------

@pytest.fixture
def signed_in():
    import app.main as web
    from browser_client import BrowserClient
    from core import db
    from core.auth import issue_session

    account = db.create_account(f"robust-{uuid.uuid4().hex}@example.test", "not-a-login-hash")
    db.set_plan_override(int(account["id"]), "gym")
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(account["id"]))
    web._rate_windows.clear()
    yield web, client
    client.close()


def _upload(client, path: Path):
    with path.open("rb") as handle:
        return client.post("/upload", headers={"Accept": "application/json"},
                           files={"video": (path.name, handle, "video/webm")},
                           data={"rights_confirmed": "true", "people_permissions_confirmed": "true",
                                 "minor_permission_status": "no_minors"})


def test_an_undecodable_upload_is_kept_and_sent_to_the_frame_picker(signed_in, tmp_path):
    web, client = signed_in
    clip = tmp_path / "fight.mp4"
    _write_clip(clip, frames=300)
    info = video.get_video_info(clip)
    with patch.object(web, "selection_frame", return_value=(None, None)), \
         patch.object(web, "opencv_decodes", return_value=False), \
         patch.object(web, "get_video_info", return_value=VideoInfo(str(clip), 25.0, 300, 320, 240, 12.0)), \
         patch.object(web, "probe_upload", return_value=(0, [])):
        response = _upload(client, clip)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["next_url"] == f"/frame/{payload['job_id']}"
    job = web.get_job(payload["job_id"])
    assert job["selection_pending"] is True
    assert job["quality"]["status"] == "unmeasured"
    # The selection page sends them back to the picker until there is a frame.
    assert client.get(f"/select/{payload['job_id']}", follow_redirects=False).status_code == 303

    # The browser captures the frame instead.
    ok, png = cv2.imencode(".png", np.full((240, 320, 3), 120, dtype=np.uint8))
    stored = client.post(f"/api/selection-frame/{payload['job_id']}/image?seconds=3.2",
                         content=png.tobytes(), headers={"Content-Type": "image/png"})
    assert stored.status_code == 200, stored.text
    job = web.get_job(payload["job_id"])
    assert job["selection_pending"] is False and job["selection_from_browser"] is True
    assert abs(job["selection_seconds"] - 3.2) < 1e-6
    assert (web.OUTPUTS / payload["job_id"] / "selection.jpg").exists()

    wrong_shape = cv2.imencode(".png", np.zeros((300, 300, 3), dtype=np.uint8))[1]
    refused = client.post(f"/api/selection-frame/{payload['job_id']}/image?seconds=1",
                          content=wrong_shape.tobytes(), headers={"Content-Type": "image/png"})
    assert refused.status_code == 400
    assert info.frame_count > 0


def test_analysis_reads_a_converted_copy_when_opencv_cannot(tmp_path):
    from core import analyzer
    from core.types import AnalysisRequest

    req = AnalysisRequest(video_path=str(tmp_path / "fight.webm"), fighter_a_box=[0, 0, 1, 1],
                          fighter_b_box=[2, 0, 3, 1], output_dir=str(tmp_path / "run"))
    seen = {}
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(b"x")

    def fake_copy(source, destination):
        seen["source"] = source
        return copy

    with patch.object(analyzer, "opencv_decodes", side_effect=lambda path: str(path) == str(copy)), \
         patch.object(analyzer, "decodable_copy", side_effect=fake_copy), \
         patch.object(analyzer, "_analyze_from_seed", side_effect=lambda r, cb: {"video_path": r.video_path,
                                                                                  "name": r.original_name}):
        result = analyzer.analyze(req)
    assert result == {"video_path": str(copy), "name": "fight.webm"}
    with patch.object(analyzer, "opencv_decodes", return_value=False), \
         patch.object(analyzer, "decodable_copy", return_value=None):
        with pytest.raises(analyzer.UnreadableVideo):
            analyzer.analyze(req)


def test_an_unreadable_video_failure_says_what_to_do():
    import app.main as web

    message = web._worker_failure_message("UnreadableVideo")
    assert "MP4 (H.264)" in message
    assert "did not reach the analysis machine" not in message
