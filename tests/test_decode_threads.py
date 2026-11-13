"""Video decoding on the shared web host (owner, 2026-11-13).

An upload of a plain 13 MB H.264 MP4 that analysed fine in September was
refused as "could not read this file as a video" (log: upload_not_decodable
reason=RuntimeError); the same file opens everywhere else. Shared hosting
counts threads against a small process limit, and a decoder starting one
thread per core of the shared server may not be able to start at all. The web
host now caps decoder threads, retries once with a single thread, and logs
the thread count if it still cannot open the file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import core.video as video

ROOT = Path(__file__).resolve().parents[1]


def _clip(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    for index in range(10):
        writer.write(np.full((48, 64, 3), index * 20, np.uint8))
    writer.release()
    return path


def test_the_web_host_caps_decoder_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("WARRIORIQ_DECODE_THREADS", "2")
    capture = video.open_capture(_clip(tmp_path))
    assert capture.isOpened() and capture.read()[0]
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        assert 1 <= capture.get(cv2.CAP_PROP_N_THREADS) <= 2
    capture.release()


def test_unset_leaves_opencv_to_choose(monkeypatch):
    monkeypatch.delenv("WARRIORIQ_DECODE_THREADS", raising=False)
    assert video._decode_threads() == 0
    monkeypatch.setenv("WARRIORIQ_DECODE_THREADS", "not a number")
    assert video._decode_threads() == 0


def test_a_decoder_that_cannot_open_is_tried_again_with_one_thread(tmp_path, monkeypatch):
    clip = _clip(tmp_path)
    real = video._raw_capture
    asked = []

    def flaky(path, threads=None):
        asked.append(threads)
        if threads != 1:
            return cv2.VideoCapture()          # never opened, as on the busy host
        return real(path, threads)

    monkeypatch.setattr(video, "_raw_capture", flaky)
    info = video.get_video_info(clip)
    assert info.frame_count == 10 and asked[-1] == 1


def test_the_web_entry_point_sets_the_caps_before_anything_imports_opencv():
    source = (ROOT / "warrioriq_wsgi.py").read_text(encoding="utf-8")
    assert source.index('os.environ.setdefault("WARRIORIQ_DECODE_THREADS", "2")') < source.index("from app.main import app")
    assert 'os.environ.setdefault("OMP_NUM_THREADS", "2")' in source
