"""Sideways video is turned upright at upload (QA, 2026-10-04).

The fighters lay across the picture and the selection page called them "left"
and "right" when one was above the other.
"""

from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from core import orientation


def _standing(frame):
    """A fake detector: finds a person only if the frame shows them upright."""
    h, w = frame.shape[:2]
    if frame[: h // 4].mean() > frame[3 * h // 4:].mean() + 20:   # bright "head" end up
        return [{"box": [w * 0.4, h * 0.1, w * 0.6, h * 0.9], "confidence": 0.9}]
    return []


def _upright():
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    frame[:100] = 200          # head end bright
    return frame


@pytest.mark.parametrize("code, turn", [(cv2.ROTATE_90_CLOCKWISE, 270),
                                        (cv2.ROTATE_90_COUNTERCLOCKWISE, 90),
                                        (cv2.ROTATE_180, 180)])
def test_a_sideways_frame_asks_for_the_turn_that_stands_it_up(code, turn):
    assert orientation.needed_turn(cv2.rotate(_upright(), code), _standing) == turn


def test_an_upright_frame_is_left_alone():
    assert orientation.needed_turn(_upright(), _standing) == 0


def test_no_detector_means_no_turn():
    assert orientation.needed_turn(_upright(), lambda frame: None) == 0


def test_wide_people_do_not_count_as_upright():
    lying = [{"box": [0, 100, 300, 180], "confidence": 0.9}]
    assert orientation.upright_score(lying, 400) == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_tag_turns_the_decoded_frames(tmp_path):
    sideways = cv2.rotate(_upright(), cv2.ROTATE_90_CLOCKWISE)
    path = tmp_path / "side.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (sideways.shape[1], sideways.shape[0]))
    for _ in range(10):
        writer.write(sideways)
    writer.release()
    assert orientation.tag_rotation(path, 270, shutil.which("ffmpeg"))
    ok, frame = cv2.VideoCapture(str(path)).read()
    assert ok and frame.shape[:2] == (400, 300)
    assert frame[:100].mean() > frame[300:].mean() + 50
