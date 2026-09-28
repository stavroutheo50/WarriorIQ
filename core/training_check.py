"""Is an uploaded training clip a real session? The check behind session points.

What it can say, and what it cannot
-----------------------------------
It runs on the web host, which has OpenCV but no pose model, so it checks
the video, not the person in it:

  * it opens and decodes as a video;
  * it lasts between MIN_SECONDS and MAX_SECONDS;
  * something moves in most of it: a still frame, a screen of text or a
    camera pointed at a wall does not count.

It cannot tell who is training or which drill it is. The points are sized
for that: a counted session is worth a little, the daily cap stops a stack
of clips from being worth a lot, and the big award waits for the athlete's
next fight to show the number actually moved (core/camp.py).
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_SECONDS = 60.0
MAX_SECONDS = 20 * 60.0
# The check runs inside the upload request on the web host, so its work is
# bounded whatever the clip: CHECKPOINTS places spread over the whole clip,
# each compared with the frame half a second later, read forward from one
# seek. Decoding every frame of a long clip to sample it would hold a worker
# for minutes; 40 seeks took 5.5 s on a 7-minute 720p bout and under a second
# on a 2-minute phone clip.
CHECKPOINTS = 40
GAP_SECONDS = 0.5
# Above 1080p even 60 seeks get expensive, and a phone can film at 1080p.
MAX_PIXELS = 1920 * 1080
WIDTH = 160
# A checkpoint is moving when this share of the picture changed by more than
# CHANGE_LEVEL (0-255 greyscale, after a light blur that removes sensor noise).
# A share rather than the frame's average change, because an athlete filmed
# from the back of a hall fills a few percent of the picture: on a wide shot
# of a real bout the average said "still" 93% of the time, the share said
# "moving" 90% of the time. A still frame, with or without added noise, scored
# 0 either way.
CHANGE_LEVEL = 25
MIN_CHANGED_SHARE = 0.005
# Share of the checkpoints that have to be moving.
MIN_MOVING_SHARE = 0.5


def check_training_video(path: str) -> dict:
    """{"verdict", "duration_seconds", "moving_share"}.

    verdict is "counted", "unreadable", "too_short", "too_long",
    "too_large" or "no_movement".
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"verdict": "unreadable", "duration_seconds": None, "moving_share": None}
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or frames <= 0:
            return {"verdict": "unreadable", "duration_seconds": None, "moving_share": None}
        duration = frames / fps
        if duration < MIN_SECONDS:
            return {"verdict": "too_short", "duration_seconds": duration, "moving_share": None}
        if duration > MAX_SECONDS:
            return {"verdict": "too_long", "duration_seconds": duration, "moving_share": None}
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        if width * height > MAX_PIXELS:
            return {"verdict": "too_large", "duration_seconds": duration, "moving_share": None}
        moving = compared = 0
        span = max(0.0, duration - GAP_SECONDS - 0.5)
        for index in range(CHECKPOINTS):
            start = span * (index + 0.5) / CHECKPOINTS
            first, second = _pair_at(capture, start, max(1, int(round(GAP_SECONDS * fps))))
            if first is None or second is None or first.shape != second.shape:
                continue
            compared += 1
            changed = float((np.abs(first - second) > CHANGE_LEVEL).mean())
            moving += int(changed >= MIN_CHANGED_SHARE)
        if compared == 0:
            return {"verdict": "unreadable", "duration_seconds": duration, "moving_share": None}
        share = moving / compared
        return {"verdict": "counted" if share >= MIN_MOVING_SHARE else "no_movement",
                "duration_seconds": round(duration, 1), "moving_share": round(share, 3)}
    finally:
        capture.release()


def _pair_at(capture, seconds: float, frames_apart: int):
    """Two small greyscale frames, `frames_apart` apart from this time, from one seek."""
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, first = capture.read()
    for _ in range(frames_apart - 1):
        capture.grab()
    ok_second, second = capture.read()
    if not (ok and ok_second) or first is None or second is None:
        return None, None
    return _small_grey(first), _small_grey(second)


def _small_grey(frame):
    height = max(1, int(round(frame.shape[0] * WIDTH / frame.shape[1])))
    grey = cv2.cvtColor(cv2.resize(frame, (WIDTH, height)), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(grey, (3, 3), 0).astype(np.int16)
