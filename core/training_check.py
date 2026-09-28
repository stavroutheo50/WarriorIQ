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
# Two samples a second at 160 px wide is enough to see a body move and cheap
# enough to run inside the upload request.
SAMPLES_PER_SECOND = 2.0
WIDTH = 160
# Mean absolute change between two samples, 0-255 greyscale. Sensor noise on
# a still phone is around 1-2; a person moving through a fifth of the frame
# is well above 4.
MOVING_DIFFERENCE = 4.0
# Share of the clip that has to be moving.
MIN_MOVING_SHARE = 0.5


def check_training_video(path: str) -> dict:
    """{"verdict", "duration_seconds", "moving_share"}.

    verdict is "counted", "unreadable", "too_short", "too_long" or
    "no_movement".
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
        step = max(1, int(round(fps / SAMPLES_PER_SECOND)))
        previous = None
        moving = compared = 0
        index = 0
        while True:
            if index % step:
                if not capture.grab():
                    break
                index += 1
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            index += 1
            height = max(1, int(round(frame.shape[0] * WIDTH / frame.shape[1])))
            grey = cv2.cvtColor(cv2.resize(frame, (WIDTH, height)), cv2.COLOR_BGR2GRAY).astype(np.int16)
            if previous is not None and previous.shape == grey.shape:
                compared += 1
                moving += int(float(np.abs(grey - previous).mean()) >= MOVING_DIFFERENCE)
            previous = grey
        if compared == 0:
            return {"verdict": "unreadable", "duration_seconds": duration, "moving_share": None}
        share = moving / compared
        return {"verdict": "counted" if share >= MIN_MOVING_SHARE else "no_movement",
                "duration_seconds": round(duration, 1), "moving_share": round(share, 3)}
    finally:
        capture.release()
