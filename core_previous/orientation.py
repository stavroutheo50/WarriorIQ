"""Find video filmed sideways, and turn it the right way up.

QA, 2026-10-04: a phone clip recorded on its side was analysed as it lay. The
fighters stood horizontally across the picture, the detector found them badly,
and the selection page described them as "left" and "right" when one was above
the other in the frame - the real positions were up and down.

A video whose file *says* it is rotated is already handled: OpenCV and the
browser both apply the rotation tag. This is for the file that does not say
so. People standing up give tall detection boxes; the same people lying on
their side in the picture give wide ones, and are found far less confidently.
So the selection frame is tried as it is and turned a quarter each way, and
the turn that finds upright people clearly better than the original wins.

The fix is a rotation tag, written by ffmpeg with the streams copied - seconds
for a whole fight, nothing re-encoded - so every reader of the file (the
analysis, the replay, the browser) sees it upright from then on.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("warrioriq.orientation")

# How much better a turned frame has to score before the video is called
# sideways. A real sideways clip scores several times higher turned; an
# upright one with a crouching fighter does not come close.
MARGIN = 2.0
MIN_SCORE = 0.6

# cv2 rotation applied to the frame -> the clockwise turn the viewer needs.
_TURNS = {cv2.ROTATE_90_CLOCKWISE: 90, cv2.ROTATE_90_COUNTERCLOCKWISE: 270, cv2.ROTATE_180: 180}


def upright_score(people: list[dict], height: int) -> float:
    """Confidence carried by people who stand upright and are big enough to be fighters."""
    score = 0.0
    for person in people:
        x1, y1, x2, y2 = person["box"]
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        if h >= 1.3 * w and h >= 0.15 * height:
            score += float(person.get("confidence", 0.0))
    return score


def needed_turn(frame: np.ndarray, detect) -> int:
    """Clockwise degrees (0, 90, 180 or 270) that make the people in ``frame`` upright.

    ``detect`` is core.person_detect.detect_people (injected for tests). 0 when
    the detector is unavailable or nothing is clearly better than as filmed.
    """
    if frame is None or frame.size == 0:
        return 0
    found = detect(frame)
    if found is None:
        return 0
    base = upright_score(found, frame.shape[0])
    best_turn, best = 0, base
    for code, turn in _TURNS.items():
        turned = cv2.rotate(frame, code)
        people = detect(turned) or []
        score = upright_score(people, turned.shape[0])
        if score > best:
            best_turn, best = turn, score
    if best_turn and best >= MIN_SCORE and best >= MARGIN * max(base, 0.05):
        LOGGER.info("video_sideways turn=%s upright=%.2f as_filmed=%.2f", best_turn, best, base)
        return best_turn
    return 0


def tag_rotation(path: str | Path, clockwise: int, ffmpeg: str | None) -> bool:
    """Rewrite ``path`` with a display rotation, streams copied. True on success.

    ffmpeg's -display_rotation is counter-clockwise, so a clockwise turn of 90
    is written as -90 (270).
    """
    if not ffmpeg or clockwise not in (90, 180, 270):
        return False
    path = Path(path)
    staged = path.with_name(f".rotated-{path.name}")
    command = [ffmpeg, "-loglevel", "error", "-y", "-display_rotation", str(-clockwise),
               "-i", str(path), "-map", "0", "-c", "copy", str(staged)]
    try:
        subprocess.run(command, check=True, timeout=300, capture_output=True)
        os.replace(staged, path)
    except (OSError, subprocess.SubprocessError) as exc:
        staged.unlink(missing_ok=True)
        LOGGER.warning("video_rotation_failed error=%s", type(exc).__name__)
        return False
    return True
