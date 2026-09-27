"""How the picture moved between two analysed frames.

The identity layer (core/identity.py) judged everything in screen positions:
where a fighter should be now, and whether somebody has been moving like a
fighter or sitting like a spectator. On a handheld phone that pans after the
action, both are wrong. The real fighter "jumps" when the camera moves and is
refused as too far away, and a seated spectator "moves" with the pan and
passes as a fighter.

Measured on two real handheld recordings: the fighters had to be found again
21-48 times a minute against 6-7 on a fixed broadcast camera, and with each
re-find the identity landed on spectators, coaches and the referee.

This estimates the frame-to-frame camera movement from the background, with
people masked out, as a partial affine (shift, uniform zoom, small rotation).
BoT-SORT already compensates its own track predictions the same way; this
gives the identity layer the same correction.
"""

from __future__ import annotations

import cv2
import numpy as np

# Work at this width: plenty for background features, cheap on CPU.
_WORK_WIDTH = 480
# Fewer inliers than this and the estimate is not trusted.
_MIN_INLIERS = 25


class CameraMotionEstimator:
    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._prev_scale = 1.0
        self.estimated = 0
        self.skipped = 0

    def update(self, frame: np.ndarray, people_boxes) -> np.ndarray | None:
        """A 2x3 matrix mapping the previous analysed frame onto this one.

        None on the first frame and whenever the background gives no reliable
        answer; the caller then carries on uncorrected, as before.
        """
        if frame is None:
            return None
        height, width = frame.shape[:2]
        scale = min(1.0, _WORK_WIDTH / float(max(1, width)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, (int(round(width * scale)), int(round(height * scale))),
                              interpolation=cv2.INTER_AREA)
        prev, prev_scale = self._prev_gray, self._prev_scale
        self._prev_gray, self._prev_scale = gray, scale
        if prev is None or prev.shape != gray.shape:
            return None

        # Background only: people move on their own and would pull the
        # estimate towards the action.
        mask = np.full(prev.shape, 255, dtype=np.uint8)
        for box in people_boxes or []:
            x1, y1, x2, y2 = (np.asarray(box, dtype=np.float32) * prev_scale).astype(int)
            pad_x, pad_y = int(0.15 * (x2 - x1)), int(0.10 * (y2 - y1))
            mask[max(0, y1 - pad_y):max(0, y2 + pad_y), max(0, x1 - pad_x):max(0, x2 + pad_x)] = 0
        points = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01,
                                         minDistance=8, mask=mask)
        if points is None or len(points) < _MIN_INLIERS:
            self.skipped += 1
            return None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, points, None,
                                                    winSize=(21, 21), maxLevel=3)
        good = status.reshape(-1) == 1
        if int(good.sum()) < _MIN_INLIERS:
            self.skipped += 1
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            points[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if matrix is None or inliers is None or int(inliers.sum()) < _MIN_INLIERS:
            self.skipped += 1
            return None
        # Back to full-resolution coordinates.
        matrix = matrix.astype(np.float64)
        matrix[:, 2] /= scale
        self.estimated += 1
        return matrix


def warp_box(box, matrix: np.ndarray) -> np.ndarray:
    """A box moved by a 2x3 camera matrix, kept axis-aligned."""
    x1, y1, x2, y2 = (float(v) for v in box)
    corners = np.asarray([[x1, y1, 1.0], [x2, y1, 1.0], [x1, y2, 1.0], [x2, y2, 1.0]])
    moved = corners @ np.asarray(matrix, dtype=np.float64).T
    return np.asarray([moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()],
                      dtype=np.float32)


def matrix_scale(matrix: np.ndarray) -> float:
    """The uniform zoom a partial-affine matrix applies."""
    m = np.asarray(matrix, dtype=np.float64)
    return float(np.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))) or 1.0
