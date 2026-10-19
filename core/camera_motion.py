"""How far the camera moved between two analysed frames.

Audit 2026-10-14 (#2): fighters were lost whenever the camera panned. BoT-SORT
compensates camera motion inside its own association (gmc_method in
models/warrioriq_botsort.yaml), but the identity manager above it compares a
candidate with the fighter's *last box in the previous frame's pixels* and
refuses anyone more than 1.65 body lengths away. A pan moves every person in
the picture at once, so the right fighter arrived "too far to be them".

This measures the shift of the background - corners tracked by optical flow,
with every person's box masked out - and the identity manager moves its
remembered boxes by the same amount before comparing. A rotation-and-scale
model (estimateAffinePartial2D) with RANSAC keeps a fighter's own movement
from being mistaken for the camera's, and an estimate without enough agreeing
background points is discarded rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Work on a small copy: the shift of a background is the same at 320 px, and
# the cost is per pixel.
WORK_WIDTH = 320
MIN_INLIERS = 20
MIN_INLIER_SHARE = 0.5
# Below this the "motion" is the estimate's own noise.
MIN_SHIFT_PX = 2.0


@dataclass
class CameraShift:
    """The camera's move as a 2x3 similarity transform in full-frame pixels."""

    matrix: np.ndarray
    inliers: int

    @property
    def dx(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def dy(self) -> float:
        return float(self.matrix[1, 2])

    @property
    def scale(self) -> float:
        return float(np.hypot(self.matrix[0, 0], self.matrix[1, 0]))

    @property
    def significant(self) -> bool:
        return abs(self.dx) >= MIN_SHIFT_PX or abs(self.dy) >= MIN_SHIFT_PX or abs(self.scale - 1.0) >= 0.02

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        m = self.matrix
        return float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2])

    def apply_box(self, box) -> np.ndarray:
        """Where a box from the previous frame sits in this frame's pixels."""
        x1, y1 = self.apply_point(float(box[0]), float(box[1]))
        x2, y2 = self.apply_point(float(box[2]), float(box[3]))
        return np.asarray([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)


class GlobalMotion:
    """Feed analysed frames in order; each call returns the camera's move since the last."""

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None
        self._factor = 1.0

    def reset(self) -> None:
        self._previous = None

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        self._factor = WORK_WIDTH / float(width) if width > WORK_WIDTH else 1.0
        small = cv2.resize(frame, (max(1, round(width * self._factor)), max(1, round(height * self._factor)))) \
            if self._factor != 1.0 else frame
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    def update(self, frame: np.ndarray, people_boxes=()) -> CameraShift | None:
        """The camera's shift from the previous frame to this one, in full-frame pixels, or None."""
        gray = self._gray(frame)
        previous, self._previous = self._previous, gray
        if previous is None or previous.shape != gray.shape:
            return None
        mask = np.full(previous.shape, 255, dtype=np.uint8)
        for box in people_boxes or ():
            x1, y1, x2, y2 = (int(round(float(v) * self._factor)) for v in box)
            mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
        corners = cv2.goodFeaturesToTrack(previous, maxCorners=200, qualityLevel=0.01, minDistance=7, mask=mask)
        if corners is None or len(corners) < MIN_INLIERS:
            return None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, corners, None)
        if moved is None:
            return None
        good = status.reshape(-1) == 1
        source, target = corners.reshape(-1, 2)[good], moved.reshape(-1, 2)[good]
        if len(source) < MIN_INLIERS:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                      ransacReprojThreshold=2.0)
        if matrix is None or inliers is None:
            return None
        count = int(inliers.sum())
        if count < MIN_INLIERS or count < MIN_INLIER_SHARE * len(source):
            return None
        # The rotation and scale are the same at any resolution; the
        # translation is in the small copy's pixels.
        full = np.asarray(matrix, dtype=np.float64).copy()
        full[:, 2] /= self._factor
        return CameraShift(matrix=full, inliers=count)
