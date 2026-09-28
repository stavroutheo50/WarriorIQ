"""The check behind training-session points (core/training_check.py)."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.training_check import check_training_video


def write_video(path: Path, seconds: float, moving: bool, fps: int = 10,
                size: tuple[int, int] = (160, 120), box: int = 40) -> str:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(int(seconds * fps)):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = (i * 7) % (width - box) if moving else width // 2
        cv2.rectangle(frame, (x, height // 4), (x + box, height // 4 + box * 2), (230, 230, 230), -1)
        writer.write(frame)
    writer.release()
    return str(path)


class TrainingCheckTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_someone_moving_for_a_minute_counts(self):
        found = check_training_video(write_video(self.root / "drill.mp4", 65, moving=True))
        self.assertEqual(found["verdict"], "counted")
        self.assertGreater(found["moving_share"], 0.9)

    def test_a_still_picture_does_not(self):
        found = check_training_video(write_video(self.root / "still.mp4", 65, moving=False))
        self.assertEqual(found["verdict"], "no_movement")

    def test_under_a_minute_is_too_short(self):
        found = check_training_video(write_video(self.root / "short.mp4", 20, moving=True))
        self.assertEqual(found["verdict"], "too_short")

    def test_an_athlete_filmed_from_far_away_still_counts(self):
        """A small figure in a wide shot: the whole-frame average barely moves."""
        found = check_training_video(write_video(self.root / "far.mp4", 65, moving=True,
                                                 size=(640, 360), box=12))
        self.assertEqual(found["verdict"], "counted")

    def test_above_1080p_is_refused_before_decoding_it(self):
        found = check_training_video(write_video(self.root / "big.mp4", 61, moving=True, fps=1,
                                                 size=(2560, 1440)))
        self.assertEqual(found["verdict"], "too_large")

    def test_something_that_is_not_a_video(self):
        path = self.root / "notes.mp4"
        path.write_bytes(b"not a video at all")
        self.assertEqual(check_training_video(str(path))["verdict"], "unreadable")


if __name__ == "__main__":
    unittest.main()
