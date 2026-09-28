"""The check behind training-session points (core/training_check.py)."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.training_check import check_training_video


def write_video(path: Path, seconds: float, moving: bool, fps: int = 10) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(int(seconds * fps)):
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        x = (i * 7) % 120 if moving else 60
        cv2.rectangle(frame, (x, 30), (x + 40, 100), (230, 230, 230), -1)
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

    def test_something_that_is_not_a_video(self):
        path = self.root / "notes.mp4"
        path.write_bytes(b"not a video at all")
        self.assertEqual(check_training_video(str(path))["verdict"], "unreadable")


if __name__ == "__main__":
    unittest.main()
