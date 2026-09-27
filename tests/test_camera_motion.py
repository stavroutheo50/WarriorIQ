"""Camera movement correction for the identity layer (core/camera_motion.py).

Found on real handheld recordings: the identity layer judged positions and
movement on screen, so a pan made the real fighter look like a jump and a
seated spectator look like a mover.
"""

from __future__ import annotations

import unittest

import numpy as np


def _textured_frame(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(45, 80, 3), dtype=np.uint8)
    import cv2

    return cv2.resize(small, (640, 360), interpolation=cv2.INTER_NEAREST)


class CameraMotionTests(unittest.TestCase):
    def test_a_known_pan_is_measured(self):
        import cv2

        from core.camera_motion import CameraMotionEstimator

        frame = _textured_frame()
        shifted = cv2.warpAffine(frame, np.float32([[1, 0, 12], [0, 1, 5]]),
                                 (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
        estimator = CameraMotionEstimator()
        self.assertIsNone(estimator.update(frame, []), "nothing to compare the first frame with")
        matrix = estimator.update(shifted, [])
        self.assertIsNotNone(matrix)
        self.assertAlmostEqual(float(matrix[0, 2]), 12.0, delta=0.5)
        self.assertAlmostEqual(float(matrix[1, 2]), 5.0, delta=0.5)

    def test_a_blank_picture_gives_no_answer_rather_than_a_guess(self):
        from core.camera_motion import CameraMotionEstimator

        blank = np.full((360, 640, 3), 90, dtype=np.uint8)
        estimator = CameraMotionEstimator()
        estimator.update(blank, [])
        self.assertIsNone(estimator.update(blank, []))

    def test_a_pan_moves_what_is_known_about_the_fighters(self):
        from core.identity import IdentityManager
        from core.types import PersonObservation

        def person(track_id, x):
            return PersonObservation(track_id=track_id, confidence=0.9,
                                     box=np.asarray([x, 100, x + 30, 180], dtype=np.float32))

        manager = IdentityManager(person(1, 100.0), person(2, 300.0), 0, source_fps=30.0)
        manager.a.last_box = person(1, 100.0).box
        manager.apply_camera_motion(np.asarray([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]]))
        self.assertAlmostEqual(float(manager.a.last_box[0]), 140.0, places=3)
        self.assertEqual(manager.camera_corrected_frames, 1)

    def test_a_seated_spectator_stays_still_while_the_camera_pans(self):
        """On screen the spectator slides 30 px a frame with the pan; relative
        to the hall they have not moved, and that is what history records."""
        from core.identity import IdentityManager
        from core.types import PersonObservation

        def person(track_id, x):
            return PersonObservation(track_id=track_id, confidence=0.9,
                                     box=np.asarray([x, 100, x + 30, 180], dtype=np.float32))

        manager = IdentityManager(person(1, 100.0), person(2, 300.0), 0, source_fps=30.0)
        for frame in range(0, 20):
            screen_x = 400.0 + 30.0 * frame
            if frame:
                manager.apply_camera_motion(np.asarray([[1.0, 0.0, 30.0], [0.0, 1.0, 0.0]]))
            manager._remember_positions([person(9, screen_x)], frame * 2)
        xs = [entry[1] for entry in manager._track_history[9]]
        self.assertLess(max(xs) - min(xs), 1e-3)

    def test_the_analysis_corrects_before_it_predicts_where_fighters_are(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "core" / "analyzer.py").read_text(encoding="utf-8")
        correct = source.index("manager.apply_camera_motion(")
        self.assertLess(correct, source.index("manager.expected_boxes()"))
        self.assertLess(correct, source.index("fighter_a, fighter_b = manager.update("))


if __name__ == "__main__":
    unittest.main()
