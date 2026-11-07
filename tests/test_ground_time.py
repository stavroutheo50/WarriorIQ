"""MMA time on the ground (core/ground_time.py) and how the report shows it."""

import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from core import ground_time
from core.types import PersonObservation

ROOT = Path(__file__).resolve().parents[1]


def _person(x1, y1, x2, y2, flat=False):
    """A detected person with shoulders above hips (standing) or beside them (flat)."""
    kp = np.zeros((17, 2))
    kc = np.full(17, .9)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if flat:
        kp[5], kp[6], kp[11], kp[12] = (x1 + 10, cy), (x1 + 12, cy), (x2 - 30, cy), (x2 - 28, cy)
        kp[15], kp[16] = (x2 - 5, cy), (x2 - 4, cy)
    else:
        kp[5], kp[6], kp[11], kp[12] = (cx - 10, y1 + 30), (cx + 10, y1 + 30), (cx - 8, cy), (cx + 8, cy)
        kp[15], kp[16] = (cx - 8, y2 - 4), (cx + 8, y2 - 4)
    return PersonObservation(track_id=None, box=np.array([x1, y1, x2, y2], float), confidence=.9,
                             keypoints=kp, keypoint_conf=kc)


STANDING = [_person(100, 100, 220, 500), _person(300, 100, 420, 500)]
DOWN = [_person(100, 380, 500, 500, flat=True), _person(120, 340, 480, 470, flat=True)]


class FrameFeatureTests(unittest.TestCase):
    def rows(self, people):
        return [{"box": p.box.tolist(), "conf": p.confidence, "kp": p.keypoints.tolist(),
                 "kc": p.keypoint_conf.tolist()} for p in people]

    def test_standing_and_flat_fighters_read_differently(self):
        up = ground_time.frame_features(self.rows(STANDING), 640, 640)
        down = ground_time.frame_features(self.rows(DOWN), 640, 640)
        self.assertEqual(len(up), 10)
        self.assertEqual(up[2], 0.0)      # nobody flat
        self.assertEqual(down[2], 1.0)
        self.assertGreater(up[3], down[3])  # the boxes are taller than wide only when standing

    def test_the_crowd_is_left_out(self):
        tiny = [_person(10, 10, 20, 40)]
        self.assertEqual(ground_time.frame_features(self.rows(tiny), 640, 640)[8], 0.0)


class _FakePictures:
    """Stands in for the image model: describes a frame by its brightness."""

    def __init__(self):
        self.batches = []

    def features(self, frames):
        self.batches.append(len(frames))
        return np.array([[float(f.mean()) / 255.0, 1.0 - float(f.mean()) / 255.0] for f in frames])


class _BrokenPictures:
    def features(self, frames):
        raise RuntimeError("no weights")


class GroundWatchTests(unittest.TestCase):
    def watch(self, pattern, model=None):
        """10 fps; `pattern` says per second whether they are down (and the frame is dark)."""
        watch = ground_time.GroundWatch(640, 640, picture_model=model or _FakePictures())
        for second, down in enumerate(pattern):
            frame = np.full((360, 640, 3), 40 if down else 200, np.uint8)
            for i in range(10):          # the watch keeps 3 of them
                watch.observe(second + i / 10, DOWN if down else STANDING,
                              1 if second < len(pattern) // 2 else 2, frame)
        return watch

    def head(self):
        watch = self.watch([False, True] * 10)
        watch.summary({"mu": np.zeros(32), "sd": np.ones(32), "w": np.zeros(32), "b": np.array(0.0)})
        X = np.stack([ground_time.second_features(watch.frames[s], watch.pictures[s]) for s in sorted(watch.frames)])
        return ground_time.fit(X, np.array([s % 2 for s in sorted(watch.frames)], float))

    def test_frames_are_described_in_batches(self):
        model = _FakePictures()
        self.watch([False] * 4, model)
        self.assertEqual(model.batches, [ground_time.PICTURE_BATCH] * (12 // ground_time.PICTURE_BATCH))

    def test_without_the_image_model_nothing_is_reported(self):
        self.assertIsNone(self.watch([True] * 12, _BrokenPictures()).summary(self.head()))

    def test_a_head_of_another_shape_is_not_used(self):
        wrong = {"mu": np.zeros(5), "sd": np.ones(5), "w": np.zeros(5), "b": np.array(0.0)}
        self.assertIsNone(self.watch([True] * 4).summary(wrong))

    def test_samples_at_three_frames_a_second_whatever_the_stride(self):
        watch = self.watch([False] * 4)
        self.assertEqual([len(watch.frames[s]) for s in range(4)], [3, 3, 3, 3])

    def test_nothing_is_reported_without_a_fitted_head(self):
        with unittest.mock.patch.object(ground_time, "load_head", return_value=None):
            self.assertIsNone(self.watch([True] * 4).summary())

    def test_ground_seconds_are_counted_per_round(self):
        summary = self.watch([False] * 6 + [True] * 4 + [True] * 2 + [False] * 8).summary(self.head())
        self.assertEqual(summary["read_seconds"], 20)
        self.assertEqual(summary["ground_seconds"], 6)
        self.assertEqual(summary["ground_seconds_list"], [6, 7, 8, 9, 10, 11])
        self.assertEqual(summary["rounds"][1], {"read_seconds": 10, "ground_seconds": 4})
        self.assertEqual(summary["rounds"][2], {"read_seconds": 10, "ground_seconds": 2})

    def test_the_committed_head_loads(self):
        head = ground_time.load_head()
        self.assertIsNotNone(head, "models/ground_time_head.npz is missing")
        self.assertEqual(head["w"].shape, (2048 + 30,))


class GroundTimeViewTests(unittest.TestCase):
    def test_shown_with_rounds_only_when_enough_was_read(self):
        from app.main import _ground_time_view
        self.assertIsNone(_ground_time_view({}))
        self.assertIsNone(_ground_time_view({"ground_time": None}))
        self.assertIsNone(_ground_time_view({"ground_time": {"read_seconds": 9, "ground_seconds": 9}}))
        view = _ground_time_view({"ground_time": {"read_seconds": 200, "ground_seconds": 50, "rounds": {
            1: {"read_seconds": 100, "ground_seconds": 10}, 2: {"read_seconds": 100, "ground_seconds": 40}}}})
        self.assertEqual(view["share"], 25)
        self.assertEqual([(r["round"], r["share"]) for r in view["rounds"]], [(1, 10), (2, 40)])

    def test_measured_for_mma_only_and_shown_with_its_limits(self):
        analyzer = (ROOT / "core" / "analyzer.py").read_text(encoding="utf-8")
        self.assertIn('GroundWatch(info.width, info.height) if sport_of(normalize_ruleset(req.ruleset)) == "mma"',
                      analyzer)
        page = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn("{% if ground_time and not identity_failed %}", page)
        self.assertIn("Who was on top and takedowns are not measured yet.", page)



if __name__ == "__main__":
    unittest.main()
