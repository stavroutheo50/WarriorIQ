"""core/peak_pose.py and tools/import_combat_sports.py: one-frame strike shapes from gym footage only."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import import_combat_sports as importer  # noqa: E402

from core.action import Sample, _feature_vector  # noqa: E402
from core.peak_pose import (FEATURES, PEAK_CLASSES, PeakPoseModel, mirror_features,  # noqa: E402
                            pose_features, scores, train)


def _pose(reach: float, rng=None) -> np.ndarray:
    """A standing skeleton in a 100x200 box whose right wrist sits ``reach`` across."""
    kp = np.zeros((17, 2), np.float32)
    kp[:, 0] = 50.0
    kp[:, 1] = np.linspace(10, 190, 17)
    kp[10] = (50.0 + reach, 60.0)
    if rng is not None:
        kp += rng.normal(0, 2.0, kp.shape).astype(np.float32)
    return kp


class FeatureTests(unittest.TestCase):
    def test_same_numbers_as_the_live_analysis(self):
        box = np.asarray([10, 20, 110, 220], np.float32)
        kp = _pose(30) + np.asarray([10, 20], np.float32)
        conf = np.linspace(0.2, 0.9, 17).astype(np.float32)
        live = _feature_vector(Sample(0, 0.0, None, box, kp, conf, None, None, None, 1.0, 1.0), None)
        self.assertEqual(pose_features(box, kp, conf).shape, (FEATURES,))
        np.testing.assert_allclose(pose_features(box, kp, conf), live[:FEATURES])

    def test_mirror_twice_is_the_same_pose_and_hidden_joints_stay_hidden(self):
        x = np.stack([pose_features([0, 0, 100, 200], _pose(r), None) for r in (5, 40)])
        x[0, 3 * 4:3 * 5] = 0.0
        np.testing.assert_allclose(mirror_features(mirror_features(x)), x, atol=1e-6)
        once = mirror_features(x)
        np.testing.assert_array_equal(once[0, 3 * 3:3 * 4], 0.0)  # left ear <- hidden right ear
        self.assertAlmostEqual(float(once[1, 3 * 9]), 1.0 - float(x[1, 3 * 10]), places=6)


class ModelTests(unittest.TestCase):
    def _data(self, seed):
        rng = np.random.default_rng(seed)
        reach = {"none": 0.0, "punch": 45.0}
        x, y = [], []
        for name in ("none", "punch"):
            for _ in range(60):
                x.append(pose_features([0, 0, 100, 200], _pose(reach[name], rng), None))
                y.append(PEAK_CLASSES.index(name))
        return np.stack(x), np.asarray(y)

    def test_learns_a_separable_shape_and_is_deterministic(self):
        x, y = self._data(0)
        model = train(x, y, epochs=200, seed=3)
        test_x, test_y = self._data(1)
        right = np.mean([p == PEAK_CLASSES[t] for p, t in zip(model.predict(test_x), test_y)])
        self.assertGreaterEqual(right, 0.95)
        np.testing.assert_array_equal(train(x, y, epochs=200, seed=3).logits(x), model.logits(x))

    def test_save_and_load_round_trip_and_refuse_other_classes(self):
        x, y = self._data(0)
        model = train(x, y, epochs=20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.npz"
            sha = model.save(path, seed=0)
            self.assertEqual(len(sha), 64)
            np.testing.assert_allclose(PeakPoseModel.load(path).logits(x), model.logits(x), rtol=1e-6)
            with np.load(path) as blob:
                arrays = dict(blob)
            arrays["classes"] = np.asarray(["none", "punch"])
            np.savez(Path(tmp) / "other.npz", **arrays)
            with self.assertRaises(ValueError):
                PeakPoseModel.load(Path(tmp) / "other.npz")

    def test_scores_count_what_they_say(self):
        result = scores(["punch", "punch", "none", "kick"], ["punch", "none", "punch", "none"])
        self.assertEqual(result["punch"], {"true": 2, "called": 2, "correct": 1, "precision": 0.5, "recall": 0.5})
        self.assertIsNone(result["kick"]["precision"])  # never called: no precision, not 0 or 1
        self.assertEqual(result["kick"]["recall"], 0.0)


def _row(name, box):
    x1, y1, x2, y2 = box
    return f"{importer.NAMES.index(name)} {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}"


PERSON = (0.1, 0.1, 0.5, 0.9)


class LabelTests(unittest.TestCase):
    def test_source_and_frame_names(self):
        self.assertEqual(importer.source_of("x/train/labels/28_mp4-0017_jpg.rf.ab12.txt"), "28")
        self.assertIsNone(importer.source_of("x/train/labels/usatsi_123_jpg.rf.ab.txt"))
        self.assertIsNone(importer.source_of("x/train/labels/frame_1_jpg.rf.ab.txt"))
        self.assertEqual(importer.frame_key("a/b/28_mp4-0017_jpg.rf.ab12.txt"),
                         importer.frame_key("c/d/28_mp4-0017_jpg.rf.ff99.txt"))

    def test_each_person_gets_one_true_answer_or_none(self):
        other = (0.6, 0.1, 0.9, 0.9)
        rows = importer.parse_labels("\n".join([
            _row("person", PERSON), _row("cross", (0.3, 0.3, 0.45, 0.4)), _row("high-guard", (0.2, 0.2, 0.3, 0.3)),
            _row("person", other), _row("low-guard", (0.65, 0.3, 0.8, 0.4)),
            _row("boxing-bag", (0.5, 0.0, 0.6, 0.9))]))
        found = importer.people_with_labels(rows)
        self.assertEqual([(f, t) for _, f, t in found], [("punch", "cross"), ("none", "guard")])

    def test_ambiguous_unlabelled_and_ownerless_people_are_skipped(self):
        mixed = [("person", list(PERSON)), ("cross", [0.2, 0.2, 0.3, 0.3]), ("kick", [0.2, 0.6, 0.4, 0.8])]
        self.assertEqual(importer.people_with_labels(mixed), [])
        self.assertEqual(importer.people_with_labels([("person", list(PERSON))]), [])
        outside = [("person", list(PERSON)), ("hook", [0.7, 0.7, 0.9, 0.9])]
        self.assertEqual(importer.people_with_labels(outside), [])
        self.assertEqual(importer.people_with_labels([("cross", [0.2, 0.2, 0.3, 0.3])]), [])

    def test_punches_merge_into_one_family(self):
        rows = [("person", list(PERSON)), ("cross", [0.2, 0.2, 0.3, 0.3]), ("hook", [0.3, 0.2, 0.4, 0.3])]
        self.assertEqual([(f, t) for _, f, t in importer.people_with_labels(rows)], [("punch", "cross+hook")])


def _archive(path: Path) -> Path:
    import cv2

    ok, jpg = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
    label = "\n".join([_row("person", PERSON), _row("cross", (0.3, 0.3, 0.45, 0.4))])
    base = "CombatSports/train"
    with zipfile.ZipFile(path, "w") as z:
        for stem in ("28_mp4-0001_jpg.rf.a", "28_mp4-0001_jpg.rf.b",   # one frame, two copies
                     "9_mp4-0002_jpg.rf.c",                            # held-out video
                     "24_mp4-0003_jpg.rf.d", "usatsi_1_jpg.rf.e"):      # not gym footage
            z.writestr(f"{base}/labels/{stem}.txt", label)
            z.writestr(f"{base}/images/{stem}.jpg", jpg.tobytes())
    return path


class ImportTests(unittest.TestCase):
    def test_only_gym_videos_once_each_split_by_video(self):
        joints = [(_pose(40), np.ones(17, np.float32))]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(importer, "keypoints_for", return_value=joints):
            rows, counts = importer.import_archive(_archive(Path(tmp) / "csd.zip"))
        self.assertEqual(rows["source"], ["28", "9"])
        self.assertEqual(rows["split"], ["train", "exam"])
        self.assertEqual(rows["family"], ["punch", "punch"])
        self.assertEqual(counts["skipped: not a gym source"], 2)
        self.assertEqual(counts["skipped: augmented copy"], 1)

    def test_writes_nothing_without_the_live_pose_model(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(importer, "keypoints_for", return_value=None):
            out = Path(tmp) / "out.npz"
            with self.assertRaises(SystemExit) as stop:
                importer.main(["--zip", str(_archive(Path(tmp) / "csd.zip")), "--out", str(out)])
            self.assertIn("RTMPose", str(stop.exception))
            self.assertFalse(out.exists())

    def test_own_footage_check_reads_each_window_at_its_last_frame(self):
        import test_model_on_own_footage as own

        class Fixed:
            def predict(self, x):
                return ["punch" if row[0] > 0 else "none" for row in x]

        quiet, peak = np.zeros((12, 102), np.float32), np.zeros((12, 102), np.float32)
        quiet[0, 0] = 1.0   # an early frame says punch; only the last one counts
        peak[-1, 0] = 1.0
        self.assertEqual(own._peak_predict(Fixed(), [quiet, peak]), {"none": 1, "punch": 1})

    def test_exam_videos_are_gym_videos(self):
        self.assertTrue(set(importer.EXAM_SOURCES) < set(importer.GYM_SOURCES))


if __name__ == "__main__":
    unittest.main()
