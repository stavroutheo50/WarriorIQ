"""The TensorRT engine has to be cached per GPU, and must never be required.

An engine is compiled for one GPU model. The remote worker used to dodge that
by pointing WARRIORIQ_POSE_ENGINE at a file called `absent.engine`, which
forced every remote run onto the .pt checkpoint - correct answers, 1.19x
slower per frame measured on an RTX 5060 at imgsz 1600 (0.0544s against
0.0457s). Building once per GPU into a Volume keeps the speed without ever
shipping an engine that belongs to another card.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import trt_engine


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:                                  # noqa: BLE001 - absence is the answer
        return False


# ensure_pose_engine returns None before it looks at anything else when
# torch.cuda.is_available() is false, so a test that asserts an engine was
# built cannot pass on a machine without a GPU. These two lived as --deselect
# lines in .github/workflows/tests.yml while tests/ had open pull requests that
# a marker would have conflicted with; the requirement belongs in the test.
needs_cuda = unittest.skipUnless(
    _cuda_available(), "builds a TensorRT engine, which needs a CUDA device")


class EngineNamingTests(unittest.TestCase):
    def test_the_filename_carries_the_gpu(self):
        """Two container types must never be handed each other's engine."""
        with tempfile.TemporaryDirectory() as tmp:
            a = trt_engine.engine_path_for(tmp, "NVIDIA GeForce RTX 5060")
            b = trt_engine.engine_path_for(tmp, "NVIDIA A10G")
        self.assertNotEqual(a.name, b.name)
        self.assertIn("5060", a.name)
        self.assertIn("a10g", b.name)

    def test_the_filename_carries_the_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = trt_engine.engine_path_for(tmp, "NVIDIA A10G", size=640)
            large = trt_engine.engine_path_for(tmp, "NVIDIA A10G", size=1600)
        self.assertNotEqual(small.name, large.name)

    def test_the_slug_is_filename_safe(self):
        slug = trt_engine.gpu_slug("NVIDIA GeForce RTX 5060 Laptop GPU")
        self.assertNotIn(" ", slug)
        self.assertEqual(slug, slug.lower())
        self.assertTrue(all(ch.isalnum() or ch == "-" for ch in slug), slug)

    def test_one_engine_is_tuned_for_640_and_accepts_1280_and_low_resolution(self):
        """Tuned for the size nearly every analysis runs at, and still able to
        serve the larger ones (QA, 2026-10-04: handle imgsz 640 and 1280).

        It was optimised at 1600, the rare small-source size, with a ceiling of
        6400 nobody uses.
        """
        from core.config import SETTINGS
        self.assertEqual(trt_engine.engine_size(), SETTINGS.default_imgsz)
        self.assertGreaterEqual(trt_engine.engine_max_size(), 1280)
        self.assertGreaterEqual(trt_engine.engine_max_size(), SETTINGS.low_resolution_imgsz)
        self.assertLess(trt_engine.engine_max_size(), 4 * SETTINGS.default_imgsz)

    def test_the_export_asks_for_that_ceiling(self):
        """Ultralytics sets the profile maximum to size x max(2, workspace)."""
        exported = {}

        class FakeYOLO:
            def __init__(self, path):
                pass

            def export(self, **kwargs):
                exported.update(kwargs)
                path = Path(tmp) / "out.engine"
                path.write_bytes(b"engine")
                return str(path)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("ultralytics.YOLO", FakeYOLO):
            trt_engine.build_pose_engine(Path(tmp) / "pose.engine", device=0)
        self.assertEqual(exported["imgsz"], trt_engine.engine_size())
        self.assertTrue(exported["dynamic"])
        self.assertEqual(exported["imgsz"] * max(2, exported["workspace"]), trt_engine.engine_max_size())

    def test_an_engine_built_under_the_old_sizing_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = trt_engine.engine_path_for(tmp, "NVIDIA A10").name
        self.assertNotEqual(name, "pose_engine_nvidia-a10_1600.engine")
        self.assertIn("_max", name)


class EnsureEngineTests(unittest.TestCase):
    @needs_cuda
    def test_an_existing_engine_is_reused_not_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = trt_engine.engine_path_for(tmp)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"pretend engine")
            with mock.patch.object(trt_engine, "build_pose_engine") as build:
                got = trt_engine.ensure_pose_engine(tmp)
            self.assertEqual(got, target)
            build.assert_not_called()

    @needs_cuda
    def test_a_missing_engine_is_built_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = trt_engine.engine_path_for(tmp)
            with mock.patch.object(trt_engine, "build_pose_engine",
                                   return_value=expected) as build:
                got = trt_engine.ensure_pose_engine(tmp)
            self.assertEqual(got, expected)
            build.assert_called_once()

    def test_a_failed_build_costs_speed_and_not_the_analysis(self):
        """Falling back to the .pt checkpoint is the same model, only slower."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(trt_engine, "build_pose_engine",
                                   side_effect=RuntimeError("no nvcc")):
                self.assertIsNone(trt_engine.ensure_pose_engine(tmp))

    def test_no_cuda_means_no_engine_and_no_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("torch.cuda.is_available", return_value=False):
                self.assertIsNone(trt_engine.ensure_pose_engine(tmp))


if __name__ == "__main__":
    unittest.main()


class EngineResolutionTests(unittest.TestCase):
    """A stale WARRIORIQ_POSE_ENGINE must not cost the run its TensorRT engine."""

    def test_the_configured_engine_wins_when_it_exists(self):
        from core.pose_tracker import resolve_pose_engine

        with tempfile.TemporaryDirectory() as tmp:
            configured = Path(tmp) / "custom.engine"
            configured.write_bytes(b"engine")
            self.assertEqual(resolve_pose_engine(str(configured), "NVIDIA A10"), configured)

    def test_a_missing_configured_path_falls_back_to_this_gpus_cached_engine(self):
        from core.pose_tracker import resolve_pose_engine

        with tempfile.TemporaryDirectory() as tmp:
            cached = trt_engine.engine_path_for(tmp, "NVIDIA A10")
            cached.write_bytes(b"engine")
            resolved = resolve_pose_engine(str(Path(tmp) / "absent.engine"), "NVIDIA A10")
            self.assertEqual(resolved, cached)

    def test_another_gpus_engine_is_never_picked_up(self):
        from core.pose_tracker import resolve_pose_engine

        with tempfile.TemporaryDirectory() as tmp:
            trt_engine.engine_path_for(tmp, "NVIDIA GeForce RTX 5060").write_bytes(b"engine")
            missing = Path(tmp) / "absent.engine"
            self.assertEqual(resolve_pose_engine(str(missing), "NVIDIA A10"), missing)


class RejectedEngineTests(unittest.TestCase):
    """An engine the runtime rejects falls back to this GPU's own engine first.

    On the analysis PC (2026-10-06) models/yolo26m-pose.engine had been built
    by a newer TensorRT than the one installed; it was rejected, and every
    analysis ran on PyTorch while the 5060's own engine sat in the same folder.
    """

    def _tracker(self, rejected):
        from core import pose_tracker

        tracker = object.__new__(pose_tracker.PoseTracker)
        tracker.model_path, tracker.device, tracker.uses_cuda = str(rejected), 0, True
        tracker._warmed, tracker._focus_model = False, None
        return tracker

    def _fake_yolo(self, refuses):
        class FakeYOLO:
            def __init__(self, path, task=None):
                assert task == "pose"
                self.path = str(path)

            def predict(self, *args, **kwargs):
                if self.path in refuses:
                    raise AttributeError("'NoneType' object has no attribute 'create_execution_context'")
                return []
        return FakeYOLO

    def test_this_gpus_engine_is_used_before_pytorch(self):
        from core import pose_tracker

        with tempfile.TemporaryDirectory() as tmp:
            rejected = Path(tmp) / "yolo26m-pose.engine"
            own = Path(tmp) / "pose_engine_gpu_640_max1920.engine"
            rejected.write_bytes(b"old")
            own.write_bytes(b"new")
            tracker = self._tracker(rejected)
            with mock.patch.object(pose_tracker, "YOLO", self._fake_yolo({str(rejected)})), \
                    mock.patch("core.trt_engine.engine_path_for", return_value=own):
                tracker.warmup(object())
            self.assertEqual(tracker.model_path, str(own))
            self.assertTrue(tracker._warmed)

    def test_pytorch_when_no_engine_loads(self):
        from core import pose_tracker
        from core.config import SETTINGS

        with tempfile.TemporaryDirectory() as tmp:
            rejected = Path(tmp) / "yolo26m-pose.engine"
            own = Path(tmp) / "pose_engine_gpu_640_max1920.engine"
            rejected.write_bytes(b"old")
            own.write_bytes(b"also old")
            tracker = self._tracker(rejected)
            with mock.patch.object(pose_tracker, "YOLO", self._fake_yolo({str(rejected), str(own)})), \
                    mock.patch("core.trt_engine.engine_path_for", return_value=own):
                tracker.warmup(object())
            self.assertEqual(tracker.model_path, SETTINGS.pose_model_pt)

    def test_the_rejected_engine_is_never_retried_as_its_own_replacement(self):
        from core import pose_tracker

        with tempfile.TemporaryDirectory() as tmp:
            rejected = Path(tmp) / "pose_engine_gpu_640_max1920.engine"
            rejected.write_bytes(b"old")
            with mock.patch("core.trt_engine.engine_path_for", return_value=rejected):
                self.assertIsNone(pose_tracker._own_engine_besides(rejected))


class PoseTaskTests(unittest.TestCase):
    """This GPU's engine file name does not say "pose"; every load must."""

    def test_ultralytics_would_guess_detect_from_the_cached_engines_name(self):
        from ultralytics.nn.tasks import guess_model_task

        name = trt_engine.engine_path_for("models", "NVIDIA GeForce RTX 5060").name
        self.assertEqual(guess_model_task(name), "detect")

    def test_every_model_load_in_the_tracker_says_pose(self):
        source = (Path(__file__).resolve().parents[1] / "core" / "pose_tracker.py").read_text(encoding="utf-8")
        loads = [line for line in source.splitlines() if "YOLO(" in line and "import" not in line]
        self.assertTrue(loads)
        for line in loads:
            self.assertIn("task=POSE_TASK", line)
