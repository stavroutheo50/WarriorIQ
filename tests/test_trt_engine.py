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

    def test_one_engine_covers_both_inference_sizes(self):
        """The engine is built at the largest size anything can ask for.

        Both the default and the low-resolution path run through the same
        engine, so building at the maximum means neither has to fall back.
        """
        from core.config import SETTINGS
        self.assertEqual(
            trt_engine.engine_size(),
            max(SETTINGS.default_imgsz, SETTINGS.low_resolution_imgsz))


class EnsureEngineTests(unittest.TestCase):
    def test_an_existing_engine_is_reused_not_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = trt_engine.engine_path_for(tmp)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"pretend engine")
            with mock.patch.object(trt_engine, "build_pose_engine") as build:
                got = trt_engine.ensure_pose_engine(tmp)
            self.assertEqual(got, target)
            build.assert_not_called()

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
