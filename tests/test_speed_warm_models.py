"""Warm models between jobs and a stage-by-stage timing record (QA, 2026-10-07, item 26).

Every job rebuilt SAM2 (SamRecovery.release dropped it), and a frame-pass
rescue after the sweep rebuilt it again mid-job. The idle warmup warmed only
the joint refiner. The report timed the SAM2 sweep and the frame pass and left
model loading, the backward pass and the report as an unexplained remainder.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from core import sam_recovery
from core.config import SETTINGS


@contextmanager
def setting(name, value):
    previous = getattr(SETTINGS, name)
    object.__setattr__(SETTINGS, name, value)
    try:
        yield
    finally:
        object.__setattr__(SETTINGS, name, previous)


class FakePredictor:
    def __init__(self):
        self.device = "cuda"
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = device
        return self


class WarmSamTests(unittest.TestCase):
    def setUp(self):
        sam_recovery._warm.update(predictor=None, model_id=None)
        self.addCleanup(sam_recovery._warm.update, predictor=None, model_id=None)
        self.builds = []

        def build():
            predictor = FakePredictor()
            self.builds.append(predictor)
            return predictor

        patcher = patch.object(sam_recovery, "_build_predictor", build)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name, value in (("sam_recovery_enabled", True), ("sam_keep_warm", True)):
            manager = setting(name, value)
            manager.__enter__()
            self.addCleanup(manager.__exit__, None, None, None)

    def test_sweep_then_rescue_then_next_job_build_once(self):
        job = sam_recovery.SamRecovery()
        self.assertTrue(job._load())
        job.release()                       # after the sweep: parked, card freed
        parked = sam_recovery._warm["predictor"]
        self.assertIs(parked, self.builds[0])
        self.assertEqual(parked.device, "cpu")
        self.assertTrue(job._load())        # a frame-pass rescue
        self.assertEqual(job.predictor.device, "cuda")
        job.release()
        following = sam_recovery.SamRecovery()
        self.assertTrue(following._load())  # the next job
        self.assertEqual(len(self.builds), 1)
        self.assertIs(following.predictor, self.builds[0])
        self.assertGreaterEqual(job.load_seconds, 0.0)

    def test_switched_off_rebuilds_as_before(self):
        with setting("sam_keep_warm", False):
            job = sam_recovery.SamRecovery()
            job._load()
            job.release()
            self.assertIsNone(sam_recovery._warm["predictor"])
            job._load()
        self.assertEqual(len(self.builds), 2)

    def test_a_different_model_is_not_reused(self):
        job = sam_recovery.SamRecovery()
        job._load()
        job.release()
        with setting("sam_model_id", "facebook/some-other-sam"):
            sam_recovery.SamRecovery()._load()
        self.assertEqual(len(self.builds), 2)

    def test_a_failed_move_falls_back_to_building(self):
        job = sam_recovery.SamRecovery()
        job._load()
        job.release()

        def broken(device):
            raise RuntimeError("CUDA out of memory")

        sam_recovery._warm["predictor"].to = broken
        self.assertTrue(sam_recovery.SamRecovery()._load())
        self.assertEqual(len(self.builds), 2)

    def test_preload_needs_a_card(self):
        with patch.object(sam_recovery.torch.cuda, "is_available", return_value=False):
            self.assertFalse(sam_recovery.preload())
        self.assertEqual(self.builds, [])


class StageSecondsTests(unittest.TestCase):
    def test_stages_add_up_to_the_wait(self):
        from core.analyzer import _stage_seconds

        stages = _stage_seconds(total=55.7, before_analysis=6.0, model_setup=3.1, sam_sweep=20.0,
                                sam_model_load=4.0, before_frame_pass=0.2, frame_pass=22.0)
        summed = sum(value for key, value in stages.items() if key != "sam_model_load_included_above")
        self.assertAlmostEqual(summed, 55.7, delta=0.05)
        self.assertAlmostEqual(stages["after_frame_pass"], 4.4, delta=0.01)
        self.assertEqual(stages["sam_model_load_included_above"], 4.0)

    def test_never_negative(self):
        from core.analyzer import _stage_seconds

        stages = _stage_seconds(total=1.0, before_analysis=0.0, model_setup=0.5, sam_sweep=0.5,
                                sam_model_load=0.0, before_frame_pass=-0.01, frame_pass=0.2)
        self.assertTrue(all(value >= 0 for value in stages.values()))

    def test_report_records_it(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "core" / "analyzer.py").read_text(encoding="utf-8")
        self.assertIn('"stage_seconds": _stage_seconds(', source)
        self.assertIn("sam_recovery.release()\n        if torch.cuda.is_available():", source)


class IdleWarmupTests(unittest.TestCase):
    def test_idle_warmup_also_warms_detector_and_sam(self):
        import worker

        calls = []

        class Tracker:
            def warmup(self, frame):
                calls.append(("pose", frame.shape))

        with patch("core.rtm_pose.warmup", lambda: calls.append(("rtm",))), \
                patch("core.analyzer.get_pose_tracker", lambda: Tracker()), \
                patch("core.sam_recovery.preload", lambda: calls.append(("sam",))):
            worker._start_idle_pose_warmup().join(10)
        self.assertEqual([call[0] for call in calls], ["rtm", "pose", "sam"])

    def test_one_failure_does_not_stop_the_rest(self):
        import worker

        calls = []

        def broken():
            raise RuntimeError("no rtmlib")

        class Tracker:
            def warmup(self, frame):
                calls.append("pose")

        with patch("core.rtm_pose.warmup", broken), \
                patch("core.analyzer.get_pose_tracker", lambda: Tracker()), \
                patch("core.sam_recovery.preload", lambda: calls.append("sam")), \
                self.assertLogs("warrioriq.worker", level="ERROR"):
            worker._start_idle_pose_warmup().join(10)
        self.assertEqual(calls, ["pose", "sam"])


if __name__ == "__main__":
    unittest.main()
