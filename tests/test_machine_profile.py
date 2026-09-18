"""The same video must plan the same stride twice.

It did not. `plan_for_budget` measured what a frame cost inside the run, which
means it read a clock, and a clock moves with machine load. Measured on fight 1
- same file, same seeds, same commit, one analysis per process on an idle card
- the planner chose stride 3 on some runs and stride 4 on others, and fighter
B's coverage went 0.260 or 0.728 accordingly.

Across all three fights the flip is worth +0.468, -0.138 and 0.009, in
opposite directions, so there is no better stride to prefer. What there is, is
a requirement that the same input plans the same way - which is what these
tests hold down.
"""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from core import machine_profile
from core.config import SETTINGS
from core.pose_tracker import QualityController


class SnapTests(unittest.TestCase):
    """Rounding is what makes a measured number stick."""

    def test_measurements_inside_one_bucket_snap_together(self):
        """Snapping narrows the window where jitter changes the answer.

        It does not close it, and this test is careful not to claim it does:
        two measurements either side of a bucket edge still snap apart, which
        is inherent to quantising anything. What closes it is that the value
        is written once and read back afterwards - see
        test_a_second_measurement_does_not_overwrite_the_first and
        test_two_runs_at_different_speeds_plan_the_same_stride, which are the
        tests that actually hold the regression down.
        """
        base = machine_profile.snap(0.1700)
        for measured in (0.1550, 0.1600, 0.1650, 0.1730):
            self.assertEqual(machine_profile.snap(measured), base)

    def test_a_boundary_is_still_a_boundary(self):
        """Stated outright rather than left as a surprise for the next reader.

        0.1730 and 0.1740 are a thousandth of a second apart and snap to
        different buckets. That is not a defect in the rounding, it is what
        rounding is - and it is why the stored value is written once rather
        than re-derived from each run's measurement.
        """
        low = machine_profile.snap(0.1730)
        high = machine_profile.snap(0.1740)
        self.assertNotEqual(low, high)
        self.assertLess(high / low, machine_profile.BUCKET_RATIO ** 2)

    def test_a_real_difference_still_moves(self):
        """Snapping must not flatten the range it has to describe."""
        self.assertNotEqual(machine_profile.snap(0.05), machine_profile.snap(0.20))
        self.assertLess(machine_profile.snap(0.05), machine_profile.snap(0.20))

    def test_snapping_is_idempotent(self):
        """Or a stored value would drift every time it was read and rewritten."""
        once = machine_profile.snap(0.1734)
        self.assertEqual(machine_profile.snap(once), once)

    def test_nonsense_is_refused_rather_than_stored(self):
        for value in (0.0, -1.0, float("nan")):
            self.assertEqual(machine_profile.snap(value), 0.0)

    def test_the_bucket_is_wider_than_the_jitter_that_caused_the_bug(self):
        """The two runs differed by enough to cross a stride boundary."""
        self.assertGreater(machine_profile.BUCKET_RATIO, 1.05)
        self.assertLess(machine_profile.BUCKET_RATIO, 1.30)


class ProfileFileTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "machine_profile.json")
        self.patch = mock.patch.dict(os.environ, {"WARRIORIQ_MACHINE_PROFILE": self.path})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_nothing_measured_yet_reads_as_nothing(self):
        self.assertIsNone(machine_profile.frame_cost("Test GPU", 1248))

    def test_what_was_written_is_what_comes_back(self):
        written = machine_profile.record_frame_cost("Test GPU", 1248, 0.173)
        self.assertEqual(machine_profile.frame_cost("Test GPU", 1248), written)

    def test_a_second_measurement_does_not_overwrite_the_first(self):
        """Write-once is the whole point.

        Blending each run's measurement in would move the stored value a
        little every time, and a stride derived from a moving number goes on
        flipping - just more slowly, and less visibly.
        """
        first = machine_profile.record_frame_cost("Test GPU", 1248, 0.173)
        second = machine_profile.record_frame_cost("Test GPU", 1248, 0.290)
        self.assertEqual(first, second)
        self.assertEqual(machine_profile.frame_cost("Test GPU", 1248), first)

    def test_different_hardware_and_sizes_are_kept_apart(self):
        machine_profile.record_frame_cost("Card A", 1248, 0.10)
        machine_profile.record_frame_cost("Card B", 1248, 0.40)
        machine_profile.record_frame_cost("Card A", 640, 0.03)
        self.assertNotEqual(machine_profile.frame_cost("Card A", 1248),
                            machine_profile.frame_cost("Card B", 1248))
        self.assertNotEqual(machine_profile.frame_cost("Card A", 1248),
                            machine_profile.frame_cost("Card A", 640))

    def test_an_unreadable_profile_costs_a_plan_and_not_the_run(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{ this is not json")
        self.assertIsNone(machine_profile.frame_cost("Test GPU", 1248))
        self.assertTrue(machine_profile.record_frame_cost("Test GPU", 1248, 0.173))

    def test_a_profile_holding_rubbish_for_this_key_is_ignored(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"Test GPU@1248": "fast"}, handle)
        self.assertIsNone(machine_profile.frame_cost("Test GPU", 1248))


class PlanningIsReproducibleTests(unittest.TestCase):
    """The property the whole module exists for."""

    def setUp(self):
        import tempfile

        self.path = os.path.join(tempfile.mkdtemp(), "machine_profile.json")
        self.patch = mock.patch.dict(os.environ, {"WARRIORIQ_MACHINE_PROFILE": self.path})
        self.patch.start()
        self.previous = SETTINGS.frame_cost_seconds
        object.__setattr__(SETTINGS, "frame_cost_seconds", 0.0)

    def tearDown(self):
        object.__setattr__(SETTINGS, "frame_cost_seconds", self.previous)
        self.patch.stop()

    def planner(self):
        quality = QualityController(30.0, 480, 220, measured_imgsz=1248)
        quality.device_name = "Test GPU"
        return quality

    def plan(self, elapsed_seconds):
        """One planning decision, from a run that took this long to calibrate."""
        quality = self.planner()
        quality.plan_for_budget(40, 5.0, elapsed_seconds, 94.3)
        return quality

    def test_two_runs_at_different_speeds_plan_the_same_stride(self):
        """The measured regression, in the form it actually took.

        The first run writes the cost down; the second reads it back and plans
        identically however fast it happened to be. Before this, a 25%
        difference in calibration speed was enough to move fighter B's
        coverage by 0.468.
        """
        first = self.plan(7.0)
        self.assertEqual(first.budget_cost_source, "measured")
        for elapsed in (5.0, 7.0, 9.0, 12.0):
            later = self.plan(elapsed)
            self.assertEqual(later.budget_cost_source, "profile")
            self.assertEqual(later.stride, first.stride,
                             "elapsed %.1fs planned differently" % elapsed)

    def test_a_configured_cost_beats_a_stored_one(self):
        self.plan(7.0)
        object.__setattr__(SETTINGS, "frame_cost_seconds", 0.40)
        quality = self.plan(7.0)
        self.assertEqual(quality.budget_cost_source, "configured")

    def test_a_pinned_stride_skips_planning_entirely(self):
        previous = SETTINGS.force_tracking_stride
        try:
            object.__setattr__(SETTINGS, "force_tracking_stride", 4)
            quality = self.plan(7.0)
        finally:
            object.__setattr__(SETTINGS, "force_tracking_stride", previous)
        self.assertEqual(quality.stride, 4)
        self.assertEqual(quality.budget_cost_source, "pinned")
        self.assertEqual(quality.budget_reason, "stride_pinned")
        self.assertIsNone(machine_profile.frame_cost("Test GPU", 1248),
                          "a pinned run must not write a profile it never measured")

    def test_a_slower_machine_still_samples_more_sparsely(self):
        """Determinism must not cost the adaptivity it was protecting.

        Two machines, each measured once: the slow one has to plan a larger
        stride, or the budget planning has stopped doing its job.
        """
        import tempfile

        strides = []
        for cost, name in ((0.02, "Fast card"), (0.40, "Slow card")):
            with mock.patch.dict(os.environ, {
                    "WARRIORIQ_MACHINE_PROFILE":
                        os.path.join(tempfile.mkdtemp(), "profile.json")}):
                quality = self.planner()
                quality.device_name = name
                quality.plan_for_budget(40, 5.0, cost * 40, 94.3)
                strides.append(quality.stride)
        self.assertLess(strides[0], strides[1])

    def test_the_report_can_explain_a_stride_without_a_second_run(self):
        quality = self.plan(7.0)
        self.assertIn(quality.budget_cost_source,
                      ("measured", "profile", "configured", "pinned"))
        self.assertIsNotNone(quality.planned_stride)


if __name__ == "__main__":
    unittest.main()


class PipelineVersionTests(unittest.TestCase):
    """A stored cost must not outlive the pipeline that measured it.

    Write-once is what makes two runs of one video plan the same stride. Its
    failure mode is that a change making frames cheaper leaves the old number in
    place for ever. That happened: fixing a second TensorRT execution context on
    2026-09-18 made frames far cheaper on this machine, and the profile went on
    saying 0.083081 s.

    A machine slower than its profile misses its deadline and plan_for_budget
    reports it. A machine FASTER than its profile silently reduces sampling,
    because the stale number says it cannot afford the frames - measured here,
    stride 2 cut to 3 with three times the needed headroom. Frames are identity
    and strike evidence, and nothing reports that loss.
    """

    def setUp(self):
        import tempfile
        from unittest import mock

        handle, path = tempfile.mkstemp(prefix="warrioriq-ver-", suffix=".json")
        os.close(handle)
        os.unlink(path)
        self.path = path
        patcher = mock.patch.dict(os.environ, {"WARRIORIQ_MACHINE_PROFILE": path})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

    def test_a_cost_from_an_older_pipeline_is_not_read(self):
        import json

        old_key = "%s@%d" % ("NVIDIA GeForce RTX 5060", 1600)   # the pre-version shape
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({old_key: 0.083081}, handle)
        self.assertIsNone(
            machine_profile.frame_cost("NVIDIA GeForce RTX 5060", 1600),
            "a cost measured by an older pipeline is still being planned from")

    def test_a_cost_from_this_pipeline_is_read(self):
        machine_profile.record_frame_cost("NVIDIA GeForce RTX 5060", 1600, 0.02)
        self.assertIsNotNone(machine_profile.frame_cost("NVIDIA GeForce RTX 5060", 1600))

    def test_the_version_is_part_of_the_key(self):
        key = machine_profile._key("NVIDIA GeForce RTX 5060", 1600)
        self.assertTrue(key.startswith(f"v{machine_profile.PIPELINE_VERSION}|"), key)
        # Determinism within a version is the thing being protected.
        self.assertEqual(key, machine_profile._key("NVIDIA GeForce RTX 5060", 1600))
