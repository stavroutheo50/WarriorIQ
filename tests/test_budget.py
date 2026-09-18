"""The realtime budget governor: plan once, never below the quality floor."""

import unittest

from core.config import SETTINGS
from core.pose_tracker import QualityController


class BudgetPlanTests(unittest.TestCase):
    def _controller(self, source_fps=31.22):
        return QualityController(source_fps, 480, 220)

    def test_a_run_already_inside_the_budget_is_left_alone(self):
        q = self._controller()
        before = q.stride
        # 40 frames in 1s of wall for 20s of video: far ahead of the budget.
        q.plan_for_budget(analyzed_frames=40, processed_seconds=20.0,
                          elapsed_seconds=1.0, segment_duration=109.0)
        self.assertEqual(q.stride, before)
        self.assertEqual(q.budget_reason, "on_track")
        self.assertTrue(q.budget_expected_met)

    def test_a_slow_run_samples_less(self):
        q = self._controller()
        before = q.stride
        # 40 frames took 30s of wall for 3s of video - badly behind.
        q.plan_for_budget(analyzed_frames=40, processed_seconds=3.0,
                          elapsed_seconds=30.0, segment_duration=109.0)
        self.assertGreater(q.stride, before)
        self.assertEqual(q.mode, "deadline")

    def test_it_never_samples_below_the_quality_floor(self):
        """Buying the budget past this point buys a number and loses the fight.

        The action windows in core/ are 0.6-1.5s; a round sampled thinner than
        min_tracking_fps cannot hold a strike.
        """
        q = self._controller()
        # Absurdly slow: the arithmetic alone would ask for an enormous stride.
        q.plan_for_budget(analyzed_frames=40, processed_seconds=0.5,
                          elapsed_seconds=300.0, segment_duration=109.0)
        self.assertGreaterEqual(q.effective_fps, min(q.source_fps, SETTINGS.min_tracking_fps) - 1e-6)
        self.assertFalse(q.budget_expected_met)
        self.assertEqual(q.budget_reason, "cannot_meet_budget_above_quality_floor")

    def test_it_plans_once_and_then_stops(self):
        """A controller that keeps re-deciding is the one that was switched off
        for making identical fights follow different frame paths."""
        q = self._controller()
        q.plan_for_budget(40, 3.0, 30.0, 109.0)
        chosen = q.stride
        q.plan_for_budget(400, 4.0, 600.0, 109.0)
        self.assertEqual(q.stride, chosen)

    def test_it_waits_for_a_real_sample_before_deciding(self):
        q = self._controller()
        before = q.stride
        q.plan_for_budget(analyzed_frames=5, processed_seconds=0.1,
                          elapsed_seconds=9.0, segment_duration=109.0)
        self.assertEqual(q.stride, before)
        self.assertFalse(q.planned)


if __name__ == "__main__":
    unittest.main()


class BudgetHonestyTests(unittest.TestCase):
    """A plan made from a stale cost must not claim it will meet the budget.

    plan_for_budget takes the frame cost from machine_profile.json so that two
    runs of one video plan the same stride - without that, the planner reads a
    clock and identical input plans differently, which moved fighter B's
    coverage from 0.260 to 0.728 on one fight.

    The cost of that determinism is that the stored number can be wrong for the
    run happening now: something else is using the card, or the workload is not
    what the profile was measured on. Measured here on 2026-09-18, a stored
    0.083 s/frame against an actual ~2.4 s/frame still produced
    budget_expected_met = True while the run missed its deadline about
    thirtyfold - and the report published that prediction to the user.

    So the stride keeps coming from the stored cost, and the PREDICTION now
    comes from what this run is actually achieving.
    """

    def _controller(self, stored_cost, source_fps=30.0):
        from unittest import mock

        from core import machine_profile
        q = QualityController(source_fps, 640, 480)
        self._patch = mock.patch.object(
            machine_profile, "frame_cost", return_value=stored_cost)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        return q

    def test_a_machine_slower_than_its_profile_stops_promising(self):
        q = self._controller(stored_cost=0.02)
        # 60 frames took 60s: one second per frame, fifty times the stored cost.
        q.plan_for_budget(analyzed_frames=60, processed_seconds=2.0,
                          elapsed_seconds=60.0, segment_duration=120.0)
        self.assertFalse(q.budget_expected_met)
        self.assertEqual(q.budget_reason, "machine_slower_than_profile")
        # Both numbers travel with the verdict, or it cannot be checked.
        self.assertAlmostEqual(q.budget_cost_planned, 0.02)
        self.assertAlmostEqual(q.budget_cost_observed, 1.0)

    def test_a_machine_matching_its_profile_still_promises(self):
        q = self._controller(stored_cost=0.02)
        # 60 frames in 1.2s is exactly the stored cost, and far inside budget.
        q.plan_for_budget(analyzed_frames=60, processed_seconds=30.0,
                          elapsed_seconds=1.2, segment_duration=120.0)
        self.assertTrue(q.budget_expected_met)
        self.assertEqual(q.budget_reason, "on_track")

    def test_ordinary_noise_does_not_flip_the_promise(self):
        """A prediction that flickers on normal variation is its own lie."""
        q = self._controller(stored_cost=0.02)
        # 40% slower than stored - real, but under the 1.5x guard - and still
        # comfortably inside the budget.
        q.plan_for_budget(analyzed_frames=60, processed_seconds=30.0,
                          elapsed_seconds=1.68, segment_duration=120.0)
        self.assertTrue(q.budget_expected_met)
        self.assertEqual(q.budget_reason, "on_track")

    def test_the_stride_is_unchanged_by_the_honesty_check(self):
        """Determinism is the reason the stored cost exists. Keep it."""
        slow = self._controller(stored_cost=0.02)
        slow.plan_for_budget(analyzed_frames=60, processed_seconds=2.0,
                             elapsed_seconds=60.0, segment_duration=120.0)
        self.assertEqual(slow.stride, slow.planned_stride)
        # The verdict changed; the frame path did not.
        self.assertFalse(slow.budget_expected_met)
