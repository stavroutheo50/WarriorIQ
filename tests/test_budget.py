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
