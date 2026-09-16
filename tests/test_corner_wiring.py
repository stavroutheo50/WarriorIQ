"""What the analyser does with a fight's corner reading.

The colour arithmetic is tested in test_corner.py and the refusal it feeds is
tested in test_corner_identity.py. What is left, and what this file covers, is
the join between them: a reading arrives, and either both fighters end up
wearing a corner or the whole signal stays off.

Off is the state that matters most. Every identity score is bit-identical to
what it was before corners existed while the region is None, so every path that
cannot produce two clearly opposite colours has to end there rather than
somewhere approximately there.
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from core.analyzer import _apply_corner_reading
from core.corner import CornerReading
from core.identity import IdentityManager
from core.types import PersonObservation


class FakeTracker:
    """Only the one method the helper is allowed to call on it."""

    def __init__(self):
        self.region = "untouched"

    def set_corner_region(self, region):
        self.region = region


def observation(track_id, x):
    box = np.array([x, 200, x + 40, 310], dtype=np.float32)
    return PersonObservation(track_id=track_id, box=box, confidence=0.9,
                             keypoints=None)


def setup():
    a, b = observation(1, 100), observation(2, 400)
    return IdentityManager(a, b, 0, 30.0), FakeTracker(), a, b


def decided(region="gear", separation=0.8, frames=40):
    return CornerReading(region=region, separation=separation,
                         frames_scored=frames, frames_clean=frames,
                         decided=True, reason="decided")


class AppliedReadingTests(unittest.TestCase):
    def test_two_opposite_colours_turn_the_signal_on(self):
        manager, tracker, a, b = setup()
        with mock.patch("core.analyzer.assign_corners", return_value=("red", "blue")):
            _apply_corner_reading(decided(), object(), a, b, manager, tracker, "round")
        self.assertEqual(manager.corner_region, "gear")
        self.assertEqual((manager.a.corner, manager.b.corner), ("red", "blue"))
        self.assertEqual(tracker.region, "gear")

    def test_the_tracker_is_told_the_same_thing_the_manager_settled_on(self):
        """Not the same thing the reading said.

        These two have to agree or detections are coloured for a region the
        manager is not judging against, which is the one combination that
        could refuse a fighter on a reading nothing checked.
        """
        manager, tracker, a, b = setup()
        with mock.patch("core.analyzer.assign_corners", return_value=(None, None)):
            _apply_corner_reading(decided(), object(), a, b, manager, tracker, "round")
        self.assertIsNone(manager.corner_region)
        self.assertEqual(tracker.region, manager.corner_region)

    def test_a_region_nobody_can_be_assigned_in_turns_the_signal_off(self):
        manager, tracker, a, b = setup()
        with mock.patch("core.analyzer.assign_corners", return_value=(None, None)):
            _apply_corner_reading(decided(), object(), a, b, manager, tracker, "round")
        self.assertIsNone(manager.corner_region)
        self.assertIsNone(manager.a.corner)
        self.assertIsNone(manager.b.corner)

    def test_an_undecided_reading_never_asks_who_is_which_colour(self):
        manager, tracker, a, b = setup()
        with mock.patch("core.analyzer.assign_corners") as assign:
            _apply_corner_reading(CornerReading(), object(), a, b,
                                  manager, tracker, "round")
        assign.assert_not_called()
        self.assertIsNone(manager.corner_region)
        self.assertIsNone(tracker.region)

    def test_a_fighter_missing_at_decision_time_turns_the_signal_off(self):
        """Identity can be mid-recovery on the frame the fight is decided on.

        There is then no box to read a colour out of, and guessing from the
        other fighter alone would assign a corner from one look at one person.
        """
        for pair in ((None, observation(2, 400)), (observation(1, 100), None)):
            manager, tracker, _, _ = setup()
            with mock.patch("core.analyzer.assign_corners") as assign:
                _apply_corner_reading(decided(), object(), pair[0], pair[1],
                                      manager, tracker, "round")
            assign.assert_not_called()
            self.assertIsNone(manager.corner_region)

    def test_where_the_decision_came_from_is_recorded(self):
        """A run that tracked the wrong fighter has to be explainable later,
        and "which frames decided the corner" is the first question."""
        manager, tracker, a, b = setup()
        with mock.patch("core.analyzer.assign_corners", return_value=("red", "blue")):
            with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
                _apply_corner_reading(decided(frames=40), object(), a, b,
                                      manager, tracker, "round")
        line = " ".join(logs.output)
        self.assertIn("analysis_corner", line)
        self.assertIn("decided_from=round", line)
        self.assertIn("frames=40", line)

    def test_a_missing_fighter_and_a_bad_read_are_told_apart(self):
        """Both end as corners off, and they want opposite fixes.

        Identity not holding both fighters on the decision frame is a timing
        problem. Both held and not reading as opposite colours is a colour
        problem. The result cannot distinguish them, so the log has to.
        """
        manager, tracker, a, b = setup()
        with mock.patch("core.analyzer.assign_corners", return_value=(None, None)):
            with self.assertLogs("warrioriq.analysis", level="INFO") as missing:
                _apply_corner_reading(decided(), object(), None, b,
                                      manager, tracker, "round")
        self.assertIn("reason=fighter_missing", " ".join(missing.output))

        manager, tracker, a, b = setup()
        with mock.patch("core.analyzer.assign_corners", return_value=(None, None)), \
             mock.patch("core.corner.score_detection", return_value=(0.1, 0.1)):
            with self.assertLogs("warrioriq.analysis", level="INFO") as unread:
                _apply_corner_reading(decided(), object(), a, b,
                                      manager, tracker, "round")
        joined = " ".join(unread.output)
        self.assertIn("reason=not_opposite", joined)
        self.assertIn("a_red=0.10", joined)

    def test_an_undecided_reading_is_not_reported_as_an_assignment_failure(self):
        """There was no region to assign anyone in, so nothing failed."""
        manager, tracker, a, b = setup()
        with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
            _apply_corner_reading(CornerReading(), object(), a, b,
                                  manager, tracker, "round")
        self.assertNotIn("analysis_corner_unassigned", " ".join(logs.output))

    def test_an_unreadable_frame_is_reported_rather_than_guessed(self):
        manager, tracker, a, b = setup()
        with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
            _apply_corner_reading(
                CornerReading(reason="no frame had two fighters to compare"),
                object(), a, b, manager, tracker, "round")
        self.assertIn("region=none", " ".join(logs.output))


if __name__ == "__main__":
    unittest.main()
