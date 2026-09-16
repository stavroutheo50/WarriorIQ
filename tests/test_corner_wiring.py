"""What the analyser does with a fight's corner reading and assignment.

The colour arithmetic is tested in test_corner.py and the refusal it feeds is
tested in test_corner_identity.py. What is left, and what this file covers, is
the join between them: a region and a pairing arrive, and either both fighters
end up wearing a corner or the whole signal stays off.

Off is the state that matters most. Every identity score is bit-identical to
what it was before corners existed while the region is None, so every path that
cannot produce two clearly opposite colours has to end there rather than
somewhere approximately there.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.analyzer import _apply_corner_reading
from core.corner import CornerAssignment, CornerReading
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
    manager = IdentityManager(observation(1, 100), observation(2, 400), 0, 30.0)
    return manager, FakeTracker()


def decided(region="gear", separation=0.8, frames=40):
    return CornerReading(region=region, separation=separation,
                         frames_scored=frames, frames_clean=frames,
                         decided=True, reason="decided")


def assigned(frames=40):
    return CornerAssignment(corner_a="red", corner_b="blue", frames=frames,
                            red_a=0.71, blue_a=0.04, red_b=0.06, blue_b=0.66)


def unassigned(frames=40):
    """Both fighters held, neither readable as one colour - the measured case."""
    return CornerAssignment(corner_a=None, corner_b=None, frames=frames,
                            red_a=0.41, blue_a=0.44, red_b=0.61, blue_b=0.26)


class AppliedReadingTests(unittest.TestCase):
    def test_two_opposite_colours_turn_the_signal_on(self):
        manager, tracker = setup()
        _apply_corner_reading(decided(), assigned(), manager, tracker, "round")
        self.assertEqual(manager.corner_region, "gear")
        self.assertEqual((manager.a.corner, manager.b.corner), ("red", "blue"))
        self.assertEqual(tracker.region, "gear")

    def test_the_tracker_is_told_the_same_thing_the_manager_settled_on(self):
        """Not the same thing the reading said.

        These two have to agree or detections are coloured for a region the
        manager is not judging against, which is the one combination that
        could refuse a fighter on a reading nothing checked.
        """
        manager, tracker = setup()
        _apply_corner_reading(decided(), unassigned(), manager, tracker, "round")
        self.assertIsNone(manager.corner_region)
        self.assertEqual(tracker.region, manager.corner_region)

    def test_a_region_nobody_can_be_assigned_in_turns_the_signal_off(self):
        manager, tracker = setup()
        _apply_corner_reading(decided(), unassigned(), manager, tracker, "round")
        self.assertIsNone(manager.corner_region)
        self.assertIsNone(manager.a.corner)
        self.assertIsNone(manager.b.corner)

    def test_an_undecided_reading_turns_the_signal_off(self):
        manager, tracker = setup()
        _apply_corner_reading(CornerReading(), assigned(), manager, tracker, "round")
        self.assertIsNone(manager.corner_region)
        self.assertIsNone(tracker.region)

    def test_no_assignment_at_all_turns_the_signal_off(self):
        manager, tracker = setup()
        _apply_corner_reading(decided(), None, manager, tracker, "round")
        self.assertIsNone(manager.corner_region)


class DiagnosticTests(unittest.TestCase):
    def test_never_holding_both_and_a_bad_read_are_told_apart(self):
        """Both end as corners off, and they want opposite fixes.

        Identity never holding both fighters together is a tracking problem.
        Holding them and never reading opposite colours is a colour problem.
        The result cannot distinguish them, so the log has to.
        """
        manager, tracker = setup()
        with self.assertLogs("warrioriq.analysis", level="INFO") as never:
            _apply_corner_reading(decided(), CornerAssignment(),
                                  manager, tracker, "round")
        self.assertIn("reason=never_both_held", " ".join(never.output))

        manager, tracker = setup()
        with self.assertLogs("warrioriq.analysis", level="INFO") as bad:
            _apply_corner_reading(decided(), unassigned(), manager, tracker, "round")
        joined = " ".join(bad.output)
        self.assertIn("reason=not_opposite", joined)
        self.assertIn("a_red=0.41", joined)
        self.assertIn("a_blue=0.44", joined)

    def test_an_undecided_reading_is_not_reported_as_an_assignment_failure(self):
        """There was no region to assign anyone in, so nothing failed."""
        manager, tracker = setup()
        with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
            _apply_corner_reading(CornerReading(), assigned(),
                                  manager, tracker, "round")
        self.assertNotIn("analysis_corner_unassigned", " ".join(logs.output))

    def test_both_windows_are_reported_not_just_the_region_one(self):
        """A run that tracked the wrong fighter has to be explainable later,
        and the two halves can be decided from very different numbers of
        frames - the region from every frame, the pairing only from frames
        where identity held both."""
        manager, tracker = setup()
        with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
            _apply_corner_reading(decided(frames=40), assigned(frames=12),
                                  manager, tracker, "round")
        line = " ".join(logs.output)
        self.assertIn("decided_from=round", line)
        self.assertIn("frames=40", line)
        self.assertIn("assigned_from=12", line)

    def test_an_unreadable_frame_is_reported_rather_than_guessed(self):
        manager, tracker = setup()
        with self.assertLogs("warrioriq.analysis", level="INFO") as logs:
            _apply_corner_reading(
                CornerReading(reason="no frame had two fighters to compare"),
                CornerAssignment(), manager, tracker, "round")
        self.assertIn("region=none", " ".join(logs.output))


if __name__ == "__main__":
    unittest.main()
