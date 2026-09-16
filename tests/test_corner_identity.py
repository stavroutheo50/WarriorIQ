"""Corner colour may argue against a candidate, and may never pick one.

Every other signal in core/identity.py compares a candidate to a remembered
template, so all of them drift together: once tracking has slid onto the other
fighter, the template *is* the other fighter and every term agrees. Corner
colour is fixed by the competition rules rather than by what was seen last,
which is why it can still disagree.

It refuses, and never endorses. A bonus would let a wrong reading lock identity
onto the wrong fighter with more confidence than before - worse than the
problem it solves. A refusal sends the fighter to recovery, which is the
behaviour this manager already prefers: missing briefly beats tracking the
wrong human.

A weighted penalty was tried first and measured useless: a conflicting
candidate that was otherwise a perfect match still scored 0.864 against a keep
threshold of 0.46 - exactly the swap this is meant to stop.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.corner import CLEAR_COLOUR
from core.identity import CORNER_MARGIN, IdentityManager
from core.types import PersonObservation


def person(track_id, x, y, *, red=None, blue=None, conf=0.9):
    box = np.array([x, y, x + 40, y + 110], dtype=np.float32)
    return PersonObservation(
        track_id=track_id, box=box, confidence=conf,
        keypoints=None, corner_red=red, corner_blue=blue)


def manager(**corners):
    a, b = person(1, 100, 200), person(2, 400, 200)
    m = IdentityManager(a, b, 0, 30.0)
    if corners:
        m.set_corners(corners.get("region", "gear"),
                      corners.get("a"), corners.get("b"))
    return m


class DefaultIsUnchangedTests(unittest.TestCase):
    """The safety property: no corner, no difference. Not approximately."""

    def test_no_corner_region_leaves_the_score_bit_identical(self):
        plain = manager()
        # Colours present on the detection but no region decided for the fight:
        # the reading must be ignored entirely, not weighed at low confidence.
        candidate = person(1, 105, 205, red=0.0, blue=1.0)
        bare = person(1, 105, 205)
        self.assertEqual(plain._score(plain.a, candidate),
                         plain._score(plain.a, bare))

    def test_a_fight_with_no_readable_corner_turns_the_signal_off(self):
        m = manager(region=None, a=None, b=None)
        self.assertIsNone(m.corner_region)
        self.assertIsNone(m.a.corner)
        candidate = person(1, 105, 205, red=0.0, blue=1.0)
        self.assertFalse(m._wears_the_other_corner(m.a, candidate))

    def test_both_fighters_in_one_colour_turns_the_signal_off(self):
        """Measured on real footage: two competitors both in blue.

        Nothing can separate them, and a corner that cannot tell them apart
        must not be allowed to penalise either.
        """
        m = manager(region="gear", a="blue", b="blue")
        self.assertIsNone(m.corner_region)
        self.assertIsNone(m.a.corner)
        self.assertIsNone(m.b.corner)


class ConflictTests(unittest.TestCase):
    def setUp(self):
        self.m = manager(region="gear", a="red", b="blue")

    def test_the_opponents_colour_is_a_conflict(self):
        wrong = person(1, 105, 205, red=0.02, blue=0.95)
        self.assertTrue(self.m._wears_the_other_corner(self.m.a, wrong))

    def test_the_fighters_own_colour_is_not(self):
        right = person(1, 105, 205, red=0.95, blue=0.02)
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, right))

    def test_an_uncoloured_detection_is_not_a_conflict(self):
        """Absent is not evidence of the opposite."""
        self.assertFalse(self.m._wears_the_other_corner(
            self.m.a, person(1, 105, 205, red=None, blue=None)))
        self.assertFalse(self.m._wears_the_other_corner(
            self.m.a, person(1, 105, 205, red=0.0, blue=0.0)))

    def test_a_close_reading_is_not_a_conflict(self):
        """Only an unambiguous conflict counts: refusing the right fighter
        is the cost of being wrong here."""
        murky = person(1, 105, 205, red=0.30, blue=0.31)
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, murky))

    def test_a_reading_below_the_clear_threshold_is_not_a_conflict(self):
        faint = person(1, 105, 205, red=0.0, blue=CLEAR_COLOUR - 0.01)
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, faint))

    def test_the_two_fighters_see_the_conflict_from_opposite_sides(self):
        blue_person = person(1, 105, 205, red=0.02, blue=0.95)
        self.assertTrue(self.m._wears_the_other_corner(self.m.a, blue_person))
        self.assertFalse(self.m._wears_the_other_corner(self.m.b, blue_person))


class ScoringTests(unittest.TestCase):
    def test_a_conflicting_candidate_is_refused_outright(self):
        m = manager(region="gear", a="red", b="blue")
        wrong = person(1, 105, 205, red=0.02, blue=0.95)
        self.assertLessEqual(m._score(m.a, wrong), -100)
        self.assertEqual(m.a.last_refusal, "wrong_corner")
        self.assertEqual(m.rejections.get("wrong_corner"), 1)

    def test_the_right_colour_earns_nothing(self):
        """It refuses and never endorses: the corner cannot pick a winner."""
        m = manager(region="gear", a="red", b="blue")
        neutral = person(1, 105, 205)
        right = person(1, 105, 205, red=0.95, blue=0.02)
        self.assertAlmostEqual(m._score(m.a, neutral), m._score(m.a, right), places=6)

    def test_the_distance_gate_still_wins_when_both_apply(self):
        """A candidate both far away and in the wrong colour is refused for
        being far away, because that gate is checked first and its count is
        what the diagnostics have always meant."""
        m = manager(region="gear", a="red", b="blue")
        far = person(1, 5000, 5000, red=0.02, blue=0.95)
        self.assertLessEqual(m._score(m.a, far), -100)
        self.assertEqual(m.a.last_refusal, "too_far_to_be_them")

    def test_the_swap_case_is_actually_stopped(self):
        """The case this exists for, and the one a penalty could not reach.

        Tracking has slid onto the other fighter: the candidate sits exactly
        where the fighter was predicted, so position and IoU are perfect, and
        the rolling template has drifted to match. Every comparative signal
        agrees. Only the corner disagrees.
        """
        m = manager(region="gear", a="red", b="blue")
        perfect_match = person(1, 100, 200)
        self.assertGreaterEqual(m._score(m.a, perfect_match), 0.46)
        same_person_wrong_corner = person(1, 100, 200, red=0.02, blue=0.95)
        self.assertLessEqual(m._score(m.a, same_person_wrong_corner), -100)

    def test_a_margin_is_required_before_refusing(self):
        m = manager(region="gear", a="red", b="blue")
        just_under = person(1, 105, 205, red=0.30, blue=0.30 + CORNER_MARGIN - 0.01)
        just_over = person(1, 105, 205, red=0.30, blue=0.30 + CORNER_MARGIN + 0.01)
        self.assertFalse(m._wears_the_other_corner(m.a, just_under))
        self.assertTrue(m._wears_the_other_corner(m.a, just_over))


if __name__ == "__main__":
    unittest.main()
