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

Two designs were measured before this one.

A weighted penalty does not work: a conflicting candidate that was otherwise a
perfect match still scored 0.864 against a keep threshold of 0.46 - exactly the
swap this is meant to stop.

A per-frame refusal works far too well. On a fight where both corners were
assigned correctly, it fired 2256 times and cost fighter B two thirds of its
coverage - 0.728 down to 0.260 - because the colour of one detection in one
frame flickers: gear leaves frame, a fighter turns, and the mat itself is red
and blue. What does not flicker is a whole track's history, which is what this
gate reads now.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.corner import CLEAR_COLOUR
from core.identity import (
    CORNER_CONFLICT_FRACTION,
    CORNER_MARGIN,
    CORNER_MIN_OBSERVATIONS,
    CORNER_WINDOW,
    IdentityManager,
)
from core.types import PersonObservation

RED = (0.95, 0.02)
BLUE = (0.02, 0.95)


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


def looked_at(m, track_id, colours, *, x=105, y=205):
    """Let the manager look at this track once per entry in `colours`.

    Goes through _remember_positions rather than writing the ledger directly,
    so these tests exercise the same path an analysis does.
    """
    for index, colour in enumerate(colours):
        red, blue = colour
        m._remember_positions([person(track_id, x, y, red=red, blue=blue)], index)


def enough(colour):
    return [colour] * CORNER_MIN_OBSERVATIONS


class DefaultIsUnchangedTests(unittest.TestCase):
    """The safety property: no corner, no difference. Not approximately."""

    def test_no_corner_region_leaves_the_score_bit_identical(self):
        plain = manager()
        looked_at(plain, 1, enough(BLUE))
        candidate = person(1, 105, 205, red=0.0, blue=1.0)
        bare = person(1, 105, 205)
        self.assertEqual(plain._score(plain.a, candidate),
                         plain._score(plain.a, bare))

    def test_a_fight_with_no_readable_corner_turns_the_signal_off(self):
        m = manager(region=None, a=None, b=None)
        self.assertIsNone(m.corner_region)
        self.assertIsNone(m.a.corner)
        looked_at(m, 1, enough(BLUE))
        self.assertFalse(m._wears_the_other_corner(m.a, person(1, 105, 205)))

    def test_both_fighters_in_one_colour_turns_the_signal_off(self):
        """Measured on real footage: two competitors both in blue.

        Nothing can separate them, and a corner that cannot tell them apart
        must not be allowed to penalise either.
        """
        m = manager(region="gear", a="blue", b="blue")
        self.assertIsNone(m.corner_region)
        self.assertIsNone(m.a.corner)
        self.assertIsNone(m.b.corner)


class OneFrameIsNotEvidenceTests(unittest.TestCase):
    """The regression this design exists to prevent."""

    def setUp(self):
        self.m = manager(region="gear", a="red", b="blue")

    def test_a_single_conflicting_look_does_not_refuse(self):
        looked_at(self.m, 1, [BLUE])
        self.assertFalse(self.m._wears_the_other_corner(
            self.m.a, person(1, 105, 205, red=0.02, blue=0.95)))

    def test_a_track_seen_briefly_is_never_refused(self):
        """A fresh track is the fighter coming back from a loss.

        Refusing it is precisely how the per-frame version destroyed coverage:
        the fighter is lost, reappears under a new track id, and is turned away
        before it has been looked at enough to judge.
        """
        looked_at(self.m, 1, [BLUE] * (CORNER_MIN_OBSERVATIONS - 1))
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))

    def test_the_last_look_needed_is_what_tips_it(self):
        looked_at(self.m, 1, [BLUE] * (CORNER_MIN_OBSERVATIONS - 1))
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))
        looked_at(self.m, 1, [BLUE])
        self.assertTrue(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))

    def test_a_few_bad_looks_among_good_ones_do_not_refuse(self):
        """Gear leaving frame for a moment must not cost the fighter."""
        looked_at(self.m, 1, [RED] * 12 + [BLUE] * 3)
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))

    def test_a_detection_with_no_track_cannot_accumulate_and_is_not_refused(self):
        self.assertFalse(self.m._wears_the_other_corner(
            self.m.a, person(None, 105, 205, red=0.02, blue=0.95)))


class SustainedConflictTests(unittest.TestCase):
    def setUp(self):
        self.m = manager(region="gear", a="red", b="blue")

    def test_a_track_that_keeps_reading_blue_is_refused_for_the_red_fighter(self):
        looked_at(self.m, 1, enough(BLUE))
        self.assertTrue(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))

    def test_the_fighters_own_colour_is_never_a_conflict(self):
        looked_at(self.m, 1, enough(RED))
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))

    def test_the_two_fighters_see_the_same_track_from_opposite_sides(self):
        looked_at(self.m, 1, enough(BLUE))
        candidate = person(1, 105, 205)
        self.assertTrue(self.m._wears_the_other_corner(self.m.a, candidate))
        self.assertFalse(self.m._wears_the_other_corner(self.m.b, candidate))

    def test_the_required_fraction_is_what_decides(self):
        total = 20
        conflicting = int(CORNER_CONFLICT_FRACTION * total)
        looked_at(self.m, 1, [BLUE] * (conflicting - 1) + [RED] * (total - conflicting + 1))
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))
        looked_at(self.m, 2, [BLUE] * conflicting + [RED] * (total - conflicting))
        self.assertTrue(self.m._wears_the_other_corner(self.m.a, person(2, 105, 205)))

    def test_the_window_forgets_and_a_track_can_recover(self):
        """A rolling window, so a track that has changed its mind is believed.

        Without this, one bad stretch would ban a track for the whole fight.
        """
        looked_at(self.m, 1, enough(BLUE))
        self.assertTrue(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))
        looked_at(self.m, 1, [RED] * CORNER_WINDOW)
        self.assertFalse(self.m._wears_the_other_corner(self.m.a, person(1, 105, 205)))


class WhatCountsAsALookTests(unittest.TestCase):
    def test_an_uncoloured_detection_is_not_recorded(self):
        m = manager(region="gear", a="red", b="blue")
        looked_at(m, 1, [(None, None)] * 20)
        self.assertFalse(m._wears_the_other_corner(m.a, person(1, 105, 205)))

    def test_a_murky_look_is_not_recorded_rather_than_recorded_as_neutral(self):
        """Absent is not evidence, and must not dilute the evidence either."""
        m = manager(region="gear", a="red", b="blue")
        looked_at(m, 1, [(0.30, 0.31)] * 20)
        self.assertEqual(len(m._track_corner.get(1, [])), 0)
        looked_at(m, 1, enough(BLUE))
        self.assertTrue(m._wears_the_other_corner(m.a, person(1, 105, 205)))

    def test_a_reading_below_the_clear_threshold_is_not_recorded(self):
        m = manager(region="gear", a="red", b="blue")
        looked_at(m, 1, [(0.0, CLEAR_COLOUR - 0.01)] * 20)
        self.assertEqual(len(m._track_corner.get(1, [])), 0)

    def test_a_look_inside_the_margin_is_not_recorded(self):
        m = manager(region="gear", a="red", b="blue")
        looked_at(m, 1, [(0.30, 0.30 + CORNER_MARGIN - 0.01)] * 20)
        self.assertEqual(len(m._track_corner.get(1, [])), 0)
        looked_at(m, 2, [(0.30, 0.30 + CORNER_MARGIN + 0.01)] * CORNER_MIN_OBSERVATIONS)
        self.assertEqual(len(m._track_corner.get(2, [])), CORNER_MIN_OBSERVATIONS)


class ScoringTests(unittest.TestCase):
    def test_a_sustained_conflict_is_refused_outright(self):
        m = manager(region="gear", a="red", b="blue")
        looked_at(m, 1, enough(BLUE))
        self.assertLessEqual(m._score(m.a, person(1, 105, 205)), -100)
        self.assertEqual(m.a.last_refusal, "wrong_corner")
        self.assertEqual(m.rejections.get("wrong_corner"), 1)

    def test_the_right_colour_earns_nothing(self):
        """It refuses and never endorses: the corner cannot pick a winner."""
        m = manager(region="gear", a="red", b="blue")
        looked_at(m, 1, enough(RED))
        neutral = person(1, 105, 205)
        right = person(1, 105, 205, red=0.95, blue=0.02)
        self.assertAlmostEqual(m._score(m.a, neutral), m._score(m.a, right), places=6)

    def test_the_distance_gate_still_wins_when_both_apply(self):
        """A candidate both far away and in the wrong colour is refused for
        being far away, because that gate is checked first and its count is
        what the diagnostics have always meant."""
        m = manager(region="gear", a="red", b="blue")
        looked_at(m, 1, enough(BLUE), x=5000, y=5000)
        self.assertLessEqual(m._score(m.a, person(1, 5000, 5000)), -100)
        self.assertEqual(m.a.last_refusal, "too_far_to_be_them")

    def test_the_swap_case_is_actually_stopped(self):
        """The case this exists for, and the one a penalty could not reach.

        Tracking has slid onto the other fighter: the candidate sits exactly
        where the fighter was predicted, so position and IoU are perfect, and
        the rolling template has drifted to match. Every comparative signal
        agrees. Only the corner disagrees - and it disagrees consistently,
        because the track really is the other fighter.
        """
        m = manager(region="gear", a="red", b="blue")
        perfect_match = person(1, 100, 200)
        self.assertGreaterEqual(m._score(m.a, perfect_match), 0.46)
        looked_at(m, 1, enough(BLUE), x=100, y=200)
        self.assertLessEqual(m._score(m.a, person(1, 100, 200)), -100)


if __name__ == "__main__":
    unittest.main()
