"""Checking each frame's choice against the fighters the user picked.

core.identity.IdentityManager._audit_against_selection: a fighter whose box
has slid onto somebody else is moved back onto the person who looks like
their selection - clearly, clearly more than the person they are on, and
clearly more than the other fighter's selection.
"""

import unittest
from unittest import mock

import numpy as np

from core import identity
from core.identity import IdentityManager, appearance_similarity
from core.types import PersonObservation


def colours(*parts: tuple[int, float]) -> np.ndarray:
    """A clothing histogram made of (block, weight) parts, 8 bins a block."""
    hist = np.full(64, 0.01, dtype=np.float32)
    for block, weight in parts:
        hist[block * 8:(block + 1) * 8] += weight
    return hist


RED_SHIRT, BLACK_SHIRT, GREY_HOODIE = 0, 1, 2


def person(track_id, x, look, referee_prob=0.01):
    return PersonObservation(
        track_id=track_id, box=np.asarray([x, 40.0, x + 40.0, 160.0], dtype=np.float32),
        confidence=0.9, appearance=look, referee_prob=referee_prob)


class SelectionAuditTests(unittest.TestCase):
    def manager(self, a_look=None, b_look=None):
        a = person(1, 100.0, colours((RED_SHIRT, 1.0)) if a_look is None else a_look)
        b = person(2, 400.0, colours((BLACK_SHIRT, 1.0)) if b_look is None else b_look)
        return IdentityManager(a, b, 0, source_fps=30.0)

    def test_a_fighter_on_a_stranger_goes_back_to_the_person_they_picked(self):
        manager = self.manager()
        # Near enough in colour to have been followed, not the fighter.
        stranger = person(3, 110.0, colours((RED_SHIRT, 0.5), (GREY_HOODIE, 0.5)))
        fighter_a = person(4, 250.0, colours((RED_SHIRT, 1.0)))
        fighter_b = person(2, 400.0, colours((BLACK_SHIRT, 1.0)))
        self.assertGreater(appearance_similarity(manager.a.anchor_appearance, stranger.appearance), 0.6)
        a, b = manager._audit_against_selection([stranger, fighter_a, fighter_b], stranger, fighter_b, 10)
        self.assertIs(a, fighter_a)
        self.assertIs(b, fighter_b)
        self.assertEqual(manager.a.current_track_id, 4)
        self.assertEqual(manager.selection_refinds, {"A": 1, "B": 0})

    def test_a_lost_fighter_is_found_again(self):
        manager = self.manager()
        fighter_a = person(4, 250.0, colours((RED_SHIRT, 1.0)))
        a, _ = manager._audit_against_selection([fighter_a], None, None, 10)
        self.assertIs(a, fighter_a)

    def test_nothing_moves_when_the_fighter_is_already_the_best_match(self):
        manager = self.manager()
        fighter_a = person(1, 105.0, colours((RED_SHIRT, 1.0)))
        similar = person(5, 300.0, colours((RED_SHIRT, 0.95), (GREY_HOODIE, 0.05)))
        a, _ = manager._audit_against_selection([fighter_a, similar], fighter_a, None, 10)
        self.assertIs(a, fighter_a)
        self.assertEqual(manager.selection_refinds, {"A": 0, "B": 0})

    def test_two_fighters_dressed_alike_are_never_traded_on_colour(self):
        both = colours((BLACK_SHIRT, 1.0))
        manager = self.manager(a_look=both, b_look=both)
        stranger = person(3, 110.0, colours((GREY_HOODIE, 1.0)))
        lookalike = person(4, 250.0, colours((BLACK_SHIRT, 1.0)))
        a, _ = manager._audit_against_selection([stranger, lookalike], stranger, None, 10)
        self.assertIs(a, stranger, "the colour cannot say which of the two it is")

    def test_the_referee_is_never_a_candidate(self):
        manager = self.manager()
        official = person(6, 250.0, colours((RED_SHIRT, 1.0)), referee_prob=0.99)
        a, _ = manager._audit_against_selection([official], None, None, 10)
        self.assertIsNone(a)

    def test_nor_is_someone_standing_perfectly_still(self):
        manager = self.manager()
        seated = person(7, 250.0, colours((RED_SHIRT, 1.0)))
        manager.prime_track_history([(frame, [seated]) for frame in range(0, 300, 2)])
        a, _ = manager._audit_against_selection([seated], None, None, 300)
        self.assertIsNone(a)

    def test_taken_from_the_other_fighter_leaves_them_unassigned(self):
        manager = self.manager()
        fighter_a = person(4, 250.0, colours((RED_SHIRT, 1.0)))
        a, b = manager._audit_against_selection([fighter_a], None, fighter_a, 10)
        self.assertIs(a, fighter_a)
        self.assertIsNone(b, "B was on A; B is missing for this frame, not moved onto a guess")

    def test_it_can_be_switched_off(self):
        from dataclasses import replace

        manager = self.manager()
        stranger = person(3, 110.0, colours((RED_SHIRT, 0.5), (GREY_HOODIE, 0.5)))
        fighter_a = person(4, 250.0, colours((RED_SHIRT, 1.0)))
        with mock.patch.object(identity, "SETTINGS", replace(identity.SETTINGS, selection_audit=False)):
            manager.update([stranger, fighter_a], 10)
        self.assertEqual(manager.selection_refinds, {"A": 0, "B": 0})


if __name__ == "__main__":
    unittest.main()
