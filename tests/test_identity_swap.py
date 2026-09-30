"""A and B put back the right way round, and which hand-offs count as suspicious.

Measured on handheld phone sparring (dataset/regression/identity_phone): when
the tracker hands a fighter's track to the opponent, every frame-to-frame check
follows the swap. Only the colours picked at selection still disagree with it.
"""

from __future__ import annotations

import unittest

import numpy as np

from core.config import SETTINGS
from core.identity import IdentityManager
from core.report import churn_rate, identity_churned
from core.types import PersonObservation


def _look(share_a: float) -> np.ndarray:
    """A clothing histogram that is `share_a` fighter A's colour, the rest B's."""
    look = np.zeros((64,), dtype=np.float32)
    look[1] = share_a
    look[2] = 1.0 - share_a
    look[10:20] = 0.05
    return look


def _person(track_id: int, x1: float, x2: float, share_a: float) -> PersonObservation:
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = np.linspace(x1 + 4, x2 - 4, 17)
    kp[:, 1] = np.linspace(100, 290, 17)
    return PersonObservation(
        track_id=track_id,
        box=np.asarray([x1, 80, x2, 310], dtype=np.float32),
        confidence=0.95,
        keypoints=kp,
        keypoint_conf=np.ones((17,), dtype=np.float32),
        appearance=_look(share_a),
    )


class SwapCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.manager = IdentityManager(_person(1, 100, 200, 1.0), _person(2, 400, 500, 0.0), 0)
        self.manager.a.current_track_id = 1
        self.manager.b.current_track_id = 2

    def _feed(self, a_obs, b_obs, frames):
        out = (a_obs, b_obs)
        for _ in range(frames):
            out = self.manager._correct_swap(a_obs, b_obs)
            if out[0] is not a_obs:
                break
        return out

    def test_each_looking_like_the_other_is_swapped_back(self):
        # A's box now wears B's colours and B's box A's.
        a_obs, b_obs = _person(2, 400, 500, 0.1), _person(1, 100, 200, 0.9)
        self.manager.a.current_track_id, self.manager.b.current_track_id = 2, 1
        a_out, b_out = self._feed(a_obs, b_obs, 30)
        self.assertIs(a_out, b_obs)
        self.assertIs(b_out, a_obs)
        self.assertEqual(self.manager.swaps_corrected, 1)
        # What is followed from here moves with them; who was picked does not.
        self.assertEqual(self.manager.a.current_track_id, 1)
        self.assertEqual(self.manager.b.current_track_id, 2)
        self.assertEqual(float(self.manager.a.anchor_appearance[1]), 1.0)

    def test_one_frame_is_not_enough(self):
        a_obs, b_obs = _person(2, 400, 500, 0.1), _person(1, 100, 200, 0.9)
        self.assertIs(self.manager._correct_swap(a_obs, b_obs)[0], a_obs)
        self.assertEqual(self.manager.swaps_corrected, 0)

    def test_the_right_way_round_is_never_swapped(self):
        a_obs, b_obs = _person(1, 100, 200, 0.9), _person(2, 400, 500, 0.1)
        self._feed(a_obs, b_obs, 200)
        self.assertEqual(self.manager.swaps_corrected, 0)

    def test_fighters_in_matching_kit_are_left_alone(self):
        """Colour cannot say who is who when both wear the same, so it must not try."""
        manager = IdentityManager(_person(1, 100, 200, 0.5), _person(2, 400, 500, 0.5), 0)
        a_obs, b_obs = _person(2, 400, 500, 0.1), _person(1, 100, 200, 0.9)
        for _ in range(200):
            manager._correct_swap(a_obs, b_obs)
        self.assertEqual(manager.swaps_corrected, 0)

    def test_a_missing_fighter_changes_nothing(self):
        b_obs = _person(1, 100, 200, 0.9)
        for _ in range(50):
            self.assertEqual(self.manager._correct_swap(None, b_obs), (None, b_obs))
        self.assertEqual(self.manager.swaps_corrected, 0)


class SuspiciousHandoffTests(unittest.TestCase):
    def setUp(self):
        self.manager = IdentityManager(_person(1, 100, 200, 1.0), _person(2, 400, 500, 0.0), 0)

    def test_a_new_track_on_the_same_person_is_not_suspicious(self):
        state = self.manager.a
        self.manager._commit(state, _person(1, 100, 200, 1.0), 1, 1.0)
        self.manager._commit(state, _person(7, 105, 205, 1.0), 2, 1.0, recovered=True)
        self.assertEqual(self.manager.track_handoffs["A"], 1)
        self.assertEqual(self.manager.suspicious_handoffs["A"], 0)

    def test_a_jump_across_the_picture_is_suspicious(self):
        state = self.manager.a
        self.manager._commit(state, _person(1, 100, 200, 1.0), 1, 1.0)
        self.manager._commit(state, _person(7, 600, 700, 1.0), 2, 1.0, recovered=True)
        self.assertEqual(self.manager.suspicious_handoffs["A"], 1)

    def test_a_change_of_colour_is_suspicious(self):
        state = self.manager.a
        self.manager._commit(state, _person(1, 100, 200, 1.0), 1, 1.0)
        self.manager._commit(state, _person(7, 105, 205, 0.0), 2, 1.0, recovered=True)
        self.assertEqual(self.manager.suspicious_handoffs["A"], 1)


class ChurnGateTests(unittest.TestCase):
    def test_suspicious_count_decides_when_present(self):
        # The phone clip followed correctly: many hand-offs, few suspicious.
        tracking = {"fighter_A_handoffs_per_minute": 19.5,
                    "fighter_A_suspicious_handoffs_per_minute": 7.8,
                    "fighter_B_handoffs_per_minute": 7.8,
                    "fighter_B_suspicious_handoffs_per_minute": 3.9}
        self.assertEqual(identity_churned(tracking), {"A": False, "B": False})
        self.assertEqual(churn_rate(tracking, "A"), (7.8, False))

    def test_a_real_mix_up_is_still_held_back(self):
        # Athens, handheld from the stands: mostly the wrong person.
        tracking = {"fighter_A_handoffs_per_minute": 21.4,
                    "fighter_A_suspicious_handoffs_per_minute": 15.0,
                    "fighter_B_handoffs_per_minute": 25.7,
                    "fighter_B_suspicious_handoffs_per_minute": 21.7}
        self.assertEqual(identity_churned(tracking), {"A": True, "B": True})

    def test_older_reports_are_judged_as_before(self):
        tracking = {"fighter_A_handoffs_per_minute": 19.5, "fighter_B_handoffs_per_minute": 7.8}
        self.assertEqual(identity_churned(tracking), {"A": True, "B": False})
        self.assertGreater(19.5, SETTINGS.max_identity_handoffs_per_minute)


if __name__ == "__main__":
    unittest.main()
