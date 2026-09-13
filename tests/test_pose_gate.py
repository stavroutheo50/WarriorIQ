"""The joint gate must refuse the impossible and allow the merely fast."""

from __future__ import annotations

import unittest

import numpy as np

from core.pose_smoothing import JointGate
from core.types import PersonObservation


def _person(points: np.ndarray) -> PersonObservation:
    # A 100px-tall fighter, so one body-height per second is 100 px/s.
    return PersonObservation(track_id=1, box=np.array([0.0, 0.0, 50.0, 100.0]),
                             confidence=0.9, keypoints=points.copy())


def _standing() -> np.ndarray:
    return np.tile(np.array([25.0, 50.0], dtype=np.float32), (17, 1))


class JointGateTests(unittest.TestCase):
    def test_a_head_cannot_cross_the_frame_in_a_tenth_of_a_second(self):
        """The failure this exists for: a confident skeleton that collapsed.

        The nose moves 100 px in 0.1 s on a 100 px body - ten body-heights per
        second, about 17 m/s. Nobody's face does that.
        """
        gate = JointGate()
        gate.apply("A", 0.0, _person(_standing()))
        moved = _standing()
        moved[0] = [125.0, 50.0]
        observation = _person(moved)
        gate.apply("A", 0.1, observation)
        self.assertEqual(gate.gated_joints, 1)
        np.testing.assert_allclose(observation.keypoints[0, :2], [25.0, 50.0])

    def test_a_kicking_ankle_is_left_alone(self):
        """The thing that must not break. An ankle genuinely reaches 15 m/s.

        The same displacement that is impossible for a nose is an ordinary kick
        at the foot, so a single limit for every joint would either clip kicks
        or admit collapsed skeletons. It cannot do both.
        """
        gate = JointGate()
        gate.apply("A", 0.0, _person(_standing()))
        kicked = _standing()
        kicked[16] = [125.0, 50.0]          # right ankle, ten body-heights/s
        observation = _person(kicked)
        gate.apply("A", 0.1, observation)
        self.assertEqual(gate.gated_joints, 0, "a real kick was refused")
        np.testing.assert_allclose(observation.keypoints[16, :2], [125.0, 50.0])

    def test_a_fighter_who_really_moved_is_not_pinned_forever(self):
        """Holding has to expire, or a gate failure becomes a frozen fighter."""
        gate = JointGate()
        gate.apply("A", 0.0, _person(_standing()))
        moved = _standing()
        moved[0] = [125.0, 50.0]
        last = None
        for step in range(1, 7):
            last = _person(moved)
            gate.apply("A", step * 0.1, last)
        self.assertAlmostEqual(float(last.keypoints[0, 0]), 125.0, places=3,
                               msg="the model insisted and was never believed")

    def test_losing_the_fighter_clears_the_history(self):
        """A stale pose must not be compared against a re-acquisition."""
        gate = JointGate()
        gate.apply("A", 0.0, _person(_standing()))
        gate.apply("A", 0.1, None)
        moved = _standing()
        moved[0] = [125.0, 50.0]
        observation = _person(moved)
        gate.apply("A", 0.2, observation)
        self.assertEqual(gate.gated_joints, 0)


if __name__ == "__main__":
    unittest.main()
