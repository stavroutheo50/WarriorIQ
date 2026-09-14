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


class MovementEvidenceTests(unittest.TestCase):
    """A movement claim has to point at the video, or a coach cannot check it."""

    def _engine(self):
        from core.metrics import MetricsAccumulator
        return MetricsAccumulator(480, 220)

    def test_extremes_spread_out_instead_of_citing_one_exchange(self):
        """The four lowest readings are usually four frames of the same moment.

        Showing a coach the same second four times is showing them one piece of
        evidence and claiming four.
        """
        engine = self._engine()
        samples = [(10.0, 0.10), (10.1, 0.11), (10.2, 0.12), (10.3, 0.13),
                   (40.0, 0.20), (70.0, 0.25), (95.0, 0.30)]
        picked = engine._extremes(samples, want_low=True, limit=4)
        self.assertEqual(len(picked), 4)
        self.assertEqual(picked, sorted(picked), "times are given in order")
        for earlier, later in zip(picked, picked[1:]):
            self.assertGreaterEqual(later - earlier, 3.0, "two citations from the same exchange")

    def test_low_and_high_pick_opposite_ends(self):
        engine = self._engine()
        samples = [(5.0, 0.9), (20.0, 0.1), (35.0, 0.8), (50.0, 0.2)]
        self.assertEqual(engine._extremes(samples, want_low=True, limit=1), [20.0])
        self.assertEqual(engine._extremes(samples, want_low=False, limit=1), [5.0])

    def test_no_samples_cites_nothing_rather_than_guessing(self):
        self.assertEqual(self._engine()._extremes([], want_low=True), [])

    def test_a_movement_claim_carries_its_moments(self):
        """The regression this exists for: evidence_times was hardcoded empty.

        On pose-only footage - which is most real footage - movement claims are
        the only ones shown, so every claim a coach saw was unverifiable.
        """
        from core.coaching import build_pose_coaching
        own = {
            "guard_index": 0.39, "balance_index": 0.6, "pressure_index": 0.12,
            "footwork_body_lengths_per_second": 0.8, "pose_coverage": 0.9,
            "moments": {"guard_index": {"low": [3.0, 9.0], "high": [8.42, 25.44]},
                        "pressure_index": {"low": [50.18, 65.98], "high": [1.0]}},
        }
        other = {"guard_index": 0.14, "balance_index": 0.7, "pressure_index": 0.30,
                 "footwork_body_lengths_per_second": 0.9, "pose_coverage": 0.9}
        coaching = build_pose_coaching("A", own, other)
        cited = [t for group in ("strengths", "improvements")
                 for item in coaching[group] for t in item["evidence_times"]]
        self.assertTrue(cited, "a movement claim with no moments is what this fixed")
