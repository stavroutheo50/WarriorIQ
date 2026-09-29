"""Plain fight numbers (core/fight_numbers.py): seconds, distances and strike timing."""

import unittest

import numpy as np

from core.fight_numbers import movement_numbers, output_numbers


def sample(t, x, y=100.0, body=100.0, guard=0.8, balance=0.8, toward=None, speed=None, gap=None):
    return {"t": t, "x": x, "y": y, "body": body, "guard": guard, "balance": balance,
            "toward": toward, "speed": speed, "gap": gap}


class MovementNumbersTests(unittest.TestCase):
    MIDDLE, RADIUS = np.asarray([500.0, 100.0]), 200.0

    def test_seconds_in_the_middle(self):
        # Ten seconds seen at 5 samples a second: 6 in the middle, 4 out at the edge.
        samples = [sample(i / 5, 500.0 if i < 30 else 680.0) for i in range(50)]
        numbers = movement_numbers(samples, self.MIDDLE, self.RADIUS)
        self.assertAlmostEqual(numbers["seen_seconds"], 10.0, places=1)
        self.assertAlmostEqual(numbers["centre_seconds"], 6.0, places=1)

    def test_forward_backward_and_circling_only_while_moving(self):
        samples = ([sample(i / 5, 500, toward=0.9, speed=1.0) for i in range(10)]
                   + [sample(2 + i / 5, 500, toward=-0.9, speed=1.0) for i in range(5)]
                   + [sample(3 + i / 5, 500, toward=0.0, speed=1.0) for i in range(5)]
                   + [sample(4 + i / 5, 500, toward=0.9, speed=0.1) for i in range(5)])
        numbers = movement_numbers(samples, self.MIDDLE, self.RADIUS)
        self.assertAlmostEqual(numbers["forward_seconds"], 2.0, places=1)
        self.assertAlmostEqual(numbers["backward_seconds"], 1.0, places=1)
        self.assertAlmostEqual(numbers["circling_seconds"], 1.0, places=1)
        self.assertAlmostEqual(numbers["standing_seconds"], 1.0, places=1)

    def test_distance_ranges(self):
        samples = ([sample(i / 5, 500, gap=0.5) for i in range(5)]
                   + [sample(1 + i / 5, 500, gap=1.2) for i in range(10)]
                   + [sample(3 + i / 5, 500, gap=2.5) for i in range(5)])
        ranges = movement_numbers(samples, self.MIDDLE, self.RADIUS)["range_seconds"]
        self.assertEqual((ranges["close"], ranges["middle"], ranges["long"]), (1.0, 2.0, 1.0))

    def test_distance_covered_ignores_jumps_across_unseen_gaps(self):
        samples = [sample(0.0, 500), sample(0.2, 600), sample(5.0, 900), sample(5.2, 1000)]
        self.assertEqual(movement_numbers(samples, None, None)["distance_body_lengths"], 2.0)

    def test_hands_down_and_off_balance(self):
        samples = ([sample(i / 5, 500, guard=0.9) for i in range(10)]
                   + [sample(2 + i / 5, 500, guard=0.1, balance=0.2) for i in range(8)]
                   + [sample(3.6 + i / 5, 500, guard=0.9) for i in range(10)])
        numbers = movement_numbers(samples, self.MIDDLE, self.RADIUS)
        self.assertAlmostEqual(numbers["longest_hands_down_seconds"], 1.4, places=1)
        self.assertEqual(numbers["off_balance_count"], 1)
        self.assertAlmostEqual(numbers["hands_up_share"], 20 / 28, places=3)

    def test_slowing_down_in_the_second_half(self):
        samples = ([sample(i / 5, 500, speed=1.0) for i in range(20)]
                   + [sample(4 + i / 5, 500, speed=0.5) for i in range(20)])
        self.assertEqual(movement_numbers(samples, self.MIDDLE, self.RADIUS)["pace_change_percent"], -50)

    def test_by_round(self):
        samples = [sample(i / 5, 500.0) for i in range(50)]
        numbers = movement_numbers(samples, self.MIDDLE, self.RADIUS, round_of=lambda t: 1 if t < 5 else 2)
        self.assertEqual(set(numbers["rounds"]), {"1", "2"})
        self.assertAlmostEqual(numbers["rounds"]["1"]["centre_seconds"], 5.0, places=1)

    def test_nothing_to_say_without_samples(self):
        self.assertIsNone(movement_numbers([sample(0, 1)], None, None))


class OutputNumbersTests(unittest.TestCase):
    def test_when_strikes_were_thrown(self):
        mine = [3.0, 4.0, 5.0, 6.0, 40.0, 95.0]
        output = output_numbers(mine, [10.0], [(1, 0.0, 60.0), (2, 60.0, 120.0)])
        self.assertEqual(output["strikes_per_minute"], 3.0)
        self.assertEqual((output["busiest_10s"], output["busiest_10s_at"]), (4, 3.0))
        self.assertEqual(output["longest_pause_seconds"], 35.0)
        self.assertEqual(output["first_strike_seconds"], 3.0)
        self.assertTrue(output["threw_first"])
        self.assertEqual(output["last_30s_by_round"], {"1": 1, "2": 1})

    def test_no_strikes(self):
        output = output_numbers([], [5.0], [(1, 0.0, 60.0)])
        self.assertEqual((output["busiest_10s"], output["first_strike_seconds"]), (0, None))
        self.assertEqual(output["longest_pause_seconds"], 60.0)


if __name__ == "__main__":
    unittest.main()
