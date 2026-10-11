"""tools/exam_drill.py: a called pad drill is its own answer key."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import exam_drill as drill  # noqa: E402


def _tone(seconds: float, hz: float) -> np.ndarray:
    t = np.arange(int(drill.RATE * seconds)) / drill.RATE
    return 0.5 * np.sin(2 * np.pi * hz * t).astype(np.float32)


def _put(track: np.ndarray, at: float, sound: np.ndarray) -> None:
    i = int(at * drill.RATE)
    track[i:i + len(sound)] += sound


class ScheduleTests(unittest.TestCase):
    def test_plan_has_enough_strikes_and_quiet_for_the_bar(self):
        plan = drill.schedule()
        self.assertGreaterEqual(len(plan["strikes"]), drill.BAR_STRIKES)
        self.assertEqual({s["family"] for s in plan["strikes"]}, {"punch", "kick", "knee"})
        rest = sum(b["end"] - b["start"] for b in plan["blocks"] if b["kind"] == "rest")
        self.assertGreaterEqual(rest, 180)

    def test_blocks_follow_each_other_and_beeps_are_far_enough_apart(self):
        plan = drill.schedule()
        for before, after in zip(plan["blocks"], plan["blocks"][1:]):
            self.assertAlmostEqual(before["end"], after["start"], places=3)
        times = [s["time"] for s in plan["strikes"]]
        # Matching windows of neighbouring beeps never overlap.
        self.assertGreater(min(np.diff(times)), drill.BEFORE + drill.AFTER)
        self.assertGreater(plan["blocks"][0]["start"], max(plan["start_beeps"]))

    def test_the_caller_page_plays_exactly_this_plan(self):
        self.assertEqual(drill.page_schedule(), drill.schedule())


class SoundTests(unittest.TestCase):
    def _recording(self, lead: float, noise: float = 0.05):
        plan = drill.schedule()
        rng = np.random.default_rng(0)
        track = rng.normal(0, noise, int(drill.RATE * (lead + plan["finished"] + 2))).astype(np.float32)
        _put(track, 1.0, _tone(0.25, drill.START_HZ))          # one stray high beep is not a start
        for s in plan["start_beeps"]:
            _put(track, lead + s, _tone(0.25, drill.START_HZ))
        for s in plan["strikes"]:
            _put(track, lead + s["time"], _tone(0.12, drill.STRIKE_HZ))
        return track, plan

    def test_finds_the_three_start_beeps_and_hears_the_strike_beeps(self):
        track, plan = self._recording(lead=7.31)
        start = drill.find_start(track)
        self.assertAlmostEqual(start, 7.31, delta=drill.FRAME * 1.5)
        times = [start + s["time"] for s in plan["strikes"]]
        self.assertEqual(drill.beeps_heard(track, times), len(plan["strikes"]))

    def test_a_wrong_start_does_not_hear_the_beeps(self):
        track, plan = self._recording(lead=7.31)
        times = [7.31 + 1.25 + s["time"] for s in plan["strikes"]]
        self.assertLess(drill.beeps_heard(track, times), len(plan["strikes"]) // 2)

    def test_no_start_in_silence(self):
        self.assertIsNone(drill.find_start(np.zeros(drill.RATE * 5, np.float32)))


class ScoreTests(unittest.TestCase):
    START = 12.0

    def _perfect(self, plan, delay=0.4):
        return [{"fighter": "A", "peak_time": self.START + s["time"] + delay, "family": s["family"]}
                for s in plan["strikes"]]

    def test_every_call_answered_on_time(self):
        plan = drill.schedule()
        result = drill.score(self._perfect(plan), plan, self.START)
        self.assertEqual((result["found"], result["detected"], result["in_rest"]), (84, 84, 0))
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["family_right"], 84)
        self.assertTrue(all(b["called"] == b["found"] == b["detected"] for b in result["blocks"]))
        self.assertIn("meets", drill.verdict(result))

    def test_rest_detections_doubles_and_late_ones_count_against_precision(self):
        plan = drill.schedule()
        events = self._perfect(plan)
        rest = next(b for b in plan["blocks"] if b["kind"] == "rest")
        events.append({"fighter": "A", "peak_time": self.START + rest["start"] + 5, "family": "punch"})
        first = plan["strikes"][0]["time"]
        events.append({"fighter": "A", "peak_time": self.START + first + 0.9, "family": "punch"})  # a second one
        events[1]["peak_time"] += 1.5                                                            # too late
        result = drill.score(events, plan, self.START)
        self.assertEqual(result["found"], 83)
        self.assertEqual(result["detected"], 86)
        self.assertEqual(result["in_rest"], 1)
        self.assertAlmostEqual(result["precision"], 83 / 86)

    def test_wrong_family_holder_and_outside_the_drill(self):
        plan = drill.schedule()
        events = self._perfect(plan)
        events[0]["family"] = "kick"
        events += [{"fighter": "B", "peak_time": self.START + 50, "family": "punch"},
                   {"fighter": "A", "peak_time": self.START - 3, "family": "punch"},       # before the start
                   {"fighter": "A", "peak_time": self.START + plan["finished"] + 3, "family": "punch"}]
        result = drill.score(events, plan, self.START)
        self.assertEqual(result["family_right"], 83)
        self.assertEqual(result["holder_detections"], 1)
        self.assertEqual(result["detected"], 84)

    def test_nothing_detected_is_not_a_precision(self):
        plan = drill.schedule()
        result = drill.score([], plan, self.START)
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0.0)
        self.assertIn("nothing to count", drill.verdict(result))

    def test_low_precision_is_said_plainly(self):
        plan = drill.schedule()
        events = self._perfect(plan)
        rest = [b for b in plan["blocks"] if b["kind"] == "rest"]
        events += [{"fighter": "A", "peak_time": self.START + b["start"] + k, "family": "punch"}
                   for b in rest for k in range(1, 8)]
        self.assertIn("is below", drill.verdict(drill.score(events, plan, self.START)))


if __name__ == "__main__":
    unittest.main()
