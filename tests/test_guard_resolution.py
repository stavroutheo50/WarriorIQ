"""Guard is measured in the fighter's own proportions, or not at all.

QA, 2026-10-07: the same clip gave Guard 27% / 45% at 1280x720 and 18% / 18%
at 160x90, and the low-resolution report carried no warning. The reading was
wrist-to-nose over a torso-based body length, with no confidence gate, no
smoothing and no floor on how small a body may be before a wrist cannot be
placed (core/guard.py).
"""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path

import numpy as np

from core import guard
from core.metrics import MetricsAccumulator
from core.types import PersonObservation

TRACK = Path(__file__).resolve().parents[1] / "dataset" / "regression" / "kicklight_stavrou_ceschia" / "track.jsonl.gz"


def _pose(scale: float = 1.0, wrist_conf: float = 0.9, shoulder_gap: float = 40.0):
    points = np.zeros((17, 2), dtype=np.float32)
    points[0] = (100, 80)
    points[5], points[6] = (100 - shoulder_gap / 2, 100), (100 + shoulder_gap / 2, 100)
    points[11], points[12] = (85, 200), (115, 200)
    points[9], points[10] = (100, 120), (140, 160)
    conf = np.full(17, 0.9, dtype=np.float32)
    conf[1:5] = 0.0
    conf[9] = conf[10] = wrist_conf
    return points * scale, conf


class ReadingTests(unittest.TestCase):
    def test_reading_is_in_the_fighters_own_proportions(self):
        big, conf = _pose(1.0)
        small, _ = _pose(0.25)
        self.assertAlmostEqual(guard.reading(big, conf)[0], guard.reading(small, conf)[0], places=4)

    def test_side_on_shoulders_do_not_inflate_it(self):
        square, conf = _pose(shoulder_gap=80.0)
        side_on, _ = _pose(shoulder_gap=6.0)
        # Square-on shoulders 80 px apart; side-on the torso floor (0.8 x 100)
        # gives the same 80 px scale instead of a 6 px one.
        self.assertAlmostEqual(guard.reading(square, conf)[0], guard.reading(side_on, conf)[0], places=4)

    def test_unclear_wrists_are_not_measured(self):
        points, conf = _pose(wrist_conf=0.2)
        self.assertEqual(guard.reading(points, conf), (None, guard.UNCLEAR))
        self.assertEqual(guard.reading(points, None), (None, guard.UNCLEAR))

    def test_too_small_a_body_is_not_measured(self):
        points, conf = _pose(scale=0.09)   # 7.2 px scale
        self.assertEqual(guard.reading(points, conf), (None, guard.TOO_SMALL))

    def test_one_misplaced_wrist_is_not_a_dropped_guard(self):
        readings = [(i / 6.0, 0.3) for i in range(60)]
        readings[30] = (30 / 6.0, 2.5)
        summary = guard.summarise(readings)
        self.assertEqual(summary["share"], 1.0)
        self.assertEqual(summary["drops"], [])


def _track_metrics(factor: float) -> dict:
    """The real kick-light track, shrunk by ``factor`` as a smaller copy would be:
    keypoints rounded to whole pixels with 0.6 px of pose jitter."""
    rng = np.random.default_rng(0)
    rows = [json.loads(line) for line in gzip.open(TRACK, "rt", encoding="utf-8")]

    def obs(entry):
        if not entry:
            return None
        points = np.round(np.asarray(entry["keypoints"], dtype=np.float32) * factor
                          + rng.normal(0.0, 0.6, (17, 2))).astype(np.float32)
        return PersonObservation(track_id=1, box=np.asarray(entry["box"], dtype=np.float32) * factor,
                                 confidence=0.9, keypoints=points,
                                 keypoint_conf=np.asarray(entry["keypoint_conf"], dtype=np.float32))

    metrics = MetricsAccumulator(int(568 * factor), int(320 * factor))
    for row in rows:
        a, b = obs(row["fighter_A"]["observation"]), obs(row["fighter_B"]["observation"])
        metrics.update("A", row["time_seconds"], 1, a, b)
        metrics.update("B", row["time_seconds"], 1, b, a)
    return metrics.finalize([], [], rows[-1]["time_seconds"])


class ResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = _track_metrics(1.0)
        cls.half = _track_metrics(0.5)
        cls.tiny = _track_metrics(0.125)

    def test_guard_holds_when_the_video_is_halved(self):
        for fighter in ("A", "B"):
            full, half = self.full[fighter]["guard_index"], self.half[fighter]["guard_index"]
            self.assertIsNotNone(full)
            self.assertIsNotNone(half)
            self.assertLessEqual(abs(full - half), 0.05, fighter)

    def test_below_the_floor_guard_is_not_measured_and_says_why(self):
        for fighter in ("A", "B"):
            own = self.tiny[fighter]
            self.assertIsNone(own["guard_index"], fighter)
            self.assertIsNone(own["numbers"]["hands_up_share"])
            self.assertIsNone(own["numbers"]["longest_hands_down_seconds"])
            self.assertEqual(own["guard_note"], guard.NOT_MEASURED_TEXT[guard.TOO_SMALL])
            self.assertEqual(own["availability"]["guard"]["reason"], guard.NOT_MEASURED_TEXT[guard.TOO_SMALL])
            # The movement measured from the same frames is still reported.
            self.assertIsNotNone(own["footwork_body_lengths_per_second"])

    def test_a_withheld_guard_is_withheld_everywhere(self):
        """At a 0.35 scale a few frames still pass the gates: too few to
        measure, so the card says "Not measured" - and the Defence table must
        not print a longest stretch with hands down from those few."""
        partial = _track_metrics(0.35)
        for fighter in ("A", "B"):
            own = partial[fighter]
            self.assertIsNone(own["guard_index"])
            self.assertIsNone(own["numbers"]["hands_up_share"])
            self.assertIsNone(own["numbers"]["longest_hands_down_seconds"])
            self.assertEqual((own["moments"].get("guard_index") or {}).get("low") or [], [])

    def test_new_reports_are_not_reconciled_as_old_ones(self):
        report = {"metrics": {"A": dict(self.full["A"])}}
        before = report["metrics"]["A"]["guard_index"]
        guard.reconcile_report_guard(report)
        self.assertEqual(report["metrics"]["A"]["guard_index"], before)
        self.assertNotIn("guard_note", report["metrics"]["A"])


if __name__ == "__main__":
    unittest.main()
