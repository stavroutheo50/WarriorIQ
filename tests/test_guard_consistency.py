"""Every guard figure in a report comes from one per-frame classification.

QA, 2026-10-07, /result/9138ca9b38a7: the top card, the comparison bars,
"Work on" and the plan said "Guard 27%" (and 45% for the opponent), while the
Defence table on the same page said "Hands up by the face 0% / 0%" and
"Longest with hands down 20 s / 0 s". Those were three summaries of the same
reading against different lines (see core/guard.py).
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from core import guard
from core.coaching import build_pose_coaching, build_training_plan
from core.metrics import MetricsAccumulator
from core.report_visuals import head_to_head
from core.types import PersonObservation

FPS = 6.0


def _person(x: float, reading: float) -> PersonObservation:
    """A standing body whose nearer wrist sits where ``reading`` puts it.

    The reading is wrist-to-chin over shoulder width (core/guard.py). Shoulders
    40 px apart over a 100 px torso give a scale of max(40, 0.8 x 100) = 80 px;
    the nose at y=80 and the shoulder line at y=100 put the chin at y=87.
    Lower readings are hands nearer the face; up is <= guard.HANDS_UP.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    nose = (x, 80.0)
    kp[0, :2] = nose
    kp[5, :2], kp[6, :2] = (x - 20, 100), (x + 20, 100)
    kp[11, :2], kp[12, :2] = (x - 15, 200), (x + 15, 200)
    kp[15, :2], kp[16, :2] = (x - 25, 330), (x + 25, 330)
    chin_y = nose[1] + guard.CHIN_FRACTION * (100.0 - nose[1])
    kp[9, :2] = (nose[0], chin_y + reading * 80.0)
    kp[10, :2] = (nose[0] + 5, chin_y + reading * 80.0 + 30)
    kp[:, 2] = 0.9
    return PersonObservation(track_id=None, box=np.array([x - 40, 60, x + 40, 340], dtype=np.float32),
                             confidence=0.9, keypoints=kp[:, :2], keypoint_conf=kp[:, 2])


def _finalize(readings_a: list[float], readings_b: list[float]) -> dict:
    metrics = MetricsAccumulator(1280, 720)
    for index, (ra, rb) in enumerate(zip(readings_a, readings_b)):
        seconds = index / FPS
        a, b = _person(400 + 3 * (index % 10), ra), _person(700 - 3 * (index % 7), rb)
        metrics.update("A", seconds, 1, a, b)
        metrics.update("B", seconds, 1, b, a)
    return metrics.finalize([], [], len(readings_a) / FPS)


def _figures(own: dict) -> dict:
    numbers = own.get("numbers") or {}
    return {"card": own.get("guard_index"), "defence": numbers.get("hands_up_share"),
            "longest_down": numbers.get("longest_hands_down_seconds"),
            "drops": ((own.get("moments") or {}).get("guard_index") or {}).get("low") or []}


class QaReproTests(unittest.TestCase):
    """The report that said 27%/45% beside 0%/0% and 20 s/0 s."""

    def setUp(self):
        # 20 s at 6 fps, in wrist-to-chin shoulder widths. A: hands low, then
        # nearer but still short of the face. B: carried at chest height the
        # whole time - the hands the old summaries called 45% guard, 0% up and
        # 0 s down at once.
        n = 120
        self.a_readings = [1.6] * 90 + [0.95] * 30
        self.b_readings = [0.95] * n
        self.result = _finalize(self.a_readings, self.b_readings)

    def test_card_and_defence_table_are_the_same_number(self):
        for fighter in ("A", "B"):
            figures = _figures(self.result[fighter])
            self.assertIsNotNone(figures["card"])
            self.assertEqual(figures["card"], figures["defence"], fighter)
            self.assertEqual(self.result[fighter]["guard_definition"], guard.GUARD_DEFINITION)

    def test_hands_never_up_reads_as_zero_with_a_hands_down_stretch(self):
        for fighter in ("A", "B"):
            figures = _figures(self.result[fighter])
            self.assertEqual(figures["card"], 0.0, fighter)
            # Nearly the whole 20 s was down, and it says so for both.
            self.assertGreater(figures["longest_down"], 19.0, fighter)

    def test_guard_drop_marker_is_where_the_hands_came_down(self):
        drops = _figures(self.result["A"])["drops"]
        self.assertEqual(drops, [0.0])

    def test_bars_coaching_and_plan_print_the_card_figure(self):
        bars = {row["key"]: row for row in head_to_head(self.result, "A", "B", {})}
        self.assertEqual(bars["guard"]["a"], 0)
        self.assertEqual(bars["guard"]["b"], 0)
        coaching = build_pose_coaching("A", self.result["A"], self.result["B"], "kickboxing")
        text = " ".join(str(item) for item in coaching["strengths"] + coaching["improvements"] + coaching["drills"])
        self.assertNotIn("27%", text)
        self.assertNotIn("45%", text)
        plan = build_training_plan(coaching, "A", self.result["A"])
        self.assertNotIn("45%", str(plan))


class InvariantTests(unittest.TestCase):
    """Guard-up share and longest hands-down must agree for the same fighter."""

    def test_random_fights_never_contradict_themselves(self):
        rng = random.Random(20261007)
        for _case in range(60):
            n = rng.randint(80, 240)
            level = rng.random() * 1.6
            readings = [max(0.0, level + rng.gauss(0, 0.35)) for _ in range(n)]
            # Some frames lose the wrists, as real footage does.
            readings = [None if rng.random() < 0.1 else r for r in readings]
            times = [i / FPS for i in range(n)]
            summary = guard.summarise(zip(times, readings))
            self.assertEqual(guard.consistency_errors(summary["share"], summary["longest_down_seconds"],
                                                      summary["measured_seconds"]), [])
            self.assertAlmostEqual(summary["up_seconds"] + summary["down_seconds"], summary["measured_seconds"],
                                   delta=0.11)
            self.assertLessEqual(summary["longest_down_seconds"], summary["down_seconds"] + 0.05)
            if summary["share"] >= 1.0:
                self.assertEqual(summary["drops"], [])
            # Classified after smoothing, so markers are checked against the
            # smoothed reading of their frame.
            kept = [(t, r) for t, r in zip(times, readings) if r is not None]
            smoothed = guard._smoothed([t for t, _ in kept], [r for _, r in kept])
            measured = {round(t, 2): value for (t, _), value in zip(kept, smoothed)}
            for moment in summary["drops"]:
                self.assertFalse(guard.is_up(measured[moment]), "a drop marker must sit on a hands-down frame")
            for moment in summary["held"]:
                self.assertTrue(guard.is_up(measured[moment]))

    def test_whole_pipeline_agrees(self):
        rng = random.Random(7)
        a = [max(0.0, 0.8 + rng.gauss(0, 0.3)) for _ in range(150)]
        b = [max(0.0, 0.6 + rng.gauss(0, 0.3)) for _ in range(150)]
        result = _finalize(a, b)
        for fighter in ("A", "B"):
            own = result[fighter]
            figures = _figures(own)
            self.assertEqual(figures["card"], figures["defence"])
            self.assertEqual(guard.consistency_errors(figures["card"], figures["longest_down"],
                                                      own["numbers"]["seen_seconds"]), [])

    def test_consistency_check_catches_the_qa_report(self):
        self.assertTrue(guard.consistency_errors(0.0, 0.0, 20.0))
        self.assertTrue(guard.consistency_errors(1.0, 5.0, 20.0))
        self.assertTrue(guard.consistency_errors(0.9, 10.0, 20.0))


class SoloReportTests(unittest.TestCase):
    def test_solo_report_keeps_its_guard_definition(self):
        """Without it a fresh solo report would be reconciled as an old one
        on load and lose its longest stretch with hands down."""
        from core.solo import SOLO_METRICS

        self.assertIn("guard_definition", SOLO_METRICS)


class LegacyReportTests(unittest.TestCase):
    """Reports saved before the fix are brought onto the one definition on load."""

    def _legacy(self):
        return {"metrics": {
            "A": {"guard_index": 0.27, "numbers": {"hands_up_share": 0.0, "longest_hands_down_seconds": 20.0},
                  "moments": {"guard_index": {"low": [3.0, 9.0]}, "balance_index": {"low": [1.0]}},
                  "spread": {"guard_index": {"standard_error": 0.01}},
                  "availability": {"guard": {"available": True}}},
            "B": {"guard_index": 0.45, "numbers": {"hands_up_share": 0.0, "longest_hands_down_seconds": 0.0}},
        }}

    def test_card_takes_the_defence_table_share_and_the_rest_is_withheld(self):
        report = guard.reconcile_report_guard(self._legacy())
        a, b = report["metrics"]["A"], report["metrics"]["B"]
        self.assertEqual((a["guard_index"], b["guard_index"]), (0.0, 0.0))
        self.assertIsNone(a["numbers"]["longest_hands_down_seconds"])
        self.assertIsNone(b["numbers"]["longest_hands_down_seconds"])
        self.assertNotIn("guard_index", a["moments"])
        self.assertIn("balance_index", a["moments"])
        self.assertNotIn("guard_index", a["spread"])
        self.assertIn("Re-run", a["guard_note"])

    def test_without_a_share_guard_is_not_measured_and_says_why(self):
        report = guard.reconcile_report_guard({"metrics": {"A": {"guard_index": 0.12}}})
        own = report["metrics"]["A"]
        self.assertIsNone(own["guard_index"])
        self.assertEqual(own["guard_note"], guard.LEGACY_NOTE)

    def test_idempotent_and_current_reports_untouched(self):
        once = guard.reconcile_report_guard(self._legacy())
        twice = guard.reconcile_report_guard(once)
        self.assertEqual(once, twice)
        current = {"metrics": {"A": {"guard_index": 0.3, "guard_definition": guard.GUARD_DEFINITION,
                                     "numbers": {"hands_up_share": 0.3, "longest_hands_down_seconds": 4.0}}}}
        self.assertEqual(guard.reconcile_report_guard(current)["metrics"]["A"]["guard_index"], 0.3)
        self.assertEqual(current["metrics"]["A"]["numbers"]["longest_hands_down_seconds"], 4.0)

    def test_report_load_gate_reconciles_before_coaching(self):
        from core.report import refresh_identity_integrity

        report = self._legacy()
        refresh_identity_integrity(report)
        self.assertEqual(report["metrics"]["A"]["guard_index"], 0.0)
        self.assertNotIn("27%", str(report.get("coaching")))


if __name__ == "__main__":
    unittest.main()
