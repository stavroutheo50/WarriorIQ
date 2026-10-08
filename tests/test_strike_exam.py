"""Per-sport strike counts that go live only on a passed accuracy exam (core/strike_exam.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import report as report_module
from core import strike_exam
from core.sport_policy import counting_policy
from core.strike_exam import BAR, CountCheck, WindowCheck, decide, family_of, passed_families, window_checks

MODEL = "a" * 64


def _passing_windows():
    return {"punch": WindowCheck("punch", 120, 100, 90, ("tkd_kick3",)),
            "kick": WindowCheck("kick", 80, 60, 40, ("tkd_kick3",)),          # 67%: fails
            "knee": WindowCheck("knee", 10, 10, 10, ())}                       # too few: fails


def _passing_counts(sport="mma"):
    rounds = tuple((f"bout{i % 4}", 20 + (i % 3), 20) for i in range(12))
    return {sport: CountCheck(sport, rounds, "ufc_stats")}


def _verdict(sport="mma"):
    return decide(_passing_windows(), _passing_counts(sport), [sport], checkpoint_sha256=MODEL)


def _report(sport="mma", model=MODEL):
    return {"scorecard": {"sport": sport}, "classifier": {"temporal_checkpoint_sha256": model}}


class MeasurementTests(unittest.TestCase):
    def test_family_of(self):
        self.assertEqual([family_of(n) for n in ("none", "jab", "spinning_backfist", "l_round_kick",
                                                  "r_push_kick", "l_knee", None)],
                         [None, "punch", "punch", "kick", "kick", "knee", None])

    def test_window_precision_and_recall(self):
        pairs = [("jab", "jab")] * 8 + [("none", "cross")] * 2 + [("l_hook", "none")] * 2 + [("l_round_kick", "jab")]
        checks = window_checks(pairs)
        punch = checks["punch"]
        self.assertEqual((punch.true_strikes, punch.predicted, punch.correct), (10, 11, 8))
        self.assertAlmostEqual(punch.precision, 8 / 11)
        self.assertAlmostEqual(punch.recall, 0.8)

    def test_too_few_strikes_never_passes(self):
        self.assertFalse(WindowCheck("knee", BAR["min_true_strikes"] - 1, 10, 10, ()).passed())
        self.assertTrue(WindowCheck("knee", BAR["min_true_strikes"], 10, 10, ()).passed())

    def test_count_check_needs_rounds_bouts_and_error(self):
        good = _passing_counts()["mma"]
        self.assertTrue(good.passed())
        self.assertLessEqual(good.median_error, BAR["max_round_error"])
        two_bouts = CountCheck("mma", tuple((f"b{i % 2}", 20, 20) for i in range(12)), "ufc_stats")
        self.assertFalse(two_bouts.passed())
        too_few = CountCheck("mma", tuple((f"b{i}", 20, 20) for i in range(5)), "ufc_stats")
        self.assertFalse(too_few.passed())
        off = CountCheck("mma", tuple((f"b{i % 4}", 30, 20) for i in range(12)), "ufc_stats")
        self.assertFalse(off.passed())

    def test_a_family_needs_both_checks(self):
        verdict = _verdict()
        self.assertEqual(verdict["sports"]["mma"]["passed_families"], ["punch"])
        no_counts = decide(_passing_windows(), {}, ["mma"], checkpoint_sha256=MODEL)
        self.assertEqual(no_counts["sports"]["mma"]["passed_families"], [])


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "strike_exam_verdict.json"
        for target, value in ((strike_exam, ("VERDICT_PATH", self.path)),
                              (report_module, ("STRIKE_COUNTS_PUBLISHED", False)),
                              (report_module, ("STRIKE_COUNTS_PRECISION_VALIDATED", False))):
            patcher = patch.object(target, *value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write(self, verdict):
        self.path.write_text(json.dumps(verdict), encoding="utf-8")

    def test_no_verdict_changes_nothing(self):
        self.assertEqual(report_module.published_families("mma", _report()), ())
        policy = counting_policy("mma", report=_report())
        self.assertEqual(policy.counted, ())

    def test_passed_family_is_published_for_the_examined_model_only(self):
        self._write(_verdict())
        self.assertEqual(report_module.published_families("mma", _report()), ("punch",))
        self.assertEqual(report_module.published_families("mma", _report(model="b" * 64)), ())
        self.assertEqual(report_module.published_families("mma", {"scorecard": {"sport": "mma"}}), ())
        self.assertEqual(report_module.published_families("mma", None), ())
        # Another sport was not examined.
        self.assertEqual(report_module.published_families("boxing", _report("boxing")), ())

    def test_policy_counts_what_passed_and_says_how_it_was_measured(self):
        self._write(_verdict())
        policy = counting_policy("mma", report=_report())
        self.assertEqual(policy.counted, ("punches",))
        self.assertIn("kicks", policy.withheld)
        self.assertIn("passed WarriorIQ's accuracy check", policy.estimate_note)
        self.assertIn("official count", policy.estimate_note)
        # Without the report there is no model to check, so nothing.
        self.assertEqual(counting_policy("mma").counted, ())

    def test_implausible_counts_still_withheld(self):
        self._write(_verdict())
        with patch.object(report_module, "counts_implausible", return_value=True):
            self.assertEqual(report_module.published_families("mma", _report()), ())

    def test_unreadable_or_foreign_verdict_is_ignored(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(passed_families("mma", _report()), ())
        self._write({"schema": "something.else", "checkpoint_sha256": MODEL,
                     "sports": {"mma": {"passed_families": ["punch"]}}})
        self.assertEqual(passed_families("mma", _report()), ())

    def test_global_switch_still_wins(self):
        with patch.object(report_module, "STRIKE_COUNTS_PUBLISHED", True):
            self.assertEqual(report_module.published_families("mma", _report(model=None)),
                             report_module.published_families("mma", None))
            self.assertTrue(report_module.published_families("mma", None))


class FingerprintTests(unittest.TestCase):
    def test_checkpoint_fingerprint_is_the_file_hash(self):
        import hashlib

        from core.temporal_model import checkpoint_sha256

        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "m.pt"
            path.write_bytes(b"weights")
            self.assertEqual(checkpoint_sha256(path), hashlib.sha256(b"weights").hexdigest())

    def test_worker_records_it_and_clears_it_when_the_model_stops(self):
        source = (Path(__file__).resolve().parents[1] / "core" / "analyzer.py").read_text(encoding="utf-8")
        self.assertIn('"temporal_checkpoint_sha256": getattr(action_engine.temporal, "checkpoint_sha256", None)',
                      source)
        self.assertIn('classifier["temporal_checkpoint_sha256"] = None', source)


if __name__ == "__main__":
    unittest.main()
