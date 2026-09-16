"""A rating that refuses to move on evidence the product itself withholds.

WarriorIQ does not show punch counts, because the action model has not passed
release validation. A rating built on those counts would put a confident
number on exactly the labels the report declines to present. These tests hold
the rating to the same gate the scorecard already applies.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import core.db as database
from core.rating import (
    BASELINE_RATING, PROVISIONAL_SESSIONS, SessionSignals, describe,
    expected_score, extract_session, is_provisional, next_rating,
    performance_score,
)


def report(*, trusted: bool, accuracy=0.5, attempts=40, coverage=0.8,
           advanced=True, rounds=3, lapses=4) -> dict:
    """The parts of a real report this reads, and nothing else."""
    return {
        "integrity": {"automated_evidence_trusted": trusted},
        "rounds": [{} for _ in range(rounds)],
        "metrics": {
            "A": {
                "advanced_metrics_available": advanced,
                "pose_coverage": coverage,
                "attacks": {"attempts": attempts, "landed": int(attempts * accuracy),
                            "accuracy": accuracy},
                "vulnerability_techniques": {"left_hook": lapses},
            }
        },
    }


class GateTests(unittest.TestCase):
    def test_untrusted_strike_numbers_never_reach_the_rating(self):
        signals = extract_session(report(trusted=False), "A", "job-1")
        self.assertFalse(signals.evidence_trusted)
        self.assertNotIn("accuracy", signals.used)
        self.assertNotIn("output", signals.used)

    def test_movement_still_counts_when_strikes_do_not(self):
        """Pose and tracking do not rest on action labels.

        This is why /compare already shows movement while withholding the
        strike table, and the rating follows the same rule.
        """
        signals = extract_session(report(trusted=False), "A", "job-1")
        self.assertEqual(signals.used, ("movement",))
        self.assertIsNotNone(performance_score(signals))

    def test_a_trusted_report_uses_the_strike_numbers_too(self):
        signals = extract_session(report(trusted=True), "A", "job-1")
        self.assertIn("accuracy", signals.used)
        self.assertIn("movement", signals.used)

    def test_a_session_with_no_usable_signal_is_not_ratable(self):
        blank = extract_session(
            report(trusted=False, advanced=False, coverage=None), "A", "job-1")
        self.assertFalse(blank.ratable)
        self.assertIsNone(performance_score(blank))

    def test_extract_never_mutates_the_report(self):
        """It consumes the scorecard; it must not touch it."""
        source = report(trusted=True)
        import copy
        before = copy.deepcopy(source)
        extract_session(source, "A", "job-1")
        self.assertEqual(source, before)


class RatingMathTests(unittest.TestCase):
    def test_a_baseline_rating_expects_an_even_result(self):
        self.assertAlmostEqual(expected_score(BASELINE_RATING), 0.5, places=6)

    def test_beating_expectation_raises_and_missing_it_lowers(self):
        up, up_delta = next_rating(BASELINE_RATING, 0, 0.9)
        down, down_delta = next_rating(BASELINE_RATING, 0, 0.1)
        self.assertGreater(up, BASELINE_RATING)
        self.assertLess(down, BASELINE_RATING)
        self.assertGreater(up_delta, 0)
        self.assertLess(down_delta, 0)

    def test_a_settled_rating_moves_less_than_a_provisional_one(self):
        _, new = next_rating(BASELINE_RATING, 0, 0.9)
        _, old = next_rating(BASELINE_RATING, PROVISIONAL_SESSIONS + 10, 0.9)
        self.assertGreater(abs(new), abs(old))

    def test_a_rating_converges_rather_than_running_away(self):
        """Repeating the same performance must approach a level, not diverge."""
        rating, deltas = BASELINE_RATING, []
        for n in range(40):
            rating, delta = next_rating(rating, n, 0.75)
            deltas.append(abs(delta))
        self.assertLess(deltas[-1], deltas[0])
        self.assertLess(abs(deltas[-1]), 2.0)


class ProvisionalTests(unittest.TestCase):
    def test_a_number_is_not_shown_before_it_is_earned(self):
        self.assertEqual(describe(1520.0, 0), "Unrated")
        self.assertIn("Provisional", describe(1520.0, 1))
        self.assertIn("Provisional", describe(1520.0, PROVISIONAL_SESSIONS - 1))

    def test_the_number_appears_once_there_are_enough_sessions(self):
        self.assertEqual(describe(1520.4, PROVISIONAL_SESSIONS), "1520")
        self.assertFalse(is_provisional(PROVISIONAL_SESSIONS))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = database.DB_PATH
        self.addCleanup(lambda: setattr(database, "DB_PATH", previous))
        database.DB_PATH = Path(self.temp.name) / "rating.sqlite3"
        database.init_db()

    def test_an_unrated_athlete_starts_at_the_baseline_and_says_so(self):
        standing = database.athlete_rating(1)
        self.assertEqual(standing["rating"], BASELINE_RATING)
        self.assertTrue(standing["provisional"])
        self.assertEqual(standing["label"], "Unrated")

    def test_a_session_is_recorded_and_moves_the_rating(self):
        signals = extract_session(report(trusted=True, accuracy=0.9), "A", "job-1")
        result = database.record_athlete_session(1, signals)
        self.assertTrue(result["rated"])
        self.assertNotEqual(result["rating"], BASELINE_RATING)
        self.assertEqual(len(database.athlete_history(1)), 1)

    def test_an_ungated_session_is_kept_but_does_not_move_the_number(self):
        """The history is worth having even when the rating cannot move."""
        blank = extract_session(
            report(trusted=False, advanced=False, coverage=None), "A", "job-1")
        result = database.record_athlete_session(1, blank)
        self.assertFalse(result["rated"])
        self.assertEqual(result["rating"], BASELINE_RATING)
        self.assertEqual(result["total_sessions"], 1)
        self.assertEqual(result["rated_sessions"], 0)
        rows = database.athlete_history(1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rated"], 0)
        self.assertEqual(rows[0]["rated_on"], "")

    def test_refiling_the_same_fight_cannot_score_twice(self):
        """Otherwise a rating goes up by pressing the button again."""
        signals = extract_session(report(trusted=True, accuracy=0.9), "A", "job-1")
        first = database.record_athlete_session(1, signals)
        second = database.record_athlete_session(1, signals)
        self.assertTrue(first["rated"])
        self.assertFalse(second["rated"])
        self.assertEqual(second["rating"], first["rating"])
        self.assertEqual(len(database.athlete_history(1)), 1)

    def test_the_label_turns_into_a_number_after_enough_rated_sessions(self):
        for n in range(PROVISIONAL_SESSIONS):
            signals = extract_session(report(trusted=True, accuracy=0.7), "A", "job-%d" % n)
            database.record_athlete_session(1, signals)
        standing = database.athlete_rating(1)
        self.assertFalse(standing["provisional"])
        self.assertTrue(standing["label"].isdigit(), standing["label"])

    def test_ungated_sessions_do_not_count_towards_leaving_provisional(self):
        """Five unratable fights are not five sessions of evidence."""
        for n in range(PROVISIONAL_SESSIONS + 2):
            blank = extract_session(
                report(trusted=False, advanced=False, coverage=None), "A", "job-%d" % n)
            database.record_athlete_session(1, blank)
        standing = database.athlete_rating(1)
        self.assertTrue(standing["provisional"])
        self.assertEqual(standing["rated_sessions"], 0)
        self.assertEqual(standing["total_sessions"], PROVISIONAL_SESSIONS + 2)

    def test_two_athletes_are_rated_independently(self):
        good = extract_session(report(trusted=True, accuracy=0.9), "A", "job-1")
        poor = extract_session(report(trusted=True, accuracy=0.1), "A", "job-2")
        database.record_athlete_session(1, good)
        database.record_athlete_session(2, poor)
        self.assertGreater(database.athlete_rating(1)["rating"],
                           database.athlete_rating(2)["rating"])


if __name__ == "__main__":
    unittest.main()
