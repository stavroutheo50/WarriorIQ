"""The movement card is a comparison, scored only where the sport is judged so.

QA, 2026-10-07: "Movement scorecard 10-10" sat next to "Score: Not scored" on
/result/b0bc06c741a7 (WT taekwondo) and /result/480b87c056cf (point
fighting), worded in ten-point-must criteria - effective aggression, ring
generalship, territory - for sports decided by counting techniques.
"""

from __future__ import annotations

import unittest

from core.generalship import judge_fight, movement_comparison
from core.squad import summarize_fight
from core.types import RoundSpec


class _Metrics:
    """Two fighters over one 20 s round: A walks B back across the area."""

    def __init__(self):
        self.timed_pressure = []
        self.timed_positions = []
        for index in range(120):
            t = index / 6.0
            self.timed_pressure += [(t, "A", 0.6), (t, "B", -0.4)]
            self.timed_positions += [(t, "A", 300.0 + index, 300.0), (t, "B", 420.0 + 2 * index, 300.0)]


def _round():
    return RoundSpec(number=1, start_seconds=0.0, end_seconds=20.5)


def _card(score_rounds: bool) -> dict:
    return judge_fight(_Metrics(), [_round()], {"A": 0.95, "B": 0.95}, 0.85, score_rounds=score_rounds)


LEGACY_10_10 = {
    "available": True, "status": "movement_criteria_only", "totals": {"A": 10, "B": 10}, "leader": None,
    "rounds_won": {"A": 0, "B": 0},
    "criteria_scored": ["effective aggression", "ring generalship", "territory"],
    "rounds": [{"number": 1, "A": 10, "B": 10, "winner": None, "margin": 0.02,
                "aggression": {"A": 0.51, "B": 0.49}, "generalship": {"A": 0.5, "B": 0.5},
                "territory": {"A": 0.52, "B": 0.48}}],
}


class AnalysisOutputTests(unittest.TestCase):
    def test_no_round_scores_unless_asked(self):
        card = _card(False)
        self.assertTrue(card["available"])
        self.assertNotIn("totals", card)
        self.assertNotIn("criteria_scored", card)
        row = card["rounds"][0]
        self.assertNotIn("A", row)
        self.assertNotIn("B", row)
        self.assertEqual(row["leader"], "A")
        self.assertEqual(card["rounds_led"], {"A": 1, "B": 0})
        self.assertGreater(row["aggression"]["A"], row["aggression"]["B"])

    def test_ten_point_must_with_scoring_on_keeps_the_round_score(self):
        card = _card(True)
        self.assertEqual(card["totals"], {"A": 10, "B": 9})
        self.assertEqual((card["rounds"][0]["A"], card["rounds"][0]["B"]), (10, 9))


class RenderViewTests(unittest.TestCase):
    NAMES = {"A": "Ana", "B": "Bea"}

    def test_counted_rulesets_get_neutral_wording_and_no_score(self):
        for ruleset in ("WT_TAEKWONDO", "ITF_TAEKWONDO", "POINT_FIGHTING", "K1"):
            for scoring in (False, True):
                with self.subTest(ruleset=ruleset, scoring=scoring):
                    view = movement_comparison(LEGACY_10_10, ruleset, scoring, self.NAMES)
                    self.assertEqual(view["title"], "Movement comparison")
                    self.assertFalse(view["judged"])
                    self.assertIsNone(view["scores"])
                    self.assertNotIn("score", view["rounds"][0])
                    text = str(view)
                    for word in ("effective aggression", "ring generalship", "judging criteria", "10-10"):
                        self.assertNotIn(word, text)
                    self.assertIn("decided by counting scoring techniques", view["basis"])
                    self.assertIn("not a score", view["basis"])

    def test_ten_point_must_keeps_criteria_wording_but_no_score_while_scoring_is_off(self):
        for ruleset in ("BOXING", "MUAY_THAI", "MMA"):
            with self.subTest(ruleset=ruleset):
                view = movement_comparison(LEGACY_10_10, ruleset, False, self.NAMES)
                self.assertTrue(view["judged"])
                self.assertIsNone(view["scores"])
                criteria = {column["criterion"] for column in view["columns"]}
                self.assertEqual(criteria, {"effective aggression", "ring generalship", None})

    def test_ten_point_must_with_scoring_on_shows_the_stored_score(self):
        view = movement_comparison(LEGACY_10_10, "BOXING", True, self.NAMES)
        self.assertEqual(view["scores"]["totals"], {"A": 10, "B": 10})
        self.assertEqual(view["rounds"][0]["score"], {"A": 10, "B": 10})

    def test_the_read_names_who_led_and_on_what(self):
        view = movement_comparison(_card(False), "WT_TAEKWONDO", False, self.NAMES)
        self.assertIn("Ana", view["rounds"][0]["read"])
        close = movement_comparison(LEGACY_10_10, "WT_TAEKWONDO", False, self.NAMES)
        self.assertEqual(close["rounds"][0]["read"], "Too close to separate on movement.")

    def test_withheld_card_keeps_its_reason(self):
        view = movement_comparison({"available": False, "reason": "Needs 85% tracking."}, "BOXING", False)
        self.assertFalse(view["available"])
        self.assertEqual(view["reason"], "Needs 85% tracking.")
        self.assertIsNone(movement_comparison({}, "BOXING", False))


class SquadTests(unittest.TestCase):
    def test_squad_verdict_reads_rounds_led_not_a_score(self):
        report = {"video": {"focus_fighter": "A"}, "movement_scorecard": _card(False),
                  "tracking": {"fighter_A_coverage": 0.9, "fighter_B_coverage": 0.9}}
        row = summarize_fight(report, {"job_id": "x"})
        self.assertEqual(row["movement_verdict"], "ahead on movement")


if __name__ == "__main__":
    unittest.main()
