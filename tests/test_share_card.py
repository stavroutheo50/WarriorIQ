"""The stats-only story card (core.report.share_card)."""

import json
import unittest

from core.report import share_card


def report(**overrides):
    base = {
        "integrity": {"identity_evidence_trusted": True},
        "statistics": {"fighters": {
            "A": {"punch_attempts": 41, "kick_attempts": 28, "knee_attempts": 7},
            "B": {"punch_attempts": 12, "kick_attempts": 30, "knee_attempts": 0},
        }},
        "scorecard": {"available": True, "sport": "kickboxing", "sport_label": "Kickboxing",
                      "totals": {"A": 29, "B": 28}},
        "coaching": {"A": {"strengths": [{"title": "Busy jab"}], "improvements": [{"title": "Guard drops after kicks"}]}},
        "video": {"original_name": "Nikos Papadopoulos vs Someone.mp4"},
        "setup": {"fighter_name": "Nikos"},
    }
    base.update(overrides)
    return base


class ShareCardTests(unittest.TestCase):
    def test_the_numbers_are_the_pages_numbers(self):
        card = share_card(report())
        self.assertEqual(card["fighters"]["A"]["strikes"], {"punch": 41, "kick": 28, "knee": 7})
        self.assertEqual(card["fighters"]["A"]["total"], 76)
        self.assertEqual(card["fighters"]["B"]["total"], 42)
        self.assertEqual(card["score"], {"A": 29, "B": 28})
        self.assertEqual(card["fighters"]["A"]["strength"], "Busy jab")
        self.assertEqual(card["fighters"]["A"]["working_on"], "Guard drops after kicks")
        self.assertIsNone(card["fighters"]["B"]["strength"])
        self.assertTrue(card["note"])

    def test_only_the_families_the_sport_scores(self):
        card = share_card(report(scorecard={"available": False, "sport": "boxing", "totals": {}}))
        self.assertEqual(card["fighters"]["A"]["strikes"], {"punch": 41})

    def test_no_card_when_it_cannot_say_whose_strikes_they_were(self):
        self.assertIsNone(share_card(report(integrity={"identity_evidence_trusted": False})))
        self.assertIsNone(share_card(report(statistics={})))

    def test_no_score_unless_the_page_shows_one(self):
        withheld = report(scorecard={"available": False, "sport": "kickboxing", "totals": {"A": 29, "B": 28}})
        self.assertIsNone(share_card(withheld)["score"])
        missing = report(scorecard={"available": True, "sport": "kickboxing", "totals": {"A": None, "B": None}})
        self.assertIsNone(share_card(missing)["score"])

    def test_no_names_on_the_card(self):
        text = json.dumps(share_card(report()))
        self.assertNotIn("Nikos", text)
        self.assertNotIn("Someone", text)


if __name__ == "__main__":
    unittest.main()
