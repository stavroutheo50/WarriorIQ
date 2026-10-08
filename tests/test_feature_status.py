"""One status per feature, read by every page that describes the product.

QA, 2026-10-07: the homepage hero, the meta description and "Confidence-gated
scorecards" promised a round-by-round score estimate while every report said
"Not scored", because strikes cannot be read reliably yet (core/features.py).
"""

from __future__ import annotations

import os
import re
import unittest
from unittest.mock import patch

from core import features


def _status(env: dict) -> dict:
    with patch.dict(os.environ, env, clear=False):
        loaded = features._load()
    return {key: {"on": f.on, "name": f.name, "label": f.label, "coming": f.coming} for key, f in loaded.items()}


OFF = _status({"WARRIORIQ_PUBLISH_STRIKE_COUNTS": "0"})


class ConfigTests(unittest.TestCase):
    def test_everything_built_on_strike_counts_follows_them(self):
        self.assertFalse(any(item["on"] for item in OFF.values()))
        on = _status({"WARRIORIQ_PUBLISH_STRIKE_COUNTS": "1"})
        self.assertTrue(all(item["on"] for item in on.values()))

    def test_a_feature_can_be_held_off_on_its_own_but_never_on_without_counts(self):
        held = _status({"WARRIORIQ_PUBLISH_STRIKE_COUNTS": "1", "WARRIORIQ_FEATURE_SCORING": "0"})
        self.assertFalse(held["scoring"]["on"])
        self.assertTrue(held["key_moments"]["on"])
        forced = _status({"WARRIORIQ_PUBLISH_STRIKE_COUNTS": "0", "WARRIORIQ_FEATURE_SCORING": "1"})
        self.assertFalse(forced["scoring"]["on"])

    def test_the_report_reads_the_same_switch(self):
        from core.report import STRIKE_COUNTS_PUBLISHED

        self.assertEqual(STRIKE_COUNTS_PUBLISHED, features.is_on("strike_counts"))

    def test_every_feature_says_it_is_coming_while_off(self):
        for item in OFF.values():
            self.assertIn("coming", item["coming"])


def _visible(html: str) -> str:
    """Rendered text without HTML comments or tags."""
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", html)


class PublicPagesWhileOffTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.main as webapp

        self.webapp = webapp
        self.client = TestClient(webapp.app)
        self.patch = patch.dict(webapp.templates.env.globals, {"features": OFF})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_homepage_and_meta_description_do_not_promise_a_score(self):
        page = self.client.get("/").text
        description = re.search(r'<meta name="description" content="([^"]*)"', page).group(1)
        self.assertNotIn("score", description.lower())
        text = _visible(page)
        self.assertNotIn("you also get a round-by-round score estimate", text)
        self.assertNotIn("Confidence-gated scorecards", text)
        self.assertNotIn("supported key moments", text)
        self.assertIn("score estimates are coming", text.lower())

    def test_pricing_report_levels_do_not_promise_moments_or_a_scorecard(self):
        page = _visible(self.client.get("/pricing").text)
        self.assertNotIn("three key moments", page)
        self.assertNotIn("estimated scorecard", page)

    def test_sports_note_does_not_claim_kicks_are_counted(self):
        from tests_support import render

        page = _visible(render("sports.html", account={"id": 1, "email": "a@example.com"}, signed_in=True, sports={}, features=OFF))
        self.assertNotIn("Reports count kicks", page)
        self.assertIn("Strike counts are coming", page)


class ReportWhileOffTests(unittest.TestCase):
    def _page(self, scoring_on: bool):
        import json
        from pathlib import Path

        import app.main as webapp
        from tests_support import render_result

        report = json.loads((Path(__file__).resolve().parent / "fixtures" / "report_sample.json")
                            .read_text(encoding="utf-8"))
        self.assertTrue(report["scorecard"]["available"], "precondition: the fixture has a score")
        with patch.object(webapp, "feature_is_on", lambda key: scoring_on):
            webapp._withhold_score_while_counts_are_off(report)
        return report, _visible(render_result(features=OFF, report=report))

    def test_no_score_anywhere_while_scoring_is_off(self):
        report, page = self._page(scoring_on=False)
        self.assertFalse(report["scorecard"]["available"])
        self.assertEqual(report["scorecard"]["status"], "scoring_off")
        self.assertIn(OFF["scoring"]["coming"], page)
        self.assertNotIn("Automatic estimated scorecard", page)
        self.assertNotIn("Estimated score", page)
        self.assertNotIn("Rounds won", page)
        self.assertNotIn("10 - 9", page)
        self.assertNotIn("10–9", page)

    def test_flags_read_as_coming(self):
        _report, page = self._page(scoring_on=False)
        self.assertIn(OFF["illegal_moves"]["coming"], page)
        self.assertNotIn("Replay the most important supported actions", page)

    def test_the_withheld_reason_names_the_switch(self):
        import app.main as webapp

        reason = webapp._score_withheld({"scorecard": {"available": False, "status": "scoring_off"}})
        self.assertIn("coming", reason["reason"])


if __name__ == "__main__":
    unittest.main()
