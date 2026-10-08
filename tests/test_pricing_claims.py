"""Pricing says what each plan really does, from the data that enforces it.

QA, 2026-10-07: "3 verified key moments", "Up to 8 verified key moments",
"Scorecard and core coaching" and "Full evidence replay and legality review"
were listed while those features were switched off on every report; the
Coach 15 card left out its 20 analyses a day; "Not enough room" said nothing;
and "No video filename shown" sat above a library searched by filename.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from core import payments


def _features(on: bool, plan_key: str) -> list[str]:
    with patch.object(payments, "feature_on", lambda key: on):
        return payments.plan_features(payments.PLANS[plan_key])


class PlanFeatureTests(unittest.TestCase):
    def test_every_plan_states_its_own_daily_limit(self):
        for key, plan in payments.PLANS.items():
            with self.subTest(plan=key):
                listed = " ".join(_features(False, key))
                if plan.get("unlimited") or plan.get("daily_limit") is None:
                    self.assertIn("Unlimited fight analyses", listed)
                else:
                    daily = plan["daily_limit"]
                    self.assertIn(f"{daily} analys{'is' if daily == 1 else 'es'} per day", listed)
        self.assertIn("20 analyses per day", _features(False, "coach_15"))

    def test_switched_off_features_are_not_sold(self):
        for key in payments.PLANS:
            with self.subTest(plan=key):
                listed = " ".join(_features(False, key)).lower()
                for claim in ("key moment", "scorecard", "legality"):
                    self.assertNotIn(claim, listed)

    def test_switched_on_they_come_from_the_plan_limits(self):
        self.assertIn("3 verified key moments", _features(True, "free"))
        self.assertIn("Up to 8 verified key moments", _features(True, "athlete"))
        self.assertIn("Scorecard and core coaching", _features(True, "athlete"))
        self.assertIn("Full evidence replay and legality review", _features(True, "coach_5"))


class PricingPageTests(unittest.TestCase):
    def test_the_page_shows_the_generated_lists_and_no_filename_promise(self):
        from fastapi.testclient import TestClient

        import app.main as webapp

        page = TestClient(webapp.app).get("/pricing?audience=coach").text
        self.assertNotIn("No video filename shown", page)
        for feature in payments.PLANS["coach_15"]["features"]:
            self.assertIn(feature, page)


if __name__ == "__main__":
    unittest.main()
