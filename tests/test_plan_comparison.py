"""Seven plans a phone can actually compare.

Stacked as cards they run about 7,500px, so comparing two of them means
scrolling past the other five and remembering. The cards are long because of
the detail somebody wants once they have chosen, so the cards are not the
thing to shorten - the choosing moves into a table instead.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from browser_client import BrowserClient as TestClient

from app.main import app
from core.payments import PLANS, comparison_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ComparisonDataTests(unittest.TestCase):
    def test_every_plan_has_a_row(self):
        self.assertEqual([r["key"] for r in comparison_rows()], list(PLANS))

    def test_the_columns_compare_the_same_quantity_down_the_page(self):
        """limit_label could not be used for this: it says "3 analyses every
        day" for an athlete plan and "Up to 5 fighters" for a coach one, so a
        column of it compares two different things."""
        for row in comparison_rows():
            with self.subTest(plan=row["key"]):
                self.assertTrue(row["fighters"])
                self.assertTrue(row["daily"])
                self.assertTrue(re.fullmatch(r"\d+|Unlimited", row["fighters"]))
                self.assertTrue(re.fullmatch(r"\d+|Unlimited", row["daily"]))

    def test_no_cell_is_ever_blank(self):
        """An empty cell in a comparison reads as "this plan does not have
        that", which is the opposite of what unlimited means."""
        gym = [r for r in comparison_rows() if r["key"] == "gym"][0]
        self.assertEqual(gym["fighters"], "Unlimited")
        self.assertEqual(gym["daily"], "Unlimited")

    def test_the_prices_are_the_ones_the_cards_show(self):
        for row in comparison_rows():
            with self.subTest(plan=row["key"]):
                self.assertEqual(row["price"], PLANS[row["key"]]["price"])

    def test_the_ladder_climbs(self):
        """A comparison that is not ordered is a list."""
        paid = [r for r in comparison_rows() if r["price"] != "€0"]
        amounts = [float(r["price"].lstrip("€")) for r in paid]
        self.assertEqual(amounts, sorted(amounts))


class ComparisonPageTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.page = self.client.get("/pricing").text

    def tearDown(self):
        self.client.close()

    def test_the_table_is_on_the_page_above_the_cards(self):
        self.assertIn('class="plan-compare"', self.page)
        self.assertLess(self.page.index('class="plan-compare"'),
                        self.page.index('class="pricing-grid"'))

    def test_it_is_a_real_table_with_real_headers(self):
        """A screen reader announces the column with the cell, which is what
        a comparison needs and what a grid of divs cannot do."""
        self.assertIn("<table", self.page)
        for column in ("Plan", "Price", "Fighters", "Report"):
            with self.subTest(column=column):
                # Report carries a class so it can drop on a narrow screen,
                # so match the scope rather than the exact tag.
                self.assertRegex(self.page, rf'<th scope="col"[^>]*>{column}</th>')
        # "Per day" is wrapped in an abbr, so it is matched on its title.
        self.assertIn('<abbr title="Analyses per day">Per day</abbr>', self.page)
        # Each row names itself, so no cell is announced without its plan.
        self.assertEqual(self.page.count('<th scope="row">'), len(PLANS))

    def test_every_row_links_to_a_card_that_exists(self):
        links = re.findall(r'<th scope="row"><a href="#(plan-[a-z0-9_]+)"', self.page)
        self.assertEqual(len(links), len(PLANS))
        for anchor in links:
            with self.subTest(anchor=anchor):
                self.assertIn(f'id="{anchor}"', self.page)

    def test_every_plan_appears_exactly_once_in_the_table(self):
        body = self.page.split('class="plan-compare"')[1].split("</table>")[0]
        for key, plan in PLANS.items():
            with self.subTest(plan=key):
                self.assertEqual(body.count(f'href="#plan-{key}"'), 1)

    def test_the_table_does_not_restate_the_early_access_badge(self):
        """The cards carry that, and saying it seven more times in a table
        whose job is to be scannable would undo the point of it."""
        body = self.page.split('class="plan-compare"')[1].split("</table>")[0]
        self.assertNotIn("Billing not open yet", body)
        self.assertNotIn("Free in early access", body)


class ComparisonLayoutTests(unittest.TestCase):
    def setUp(self):
        self.css = (PROJECT_ROOT / "app" / "static" / "components.css").read_text(encoding="utf-8")

    def test_the_least_useful_column_is_the_one_that_drops(self):
        """Every paid plan above Athlete says "Complete", so report depth
        separates almost nothing while costing width price and roster need."""
        after = self.css.split("@media (max-width:560px)")[1]
        # Up to the line that closes the media block at column zero.
        narrow = after.split("\n}")[0]
        self.assertIn(".plan-compare .compare-report{display:none}", narrow)

    def test_numbers_line_up(self):
        """Comparing 10 against 9.99 in a proportional font is comparing
        their widths."""
        self.assertIn("font-variant-numeric:tabular-nums", self.css)


if __name__ == "__main__":
    unittest.main()
