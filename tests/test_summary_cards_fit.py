"""The final coaching summary cards fill the row (QA, 2026-10-07, item 24).

.performance-insights was a fixed four-column grid; the summary usually has
two cards, so they took half the row. A long fighter name wrapped over several
lines in the eyebrow and in "This target comes from <name>'s measured report".
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SummaryCardTests(unittest.TestCase):
    def test_grid_fits_however_many_cards_there_are(self):
        css = (ROOT / "app" / "static" / "fixes.css").read_text(encoding="utf-8")
        rules = re.findall(r"\.performance-insights\{[^}]*grid-template-columns:([^;}]*)", css)
        self.assertTrue(rules)
        for columns in rules:
            self.assertNotIn("repeat(4", columns)
            self.assertNotIn("repeat(2,1fr)", columns)
        self.assertIn(".performance-insights{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))", css)
        self.assertIn(".fighter-name-fit{display:inline-block;max-width:16em;overflow:hidden;text-overflow:ellipsis;"
                      "white-space:nowrap", css)

    def test_long_names_are_cut_with_a_title(self):
        page = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        summary = page[page.index('id="report-summary"'):]
        summary = summary[:summary.index("</section>")]
        self.assertEqual(summary.count('<span class="fighter-name-fit" title="{{ summary_name }}">{{ summary_name }}</span>'), 2)
        self.assertNotIn("names.get(primary, 'Fighter ' ~ primary) }}’s measured report", summary)


if __name__ == "__main__":
    unittest.main()
