"""Who each column is, and what each selection radio is for (QA, 2026-10-07, item 23).

The "How the fight went" groups are role="table" with a row header and two
cells per row, and no header row: nothing, visible or spoken, said which
number was yours. On the selection page the person lists for Fighter A and
Fighter B gave their radios identical names, and the report-focus radios
read "A Fighter A" because the badge letter was part of the label.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from tests_support import render_result

ROOT = Path(__file__).resolve().parents[1]

NUMBERS = {"centre_seconds": 41.0, "seen_seconds": 80.0, "forward_seconds": 20.0, "backward_seconds": 12.0,
           "circling_seconds": 30.0, "standing_seconds": 18.0, "distance_body_lengths": 33.0,
           "range_seconds": {"close": 10.0, "middle": 40.0, "long": 30.0}, "hands_up_share": 0.4,
           "longest_hands_down_seconds": 6.0, "off_balance_count": 2, "pace_change_percent": -8, "rounds": {}}


class NumbersHeaderTests(unittest.TestCase):
    def setUp(self):
        report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
        for side in ("A", "B"):
            report["metrics"].setdefault(side, {})["numbers"] = dict(NUMBERS)
        self.page = render_result(report=report, names={"A": "Alex Kicks", "B": "Opponent"})
        section = self.page[self.page.index('id="report-numbers"'):]
        self.section = section[:section.index("</section>")]

    def test_every_group_has_a_header_row(self):
        groups = re.findall(r'<div class="fight-numbers-group" role="table".*?(?=<div class="fight-numbers-group"|$)',
                            self.section, flags=re.S)
        self.assertGreaterEqual(len(groups), 4)
        for group in groups:
            headers = re.findall(r'role="columnheader"[^>]*>([^<]*)<', group)
            self.assertEqual(headers, ["Measure", "Alex Kicks", "Opponent"], group[:80])

    def test_long_names_carry_a_title(self):
        self.assertIn('role="columnheader" title="Alex Kicks"', self.section)

    def test_header_columns_line_up_with_the_numbers(self):
        css = (ROOT / "app" / "static" / "components.css").read_text(encoding="utf-8")
        self.assertIn(".fight-numbers-head [role=columnheader]:nth-child(2){grid-column:2}", css)
        self.assertIn("text-overflow:ellipsis", css[css.index(".fight-numbers-head [role=columnheader]{"):][:300])


class SelectionRadioTests(unittest.TestCase):
    def setUp(self):
        self.page = (ROOT / "app" / "templates" / "select.html").read_text(encoding="utf-8")

    def test_focus_badges_are_not_read(self):
        self.assertIn('<i aria-hidden="true">A</i> Fighter A', self.page)
        self.assertIn('<i aria-hidden="true">B</i> Fighter B', self.page)

    def test_person_radios_say_which_fighter_they_are_for(self):
        self.assertIn("input.setAttribute('aria-label',`Fighter ${side}: ${text.textContent}`)", self.page)

    def test_every_static_radio_is_inside_a_label_with_text(self):
        for match in re.finditer(r'<label>(<input type="radio"[^>]*>)<span>(.*?)</span></label>', self.page):
            text = re.sub(r"<[^>]*>|\{[{%].*?[%}]\}", "", match.group(2)).strip()
            self.assertTrue(text, match.group(1))


if __name__ == "__main__":
    unittest.main()
