"""The report leads with its answer: fix this, keep this, then the rest folded (2026-10-09)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from core.report import _clock as fix_clock, did_well

ROOT = Path(__file__).resolve().parents[1]


def _report(*, trusted=True, strengths=None, improvements=None):
    return {
        "integrity": {"identity_evidence_trusted": trusted},
        "coaching": {"A": {
            "strengths": strengths if strengths is not None else [
                {"title": "Consistent guard · hands up 78% of the time",
                 "detail": "Hands were up by the face 78% of the time they could be measured.",
                 "evidence_times": [40.0, 9.5]}],
            "improvements": improvements if improvements is not None else [{
                "title": "Work on: Pressure 12%", "detail": "You 12%, them 30%.",
                "evidence_times": [20.0], "metric": "pressure_index"}],
            "drills": [],
        }},
        "training_plan": {"A": []},
    }


class DidWellTests(unittest.TestCase):
    def test_first_strength_and_its_first_moment(self):
        card = did_well(_report(), "A")
        self.assertEqual(card["title"], "Consistent guard · hands up 78% of the time")
        self.assertEqual(card["moment_seconds"], 9.5)
        self.assertEqual(card["moment_clock"], fix_clock(9.5))

    def test_no_strength_or_failed_identity_means_no_card(self):
        self.assertIsNone(did_well(_report(strengths=[]), "A"))
        self.assertIsNone(did_well(_report(trusted=False), "A"))
        self.assertIsNone(did_well({}, "B"))

    def test_a_strength_without_a_moment_has_no_jump(self):
        card = did_well(_report(strengths=[{"title": "Active defence · 4 actions", "detail": "d", "evidence_times": []}]), "A")
        self.assertIsNone(card["moment_seconds"])


class PageTests(unittest.TestCase):
    def _page(self, extra):
        from tests_support import render_result

        report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
        report.update(extra)
        return render_result(report=report, numbers={"state": "ok"}, job_id="abc")

    def test_answer_first_then_everything_else_folded(self):
        page = self._page(_report())
        fix, keep = page.index('id="fix-first"'), page.index('id="did-well"')
        fold = page.index('<details class="report-all" id="reportAll">')
        self.assertLess(fix, keep)
        self.assertLess(keep, fold)
        self.assertIn('href="/replay/abc?t=8.50"', page)
        folded = page[fold:page.index('<details class="report-deep-dive">')]
        self.assertTrue(folded.rstrip().endswith("</details>"))
        self.assertIn('class="numbers-first-report"', folded)

    def test_without_fix_this_first_nothing_is_folded(self):
        page = self._page(_report(improvements=[]))
        self.assertNotIn('id="fix-first"', page)
        self.assertNotIn('id="reportAll"', page)

    def test_links_into_a_fold_open_it(self):
        source = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn("fold.open=true", source)


if __name__ == "__main__":
    unittest.main()
