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

    def test_the_answer_comes_before_the_fight_details(self):
        # 2026-10-09: on a phone the details filled the first screen.
        page = self._page(_report())
        self.assertLess(page.index('id="did-well"'), page.index('class="result-overview"'))
        self.assertIn('class="result-context-focus"', page)

    def test_without_fix_this_first_nothing_is_folded(self):
        page = self._page(_report(improvements=[]))
        self.assertNotIn('id="fix-first"', page)
        self.assertNotIn('id="reportAll"', page)

    def test_links_into_a_fold_open_it(self):
        source = (ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn("fold.open=true", source)


if __name__ == "__main__":
    unittest.main()


class NumbersLeadTests(unittest.TestCase):
    """2026-10-13: opening a report showed notes, "Fix this first" and the
    fight details before any number; the six numbers now lead."""

    def _page(self, extra, **context):
        from tests_support import render_result

        report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
        report.update(extra)
        return render_result(report=report, numbers={"state": "ok"}, job_id="abc", **context)

    def test_the_six_numbers_come_first(self):
        page = self._page(_report(), recording_warnings=["The fighters are small."])
        vitals = page.index('class="fight-vitals"')
        self.assertLess(vitals, page.index('id="fix-first"'))
        self.assertLess(vitals, page.index('class="result-overview"'))
        self.assertLess(vitals, page.index('id="recordingWarnings"'))
        self.assertIn("About this recording &middot; 1 note", page)
        self.assertEqual(page.count('id="fix-first"'), 1)
        self.assertEqual(page.count('class="result-overview"'), 1)

    def test_a_missing_start_stays_in_view(self):
        span = {"label": "0:30–1:56 of 1:56", "note": "The first 0:30 of the video is not in this report.",
                "start_seconds": 30.0, "end_seconds": 116.0, "duration_seconds": 116.0}
        page = self._page(_report(), analysed_span=span)
        self.assertLess(page.index("The first 0:30"), page.index('class="fight-vitals"'))

    def test_a_number_is_not_shrunk_by_the_label_style(self):
        css = (ROOT / "app" / "static" / "redesign.css").read_text(encoding="utf-8")
        self.assertNotIn(".fight-vitals span {", css)
        self.assertIn(".fight-vitals article > span {", css)


class StrikeStripTests(unittest.TestCase):
    """2026-10-13: the counts go up top, under the six numbers, as estimates."""

    def _page(self, **context):
        from tests_support import render_result

        report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
        report.update(_report())
        report["statistics"] = {"fighters": {
            "A": {"attempts": 14, "punch_attempts": 11, "kick_attempts": 3, "knee_attempts": 0},
            "B": {"attempts": 9, "punch_attempts": 9, "kick_attempts": 0, "knee_attempts": 0}}}
        base = {"report": report, "numbers": {"state": "ok"}, "job_id": "abc",
                "strike_counts_published": True, "families_shown": ("punch", "kick")}
        base.update(context)
        return render_result(**base)

    def test_counts_sit_under_the_six_numbers(self):
        page = self._page()
        strip = page.index('id="strikeStrip"')
        self.assertLess(page.index('class="fight-vitals"'), strip)
        self.assertLess(strip, page.index('id="fix-first"'))
        self.assertIn("11 punches &middot; 3 kicks", page)
        self.assertIn("Estimates, counted automatically", page)
        self.assertNotIn("knee", page[strip:page.index('id="fix-first"')])

    def test_estimated_outcomes_show_with_the_unclear_count(self):
        """2026-10-14 (owner request): landed/missed/blocked as estimates, never accuracy."""
        from tests_support import render_result

        report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
        report.update(_report())
        estimated = {"landed": 15, "missed": 21, "blocked": 2, "evaded": 0, "uncertain": 15}
        report["statistics"] = {"fighters": {
            "A": {"attempts": 53, "punch_attempts": 37, "kick_attempts": 14, "estimated": estimated},
            "B": {"attempts": 56, "punch_attempts": 36, "kick_attempts": 18, "estimated": estimated}}}
        page = render_result(report=report, numbers={"state": "ok"}, job_id="abc",
                             strike_counts_published=True, families_shown=("punch", "kick"))
        strip = page[page.index('id="strikeStrip"'):page.index('id="fix-first"')]
        self.assertIn("~15 landed &middot; 21 missed &middot; 2 blocked &middot; 15 unclear", strip)
        self.assertNotIn("evaded", strip)
        self.assertIn("no accuracy is given", strip)

    def test_no_estimated_outcomes_on_older_reports(self):
        self.assertNotIn("unclear", self._page()[:self._page().index('id="fix-first"')])

    def test_no_strip_when_counts_are_not_published_or_unverified(self):
        self.assertNotIn('id="strikeStrip"', self._page(strike_counts_published=False))
        self.assertNotIn('id="strikeStrip"', self._page(numbers={"state": "unverified"}))
