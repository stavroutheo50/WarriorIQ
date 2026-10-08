"""The report's "Fix this first" card: one measured fault, its moment, its drill."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from core.report import fix_first

ROOT = Path(__file__).resolve().parents[1]


def _report(*, trusted=True, improvements=None, drills=None, plan=None):
    return {
        "integrity": {"identity_evidence_trusted": trusted},
        "coaching": {"A": {
            "improvements": improvements if improvements is not None else [{
                "title": "Work on: Guard 22%", "detail": "You 22%, them 31% - behind your opponent here.",
                "evidence_times": [74.5, 12.0, 30.0], "metric": "guard_index"}],
            "drills": drills if drills is not None else [{
                "name": "Fighter A · Hands-home rounds", "prescription": "3 x 2 min: return both hands to the chin.",
                "why": "You 22%, them 31% - behind your opponent here.", "metric": "guard_index"}],
        }},
        "training_plan": {"A": plan if plan is not None else [{
            "work": "3 x 2 min: return both hands to the chin.", "goal": "Raise guard from 22% to 30%."}]},
    }


class FixFirstTests(unittest.TestCase):
    def test_fault_moment_and_drill(self):
        card = fix_first(_report(), "A")
        self.assertEqual(card["title"], "Guard 22%")
        self.assertEqual(card["moment_seconds"], 12.0)
        self.assertEqual(card["moment_clock"], "0:12")
        self.assertEqual(card["more_moments"], 2)
        self.assertEqual(card["drill"], {"name": "Hands-home rounds",
                                         "prescription": "3 x 2 min: return both hands to the chin.",
                                         "goal": "Raise guard from 22% to 30%."})

    def test_older_reports_match_the_drill_by_its_sentence(self):
        report = _report()
        del report["coaching"]["A"]["improvements"][0]["metric"]
        del report["coaching"]["A"]["drills"][0]["metric"]
        self.assertIsNotNone(fix_first(report, "A")["drill"])

    def test_a_drill_for_another_measurement_is_never_borrowed(self):
        report = _report(drills=[{"name": "x · Pressure", "prescription": "p", "why": "other", "metric": "pressure_index"}])
        card = fix_first(report, "A")
        self.assertIsNone(card["drill"])
        self.assertEqual(card["moment_clock"], "0:12")

    def test_plan_without_training_hides_the_drill(self):
        self.assertIsNone(fix_first(_report(), "A", training_items=0)["drill"])

    def test_nothing_to_fix_or_identity_failed_means_no_card(self):
        nothing = _report(improvements=[{"title": "Nothing behind your opponent", "detail": "...",
                                         "evidence_times": []}], drills=[])
        self.assertIsNone(fix_first(nothing, "A"))
        self.assertIsNone(fix_first(_report(trusted=False), "A"))
        self.assertIsNone(fix_first(_report(improvements=[]), "A"))

    def test_new_findings_carry_their_measurement(self):
        source = (ROOT / "core" / "coaching.py").read_text(encoding="utf-8")
        self.assertIn('"metric": _key,\n        })', source)


class TemplateTests(unittest.TestCase):
    def _page(self, report_extra):
        from tests_support import render_result

        report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
        report.update(report_extra)
        return render_result(report=report, numbers={"state": "ok"}, job_id="abc")

    def test_card_renders_with_a_replay_jump_and_fix_next_is_not_repeated(self):
        page = self._page(_report())
        self.assertIn('id="fix-first"', page)
        self.assertIn("Fix this first", page)
        self.assertIn('href="/replay/abc?t=11.00"', page)
        self.assertIn("Watch it at 0:12", page)
        self.assertIn("not a replacement for your coach", page)
        self.assertNotIn("<span>Fix next</span>", page)

    def test_without_a_fault_the_old_row_stays(self):
        page = self._page(_report(improvements=[], drills=[]))
        self.assertNotIn('id="fix-first"', page)
        self.assertIn("<span>Fix next</span>", page)


if __name__ == "__main__":
    unittest.main()
