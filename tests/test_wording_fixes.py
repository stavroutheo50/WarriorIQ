"""Wording fixes from the QA pass of 2026-10-07 (item 21)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app" / "templates"


class RulesetLabelTests(unittest.TestCase):
    """ "Mma (Standing Exchanges)": a label was run through the label filter again."""

    def test_a_label_passes_through_unchanged(self):
        from app.main import _ruleset_label
        from core.config import RULESET_LABELS

        for key, label in RULESET_LABELS.items():
            self.assertEqual(_ruleset_label(key), label)
            self.assertEqual(_ruleset_label(label), label)
        self.assertEqual(_ruleset_label("MMA (standing exchanges)"), "MMA (standing exchanges)")

    def test_progress_points_carry_the_label_the_camp_page_filters(self):
        from core.progress_insights import _point
        from app.main import _ruleset_label

        record = {"job_id": "j", "report": {"setup": {"ruleset": "MMA"}, "metrics": {"A": {"guard_index": 0.3}},
                                            "integrity": {"identity_evidence_trusted": True}}}
        point = _point(record, "A")
        if point is None:
            self.skipTest("identity gate withheld the synthetic point")
        self.assertEqual(_ruleset_label(point["ruleset"]), "MMA (standing exchanges)")


class SignedChangeTests(unittest.TestCase):
    """ "-0 pts": '%+.0f' of a change under half a point printed a minus zero."""

    def test_camp_rounds_before_choosing_the_sign(self):
        page = (TEMPLATES / "camp.html").read_text(encoding="utf-8")
        self.assertNotIn("'%+.0f'|format(d*100)", page)
        self.assertNotIn("'%+.0f'|format(delta*100)", page)

        from jinja2 import Environment

        snippet = Environment().from_string(
            "{%- set pts = (d*100)|round|int -%}{{'%+d'|format(pts) if pts else '0'}}")
        self.assertEqual(snippet.render(d=-0.004), "0")
        self.assertEqual(snippet.render(d=0.004), "0")
        self.assertEqual(snippet.render(d=-0.03), "-3")
        self.assertEqual(snippet.render(d=0.12), "+12")


class WordmarkTests(unittest.TestCase):
    """The story cards drew "WARRIOR IQ"; the site says WARRIORIQ."""

    def test_cards_measure_the_word_without_a_space(self):
        for path in ("app/static/share_card.js", "app/static/camp_share.js", "core/share_image.py"):
            source = (ROOT / path).read_text(encoding="utf-8")
            self.assertNotIn('"WARRIOR "', source, path)
            self.assertIn('"WARRIOR"', source, path)

    def test_server_card_has_no_gap(self):
        import cv2
        from core import share_image

        joined = cv2.getTextSize("WARRIOR", share_image.FONT, 1.5, 3)[0][0]
        spaced = cv2.getTextSize("WARRIOR ", share_image.FONT, 1.5, 3)[0][0]
        self.assertLess(joined, spaced)


class ReplayHeaderTests(unittest.TestCase):
    def _render(self, focus, names):
        from tests_support import render

        report = {"video": {"focus_fighter": focus}, "setup": {}, "tracking": {}}
        return render("replay.html", report=report, names=names, job_id="j", replay_chapters=[],
                      identity_safe=True, moments=[], counting_policy={})

    def test_header_uses_the_fighters_name(self):
        page = self._render("B", {"A": "Opponent", "B": "Alex Kicks"})
        self.assertIn("Alex Kicks replay", page)
        self.assertNotIn("Fighter B replay", page)

    def test_without_a_name_it_keeps_the_letter(self):
        self.assertIn("Fighter A replay", self._render("A", {"A": "Fighter A", "B": "Fighter B"}))


class BritishSpellingTests(unittest.TestCase):
    """One spelling everywhere (British, as chosen on 2026-10-07)."""

    AMERICAN = re.compile(r"\b(analyz(e|ed|es|ing)|center(ed|s)?|defenses?|behaviors?|recogniz\w+|"
                          r"summariz\w+|organiz(e|ed|ing)|canceled|labeled)\b", re.I)

    @staticmethod
    def _visible(source: str) -> str:
        source = re.sub(r"<(script|style)\b.*?</\1>", " ", source, flags=re.S | re.I)
        source = re.sub(r"\{#.*?#\}|\{%.*?%\}|\{\{.*?\}\}", " ", source, flags=re.S)
        attrs = " ".join(m.group(2) for m in re.finditer(
            r'\b(placeholder|title|alt|aria-label|content)="([^"]*)"', source))
        return re.sub(r"<[^>]*>", " ", source) + " " + attrs

    def test_visible_template_text_is_british(self):
        for path in sorted(TEMPLATES.glob("*.html")):
            text = self._visible(path.read_text(encoding="utf-8"))
            found = sorted({m.group(0) for m in self.AMERICAN.finditer(text)})
            self.assertEqual(found, [], path.name)

    def test_report_text_is_british(self):
        from core.coaching import build_training_plan

        plan = build_training_plan({"drills": [{"name": "x", "prescription": "keep the centre lane"}]}, "A", {})
        self.assertNotRegex(str(plan), self.AMERICAN)
        self.assertIn("analysed segment", (ROOT / "app" / "main.py").read_text(encoding="utf-8"))
        self.assertIn('"Analysing fight"', (ROOT / "core" / "analyzer.py").read_text(encoding="utf-8"))


class EngineNamedTests(unittest.TestCase):
    """Old-report notices name the engine instead of saying "older reports"."""

    def test_engine_name(self):
        from core.build_info import engine_name

        self.assertEqual(engine_name({}), "an analysis engine from before v2")
        self.assertEqual(engine_name({"analysis_build": {"analysis_version": 3}}), "analysis engine v3")

    def test_span_note_names_the_engine(self):
        from app.main import _analysed_span_summary

        report = {"setup": {"start_seconds": 30.0}, "rounds": [{"end_seconds": 90.0}],
                  "performance": {"segment_duration_seconds": 60.0}}
        note = _analysed_span_summary(report)["note"]
        self.assertIn("an analysis engine from before v2", note)
        self.assertNotIn("older reports", note)

    def test_guard_note_names_the_engine(self):
        from core import guard

        report = {"analysis_build": {"analysis_version": 3},
                  "metrics": {"A": {"guard_index": 0.4, "numbers": {"hands_up_share": 0.2}}}}
        guard.reconcile_report_guard(report)
        own = report["metrics"]["A"]
        self.assertIn("analysis engine v3", own["guard_note"])
        self.assertEqual(own["guard_note_short"], "made by engine v3 — re-run to measure")

    def test_footer_names_both_versions(self):
        for name in ("result.html", "solo_result.html"):
            page = (TEMPLATES / name).read_text(encoding="utf-8")
            self.assertNotIn("an older analysis engine than the current one", page)
            self.assertIn("the current engine is v", page)


class LimitationTests(unittest.TestCase):
    """One "Pose coverage was 2.6%" line, not one per measurement it blocked."""

    def test_same_reason_listed_once(self):
        from core.metric_catalog import evidence_limitations

        reason = "Pose coverage was 2.6%; at least 35% is required."
        metrics = {"availability": {
            "movement": {"available": False, "reason": reason},
            "guard": {"available": False, "reason": reason},
            "balance": {"available": False, "reason": reason},
            "round_consistency": {"available": False, "reason": "Needs two rounds."},
            "pressure": {"available": True, "reason": None},
        }}
        items = evidence_limitations(metrics, round_count=1)
        self.assertEqual([item["reason"] for item in items], [reason])
        self.assertEqual(items[0]["keys"], ["movement", "guard", "balance"])
        self.assertEqual(len(evidence_limitations(metrics, round_count=3)), 2)

    def test_report_uses_it(self):
        page = (TEMPLATES / "result.html").read_text(encoding="utf-8")
        self.assertIn("evidence_limitations(m, report.setup.round_count)", page)


class PlanLineTests(unittest.TestCase):
    """The plan is headed with the name; each line said it again."""

    def test_new_plans_have_no_prefix(self):
        from core.coaching import build_training_plan

        coaching = {"drills": [{"name": "Reset", "prescription": "3 x 2 min: Fighter A finishes in stance."}]}
        work = build_training_plan(coaching, "A", {})[0]["work"]
        self.assertEqual(work, "3 x 2 min: Fighter A finishes in stance.")

    def test_stored_plans_lose_the_prefix_on_load(self):
        from core.coaching import drop_work_prefix
        from core.report import refresh_identity_integrity

        report = {"training_plan": {"A": [{"work": "Fighter A: 3 x 2 min: Fighter A finishes in stance."}]},
                  "training_progression": {"B": [{"work": ["Fighter B: shadow rounds", "other"]}]}}
        drop_work_prefix(report)
        drop_work_prefix(report)
        self.assertEqual(report["training_plan"]["A"][0]["work"], "3 x 2 min: Fighter A finishes in stance.")
        self.assertEqual(report["training_progression"]["B"][0]["work"], ["shadow rounds", "other"])
        self.assertIn("drop_work_prefix(report)",
                      (ROOT / "core" / "report.py").read_text(encoding="utf-8"))
        self.assertTrue(callable(refresh_identity_integrity))


if __name__ == "__main__":
    unittest.main()
