"""Boxing and taekwondo reports: what they claim when strike counts are withheld.

Found by building boxing, WT and ITF taekwondo reports with the real report
code and reading the rendered pages. Each test is one sentence that was wrong.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "report_sample.json"


def _sample():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class WithheldScoreReasonTests(unittest.TestCase):
    """The reason beside "Not scored" blamed the footage for a product limit."""

    def test_boxing_is_told_the_real_reason(self):
        from app.main import _score_withheld

        report = {"scorecard": {"available": False, "status": "punch_counting_unavailable",
                                "sport": "boxing"},
                  "tracking": {"fighter_A_coverage": .89, "fighter_B_coverage": .87}}
        withheld = _score_withheld(report)
        self.assertIn("Boxing is scored on punches", withheld["reason"])
        self.assertNotIn("Tracking", withheld["reason"])
        self.assertIn("Nothing to redo", withheld["fix"])

    def test_unvalidated_outcomes_are_not_blamed_on_the_footage(self):
        from app.main import _score_withheld

        report = {"scorecard": {"available": False, "status": "insufficient_scoring_actions",
                                "sport": "taekwondo", "evidence": {"scoring_action_candidates": 0}},
                  "integrity": {"action_metrics_trusted": False}}
        withheld = _score_withheld(report)
        self.assertNotIn("clear enough", withheld["reason"])
        self.assertIn("which strikes landed", withheld["reason"])

    def test_trusted_outcomes_keep_the_count_based_reason(self):
        from app.main import _score_withheld

        report = {"scorecard": {"available": False, "status": "insufficient_scoring_actions",
                                "evidence": {"scoring_action_candidates": 3}},
                  "integrity": {"action_metrics_trusted": True}}
        self.assertIn("Only 3 were clear enough", _score_withheld(report)["reason"])


class BoxingScorecardDisclaimerTests(unittest.TestCase):
    """Boxing's scorecard spoke of hands and feet and a leg-strike count."""

    def test_boxing_disclaimer_describes_boxing(self):
        from core.report import STRIKE_COUNTS_PRECISION_VALIDATED, build_report
        from core.types import AnalysisRequest, RoundSpec, StrikeEvent

        if STRIKE_COUNTS_PRECISION_VALIDATED:
            self.skipTest("punch counts are published")
        sample = _sample()
        req = AnalysisRequest(video_path="none.mp4", fighter_a_box=[0, 0, 10, 10],
                              fighter_b_box=[20, 0, 30, 10], ruleset="BOXING", round_count=1,
                              round_duration_seconds=60.0, break_duration_seconds=0.0)
        # A boxing bout has punches in it; with none the report stops at
        # "no scoring candidates" and never reaches the boxing wording.
        punches = [StrikeEvent(
            fighter=who, opponent="B" if who == "A" else "A", round_number=1,
            start_frame=i * 30, peak_frame=i * 30 + 3, end_frame=i * 30 + 6,
            start_time=i * 2.0, peak_time=i * 2.0 + .2, end_time=i * 2.0 + .4,
            technique=technique, family="punch", limb="left_hand", outcome="clean", landed=True,
            target="head", confidence=.9, contact_confidence=.8)
            for i, (who, technique) in enumerate(
                [("A", "jab"), ("B", "cross"), ("A", "left_hook"), ("B", "jab")] * 4)]
        report = build_report(req, "clip.mp4", [RoundSpec(1, 0.0, 60.0)], punches, [],
                              sample["metrics"], sample["tracking"], sample["performance"],
                              sample["classifier"])
        disclaimer = report["scorecard"]["disclaimer"]
        if report["scorecard"].get("status") != "punch_counting_unavailable":
            self.skipTest("boxing was withheld for another reason on this sample")
        self.assertIn("Boxing is scored on punches", disclaimer)
        self.assertNotIn("feet", disclaimer)
        self.assertNotIn("leg-strike", disclaimer)


class VisualsWithoutCountedOutcomesTests(unittest.TestCase):
    """"Whether a kick landed is not counted yet" above "Kicks landed 7"."""

    @staticmethod
    def _report(sport="taekwondo"):
        report = _sample()
        report.setdefault("scorecard", {})["sport"] = sport
        return report

    def test_nothing_says_what_landed_when_outcomes_are_not_counted(self):
        from core.report_visuals import build

        visuals = build(self._report(), "A", outcomes_counted=False)
        self.assertFalse(visuals["strikes_shown"])
        self.assertNotIn("landed", {row["key"] for row in visuals["head_to_head"]})
        self.assertEqual(visuals["timeline"], [])
        self.assertEqual(sum(visuals["landed"].values()) + sum(visuals["taken"].values()), 0)
        # The pose rows stay: they are measured, not inferred from strikes.
        self.assertIn("guard", {row["key"] for row in visuals["head_to_head"]})

    def test_boxing_never_gets_a_kicks_only_map(self):
        from core.report_visuals import build

        visuals = build(self._report("boxing"), "A", outcomes_counted=True)
        self.assertFalse(visuals["strikes_shown"])

    def test_counted_outcomes_keep_the_strike_sections(self):
        from core.report_visuals import build

        visuals = build(self._report("kickboxing"), "A", outcomes_counted=True)
        self.assertTrue(visuals["strikes_shown"])
        self.assertIn("landed", {row["key"] for row in visuals["head_to_head"]})

    def test_the_page_passes_whether_outcomes_were_counted(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('outcomes_counted=bool((report.get("statistics") or {}).get("action_labels_available"))',
                      source)


class TaekwondoWordingTests(unittest.TestCase):
    """Taekwondo awards no knees, and its pages said knees were withheld."""

    def test_taekwondo_withholds_punches_only(self):
        from app.main import _reported_strike_families
        from core.report import STRIKE_COUNTS_PRECISION_VALIDATED

        if STRIKE_COUNTS_PRECISION_VALIDATED:
            self.skipTest("punch counts are published")
        self.assertEqual(_reported_strike_families("taekwondo")["withheld_families"], "punches")

    def test_the_striking_aside_only_sits_beside_a_score(self):
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(
            encoding="utf-8")
        self.assertIn("{% if report.scorecard.coverage_note and report.scorecard.available %}", page)
        self.assertNotIn("counts the punches, kicks and knees it observed", page)


if __name__ == "__main__":
    unittest.main()


class MuayThaiAndMmaReportTests(unittest.TestCase):
    """Found reading full-plan Muay Thai and MMA reports; true of every sport."""

    def test_a_level_measurement_is_not_something_to_work_on(self):
        """77.0% against 77.1% was "Work on: Holding the middle - level with
        your opponent here", with a drill attached."""
        from core.coaching import build_pose_coaching

        own = {"guard_index": 0.12, "ring_center_control": 0.769, "balance_index": 0.76}
        them = {"guard_index": 0.24, "ring_center_control": 0.771, "balance_index": 0.71}
        titles = [item["title"] for item in build_pose_coaching("A", own, them)["improvements"]]
        self.assertTrue(any("Guard" in title for title in titles), titles)
        self.assertFalse(any("Holding the middle" in title for title in titles), titles)

    def test_the_scorecard_box_agrees_with_the_reason_at_the_top(self):
        """The stored text said "this analysis verified 4" beside "4
        unverified scoring candidates" and "no strike is counted"."""
        from app.main import _score_withheld

        for status, sport in (("insufficient_scoring_actions", "muay_thai"),
                              ("punch_counting_unavailable", "mma"),
                              ("punch_counting_unavailable", "boxing")):
            with self.subTest(status=status, sport=sport):
                withheld = _score_withheld({
                    "scorecard": {"available": False, "status": status, "sport": sport},
                    "integrity": {"action_metrics_trusted": False}})
                self.assertIn("No score is shown", withheld["disclaimer"])
                self.assertNotIn("verified", withheld["disclaimer"])
                self.assertNotIn("timeline", withheld["disclaimer"])
        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('report["scorecard"]["disclaimer"] = score_withheld["disclaimer"]', source)

    def test_new_analyses_do_not_call_candidates_verified(self):
        source = (Path(__file__).resolve().parents[1] / "core" / "report.py").read_text(encoding="utf-8")
        self.assertNotIn('f"analysis verified {candidate_count}', source)

    def test_the_kickboxing_rules_note_stays_on_kickboxing(self):
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(
            encoding="utf-8")
        self.assertIn("{% if report.scorecard.sport|default('kickboxing', true) == 'kickboxing' %}"
                      "<p class=\"small muted\">{{report.integrity.rules_reference}}</p>", page)

    def test_the_model_is_named_by_file_not_by_local_path(self):
        from jinja2 import Environment

        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(
            encoding="utf-8")
        expression = "{{ (report.performance.pose_model|string).replace('\\\\', '/').split('/')|last }}"
        self.assertIn(expression, page)
        rendered = Environment().from_string(expression).render(report={"performance": {
            "pose_model": r"C:\Users\User\Documents\WarriorIQ\models\yolo26m-pose.engine"}})
        self.assertEqual(rendered, "yolo26m-pose.engine")
