"""Regression tests for the 2026-09-25 site audit.

Each class is one finding: what the page or the analysis said, why that was
wrong, and the behaviour that replaces it. Kept together so the audit can be
re-checked in one run.
"""

from __future__ import annotations

import re
import unittest

import numpy as np


class UnattributedKickTotalTests(unittest.TestCase):
    """Identity failed, and the page printed 31, 24 and 34 for the same kicks."""

    @staticmethod
    def _report(a=(9, None), b=(22, None), knees=(1, 2)):
        return {"statistics": {"fighters": {
            "A": {"kick_attempts": a[0], "kicks_landed": a[1], "knee_attempts": knees[0]},
            "B": {"kick_attempts": b[0], "kicks_landed": b[1], "knee_attempts": knees[1]},
        }}}

    def test_one_total_of_kick_attempts_without_knees(self):
        from core.report import unattributed_kick_total

        total = unattributed_kick_total(self._report())
        self.assertEqual(total["attempts"], 31)

    def test_landed_is_withheld_unless_both_sides_counted_it(self):
        from core.report import unattributed_kick_total

        self.assertIsNone(unattributed_kick_total(self._report(a=(9, 3)))["landed"])
        self.assertEqual(unattributed_kick_total(self._report(a=(9, 3), b=(22, 5)))["landed"], 8)

    def test_no_statistics_means_no_total(self):
        from core.report import unattributed_kick_total

        self.assertIsNone(unattributed_kick_total({}))


class ProgressLeavesOutIdentityFailuresTests(unittest.TestCase):
    """"Guard 20%, +11 pts" came from a run whose own report said "not safe"."""

    @staticmethod
    def _record(job_id, created, guard, trusted=True, per_fighter=None):
        integrity = {"identity_evidence_trusted": trusted}
        if per_fighter is not None:
            integrity["fighter_identity_trusted"] = per_fighter
        return {"job_id": job_id, "created_at": created, "report": {
            "video": {"analysis_target": "BOTH", "focus_fighter": "A"},
            "setup": {"ruleset": "K1"}, "integrity": integrity,
            "metrics": {"A": {"pose_coverage": .8, "guard_index": guard}},
            "coaching": {}, "training_plan": {},
        }}

    def test_an_identity_failed_run_is_not_a_point_or_a_trend(self):
        from core.progress_insights import build_progress

        progress = build_progress([
            self._record("one", "2026-09-01", .09),
            self._record("two", "2026-09-02", .20, trusted=False),
        ], "A")
        self.assertEqual(progress["fight_count"], 1)
        self.assertEqual(progress["latest"]["guard"], .09)
        self.assertIsNone(progress["trends"]["guard"])
        self.assertEqual(progress["identity_failed_count"], 1)

    def test_the_followed_fighter_failing_alone_is_enough(self):
        from core.progress_insights import build_progress

        progress = build_progress([
            self._record("one", "2026-09-01", .3, per_fighter={"A": False, "B": True}),
        ], "A")
        self.assertEqual(progress["fight_count"], 0)

    def test_legacy_rows_without_a_flag_still_count(self):
        from core.progress_insights import build_progress

        record = self._record("old", "2026-08-01", .4)
        del record["report"]["integrity"]["identity_evidence_trusted"]
        self.assertEqual(build_progress([record], "A")["fight_count"], 1)

    def test_since_your_last_fight_skips_an_identity_failure(self):
        from core.squad import summarize_fight

        report = {"video": {"focus_fighter": "A"},
                  "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9},
                  "integrity": {"identity_evidence_trusted": False}}
        self.assertFalse(summarize_fight(report, {"job_id": "x"})["usable"])
        report["integrity"]["identity_evidence_trusted"] = True
        self.assertTrue(summarize_fight(report, {"job_id": "x"})["usable"])


class AthleteNameTests(unittest.TestCase):
    """The Progress page said "My Athlete" over fights filed under Theodoulos."""

    def test_the_name_fights_were_filed_under_wins(self):
        from app.main import _athlete_name

        fights = [{"fighter_name": "Theodoulos"}, {"fighter_name": "Theodoulos"},
                  {"fighter_name": "Nikos"}, {"fighter_name": None}]
        self.assertEqual(_athlete_name({"display_name": "My Athlete"}, fights), "Theodoulos")

    def test_the_display_name_is_the_fallback(self):
        from app.main import _athlete_name

        self.assertEqual(_athlete_name({"display_name": "Coach Eleni"}, []), "Coach Eleni")


class CentreControlTellsFightersApartTests(unittest.TestCase):
    """"Held the centre" read 50-52% for both fighters on both fights."""

    def test_a_fighter_pinned_to_one_side_scores_lower_than_the_one_pinning(self):
        from core.metrics import MetricsAccumulator

        metrics = MetricsAccumulator(width=1920, height=1080)
        # Early on they use the whole mat; then A walks B onto the right-hand
        # edge and keeps them there. Measured against the average of their
        # positions - the point between them - both scored about the same.
        for step in range(40):
            metrics.positions["A"].append(np.asarray([200.0 + step * 10, 500.0], dtype=np.float32))
            metrics.positions["B"].append(np.asarray([1200.0 - step * 10, 500.0], dtype=np.float32))
        for _ in range(160):
            metrics.positions["A"].append(np.asarray([1000.0, 500.0], dtype=np.float32))
            metrics.positions["B"].append(np.asarray([1180.0, 500.0], dtype=np.float32))
        a, b = metrics._center_control("A"), metrics._center_control("B")
        self.assertGreater(a - b, 0.15, (a, b))

    def test_a_round_spent_in_the_middle_reads_high_for_both(self):
        """Both can hold the centre; the pair is not split around one half."""
        from core.metrics import MetricsAccumulator

        metrics = MetricsAccumulator(width=1920, height=1080)
        for step in range(50):
            # One lap of the edge each, which is what sets the size of the mat.
            angle = step / 50.0 * 6.28318
            for side, sign in (("A", 1.0), ("B", -1.0)):
                metrics.positions[side].append(np.asarray(
                    [500.0 + sign * 400 * np.cos(angle), 500.0 + sign * 400 * np.sin(angle)],
                    dtype=np.float32))
        for step in range(150):
            metrics.positions["A"].append(np.asarray([480.0, 500.0], dtype=np.float32))
            metrics.positions["B"].append(np.asarray([520.0, 500.0], dtype=np.float32))
        for side in ("A", "B"):
            self.assertGreater(metrics._center_control(side), 0.6, side)


class CornerLabelTests(unittest.TestCase):
    """Fighter A was always "Red corner", and on the WAKO clip A was blue."""

    def test_the_stated_corner_is_used_and_the_other_follows(self):
        from app.main import _corner_labels

        self.assertEqual(_corner_labels({"fighter_a_corner": "blue"}),
                         {"A": "Blue corner", "B": "Red corner"})
        self.assertEqual(_corner_labels({"fighter_a_corner": "red"}),
                         {"A": "Red corner", "B": "Blue corner"})

    def test_an_unstated_corner_is_not_guessed(self):
        from app.main import _corner_labels

        for job in ({}, {"fighter_a_corner": None}, {"fighter_a_corner": "green"}):
            self.assertEqual(_corner_labels(job), {"A": None, "B": None})

    def test_the_report_no_longer_hard_codes_the_colours(self):
        from pathlib import Path

        page = Path("app/templates/result.html").read_text(encoding="utf-8")
        self.assertNotIn("<small>Red corner</small>", page)
        self.assertNotIn("<small>Blue corner</small>", page)


class ReportedFamiliesTests(unittest.TestCase):
    """"We count punches, kicks and knees" on every sport, then no punches."""

    def test_boxing_is_told_it_gets_no_strike_counts(self):
        from app.main import _reported_strike_families, _sport_coverage_badge
        from core.report import STRIKE_COUNTS_PRECISION_VALIDATED

        if STRIKE_COUNTS_PRECISION_VALIDATED:
            self.skipTest("punch counts are published")
        self.assertTrue(_reported_strike_families("boxing")["no_strike_counts"])
        self.assertEqual(_sport_coverage_badge("boxing")["label"], "No punch counts yet")

    def test_kickboxing_says_kicks_only(self):
        from app.main import _reported_strike_families
        from core.report import STRIKE_COUNTS_PRECISION_VALIDATED

        if STRIKE_COUNTS_PRECISION_VALIDATED:
            self.skipTest("punch counts are published")
        families = _reported_strike_families("kickboxing")
        self.assertEqual(families["reported_families"], "kicks")
        self.assertIn("punches", families["withheld_families"])


class AnkleVisibilityTests(unittest.TestCase):
    """"Nothing obviously wrong, fighters fill 100% of height" - legs cropped."""

    class _Keypoints:
        def __init__(self, conf):
            self.conf = np.asarray(conf, dtype=np.float32)

    class _Result:
        def __init__(self, conf):
            self.keypoints = AnkleVisibilityTests._Keypoints(conf)

    def test_ankles_are_read_from_the_two_tallest_people(self):
        from core.preflight import _ankles_seen

        boxes = np.asarray([[0, 0, 10, 100], [20, 0, 30, 90], [40, 0, 50, 20]], dtype=np.float32)
        conf = np.zeros((3, 17), dtype=np.float32)
        conf[0, 15] = 0.9          # tallest: an ankle is visible
        conf[2, 16] = 0.9          # the small person does not count
        self.assertEqual(sorted(_ankles_seen(self._Result(conf), boxes)), [False, True])

    def test_no_keypoints_means_no_verdict(self):
        from core.preflight import _ankles_seen

        class Bare:
            keypoints = None

        self.assertEqual(_ankles_seen(Bare(), np.zeros((1, 4), dtype=np.float32)), [])

    def test_cropped_feet_are_warned_about(self):
        from core.preflight import Preflight, _judge

        report = Preflight(width=1920, height=1080, fps=30.0, frame_count=900, measured=True,
                           subject_height_px=900, subject_share_of_height=.83,
                           subject_px_in_network=400, people_in_frame=2,
                           ankles_visible_share=.2)
        _judge(report)
        self.assertTrue(any("feet are out of shot" in w for w in report.warnings))

    def test_the_browser_check_looks_for_cut_off_feet(self):
        from pathlib import Path

        script = Path("app/static/preflight.js").read_text(encoding="utf-8")
        self.assertIn("touchesBottom", script)
        self.assertIn("feetCropped", script)
        page = Path("app/templates/analyze.html").read_text(encoding="utf-8")
        self.assertNotIn("Nothing obviously wrong", page)
        # "Anyway" only follows a problem.
        self.assertIn("tone==='ok'?'.':'. You can upload it anyway.'", page)


class ContentSecurityPolicyTests(unittest.TestCase):
    """script-src 'unsafe-inline', and form-action naming github.com everywhere."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import app.main as webapp

        cls.client = TestClient(webapp.app)

    @staticmethod
    def _directive(policy, name):
        return next((part.strip() for part in policy.split(";")
                     if part.strip().startswith(name + " ")), "")

    def test_scripts_are_allowed_by_nonce_not_by_unsafe_inline(self):
        response = self.client.get("/")
        script_src = self._directive(response.headers["Content-Security-Policy"], "script-src")
        self.assertNotIn("'unsafe-inline'", script_src)
        nonce = re.search(r"'nonce-([^']+)'", script_src)
        self.assertIsNotNone(nonce)
        tags = re.findall(r"<script\b[^>]*>", response.text)
        self.assertTrue(tags)
        for tag in tags:
            self.assertIn(f'nonce="{nonce.group(1)}"', tag)

    def test_the_nonce_changes_on_every_response(self):
        first = self.client.get("/").headers["Content-Security-Policy"]
        second = self.client.get("/").headers["Content-Security-Policy"]
        self.assertNotEqual(re.search(r"'nonce-([^']+)'", first).group(1),
                            re.search(r"'nonce-([^']+)'", second).group(1))

    def test_no_template_uses_an_inline_event_handler(self):
        """A nonce cannot cover onsubmit= or onerror=, so they would stop working."""
        from pathlib import Path

        for path in Path("app/templates").glob("*.html"):
            with self.subTest(template=path.name):
                self.assertIsNone(re.search(r"\son[a-z]+=\"", path.read_text(encoding="utf-8")))

    def test_provider_origins_only_on_the_sign_in_pages(self):
        from core.social_auth import SOCIAL_AUTH

        self.assertEqual(SOCIAL_AUTH.form_action_origins_for("/pricing"), [])
        self.assertEqual(SOCIAL_AUTH.form_action_origins_for("/result/abc"), [])
        self.assertNotIn("https://github.com", SOCIAL_AUTH.form_action_origins_for("/signup"))
        policy = self.client.get("/pricing").headers["Content-Security-Policy"]
        self.assertEqual(self._directive(policy, "form-action"), "form-action 'self'")


class SameKitCheckTests(unittest.TestCase):
    """Two fighters in the same kit were found out 2-3 minutes too late."""

    def test_the_selection_page_asks_as_soon_as_both_boxes_exist(self):
        from pathlib import Path

        import app.main as webapp

        page = Path("app/templates/select.html").read_text(encoding="utf-8")
        self.assertIn("/api/pair-check/", page)
        self.assertIn('id="pairWarning"', page)
        paths = {getattr(route, "path", "") for route in webapp.app.routes}
        self.assertIn("/api/pair-check/{job_id}", paths)


class HeaderChipFollowsTheJobTests(unittest.TestCase):
    """"Analyzing 0%" beside a page reading 35%, and after the run finished."""

    def test_the_chip_is_updated_after_page_load(self):
        from pathlib import Path

        base = Path("app/templates/base.html").read_text(encoding="utf-8")
        self.assertIn("window.wiqUpdateAnalysisChip", base)
        self.assertIn("/api/active-analysis", base)
        progress = Path("app/templates/progress.html").read_text(encoding="utf-8")
        self.assertIn("window.wiqUpdateAnalysisChip?.(", progress)


class SportChipBelongsToTheFightTests(unittest.TestCase):
    """Opening Muay Thai in another tab relabelled a kickboxing result."""

    def test_a_fight_page_names_the_fights_own_sport(self):
        from types import SimpleNamespace

        from app.main import _pin_sport_to_fight

        request = SimpleNamespace(state=SimpleNamespace(active_sport="muay thai from the cookie"))
        _pin_sport_to_fight(request, "kickboxing")
        self.assertEqual(request.state.active_sport.key, "kickboxing")
        # An unknown sport leaves the header alone rather than blanking it.
        _pin_sport_to_fight(request, None)
        self.assertEqual(request.state.active_sport.key, "kickboxing")


class StepCountTests(unittest.TestCase):
    """"Four steps" on the homepage, "Step 2 of 3" on setup, a 3-step tracker."""

    def test_every_surface_counts_four_steps(self):
        from pathlib import Path

        templates = Path("app/templates")
        for name in ("sports.html", "analyze.html"):
            with self.subTest(template=name):
                text = (templates / name).read_text(encoding="utf-8")
                self.assertIn("of 4", text)
                self.assertNotIn("of 3", text)
        for name in ("frame.html", "select.html"):
            with self.subTest(template=name):
                tracker = (templates / name).read_text(encoding="utf-8").split(
                    '<ol class="workflow-steps"', 1)[1].split("</ol>", 1)[0]
                self.assertEqual(tracker.count("<li"), 4)


if __name__ == "__main__":
    unittest.main()
