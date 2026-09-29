"""The Progress page (/dashboard): whose fights it charts, and which it trusts.

Found by giving an account a realistic library - one athlete's fights, one
teammate's, a fight the athlete fought as Fighter B, and two whose identity
check failed - and reading the page.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "report_sample.json"


def _record(job_id, created, guard, focus="A", identity=None, fighter_id=1):
    report = {
        "video": {"analysis_target": focus, "focus_fighter": focus},
        "setup": {"ruleset": "K1"},
        "metrics": {focus: {"pose_coverage": .85, "guard_index": guard}},
        "coaching": {}, "training_plan": {},
    }
    if identity is not None:
        report["integrity"] = identity
    return {"job_id": job_id, "created_at": created, "fighter_id": fighter_id, "report": report}


class FollowedSideTests(unittest.TestCase):
    """A fight where the athlete was boxed second vanished from their progress."""

    def test_a_fight_fought_as_fighter_b_is_charted(self):
        from core.progress_insights import build_progress

        progress = build_progress([
            _record("one", "2026-08-01", .10, focus="A"),
            _record("two", "2026-08-08", .16, focus="B"),
        ], "A")
        self.assertEqual(progress["fight_count"], 2)
        self.assertEqual(progress["latest"]["guard"], .16)
        self.assertEqual(progress["latest"]["side"], "B")
        self.assertAlmostEqual(progress["trends"]["guard"], .06)

    def test_a_both_fighter_analysis_uses_the_profile_default(self):
        from core.progress_insights import followed_side

        record = _record("x", "2026-08-01", .1, focus="BOTH")
        record["report"]["video"] = {"analysis_target": "BOTH"}
        self.assertEqual(followed_side(record, "B"), "B")

    def test_identity_is_checked_on_the_side_that_was_followed(self):
        from core.progress_insights import build_progress

        progress = build_progress([
            _record("x", "2026-08-01", .2, focus="B",
                    identity={"fighter_identity_trusted": {"A": True, "B": False}}),
        ], "A")
        self.assertEqual(progress["fight_count"], 0)
        self.assertEqual(progress["identity_failed_count"], 1)


class SeparabilityIsGatedTests(unittest.TestCase):
    """A "fighters look too alike" fight was charted: the stored flag ignored it."""

    def test_build_report_stores_the_same_verdict_the_report_page_shows(self):
        from core.report import build_report
        from core.types import AnalysisRequest, RoundSpec

        sample = json.loads(FIXTURE.read_text(encoding="utf-8"))
        tracking = dict(sample["tracking"], fighters_separable=False)
        req = AnalysisRequest(video_path="none.mp4", fighter_a_box=[0, 0, 10, 10],
                              fighter_b_box=[20, 0, 30, 10], ruleset="K1", round_count=1,
                              round_duration_seconds=60.0, break_duration_seconds=0.0)
        report = build_report(req, "clip.mp4", [RoundSpec(1, 0.0, 60.0)], [], [],
                              sample["metrics"], tracking, sample["performance"], sample["classifier"])
        self.assertFalse(report["integrity"]["identity_evidence_trusted"])

    def test_the_progress_snapshot_carries_what_the_gate_needs(self):
        from core.report import identity_tracking

        kept = identity_tracking({"fighters_separable": False, "fighter_A_coverage": .9,
                                  "coverage_windows": [1, 2, 3]})
        self.assertEqual(kept, {"fighters_separable": False, "fighter_A_coverage": .9})
        for path in ("core/db.py", "core/analyzer.py"):
            source = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
            self.assertIn('"tracking": identity_tracking(report.get("tracking", {}))', source, path)

    def test_progress_reapplies_the_gate_to_saved_snapshots(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("report = refresh_identity_integrity(deepcopy(compact))", source)

    def test_the_coach_squad_does_not_count_an_inseparable_fight(self):
        from core.squad import summarize_fight

        report = {"video": {"focus_fighter": "A"},
                  "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9,
                               "fighters_separable": False},
                  "integrity": {"identity_evidence_trusted": True}}
        self.assertFalse(summarize_fight(report, {"job_id": "x"})["usable"])


class OneAthleteTests(unittest.TestCase):
    """A teammate's fight became the athlete's "last fight": -18 points."""

    def test_the_page_follows_the_fighter_most_fights_are_filed_under(self):
        from app.main import _athlete_fighter_id

        fights = [{"fighter_id": 1}, {"fighter_id": 1}, {"fighter_id": 2}, {"fighter_id": None}]
        self.assertEqual(_athlete_fighter_id(fights), 1)
        self.assertIsNone(_athlete_fighter_id([{"fighter_id": None}]))

    def test_the_dashboard_filters_to_that_fighter(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('if athlete_id is None or record.get("fighter_id") in (None, athlete_id)]', source)


if __name__ == "__main__":
    unittest.main()


class CoachPageTests(unittest.TestCase):
    """Found reading /coach with the same library."""

    def test_movement_values_use_the_units_of_every_other_page(self):
        """Coach printed pressure 0.12 and centre 0.64 for a fight the report
        shows as 56 of 100 and 64%."""
        from core.squad import movement_value

        self.assertEqual(movement_value(0.12, "pressure"), "56")
        self.assertEqual(movement_value(0.64, "centre"), "64%")
        self.assertEqual(movement_value(1.17, "footwork"), "1.2")
        self.assertEqual(movement_value(None, "centre"), "—")

    def test_a_fight_set_aside_for_identity_says_so(self):
        """"Too low to compare" beside 86% seen, on a fight set aside because
        the two fighters looked alike."""
        from core.squad import build_squad_view, summarize_fight

        alike = {"video": {"focus_fighter": "A"},
                 "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9,
                              "fighters_separable": False}}
        thin = {"video": {"focus_fighter": "A"},
                "tracking": {"fighter_A_coverage": .6, "fighter_B_coverage": .9}}
        self.assertEqual(summarize_fight(alike, {"job_id": "a"})["unusable_reason"], "identity")
        self.assertEqual(summarize_fight(thin, {"job_id": "b"})["unusable_reason"], "coverage")
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "camp.html").read_text(
            encoding="utf-8")
        self.assertIn("couldn't tell who was who", page)
        self.assertIn("squad.unusable_identity", page)

    def test_the_coach_page_uses_the_current_name_for_centre(self):
        from core.metric_catalog import RETIRED_NAMES

        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "camp.html").read_text(
            encoding="utf-8")
        for retired in RETIRED_NAMES["ring_center_control"]:
            self.assertNotIn(f"'{retired}'", page, retired)

    def test_since_your_last_fight_carries_the_key_it_formats_by(self):
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(
            encoding="utf-8")
        self.assertIn("c.now|movement_value(c.key|default(''))", page)


class ComparePageTests(unittest.TestCase):
    """Found reading /compare on five pairs from the same library."""

    @staticmethod
    def _report(focus="A", sport="kickboxing", integrity=None, tracking=None, **metrics):
        return {
            "video": {"focus_fighter": focus},
            "scorecard": {"sport": sport},
            "integrity": integrity or {},
            "tracking": {"fighter_A_coverage": .95, "fighter_B_coverage": .95, **(tracking or {})},
            "metrics": {focus: dict(metrics)},
        }

    def test_only_guard_and_balance_are_called_better_when_higher(self):
        """"Higher is better on all five", while metric_catalog gives pressure,
        centre and movement no direction."""
        from core.squad import compare_movement

        rows = {row["key"]: row for row in compare_movement([
            self._report(pressure_index=.1, ring_center_control=.6, guard_index=.1, balance_index=.7),
            self._report(pressure_index=.3, ring_center_control=.4, guard_index=.3, balance_index=.8),
        ])["rows"]}
        self.assertTrue(rows["guard_index"]["higher_is_better"])
        self.assertTrue(rows["balance_index"]["higher_is_better"])
        for key in ("pressure_index", "ring_center_control"):
            self.assertFalse(rows[key]["higher_is_better"], key)
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "compare.html").read_text(
            encoding="utf-8")
        self.assertNotIn("Higher is better on all five", page)
        self.assertIn("row.higher_is_better and row.leader == 'a'", page)

    def test_the_rows_use_the_measurements_own_names(self):
        from core.squad import compare_movement

        labels = [row["label"] for row in compare_movement([
            self._report(pressure_index=.1, ring_center_control=.6,
                         footwork_body_lengths_per_second=1.0, guard_index=.1, balance_index=.7),
            self._report(pressure_index=.2, ring_center_control=.5,
                         footwork_body_lengths_per_second=1.1, guard_index=.2, balance_index=.8),
        ])["rows"]]
        self.assertEqual(labels, ["Pressure", "Centre", "Movement", "Guard", "Balance"])

    def test_a_failed_identity_check_is_said_before_the_numbers(self):
        from core.squad import compare_movement

        result = compare_movement([
            self._report(guard_index=.1),
            self._report(guard_index=.3, tracking={"fighters_separable": False}),
        ])
        self.assertEqual(result["identity_failed"], [False, True])
        result = compare_movement([
            self._report(focus="B", guard_index=.1,
                         integrity={"fighter_identity_trusted": {"A": True, "B": False}}),
            self._report(guard_index=.3),
        ])
        self.assertEqual(result["identity_failed"], [True, False])

    def test_different_sports_are_flagged(self):
        from core.squad import compare_movement

        self.assertTrue(compare_movement([
            self._report(sport="boxing", guard_index=.2), self._report(guard_index=.2),
        ])["different_sports"])
        self.assertFalse(compare_movement([
            self._report(guard_index=.2), self._report(guard_index=.2),
        ])["different_sports"])

    def test_two_different_fighters_are_named(self):
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "compare.html").read_text(
            encoding="utf-8")
        self.assertIn("Two different fighters.", page)
        self.assertIn("picked[0].fighter_id != picked[1].fighter_id", page)
        self.assertNotIn("Movement progress is still compared below", page)


class SharedCoachLinkTests(unittest.TestCase):
    """Found opening /s/ links logged out, as the coach they are sent to."""

    @staticmethod
    def _page():
        return (Path(__file__).resolve().parents[1] / "app" / "templates" / "shared.html").read_text(
            encoding="utf-8")

    def test_a_failed_identity_check_hides_the_numbers(self):
        """An identity-failed fight still showed its numbers and priorities,
        and told the coach to "return to fighter selection"."""
        page = self._page()
        self.assertIn("identity_ok=report.integrity.identity_evidence_trusted", page)
        self.assertIn("Identity check failed.", page)
        self.assertIn("{% if identity_ok %}<div class=\"metric-grid\">", page)

    def test_the_movement_numbers_use_the_reports_units(self):
        page = self._page()
        for key, unit in (("footwork_body_lengths_per_second", "footwork"), ("pressure_index", "pressure"),
                          ("ring_center_control", "centre"), ("guard_index", "centre"),
                          ("balance_index", "centre"), ("pose_coverage", "centre")):
            self.assertIn(f"{{{{m.{key}|default(none)|movement_value('{unit}')}}}}", page, key)
        self.assertNotIn("Pose evidence", page)
        self.assertNotIn("evidence validation gate", page)

    def test_the_expiry_is_a_date_and_a_dead_link_says_why(self):
        from app.main import _friendly_date

        self.assertEqual(_friendly_date("2026-10-03T17:02:44.079794+00:00"), "03 Oct 2026")
        self.assertIn("{{expires_label or expires_at}}", self._page())
        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("This coach link has expired or was turned off.", source)

    def test_an_unmeasured_likeness_is_not_reported_as_zero(self):
        """"Their kit matches at 0%" - the opposite of "too alike" - when an
        older report had no similarity measurement."""
        from core.report import refresh_identity_integrity

        unknown = refresh_identity_integrity({"tracking": {"fighters_separable": False},
                                              "video": {"analysis_target": "A"}})
        self.assertNotIn("0%", unknown["scorecard"]["disclaimer"])
        measured = refresh_identity_integrity({"tracking": {"fighters_separable": False,
                                                            "fighter_pair_similarity": .83},
                                               "video": {"analysis_target": "A"}})
        self.assertIn("matches at 83%", measured["scorecard"]["disclaimer"])


class StyleIsNotAFaultTests(unittest.TestCase):
    """"Work on: Walking them down" on the Coach squad, from coaching saved at
    analysis time, after the report page had stopped saying it."""

    def test_the_squad_priority_uses_the_current_coaching(self):
        import tempfile
        from core.squad import build_squad_view

        report = {
            "video": {"focus_fighter": "A", "analysis_target": "A"},
            "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9,
                         "fighter_A_initial_lock_safe": True},
            "integrity": {"identity_evidence_trusted": True, "action_metrics_trusted": False},
            "metrics": {"A": {"pose_coverage": .9, "guard_index": .2, "balance_index": .7,
                              "pressure_index": -.1, "ring_center_control": .6},
                        "B": {"pose_coverage": .9, "guard_index": .2, "balance_index": .7,
                              "pressure_index": .2, "ring_center_control": .6}},
            "coaching": {"A": {"improvements": [{"title": "Work on: Walking them down 45"}]}},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            view = build_squad_view([{"job_id": "x", "report_path": str(path),
                                      "created_at": "2026-09-01", "ruleset": "K1"}])
        priorities = [row.get("priority") for row in view["fights"]]
        self.assertTrue(priorities, view)
        self.assertNotIn("Work on: Walking them down 45", priorities)


class RealFightFindingsTests(unittest.TestCase):
    """Found running a real 94-second Kick Light bout, shared from a phone at
    480x220, through upload, fighter pick, analysis and report."""

    def test_low_coverage_from_a_small_video_says_to_send_the_original(self):
        from app.main import _score_withheld

        advice = ("The framing is fine - one fighter fills 35% of the picture - but the video "
                  "is only 220 pixels tall, so there is not enough detail.")
        report = {"scorecard": {"available": False, "status": "insufficient_observation_coverage"},
                  "tracking": {"fighter_A_coverage": .49, "fighter_B_coverage": .54,
                               "recording": {"measured": True, "advice": [advice]}}}
        self.assertEqual(_score_withheld(report)["fix"], advice)
        # Without a recording problem the old advice still applies.
        del report["tracking"]["recording"]
        self.assertIn("clearly apart", _score_withheld(report)["fix"])


class HandheldCameraIdentityTests(unittest.TestCase):
    """Athens European Cup, filmed handheld from the stands: coverage 82%, the
    identity check passed, and the boxes were on spectators, coaches and the
    opponent for most of the bout."""

    @staticmethod
    def _tracking(rate_a, rate_b=5.0):
        tracking = {"fighter_A_seed_source": "manual_anchor", "fighter_B_seed_source": "manual_anchor",
                    "fighter_A_coverage": .82, "fighter_B_coverage": .81, "fighters_separable": True}
        if rate_a is not None:
            tracking.update(fighter_A_handoffs_per_minute=rate_a, fighter_B_handoffs_per_minute=rate_b)
        return tracking

    def test_the_manager_counts_moves_between_real_tracks_only(self):
        import numpy as np

        from core.identity import IdentityManager
        from core.types import PersonObservation

        def person(track_id, x=100.0):
            return PersonObservation(track_id=track_id, confidence=0.9,
                                     box=np.asarray([x, 100, x + 30, 180], dtype=np.float32))

        manager = IdentityManager(person(1), person(2, 300.0), 0, source_fps=30.0)
        for frame, track in enumerate([5, 5, -1001, 5, 7, 7, -1001, 9]):
            manager._commit(manager.a, person(track), frame, 0.8)
        self.assertEqual(manager.track_handoffs["A"], 2, "5 to 7 and 7 to 9; stand-ins are not tracks")

    def test_a_fighter_found_again_too_often_fails_the_identity_check(self):
        from core.report import refresh_identity_integrity

        report = refresh_identity_integrity({"tracking": self._tracking(21.4),
                                             "video": {"analysis_target": "A"}})
        self.assertFalse(report["integrity"]["identity_evidence_trusted"])
        self.assertIn("find them again 21 times a minute", report["scorecard"]["disclaimer"])
        self.assertIn("steadier recording", report["scorecard"]["disclaimer"])

    def test_a_steady_camera_and_an_older_report_pass_as_before(self):
        from core.report import refresh_identity_integrity

        for tracking in (self._tracking(6.4), self._tracking(None)):
            report = refresh_identity_integrity({"tracking": tracking, "video": {"analysis_target": "A"}})
            self.assertTrue(report["integrity"]["identity_evidence_trusted"], tracking)

    def test_the_page_asks_for_a_steadier_recording_not_a_re_pick(self):
        from app.main import _score_withheld

        report = {"scorecard": {"available": False, "status": "identity_integrity_failed"},
                  "tracking": self._tracking(25.7)}
        withheld = _score_withheld(report)
        self.assertIn("camera moves a lot", withheld["reason"])
        self.assertIn("fixed spot", withheld["fix"])

    def test_the_saved_report_fails_the_check_too(self):
        """The page refused a fight the saved report - and so the Progress
        snapshot - still called trusted: build_report had its own copy of the
        gate without this rule."""
        from core.report import build_report
        from core.types import AnalysisRequest, RoundSpec

        sample = json.loads(FIXTURE.read_text(encoding="utf-8"))
        tracking = dict(sample["tracking"], fighter_A_handoffs_per_minute=21.4,
                        fighter_B_handoffs_per_minute=25.7)
        req = AnalysisRequest(video_path="none.mp4", fighter_a_box=[0, 0, 10, 10],
                              fighter_b_box=[20, 0, 30, 10], ruleset="K1", round_count=1,
                              round_duration_seconds=60.0, break_duration_seconds=0.0)
        report = build_report(req, "clip.mp4", [RoundSpec(1, 0.0, 60.0)], [], [],
                              sample["metrics"], tracking, sample["performance"], sample["classifier"])
        self.assertFalse(report["integrity"]["identity_evidence_trusted"])

    def test_the_rest_of_the_page_stops_saying_pick_the_fighters_again(self):
        """On the real handheld bout the headline asked for a steadier
        recording while three other places said to re-pick the fighters."""
        from app.main import _identity_lost_to_camera

        self.assertTrue(_identity_lost_to_camera({"tracking": self._tracking(25.7)}))
        self.assertFalse(_identity_lost_to_camera({"tracking": self._tracking(6.4)}))
        self.assertFalse(_identity_lost_to_camera({"tracking": self._tracking(None)}))
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(
            encoding="utf-8")
        self.assertIn("{% if identity_failed and not camera_lost %}<a class=\"btn\" href=\"/select/", page)
        for repick in ("Pick the fighters again and they will be yours alone.",
                       "<a class=\"text-link\" href=\"/select/{{job_id}}\">Show me who is who</a>",
                       "Choose the fighters again at the top of this page"):
            before = page[:page.index(repick)]
            self.assertIn("{% if camera_lost %}", before[-700:], repick)

    def test_the_progress_snapshot_keeps_what_the_check_reads(self):
        from core.report import IDENTITY_TRACKING_KEYS

        self.assertIn("fighter_A_handoffs_per_minute", IDENTITY_TRACKING_KEYS)
        self.assertIn("fighter_B_handoffs_per_minute", IDENTITY_TRACKING_KEYS)
