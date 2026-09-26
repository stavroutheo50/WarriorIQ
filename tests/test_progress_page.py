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
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "coach.html").read_text(
            encoding="utf-8")
        self.assertIn("identity check failed", page)
        self.assertIn("squad.unusable_identity", page)

    def test_the_coach_page_uses_the_current_name_for_centre(self):
        from core.metric_catalog import RETIRED_NAMES

        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "coach.html").read_text(
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
