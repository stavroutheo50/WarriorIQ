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
