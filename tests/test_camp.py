"""Fight Camp missions (core/camp.py) and the page that shows them."""

import unittest
from datetime import date

from core.camp import (camp_standing, fight_camp_missions, improved_by, mission_result,
                       missions_from_report)


def report(trusted=True, drills=True):
    why = "You 10%, them 16% - behind your opponent here."
    return {
        "integrity": {"identity_evidence_trusted": trusted},
        "coaching": {"A": {
            "improvements": [{"title": "Work on: Guard 10%", "detail": why, "evidence_times": [4.07, 7.13]}],
            "drills": [{"name": "Fighter A · Guard-return audit", "why": why, "metric": "guard_index",
                        "label": "Guard", "measured": 0.10,
                        "prescription": "4 x 90 sec: freeze in stance after every exchange."}] if drills else [],
        }},
        "training_plan": {"A": [{"session_block": 1, "focus": "Block 1: Fighter A · Guard-return audit",
                                 "goal": "Raise guard from 10.2% to 18.2%."}] if drills else []},
    }


class MissionTests(unittest.TestCase):
    def test_a_drill_becomes_a_mission_with_its_target_and_moment(self):
        found = missions_from_report(report(), "A")
        self.assertIsNone(found["reason"])
        mission = found["missions"][0]
        self.assertEqual(mission["title"], "Guard-return audit")
        self.assertEqual(mission["exercise"], "4 x 90 sec: freeze in stance after every exchange.")
        self.assertEqual(mission["target"], "Raise guard from 10.2% to 18.2%.")
        self.assertEqual(mission["evidence_seconds"], 4.07)

    def test_no_drill_no_mission_and_it_says_why(self):
        found = missions_from_report(report(drills=False), "A")
        self.assertEqual(found, {"missions": [], "reason": "no_drill"})

    def test_the_other_fighters_coaching_is_not_used(self):
        self.assertEqual(missions_from_report(report(), "B")["reason"], "no_drill")

    def test_a_fight_whose_identity_failed_gives_no_missions(self):
        self.assertEqual(missions_from_report(report(trusted=False), "A")["reason"], "identity")

    def test_the_newest_usable_fight_is_used_and_taken_missions_are_marked(self):
        records = [
            {"job_id": "new", "report": report(trusted=False), "fighter": "A", "created_at": "2026-09-20"},
            {"job_id": "old", "report": report(), "fighter": "A", "created_at": "2026-09-01"},
        ]
        camp = fight_camp_missions(records, [{"title": "guard-return audit", "status": "active"}])
        self.assertEqual(camp["job_id"], "old")
        self.assertTrue(camp["skipped_newer_fight"])
        self.assertEqual(camp["missions"][0]["status"], "active")

    def test_no_fights(self):
        camp = fight_camp_missions([], [])
        self.assertEqual((camp["missions"], camp["reason"]), ([], "no_fight"))


class MissionResultTests(unittest.TestCase):
    MISSION = {"metric": "guard_index", "measured": 0.10}

    def later(self, value, trusted=True, fighter="A"):
        return {"job_id": "next", "fighter": fighter, "report": {
            "integrity": {"identity_evidence_trusted": trusted},
            "metrics": {fighter: {"guard_index": value}}}}

    def test_a_real_improvement_counts(self):
        result = mission_result(self.MISSION, [self.later(0.14)])
        self.assertTrue(result["improved"])
        self.assertEqual((result["before"], result["after"]), (0.10, 0.14))

    def test_noise_is_not_an_improvement(self):
        self.assertFalse(mission_result(self.MISSION, [self.later(0.12)])["improved"])

    def test_a_fight_it_could_not_attribute_is_skipped(self):
        result = mission_result(self.MISSION, [self.later(0.50, trusted=False), self.later(0.11)])
        self.assertFalse(result["improved"])

    def test_no_later_fight_means_no_result_yet(self):
        self.assertIsNone(mission_result(self.MISSION, []))

    def test_the_margin_follows_the_units(self):
        self.assertEqual(improved_by("guard_index"), 0.03)
        self.assertEqual(improved_by("pressure_index"), 0.06)


class StandingTests(unittest.TestCase):
    TODAY = date(2026, 9, 30)      # a Wednesday

    def session(self, day, verdict="counted"):
        return {"created_at": day + "T10:00:00+00:00", "verdict": verdict}

    def test_points_and_level(self):
        standing = camp_standing([{"points": 100}, {"points": 25}, {"points": 10}], [], self.TODAY)
        self.assertEqual((standing["points"], standing["level"], standing["into_level"]), (135, 2, 35))

    def test_streak_counts_weeks_in_a_row(self):
        sessions = [self.session("2026-09-29"), self.session("2026-09-22"), self.session("2026-09-15"),
                    self.session("2026-09-01")]
        self.assertEqual(camp_standing([], sessions, self.TODAY)["streak_weeks"], 3)

    def test_a_week_not_over_yet_does_not_break_the_streak(self):
        sessions = [self.session("2026-09-22"), self.session("2026-09-15")]
        self.assertEqual(camp_standing([], sessions, self.TODAY)["streak_weeks"], 2)

    def test_sessions_that_did_not_count_do_not_make_a_streak(self):
        sessions = [self.session("2026-09-29", "no_movement"), self.session("2026-09-22", "duplicate")]
        self.assertEqual(camp_standing([], sessions, self.TODAY)["streak_weeks"], 0)


if __name__ == "__main__":
    unittest.main()
