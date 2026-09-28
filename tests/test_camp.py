"""Fight Camp missions (core/camp.py) and the page that shows them."""

import unittest

from core.camp import fight_camp_missions, missions_from_report


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


if __name__ == "__main__":
    unittest.main()
