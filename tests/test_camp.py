"""Fight Camp missions (core/camp.py) and the page that shows them."""

import unittest
from datetime import date

from core.camp import (DAILY_COUNTED_SESSIONS, WEEKLY_TARGET, camp_standing, fight_camp_missions, improved_by,
                       mission_board, mission_progress, mission_result, missions_from_report, next_step,
                       paid_sessions_on, points_history)


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

    def test_spending_lowers_the_balance_not_the_level(self):
        standing = camp_standing([{"points": 250}, {"points": -200}], [], self.TODAY)
        self.assertEqual((standing["level"], standing["balance"]), (3, 50))

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

    def test_a_rank_for_each_stretch_of_levels(self):
        def rank(points):
            standing = camp_standing([{"points": points}], [], self.TODAY)
            return standing["rank"], standing["next_rank"], standing["next_rank_level"]
        self.assertEqual(rank(0), ("Rookie", "Prospect", 3))
        self.assertEqual(rank(250), ("Prospect", "Contender", 5))
        self.assertEqual(rank(1100), ("Champion", None, None))

    def test_this_weeks_counted_sessions_against_the_target(self):
        sessions = [self.session(self.TODAY.isoformat()), self.session(self.TODAY.isoformat(), "no_movement"),
                    self.session("2026-09-01")]
        standing = camp_standing([], sessions, self.TODAY)
        self.assertEqual((standing["week_sessions"], standing["week_target"]), (1, WEEKLY_TARGET))

    def test_points_history_newest_first_and_named(self):
        points = [{"id": 1, "points": 10, "reason": "session", "created_at": "2026-09-28T09:00:00+00:00"},
                  {"id": 2, "points": 100, "reason": "improved", "created_at": "2026-09-28T10:00:00+00:00"},
                  {"id": 3, "points": -200, "reason": "redeem_analysis", "created_at": "2026-09-28T11:00:00+00:00"}]
        history = points_history(points, limit=2)
        self.assertEqual([(item["points"], item["label"]) for item in history],
                         [(-200, "Extra analysis"), (100, "Your fight showed the improvement")])



class MissionProgressTests(unittest.TestCase):
    def test_the_bar_runs_from_the_measured_number_to_the_plans_goal(self):
        bar = mission_progress("guard_index", 0.10)
        # The training plan's sentence for the same number says 10% to 18%.
        self.assertEqual((bar["label"], bar["before"], bar["goal"]), ("Guard", "10.0%", "18.0%"))
        self.assertEqual(bar["unlock"], "13.0%")
        self.assertEqual(bar["tick"], 38)          # 3 of the 8 points to the goal
        self.assertEqual((bar["fill"], bar["after"], bar["improved"]), (0, None, None),
                         "nothing is filled before a fight has measured it")

    def test_a_later_fight_fills_it_with_what_it_measured(self):
        bar = mission_progress("guard_index", 0.10, {"after": 0.14, "improved": True})
        self.assertEqual((bar["after"], bar["fill"], bar["improved"]), ("14.0%", 50, True))
        worse = mission_progress("guard_index", 0.10, {"after": 0.07, "improved": False})
        self.assertEqual((worse["after"], worse["fill"], worse["improved"]), ("7.0%", 0, False))
        past = mission_progress("guard_index", 0.10, {"after": 0.40, "improved": True})
        self.assertEqual(past["fill"], 100)

    def test_the_units_follow_the_number(self):
        pressure = mission_progress("pressure_index", -0.20)
        self.assertEqual((pressure["before"], pressure["goal"]), ("40 out of 100", "46 out of 100"))
        feet = mission_progress("footwork_body_lengths_per_second", 0.8)
        self.assertEqual((feet["before"], feet["unlock"]), ("0.8 body lengths a second", "0.9 body lengths a second"))

    def test_nothing_to_draw_without_a_measured_number(self):
        self.assertIsNone(mission_progress(None, None))
        self.assertIsNone(mission_progress("guard_index", None))
        self.assertIsNone(mission_progress("guard_index", 0.97), "no room left to a goal")


class BoardTests(unittest.TestCase):
    def camp(self, status=None):
        mission = missions_from_report(report(), "A")["missions"][0]
        mission["status"] = status
        return {"job_id": "f1", "missions": [mission], "reason": None}

    @staticmethod
    def item(item_id, title, status="active"):
        return {"id": item_id, "title": title, "detail": "", "status": status}

    def test_a_mission_not_taken_yet_is_new_with_its_bar(self):
        board = mission_board(self.camp(), [], {}, {}, {})
        self.assertEqual([card["stage"] for card in board], ["new"])
        self.assertEqual(board[0]["index"], 0)
        self.assertEqual(board[0]["progress"]["before"], "10.0%")

    def test_each_mission_is_one_card_at_its_stage(self):
        items = [self.item(1, "Guard-return audit"), self.item(2, "Own drill", "complete"),
                 self.item(3, "Waited on", "complete"), self.item(4, "Judged", "complete")]
        linked = {1: {"metric": "guard_index", "measured": 0.10}, 3: {"metric": "guard_index", "measured": 0.2},
                  4: {"metric": "guard_index", "measured": 0.2}}
        results = {4: {"after": 0.3, "improved": True}}
        board = mission_board(self.camp("active"), items, linked, results, {1: 2})
        stages = {card["title"]: card["stage"] for card in board}
        self.assertEqual(stages, {"Guard-return audit": "training", "Own drill": "done",
                                  "Waited on": "waiting", "Judged": "improved"})
        self.assertEqual(len(board), 4, "the taken mission is not shown twice")
        training = board[0]
        self.assertEqual((training["title"], training["sessions"]), ("Guard-return audit", 2))
        self.assertEqual(training["evidence_seconds"], 4.07, "the latest fight's moment for it")
        self.assertEqual([card["finished"] for card in board], [False, False, True, True])

    def test_a_judged_mission_still_on_the_list_stays_with_the_work(self):
        board = mission_board(self.camp("active"), [self.item(1, "Guard-return audit")],
                              {1: {"metric": "guard_index", "measured": 0.10}},
                              {1: {"after": 0.11, "improved": False}}, {})
        self.assertEqual((board[0]["stage"], board[0]["finished"]), ("not_yet", False))


class NextStepTests(unittest.TestCase):
    def board(self, *cards):
        return list(cards)

    @staticmethod
    def card(stage, sessions=0, status=None, title="x"):
        assignment = None if status is None else {"id": 1, "status": status}
        return {"stage": stage, "assignment": assignment, "sessions": sessions, "title": title}

    def test_train_the_item_trained_least(self):
        busy, quiet = self.card("training", 3, "active", "busy"), self.card("training", 1, "active", "quiet")
        step = next_step({}, [busy, quiet], 0, True)
        self.assertEqual((step["kind"], step["card"]["title"]), ("upload", "quiet"))

    def test_rest_once_todays_points_are_in(self):
        step = next_step({}, [self.card("training", 1, "active")], DAILY_COUNTED_SESSIONS, True)
        self.assertEqual(step["kind"], "rested")

    def test_then_take_a_mission_then_the_fight_that_judges_the_training(self):
        self.assertEqual(next_step({}, [self.card("new"), self.card("waiting", 2, "complete")], 0, True)["kind"], "take")
        self.assertEqual(next_step({}, [self.card("waiting", 2, "complete")], 0, True)["kind"], "next_fight")

    def test_otherwise_analyse_a_fight(self):
        self.assertEqual(next_step({"reason": "no_fight"}, [], 0, False)["kind"], "first_fight")
        step = next_step({"reason": "identity"}, [self.card("done", status="complete")], 0, True)
        self.assertEqual((step["kind"], step["reason"]), ("analyse", "identity"))

    def test_only_sessions_that_earned_points_today_count_against_the_cap(self):
        sessions = [{"points": 10, "created_at": "2026-09-28T09:00:00+00:00"},
                    {"points": 0, "created_at": "2026-09-28T10:00:00+00:00"},
                    {"points": 10, "created_at": "2026-09-27T23:00:00+00:00"}]
        self.assertEqual(paid_sessions_on(sessions, date(2026, 9, 28)), 1)


if __name__ == "__main__":
    unittest.main()
