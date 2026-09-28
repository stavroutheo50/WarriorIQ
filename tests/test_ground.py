"""Moments someone went down (core/ground.py)."""

import unittest
from types import SimpleNamespace

import numpy as np

from core import ground
from core.types import RoundSpec

CONF = np.full(17, 0.9)


def skeleton(shoulders, hips, ankles):
    """17 COCO joints; only shoulders (5, 6), hips (11, 12) and ankles (15, 16) matter here."""
    points = np.zeros((17, 2))
    for index, point in zip((5, 6, 11, 12, 15, 16), (*shoulders, *hips, *ankles)):
        points[index] = point
    return points


def standing(x, feet=200.0):
    return skeleton([(x - 10, feet - 150), (x + 10, feet - 150)],
                    [(x - 8, feet - 90), (x + 8, feet - 90)], [(x - 8, feet), (x + 8, feet)])


def lying(x, floor=200.0):
    return skeleton([(x - 60, floor - 20), (x - 60, floor - 10)],
                    [(x, floor - 15), (x, floor - 5)], [(x + 60, floor - 10), (x + 60, floor)])


def sitting(x, floor=200.0):
    return skeleton([(x - 10, floor - 80), (x + 10, floor - 80)],
                    [(x - 8, floor - 10), (x + 8, floor - 10)], [(x - 30, floor), (x + 30, floor)])


def person(box, keypoints, confidence=0.9):
    return SimpleNamespace(box=np.array(box, dtype=float), keypoints=keypoints,
                           keypoint_conf=CONF, confidence=confidence)


def fighter(box):
    return SimpleNamespace(box=box)


class PoseDownTests(unittest.TestCase):
    def test_upright_lying_and_sitting(self):
        self.assertIsNone(ground.pose_down(standing(100), CONF))
        self.assertEqual(ground.pose_down(lying(100), CONF), "lying")
        self.assertEqual(ground.pose_down(sitting(100), CONF), "sitting")

    def test_joints_it_cannot_see_are_not_a_fall(self):
        unseen = CONF.copy()
        unseen[[11, 12]] = 0.1
        self.assertIsNone(ground.pose_down(lying(100), unseen))
        self.assertIsNone(ground.pose_down(None, None))


class DownWatchTests(unittest.TestCase):
    A_BOX = [80, 40, 120, 200]
    B_BOX = [180, 40, 220, 200]

    def feed(self, watch, start, seconds, people_at, fps=4, tracked=True):
        for i in range(int(seconds * fps)):
            t = start + i / fps
            a = fighter(self.A_BOX) if tracked else None
            b = fighter(self.B_BOX) if tracked else None
            watch.observe(t, people_at(t), a, b)

    def both_standing(self, t):
        return [person(self.A_BOX, standing(100)), person(self.B_BOX, standing(200))]

    def a_lying(self, t):
        return [person([40, 170, 160, 200], lying(100)), person(self.B_BOX, standing(200))]

    def test_a_takedown_after_standing_is_one_moment(self):
        watch = ground.DownWatch()
        self.feed(watch, 0, 5, self.both_standing)
        self.feed(watch, 5, 6, self.a_lying, tracked=False)
        moments = watch.moments()
        self.assertEqual(len(moments), 1)
        self.assertAlmostEqual(moments[0]["seconds"], 5.0)
        self.assertEqual(moments[0]["down_seconds"], 6)

    def test_nothing_is_listed_while_both_stand(self):
        watch = ground.DownWatch()
        self.feed(watch, 0, 30, self.both_standing)
        self.assertEqual(watch.moments(), [])

    def test_one_collapsed_frame_is_not_a_moment(self):
        watch = ground.DownWatch()
        self.feed(watch, 0, 10, lambda t: self.a_lying(t) if t == 4.0 else self.both_standing(t))
        self.assertEqual(watch.moments(), [])

    def test_a_short_gap_does_not_split_one_ground_spell(self):
        watch = ground.DownWatch()
        self.feed(watch, 0, 3, self.a_lying)
        self.feed(watch, 3, 4, self.both_standing)      # missed seconds, not a stand-up
        self.feed(watch, 7, 3, self.a_lying)
        self.assertEqual(len(watch.moments()), 1)

    def test_a_spectator_sitting_behind_the_fighters_is_not_a_fall(self):
        """The false moment on a real bout: someone cross-legged at the mat
        edge, overlapping a standing fighter's box in the picture."""
        watch = ground.DownWatch()
        spectator = person([175, 100, 225, 165], sitting(200, floor=165))
        self.feed(watch, 0, 10, lambda t: self.both_standing(t) + [spectator])
        self.assertEqual(watch.moments(), [])

    def test_a_fighter_sitting_where_they_are_tracked_is_down(self):
        watch = ground.DownWatch()
        down_a = [60, 130, 140, 200]
        for i in range(20):
            watch.observe(i / 4, [person(down_a, sitting(100)), person(self.B_BOX, standing(200))],
                          fighter(down_a), fighter(self.B_BOX))
        self.assertEqual(len(watch.moments()), 1)

    def test_someone_lying_far_from_both_fighters_is_ignored(self):
        """Another bout on the next mat."""
        watch = ground.DownWatch()
        far = person([600, 170, 720, 200], lying(660))
        self.feed(watch, 0, 10, lambda t: self.both_standing(t) + [far])
        self.assertEqual(watch.moments(), [])

    def test_moments_outside_the_selected_rounds_are_dropped(self):
        watch = ground.DownWatch()
        self.feed(watch, 0, 5, self.a_lying)
        self.feed(watch, 20, 5, self.a_lying)
        rounds = [RoundSpec(1, 0.0, 10.0, selected=False), RoundSpec(2, 15.0, 40.0)]
        moments = watch.moments(rounds)
        self.assertEqual([m["round"] for m in moments], [2])
        self.assertAlmostEqual(moments[0]["seconds"], 20.0)

    def test_the_summary_says_it_is_not_attributed(self):
        summary = ground.DownWatch().summary()
        self.assertFalse(summary["attributed"])
        self.assertEqual(summary["moments"], [])
        self.assertTrue(summary["note"])



class DownCheckExportTests(unittest.TestCase):
    def test_the_summary_counts_false_alarms_and_who_went_down(self):
        import json
        import tempfile
        from pathlib import Path

        from tools.export_down_checks import summary
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, answers in (("downs_one", ["A", "nobody", None]), ("downs_two", ["B", "A"])):
                (root / name).mkdir()
                (root / name / "labels.json").write_text(json.dumps({"moments": [
                    {"seconds": float(i), "answer": answer} for i, answer in enumerate(answers)]}))
            result = summary(root)
        self.assertEqual((result["fights"], result["answered"], result["nobody"]), (2, 4, 1))
        self.assertEqual((result["fighter_A"], result["fighter_B"], result["flags_right"]), (2, 1, 0.75))


class GrapplingTests(unittest.TestCase):
    """Measured down time: Kick Light 2 s (two knockdowns), taekwondo none,
    pankration bouts 2+3+7+27 and 23 seconds."""

    def moments(self, *seconds):
        return {"moments": [{"seconds": float(i), "down_seconds": s} for i, s in enumerate(seconds)]}

    def test_striking_bouts_measured_on_real_footage_stay_quiet(self):
        self.assertIsNone(ground.looks_like_grappling(self.moments(2)))
        self.assertIsNone(ground.looks_like_grappling(self.moments()))
        self.assertIsNone(ground.looks_like_grappling(None))

    def test_the_mma_style_bouts_are_recognised(self):
        self.assertEqual(ground.looks_like_grappling(self.moments(2, 3, 7, 27)),
                         {"ground_seconds": 39, "longest_seconds": 27})
        self.assertIsNotNone(ground.looks_like_grappling(self.moments(23)))

    def test_several_short_knockdowns_are_not_a_ground_fight(self):
        self.assertIsNone(ground.looks_like_grappling(self.moments(5, 5, 5, 5)))

    def test_one_fighter_slow_to_get_up_is_not_either(self):
        self.assertIsNone(ground.looks_like_grappling(self.moments(12)))


if __name__ == "__main__":
    unittest.main()
