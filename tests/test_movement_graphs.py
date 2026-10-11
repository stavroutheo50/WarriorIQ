"""The report's movement graphs: core.fight_numbers.over_time and core/movement_graphs.py."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from core import movement_graphs
from core.fight_numbers import SERIES_POINTS, movement_numbers, over_time
from core.report import withhold_pose_figures


def _samples(seconds: float, rate: float = 5.0, speed=lambda t: 1.0, guard=lambda t: 0.5):
    return [{"t": i / rate, "x": 100.0 + i, "y": 100.0, "body": 100.0, "guard": guard(i / rate),
             "balance": 0.8, "toward": 0.9, "speed": speed(i / rate), "gap": 1.2}
            for i in range(int(seconds * rate))]


class OverTimeTests(unittest.TestCase):
    def test_five_second_points_with_speed_and_hands_up(self):
        samples = _samples(30, speed=lambda t: 2.0 if t >= 15 else 1.0,
                           guard=lambda t: 0.5 if t < 10 else 0.9)   # up (<= 0.7) for the first 10 s
        weights = [0.2] * len(samples)
        series = over_time(samples, weights)
        self.assertEqual(series["step_seconds"], 5.0)
        self.assertEqual([p["t"] for p in series["points"]], [0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
        self.assertEqual([p["speed"] for p in series["points"]], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.assertEqual([p["hands_up"] for p in series["points"]], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(series["points"][0]["seen"], 5.0)

    def test_an_unseen_stretch_is_a_gap_not_a_zero(self):
        samples = [s for s in _samples(30) if not 10 <= s["t"] < 15]
        series = over_time(samples, [0.2] * len(samples))
        gap = series["points"][2]
        self.assertEqual((gap["seen"], gap["speed"], gap["hands_up"]), (0.0, None, None))

    def test_a_long_fight_never_has_more_than_the_point_limit(self):
        samples = _samples(9 * 60, rate=2.0)
        series = over_time(samples, [0.5] * len(samples))
        self.assertLessEqual(len(series["points"]), SERIES_POINTS)
        self.assertEqual(series["step_seconds"] % 5.0, 0.0)

    def test_withheld_guard_draws_no_hands_up(self):
        samples = _samples(30)
        self.assertTrue(all(p["hands_up"] is None
                            for p in over_time(samples, [0.2] * len(samples), guard_measured=False)["points"]))

    def test_too_short_to_draw(self):
        samples = _samples(6)
        self.assertIsNone(over_time(samples, [0.2] * len(samples)))

    def test_movement_numbers_carries_the_series(self):
        numbers = movement_numbers(_samples(30), None, None)
        self.assertEqual(len(numbers["series"]["points"]), 6)

    def test_sideways_video_withholds_the_hands_up_points_too(self):
        report = {"metrics": {"A": {"numbers": movement_numbers(_samples(30), None, None)}}}
        withhold_pose_figures(report, "note", "short")
        points = report["metrics"]["A"]["numbers"]["series"]["points"]
        self.assertTrue(all(p["hands_up"] is None for p in points))
        self.assertTrue(all(p["speed"] is not None for p in points))   # movement stays


def _report(with_series: bool = True) -> dict:
    a = movement_numbers(_samples(30, speed=lambda t: 1.5), None, None)
    b = movement_numbers(_samples(30, speed=lambda t: 0.8), None, None)
    a.update({"forward_seconds": 12.0, "backward_seconds": 6.0, "circling_seconds": 6.0, "standing_seconds": 6.0,
              "range_seconds": {"close": 3.0, "middle": 21.0, "long": 6.0}, "pace_change_percent": -12,
              "longest_hands_down_seconds": 4.2, "off_balance_count": 1})
    if not with_series:
        a.pop("series")
        b.pop("series")
    return {"metrics": {"A": {"numbers": a}, "B": {"numbers": b}}}


class GraphTests(unittest.TestCase):
    def test_how_you_moved_and_distance_as_shares(self):
        graphs = movement_graphs.build(_report(), "A")
        moved = graphs["moved"]["A"]
        self.assertEqual([p["label"] for p in moved], ["Forward", "Backing up", "Circling", "Standing still"])
        self.assertEqual([p["share"] for p in moved], [40.0, 20.0, 20.0, 20.0])
        self.assertEqual(graphs["ranges"]["A"][1]["share"], 70.0)
        self.assertIsNotNone(graphs["moved"]["B"])   # the opponent beside you

    def test_energy_lines_for_both_and_a_guard_bar_per_point(self):
        graphs = movement_graphs.build(_report(), "A")
        self.assertEqual(set(graphs["energy"]["lines"]), {"A", "B"})
        self.assertEqual(graphs["energy"]["top"], 1.5)
        self.assertEqual(len(graphs["guard"]["bars"]), 6)
        self.assertEqual(graphs["energy"]["axis"][0], "0:00")

    def test_a_gap_breaks_the_line(self):
        report = _report()
        report["metrics"]["A"]["numbers"]["series"]["points"][3]["speed"] = None
        lines = movement_graphs.build(report, "A")["energy"]["lines"]["A"]
        self.assertEqual(len(lines), 2)

    def test_plain_facts(self):
        facts = {f["label"]: f for f in movement_graphs.build(_report(), "A")["facts"]}
        self.assertEqual(facts["pace, second half"]["sentence"],
                         "You moved 12% slower in the second half than the first.")
        self.assertEqual(facts["time off balance"]["value"], "1")
        self.assertEqual(facts["longest with hands down"]["value"], "0:04")

    def test_each_graph_says_what_it_shows_with_its_numbers(self):
        reads = movement_graphs.build(_report(), "A")["reads"]
        self.assertEqual(reads["moved"], "You went forward 40% of the time, backed up 20%, circled 20% and stood "
                                         "still 20%. Your opponent went forward 100% of the time.")
        self.assertEqual(reads["ranges"], "You were close to your opponent 10% of the time, at middle distance "
                                          "70% and far apart 20%.")
        self.assertIn("On average you moved 1.5 body lengths a second, your opponent 0.8.", reads["energy"])
        self.assertTrue(reads["guard"].startswith("Your hands were up 100% of the time."))

    def test_an_older_report_still_gets_the_bars_without_the_lines(self):
        graphs = movement_graphs.build(_report(with_series=False), "A")
        self.assertIsNone(graphs["energy"])
        self.assertIsNone(graphs["guard"])
        self.assertIsNotNone(graphs["moved"]["A"])

    def test_nothing_measured_nothing_drawn(self):
        self.assertIsNone(movement_graphs.build({"metrics": {"A": {}}}, "A"))
        self.assertIsNone(movement_graphs.build({}, "A"))


class PageTests(unittest.TestCase):
    def _render(self, trusted: bool):
        from tests_support import render_result

        report = json.loads((Path(__file__).resolve().parent / "fixtures" / "report_sample.json")
                            .read_text(encoding="utf-8"))
        report["integrity"]["identity_evidence_trusted"] = trusted
        built = _report()
        for side in ("A", "B"):
            report["metrics"][side]["numbers"] = copy.deepcopy(built["metrics"][side]["numbers"])
        return render_result(report=report, numbers={"state": "ok" if trusted else "unverified"},
                             graphs=movement_graphs.build(report, "A") if trusted else None)

    def test_graphs_follow_the_six_numbers(self):
        page = self._render(True)
        self.assertIn('id="report-graphs"', page)
        self.assertLess(page.index('class="fight-vitals"'), page.index('id="report-graphs"'))
        self.assertIn('class="graph-line mine"', page)
        self.assertIn("You moved 12% slower in the second half", page)
        self.assertIn('class="graph-read"', page)
        self.assertIn("<span>40%</span>", page)   # the share printed inside its bar

    def test_numbers_count_up_and_graphs_draw_in_but_never_against_reduced_motion(self):
        import re

        page = self._render(True)
        vitals = page[page.index('class="fight-vitals"'):page.index('id="report-graphs"')]
        self.assertEqual(vitals.count("data-count-up"), 6)
        self.assertIn('pathLength="1"', page)
        self.assertEqual(page.count('class="stack-part'), len(re.findall(r'class="stack-part[^"]*" style="--w:', page)),
                         "one style attribute per bar part, carrying its width")
        css = (Path(__file__).resolve().parents[1] / "app" / "static" / "components.css").read_text(encoding="utf-8")
        motion = css[css.index("The graphs build themselves"):]
        self.assertIn("@media (prefers-reduced-motion:no-preference)", motion)
        # Hidden only while motion.js has marked it and it is not yet in view.
        self.assertIn(".fight-graphs.wiq-reveal:not(.is-visible) .stack-part{width:0}", motion)

    def test_no_graphs_when_the_fighters_could_not_be_followed(self):
        self.assertNotIn('id="report-graphs"', self._render(False))


if __name__ == "__main__":
    unittest.main()


class CompareTests(unittest.TestCase):
    def _two(self):
        first, second = _report(), _report()
        # A busier second fight: every speed doubled.
        for point in second["metrics"]["A"]["numbers"]["series"]["points"]:
            point["speed"] = point["speed"] * 2
        return [first, second]

    def test_fight_1_against_fight_2_on_one_scale(self):
        graphs = movement_graphs.compare(self._two(), ["A", "A"], [False, False])
        self.assertEqual(graphs["labels"], ["Fight 1", "Fight 2"])
        self.assertEqual([row["label"] for row in graphs["moved"]], ["Fight 1", "Fight 2"])
        self.assertEqual(graphs["energy_top"], 3.0)
        # Same scale: the first fight's line sits at half height (y = 55 of 110).
        first_y = {float(pair.split(",")[1]) for pair in graphs["energy"][0]["lines"][0].split()}
        self.assertEqual(first_y, {55.0})
        self.assertEqual(len(graphs["guard"]), 2)
        pace = next(row for row in graphs["facts"] if row["label"] == "pace, second half")
        self.assertEqual(pace["values"], ["-12%", "-12%"])

    def test_a_fight_that_could_not_be_followed_is_left_out_with_a_reason(self):
        graphs = movement_graphs.compare(self._two(), ["A", "A"], [False, True])
        self.assertEqual([row["label"] for row in graphs["moved"]], ["Fight 1"])
        self.assertEqual(graphs["notes"], ["Fight 2's fighters could not be followed, so it has no graphs."])
        self.assertEqual(graphs["facts"][0]["values"][1], "—")

    def test_nothing_to_compare(self):
        self.assertIsNone(movement_graphs.compare(self._two(), ["A", "A"], [True, True]))

    def test_the_compare_page_shows_them_after_the_movement_table(self):
        from tests_support import render

        reports = self._two()
        page = render("compare.html", signed_in=True, fights=[{"job_id": "a"}, {"job_id": "b"}],
                      a="a", b="b", reports=reports, picked=None, solo_picked=False,
                      movement={"available": True, "rows": [], "identity_failed": [False, False],
                                "untrusted": [False, False], "coverage": [1, 1]},
                      compare_graphs=movement_graphs.compare(reports, ["A", "A"], [False, False]))
        self.assertIn('id="compare-graphs"', page)
        self.assertLess(page.index('id="movement-comparison"'), page.index('id="compare-graphs"'))
        self.assertIn("/static/components.css", page)
        self.assertIn("Energy through each fight", page)
