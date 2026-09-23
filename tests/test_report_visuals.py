"""The parts of the report a fighter reads without reading.

The analysis was already computing defence counts, combination lengths, which
targets a fighter was caught on, and the seconds their guard dropped. The
report showed none of it and spent ten thousand characters on prose instead.

These cover the reshaping, and - more importantly - that it cannot leak the
punch count the product does not trust.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from core.report import ARRIVED_OUTCOMES
from core.report_visuals import ZONES, build, head_to_head

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _event(fighter="A", technique="left_low_kick", outcome="clean", target="leg", at=1.0):
    return {"fighter": fighter, "technique": technique, "outcome": outcome,
            "target": target, "peak_time": at}


def _report(**overrides):
    base = {
        "metrics": {
            "A": {"guard_index": 0.11, "balance_index": 0.54, "ring_center_control": 0.56,
                  "footwork_body_lengths_per_second": 0.91,
                  "defenses": {"slip": 6, "block": 5}, "combinations": {"count": 2, "max_length": 3,
                  "evidence": [[1.0, 1.5, 2.0], [4.0, 4.5]]},
                  "moments": {"guard_index": {"low": [5.0, 8.27]}}},
            "B": {"guard_index": 0.13, "balance_index": 0.60, "ring_center_control": 0.46,
                  "footwork_body_lengths_per_second": 0.83},
        },
        "events": [], "rounds": [{"end_seconds": 30.0}],
    }
    base.update(overrides)
    return base


class PunchGateTests(unittest.TestCase):
    """The reason this module is allowed to exist at all.

    Punch counting is switched off - checked against video, the count was
    overstated by eleven in two bouts of three. A body map fed from the raw
    event stream would republish exactly that number in a prettier shape.
    """

    def test_a_punch_never_reaches_a_body_map(self):
        report = _report(events=[
            _event(technique="jab", target="head"),
            _event(technique="cross", target="head"),
            _event(technique="left_low_kick", target="leg"),
        ])
        visuals = build(report, "A")
        self.assertEqual(visuals["landed"]["head"], 0,
                         "a punch was counted onto the body map")
        self.assertEqual(visuals["landed"]["leg"], 1)

    def test_a_punch_never_reaches_the_timeline(self):
        report = _report(events=[
            _event(technique="jab", outcome="clean", at=2.0),
            _event(technique="right_low_kick", outcome="clean", at=4.0),
        ])
        self.assertEqual(len(build(report, "A")["timeline"]), 1)

    def test_a_punch_never_reaches_the_strike_count(self):
        report = _report(events=[_event(technique="jab") for _ in range(9)])
        row = [r for r in build(report, "A")["head_to_head"] if r["key"] == "landed"][0]
        self.assertEqual(row["a"], 0)

    def test_knees_count_with_kicks(self):
        """The report already treats them as one family: by eye a knee and a
        round kick are a coin flip."""
        report = _report(events=[_event(technique="left_knee", target="body")])
        self.assertEqual(build(report, "A")["landed"]["body"], 1)

    def test_only_strikes_that_arrived_are_counted(self):
        report = _report(events=[
            _event(outcome="clean"), _event(outcome="missed"),
            _event(outcome="uncertain"), _event(outcome="blocked"),
        ])
        # clean and blocked arrived; missed and uncertain did not.
        self.assertEqual(build(report, "A")["landed"]["leg"], 2)
        self.assertTrue({"clean", "blocked"} <= set(ARRIVED_OUTCOMES))

    def test_the_page_says_what_it_counted(self):
        self.assertIn("Hands are not counted", build(_report(), "A")["counts"])


class HeadToHeadTests(unittest.TestCase):
    """The opponent is the benchmark, which beats any invented target band."""

    def test_both_fighters_are_measured_on_the_same_row(self):
        rows = {r["key"]: r for r in build(_report(), "A")["head_to_head"]}
        self.assertEqual(rows["guard"]["a"], 11)
        self.assertEqual(rows["guard"]["b"], 13)
        self.assertEqual(rows["guard"]["leader"], "b")

    def test_a_percentage_is_read_against_a_hundred_not_against_the_pair(self):
        """Guard at 11% against 13% is two small numbers, and the bars must
        say so. Scaling them to each other would draw 11% as a full bar."""
        rows = {r["key"]: r for r in build(_report(), "A")["head_to_head"]}
        self.assertLess(rows["guard"]["a_width"], 20)
        self.assertLess(rows["guard"]["b_width"], 20)

    def test_a_count_is_read_against_the_pair(self):
        """A strike count has no natural ceiling, so the bigger of the two
        sets the scale."""
        report = _report(events=[_event() for _ in range(9)])
        rows = {r["key"]: r for r in build(report, "A")["head_to_head"]}
        self.assertGreater(rows["landed"]["a_width"], 70)

    def test_a_measurement_missing_for_either_fighter_is_dropped(self):
        """One bar against an empty space reads as a win."""
        report = _report()
        report["metrics"]["B"].pop("guard_index")
        keys = {r["key"] for r in build(report, "A")["head_to_head"]}
        self.assertNotIn("guard", keys)
        self.assertIn("balance", keys)

    def test_a_zero_still_draws_something(self):
        rows = head_to_head({"A": {}, "B": {}}, "A", "B", {"A": 0, "B": 5})
        landed = [r for r in rows if r["key"] == "landed"][0]
        self.assertGreater(landed["a_width"], 0, "zero should be a visible mark")


class ShapeTests(unittest.TestCase):
    def test_the_two_maps_are_opposites(self):
        report = _report(events=[
            _event(fighter="A", target="head"), _event(fighter="B", target="leg"),
        ])
        visuals = build(report, "A")
        self.assertEqual(visuals["landed"]["head"], 1)
        self.assertEqual(visuals["taken"]["leg"], 1)
        self.assertEqual(visuals["taken"]["head"], 0)

    def test_every_zone_is_present_even_at_zero(self):
        """A missing zone would collapse the silhouette's layout."""
        visuals = build(_report(), "A")
        for zone in ZONES:
            self.assertIn(zone, visuals["landed"])
            self.assertIn(zone, visuals["taken"])

    def test_chains_are_longest_first(self):
        visuals = build(_report(), "A")
        self.assertEqual(visuals["chains"], sorted(visuals["chains"], reverse=True))

    def test_defences_are_ordered_and_totalled(self):
        visuals = build(_report(), "A")
        self.assertEqual(list(visuals["defences"]), ["slip", "block"])
        self.assertEqual(visuals["defence_total"], 11)
        self.assertEqual(visuals["defence_max"], 6)

    def test_a_report_with_no_metrics_returns_nothing(self):
        """The section must not render half-built."""
        self.assertIsNone(build({"metrics": {}}, "A"))
        self.assertIsNone(build({}, "A"))

    def test_an_empty_fight_does_not_divide_by_zero(self):
        report = _report(rounds=[], events=[])
        report["metrics"]["A"]["defenses"] = {}
        report["metrics"]["A"]["combinations"] = {}
        visuals = build(report, "A")
        self.assertEqual(visuals["span_seconds"], 0.0)
        self.assertEqual(visuals["defence_max"], 0)
        self.assertEqual(visuals["chains"], [])


class RealReportTests(unittest.TestCase):
    def setUp(self):
        stored = PROJECT_ROOT / "outputs" / "athens_hd" / "report.json"
        if not stored.exists():
            self.skipTest("no stored analysis in this checkout")
        self.visuals = build(json.loads(stored.read_text(encoding="utf-8")), "A")

    def test_it_reshapes_a_real_report(self):
        self.assertEqual(self.visuals["landed"], {"head": 1, "body": 4, "leg": 4})
        self.assertEqual(self.visuals["taken"], {"head": 0, "body": 2, "leg": 1})
        self.assertEqual(self.visuals["defence_total"], 22)
        self.assertEqual(self.visuals["chain_longest"], 8)
        self.assertEqual(len(self.visuals["head_to_head"]), 5)

    def test_it_works_on_an_analysis_that_predates_it(self):
        """Built at render time, so nothing has to be re-run to gain it."""
        self.assertTrue(self.visuals["timeline"])
        self.assertTrue(self.visuals["guard_low"])


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.page = (PROJECT_ROOT / "app" / "templates" / "result.html").read_text(encoding="utf-8")

    def test_the_section_is_not_buried_in_the_deep_dive_accordion(self):
        """It was, on the first attempt: the headline visuals rendered inside
        a collapsed <details> and could not be scrolled to."""
        self.assertIn('id="report-visual"', self.page)
        self.assertLess(self.page.index('id="report-visual"'),
                        self.page.index('<details class="report-deep-dive">'),
                        "the visual report is inside the collapsed accordion")

    def test_the_corner_colours_match_the_words(self):
        """The page says "Red corner" over Fighter A and rendered A blue."""
        css = (PROJECT_ROOT / "app" / "static" / "components.css").read_text(encoding="utf-8")
        self.assertIn("--mine:#d9545e", css)
        self.assertIn("--theirs:#5b8ff0", css)

    def test_identity_survives_forced_colours(self):
        """A high-contrast palette drops the fills, so the hue alone would
        stop telling the two fighters apart."""
        css = (PROJECT_ROOT / "app" / "static" / "components.css").read_text(encoding="utf-8")
        self.assertIn("forced-colors: active", css)


if __name__ == "__main__":
    unittest.main()
