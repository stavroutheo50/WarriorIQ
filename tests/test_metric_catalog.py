"""One name per measurement, and a reference beside every number.

The report showed six distinct numbers under up to four names each, and most
of them as bare percentages. "Guard 14%" told a fighter nothing: 14% of what,
and is that good?
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from core import metric_catalog
from core.metric_catalog import BY_KEY, CATALOG, RETIRED_NAMES, readings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_HTML = PROJECT_ROOT / "app" / "templates" / "result.html"


class OneNamePerMeasurementTests(unittest.TestCase):
    def test_no_two_measurements_share_a_name(self):
        names = [metric.name for metric in CATALOG]
        self.assertEqual(len(names), len(set(names)))

    def test_no_retired_name_is_still_on_the_page(self):
        """Each of these was on the report at the same time as the canonical
        name, for the same number."""
        page = RESULT_HTML.read_text(encoding="utf-8")
        for key, retired in RETIRED_NAMES.items():
            for name in retired:
                with self.subTest(metric=key, retired=name):
                    self.assertNotIn(
                        f"<span>{name}</span>", page,
                        f"{name} is a second name for {BY_KEY[key].name}")

    def test_the_canonical_names_are_the_ones_rendered(self):
        page = RESULT_HTML.read_text(encoding="utf-8")
        for key in ("pose_coverage", "guard_index", "balance_index", "ring_center_control"):
            with self.subTest(metric=key):
                self.assertIn(f"<span>{BY_KEY[key].name}</span>", page)

    def test_the_deep_dive_no_longer_restates_the_numbers_above_it(self):
        """The accordion should add context, not repeat a value under a
        different label.

        Checked by what the block renders, not by its CSS class: the
        .performance-insights class is also used by the coaching summary for
        "What worked" and "Next training priority", which are narrative rather
        than a second printing of a measurement, and must stay.
        """
        page = RESULT_HTML.read_text(encoding="utf-8")
        self.assertIn('class="metric-standing"', page)
        # Pose coverage was rendered three times; the table's copy is gone.
        self.assertNotIn("<td>Pose coverage</td>", page)
        # The four metric values are no longer printed a second time inside
        # the accordion. Each of these was a whole <article> restating a
        # number from the rail above it.
        for restated in ("m.guard_index*100)}}%{% else %}Not measured",
                         "m.balance_index*100)}}%{% else %}Not measured",
                         "m.ring_center_control*100)}}%{% else %}Not measured"):
            with self.subTest(restated=restated):
                self.assertNotIn(restated, page)


class DirectionAndBenchmarkTests(unittest.TestCase):
    def test_every_metric_says_which_way_is_better_or_that_nobody_knows(self):
        for metric in CATALOG:
            with self.subTest(metric=metric.key):
                self.assertIn(metric.direction, {"higher", "lower", "unknown"})

    def test_a_metric_with_no_benchmark_says_so_rather_than_showing_a_bare_number(self):
        reading = BY_KEY["ring_center_control"].reading(0.53)
        self.assertEqual(reading["standing"], "no_reference")
        self.assertIn("no reference", reading["reference"])

    def test_the_audits_own_example_now_carries_its_direction(self):
        """"Guard 14%" was the complaint. It should now say 14% is low."""
        reading = BY_KEY["guard_index"].reading(0.14)
        self.assertEqual(reading["standing"], "needs_work")
        self.assertIn("42%", reading["reference"])

    def test_the_bands_come_from_the_coaching_thresholds_they_claim_to(self):
        """If core/coaching.py retunes, this file must not keep quoting the
        old numbers as though they were what the product acts on."""
        coaching = (PROJECT_ROOT / "core" / "coaching.py").read_text(encoding="utf-8")
        self.assertIn(f"guard >= {metric_catalog.GUARD_STRENGTH}", coaching)
        self.assertIn(f"guard < {metric_catalog.GUARD_WEAKNESS}", coaching)
        self.assertIn(f"balance < {metric_catalog.BALANCE_WEAKNESS}", coaching)

    def test_a_band_is_never_described_as_two_bounds_when_it_has_one(self):
        """94% coverage read as "between the two: under 70% is worth
        training" - both bounds' worth of confidence about one."""
        for metric in CATALOG:
            if metric.band is None:
                continue
            low, high = metric.band
            if low is None or high is None:
                for value in (0.0, 0.5, 0.99):
                    with self.subTest(metric=metric.key, value=value):
                        self.assertNotIn("Between", metric.reading(value)["reference"])

    def test_a_missing_value_is_not_dressed_up_as_a_measurement(self):
        for metric in CATALOG:
            with self.subTest(metric=metric.key):
                reading = metric.reading(None)
                self.assertEqual(reading["standing"], "not_measured")
                self.assertIsNone(reading["value"])

    def test_readings_cover_the_catalogue_even_from_an_empty_report(self):
        self.assertEqual(len(readings({})), len(CATALOG))
        self.assertEqual(len(readings(None)), len(CATALOG))


class HonestyIsNotTidiedAwayTests(unittest.TestCase):
    """Audit item 12: the best thing in the report must survive items 10/11.

    "Punch and kick counting is switched off. It is not accurate yet, so
    WarriorIQ does not show it. Everything above is measured from your body
    and is real."
    """

    def test_the_punch_disclosure_is_still_on_the_page(self):
        page = RESULT_HTML.read_text(encoding="utf-8")
        haystack = re.sub(r"\s+", " ", page)
        self.assertIn("is switched off", haystack)
        self.assertIn("measured from your body", haystack)

    def test_no_metric_claims_a_benchmark_it_does_not_have(self):
        """The same standard applied to the new reference sentences: where
        there is no band, the page says there is none."""
        for metric in CATALOG:
            if metric.band is not None:
                self.assertTrue(metric.band_source,
                                f"{metric.key} has a band but does not say where it came from")
            else:
                with self.subTest(metric=metric.key):
                    self.assertIn("no reference", metric.reading(0.5)["reference"])


if __name__ == "__main__":
    unittest.main()
