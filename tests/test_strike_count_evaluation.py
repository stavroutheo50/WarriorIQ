"""The strike-count harness, and the guarantee that punches stay measurable.

The punch head is switched off for display. If it were ever switched off at
the source - stopped being computed, or stripped from the stored report - then
no amount of labelling would make it measurable again and the decision to
withhold it could never be revisited. These tests hold that door open.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_strike_counts import (
    ARRIVED_OUTCOMES,
    Tally,
    evaluate,
    family_of,
    summarise,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FamilyTests(unittest.TestCase):
    def test_knees_are_counted_with_kicks(self):
        """The report treats them as one family, so the measurement must too:
        by eye a knee and a round kick are a coin flip."""
        for name in ("left_knee", "right_knee", "left_low_kick", "right_push_kick"):
            self.assertEqual(family_of(name), "kick")
        for name in ("jab", "cross", "left_hook", "left_uppercut", "backfist"):
            self.assertEqual(family_of(name), "punch")

    def test_an_unknown_technique_is_not_silently_a_kick(self):
        self.assertEqual(family_of(""), "punch")
        self.assertEqual(family_of("something_new"), "punch")


class TallyArithmeticTests(unittest.TestCase):
    def test_unlabelled_proposals_are_not_counted_as_wrong(self):
        """The bug this harness had on its first run.

        44 of pack_1mp4's 49 punch proposals had never been labelled. Counting
        them as misses reported 2% precision for a detector nobody had judged.
        """
        tally = Tally(proposed=49, confirmed=1, unsure=4, unlabelled=44)
        self.assertEqual(tally.judged, 1)
        self.assertEqual(tally.precision, 1.0)
        self.assertEqual(tally.over_count, 0)

    def test_unsure_is_neither_right_nor_wrong(self):
        tally = Tally(proposed=10, confirmed=4, unsure=6)
        self.assertEqual(tally.judged, 4)
        self.assertEqual(tally.precision, 1.0)

    def test_precision_is_undefined_rather_than_zero_with_no_evidence(self):
        """Zero would read as "this is terrible"; the truth is "nobody looked"."""
        self.assertIsNone(Tally(proposed=8, unlabelled=8).precision)
        self.assertIsNone(Tally().arrived_precision)

    def test_merging_keeps_every_counter(self):
        a = Tally(proposed=2, confirmed=1, unsure=1, arrived_judged=1, arrived_confirmed=1)
        b = Tally(proposed=3, confirmed=2, unlabelled=1, arrived_judged=2, arrived_confirmed=1)
        a.merge(b)
        self.assertEqual((a.proposed, a.confirmed, a.unsure, a.unlabelled), (5, 3, 1, 1))
        self.assertEqual((a.arrived_judged, a.arrived_confirmed), (3, 2))


class RealLabelTests(unittest.TestCase):
    """Run against the project's actual label packs."""

    def setUp(self):
        self.labels = PROJECT_ROOT / "tools" / "labels_athens_hd_claude.json"
        if not self.labels.exists():
            self.skipTest("athens_hd labels are not in this checkout")
        # The labels file is tracked; the pack it refers to is not. labelpack/
        # is gitignored - the clips are crops of identifiable people at a real
        # tournament - so on a fresh checkout `evaluate` finds no index.json
        # and returns None. Guarding on the labels file alone meant these four
        # tests ran on CI and failed there every time, while passing on any
        # machine that happened to have the pack. Skip on what is actually
        # needed, which is the pack.
        self.result = evaluate(self.labels)
        if self.result is None:
            self.skipTest("labelpack/athens_hd is not in this checkout (it is gitignored)")

    def test_it_reads_the_pack_and_finds_both_families(self):
        self.assertIsNotNone(self.result)
        self.assertIn("punch", self.result.families)
        self.assertIn("kick", self.result.families)

    def test_punches_are_still_being_measured(self):
        """The point of the whole file.

        Punches are withheld from the report. They must not be withheld from
        the evidence, or the decision to withhold them becomes permanent.
        """
        punch = self.result.families["punch"]
        self.assertGreater(punch.proposed, 0, "no punch proposals reached the label pack")
        self.assertGreater(punch.judged, 0, "no punch proposal has a human verdict")

    def test_the_summary_is_json_and_carries_its_caveats(self):
        summary = summarise([self.result])
        json.loads(json.dumps(summary))
        self.assertEqual(summary["schema"], "warrioriq.strike_counts.v1")
        # Recall must stay explicitly disclaimed: these labels sit on
        # proposals, so a strike never proposed is invisible here.
        self.assertIn("not measurable", summary["measures"]["recall"])

    def test_a_baseline_round_trips(self):
        summary = summarise([self.result])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), summary)


class PublishedSubsetTests(unittest.TestCase):
    def test_the_shown_outcomes_match_the_reports_own_list(self):
        """If core.report widens what it publishes, this measurement has to
        follow, or "shown" stops meaning what a reader sees."""
        from core.report import ARRIVED_OUTCOMES as REPORT_OUTCOMES

        self.assertEqual(set(ARRIVED_OUTCOMES), set(REPORT_OUTCOMES))


class SuppressionIsDisplayOnlyTests(unittest.TestCase):
    """Punches may be hidden. They may not be deleted."""

    def test_the_switch_is_a_single_named_constant(self):
        from core import report

        self.assertIsInstance(report.STRIKE_COUNTS_PRECISION_VALIDATED, bool)

    def test_the_event_stream_is_not_filtered_by_family(self):
        """core.report filters what is *shown*. A family filter applied to the
        stored events would destroy the evidence this harness runs on."""
        source = (PROJECT_ROOT / "core" / "report.py").read_text(encoding="utf-8")
        self.assertIn("deliberately does not touch `events`", source)

    def test_a_stored_report_still_carries_its_punches(self):
        stored = PROJECT_ROOT / "outputs" / "athens_hd" / "report.json"
        if not stored.exists():
            self.skipTest("no stored analysis in this checkout")
        events = json.loads(stored.read_text(encoding="utf-8")).get("events") or []
        punches = [e for e in events if family_of(str(e.get("technique") or "")) == "punch"]
        self.assertGreater(
            len(punches), 0,
            "the stored report has no punch events, so the punch head cannot be "
            "evaluated from it any more")


class ProvenanceTests(unittest.TestCase):
    """A number is only as good as who produced the labels under it.

    Every label set in this repository was written by Claude and says so in
    its own header. The harness read them as though a person had, and
    reported a precision figure that was one model grading another model's
    output - on exactly the case in dispute, whether a few pixels of arm
    movement was a punch.
    """

    def test_a_machine_label_set_is_recognised_as_one(self):
        from tools.evaluate_strike_counts import provenance

        machine = provenance({
            "labeller": "claude-opus-5",
            "_what_this_is": ["MACHINE-GENERATED, by Claude, from the filmstrips."],
        })
        self.assertTrue(machine["machine_generated"])
        self.assertEqual(machine["labeller"], "claude-opus-5")

    def test_a_human_label_set_is_not_flagged(self):
        from tools.evaluate_strike_counts import provenance

        human = provenance({"labeller": "a coach", "_what_this_is": ["Watched on video."]})
        self.assertFalse(human["machine_generated"])

    def test_the_summary_says_whether_it_rests_on_human_judgement(self):
        from tools.evaluate_strike_counts import discover, evaluate, summarise

        results = [r for r in (evaluate(p) for p in discover()) if r]
        if not results:
            self.skipTest("no label packs in this checkout")
        summary = summarise(results)
        self.assertIn("human_ground_truth", summary)
        self.assertIn("label_sources", summary)
        # Every pack in the repository is machine-labelled today, so this is
        # False. If it ever becomes True somebody has done real labelling and
        # the warning should stop being printed.
        self.assertFalse(summary["human_ground_truth"])

    def test_the_report_warns_when_nothing_was_judged_by_a_person(self):
        from tools.evaluate_strike_counts import discover, evaluate, render

        results = [r for r in (evaluate(p) for p in discover()) if r]
        if not results:
            self.skipTest("no label packs in this checkout")
        text = render(results)
        self.assertIn("WRITTEN BY A MODEL", text)
        self.assertIn("never the evidence that settles it", text)

    def test_thinness_is_measured_rather_than_read_off_the_header(self):
        """The first version matched "could not tell" in a header and flagged
        athens_hd, whose header contains that phrase while describing a
        different pack - to explain why that one is weak and this one is not.
        """
        from tools.evaluate_strike_counts import discover, evaluate, thinness

        by_pack = {r.pack: thinness(r) for r in
                   (x for x in (evaluate(p) for p in discover()) if x)}
        if "athens_hd" not in by_pack:
            self.skipTest("no label packs in this checkout")
        self.assertLess(by_pack["athens_hd"], 0.5,
                        "athens_hd is well judged and must not be flagged thin")
        self.assertGreater(by_pack.get("pack_1mp4", 0), 0.5,
                           "pack_1mp4 is mostly unjudged and should be flagged")


if __name__ == "__main__":
    unittest.main()
