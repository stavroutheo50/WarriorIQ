"""The strike detector, scored against a competitor's answers on a real fight.

tools/benchmark_labelled_fight.py replays the action engine over a stored pose
track and matches its proposals to 72 hand-labelled moments. This pins the
result: a change to detection that moves any number fails here until the new
numbers are looked at and written with --write-baseline. That is the point -
"the detector got better" becomes a diff of this file, not an impression.
"""
from __future__ import annotations

import json
import unittest

from tools import benchmark_labelled_fight as bench


class LabelledFightBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.now = bench.run()
        cls.baseline = json.loads((bench.FIGHT / "baseline.json").read_text(encoding="utf-8"))

    def test_the_numbers_match_the_stored_baseline(self):
        self.assertEqual(self.now, self.baseline,
                         "detection changed: review the numbers, then run "
                         "python tools/benchmark_labelled_fight.py --write-baseline")

    def test_every_labelled_fight_matches_its_baseline(self):
        for fight in bench.labelled_fights():
            with self.subTest(fight=fight.name):
                baseline = json.loads((fight / "baseline.json").read_text(encoding="utf-8"))
                self.assertEqual(bench.run(fight), baseline)

    def test_a_one_tap_answer_becomes_a_benchmark_label(self):
        from unittest import mock

        from tools import export_strike_checks as export
        stored = [
            {"event_time": 3.25, "predicted": {"family": "punch"},
             "corrected": {"fighter": "A", "verdict": "kick", "source": "strike_check"}},
            {"event_time": 5.0, "predicted": {"family": "kick"},
             "corrected": {"fighter": "B", "verdict": "right", "source": "strike_check"}},
            {"event_time": 6.0, "predicted": {"family": "punch"},
             "corrected": {"fighter": "B", "verdict": "not_a_strike", "source": "strike_check"}},
            {"event_time": 8.0, "predicted": {}, "corrected": {"technique": "jab"}},   # a full correction
        ]
        with mock.patch.object(export, "get_annotations", return_value=stored):
            labels = export.labels_for("x")
        self.assertEqual([(l["fighter"], l["answer"]) for l in labels],
                         [("A", "kick"), ("B", "kick"), ("B", "none")])
        self.assertEqual({bench._family(l["answer"]) for l in labels}, {"kick", None})

    def test_the_replay_reproduces_the_analysis(self):
        # The analysis of this track produced 42 events. A replay that did not
        # would be measuring a different pipeline from the one that runs.
        self.assertEqual(self.now["all_proposals"]["proposals_raw"], 42)

    def test_every_counted_proposal_is_accounted_for(self):
        parts = ("real_strike", "nothing_happened", "unsure", "unlabelled")
        for section in self.now.values():
            self.assertEqual(sum(section[p] for p in parts), section["proposals_counted"])

    def test_a_punch_thrown_at_nothing_is_not_counted_but_a_missed_kick_is(self):
        from types import SimpleNamespace

        from core.analyzer import _punch_thrown_at_nothing
        self.assertTrue(_punch_thrown_at_nothing(SimpleNamespace(family="punch", outcome="missed")))
        self.assertFalse(_punch_thrown_at_nothing(SimpleNamespace(family="kick", outcome="missed")))
        self.assertFalse(_punch_thrown_at_nothing(SimpleNamespace(family="punch", outcome="blocked")))

    def test_the_ambiguous_moment_is_decided_by_majority(self):
        moments = {(m["fighter"], m["time"]): m for m in bench.load_moments()}
        self.assertIsNone(moments[("A", 87.67)]["truth"])      # cross, none, wrong person, none
        self.assertEqual(moments[("A", 83.83)]["truth"], "punch")


if __name__ == "__main__":
    unittest.main()
