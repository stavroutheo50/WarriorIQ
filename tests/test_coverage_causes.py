"""Per-fighter attribution of why a fighter was lost.

The red corner is tracked worse than the blue one on every fight sampled -
64-74% against 82-88%, never once close. `blocked_recovery_reasons` already
counted why *somebody* was unassigned, but it pooled A and B, so it could not
say which of them a cause belonged to. A cause that is most of A's losses and
none of B's is indistinguishable, pooled, from one shared evenly between them,
and those call for opposite fixes.

These cover the split, not the bug: naming the cause is the step the audit
asked for before anybody picks a fix.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from core.identity import IdentityManager
from core.types import PersonObservation
from tools.report_coverage_causes import causes, render

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _person(track_id: int, x1: float, x2: float, appearance_index: int) -> PersonObservation:
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = np.linspace(x1 + 4, x2 - 4, 17)
    kp[:, 1] = np.linspace(100, 290, 17)
    appearance = np.zeros((64,), dtype=np.float32)
    appearance[appearance_index] = 1.0
    return PersonObservation(
        track_id=track_id,
        box=np.asarray([x1, 80, x2, 310], dtype=np.float32),
        confidence=0.95,
        keypoints=kp,
        keypoint_conf=np.ones((17,), dtype=np.float32),
        appearance=appearance,
    )


class BlockedAttributionTests(unittest.TestCase):
    def setUp(self):
        self.manager = IdentityManager(
            _person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)

    def test_a_reason_lands_on_the_fighter_it_belongs_to(self):
        self.manager._blocked("scored_below_empty_slot", "A")
        self.manager._blocked("scored_below_empty_slot", "A")
        self.manager._blocked("scored_below_empty_slot", "B")
        by_fighter = self.manager.blocked_recovery_by_fighter
        self.assertEqual(by_fighter["A"]["scored_below_empty_slot"], 2)
        self.assertEqual(by_fighter["B"]["scored_below_empty_slot"], 1)

    def test_the_pooled_total_still_agrees_with_the_split(self):
        """The pooled dict is what existing reports and tests read, so it must
        not change meaning; it must simply stop being the only view."""
        for fighter in ("A", "A", "B"):
            self.manager._blocked("lost_to_the_other_fighter", fighter)
        pooled = self.manager.blocked_recovery["lost_to_the_other_fighter"]
        split = sum(side.get("lost_to_the_other_fighter", 0)
                    for side in self.manager.blocked_recovery_by_fighter.values())
        self.assertEqual(pooled, 3)
        self.assertEqual(pooled, split)

    def test_an_unnamed_fighter_counts_pooled_and_does_not_raise(self):
        """A counter must never be able to end a paid run."""
        self.manager._blocked("some_reason")
        self.manager._blocked("some_reason", "not a fighter")
        self.assertEqual(self.manager.blocked_recovery["some_reason"], 2)
        self.assertEqual(self.manager.blocked_recovery_by_fighter["A"], {})
        self.assertEqual(self.manager.blocked_recovery_by_fighter["B"], {})

    def test_losing_a_fighter_records_a_cause_against_that_fighter(self):
        """Drive the real update path rather than the counter directly.

        Only B's man is on screen. A has nobody, so exactly one fighter should
        be recorded as missing and the cause should be A's.
        """
        before = dict(self.manager.missing_frames_by_fighter)
        a_obs, b_obs = self.manager.update([_person(2, 400, 500, 2)], 1)
        self.assertIsNone(a_obs, "A should not be assigned to B's man")
        self.assertIsNotNone(b_obs)
        self.assertEqual(self.manager.missing_frames_by_fighter["A"], before["A"] + 1)
        self.assertEqual(self.manager.missing_frames_by_fighter["B"], before["B"])
        self.assertTrue(self.manager.blocked_recovery_by_fighter["A"],
                        "A was lost but no cause was recorded against A")
        self.assertFalse(self.manager.blocked_recovery_by_fighter["B"])

    def test_an_empty_frame_is_not_attributed_to_either_fighter(self):
        """Nobody detected at all is a different fact from a fighter losing a
        contest for a detection, and must not be filed as one."""
        self.manager.update([], 1)
        self.assertFalse(self.manager.blocked_recovery_by_fighter["A"])
        self.assertFalse(self.manager.blocked_recovery_by_fighter["B"])


class AnalyzerEmitsTheSplitTests(unittest.TestCase):
    def test_the_tracking_block_carries_the_per_fighter_causes(self):
        source = (PROJECT_ROOT / "core" / "analyzer.py").read_text(encoding="utf-8")
        self.assertIn('"blocked_recovery_by_fighter"', source)
        self.assertIn('"missing_frames_by_fighter"', source)
        # The pooled view stays: reports and tests already read it.
        self.assertIn('"blocked_recovery_reasons"', source)


class ReportRenderingTests(unittest.TestCase):
    def _report(self, **tracking) -> dict:
        base = {"fighter_A_coverage": 0.64, "fighter_B_coverage": 0.86,
                "analyzed_frames": 500}
        base.update(tracking)
        return {"tracking": base}

    def test_it_names_the_cause_that_separates_the_fighters(self):
        text = render("fight", causes(self._report(
            missing_frames_by_fighter={"A": 180, "B": 70},
            blocked_recovery_by_fighter={
                "A": {"scored_below_empty_slot": 150, "cannot_tell_a_from_b": 30},
                "B": {"scored_below_empty_slot": 10, "cannot_tell_a_from_b": 60},
            })))
        self.assertIn("scored_below_empty_slot", text)
        # 83% of A's missing frames against 14% of B's is the finding; the
        # renderer must lead with it rather than with the biggest total.
        self.assertIn("83%", text)
        first_cause = [l for l in text.splitlines() if "scored_below_empty_slot" in l][0]
        rows = [l for l in text.splitlines() if "cannot_tell_a_from_b" in l]
        self.assertLess(text.index(first_cause), text.index(rows[0]))

    def test_it_says_so_when_no_cause_separates_them(self):
        text = render("fight", causes(self._report(
            missing_frames_by_fighter={"A": 100, "B": 100},
            blocked_recovery_by_fighter={
                "A": {"lost_to_the_other_fighter": 50},
                "B": {"lost_to_the_other_fighter": 50},
            })))
        self.assertIn("No cause separates", text)
        self.assertIn("look upstream", text)

    def test_an_older_report_is_handled_rather_than_crashed(self):
        """Every stored analysis predates this split."""
        text = render("fight", causes(self._report(
            blocked_recovery_reasons={"lost_to_the_other_fighter": 38})))
        self.assertIn("No per-fighter causes in this report", text)
        self.assertIn("lost_to_the_other_fighter", text)

    def test_a_real_stored_report_renders(self):
        stored = PROJECT_ROOT / "outputs" / "athens_hd" / "report.json"
        if not stored.exists():
            self.skipTest("no stored analysis in this checkout")
        text = render("athens_hd", causes(json.loads(stored.read_text(encoding="utf-8"))))
        self.assertIn("coverage", text)


if __name__ == "__main__":
    unittest.main()
