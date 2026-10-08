"""An empty coaching card says why it is empty - and only blames the footage
when the footage is the reason.

QA, 2026-10-07 (/result/b0bc06c741a7): "No measured target yet: this fight did
not give enough evidence to set one" beside "100% seen". The plan was empty
because nothing measured was clearly behind the opponent.
"""

from __future__ import annotations

import unittest

from core.coaching import (NO_CLEAR_GAP, NO_DIRECTION, NOTHING_UNUSUAL, TOO_LITTLE, WITHHELD,
                           build_pose_coaching, build_training_plan, coaching_gaps)

FOOTAGE_WORDS = ("enough evidence", "footage", "longer clip", "cleaner sample")


def _fighter(**values):
    base = {"pose_coverage": 1.0, "guard_index": 0.30, "balance_index": 0.72, "ring_center_control": 0.5,
            "pressure_index": 0.02, "footwork_body_lengths_per_second": 1.0}
    base.update(values)
    return base


def _report(a, b=None, mode="pose_only"):
    report = {"metrics": {"A": a}, "integrity": {"coaching_evidence_mode": mode}}
    if b is not None:
        report["metrics"]["B"] = b
    return report


class GapTests(unittest.TestCase):
    def test_the_qa_report_level_with_its_opponent_and_fully_seen(self):
        a, b = _fighter(), _fighter(guard_index=0.31, balance_index=0.73)
        coaching = build_pose_coaching("A", a, b, "taekwondo")
        self.assertEqual(build_training_plan(coaching, "A", a), [], "the precondition: no plan")
        gaps = coaching_gaps(_report(a, b), "A", "taekwondo")
        self.assertEqual(gaps["code"], NO_CLEAR_GAP)
        for text in (gaps["keep"], gaps["fix"], gaps["plan"]):
            for word in FOOTAGE_WORDS:
                self.assertNotIn(word, text)
        self.assertIn("not clearly behind your opponent", gaps["plan"])
        self.assertIn("balance", gaps["plan"])

    def test_too_little_measured_says_so_with_what_was_seen(self):
        empty = {"pose_coverage": 1.0, "guard_note": "Not measured: the fighter is too small in this video."}
        gaps = coaching_gaps(_report(empty, _fighter()), "A")
        self.assertEqual(gaps["code"], TOO_LITTLE)
        self.assertIn("100%", gaps["plan"])
        self.assertIn("too small", gaps["plan"])

    def test_nothing_with_a_better_direction(self):
        only_style = {"pose_coverage": 1.0, "ring_center_control": 0.5, "pressure_index": 0.1,
                      "footwork_body_lengths_per_second": 1.2}
        gaps = coaching_gaps(_report(only_style, only_style), "A", "kickboxing")
        self.assertEqual(gaps["code"], NO_DIRECTION)
        self.assertIn("depend on how you fight", gaps["plan"])

    def test_no_opponent_to_compare_against(self):
        gaps = coaching_gaps(_report(_fighter()), "A")
        self.assertEqual(gaps["code"], NOTHING_UNUSUAL)
        self.assertIn("no opponent", gaps["plan"])

    def test_withheld_coaching_keeps_its_own_reason(self):
        report = _report(_fighter(), _fighter(), mode="withheld_identity_failure")
        report["coaching"] = {"A": {"note": "Coaching withheld because WarriorIQ could not confirm who was who."}}
        gaps = coaching_gaps(report, "A")
        self.assertEqual(gaps["code"], WITHHELD)
        self.assertIn("who was who", gaps["plan"])


class RenderTests(unittest.TestCase):
    def test_the_report_page_prints_the_reason_not_the_old_sentence(self):
        import json
        from pathlib import Path

        from jinja2 import ChainableUndefined, Environment, FileSystemLoader

        from app.main import _analysis_quality_summary, sport_identity
        from app.main import templates as app_templates

        root = Path(__file__).resolve().parents[1] / "app" / "templates"
        env = Environment(loader=FileSystemLoader(str(root)), undefined=ChainableUndefined)
        env.filters.update(app_templates.env.filters)
        env.globals.update(app_templates.env.globals)
        report = json.loads((Path(__file__).resolve().parent / "fixtures" / "report_sample.json")
                            .read_text(encoding="utf-8"))
        report["metrics"]["A"].update(_fighter())
        report["metrics"]["B"].update(_fighter(guard_index=0.31, balance_index=0.73))
        report["coaching"] = {side: {"strengths": [], "improvements": [], "drills": []} for side in "AB"}
        report["training_plan"] = {"A": [], "B": []}

        class Stub:
            def __init__(self, **kw): self.__dict__.update(kw)
            def __getattr__(self, key): return Stub()
            def __getitem__(self, key): return Stub()
            def __str__(self): return ""
            def __bool__(self): return False
            def __iter__(self): return iter(())

        page = env.get_template("result.html").render(
            request=Stub(url=Stub(path="/report/abc"), state=Stub(account=None), cookies={}, headers={}),
            job_id="abc", report=report, identity=sport_identity("kickboxing"),
            report_access={"report_tier": "full", "report_label": "Full", "label": "Full"},
            analysis_quality=_analysis_quality_summary(report), unavailable=[])
        self.assertNotIn("did not give enough evidence", page)
        self.assertNotIn("No safe plan could be generated", page)
        self.assertNotIn("A longer clip with both of you in view", page)
        self.assertIn("not clearly behind your opponent", page)
        self.assertIn("No drill from this fight", page)


if __name__ == "__main__":
    unittest.main()
