"""One "Analysed" figure in the report header (QA, 2026-10-07, item 22).

The header showed "Analysed: Whole video · 0:20" beside "Fight footage
analysed: 0:20 of 0:20": two summaries of the same footage, built separately
(_analysed_span_summary and _fight_footage_summary) and printed side by side.
"""

from __future__ import annotations

import unittest

from app.main import _analysed_label, _analysed_span_summary, _fight_footage_summary


def _report(fight_seconds, start=0.0, end=20.0, duration=20.0):
    video = {"analysed_span": {"start_seconds": start, "end_seconds": end, "video_duration_seconds": duration}}
    if fight_seconds is not None:
        video["fight_footage"] = {"fight_seconds": fight_seconds, "analysed_seconds": end - start,
                                  "excluded_seconds": (end - start) - fight_seconds}
    return {"video": video}


def _label(report):
    return _analysed_label(_analysed_span_summary(report), _fight_footage_summary(report))


class AnalysedLabelTests(unittest.TestCase):
    def test_qa_clip_says_it_once(self):
        self.assertEqual(_label(_report(20.0)), "Whole video · 0:20")

    def test_fight_share_added_only_when_it_differs(self):
        self.assertEqual(_label(_report(14.0)), "Whole video · 0:20 · 0:14 of it fight footage")
        self.assertEqual(_label(_report(19.6)), "Whole video · 0:20")

    def test_part_of_the_video(self):
        self.assertEqual(_label(_report(50.0, start=30.0, end=90.0, duration=116.0)),
                         "0:30–1:30 of 1:56 · 0:50 of it fight footage")

    def test_without_fight_footage_record(self):
        self.assertEqual(_label(_report(None)), "Whole video · 0:20")
        self.assertIsNone(_analysed_label(None, None))

    def test_header_has_a_single_row(self):
        from pathlib import Path

        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertEqual(page.count("<span>Analysed</span>"), 1)
        self.assertNotIn("<span>Fight footage analysed</span>", page)


if __name__ == "__main__":
    unittest.main()
