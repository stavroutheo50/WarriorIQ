"""Decoding ahead on a helper thread must hand the loop exactly what inline
decoding would: the same frames, in order, converted on the same rule."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.frame_feed import FrameFeed


def _video(path: Path, frames: int = 90) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index % 250, np.uint8))
    writer.release()


def _walk(path: Path, ahead: bool, *, retrieve_all=False, history_step=None, seed=7):
    """Drive a feed the way the analysis loop does, with a varying stride."""
    rng = random.Random(seed)
    cap = cv2.VideoCapture(str(path))
    cap.grab()  # the loop starts after the already-consumed first frame
    gate = 1 + 3
    feed = FrameFeed(cap, first_source_frame=1, next_inference_frame=gate,
                     retrieve_all=retrieve_all, history_step=history_step,
                     next_history_frame=30, ahead=ahead)
    seen = []
    try:
        while True:
            fed = feed.next()
            if fed is None:
                break
            converted = fed.frame is not None
            seen.append((fed.source_frame, round(fed.pts_ms, 3), converted,
                         int(fed.frame.mean()) if converted else None))
            if fed.source_frame >= gate:
                gate = fed.source_frame + rng.choice((1, 2, 3, 5, 9))
                feed.allow_through(gate)
    finally:
        feed.close()
        cap.release()
    return seen


class FrameFeedTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "clip.mp4"
        _video(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_ahead_matches_inline_exactly(self):
        for options in ({}, {"retrieve_all": True}, {"history_step": 30}):
            with self.subTest(**options):
                inline = _walk(self.path, ahead=False, **options)
                ahead = _walk(self.path, ahead=True, **options)
                self.assertEqual(ahead, inline)
                self.assertEqual([item[0] for item in ahead], list(range(1, 90)))

    def test_only_wanted_frames_are_converted(self):
        seen = _walk(self.path, ahead=True)
        converted = [index for index, _, done, _ in seen if done]
        self.assertLess(len(converted), len(seen))
        self.assertTrue(converted)

    def test_closing_early_stops_the_helper(self):
        cap = cv2.VideoCapture(str(self.path))
        feed = FrameFeed(cap, first_source_frame=0, next_inference_frame=10_000,
                         retrieve_all=False, history_step=None, next_history_frame=0,
                         ahead=True, depth=2)
        self.assertIsNotNone(feed.next())
        feed.close()
        self.assertFalse(feed._thread.is_alive())
        cap.release()


class FrameBreakdownTests(unittest.TestCase):
    def test_breakdown_is_per_frame_and_names_the_rest(self):
        from core.analyzer import _frame_breakdown

        steps = {"pose_model": 6.0, "reading_video": 1.0}
        breakdown = _frame_breakdown(steps, total=10.0, frames=100)
        self.assertEqual(breakdown, {"pose_model": 0.06, "reading_video": 0.01, "other": 0.03})
        self.assertIsNone(_frame_breakdown(steps, total=10.0, frames=0))

    def test_report_shows_the_breakdown(self):
        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn('id="report-frame-breakdown"', page)
        self.assertIn("perf.frame_pass_breakdown_per_frame", page)


if __name__ == "__main__":
    unittest.main()
