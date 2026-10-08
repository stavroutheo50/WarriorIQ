"""A sideways video can be turned before the analysis, or is not measured as upright.

QA, 2026-10-07 (/result/f5f685f3b5cc): the upload said "filmed sideways,
could not turn it", offered nothing but uploading again, and a solo analysis
then printed "100% seen, guard 39%" from a body lying across the picture.
"""

from __future__ import annotations

import re
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.video import decoding_rotation, get_video_info, open_capture, quarter_turn, read_frame, rotate_frame


def _marked_frame(width=64, height=48):
    """A grey frame with a white block in the top-left corner."""
    frame = np.full((height, width, 3), 60, np.uint8)
    frame[0:8, 0:12] = 255
    return frame


def _corner(frame) -> str:
    h, w = frame.shape[:2]
    corners = {"top-left": frame[:4, :4], "top-right": frame[:4, w - 4:],
               "bottom-left": frame[h - 4:, :4], "bottom-right": frame[h - 4:, w - 4:]}
    return max(corners, key=lambda key: float(corners[key].mean()))


class DecodingTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "sideways.mp4"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for _ in range(10):
            writer.write(_marked_frame())
        writer.release()

    def tearDown(self):
        self.folder.cleanup()

    def test_every_reader_sees_the_turned_video_inside_the_block_only(self):
        self.assertEqual(_corner(read_frame(self.path, 2)), "top-left")
        with decoding_rotation(self.path, 90):
            info = get_video_info(self.path)
            self.assertEqual((info.width, info.height), (48, 64))
            frame = read_frame(self.path, 2)
            self.assertEqual(frame.shape[:2], (64, 48))
            self.assertEqual(_corner(frame), "top-right")
            capture = open_capture(self.path)
            ok, first = capture.read()
            self.assertTrue(ok)
            self.assertEqual(first.shape[:2], (64, 48))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 48)
            capture.release()
        info = get_video_info(self.path)
        self.assertEqual((info.width, info.height), (64, 48))

    def test_the_analysis_decodes_inside_the_turn(self):
        from core import analyzer
        from core.types import AnalysisRequest

        seen = {}

        def fake_run(req, progress):
            info = get_video_info(req.video_path)
            seen["size"] = (info.width, info.height)
            seen["corner"] = _corner(read_frame(req.video_path, 1))
            return {}

        request = AnalysisRequest(video_path=str(self.path), fighter_a_box=[1, 1, 40, 60],
                                  fighter_b_box=[10, 1, 47, 60], rotate_clockwise=90)
        with patch.object(analyzer, "_analyze_from_seed", fake_run):
            analyzer.analyze(request)
        self.assertEqual(seen, {"size": (48, 64), "corner": "top-right"})
        self.assertEqual(get_video_info(self.path).width, 64)

    def test_quarter_turns_only(self):
        self.assertEqual([quarter_turn(v) for v in (0, 90, 180, 270, 360, 450, 45, None, "x")],
                         [0, 90, 180, 270, 0, 90, 0, 0, 0])
        self.assertIs(rotate_frame(None, 90), None)


class SelectionPageTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.main as webapp
        from app.state import create_job

        self.webapp = webapp
        self.client = TestClient(webapp.app)
        self.client.get("/")
        guest = self.client.cookies.get(webapp.GUEST_COOKIE)
        self.job_id = "rotate0001"
        self.video = webapp.UPLOADS / f"{self.job_id}.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for _ in range(30):
            writer.write(_marked_frame())
        writer.release()
        job_dir = webapp.OUTPUTS / self.job_id
        job_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(job_dir / "selection.jpg"), _marked_frame())
        create_job(self.job_id, {
            "owner_key": f"guest:{guest}", "status": "selection", "auto_frame_done": True,
            "video_path": str(self.video), "original_name": "fight.mp4",
            "video_width": 64, "video_height": 48, "video_duration": 3.0,
            "start_seconds": 0.0, "selection_frame": 0, "selection_seconds": 0.0,
            "fight_type": "sparring", "ruleset": "K1", "round_count": 1,
            "round_duration_seconds": 3.0, "break_duration_seconds": 0.0,
            "orientation": {"turned_clockwise": 0, "sideways": True,
                            "warning": self.webapp.SIDEWAYS_UPLOAD_WARNING},
        })

    def tearDown(self):
        from app.state import delete_job

        delete_job(self.job_id)
        self.video.unlink(missing_ok=True)

    def _csrf(self):
        page = self.client.get(f"/select/{self.job_id}").text
        return {"X-CSRF-Token": re.search(r'name="csrf-token" content="([^"]+)"', page).group(1)}

    def _rotate(self, still_sideways: bool):
        with patch.object(self.webapp, "needed_turn", return_value=90 if still_sideways else 0):
            return self.client.post(f"/api/selection-frame/{self.job_id}/rotate", json={"clockwise": 90},
                                    headers=self._csrf())

    def test_the_page_offers_the_turn(self):
        page = self.client.get(f"/select/{self.job_id}").text
        self.assertIn('id="rotateFrame"', page)
        self.assertIn("Rotate 90°", page)
        self.assertIn("filmed sideways", page)

    def test_turning_upright_turns_the_frame_and_the_analysis(self):
        from app.state import get_job

        response = self._rotate(still_sideways=False)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["rotate_clockwise"], 90)
        frame = cv2.imread(str(self.webapp.OUTPUTS / self.job_id / "selection.jpg"))
        self.assertEqual(frame.shape[:2], (64, 48))
        self.assertEqual(_corner(frame), "top-right")
        job = get_job(self.job_id)
        self.assertFalse(job["orientation"]["sideways"])
        self.assertIsNone(job["orientation"]["warning"])
        request = self.webapp._analysis_request(self.job_id, job, [1, 1, 40, 60], [10, 1, 47, 60], "A")
        self.assertEqual((request.rotate_clockwise, request.sideways), (90, False))
        payload = self.webapp._remote_job_payload(self.job_id, {**job, "fighter_a_box": [1, 1, 2, 2]})
        self.assertEqual((payload["rotate_clockwise"], payload["sideways"]), (90, False))
        import worker

        remote = worker._request_from_job(self.job_id, {**payload, "video_path": "x.mp4"})
        self.assertEqual((remote.rotate_clockwise, remote.sideways), (90, False))
        page = self.client.get(f"/select/{self.job_id}").text
        self.assertIn("Turned 90° by you", page)

    def test_still_sideways_after_turning_is_said_and_carried(self):
        from app.state import get_job

        self._rotate(still_sideways=True)
        self._rotate(still_sideways=True)
        job = get_job(self.job_id)
        self.assertEqual(job["rotate_clockwise"], 180)
        self.assertTrue(job["orientation"]["sideways"])
        self.assertEqual(job["orientation"]["warning"], self.webapp.SIDEWAYS_STILL_WARNING)
        request = self.webapp._analysis_request(self.job_id, job, [1, 1, 40, 40], [41, 1, 63, 40], "A")
        self.assertTrue(request.sideways)

    def test_no_turn_once_the_analysis_has_started(self):
        from app.state import update_job

        csrf = self._csrf()
        update_job(self.job_id, {"status": "queued"})
        response = self.client.post(f"/api/selection-frame/{self.job_id}/rotate", json={"clockwise": 90},
                                    headers=csrf)
        self.assertEqual(response.status_code, 409)


def _report(sideways):
    own = {"guard_index": 0.39, "balance_index": 0.8, "footwork_body_lengths_per_second": 0.7,
           "pose_coverage": 1.0, "numbers": {"hands_up_share": 0.39, "longest_hands_down_seconds": 2.0,
                                             "off_balance_count": 1},
           "moments": {"guard_index": {"low": [1.0]}, "balance_index": {"low": [2.0]}},
           "availability": {"guard": {"available": True}, "balance": {"available": True}}}
    report = {"mode": "solo", "metrics": {"A": own}}
    if sideways is not None:
        report["video"] = {"orientation": {"rotated_clockwise": 0, "sideways": sideways}}
    return report


class ReportTests(unittest.TestCase):
    def test_a_video_still_sideways_reports_no_guard_or_balance(self):
        from core.report import withhold_for_sideways

        own = withhold_for_sideways(_report(True))["metrics"]["A"]
        self.assertIsNone(own["guard_index"])
        self.assertIsNone(own["balance_index"])
        self.assertIsNone(own["numbers"]["hands_up_share"])
        self.assertIsNone(own["numbers"]["off_balance_count"])
        self.assertEqual(own["moments"], {})
        self.assertIn("sideways", own["pose_note"])
        self.assertIn("Rotate 90°", own["pose_note"])
        self.assertFalse(own["availability"]["balance"]["available"])
        # Movement comes from where the body was, not its shape, and stays.
        self.assertEqual(own["footwork_body_lengths_per_second"], 0.7)

    def test_an_upright_video_is_untouched(self):
        from core.report import withhold_for_sideways

        self.assertEqual(withhold_for_sideways(_report(False))["metrics"]["A"]["guard_index"], 0.39)

    def test_a_saved_report_takes_the_uploads_finding(self):
        """The QA report was saved before reports recorded their orientation."""
        from app.main import SIDEWAYS_UPLOAD_WARNING, _attach_orientation
        from core.report import withhold_for_sideways

        legacy_job = {"orientation": {"turned_clockwise": 0, "warning": SIDEWAYS_UPLOAD_WARNING}}
        report = withhold_for_sideways(_attach_orientation(_report(None), legacy_job))
        self.assertIsNone(report["metrics"]["A"]["guard_index"])
        turned_job = {**legacy_job, "rotate_clockwise": 90}
        report = withhold_for_sideways(_attach_orientation(_report(None), turned_job))
        self.assertEqual(report["metrics"]["A"]["guard_index"], 0.39)


if __name__ == "__main__":
    unittest.main()
