"""The setup page finds a clear moment and the people in it by itself.

Covers core/person_detect.py's choice of a starting pair, and the selection
page's use of it in app/main.py: a fresh upload is moved to the clearest early
moment once, and a moment somebody chose is never overridden.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np


def _person(x1, y1, x2, y2, confidence=0.8):
    return {"box": [float(x1), float(y1), float(x2), float(y2)], "confidence": confidence}


class PairScoreTests(unittest.TestCase):
    HEIGHT = 480

    def test_two_people_apart_make_a_clear_start(self):
        from core.person_detect import pair_score

        score, pair = pair_score([_person(100, 100, 220, 420), _person(500, 110, 610, 420)], self.HEIGHT)
        self.assertEqual(pair, (0, 1))
        self.assertGreater(score, 0.5)

    def test_overlapping_people_are_not_a_clear_start(self):
        from core.person_detect import pair_score

        score, pair = pair_score([_person(100, 100, 260, 420), _person(150, 100, 300, 420)], self.HEIGHT)
        self.assertIsNone(pair)
        self.assertEqual(score, 0.0)

    def test_a_distant_figure_is_not_paired_with_a_near_one(self):
        """A spectator far behind is a fifth of a fighter's height."""
        from core.person_detect import pair_score

        _, pair = pair_score([_person(100, 60, 260, 470), _person(500, 200, 530, 270)], self.HEIGHT)
        self.assertIsNone(pair)

    def test_someone_standing_across_a_fighter_lowers_the_score(self):
        from core.person_detect import pair_score

        clear, _ = pair_score([_person(100, 100, 220, 420), _person(500, 110, 610, 420)], self.HEIGHT)
        crossed, _ = pair_score([_person(100, 100, 220, 420), _person(500, 110, 610, 420),
                                 _person(170, 110, 280, 420)], self.HEIGHT)
        self.assertLess(crossed, clear)

    def test_cut_off_at_the_frame_edge_is_not_a_clear_start(self):
        """QA 2026-09: the pick must show two full-body people."""
        from core.person_detect import pair_score

        whole, _ = pair_score([_person(100, 100, 220, 420), _person(500, 110, 610, 420)], self.HEIGHT)
        cut, pair = pair_score([_person(100, 100, 220, 480), _person(500, 110, 610, 420)], self.HEIGHT)
        head_and_shoulders, other = pair_score([_person(100, 0, 220, 300), _person(500, 0, 610, 300)], self.HEIGHT)
        self.assertGreater(whole, 0.0)
        self.assertIsNone(pair)
        self.assertIsNone(other)
        self.assertEqual((cut, head_and_shoulders), (0.0, 0.0))

    def test_two_people_too_far_apart_to_fight_are_not_a_clear_start(self):
        """A singer and a guitarist at opposite ends of a stage are not a bout."""
        from core.person_detect import pair_score

        _, pair = pair_score([_person(20, 300, 60, 420), _person(1200, 300, 1240, 420)], self.HEIGHT)
        self.assertIsNone(pair)


class LivePageStillsTests(unittest.TestCase):
    def test_live_page_switches_to_stills_when_the_video_cannot_play(self):
        from pathlib import Path

        page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
        self.assertIn('id="liveStill"', page)
        self.assertIn("video.addEventListener('error',useStills)", page)
        self.assertIn("if(!(video.videoWidth>0))useStills()", page)
        self.assertIn("/live-frame/${jobId}?t=", page)
        self.assertNotIn("The video preview could not load", page)


class DetectorSwitchTests(unittest.TestCase):
    def test_switched_off_detector_finds_nothing_and_says_so(self):
        """None, not an empty list: "unavailable" is not "nobody there"."""
        from core import person_detect

        with patch.object(person_detect, "ENABLED", False), \
                patch.object(person_detect, "_net", None), \
                patch.object(person_detect, "_unavailable", False):
            self.assertIsNone(person_detect.detect_people(np.zeros((48, 64, 3), np.uint8)))
            self.assertIsNone(person_detect.find_clear_moment("missing.mp4"))


class AutoFrameRouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.main as webapp

        self.webapp = webapp
        self.client = TestClient(webapp.app)
        self.client.get("/")
        self.guest = self.client.cookies.get(webapp.GUEST_COOKIE)
        self.job_id = "autoframe01"
        self.video = webapp.UPLOADS / f"{self.job_id}.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for value in range(30):
            writer.write(np.full((48, 64, 3), value * 8, np.uint8))
        writer.release()
        job_dir = webapp.OUTPUTS / self.job_id
        job_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(job_dir / "selection.jpg"), np.zeros((48, 64, 3), np.uint8))
        from app.state import create_job

        create_job(self.job_id, {
            "owner_key": f"guest:{self.guest}", "status": "selection",
            "video_path": str(self.video), "original_name": "fight.mp4",
            "video_width": 64, "video_height": 48, "video_duration": 3.0,
            "start_seconds": 0.0, "selection_frame": 0,
        })

    def tearDown(self):
        from app.state import delete_job

        delete_job(self.job_id)
        self.video.unlink(missing_ok=True)

    def _moment(self):
        return {"frame_index": 15, "seconds": 1.5, "frame": None, "score": 0.7, "pair": (0, 1),
                "people": [_person(2, 4, 20, 46), _person(40, 4, 60, 46)]}

    def test_a_fresh_upload_moves_to_the_clear_moment_once(self):
        from app.state import get_job

        with patch.object(self.webapp, "find_clear_moment", return_value=self._moment()) as scan:
            first = self.client.get(f"/api/detect/{self.job_id}").json()
            second = self.client.get(f"/api/detect/{self.job_id}").json()
        self.assertTrue(first["frame_moved"])
        self.assertEqual(len(first["people"]), 2)
        self.assertAlmostEqual(first["seconds"], 1.5, places=1)
        self.assertFalse(second.get("frame_moved"))
        self.assertEqual(scan.call_count, 1)
        job = get_job(self.job_id)
        self.assertTrue(job["auto_frame_done"])
        self.assertAlmostEqual(float(job["start_seconds"]), 1.5, places=1)

    def test_a_chosen_moment_is_never_replaced(self):
        from app.state import get_job

        chosen = self.client.post(f"/api/selection-frame/{self.job_id}", json={"seconds": 0.5},
                                  headers=self._csrf())
        self.assertEqual(chosen.status_code, 200, chosen.text)
        with patch.object(self.webapp, "find_clear_moment", return_value=self._moment()) as scan:
            self.client.get(f"/api/detect/{self.job_id}")
        scan.assert_not_called()
        self.assertAlmostEqual(float(get_job(self.job_id)["start_seconds"]), 0.5, places=1)

    def test_no_clear_moment_keeps_the_frame_and_stops_looking(self):
        from app.state import get_job

        with patch.object(self.webapp, "find_clear_moment", return_value=None) as scan:
            self.client.get(f"/api/detect/{self.job_id}")
            self.client.get(f"/api/detect/{self.job_id}")
        self.assertEqual(scan.call_count, 1)
        job = get_job(self.job_id)
        self.assertTrue(job["auto_frame_done"])
        self.assertEqual(float(job["start_seconds"]), 0.0)

    def test_the_light_detector_runs_even_with_pose_detection_switched_off(self):
        """On Render WARRIORIQ_SELECTION_DETECTION is off, and no box was ever drawn."""
        import dataclasses

        people = [_person(2, 4, 20, 46), _person(40, 4, 60, 46)]
        settings = dataclasses.replace(self.webapp.SETTINGS, selection_detection_enabled=False)
        with patch.object(self.webapp, "SETTINGS", settings), \
                patch.object(self.webapp, "find_clear_moment", return_value=None), \
                patch.object(self.webapp, "detect_people_in_frame", return_value=people):
            answer = self.client.get(f"/api/detect/{self.job_id}").json()
        self.assertEqual(answer["availability"], "candidates_ready")
        self.assertEqual(len(answer["people"]), 2)
        self.assertTrue(answer["pair_found"])
        with patch.object(self.webapp, "SETTINGS", settings), \
                patch.object(self.webapp, "detect_people_in_frame", return_value=None):
            unavailable = self.client.get(f"/api/detect/{self.job_id}").json()
        self.assertEqual(unavailable["availability"], "manual_only")
        self.assertFalse(unavailable["pair_found"])

    def test_the_page_says_when_the_frame_was_chosen_for_them(self):
        from app.state import update_job

        update_job(self.job_id, {"selection_source": "auto", "selection_seconds": 12.0})
        page = self.client.get(f"/select/{self.job_id}").text
        self.assertIn("WarriorIQ chose this moment (0:12) for you.", page)
        self.assertIn(f'href="/frame/{self.job_id}">Pick a different moment', page)
        # Until boxes are drawn, the copy asks for drawn boxes, not taps.
        self.assertIn('<p id="selectIntro">Drag a box around each fighter', page)
        self.client.post(f"/api/selection-frame/{self.job_id}", json={"seconds": 0.5}, headers=self._csrf())
        self.assertNotIn('id="frameNote"', self.client.get(f"/select/{self.job_id}").text)

    def test_moving_to_a_clear_moment_records_who_chose_it(self):
        from app.state import get_job

        with patch.object(self.webapp, "find_clear_moment", return_value=self._moment()):
            answer = self.client.get(f"/api/detect/{self.job_id}").json()
        self.assertEqual(answer["frame_source"], "auto_pair")
        self.assertTrue(answer["pair_found"])
        self.assertEqual(get_job(self.job_id)["selection_source"], "auto_pair")

    def test_live_frame_serves_the_moment_being_analysed(self):
        """The live page's stills, for a browser that cannot play the upload."""
        response = self.client.get(f"/live-frame/{self.job_id}?t=1.0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (48, 64))
        # Frame 10 of the clip was written with value 80; JPEG keeps it close.
        self.assertAlmostEqual(float(image.mean()), 80.0, delta=6.0)

    def test_live_frame_is_private_to_the_fight_owner(self):
        from fastapi.testclient import TestClient

        stranger = TestClient(self.webapp.app)
        self.assertEqual(stranger.get(f"/live-frame/{self.job_id}?t=1.0").status_code, 404)

    def _csrf(self):
        import re

        page = self.client.get(f"/select/{self.job_id}").text
        return {"X-CSRF-Token": re.search(r'name="csrf-token" content="([^"]+)"', page).group(1)}

if __name__ == "__main__":
    unittest.main()
