"""/progress/<id> has nothing to show before the fighters are picked or after
the analysis finished (QA, 2026-10-07: "Analysis running 0.0% - Loading the
analysis models" forever on a job still waiting for its fighters)."""

from __future__ import annotations

import unittest


class ProgressRedirectTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.main as webapp
        from app.state import create_job

        self.client = TestClient(webapp.app)
        self.client.get("/")
        guest = self.client.cookies.get(webapp.GUEST_COOKIE)
        self.job_id = "progredir1"
        create_job(self.job_id, {"owner_key": f"guest:{guest}", "video_path": "x.mp4",
                                 "fight_type": "sparring", "ruleset": "K1"})

    def tearDown(self):
        from app.state import delete_job

        delete_job(self.job_id)

    def _get(self):
        return self.client.get(f"/progress/{self.job_id}", follow_redirects=False)

    def test_before_fighters_are_picked_it_goes_to_selection(self):
        response = self._get()
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], f"/select/{self.job_id}")

    def test_a_finished_analysis_goes_to_its_result(self):
        from app.state import update_job

        update_job(self.job_id, {"status": "complete"})
        response = self._get()
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], f"/result/{self.job_id}")

    def test_a_running_analysis_shows_progress(self):
        from app.state import update_job

        for status in ("queued", "running", "interrupted", "error"):
            update_job(self.job_id, {"status": status})
            self.assertEqual(self._get().status_code, 200, status)


if __name__ == "__main__":
    unittest.main()
