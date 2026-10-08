"""Cancelling a pending upload says so.

QA, 2026-10-07: "Cancel" on a pending upload deleted it and redirected to
/history#pending with no message, so the fight simply vanished and the
fighter could not tell whether the cancel had worked or the page had lost it.
"""

from __future__ import annotations

import unittest
import uuid

from browser_client import BrowserClient

import app.main as webapp
from app.state import create_job, delete_job
from core.auth import register


class CancelNoticeTests(unittest.TestCase):
    def setUp(self):
        self.client = BrowserClient(webapp.app)
        email = f"cancel-{uuid.uuid4().hex[:8]}@example.com"
        account = register(email, "Strong-Local-Password")
        self.client.post("/login", data={"email": email, "password": "Strong-Local-Password",
                                         "accept_policies": "true"})
        self.job_id = f"cancel{uuid.uuid4().hex[:6]}"
        create_job(self.job_id, {"owner_key": f"account:{account['id']}", "video_path": "x.mp4",
                                 "fight_type": "sparring", "ruleset": "K1"})

    def tearDown(self):
        self.client.close()
        delete_job(self.job_id)

    def test_cancelling_lands_on_a_confirmation(self):
        self.assertIn(f"/pending/{self.job_id}/cancel", self.client.get("/history").text)
        response = self.client.post(f"/pending/{self.job_id}/cancel", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertIn("cancelled=1", response.headers["location"])
        landed = self.client.get(response.headers["location"]).text
        self.assertIn("Upload cancelled.", landed)
        self.assertIn('role="status"', landed)
        self.assertNotIn(f"/pending/{self.job_id}/cancel", landed)

    def test_no_notice_without_a_cancellation(self):
        self.assertNotIn("Upload cancelled.", self.client.get("/history").text)

    def test_signed_out_visitor_never_sees_the_notice(self):
        self.assertNotIn("Upload cancelled.", BrowserClient(webapp.app).get("/history?cancelled=1").text)


if __name__ == "__main__":
    unittest.main()
