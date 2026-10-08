"""Opening a form-only address in a browser is an ordinary not-found page.

QA, 2026-10-07: /share/<anything> answered "405 - Method Not Allowed". The
share, story, cancel and revoke addresses only accept their form's POST, so a
pasted or bookmarked link to one reached the framework's method check.
"""

from __future__ import annotations

import unittest

from browser_client import BrowserClient

import app.main as webapp

ACTION_ADDRESSES = ("/share/anything", "/shares/anything/revoke", "/pending/anything/cancel", "/story/anything")


class ActionAddressTests(unittest.TestCase):
    def setUp(self):
        self.client = BrowserClient(webapp.app)

    def tearDown(self):
        self.client.close()

    def test_a_browser_gets_the_normal_404_page(self):
        expected = self.client.get("/no-such-page").text
        self.assertIn("That page left the ring", expected)
        for path in ACTION_ADDRESSES:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404, path)
            self.assertIn("That page left the ring", response.text, path)
            self.assertNotIn("Method Not Allowed", response.text, path)
            self.assertNotIn("allow", {key.lower() for key in response.headers}, path)

    def test_api_callers_still_see_the_real_status(self):
        response = self.client.get("/share/anything", headers={"accept": "application/json"})
        self.assertEqual(response.status_code, 405)

    def test_the_form_post_still_reaches_the_route(self):
        # Signed out, so the route itself refuses with its own 403.
        response = self.client.post("/share/anything", follow_redirects=False)
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
