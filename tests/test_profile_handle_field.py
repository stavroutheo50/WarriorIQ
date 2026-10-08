"""The username box holds exactly what a username may be.

QA, 2026-10-07: the hint said 3-30 characters but the box took 31. Its
maxlength left room for an optional leading "@", so a 31-character name
without one could be typed and was only refused by the server on save.
"""

from __future__ import annotations

import re
import unittest
import uuid

from browser_client import BrowserClient

import app.main as webapp
from core.auth import register
from core.social import HANDLE_PATTERN, normalize_handle


class HandleFieldTests(unittest.TestCase):
    def setUp(self):
        self.client = BrowserClient(webapp.app)
        email = f"handle-{uuid.uuid4().hex[:8]}@example.com"
        register(email, "Strong-Local-Password")
        self.client.post("/login", data={"email": email, "password": "Strong-Local-Password",
                                         "accept_policies": "true"})
        page = self.client.get("/profile").text
        self.field = re.search(r'<input id="profileHandle"[^>]*>', page).group(0)
        self.page = page

    def tearDown(self):
        self.client.close()

    def _attr(self, name):
        return re.search(rf'\s{name}="([^"]*)"', self.field).group(1)

    def test_box_limits_match_the_hint_and_the_server(self):
        self.assertEqual(self._attr("maxlength"), "30")
        self.assertEqual(self._attr("minlength"), "3")
        self.assertIn("3-30", self.page)
        browser_rule = re.compile(f"^(?:{self._attr('pattern')})$")
        samples = ["ab", "abc", "a" * 30, "a" * 31, "alex.kicks", "Alex_K", ".alex", "alex.", "al ex", "@alex"]
        for sample in samples:
            self.assertEqual(bool(browser_rule.match(sample)), bool(HANDLE_PATTERN.match(sample.lower())), sample)

    def test_server_still_accepts_a_leading_at(self):
        # The page strips a typed or pasted "@"; the server keeps accepting it
        # for anything that posts the form directly.
        self.assertEqual(normalize_handle("@Alex.Kicks"), "alex.kicks")
        with self.assertRaises(ValueError):
            normalize_handle("a" * 31)

    def test_page_strips_a_leading_at(self):
        self.assertIn("replace(/^@+/,'')", self.page)


if __name__ == "__main__":
    unittest.main()
