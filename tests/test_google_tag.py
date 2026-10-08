"""The Google tag loads only on public marketing pages, only after consent.

QA, 2026-10-07: a Google Ads page-view request (pagead2.googlesyndication.com/
ccm/collect, AW- destination) fired on every page, private ones included, and
carried the full report URL (/result/<id>) before any consent.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager

import app.main as webapp


@contextmanager
def setting(name, value):
    """SETTINGS is frozen; tests in this suite swap a field the same way."""
    previous = getattr(webapp.SETTINGS, name)
    object.__setattr__(webapp.SETTINGS, name, value)
    try:
        yield
    finally:
        object.__setattr__(webapp.SETTINGS, name, previous)

ACCEPTED = {"analytics": True, "decided": True}
DECLINED = {"analytics": False, "decided": True}


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.patch = setting("analytics_measurement_id", "GT-TEST")
        self.patch.__enter__()

    def tearDown(self):
        self.patch.__exit__(None, None, None)

    def test_marketing_pages_after_consent_only(self):
        for path in webapp.GOOGLE_TAG_PATHS:
            self.assertTrue(webapp.google_tag_allowed(path, ACCEPTED), path)
            self.assertFalse(webapp.google_tag_allowed(path, DECLINED), path)
            self.assertFalse(webapp.google_tag_allowed(path, {}), path)

    def test_never_on_private_or_unlisted_routes(self):
        for path in ("/result/9138ca9b38a7", "/replay/x", "/progress/x", "/select/x", "/history", "/camp",
                     "/profile", "/s/token", "/share/x", "/admin", "/login", "/signup", "/f/token",
                     "/privacy", "/cookies", "/analyze", "/feed"):
            self.assertFalse(webapp.google_tag_allowed(path, ACCEPTED), path)

    def test_marketing_list_holds_no_private_prefix(self):
        for path in webapp.GOOGLE_TAG_PATHS:
            self.assertFalse(path.startswith(webapp.PRIVATE_ROUTE_PREFIXES), path)

    def test_nothing_configured_loads_nothing(self):
        with setting("analytics_measurement_id", ""), setting("gtm_container_id", ""):
            self.assertFalse(webapp.google_tag_allowed("/", ACCEPTED))


class PolicyTextTests(unittest.TestCase):
    def test_cookie_and_subprocessor_pages_describe_what_happens(self):
        from core.legal import LEGAL_DOCUMENTS, subprocessor_sections

        cookies = " ".join(body for _title, body in LEGAL_DOCUMENTS["cookies"]["sections"])
        self.assertNotIn("loads on every page", cookies)
        self.assertIn("not loaded at all until you choose Accept All", cookies)
        self.assertIn("never loaded on sign-in, account, upload", cookies)
        with setting("analytics_measurement_id", "GT-TEST"):
            google = dict(subprocessor_sections())["Google (analytics and advertising measurement)"]
        self.assertIn("only after you accept analytics", google)
        self.assertIn("never loads on sign-in", google)


if __name__ == "__main__":
    unittest.main()
