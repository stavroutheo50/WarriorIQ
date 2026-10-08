"""The site's link preview is a 1200x630 card (QA, 2026-10-07, item 25).

og:image was the bare square logo and twitter:card "summary", so shared
links showed a small thumbnail.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.main as webapp
from core.share_image import SITE_CARD_LINES, site_card_png

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "app" / "static" / "warrioriq-social-card.png"


@contextmanager
def setting(name, value):
    previous = getattr(webapp.SETTINGS, name)
    object.__setattr__(webapp.SETTINGS, name, value)
    try:
        yield
    finally:
        object.__setattr__(webapp.SETTINGS, name, previous)


class SocialCardTests(unittest.TestCase):
    def test_committed_card_is_1200_by_630(self):
        image = cv2.imread(str(CARD))
        self.assertIsNotNone(image)
        self.assertEqual(image.shape[:2], (630, 1200))

    def test_generator_draws_the_same_size(self):
        logo = cv2.imread(str(ROOT / "app" / "static" / "warrioriq-logo-512.png"), cv2.IMREAD_UNCHANGED)
        drawn = cv2.imdecode(np.frombuffer(site_card_png(logo), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(drawn.shape[:2], (630, 1200))

    def test_card_words_claim_only_what_reports_contain(self):
        text = " ".join(SITE_CARD_LINES).lower()
        for word in ("score", "strike", "punch", "kick", "judge", "illegal"):
            self.assertNotIn(word, text)

    def test_pages_point_at_the_card_with_large_image(self):
        with setting("public_base_url", "https://warrioriq.eu"):
            page = TestClient(webapp.app, base_url="https://warrioriq.eu").get("/").text
        self.assertIn('property="og:image" content="https://warrioriq.eu/static/warrioriq-social-card.png"', page)
        self.assertIn('property="og:image:width" content="1200"', page)
        self.assertIn('property="og:image:height" content="630"', page)
        self.assertIn('name="twitter:card" content="summary_large_image"', page)
        self.assertIn('name="twitter:image" content="https://warrioriq.eu/static/warrioriq-social-card.png"', page)

    def test_without_a_public_address_no_image_and_plain_summary(self):
        with setting("public_base_url", ""):
            page = TestClient(webapp.app).get("/").text
        self.assertNotIn('property="og:image"', page)
        self.assertIn('name="twitter:card" content="summary"', page)

    def test_card_is_served(self):
        response = TestClient(webapp.app).get("/static/warrioriq-social-card.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")


if __name__ == "__main__":
    unittest.main()
