"""Muay Thai and MMA guide pages (owner's brief, 2026-11-14): only what the analysis does."""

from __future__ import annotations

import app.main as web
from browser_client import BrowserClient
from core.sport_policy import sport_unobserved


def test_the_new_guides_are_public_indexed_and_in_the_sitemap():
    client = BrowserClient(web.app)
    for path in ("/muay-thai-fight-analysis", "/mma-fight-analysis"):
        page = client.get(path)
        assert page.status_code == 200 and "<h1" in page.text
        assert path in web.PUBLIC_INDEX_ROUTES


def test_each_guide_names_what_its_sport_does_not_observe():
    for slug, sport in (("muay-thai-fight-analysis", "muay_thai"), ("mma-fight-analysis", "mma")):
        guide = web.SEARCH_GUIDES[slug]
        text = " ".join([guide["intro"], *(section["body"] for section in guide["sections"])]).lower()
        for item in sport_unobserved(sport):
            words = [word for word in item.lower().replace(",", " ").split() if len(word) > 4]
            assert any(word in text for word in words), (slug, item)
