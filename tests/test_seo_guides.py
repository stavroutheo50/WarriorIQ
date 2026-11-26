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


def test_the_boxing_taekwondo_and_sparring_guides_exist_and_are_honest():
    # Owner, 2026-11-26: the competitor teardown found no WarriorIQ page for
    # boxing, taekwondo or "AI sparring analysis" searches.
    client = BrowserClient(web.app)
    for path in ("/boxing-fight-analysis", "/taekwondo-fight-analysis", "/ai-sparring-analysis"):
        page = client.get(path)
        assert page.status_code == 200 and "<h1" in page.text, path
        assert path in web.PUBLIC_INDEX_ROUTES
        assert path in client.get("/").text            # linked from the home page's guides
        assert path in client.get("/llms.txt").text    # and described for AI assistants
    guide = web.SEARCH_GUIDES["taekwondo-fight-analysis"]
    text = " ".join([guide["intro"], *(section["body"] for section in guide["sections"])]).lower()
    for item in sport_unobserved("taekwondo"):
        words = [word for word in item.lower().replace(",", " ").split() if len(word) > 4]
        assert any(word in text for word in words), item
    sparring = web.SEARCH_GUIDES["ai-sparring-analysis"]
    assert f"{web.SETTINGS.saved_video_retention_days} days" in sparring["sections"][3]["body"]
