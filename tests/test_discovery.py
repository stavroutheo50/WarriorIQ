"""Being found by search engines and AI tools (app/discovery.py, 2026-11-25)."""

from __future__ import annotations

import json
import re

import app.main as web
from app import discovery
from browser_client import BrowserClient


def test_llms_txt_describes_warrioriq_with_links_and_no_fight_statistics(monkeypatch):
    text = discovery.llms_txt("https://warrioriq.eu")
    assert text.startswith("# WarriorIQ\n\n> ")
    for path in ("/analyze", "/pricing", "/scorer", "/privacy", "/ai-transparency"):
        assert f"https://warrioriq.eu{path})" in text
    assert "estimates" in text                                 # says what it cannot do yet
    assert not re.search(r"\d+(\.\d+)?%", text)                # no percentages of any kind
    page = BrowserClient(web.app).get("/llms.txt")
    assert page.status_code == 200 and page.text.startswith("# WarriorIQ")


def test_the_indexnow_key_is_made_once_and_served(tmp_path, monkeypatch):
    first = discovery.indexnow_key(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{32}", first) and discovery.indexnow_key(tmp_path) == first
    monkeypatch.setattr(web, "DATA_ROOT", tmp_path)
    assert BrowserClient(web.app).get("/indexnow-key.txt").text == first


def test_indexnow_is_told_every_page_with_the_key_location():
    sent = {}

    class Response:
        status = 202
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    def opener(request, timeout):
        sent["url"], sent["body"] = request.full_url, json.loads(request.data)
        return Response()

    result = discovery.submit_to_indexnow("https://warrioriq.eu", ("/", "/pricing"), "k" * 32, opener=opener)
    assert result == {"ok": True, "status": 202, "count": 2}
    assert sent["url"] == discovery.INDEXNOW_ENDPOINT
    assert sent["body"] == {"host": "warrioriq.eu", "key": "k" * 32,
                            "keyLocation": "https://warrioriq.eu/indexnow-key.txt",
                            "urlList": ["https://warrioriq.eu/", "https://warrioriq.eu/pricing"]}
    assert not discovery.submit_to_indexnow("http://127.0.0.1:8765", ("/",), "k" * 32, opener=opener)["ok"]


def test_only_an_administrator_can_send_it():
    assert BrowserClient(web.app).post("/admin/indexnow", follow_redirects=False).status_code in {403, 404}


def test_the_bing_ownership_tag_renders_when_set(monkeypatch):
    monkeypatch.setenv("WARRIORIQ_BING_VERIFICATION", "ABC123")
    assert '<meta name="msvalidate.01" content="ABC123">' in BrowserClient(web.app).get("/").text
    monkeypatch.delenv("WARRIORIQ_BING_VERIFICATION")
    assert "msvalidate.01" not in BrowserClient(web.app).get("/").text


def test_robots_still_lets_every_crawler_in():
    previous = web.SETTINGS.public_base_url
    object.__setattr__(web.SETTINGS, "public_base_url", "https://warrioriq.eu")
    try:
        text = BrowserClient(web.app).get("/robots.txt").text
    finally:
        object.__setattr__(web.SETTINGS, "public_base_url", previous)
    assert "User-agent: *" in text and "Allow: /" in text
    for bot in ("GPTBot", "ClaudeBot", "PerplexityBot", "Google-Extended", "CCBot"):
        assert bot not in text                                  # none singled out to be blocked
