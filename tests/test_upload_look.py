"""The upload pages in the home page's look (owner, 2026-11-09)."""

from __future__ import annotations

from pathlib import Path

from core.scoring import SPORTS

ROOT = Path(__file__).resolve().parents[1]


def test_every_sport_has_its_still_and_clip():
    for sport in SPORTS:
        name = sport.replace("_", "")
        assert (ROOT / "app" / "static" / "home" / f"{name}.jpg").is_file()
        assert (ROOT / "app" / "static" / "home" / f"{name}.mp4").is_file()


def test_the_templates_use_them_and_credit_cc_by():
    picker = (ROOT / "app" / "templates" / "sports.html").read_text(encoding="utf-8")
    setup = (ROOT / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
    # The Muay Thai and kickboxing clips are Pexels since 2026-11-13 (owner:
    # real sparring in place of gym pad work); every clip is still credited.
    assert 'class="sport-card-photo"' in picker and "Pexels (Pexels License)" in picker
    assert 'class="setup-clip"' in setup and "data-home-clip" in setup and "/static/home.js" in setup
    assert "sport == 'muay_thai'" in setup and "Pexels (Pexels License)" in setup


def test_the_scorer_head_plays_the_ringside_clip_and_credits_it():
    page = (ROOT / "app" / "templates" / "ringside.html").read_text(encoding="utf-8")
    assert 'class="ringside-head-clip" data-home-clip' in page and "/static/home.js" in page
    # Men since 2026-11-10 (owner): Mixkit 40967, not the CC BY women's bout.
    assert "Mixkit" in page and "Jorina" not in page
