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
    assert 'class="sport-card-photo"' in picker and "CC BY 2.0" in picker
    assert 'class="setup-clip"' in setup and "data-home-clip" in setup and "/static/home.js" in setup
    assert "sport == 'muay_thai'" in setup and "CC BY 2.0" in setup
