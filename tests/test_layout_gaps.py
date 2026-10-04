"""No large empty areas: under the selection canvas, between home sections.

QA, 2026-10-04. Measured in Chromium at 1280 px after the change: the
selection page's canvas column and sidebar are both 1076 px (the canvas column
used to end at the picture while the sidebar ran on), and the homepage's
section-to-section gap is 108 px instead of about 220.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_pick_list_and_rules_sit_under_the_canvas():
    page = (ROOT / "app" / "templates" / "select.html").read_text(encoding="utf-8")
    stage = page.split('<section class="fighter-stage-panel"', 1)[1].split("</section>", 1)[0]
    assert '<div class="stage-extras">' in stage
    assert 'id="pickFromList"' in stage and 'class="selection-rules"' in stage
    aside = page.split('<aside class="fighter-lock-guide"', 1)[1].split("</aside>", 1)[0]
    assert 'id="pickFromList"' not in aside


def test_home_sections_share_one_gap():
    css = (ROOT / "app" / "static" / "product.css").read_text(encoding="utf-8")
    assert ("body[data-page=home] main > .product-section,body[data-page=home] main > .trust-panel"
            "{margin-top:0;padding-top:clamp(44px,6vh,64px);padding-bottom:clamp(44px,6vh,64px)}") in css
