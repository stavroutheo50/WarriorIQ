"""The button and motion layer the owner chose from a live mockup (2026-11-24).

B ("solid premium") for buttons, A ("real glass") over a photo or video, a
sliding menu highlight, cards that lift, pages that fade into each other and
numbers that count up. These pin what made the old buttons dead, so it does
not come back.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"


def test_the_layer_loads_last_before_the_accessibility_floors():
    from app.main import CSS_BUNDLES

    shell = CSS_BUNDLES["shell"]
    assert shell[-2:] == ("interaction.css", "a11y.css")


def test_nothing_cancels_the_hover_lift_or_glow_any_more():
    redesign = (STATIC / "redesign.css").read_text(encoding="utf-8")
    assert ".btn:hover {  transform: none" not in redesign
    assert ".btn, .btn:hover, .nav-signup, .wiq-primary-motion { box-shadow: none !important; }" not in redesign
    assert ".btn { box-shadow: inset 0 -1px 0 rgba(0, 0, 0, .25) !important; }" not in redesign


def test_buttons_animate_what_actually_changes_and_never_clip_their_label():
    css = (STATIC / "interaction.css").read_text(encoding="utf-8")
    for prop in ("transform", "box-shadow", "filter", "background-color"):
        assert prop + " .22s" in css
    # overflow:hidden let a flex row shrink "View analysis" into a circle on phones.
    block = css.split(".btn, a.btn, button.btn{", 1)[1].split("}", 1)[0]
    assert "overflow" not in block
    assert ".btn:active{transform:translateY(0) scale(.96)" in css
    assert "prefers-reduced-motion: reduce" in css


def test_glass_is_only_for_buttons_over_imagery():
    css = (STATIC / "interaction.css").read_text(encoding="utf-8")
    assert ".on-media .btn:not(.secondary):not(.danger), .btn.btn-glass{" in css
    assert "backdrop-filter:blur(16px)" in css


def test_pages_fade_and_the_menu_highlight_slides():
    css = (STATIC / "interaction.css").read_text(encoding="utf-8")
    assert "@view-transition{navigation:auto}" in css and "view-transition-name:wiq-nav-pill" in css
    js = (STATIC / "motion.js").read_text(encoding="utf-8")
    assert "pill.className='nav-pill'" in js and "duration:900" in js


def test_the_home_page_no_longer_offers_two_step_sign_in():
    page = (ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "Two-step sign-in" not in page and "2FA" not in page
