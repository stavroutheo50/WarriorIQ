"""The share dialog fits the screen and opens at its top (QA, 2026-10-04).

Measured in Chromium before: opened already scrolled 368-404 px down
(showModal focused a control below the card), running to the bottom edge.
After, at 390x667, 1280x600, 1280x1100 and 360x560: 16 px from the top,
scrollTop 0, focus on the title, scrolling inside when taller than the screen.
"""

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


def test_the_dialog_is_capped_to_the_viewport_and_scrolls_inside():
    css = (STATIC / "components.css").read_text(encoding="utf-8")
    rule = css.split(".share-card-dialog{", 1)[1].split("}", 1)[0]
    for part in ("max-height:calc(100dvh - 32px)", "inset:0", "margin:auto", "overflow-y:auto"):
        assert part in rule, part


def test_it_opens_at_the_top_on_its_title():
    js = (STATIC / "share_card.js").read_text(encoding="utf-8")
    opener = js.split('open.addEventListener("click", function () {', 1)[1].split("dialog.scrollTop = 0;", 1)[0] + "dialog.scrollTop = 0;"
    assert "dialog.showModal();" in opener
    assert opener.index("dialog.showModal();") < opener.index("dialog.scrollTop = 0;")
    assert 'title.focus({ preventScroll: true })' in opener
