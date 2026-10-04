"""The header links do not slide when the "Analyzing" pill appears (QA, 2026-10-04).

Measured in Chromium at 1280: with the old space-between row the links moved
60 px left when the pill was added (422 -> 362); with the grid they stay at
441 at 1280 and 301 at 1000.
"""

from pathlib import Path

CSS = (Path(__file__).resolve().parents[1] / "app" / "static" / "product.css").read_text(encoding="utf-8")


def test_the_desktop_header_has_equal_side_columns():
    assert (".nav{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center}"
            ".nav>.brand{justify-self:start}.nav>.account-nav{justify-self:end}\n"
            "@media(max-width:900px){.nav{display:flex}}") in CSS
