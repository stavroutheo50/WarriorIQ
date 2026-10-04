"""Selection page layout with long filed fighter names (QA, 2026-10-04).

A 48-character name squeezed "Which one is ...?" to one word per line: the
copy column was minmax(0,1fr) beside an `auto` column holding two radio
buttons that each repeated the full name. Measured in Chromium before the fix
the copy column was 6 px wide at 1280; after it, 411 px (2 lines) at 1280 and
the full width below 1120.
"""

from pathlib import Path

CSS = (Path(__file__).resolve().parents[1] / "app" / "static" / "frame-picker.css").read_text(encoding="utf-8")


def test_the_question_column_has_a_floor():
    assert ".analysis-choice{display:grid;grid-template-columns:minmax(260px,1fr) minmax(0,auto)" in CSS


def test_the_name_buttons_cannot_take_the_whole_row():
    assert ".analysis-choice .target-options label{min-width:104px;max-width:220px}" in CSS
    assert "overflow-wrap:anywhere" in CSS


def test_it_stacks_before_the_question_gets_narrow():
    assert "@media(max-width:1120px){.analysis-choice{grid-template-columns:1fr}" in CSS
