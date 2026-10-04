"""The reason Start is disabled sits next to the button (QA, 2026-10-04).

It was only in the status line at the top of the sidebar. Checked in
Chromium: the reason renders 6 px under the button at 1280 and 390 px, and
follows the selection - "Box Fighter A", "Now box Fighter B", "The two boxes
cover the same person", and nothing once Start is enabled.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "app" / "templates" / "select.html").read_text(encoding="utf-8")


def test_the_reason_is_inside_the_start_column_and_announced():
    wrap = PAGE.split('<div class="start-wrap">', 1)[1].split("</div>", 1)[0]
    assert 'id="start"' in wrap and 'id="startReason"' in wrap
    assert 'aria-describedby="startReason"' in PAGE
    assert 'role="status" aria-live="polite"' in wrap


def test_every_disabled_state_has_a_reason():
    function = PAGE.split("function startReasonText(valid){", 1)[1].split("\n}", 1)[0]
    for reason in ("Box Fighter A to start.", "Now box Fighter B to start.",
                   "Box the person training to start.", "cover the same person"):
        assert reason in function
    assert "startReason.textContent=startReasonText(valid)" in PAGE


def test_a_solo_selection_does_not_read_fighter_b():
    assert "detectorMatch(boxA)&&(solo||detectorMatch(boxB))" in PAGE
