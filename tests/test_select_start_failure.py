"""A failed start clears "Starting the analysis..." (QA, 2026-10-07).

Reproduced in Chromium: with /api/start answering 400 the error showed and
the line under the button kept saying "Starting the analysis...". The catch
branch now refreshes that line, and only that line, so the error stays."""

from __future__ import annotations

from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "app" / "templates" / "select.html").read_text(encoding="utf-8")


def test_the_failure_branch_refreshes_the_start_reason():
    start = PAGE.index("async function startAnalysis(){")
    body = PAGE[start:PAGE.index("\n", start)]
    failure = body[body.index("catch(error){"):]
    assert failure.index("starting=false") < failure.index("startReason.textContent=startReasonText(")
    assert "startReason.hidden=!startReason.textContent" in failure
    # Not a full redraw, which would overwrite the error in the status box.
    assert "draw()" not in failure


def test_the_reason_text_says_starting_only_while_starting():
    assert "if(starting)return 'Starting the analysis…';" in PAGE
