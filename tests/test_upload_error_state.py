"""A failed upload's messages clear when a new file is chosen, and the
"keep this tab open" hint never sits under an error (QA, 2026-10-07).

Checked in Chromium while fixing it: before, the red progress strip and the
hint stayed up over the newly chosen file. These pin the two code paths."""

from __future__ import annotations

import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")


def _function(name: str) -> str:
    start = PAGE.index(f"const {name}=")
    return PAGE[start:PAGE.index("};", start) + 2]


def test_the_in_flight_hint_is_hidden_in_the_error_state():
    show = _function("showProgress")
    assert "uploadProgressHint" in show
    assert re.search(r"hint\.hidden\s*=\s*tone\s*===\s*'error'", show)


def test_choosing_a_file_clears_both_error_displays():
    hide = _function("hideUploadError")
    assert "uploadError.hidden=true" in hide
    assert "uploadProgress" in hide and "dataset.tone==='error'" in hide
    assert "videoInput.addEventListener('change',()=>{hideUploadError();" in PAGE


def test_a_failure_shows_the_error_state_through_show_progress():
    restore = _function("restore")
    assert "showProgress('error')" in restore


def test_a_rejected_file_cancels_the_footage_check_still_running_on_it():
    """QA, 2026-10-07: a 1-second clip showed "needs at least 2 seconds" and a
    quality panel saying the recording looked usable. Reproduced in Chromium:
    the duration check rejected the file, then the footage check on the same
    file finished and showed its verdict. Rejecting now invalidates it."""
    reject = PAGE[PAGE.index("const rejectFile="):]
    reject = reject[:reject.index("};") + 2]
    assert "preflightToken++" in reject
    preflight = PAGE[PAGE.index("const runPreflight="):]
    assert "if(token!==preflightToken)return;" in preflight
