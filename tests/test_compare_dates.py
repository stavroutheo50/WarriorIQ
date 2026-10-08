"""Compare: one date formatter in the reader's zone, and a real reason for solo.

QA, 2026-10-07: the result header read "7 Oct, 13:59" (UTC) over a picker
reading "Oct 7 2026, 04:59 PM" (local), and comparing a solo session with a
fight said "Choose two different saved fights". Checked in Chromium
(Europe/Athens): headers and picker now read the same local time, and the
solo case says why.
"""

from __future__ import annotations

from pathlib import Path

from core.squad import fight_choice_stamp

ROOT = Path(__file__).resolve().parents[1] / "app" / "templates"
BASE = (ROOT / "base.html").read_text(encoding="utf-8")
COMPARE = (ROOT / "compare.html").read_text(encoding="utf-8")


def test_every_relabel_uses_the_one_formatter():
    assert "window.wiqLocalTime=" in BASE
    relabel = BASE[BASE.index("window.wiqLocalTime="):BASE.index("// The header chip was rendered once")]
    assert relabel.count("toLocaleString(") == 1
    assert "el.textContent=window.wiqLocalTime(when,el.dataset.local!=='date')" in relabel
    assert "el.textContent.replace(stamp,window.wiqLocalTime(when))" in relabel


def test_a_stamp_left_unswapped_still_says_it_is_utc():
    assert fight_choice_stamp("2026-10-07T13:59:00+00:00") == "7 Oct, 13:59 UTC"


def test_a_solo_session_is_named_as_the_reason():
    assert "Solo sessions cannot be compared with fights" in COMPARE
    # The page's own script used to overwrite it on load.
    assert "if(serverNotice)return true;" in COMPARE


def test_the_compare_route_detects_a_solo_pick():
    from fastapi.testclient import TestClient

    import app.main as webapp

    source = Path(webapp.__file__).read_text(encoding="utf-8")
    assert '"solo_picked": bool(solo_ids)' in source
