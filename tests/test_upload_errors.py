"""Upload errors look like errors and speak the device's language (QA, 2026-10-04).

Checked in Chromium: a .txt picked on desktop reads "Pick the clip from your
files." in the danger colour (rgb(239,120,145)); on an iPhone 13 emulation it
says "your camera roll".
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_a_rejected_file_line_has_the_error_colour():
    css = (ROOT / "app" / "static" / "product.css").read_text(encoding="utf-8")
    assert ".file-status[data-tone=bad]{color:var(--wiq-danger" in css


def test_camera_roll_only_on_touch_devices():
    page = (ROOT / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
    assert "matchMedia('(pointer: coarse)').matches)?'your camera roll':'your files'" in page
    assert "Pick the clip from your camera roll" not in page
