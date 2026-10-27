"""A big video keeps its size advice once its length is known (2026-10-27).

The owner's 3-minute, 330 MB phone video took 11 minutes to upload. The page
had advice for files over 100 MB - roughly how long, and the camera setting
that makes the next one several times smaller - but the line reporting the
video's length replaced it a moment after it appeared.
"""
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")


def test_the_length_line_keeps_the_size_advice():
    handler = PAGE[PAGE.index("probe.onloadedmetadata="):PAGE.index("probe.onerror=unsure")]
    assert "const advice=sizeAdvice(f.size);" in handler
    assert "if(advice.tone!=='ok'){fileStatus.textContent=`${mins}:${secs} — ${advice.text}`" in handler


def test_the_advice_names_the_setting_that_makes_the_next_video_smaller():
    assert "Settings → Camera → Formats → High Efficiency" in PAGE
