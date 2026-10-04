"""One "Pick a different moment" link, not two (QA, 2026-10-04).

Checked in Chromium: one visible link on an automatically chosen frame (in the
note) and one on a frame the person chose (in the header).
"""

from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "app" / "templates" / "select.html").read_text(encoding="utf-8")


def test_the_header_link_hides_while_the_note_carries_it():
    assert ('<a class="quiet-back" id="pickMomentLink" href="/frame/{{job_id}}"'
            "{% if frame_source in ['auto', 'auto_pair'] %} hidden{% endif %}>") in PAGE


def test_the_header_link_returns_when_the_note_is_hidden():
    assert "note.hidden=true;document.getElementById('pickMomentLink').hidden=false;" in PAGE
