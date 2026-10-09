"""One visible "Pick a different moment" link, not two (QA, 2026-10-04).

On an automatically chosen frame the note under the heading carries it. The
other copy now lives under "More options" on the who-is-who page (2026-10-09),
folded away, so the two are never on screen together.
"""

from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "app" / "templates" / "select.html").read_text(encoding="utf-8")


def test_the_second_link_waits_under_more_options():
    more = PAGE.split('<details class="select-more"', 1)[1].split("</details>", 1)[0]
    assert 'id="pickMomentLink" href="/frame/{{job_id}}"' in more
    header = PAGE.split('<header class="workflow-heading">', 1)[1].split("</header>", 1)[0]
    assert "pickMomentLink" not in header


def test_the_link_is_never_left_hidden_when_the_note_goes():
    assert "note.hidden=true;document.getElementById('pickMomentLink').hidden=false;" in PAGE
