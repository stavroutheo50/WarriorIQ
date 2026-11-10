"""Skeleton and box colours follow the corner chosen on the selection page
(owner, 2026-11-10: in the replay "the fighters were mixed up"). Fighter A is
whoever was boxed first; the report already showed a red-corner A in red, but
the replay, the live page and the selection boxes drew A blue regardless."""

from __future__ import annotations

import json
from pathlib import Path

import app.main as web

TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "templates"


def _line(name: str, marker: str) -> str:
    return next(line for line in (TEMPLATES / name).read_text(encoding="utf-8").splitlines() if marker in line)


def test_replay_colours_follow_the_corners():
    line = _line("replay.html", "const CORNER_COLOUR=")
    template = web.templates.env.from_string(line)
    def colours(job):
        rendered = template.render(corners=web._corner_labels(job))
        return json.loads(rendered.split("=", 1)[1].rstrip(";"))
    assert colours({"fighter_a_corner": "red"}) == {"A": "#ff6470", "B": "#55a7ff"}
    assert colours({"fighter_a_corner": "blue"}) == {"A": "#55a7ff", "B": "#ff6470"}
    assert colours({}) == {"A": "#55a7ff", "B": "#ff6470"}
    assert "CORNER_COLOUR.A" in _line("replay.html", "function draw(")


def test_live_page_and_selection_follow_the_corners():
    line = _line("progress.html", "drawSkeleton(observation?.fighters?.A")
    red = web.templates.env.from_string(line).render(a_red_corner=True)
    assert "drawSkeleton(observation?.fighters?.A?.keypoints,\"#ff6470\"" in red
    assert 'a-red-corner' in _line("progress.html", 'id="analysisPage"')
    assert "aIsRed()?'#ff6470':'#55a7ff'" in _line("select.html", "drawBox(boxA,")
