"""Owner requests, 2026-11-13.

- Menus and pop-ups were see-through: they are solid now.
- Choosing the fighter is its own upload step, after the rules.
- The live analysis page fits one screen and shows what has been measured
  so far: video analysed, fighters in view, round, strike estimate.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLASS = (ROOT / "app" / "static" / "glass.css").read_text(encoding="utf-8")
ANALYZE = (ROOT / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
PROGRESS = (ROOT / "app" / "templates" / "progress.html").read_text(encoding="utf-8")


def _last_rule_for(selector: str) -> str:
    """The body of the last rule whose selector list names `selector`."""
    bodies = [body for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", GLASS)
              if selector in [s.strip() for s in selectors.split(",")]]
    assert bodies, selector
    return bodies[-1]


def test_menus_and_popups_are_solid():
    for selector in (".profile-menu-panel", ".mobile-menu", ".nav-more-panel", ".cookie-notice", ".app-toast"):
        body = _last_rule_for(selector)
        assert "var(--glass-fill-solid) !important" in body, selector
        assert "backdrop-filter: none !important" in body, selector
    solid = re.search(r"--glass-fill-solid:\s*([^;]+);", GLASS).group(1)
    assert "rgba" not in solid, "the solid fill must not be transparent"
    assert "select option" in GLASS


def test_fighter_choice_is_its_own_step_after_the_rules():
    assert "const stepOrder=['video','rules','who','check'];" in ANALYZE
    steps = re.search(r'<ol class="upload-steps"[^>]*>(.*?)</ol>', ANALYZE).group(1)
    assert re.findall(r'data-step="(\w+)"', steps) == ["video", "rules", "who", "check"]
    assert ">Fighter</li>" in steps and ">Rules</li>" in steps
    rules = ANALYZE.index('data-upload-step="rules"')
    assert ANALYZE.index('id="fightRuleset"', rules) < ANALYZE.index('data-upload-step="who"')
    # The fighter questions stay on the "who" step.
    for field in ('id="fighterName"', 'id="fighterPick"'):
        at = ANALYZE.index(field)
        assert ANALYZE.rfind('data-upload-step="who"', 0, at) > ANALYZE.rfind('data-upload-step="rules"', 0, at)


def test_live_page_shows_measured_facts_and_folds_the_step_list():
    for fact in ("factAnalysed", "factInView", "factRound"):
        assert f'id="{fact}"' in PROGRESS
    assert 'id="factStrikes"' in PROGRESS and "<small>estimate</small>" in PROGRESS
    assert "function renderFacts(d)" in PROGRESS and "latestData=d;renderFacts(d);" in PROGRESS
    # Video analysed moves only in the reading pass, never during tracking.
    assert "['analysis','report'].includes(d.stage)" in PROGRESS
    assert '<details class="analysis-process-card live-steps">' in PROGRESS
    assert 'id="analysisTimeline" aria-label="Detected event timeline"{% if not live_moments %} hidden{% endif %}' in PROGRESS
