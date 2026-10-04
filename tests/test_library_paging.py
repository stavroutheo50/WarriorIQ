"""The fight library pages twelve at a time (QA, 2026-10-04: 36 on one page).

Checked in Chromium at 390 px with 37 saved fights: 12 shown, Next moves to
13-24, a search covers every fight (not just the current page) and the pager
hides when one page holds the results.
"""

from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "app" / "templates" / "history.html").read_text(encoding="utf-8")


def test_the_pager_exists_and_is_hidden_until_needed():
    assert '<nav class="archive-pager" id="historyPager" aria-label="Fight library pages" hidden>' in PAGE
    assert 'id="historyPageInfo" aria-live="polite"' in PAGE


def test_paging_runs_after_search_filter_and_sort():
    script = PAGE.split("const apply=(keepPage)=>{", 1)[1]
    assert script.index("matches.push(item)") < script.index("paginate(matches)")
    assert "PAGE_SIZE=12" in PAGE


def test_a_new_search_starts_at_the_first_page():
    assert "if(keepPage!==true)page=0;" in PAGE
