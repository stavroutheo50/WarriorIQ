"""Library search finds a fight by its date however it is typed.

QA, 2026-10-07: cards displayed "OCT 7 2026" but only "7 oct" matched; "oct
7", "2026-10-07" and "october" returned nothing. Verified in Chromium with a
seeded library: every form below now matches, nonsense shows the empty state
with "Clear search", and clearing restores "1-12 of 14 fights".
"""

from __future__ import annotations

from pathlib import Path

from app.main import _date_search_terms

PAGE = (Path(__file__).resolve().parents[1] / "app" / "templates" / "history.html").read_text(encoding="utf-8")


def test_the_stored_date_is_indexed_in_every_common_form():
    terms = _date_search_terms("2026-10-07T13:59:00+00:00")
    for typed in ("7 oct 2026", "oct 7", "october 7 2026", "7 october", "october", "2026-10-07",
                  "07/10/2026", "10/07/2026", "7/10/2026", "7.10.2026"):
        assert typed in terms, typed


def test_a_bad_date_indexes_nothing():
    assert _date_search_terms("not a date") == ""
    assert _date_search_terms(None) == ""


def test_the_card_indexes_the_terms_and_the_date_as_the_browser_shows_it():
    assert "f.created_at|date_search_terms" in PAGE
    # The viewer's own time zone and the card's displayed text are added.
    assert "toLocaleDateString(loc,opts)" in PAGE and "card.querySelector('time')" in PAGE
    # "Oct 7, 2026": commas and repeated spaces do not stop a match.
    assert "replace(/,/g,' ')" in PAGE and "const query=normal(search.value)" in PAGE


def test_no_results_never_prints_a_one_to_zero_range():
    assert "if(!matches.length){pager.hidden=true;pageInfo.textContent='';return}" in PAGE
    assert 'id="historyNoResultsAction"' in PAGE
