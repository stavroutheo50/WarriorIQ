"""Fixes from the site walkthrough of 19 Oct 2026, numbered as reported.

1  Fight Camp leaves out fights with too little fight footage to measure
2  an upload waiting for its fighters says it holds the analysis
3  the header links never wrap while an analysis badge shows
4  the badge speaks in the progress page's steps
5  the opponent's name on the selection and progress pages
6  the score once on the report
7  strike counts once on the report
8  no "measured and real" beside a guard that was not measured
9  "You vs your opponent" says which colour is who
10 the replay box describes both skeletons
11 a no-fight report keeps its notice inside the page; illegal moves in one line when off
12 library cards keep four columns; no stray empty box
13 the profile lists fights by title
14/15 a shorter homepage; accuracy on its own page
16 the signed-out sport chooser has an H1
17 no probe of the switched-off upload path
18 the Compare picker keeps its text column without a picture
"""

from __future__ import annotations

import time
from pathlib import Path

from browser_client import BrowserClient

import app.main as web
from core import progress_insights
from core.squad import summarize_fight

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app" / "templates"
CSS = (ROOT / "app" / "static" / "system.css").read_text(encoding="utf-8")


def _page(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------- 1
def _record(sufficient: bool) -> dict:
    return {"job_id": "x", "created_at": "2026-10-19", "report": {
        "video": {"focus_fighter": "A"},
        "integrity": {"identity_evidence_trusted": True, "fight_footage_sufficient": sufficient},
        "metrics": {"A": {"pose_coverage": 0.0 if not sufficient else 0.8, "balance_index": 0.7}},
    }}


def test_fight_camp_leaves_out_a_fight_with_no_fight_footage():
    progress = progress_insights.build_progress([_record(False)], "A")
    assert progress["fight_count"] == 0 and progress["latest"] is None
    assert progress_insights.build_progress([_record(True)], "A")["fight_count"] == 1


def test_the_squad_table_marks_it_as_no_fight_footage():
    report = {"integrity": {"identity_evidence_trusted": True, "fight_footage_sufficient": False},
              "tracking": {"fighter_A_coverage": 1.0, "fighter_B_coverage": 1.0},
              "video": {"focus_fighter": "A"}, "metrics": {"A": {}, "B": {}}, "scorecard": {}}
    row = summarize_fight(report, {"job_id": "x", "ruleset": "K1", "created_at": "2026-10-19"})
    assert row["usable"] is False and row["unusable_reason"] == "no_fight"
    assert "No fight footage to measure" in _page("camp.html")


# ---------------------------------------------------------------------- 2
def test_an_upload_waiting_for_fighters_says_it_holds_the_analysis():
    page = _page("analyze.html")
    assert "Your analysis is held by <strong>{{ held_upload.name }}</strong>" in page
    assert "the analysis it was holding is given back" in _page("select.html")
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "Your analysis is being held by {held['name']}" in source


# ------------------------------------------------------------------ 3 4
def test_header_links_never_wrap():
    assert ".navlinks>a,.nav-more>summary{white-space:nowrap}" in CSS
    assert ".account-nav:has(.active-analysis-chip) .profile-email{display:none}" in CSS


def test_the_badge_uses_the_progress_page_steps():
    assert web.analysis_chip_label({"status": "running", "stage": "analysis", "stage_fraction": 0.17}) \
        == "Analysing step 3/4 · 17%"
    job = {"status": "running", "stage": "analysis", "stage_fraction": 0.17, "queued_at_epoch": time.time(),
           "video_duration": 10.0}
    assert web._public_job_status("x", job)["chip_label"] == "Analysing step 3/4 · 17%"
    assert "d.chip_label||" in _page("base.html")


# ---------------------------------------------------------------------- 5
def test_the_opponent_is_named_on_selection_and_progress():
    select = _page("select.html")
    assert "const opponentName={{ (opponent_name or '')|tojson }};" in select
    assert "(opponentName||'Opponent')" in select
    progress = _page("progress.html")
    assert "const fighterNames={{ fighter_names|default({}, true)|tojson }};" in progress
    assert "Fighter ${event.fighter}" not in progress


# ------------------------------------------------------------- 6 7 8 9 10
def test_the_report_shows_score_and_strike_counts_once():
    page = _page("result.html")
    assert 'class="numbers-score"' not in page
    assert "{% elif observed and not strip_shown %}" in page
    assert "{% if strike_counts_published and not strip_shown %}" in page


def test_no_claim_that_unmeasured_numbers_are_real():
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "measured and real" not in source


def test_the_bars_say_which_colour_is_who_and_the_replay_copy_is_current():
    page = _page("result.html")
    assert 'class="h2h-legend small"' in page and "(left)" in page and "(right)" in page
    assert "The overlay follows only the athlete selected" not in page
    assert "Both fighters&rsquo; skeletons are drawn" in page


# --------------------------------------------------------------------- 11
def test_the_deep_dive_closes_only_where_it_opened():
    page = _page("result.html")
    assert "{% if numbers.state != 'no_fight' %}</div></details>{% endif %}" in page
    assert 'class="small muted illegal-moves-off"' in page


# ------------------------------------------------------------------ 12 13
def test_library_cards_and_profile_titles():
    assert ".fight-record>.fight-select{position:absolute" in CSS
    assert ".archive-no-results[hidden]{display:none}" in CSS
    assert "<strong>Fight analysis</strong>" not in _page("profile.html")
    assert "{{ f|fight_title }}" in _page("profile.html")


# ------------------------------------------------------------------ 14 15
def test_the_homepage_is_shorter_and_accuracy_has_its_own_page():
    home = BrowserClient(web.app).get("/").text
    assert home.count("<section") <= 7
    for gone in ("Your next round starts here.", "Evidence before claims.", 'id="accuracy"'):
        assert gone not in home
    accuracy = BrowserClient(web.app).get("/accuracy")
    assert accuracy.status_code == 200 and "11 of 16" in accuracy.text
    assert accuracy.text.count("<h1") == 1


# ------------------------------------------------------------- 16 17 18
def test_the_signed_out_sport_chooser_has_one_h1():
    page = BrowserClient(web.app).get("/analyze").text
    assert page.count("<h1") == 1


def test_the_switched_off_upload_path_is_not_probed():
    page = _page("analyze.html")
    assert "const CHUNKED_ENABLED = {{ 'true' if chunked_upload_enabled else 'false' }};" in page
    assert "if (!CHUNKED_ENABLED) throw FALLBACK;" in page


def test_the_compare_picker_keeps_its_text_column():
    assert ".compare-option-text{grid-column:3}" in CSS
