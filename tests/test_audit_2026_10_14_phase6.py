"""Audit of 14 Oct 2026, phase 6: report, replay, library, Compare, Fight Camp.

#26 a replay chapter brings the video into view
#27 both skeletons are drawn, with a marker where a fighter is not tracked
#29 one H1 on the Compare, report and replay pages
#30 optional event and opponent names; titles "Sport · Date", never the file name
#31 the fighters' names everywhere, including who is ahead
#33 library filters and bulk delete
#34 fights that failed the identity check are marked and not compared by default
#36 a comparison has its own address and lands on the results
#37 the Compare picker shows picture, title, date, ruleset, score and status
#42 /dashboard is a permanent redirect to /camp
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest
from browser_client import BrowserClient

import app.main as web
from core import db as database

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app" / "templates"
FIXTURE = ROOT / "tests" / "fixtures" / "report_sample.json"


def _signed_in():
    client = BrowserClient(web.app)
    email = f"p6-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/signup", data={"email": email, "password": "Strong-Local-Password", "accept_terms": "true",
                                 "age_confirmed": "true"}, follow_redirects=False)
    profile_id = int(database.get_account_by_email(email)["profile_id"])
    return client, profile_id


def _saved_fight(profile_id, *, failed=False, name="3.mp4", **details):
    job_id = "p6" + uuid.uuid4().hex[:10]
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if failed:
        report["integrity"]["identity_evidence_trusted"] = False
    directory = web.OUTPUTS / job_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    database.save_fight(job_id, profile_id, name, str(web.UPLOADS / f"{job_id}.mp4"), str(directory / "report.json"),
                        "sparring", "K1", "A", {"progress_report": {"integrity": report["integrity"]}},
                        "2999-01-01T00:00:00+00:00", **details)
    return job_id


# --------------------------------------------------------------- #26 #27
def test_a_chapter_brings_the_video_into_view():
    page = (TEMPLATES / "replay.html").read_text(encoding="utf-8")
    assert "function revealVideo()" in page
    assert "document.querySelectorAll('#timeline button').forEach(b=>b.onclick=()=>{revealVideo();" in page


def test_both_skeletons_are_drawn_with_a_marker_for_a_lost_fighter():
    page = (TEMPLATES / "replay.html").read_text(encoding="utf-8")
    assert "for(const[key,label,color]of[['fighter_A','A','#55a7ff'],['fighter_B','B','#f1b84b']])" in page
    assert "not tracked here" in page and "not tracked at this moment" in page
    assert "if(target==='A'||target==='BOTH')skeleton(" not in page


# --------------------------------------------------------------------- #29
@pytest.mark.parametrize("name", ["compare.html", "replay.html", "result.html"])
def test_one_h1_per_page(name):
    assert (TEMPLATES / name).read_text(encoding="utf-8").count("<h1") == 1


def test_the_compare_page_renders_one_h1():
    client, _ = _signed_in()
    assert client.get("/compare").text.count("<h1") == 1


# --------------------------------------------------------------- #30 #31
def test_titles_are_the_event_or_sport_and_date_never_the_file_name():
    from core.squad import fight_title

    assert fight_title({"ruleset": "K1", "created_at": "2026-10-14T10:00:00+00:00", "original_name": "3.mp4"}) \
        == "Kickboxing · 14 Oct 2026"
    assert fight_title({"event_name": "Club championship", "ruleset": "K1"}) == "Club championship"


def test_the_library_uses_titles_and_names():
    client, profile_id = _signed_in()
    named = _saved_fight(profile_id, event_name="Club championship", opponent_name="Nikos")
    plain = _saved_fight(profile_id)
    page = client.get("/history").text
    assert "Club championship" in page and "Nikos" in page
    assert re.search(r'class="fight-title"><strong>Kickboxing · \d+ \w{3} \d{4}</strong>', page)
    assert f'id="fight-{named}"' in page and f'id="fight-{plain}"' in page


def test_names_can_be_edited_only_by_the_owner():
    client, profile_id = _signed_in()
    job_id = _saved_fight(profile_id)
    response = client.post(f"/fights/{job_id}/details", data={"event_name": "  Derby  ", "opponent_name": "Maria"},
                           follow_redirects=False)
    assert response.status_code == 303
    fight = database.get_fight(job_id)
    assert (fight["event_name"], fight["opponent_name"]) == ("Derby", "Maria")
    stranger, _ = _signed_in()
    assert stranger.post(f"/fights/{job_id}/details", data={"event_name": "x"}).status_code == 404


def test_who_is_ahead_is_named():
    page = (TEMPLATES / "history.html").read_text(encoding="utf-8")
    assert "{{ nm.get(card.score.winner, 'Fighter ' ~ card.score.winner) }} ahead (estimate)" in page
    names = web._library_names({"analysis_target": "A", "fighter_name": "Theo", "opponent_name": "Nikos"})
    assert (names["A"], names["B"]) == ("Theo", "Nikos")
    assert web._library_names({"analysis_target": "B", "fighter_name": "Theo"})["A"] == "Opponent"


def test_the_upload_form_asks_for_the_names_optionally():
    page = (TEMPLATES / "analyze.html").read_text(encoding="utf-8")
    assert 'name="event_name"' in page and 'name="opponent_name"' in page


def test_the_schema_change_is_backward_compatible():
    database.init_db()
    with database.connection() as con:
        columns = {row[1]: row for row in con.execute("PRAGMA table_info(fights)")}
    assert columns["event_name"][3] == 0 and columns["opponent_name"][3] == 0     # nullable


# --------------------------------------------------------------------- #33
def test_bulk_delete_removes_only_the_owners_fights():
    client, profile_id = _signed_in()
    mine = [_saved_fight(profile_id) for _ in range(2)]
    stranger, other_profile = _signed_in()
    theirs = _saved_fight(other_profile)
    response = client.post("/history/delete", data={"job_id": [*mine, theirs]}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/history?deleted=2"
    assert all(database.get_fight(job_id) is None for job_id in mine)
    assert database.get_fight(theirs) is not None


def test_the_library_offers_filters_and_bulk_delete():
    page = (TEMPLATES / "history.html").read_text(encoding="utf-8")
    for control in ('id="historyStatus"', 'data-history-select="sport"', 'data-history-select="ruleset"',
                    'id="bulkDeleteForm"', 'form="bulkDeleteForm" data-bulk-pick'):
        assert control in page


# ------------------------------------------------------------ #34 #36 #37
def test_a_failed_fight_is_marked_and_not_compared_by_default():
    client, profile_id = _signed_in()
    good = _saved_fight(profile_id)
    failed = _saved_fight(profile_id, failed=True)
    page = client.get(f"/compare?a={good}&b={failed}").text
    assert "failed the identity check" in page
    assert re.search(rf'value="{failed}"[^>]*disabled', page)
    assert 'id="results"' not in page
    included = client.get(f"/compare?a={good}&b={failed}&include_failed=1").text
    assert 'id="results"' in included


def test_a_comparison_has_its_own_address_and_lands_on_the_results():
    client, profile_id = _signed_in()
    a, b = _saved_fight(profile_id), _saved_fight(profile_id)
    page = client.get(f"/compare?a={a}&b={b}").text
    assert 'action="/compare#results"' in page
    assert f"/compare?a={a}&amp;b={b}" in page
    # The address only opens for its owner.
    stranger, _ = _signed_in()
    assert 'id="results"' not in stranger.get(f"/compare?a={a}&b={b}").text


def test_the_picker_shows_picture_title_date_ruleset_score_and_status():
    page = (TEMPLATES / "compare.html").read_text(encoding="utf-8")
    for part in ('<img src="/selection-image/{{f.job_id}}"', "{{ f.title }}", "{{ f.ruleset|ruleset_label }}",
                 "score {{ f.card.score.totals.A }}", "Report ready", '<input type="radio" name="{{field}}"'):
        assert part in page, part


# --------------------------------------------------------------------- #42
def test_dashboard_is_a_permanent_redirect_to_camp():
    client = BrowserClient(web.app)
    response = client.get("/dashboard?tab=squad", follow_redirects=False)
    assert (response.status_code, response.headers["location"]) == (301, "/camp?tab=squad")
    links = "".join(path.read_text(encoding="utf-8") for path in TEMPLATES.glob("*.html"))
    assert 'href="/dashboard"' not in links
