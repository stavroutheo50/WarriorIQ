"""Audit of 14 Oct 2026, phase 7: navigation, account, global.

#43 a GET never changes the saved sport; choosing one is a POST
#44 the avatar letter comes from the athlete's name, not the email
#45 the account menu has Settings, Plans and billing, Help and Privacy
#46 the sport switcher marks the current sport
#47 the header collapses to the menu below 1024px instead of wrapping
#48 the footer copyright stays on one line
#49 a signed-out visitor is sent to sign in, without learning whether an id exists
#50 GitHub and Google are equal buttons on /login
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest
from browser_client import BrowserClient

import app.main as web
from core import db as database

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app" / "templates"
STATIC = ROOT / "app" / "static"


def _signed_in():
    client = BrowserClient(web.app)
    email = f"p7-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/signup", data={"email": email, "password": "Strong-Local-Password", "accept_terms": "true",
                                 "age_confirmed": "true"}, follow_redirects=False)
    return client, email


# --------------------------------------------------------------------- #43
def test_opening_a_sport_page_saves_nothing():
    client = BrowserClient(web.app)
    response = client.get("/analyze/boxing")
    assert response.status_code == 200
    assert web.ACTIVE_SPORT_COOKIE not in response.headers.get("set-cookie", "")


def test_choosing_a_sport_is_a_post():
    client = BrowserClient(web.app)
    client.get("/analyze")
    response = client.post("/sport", data={"sport": "muay_thai", "next_path": "/camp"}, follow_redirects=False)
    assert (response.status_code, response.headers["location"]) == (303, "/camp")
    assert f"{web.ACTIVE_SPORT_COOKIE}=muay_thai" in response.headers["set-cookie"]
    assert client.post("/sport", data={"sport": "chess"}, follow_redirects=False).status_code == 400
    # An off-site next falls back to the sport's own page.
    away = client.post("/sport", data={"sport": "boxing", "next_path": "//evil.example"}, follow_redirects=False)
    assert away.headers["location"] == "/analyze/boxing"


# --------------------------------------------------------------------- #44
def test_the_avatar_uses_the_athletes_name():
    assert web.account_display({"email": "stavros@example.com", "display_name": "Theo Stavrou"})["initial"] == "T"
    # The placeholder every new profile starts with is not a name.
    placeholder = web.account_display({"email": "stavros@example.com", "display_name": web.SETTINGS.default_profile_name})
    assert (placeholder["initial"], placeholder["label"]) == ("S", "stavros@example.com")


def test_the_header_shows_the_name_once_it_is_set():
    client, email = _signed_in()
    profile_id = int(database.get_account_by_email(email)["profile_id"])
    with database.connection() as con:
        con.execute("UPDATE profiles SET display_name=? WHERE id=?", ("Theo Stavrou", profile_id))
    page = client.get("/camp").text
    assert '<span class="profile-initial" aria-hidden="true">T</span>' in page
    assert ">Theo Stavrou</span>" in page


# --------------------------------------------------------------------- #45
def test_the_account_menu_has_the_account_pages():
    client, _ = _signed_in()
    page = client.get("/camp").text
    panel = page[page.index('class="profile-menu-panel"'):]
    panel = panel[:panel.index("</details>")]
    for href in ("/profile", "/settings", "/settings/billing", "/feed", "/how-to-record-a-fight-for-analysis",
                 "/privacy"):
        assert f'href="{href}"' in panel, href
    assert 'href="/history"' not in panel          # already in the bar


# --------------------------------------------------------------------- #46
def test_the_switcher_marks_the_current_sport():
    page = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert '{% if sport.key == request.state.active_sport.key %} aria-current="true"{% endif %}' in page
    assert "(current sport)" in page and "sport-menu-tick" in page


# --------------------------------------------------------------------- #47
def test_the_header_collapses_below_1024px():
    fixes = (STATIC / "fixes.css").read_text(encoding="utf-8")
    product = (STATIC / "product.css").read_text(encoding="utf-8")
    assert ("@media(max-width:1023px){.navlinks{display:none}.nav-more{display:none}.account-nav{display:none}"
            ".mobile-menu-button{display:block}}") in fixes
    assert "@media(max-width:1023px){.nav{display:flex}}" in product


# --------------------------------------------------------------------- #48
def test_the_copyright_keeps_one_line():
    assert 'class="small muted footer-copyright">© ' in (TEMPLATES / "base.html").read_text(encoding="utf-8")
    assert ".footer-copyright{flex:0 0 auto;white-space:nowrap}" in (STATIC / "system.css").read_text(encoding="utf-8")


# --------------------------------------------------------------------- #49
@pytest.mark.parametrize("path", ["/result/{id}", "/replay/{id}?fighter=A", "/progress/{id}"])
def test_a_signed_out_visitor_is_sent_to_sign_in(path):
    # The same answer for an id that exists and one that does not.
    client, email = _signed_in()
    profile_id = int(database.get_account_by_email(email)["profile_id"])
    real = "p7" + uuid.uuid4().hex[:10]
    database.save_fight(real, profile_id, "a.mp4", "", "", "sparring", "K1", "A", {}, "2999-01-01T00:00:00+00:00")
    guest = BrowserClient(web.app)
    answers = []
    for job_id in (real, "p7" + uuid.uuid4().hex[:10]):
        response = guest.get(path.format(id=job_id), follow_redirects=False)
        location = response.headers["location"]
        answers.append((response.status_code, location.replace(job_id, "ID")))
        assert location.startswith("/login?next=/")
    assert answers[0] == answers[1] and answers[0][0] == 303


def test_signed_in_someone_elses_fight_is_still_404():
    owner, email = _signed_in()
    profile_id = int(database.get_account_by_email(email)["profile_id"])
    job_id = "p7" + uuid.uuid4().hex[:10]
    database.save_fight(job_id, profile_id, "a.mp4", "", "", "sparring", "K1", "A", {}, "2999-01-01T00:00:00+00:00")
    stranger, _ = _signed_in()
    for path in (f"/result/{job_id}", f"/replay/{job_id}", f"/progress/{job_id}", "/result/nosuchid0000"):
        assert stranger.get(path, follow_redirects=False).status_code == 404


def test_the_api_documentation_stays_private():
    client = BrowserClient(web.app)
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404


# --------------------------------------------------------------------- #50
def test_github_is_a_button_like_google_on_sign_in_only():
    page = (TEMPLATES / "auth.html").read_text(encoding="utf-8")
    assert "{% set legacy_providers = request.state.legacy_oauth_providers if mode != 'signup' else [] %}" in page
    assert "{% for provider in request.state.oauth_providers|list + legacy_providers|list %}" in page
    assert re.search(r'<button class="text-link"[^>]*formaction="/auth/', page) is None
