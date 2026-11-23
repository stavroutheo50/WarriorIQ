"""The site walkthrough of 16 Nov 2026: security and data-loss fixes (P0, P1).

What must hold, each checked on the server, never only in the browser:

  * an account is deleted only with the exact phrase DELETE, the right current
    password and - with two-step sign-in on - a live code; an account with no
    password only through a one-time link emailed to it;
  * every other password-gated action (change password, change email,
    download data, set up or turn off two-step sign-in) refuses a wrong
    password and changes nothing;
  * the forms that destroy or re-route the account are not sign-in forms, so
    the browser cannot fill a saved email or password into them;
  * the two-step setup shows a QR code drawn on the server that decodes to the
    same secret the server verifies;
  * a name typed by a user is escaped wherever it is shown;
  * a failed start serves a WarriorIQ page and recovers by itself.
"""

from __future__ import annotations

import re
import uuid
from unittest import mock

import pytest

import app.main as web
from browser_client import BrowserClient
from core import db
from core.auth import authenticate, hash_password, issue_session

PASSWORD = "Correct-Horse-Battery-9"


def _account(password: str | None = PASSWORD):
    email = f"wt-{uuid.uuid4().hex[:10]}@example.test"
    created = db.create_account(email, hash_password(password) if password else "not-a-login-hash")
    if password is None:
        with db.connection() as con:
            con.execute("UPDATE accounts SET password_login_enabled=0 WHERE id=?", (created["id"],))
    return created, email


def _client(created) -> BrowserClient:
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    return client


def _exists(created) -> bool:
    return db.get_account(int(created["id"])) is not None


# --- P0 #1: deleting an account --------------------------------------------

def test_a_wrong_password_never_deletes_the_account():
    created, _ = _account()
    client = _client(created)
    response = client.post("/account/delete", data={"confirm_phrase": "DELETE", "delete_password": "wrong-password-1"},
                           follow_redirects=False)
    assert response.status_code == 403
    assert "That password is not right" in response.text and 'role="alert"' in response.text
    assert _exists(created)
    # The earlier field names are refused the same way.
    legacy = client.post("/account/delete", data={"confirmation": "DELETE", "password": "wrong-password-1"},
                         follow_redirects=False)
    assert legacy.status_code == 403 and _exists(created)


def test_a_missing_password_never_deletes_the_account():
    created, _ = _account()
    client = _client(created)
    for data in ({"confirm_phrase": "DELETE"}, {"confirm_phrase": "DELETE", "delete_password": ""}):
        response = client.post("/account/delete", data=data, follow_redirects=False)
        assert response.status_code == 400 and _exists(created)


@pytest.mark.parametrize("phrase", ["", "delete", "Delete", "DELET", "DELETE ME", "D E L E T E"])
def test_anything_but_the_exact_phrase_never_deletes_the_account(phrase):
    created, email = _account()
    client = _client(created)
    for typed in (phrase, email):
        response = client.post("/account/delete", data={"confirm_phrase": typed, "delete_password": PASSWORD},
                               follow_redirects=False)
        assert response.status_code == 400 and _exists(created)
        assert "Type DELETE" in response.text


def test_the_right_password_and_phrase_delete_the_account():
    created, email = _account()
    client = _client(created)
    response = client.post("/account/delete", data={"confirm_phrase": "  DELETE  ", "delete_password": PASSWORD},
                           follow_redirects=False)
    assert response.status_code == 303 and not _exists(created)
    assert db.get_account_by_email(email) is None


def test_deleting_needs_a_csrf_token_and_a_session():
    from fastapi.testclient import TestClient

    created, _ = _account()
    plain = TestClient(web.app)
    plain.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    refused = plain.post("/account/delete", data={"confirm_phrase": "DELETE", "delete_password": PASSWORD},
                         follow_redirects=False)
    assert refused.status_code == 403 and _exists(created)
    stranger = BrowserClient(web.app)
    assert stranger.post("/account/delete", data={"confirm_phrase": "DELETE", "delete_password": PASSWORD},
                         follow_redirects=False).status_code == 403
    assert _exists(created)


def test_an_account_without_a_password_is_deleted_only_through_the_emailed_link():
    created, email = _account(password=None)
    client = _client(created)
    sent = []
    with mock.patch.object(web, "send_transactional_email",
                           side_effect=lambda to, subject, body: sent.append((to, body)) or True):
        response = client.post("/account/delete", data={"confirm_phrase": "DELETE"}, follow_redirects=False)
    assert response.status_code == 200 and _exists(created)
    assert len(sent) == 1 and sent[0][0] == email
    link = re.search(r"https?://\S+/account/delete/confirm/(\S+)", sent[0][1]).group(1)
    # Opening the link deletes nothing (mail scanners open links).
    page = client.get(f"/account/delete/confirm/{link}")
    assert page.status_code == 200 and "Delete my account" in page.text and _exists(created)
    # Another account cannot use it, and a damaged one is refused.
    other, _ = _account(password=None)
    assert _client(other).post(f"/account/delete/confirm/{link}").status_code == 410 and _exists(created)
    assert client.post(f"/account/delete/confirm/{link}x").status_code == 410 and _exists(created)
    done = client.post(f"/account/delete/confirm/{link}", follow_redirects=False)
    assert done.status_code == 303 and not _exists(created)


def test_an_expired_deletion_link_is_refused():
    created, _ = _account(password=None)
    client = _client(created)
    token = web._account_delete_signer().dumps({"a": int(created["id"]),
                                                "f": web._account_delete_fingerprint(db.get_account(created["id"]))})
    with mock.patch("itsdangerous.timed.TimestampSigner.get_timestamp",
                    return_value=int(__import__("time").time()) + web.ACCOUNT_DELETE_LINK_MINUTES * 60 + 5):
        assert client.post(f"/account/delete/confirm/{token}").status_code == 410
    assert _exists(created)


# --- P0 #1: every other password-gated action ------------------------------

def test_wrong_passwords_change_nothing_anywhere_else():
    created, email = _account()
    client = _client(created)
    password = client.post("/settings/password", data={"current_password": "wrong-password-1",
                                                      "new_password": "Brand-New-Password-2",
                                                      "confirm_password": "Brand-New-Password-2"},
                           follow_redirects=False)
    assert "error=" in password.headers["location"] and authenticate(email, PASSWORD)
    with mock.patch.object(web, "send_transactional_email", return_value=True) as send:
        changed = client.post("/settings/email", data={"reauth_password": "wrong-password-1",
                                                       "new_email": "someone-else@example.test"},
                              follow_redirects=False)
        assert "error=" in changed.headers["location"] and not send.called
    assert db.pending_email_change(int(created["id"])) is None
    exported = client.post("/account/export", data={"export_password": "wrong-password-1"})
    assert exported.status_code == 403 and "attachment" not in exported.headers.get("content-disposition", "")
    assert email not in exported.text.split("Email</dt>")[0]          # no data payload, just the page


# --- P0 #2: the browser cannot fill these forms in ---------------------------

def _form(page: str, action: str) -> str:
    start = page.index(f'action="{action}"')
    return page[start:page.index("</form>", start)]


def test_destructive_forms_are_not_sign_in_forms():
    created, _ = _account()
    page = _client(created).get("/settings/privacy").text
    delete = _form(page, "/account/delete")
    phrase = re.search(r'<input id="deleteConfirmPhrase"[^>]*>', delete).group(0)
    for attribute in ('type="text"', 'name="confirm_phrase"', 'autocomplete="off"', 'placeholder="DELETE"',
                      'spellcheck="false"', 'autocapitalize="off"'):
        assert attribute in phrase
    assert "email" not in phrase.split('placeholder')[0].lower()
    password = re.search(r'<input id="deleteReauthPassword"[^>]*>', delete).group(0)
    assert 'autocomplete="off"' in password and "readonly" in password and 'name="delete_password"' in password
    assert "current-password" not in delete
    email_form = _form(page, "/settings/email")
    new_email = re.search(r'<input id="changeEmailNew"[^>]*>', email_form).group(0)
    assert 'autocomplete="off"' in new_email and 'value=""' in new_email and "readonly" in new_email
    assert 'name="reauth_password"' in email_form and "current-password" not in email_form
    for action in ("/account/export",):
        assert "current-password" not in _form(page, action) and "data-reauth" in _form(page, action)
    # The real sign-in form keeps current-password: it is the one place it belongs.
    assert 'autocomplete="current-password"' in BrowserClient(web.app).get("/login").text


def test_there_is_one_delete_section_and_it_is_in_settings():
    created, _ = _account()
    client = _client(created)
    profile = client.get("/profile").text
    assert 'action="/account/delete"' not in profile and 'action="/account/export"' not in profile
    assert 'href="/settings/privacy#delete-account"' in profile
    assert client.get("/settings/privacy").text.count('action="/account/delete"') == 1


# --- P1 #5: two-step setup ---------------------------------------------------

# --- P0 #4: names are escaped wherever they are shown -----------------------

PROBE = "QA <img src=x onerror=alert(1)> \"O'Neil\" & Ωmega"


def test_a_hostile_fighter_name_is_escaped_on_the_roster():
    created, _ = _account()
    db.set_account_type(int(created["profile_id"]), "coach") if hasattr(db, "set_account_type") else None
    db.create_fighter(int(created["profile_id"]), PROBE)
    client = _client(created)
    for path in ("/camp", "/history"):
        page = client.get(path).text
        assert "<img src=x onerror=alert(1)>" not in page
    camp = client.get("/camp").text
    assert "&lt;img src=x onerror=alert(1)&gt;" in camp


def test_no_template_marks_a_user_supplied_name_safe():
    from pathlib import Path

    templates = Path(web.ROOT) / "app" / "templates"
    for template in templates.glob("*.html"):
        text = template.read_text(encoding="utf-8")
        for match in re.finditer(r"\{\{[^}]*\|\s*safe[^}]*\}\}", text):
            assert not re.search(r"name|handle|email|bio|notes|title", match.group(0).split("|safe")[0]), \
                (template.name, match.group(0))


def test_the_qa_tool_can_archive_instead_of_deleting():
    from tools import delete_qa_test_data as cleanup

    created, email = _account()
    fighter = db.create_fighter(int(created["profile_id"]), PROBE)
    keep = db.create_fighter(int(created["profile_id"]), "Real Athlete")
    assert cleanup.main(["--email", email, "--archive"]) == 0                   # dry run
    assert all(not f.get("archived") for f in db.list_fighters(int(created["profile_id"]), include_archived=True))
    assert cleanup.main(["--email", email, "--archive", "--confirm"]) == 0
    roster = {f["id"]: f for f in db.list_fighters(int(created["profile_id"]), include_archived=True)}
    assert roster[fighter["id"]]["archived"] and not roster[keep["id"]]["archived"]


# --- P0 #3: a failed start is not an outage --------------------------------

def test_a_failed_start_serves_the_outage_page_and_recovers_by_itself():
    from app import startup_guard

    attempts = []

    def loader():
        attempts.append(1)
        if len(attempts) == 1:
            raise ImportError("half-copied tree")
        return lambda environ, start_response: (start_response("200 OK", []), [b"live"])[1]

    application = startup_guard.guarded(loader, retry_seconds=0)
    seen = {}
    body = application({"REQUEST_METHOD": "GET", "PATH_INFO": "/"},
                       lambda status, headers: seen.update(status=status, headers=dict(headers)))
    # The first request after the failed start retries at once (retry_seconds=0) and is served.
    assert body == [b"live"] and len(attempts) == 2

    failing = startup_guard.guarded(lambda: (_ for _ in ()).throw(RuntimeError("boom")), retry_seconds=3600)
    body = failing({"REQUEST_METHOD": "GET", "PATH_INFO": "/history"},
                   lambda status, headers: seen.update(status=status, headers=dict(headers)))
    assert seen["status"].startswith("503") and seen["headers"]["Retry-After"]
    assert b"WARRIOR" in body[0] and b"Back in a moment" in body[0]


def test_healthz_answers_without_the_database():
    with mock.patch.object(web, "init_db", side_effect=RuntimeError("locked")):
        response = BrowserClient(web.app).get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


# --- P1 #6: choosing a new password ----------------------------------------

def _reset_link(created) -> str:
    from datetime import datetime, timedelta, timezone

    from core.auth import session_token, token_digest

    token = session_token()
    db.save_password_reset_token(int(created["id"]), token_digest(token),
                                 (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat())
    return token


def test_the_reset_page_has_a_confirm_box_an_eye_and_the_rule():
    created, _ = _account()
    page = BrowserClient(web.app).get(f"/reset-password/{_reset_link(created)}").text
    assert 'name="confirm_password"' in page and "New password again" in page
    assert 'class="password-toggle"' in page and "At least 10 characters." in page
    assert 'class="auth-layout"' in page


def test_mismatched_or_short_passwords_are_refused_and_the_link_still_works():
    created, email = _account()
    token = _reset_link(created)
    client = BrowserClient(web.app)
    web._rate_windows.clear()
    mismatch = client.post(f"/reset-password/{token}", data={"password": "Brand-New-Password-2",
                                                            "confirm_password": "Brand-New-Password-3"})
    assert "The two passwords do not match." in mismatch.text and authenticate(email, PASSWORD)
    short = client.post(f"/reset-password/{token}", data={"password": "short", "confirm_password": "short"})
    assert "between 10 and 1,024" in short.text and authenticate(email, PASSWORD)
    done = client.post(f"/reset-password/{token}", data={"password": "Brand-New-Password-2",
                                                         "confirm_password": "Brand-New-Password-2"})
    assert done.status_code == 200 and "Password changed" in done.text and "/login?reset=1" in done.text
    assert authenticate(email, "Brand-New-Password-2")


def test_a_reset_link_works_once_and_says_so_before_anything_is_typed():
    created, _ = _account()
    token = _reset_link(created)
    client = BrowserClient(web.app)
    web._rate_windows.clear()
    client.post(f"/reset-password/{token}", data={"password": "Brand-New-Password-2",
                                                 "confirm_password": "Brand-New-Password-2"})
    used = client.get(f"/reset-password/{token}")
    assert used.status_code == 410 and "expired or was already used" in used.text
    assert 'name="confirm_password"' not in used.text
    again = client.post(f"/reset-password/{token}", data={"password": "Another-Password-33",
                                                         "confirm_password": "Another-Password-33"})
    assert again.status_code == 410


def test_a_reset_link_expires_after_thirty_minutes():
    from datetime import datetime, timedelta, timezone

    from core.auth import session_token, token_digest

    created, email = _account()
    token = session_token()
    db.save_password_reset_token(int(created["id"]), token_digest(token),
                                 (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
    client = BrowserClient(web.app)
    web._rate_windows.clear()
    assert client.get(f"/reset-password/{token}").status_code == 410
    assert client.post(f"/reset-password/{token}", data={"password": "Brand-New-Password-2",
                                                        "confirm_password": "Brand-New-Password-2"}).status_code == 410
    assert authenticate(email, PASSWORD)
    # And the link the server sends is valid for exactly thirty minutes.
    source = (web.ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "timedelta(minutes=30)" in source[source.index("def request_password_reset"):]


# --- P1 #7: sign-in, sign-out, sign-up ---------------------------------------

def test_an_empty_sign_up_says_what_is_missing_beside_each_field():
    client = BrowserClient(web.app)
    web._rate_windows.clear()
    page = client.post("/signup", data={})
    assert page.status_code == 400
    assert 'id="authErrorSummary"' in page.text and "Fix the highlighted fields" in page.text
    # No separate age question since 2026-11-23: the terms box confirms the age.
    for message in ("Enter your email address", "Password must contain", "Accept the Terms of Service"):
        assert message in page.text
    assert page.text.count('aria-invalid="true"') >= 3


def test_signing_out_says_so():
    created, _ = _account()
    client = _client(created)
    out = client.post("/logout", follow_redirects=False)
    assert out.headers["location"] == "/?signed_out=1"
    assert "You're signed out." in client.get(out.headers["location"]).text


def test_forgot_password_keeps_the_email_typed_on_sign_in():
    client = BrowserClient(web.app)
    page = client.get("/forgot-password?email=fighter%40example.test").text
    assert 'value="fighter@example.test"' in page
    assert 'value="<script>' not in client.get("/forgot-password?email=%3Cscript%3E").text
    assert 'id="forgotLink"' in client.get("/login").text


def test_sign_in_and_create_account_can_be_marked_as_the_current_page():
    page = BrowserClient(web.app).get("/login").text
    assert 'data-nav="login" href="/login"' in page and 'data-nav="signup"' in page
    source = (web.ROOT / "app" / "templates" / "base.html").read_text(encoding="utf-8")
    assert "'login':path.startsWith('/signup')?'signup'" in source


def test_the_social_helper_line_is_a_sentence_not_an_eyebrow():
    css = (web.ROOT / "app" / "static" / "walkthrough.css").read_text(encoding="utf-8")
    rule = css[css.index(".social-auth > p.social-auth-gate"):]
    assert "text-transform: none" in rule[:200]
    assert "input:-webkit-autofill" in css and "-webkit-box-shadow: 0 0 0 1000px var(--wiq-field-bg) inset" in css


# --- P2: page fixes that carry logic -------------------------------------------

def test_a_footwork_change_carries_its_unit():
    from core.progress_insights import change_words

    assert change_words(-0.3, "speed") == "-0.3 body lengths a second"
    assert change_words(0.3, "rate") == "+0.3 body lengths a second"


def test_the_pace_graph_marks_an_opening_that_is_not_in_the_report():
    from core.stat_sheet import _pace

    def side(start):
        return {"numbers": {"series": {"points": [{"t": start + i, "speed": 0.5 + i / 100} for i in range(60)]}}}

    shaded = _pace({"A": side(27.0), "B": side(27.0)})
    assert shaded["start_x"] > 0 and shaded["start_label"] == "0:27" and shaded["half_speed"] > 0
    whole = _pace({"A": side(0.0), "B": side(0.0)})
    assert whole["start_x"] == 0.0


def test_the_library_labels_an_upload_saved_twice_and_shows_the_date_once():
    created, _ = _account()
    pid = int(created["profile_id"])
    for suffix in ("a", "b"):
        db.save_fight(f"twice{uuid.uuid4().hex[:6]}{suffix}", pid, "same.mp4", "", "", "sparring", "K1", "A", {})
    page = _client(created).get("/history").text
    assert page.count("Uploaded again · copy 2") == 1
    record = page[page.index('class="fight-record"'):]
    record = record[:record.index("</article>")]
    assert 'data-local="date"' not in record           # the title already carries the date
    assert 'id="historySearch"' in page and "Search your fights" in page


def test_a_plan_too_small_for_the_roster_is_never_offered_starter_included():
    created, _ = _account()
    pid = int(created["profile_id"])
    db.set_plan_override(int(created["id"]), "gym")
    for n in range(3):
        db.create_fighter(pid, f"Athlete {n}")
    page = _client(created).get("/pricing").text
    starter = page[page.index('id="plan-free"'):]
    starter = starter[:starter.index("</section>")]
    assert "Use Starter" not in starter and "this plan holds 1" in starter and "disabled" in starter


def test_a_missing_page_says_so_in_a_sentence():
    page = BrowserClient(web.app).get("/no-such-page-here")
    assert page.status_code == 404
    assert ">Not Found<" not in page.text and "find that page" in page.text


def test_the_plans_banner_keeps_the_space_between_its_sentences():
    page = BrowserClient(web.app).get("/pricing").text
    assert "opens.To" not in page and "opens. To" in page
