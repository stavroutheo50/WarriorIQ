"""Signing in and signing up made simple (owner, 2026-11-23).

"It took me 10 entire minutes just to log in ... We're not a banking app."
Three things went: the authenticator-app second step (at sign-in and before
deleting an account), the separate age question on the sign-up form, and the
"Choose Create account first" wall when someone new pressed Continue with
Google on the sign-in page. Email and password, or Google, is the whole of it.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

from starlette.responses import RedirectResponse

import app.main as web
from browser_client import BrowserClient
from core import db, two_factor
from core.auth import hash_password, issue_session
from core.social_auth import SocialIdentity

PASSWORD = "correct-horse-battery-9"


def _account(email=None):
    email = email or f"simple-{uuid.uuid4().hex[:10]}@example.test"
    return db.create_account(email, hash_password(PASSWORD)), email


def _with_old_authenticator(created) -> None:
    """An account that had turned the authenticator step on before it was removed."""
    db.begin_two_factor_setup(int(created["id"]), two_factor.seal(two_factor.new_secret()))
    db.enable_two_factor(int(created["id"]), two_factor.current_step(), [two_factor.recovery_hash("abcd-efgh-ijkl")])
    assert db.two_factor_row(int(created["id"]))["enabled"]


def test_a_password_signs_in_even_where_the_authenticator_was_on():
    created, email = _account()
    _with_old_authenticator(created)
    client = BrowserClient(web.app)
    web._rate_windows.clear()
    response = client.post("/login", data={"email": email, "password": PASSWORD, "next_path": "/camp"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert "/login/2fa" not in response.headers["location"]
    assert web.SESSION_COOKIE in client.cookies


def test_an_old_second_step_link_goes_back_to_sign_in():
    response = BrowserClient(web.app).get("/login/2fa", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_settings_no_longer_offers_an_authenticator():
    created, _ = _account()
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    page = client.get("/settings/privacy").text
    assert "Two-step sign-in" not in page and "/settings/2fa/" not in page
    assert client.post("/settings/2fa/start", data={"reauth_password": PASSWORD}).status_code in {404, 405}


def test_deleting_an_account_needs_the_password_only():
    created, _ = _account()
    _with_old_authenticator(created)
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    assert 'name="code"' not in client.get("/settings/privacy").text
    web._rate_windows.clear()
    done = client.post("/account/delete", data={"confirm_phrase": "DELETE", "delete_password": PASSWORD},
                       follow_redirects=False)
    assert done.status_code == 303 and db.get_account(int(created["id"])) is None


def test_sign_up_is_email_password_and_one_box():
    client = BrowserClient(web.app)
    page = client.get("/signup").text
    assert 'name="age_group" value="adult"' in page and 'type="radio"' not in page
    assert 'id="underAge"' in page and "or older and </span>I agree" in page
    email = f"oneclick-{uuid.uuid4().hex[:8]}@example.test"
    web._rate_windows.clear()
    created = client.post("/signup", data={"email": email, "password": PASSWORD, "accept_terms": "true"},
                          follow_redirects=False)
    assert created.status_code == 303
    account = db.get_account_by_email(email)
    assert account and account.get("guardian_approval_status") in {None, "", "not_applicable"}


def test_guardian_details_alone_choose_the_under_age_route():
    """Works without JavaScript: filled-in guardian details mean "Under 18?" was used."""
    client = BrowserClient(web.app)
    email = f"young-{uuid.uuid4().hex[:8]}@example.test"
    web._rate_windows.clear()
    with patch.object(web, "send_transactional_email", return_value=True):
        response = client.post("/signup", data={
            "email": email, "password": PASSWORD, "accept_terms": "true",
            "guardian_name": "Parent Name", "guardian_email": f"parent-{uuid.uuid4().hex[:6]}@example.test"},
            follow_redirects=False)
    assert response.status_code == 303
    assert db.get_account_by_email(email)["guardian_approval_status"] == web.GUARDIAN_PENDING


def test_a_half_filled_guardian_route_reopens_under_age():
    client = BrowserClient(web.app)
    web._rate_windows.clear()
    page = client.post("/signup", data={"email": f"half-{uuid.uuid4().hex[:8]}@example.test", "password": PASSWORD,
                                        "accept_terms": "true", "guardian_name": "Parent Name"})
    assert page.status_code == 400 and '<details class="under-age" id="underAge" open>' in page.text


class _FakeGoogle:
    async def authorize_redirect(self, request, redirect_uri, **kwargs):
        return RedirectResponse("https://identity.example/authorize?state=simple-state")

    async def authorize_access_token(self, request):
        return {"access_token": "discarded"}


def test_continue_with_google_on_the_sign_in_page_creates_the_account():
    email = f"google-new-{uuid.uuid4().hex[:8]}@example.test"
    identity = SocialIdentity(provider="google", subject=f"g-{uuid.uuid4().hex[:10]}", email=email,
                              display_name="New Fighter", email_verified=True)
    client = BrowserClient(web.app, base_url="https://warrioriq.eu")
    web._rate_windows.clear()
    with patch.object(web.SOCIAL_AUTH, "client", return_value=_FakeGoogle()), \
            patch.object(web.SOCIAL_AUTH, "identity_from_token", new=AsyncMock(return_value=identity)):
        # From /login, with no consent boxes at all.
        started = client.post("/auth/google/start", data={"mode": "login", "next_path": "/camp"},
                              follow_redirects=False)
        assert started.status_code == 307
        done = client.get("/auth/google/callback?state=simple-state", follow_redirects=False)
    assert done.status_code == 303 and done.headers["location"] == "/camp"
    account = db.get_account_by_email(email)
    assert account is not None and web.SESSION_COOKIE in client.cookies
    kinds = {row["kind"] for row in db.list_legal_acceptances(profile_id=account["profile_id"])}
    assert {"terms_acceptance", "privacy_acknowledgement", "age_18_plus_confirmation"} <= kinds


def test_both_pages_say_what_continuing_with_google_means():
    from core.social_auth import SocialAuthRegistry

    registry = SocialAuthRegistry()
    registry._enabled = {"google"}
    with patch.object(web, "SOCIAL_AUTH", registry):
        for path in ("/login", "/signup"):
            page = BrowserClient(web.app).get(path).text
            assert 'formaction="/auth/google/start"' in page
            assert "Continuing creates your account" in page
            assert 'id="socialGate"' not in page and "disabled" not in page.split('formaction="/auth/google/start"')[1][:200]
