from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.responses import RedirectResponse

import app.main as webapp
import core.db as database
import core.retention as retention
from app.main import GUEST_COOKIE, SESSION_COOKIE, app
from app.state import create_job, delete_job
from core.auth import register
from core.payments import PLANS
from core.social_auth import SocialIdentity


class _FakeSocialClient:
    async def authorize_redirect(self, request, redirect_uri, **kwargs):
        return RedirectResponse("https://identity.example/authorize?state=test-state")

    async def authorize_access_token(self, request):
        return {"access_token": "discarded-by-warrioriq"}


class AccountAndProductIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_db = database.DB_PATH
        self.original_outputs = webapp.OUTPUTS
        self.original_uploads = webapp.UPLOADS
        self.original_retention_outputs = retention.OUTPUTS
        self.original_retention_uploads = retention.UPLOADS
        database.DB_PATH = Path(self.temp.name) / "product-test.sqlite3"
        webapp.OUTPUTS = Path(self.temp.name) / "outputs"
        webapp.UPLOADS = Path(self.temp.name) / "uploads"
        retention.OUTPUTS = webapp.OUTPUTS
        retention.UPLOADS = webapp.UPLOADS
        webapp.OUTPUTS.mkdir()
        webapp.UPLOADS.mkdir()
        database.init_db()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        database.DB_PATH = self.original_db
        webapp.OUTPUTS = self.original_outputs
        webapp.UPLOADS = self.original_uploads
        retention.OUTPUTS = self.original_retention_outputs
        retention.UPLOADS = self.original_retention_uploads
        self.temp.cleanup()

    def test_signup_creates_private_session_and_workspace(self):
        response = self.client.post(
            "/signup",
            data={
                "email": "athlete@example.com", "password": "Strong-Local-Password", "next_path": "/dashboard",
                "accept_terms": "true", "age_confirmed": "true",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/dashboard")
        self.assertIn(SESSION_COOKIE, self.client.cookies)
        dashboard = self.client.get("/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("Your baseline starts with one fight", dashboard.text)
        self.assertNotIn("Create your private athlete workspace", dashboard.text)
        account = database.get_account_by_email("athlete@example.com")
        acceptances = database.list_legal_acceptances(profile_id=account["profile_id"])
        self.assertEqual({item["kind"] for item in acceptances}, {
            "terms_acceptance", "privacy_acknowledgement", "age_18_plus_confirmation", "marketing_consent",
        })
        saved_account = database.get_account(account["id"])
        self.assertIsNotNone(saved_account["policies_accepted_at"])
        self.assertEqual(saved_account["terms_version"], webapp.SETTINGS.policy_version)
        self.assertEqual(saved_account["privacy_version"], webapp.SETTINGS.policy_version)
        self.assertEqual(saved_account["marketing_consent"], 0)

    def test_compare_explains_why_it_is_unavailable_before_two_fights(self):
        self.client.post(
            "/signup",
            data={
                "email": "athlete@example.com",
                "password": "Strong-Local-Password",
                "accept_terms": "true",
                "age_confirmed": "true",
            },
        )
        comparison = self.client.get("/compare")
        self.assertEqual(comparison.status_code, 200)
        self.assertIn("Two fights required", comparison.text)
        self.assertIn('href="/#analyze">Analyze another fight', comparison.text)
        self.assertNotIn('id="compareForm"', comparison.text)

    def test_signup_requires_account_manager_terms_and_privacy_acceptance(self):
        response = self.client.post(
            "/signup",
            data={"email": "athlete@example.com", "password": "Strong-Local-Password"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least 18", response.text)
        self.assertIn("Terms of Service and Privacy Policy", response.text)
        self.assertIsNone(database.get_account_by_email("athlete@example.com"))

    def test_marketing_consent_is_optional_separate_and_withdrawable(self):
        response = self.client.post(
            "/signup",
            data={
                "email": "marketing@example.com", "password": "Strong-Local-Password",
                "accept_terms": "true", "age_confirmed": "true", "marketing_consent": "true",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        account = database.get_account_by_email("marketing@example.com")
        self.assertEqual(account["marketing_consent"], 1)
        self.assertIsNotNone(account["marketing_consent_at"])
        privacy_page = self.client.get("/settings/privacy")
        self.assertEqual(privacy_page.status_code, 200)
        self.assertIn("Delete Account &amp; Data", privacy_page.text)
        self.assertIn("Change cookie preferences", privacy_page.text)
        withdrawn = self.client.post("/settings/marketing", data={}, follow_redirects=False)
        self.assertEqual(withdrawn.status_code, 303)
        self.assertEqual(database.get_account(account["id"])["marketing_consent"], 0)
        records = database.list_legal_acceptances(profile_id=account["profile_id"])
        self.assertEqual(records[0]["kind"], "marketing_consent")
        self.assertEqual(records[0]["current_status"], "withdrawn")

    def test_cookie_choices_are_server_stored_and_nonessential_defaults_off(self):
        page = self.client.get("/")
        self.assertIn("Accept All", page.text)
        self.assertIn("Reject Non-Essential", page.text)
        self.assertFalse(page.request.headers.get("x-analytics-enabled"))
        saved = self.client.post(
            "/cookie-preferences",
            data={"choice": "custom", "analytics": "true", "next_path": "/cookies"},
            follow_redirects=False,
        )
        self.assertEqual(saved.status_code, 303)
        self.assertEqual(self.client.cookies.get(webapp.COOKIE_PREFERENCES_COOKIE), "custom-analytics")
        self.assertNotIn("Accept All", self.client.get("/").text)
        guest_id = self.client.cookies.get(GUEST_COOKIE)
        records = database.list_legal_acceptances(guest_id=guest_id)
        self.assertEqual(records[0]["metadata"], {"analytics": True, "marketing": False})

    def test_password_reset_is_one_time_and_revokes_existing_sessions(self):
        account = register("reset@example.com", "Strong-Local-Password")
        with patch.object(webapp, "send_transactional_email", return_value=True) as send_email:
            requested = self.client.post("/forgot-password", data={"email": "reset@example.com"})
        self.assertEqual(requested.status_code, 200)
        self.assertEqual(database.list_outbound_messages(account["id"]), [])
        email_body = send_email.call_args.args[2]
        reset_path = "/reset-password/" + email_body.split("/reset-password/", 1)[1].split()[0]
        changed = self.client.post(reset_path, data={"password": "New-Strong-Password"}, follow_redirects=False)
        self.assertEqual(changed.status_code, 303)
        self.assertIsNone(webapp.authenticate("reset@example.com", "Strong-Local-Password"))
        self.assertIsNotNone(webapp.authenticate("reset@example.com", "New-Strong-Password"))
        reused = self.client.post(reset_path, data={"password": "Another-Strong-Password"})
        self.assertEqual(reused.status_code, 410)

    def test_withdrawal_is_separate_from_cancellation_and_queues_confirmation(self):
        self.client.post(
            "/signup",
            data={
                "email": "billing@example.com", "password": "Strong-Local-Password",
                "accept_terms": "true", "age_confirmed": "true",
            },
        )
        account = database.get_account_by_email("billing@example.com")
        with database.connection() as con:
            con.execute(
                "UPDATE accounts SET plan='athlete',stripe_subscription_id='sub_test',subscription_status='active' WHERE id=?",
                (account["id"],),
            )
        response = self.client.post(
            "/settings/billing/withdraw", data={"confirm": "true"}, follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        action = database.list_subscription_actions(account["id"])[0]
        self.assertEqual(action["action_type"], "eu_withdrawal")
        self.assertEqual(action["status"], "pending_review")
        self.assertTrue(action["metadata"]["eligibility_not_determined"])
        self.assertEqual(database.list_outbound_messages(account["id"])[0]["message_type"], "withdrawal_request_confirmation")
        billing_page = self.client.get("/settings/billing")
        self.assertEqual(billing_page.status_code, 200)
        self.assertIn("Cancel Subscription", billing_page.text)
        self.assertIn("Withdraw from Contract", billing_page.text)

    def test_copyright_intake_is_private_and_admin_is_closed_by_default(self):
        response = self.client.post(
            "/copyright-report",
            data={
                "email": "rights@example.com", "resource_id": "fight-123",
                "details": "I own the identified footage and request review and removal of this unauthorised copy.",
                "good_faith": "true",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Report recorded", response.text)
        self.assertEqual(database.list_moderation_reports()[0]["resource_id"], "fight-123")
        self.assertEqual(self.client.get("/admin").status_code, 404)

    def test_login_requires_and_records_policy_acceptance(self):
        account = register("athlete@example.com", "Strong-Local-Password")
        denied = self.client.post(
            "/login",
            data={"email": "athlete@example.com", "password": "Strong-Local-Password"},
            follow_redirects=False,
        )
        self.assertEqual(denied.status_code, 400)
        self.assertNotIn(SESSION_COOKIE, self.client.cookies)

        accepted = self.client.post(
            "/login",
            data={
                "email": "athlete@example.com",
                "password": "Strong-Local-Password",
                "accept_policies": "true",
            },
            follow_redirects=False,
        )
        self.assertEqual(accepted.status_code, 303)
        self.assertIn(SESSION_COOKIE, self.client.cookies)
        acceptances = database.list_legal_acceptances(profile_id=account["profile_id"])
        self.assertEqual(acceptances[0]["kind"], "account_signin_policies")

    def test_social_signup_links_stable_identity_without_enabling_password_login(self):
        social_client = TestClient(app, base_url="https://warrioriq.eu")
        identity = SocialIdentity(
            provider="google", subject="provider-user-123",
            email="social@example.com", display_name="Social Athlete", email_verified=True,
        )
        try:
            with (
                patch.object(webapp.SOCIAL_AUTH, "client", return_value=_FakeSocialClient()),
                patch.object(
                    webapp.SOCIAL_AUTH, "identity_from_token",
                    new=AsyncMock(return_value=identity),
                ),
            ):
                started = social_client.post(
                    "/auth/google/start",
                    data={
                        "mode": "signup", "next_path": "/dashboard",
                        "accept_terms": "true", "age_confirmed": "true",
                    },
                    follow_redirects=False,
                )
                self.assertEqual(started.status_code, 307)
                self.assertIn("state=test-state", started.headers["location"])
                completed = social_client.get(
                    "/auth/google/callback?state=test-state", follow_redirects=False,
                )
            self.assertEqual(completed.status_code, 303)
            self.assertEqual(completed.headers["location"], "/dashboard")
            self.assertIn(SESSION_COOKIE, social_client.cookies)
            account = database.get_account_by_email("social@example.com")
            self.assertEqual(account["password_login_enabled"], 0)
            self.assertIsNotNone(account["email_verified_at"])
            self.assertIsNone(webapp.authenticate("social@example.com", "any-password"))
            identities = database.list_oauth_identities(int(account["id"]))
            self.assertEqual(
                [(item["provider"], item["subject"]) for item in identities],
                [("google", "provider-user-123")],
            )
            self.assertNotIn("access_token", json.dumps(identities))
        finally:
            social_client.close()

    def test_social_signup_never_auto_links_an_existing_email(self):
        register("existing@example.com", "Strong-Local-Password")
        social_client = TestClient(app, base_url="https://warrioriq.eu")
        identity = SocialIdentity(
            provider="google", subject="different-provider-user",
            email="existing@example.com", display_name="Existing Athlete",
        )
        try:
            with (
                patch.object(webapp.SOCIAL_AUTH, "client", return_value=_FakeSocialClient()),
                patch.object(
                    webapp.SOCIAL_AUTH, "identity_from_token",
                    new=AsyncMock(return_value=identity),
                ),
            ):
                social_client.post(
                    "/auth/google/start",
                    data={
                        "mode": "signup", "accept_terms": "true",
                        "age_confirmed": "true",
                    },
                    follow_redirects=False,
                )
                completed = social_client.get(
                    "/auth/google/callback?state=test-state", follow_redirects=False,
                )
            self.assertEqual(completed.status_code, 400)
            self.assertIn("Sign in with its password first", completed.text)
            account = database.get_account_by_email("existing@example.com")
            self.assertEqual(database.list_oauth_identities(int(account["id"])), [])
            self.assertNotIn(SESSION_COOKIE, social_client.cookies)
        finally:
            social_client.close()

    def test_cross_site_state_change_is_blocked(self):
        response = self.client.post(
            "/signup",
            data={
                "email": "athlete@example.com", "password": "Strong-Local-Password",
                "accept_terms": "true", "age_confirmed": "true",
            },
            headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(database.get_account_by_email("athlete@example.com"))

    def test_account_export_requires_password_and_excludes_password_hash(self):
        self.client.post(
            "/signup",
            data={
                "email": "athlete@example.com", "password": "Strong-Local-Password",
                "accept_terms": "true", "age_confirmed": "true",
            },
        )
        denied = self.client.post("/account/export", data={"password": "wrong-password"})
        self.assertEqual(denied.status_code, 400)
        exported = self.client.post("/account/export", data={"password": "Strong-Local-Password"})
        self.assertEqual(exported.status_code, 200)
        self.assertIn("attachment", exported.headers["content-disposition"])
        self.assertEqual(exported.json()["account"]["email"], "athlete@example.com")
        self.assertNotIn("password_hash", exported.text)
        self.assertNotIn("video_path", exported.text)
        self.assertNotIn("report_path", exported.text)
        self.assertNotIn("sequence_path", exported.text)

    def test_checkout_fails_closed_until_real_operator_details_are_configured(self):
        self.client.post(
            "/signup",
            data={
                "email": "athlete@example.com", "password": "Strong-Local-Password",
                "accept_terms": "true", "age_confirmed": "true",
            },
        )
        response = self.client.post(
            "/checkout/athlete", data={"billing_acceptance": "true"}, follow_redirects=False,
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("operator", response.text.lower())

    def test_login_failure_is_generic_and_open_redirect_is_rejected(self):
        register("athlete@example.com", "Strong-Local-Password")
        response = self.client.post(
            "/login",
            data={"email": "athlete@example.com", "password": "incorrect-one", "next_path": "https://attacker.example", "accept_policies": "true"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("email or password is incorrect", response.text)
        success = self.client.post(
            "/login",
            data={"email": "athlete@example.com", "password": "Strong-Local-Password", "next_path": "//attacker.example", "accept_policies": "true"},
            follow_redirects=False,
        )
        self.assertEqual(success.status_code, 303)
        self.assertEqual(success.headers["location"], "/dashboard")

    def test_starter_daily_allowance_is_atomic_and_failed_use_is_returned(self):
        account = register("athlete@example.com", "Strong-Local-Password")
        today = datetime.now(timezone.utc)
        self.assertTrue(database.reserve_analysis(account["id"], "starter-1", today))
        self.assertTrue(database.reserve_analysis(account["id"], "starter-1", today))
        self.assertFalse(database.reserve_analysis(account["id"], "starter-2", today))
        self.assertTrue(database.release_analysis(account["id"], "starter-1"))
        self.assertTrue(database.reserve_analysis(account["id"], "starter-2", today))
        self.assertTrue(database.reserve_analysis(account["id"], "starter-next-day", today + timedelta(days=1)))
        self.client.post("/login", data={"email": "athlete@example.com", "password": "Strong-Local-Password", "accept_policies": "true"})
        home = self.client.get("/")
        self.assertIn("Limit reached", home.text)
        self.assertIn("Analysis allowance used", home.text)

    def test_checkout_webhook_event_is_idempotent(self):
        account = register("athlete@example.com", "Strong-Local-Password")
        self.assertTrue(database.apply_checkout_event("evt_123", "checkout.session.completed", account["id"], "athlete", 0))
        self.assertFalse(database.apply_checkout_event("evt_123", "checkout.session.completed", account["id"], "athlete", 0))
        self.assertEqual(database.get_account(account["id"])["plan"], "athlete")

    def test_permanent_plan_override_survives_billing_changes(self):
        account = register("owner@example.com", "Strong-Local-Password")
        self.assertTrue(database.set_plan_override(account["id"], "gym"))
        self.assertTrue(database.apply_checkout_event("evt-owner", "checkout.session.completed", account["id"], "athlete", 0))
        saved = database.get_account(account["id"])
        self.assertEqual(saved["plan"], "athlete")
        self.assertEqual(saved["plan_override"], "gym")
        allowance = database.analysis_allowance(account["id"])
        self.assertIs(allowance["plan"], PLANS["gym"])
        self.assertIsNone(allowance["limit"])

        database.save_session(account["id"], "owner-session", "2999-01-01T00:00:00+00:00")
        session_account = database.account_for_session("owner-session")
        self.assertEqual(session_account["plan_override"], "gym")

    def test_paid_plan_allowances_match_the_pricing_contract(self):
        today = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)
        athlete = register("daily@example.com", "Strong-Local-Password")
        database.apply_checkout_event("evt-athlete", "checkout.session.completed", athlete["id"], "athlete", 0)
        for index in range(3):
            self.assertTrue(database.reserve_analysis(athlete["id"], f"athlete-{index}", today))
        self.assertFalse(database.reserve_analysis(athlete["id"], "athlete-4", today))
        self.assertTrue(database.reserve_analysis(athlete["id"], "athlete-tomorrow", today + timedelta(days=1)))

        # The Coach plan was withdrawn. A subscriber still carrying that key
        # must keep equivalent access rather than dropping to the free tier,
        # which is what an unrecognised plan would otherwise give them.
        legacy = register("coach@example.com", "Strong-Local-Password")
        database.apply_checkout_event("evt-coach", "checkout.session.completed", legacy["id"], "coach", 30)
        for index in range(10):
            self.assertTrue(database.reserve_analysis(legacy["id"], f"legacy-{index}", today))
        self.assertFalse(database.reserve_analysis(legacy["id"], "legacy-11", today))

        gym = register("gym@example.com", "Strong-Local-Password")
        database.apply_checkout_event("evt-gym", "checkout.session.completed", gym["id"], "gym", 0)
        for index in range(40):
            self.assertTrue(database.reserve_analysis(gym["id"], f"gym-{index}", today))

        self.assertEqual(PLANS["athlete_pro"]["daily_limit"], 10)
        self.assertEqual(PLANS["athlete_pro"]["report_tier"], "full")
        self.assertTrue(PLANS["gym"]["unlimited"])

    def test_assignment_cannot_be_toggled_by_another_profile(self):
        first = register("first@example.com", "Strong-Local-Password")
        second = register("second@example.com", "Another-Local-Password")
        assignment_id = database.add_assignment(first["profile_id"], "Guard recovery", "Three rounds")
        self.assertFalse(database.toggle_assignment(assignment_id, second["profile_id"]))
        self.assertTrue(database.toggle_assignment(assignment_id, first["profile_id"]))
        self.assertEqual(database.list_assignments(first["profile_id"])[0]["status"], "complete")

    def test_duplicate_active_coach_assignment_is_not_added_twice(self):
        account = register("coach@example.com", "Strong-Local-Password")
        first = database.add_assignment(account["profile_id"], "Guard return", "Four rounds")
        second = database.add_assignment(account["profile_id"], "Guard return", "Four rounds")
        self.assertEqual(first, second)
        self.assertEqual(len(database.list_assignments(account["profile_id"])), 1)

    def test_coach_assignment_can_be_created_and_completed_from_dashboard(self):
        self.client.post(
            "/signup",
            data={
                "email": "coach@example.com",
                "password": "Strong-Local-Password",
                "accept_terms": "true",
                "age_confirmed": "true",
            },
        )
        created = self.client.post(
            "/coach/assignments",
            data={
                "title": "Guard recovery",
                "detail": "Four controlled rounds",
                "next_path": "/dashboard",
            },
            follow_redirects=False,
        )
        self.assertEqual(created.status_code, 303)
        self.assertEqual(created.headers["location"], "/dashboard")
        account = database.get_account_by_email("coach@example.com")
        assignment = database.list_assignments(account["profile_id"])[0]
        dashboard = self.client.get("/dashboard")
        self.assertIn("Guard recovery", dashboard.text)
        self.assertIn("Four controlled rounds", dashboard.text)

        completed = self.client.post(
            f"/coach/assignments/{assignment['id']}/toggle",
            data={"next_path": "/dashboard"},
            follow_redirects=False,
        )
        self.assertEqual(completed.status_code, 303)
        self.assertEqual(completed.headers["location"], "/dashboard")
        self.assertEqual(database.list_assignments(account["profile_id"])[0]["status"], "complete")

    def test_sensitive_pages_send_security_headers(self):
        response = self.client.get("/dashboard")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])

    def test_replay_page_renders_identity_state_in_every_template_block(self):
        self.client.get("/")
        guest_id = self.client.cookies.get(GUEST_COOKIE)
        self.assertIsNotNone(guest_id)
        job_id = "replayrender1"
        job_dir = webapp.OUTPUTS / job_id
        job_dir.mkdir()
        video_path = webapp.UPLOADS / f"{job_id}.mp4"
        video_path.write_bytes(b"video-placeholder")
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False},
            "setup": {"ruleset": "K1", "start_seconds": 0, "end_seconds": 10},
            "video": {"analysis_target": "BOTH"},
            "performance": {"segment_duration_seconds": 10},
            "rounds": [{"number": 1, "start_seconds": 0, "end_seconds": 10, "selected": True}],
            "events": [], "key_moments": [], "illegal_moves": [], "integrity": {},
            "scorecard": {"ruleset_label": "K-1"}, "metrics": {},
            "tracking": {
                "fighter_A_coverage": .90, "fighter_B_coverage": .91,
                "fighter_A_seed_source": "manual_anchor", "fighter_B_seed_source": "manual_anchor",
            },
        }
        (job_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        create_job(job_id, {
            "owner_key": f"guest:{guest_id}", "status": "complete", "video_path": str(video_path),
        })
        try:
            response = self.client.get(f"/replay/{job_id}")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Movement with skeletons", response.text)
            self.assertIn("Play full fight", response.text)
            self.assertIn("Jump anywhere in the fight", response.text)
            self.assertNotIn("No verified key moments are available", response.text)
            self.assertIn("identitySafe=true", response.text)
        finally:
            delete_job(job_id)

    def test_public_navigation_destinations_render(self):
        for path in ("/", "/dashboard", "/history", "/compare", "/coach", "/validation", "/pricing", "/privacy", "/login", "/signup"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("WARRIOR", response.text)

    def test_primary_actions_are_real_and_explain_their_state(self):
        self._sign_in("actions@example.com")
        home = self.client.get("/").text
        self.assertIn('href="#analyze">Analyze a fight', home)
        self.assertIn('id="uploadSubmit" type="submit"', home)
        self.assertIn("Choose a video to continue", home)
        self.assertIn('class="nav-more"', home)
        self.assertIn('href="/history">Fight library', home)

        pricing = self.client.get("/pricing").text
        self.assertNotIn("checkout disabled</span>", pricing)
        self.assertNotIn("Private beta onboarding</span>", pricing)
        self.assertIn("1 analysis per day", pricing)
        self.assertIn("3 analyses per day", pricing)
        self.assertIn("10 analyses per day", pricing)
        self.assertIn("Unlimited fight analyses", pricing)
        self.assertIn("€89.99", pricing)
        self.assertIn('class="plan-banner">Most flexible', pricing)

    def _sign_in(self, email="uploader@example.com"):
        """Analysis is account-only, so upload tests need a signed-in browser."""
        register(email, "Strong-Local-Password")
        self.client.post("/login", data={
            "email": email, "password": "Strong-Local-Password", "accept_policies": "true",
        })

    def test_anonymous_upload_is_refused_and_no_footage_is_stored(self):
        """Analysis is account-only.

        The refusal has to happen before the file is written: fight footage
        shows identifiable athletes, so an anonymous browser session must not
        leave a video on disk that no account is answerable for.
        """
        source = Path(self.temp.name) / "source.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (640, 480))
        for index in range(30):
            frame = np.full((480, 640, 3), 80 + index, dtype=np.uint8)
            cv2.rectangle(frame, (120, 80), (230, 420), (220, 220, 220), -1)
            cv2.rectangle(frame, (410, 80), (520, 420), (150, 180, 230), -1)
            writer.write(frame)
        writer.release()
        before = {path.name for path in webapp.UPLOADS.glob("*")}
        with source.open("rb") as handle:
            response = self.client.post(
                "/upload",
                data={
                    "rights_confirmed": "true", "people_permissions_confirmed": "true",
                    "minor_permission_status": "no_minors",
                },
                files={"video": ("private-fight-name.mp4", handle, "video/mp4")},
                follow_redirects=False,
            )
        self.assertEqual(response.status_code, 401)
        self.assertIn("sign in", response.text.lower())
        self.assertEqual({path.name for path in webapp.UPLOADS.glob("*")}, before)
        self.assertEqual(database.list_fights(1), [])

    def test_mobile_async_upload_returns_the_exact_next_step(self):
        self._sign_in("mobile@example.com")
        source = Path(self.temp.name) / "mobile-upload.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (320, 240))
        for index in range(24):
            frame = np.full((240, 320, 3), 70 + index, dtype=np.uint8)
            cv2.rectangle(frame, (35, 25), (120, 220), (220, 220, 220), -1)
            cv2.rectangle(frame, (200, 25), (285, 220), (150, 180, 230), -1)
            writer.write(frame)
        writer.release()
        with source.open("rb") as handle:
            response = self.client.post(
                "/upload",
                headers={"Accept": "application/json", "User-Agent": "Mobile Safari"},
                data={
                    "rights_confirmed": "true", "people_permissions_confirmed": "true",
                    "minor_permission_status": "no_minors",
                },
                files={"video": ("phone-fight.mp4", handle, "video/mp4")},
                follow_redirects=False,
            )
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertRegex(payload["job_id"], r"^[a-f0-9]{12}$")
        self.assertEqual(payload["next_url"], f"/frame/{payload['job_id']}")
        self.assertTrue(response.cookies.get("warrioriq_active_analysis"))

    def test_failed_selection_frame_creation_removes_partial_upload(self):
        self._sign_in("partial@example.com")
        source = Path(self.temp.name) / "selection-failure.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (320, 240))
        for _ in range(12):
            writer.write(np.full((240, 320, 3), 90, dtype=np.uint8))
        writer.release()

        with source.open("rb") as handle, patch.object(webapp.cv2, "imwrite", return_value=False):
            response = self.client.post(
                "/upload",
                data={
                    "rights_confirmed": "true",
                    "people_permissions_confirmed": "true",
                    "minor_permission_status": "no_minors",
                },
                files={"video": ("fight.mp4", handle, "video/mp4")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(list(webapp.UPLOADS.iterdir()), [])
        self.assertEqual(list(webapp.OUTPUTS.iterdir()), [])

    def test_saved_video_is_owner_scoped_and_can_be_deleted_without_report(self):
        owner = register("owner@example.com", "Strong-Local-Password")
        other = register("other@example.com", "Another-Local-Password")
        job_id = "privateowner1"
        video_path = webapp.UPLOADS / f"{job_id}.mp4"
        video_path.write_bytes(b"private-video")
        report_dir = webapp.OUTPUTS / job_id
        report_dir.mkdir()
        report_path = report_dir / "report.json"
        report_path.write_text("{}", encoding="utf-8")
        database.save_fight(
            job_id, owner["profile_id"], "hidden.mp4", str(video_path), str(report_path),
            "competition", "K1", "A", {}, "2999-01-01T00:00:00+00:00",
        )
        other_client = TestClient(app)
        try:
            other_client.post(
                "/login",
                data={"email": other["email"], "password": "Another-Local-Password", "accept_policies": "true"},
            )
            self.assertEqual(other_client.get(f"/media/{job_id}").status_code, 404)
            self.assertEqual(other_client.post(f"/settings/videos/{job_id}/delete").status_code, 404)
        finally:
            other_client.close()
        self.client.post(
            "/login",
            data={"email": owner["email"], "password": "Strong-Local-Password", "accept_policies": "true"},
        )
        deleted = self.client.post(f"/settings/videos/{job_id}/delete", follow_redirects=False)
        self.assertEqual(deleted.status_code, 303)
        self.assertFalse(video_path.exists())
        self.assertTrue(report_path.exists())
        saved = database.get_fight(job_id)
        self.assertEqual(saved["video_path"], "")
        self.assertIsNotNone(saved["video_deleted_at"])

    def test_selected_fighter_is_report_focus_while_engine_analyzes_both(self):
        self.client.get("/")
        guest_id = self.client.cookies.get(GUEST_COOKIE)
        job_id = "focusfight1"
        job_dir = webapp.OUTPUTS / job_id
        job_dir.mkdir()
        selection = np.full((360, 640, 3), 90, dtype=np.uint8)
        cv2.imwrite(str(job_dir / "selection.jpg"), selection)
        video_path = webapp.UPLOADS / f"{job_id}.mp4"
        video_path.write_bytes(b"video-placeholder")
        create_job(job_id, {
            "owner_key": f"guest:{guest_id}", "status": "selection",
            "video_path": str(video_path), "original_name": "hidden.mp4",
            "fight_type": "competition", "ruleset": "K1", "start_seconds": 0.0,
            "round_count": 1, "round_duration_seconds": 120.0,
            "break_duration_seconds": 60.0, "selected_rounds": [1], "end_seconds": None,
            "profile_id": 0, "persist_result": False, "openai_identity_recovery": False,
        })
        try:
            with patch.object(webapp.executor, "submit") as submit:
                response = self.client.post(f"/api/start/{job_id}", json={
                    "fighter_a_box": [80, 40, 250, 340],
                    "fighter_b_box": [390, 40, 560, 340],
                    "focus_fighter": "B",
                })
            self.assertEqual(response.status_code, 200)
            request = submit.call_args.args[2]
            self.assertEqual(request.analysis_target, "BOTH")
            self.assertEqual(request.focus_fighter, "B")
            saved = webapp.get_job(job_id)
            self.assertEqual(saved["analysis_target"], "BOTH")
            self.assertEqual(saved["focus_fighter"], "B")
        finally:
            delete_job(job_id)

    def test_start_rejects_invalid_or_overlapping_fighter_boxes(self):
        self.client.get("/")
        guest_id = self.client.cookies.get(GUEST_COOKIE)
        job_id = "invalidboxes1"
        job_dir = webapp.OUTPUTS / job_id
        job_dir.mkdir()
        selection = np.full((360, 640, 3), 90, dtype=np.uint8)
        cv2.imwrite(str(job_dir / "selection.jpg"), selection)
        video_path = webapp.UPLOADS / f"{job_id}.mp4"
        video_path.write_bytes(b"video-placeholder")
        create_job(job_id, {
            "owner_key": f"guest:{guest_id}", "status": "selection",
            "video_path": str(video_path), "original_name": "hidden.mp4",
            "video_width": 640, "video_height": 360,
            "fight_type": "competition", "ruleset": "K1", "start_seconds": 0.0,
            "round_count": 1, "round_duration_seconds": 120.0,
            "break_duration_seconds": 60.0, "selected_rounds": [1], "end_seconds": None,
            "profile_id": 0, "persist_result": False, "openai_identity_recovery": False,
        })
        try:
            invalid = self.client.post(f"/api/start/{job_id}", json={
                "fighter_a_box": [-20, 30, 180, 330],
                "fighter_b_box": [390, 30, 560, 330],
                "focus_fighter": "A",
            })
            self.assertEqual(invalid.status_code, 400)
            self.assertIn("inside the video frame", invalid.json()["detail"])

            overlapping = self.client.post(f"/api/start/{job_id}", json={
                "fighter_a_box": [80, 30, 300, 340],
                "fighter_b_box": [100, 40, 310, 340],
                "focus_fighter": "A",
            })
            self.assertEqual(overlapping.status_code, 400)
            self.assertIn("separate fighter", overlapping.json()["detail"])
        finally:
            delete_job(job_id)

    def test_missing_selection_model_keeps_manual_fighter_selection_available(self):
        self.client.get("/")
        guest_id = self.client.cookies.get(GUEST_COOKIE)
        job_id = "manualfallback1"
        job_dir = webapp.OUTPUTS / job_id
        job_dir.mkdir()
        selection = np.full((360, 640, 3), 90, dtype=np.uint8)
        cv2.imwrite(str(job_dir / "selection.jpg"), selection)
        create_job(job_id, {
            "owner_key": f"guest:{guest_id}", "status": "selection",
            "video_path": str(webapp.UPLOADS / f"{job_id}.mp4"),
            "original_name": "hidden.mp4", "video_width": 640, "video_height": 360,
        })
        try:
            with patch.object(webapp, "_get_pose_tracker", side_effect=RuntimeError("missing private checkpoint")):
                response = self.client.get(f"/api/detect/{job_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["people"], [])
            self.assertEqual(response.json()["availability"], "manual_only")
            self.assertNotIn("checkpoint", response.text)
        finally:
            delete_job(job_id)

    def test_cleanup_never_removes_a_protected_running_guest_job(self):
        metadata = retention.mark_guest_job("active123", "guest-token", str(webapp.UPLOADS / "active123.mp4"))
        metadata["expires_at"] = "2000-01-01T00:00:00+00:00"
        marker = webapp.OUTPUTS / "active123" / "guest.json"
        marker.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertEqual(retention.cleanup_expired_guest_jobs({"active123"}), [])
        self.assertTrue(marker.exists())
        self.assertEqual(retention.cleanup_expired_guest_jobs(), ["active123"])
        self.assertFalse(marker.exists())

    def test_abandoned_processing_cleanup_preserves_active_and_saved_jobs(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).timestamp()
        for job_id in ("abandoned1", "active1", "saved1"):
            video = webapp.UPLOADS / f"{job_id}.mp4"
            video.write_bytes(b"video")
            folder = webapp.OUTPUTS / job_id
            folder.mkdir()
            (folder / "selection.jpg").write_bytes(b"frame")
            os.utime(video, (old, old))
            os.utime(folder, (old, old))
        report_path = webapp.OUTPUTS / "saved1" / "report.json"
        report_path.write_text("{}", encoding="utf-8")
        database.save_fight(
            "saved1", 1, "hidden.mp4", str(webapp.UPLOADS / "saved1.mp4"), str(report_path),
            "competition", "K1", "A", {}, "2999-01-01T00:00:00+00:00",
        )
        removed = retention.cleanup_abandoned_processing_files(
            {"active1"}, {"saved1"}, older_than_hours=24,
        )
        self.assertEqual(removed, ["abandoned1"])
        self.assertFalse((webapp.UPLOADS / "abandoned1.mp4").exists())
        self.assertTrue((webapp.UPLOADS / "active1.mp4").exists())
        self.assertTrue((webapp.UPLOADS / "saved1.mp4").exists())

    def test_account_deletion_is_blocked_during_analysis(self):
        self.client.post(
            "/signup",
            data={
                "email": "athlete@example.com", "password": "Strong-Local-Password", "next_path": "/dashboard",
                "accept_terms": "true", "age_confirmed": "true",
            },
        )
        account = database.get_account_by_email("athlete@example.com")
        create_job("running123", {"owner_key": f"account:{account['id']}", "status": "running"})
        try:
            response = self.client.post(
                "/account/delete",
                data={"password": "Strong-Local-Password", "confirmation": "DELETE"},
            )
            self.assertEqual(response.status_code, 409)
            self.assertIsNotNone(database.get_account(account["id"]))
        finally:
            delete_job("running123")

    def test_owner_can_review_candidates_add_missed_actions_and_complete(self):
        account = register("reviewer@example.com", "Strong-Local-Password")
        database.set_plan_override(account["id"], "gym")
        self.client.post("/login", data={"email": "reviewer@example.com", "password": "Strong-Local-Password", "accept_policies": "true"})
        job_id = "reviewfight1"
        job_dir = webapp.OUTPUTS / job_id
        job_dir.mkdir()
        video_path = webapp.UPLOADS / f"{job_id}.mp4"
        video_path.write_bytes(b"video-placeholder")
        candidate = {
            "peak_time": 3.0, "fighter": "A", "technique": "left_head_kick", "family": "kick",
            "limb": "left_leg", "target": "head", "outcome": "clean", "confidence": .94,
            "contact_confidence": .94,
        }
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False},
            "setup": {"ruleset": "K1", "start_seconds": 0, "end_seconds": 10},
            "video": {"analysis_target": "BOTH"}, "performance": {"segment_duration_seconds": 10},
            "rounds": [{"number": 1, "start_seconds": 0, "end_seconds": 10, "selected": True}],
            "events": [candidate], "key_moments": [], "illegal_moves": [], "integrity": {},
            "scorecard": {"ruleset_label": "K-1"}, "metrics": {},
            "tracking": {
                "fighter_A_coverage": .90, "fighter_B_coverage": .91,
                "fighter_A_seed_source": "manual_anchor", "fighter_B_seed_source": "manual_anchor",
                "initial_iou_A": 0.0, "initial_iou_B": 0.0,
            },
        }
        report_path = job_dir / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        database.save_fight(
            job_id, account["profile_id"], "hidden-name.mp4", str(video_path), str(report_path),
            "competition", "K1", "BOTH", {},
        )

        review = self.client.get(f"/review/{job_id}")
        self.assertEqual(review.status_code, 200)
        self.assertIn("Review action candidates", review.text)
        self.assertIn("Not an action", review.text)

        candidate_label = self.client.post(f"/api/annotations/{job_id}", json={
            "event_time": 3.0, "predicted": candidate, "fighter": "A", "technique": "none",
            "target": "none", "outcome": "uncertain", "manual": False,
        })
        self.assertEqual(candidate_label.status_code, 200)
        self.assertFalse(candidate_label.json()["training_consent"])
        missed_label = self.client.post(f"/api/annotations/{job_id}", json={
            "event_time": 6.2, "predicted": {}, "fighter": "B", "technique": "right_low_kick",
            "target": "leg", "outcome": "clean", "manual": True,
        })
        self.assertEqual(missed_label.status_code, 200)
        self.client.post(
            "/profile",
            data={"display_name": "Reviewer", "default_fighter": "A", "allow_model_training": "true"},
        )
        opted_in_label = self.client.post(f"/api/annotations/{job_id}", json={
            "event_time": 7.2, "predicted": {}, "fighter": "A", "technique": "jab",
            "target": "head", "outcome": "clean", "manual": True,
        })
        self.assertEqual(opted_in_label.status_code, 200)
        self.assertTrue(opted_in_label.json()["training_consent"])
        completed = self.client.post(f"/review/{job_id}/complete", data={"complete": "true"}, follow_redirects=False)
        self.assertEqual(completed.status_code, 303)
        self.assertEqual(database.get_fight_review(job_id)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
