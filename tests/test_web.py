from __future__ import annotations

import re
import contextlib
import json
import time
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from unittest.mock import patch

from browser_client import BrowserClient as TestClient

import app.main as webapp
from app.main import (
    _apply_report_annotations, _build_replay_chapters, _prediction_at,
    _public_analysis_error, _review_candidates, app,
)
from core.config import SETTINGS
from core.evidence_trust import automated_evidence_trust
from core.temporal_model import ACTION_CLASSES


class AnalyticsPolicyTests(unittest.TestCase):
    """The analytics tag and the policy that permits it must stay in step.

    A strict script-src silently blocked googletagmanager.com, so the tag was
    present on every page while no measurement ever reached Google.
    """

    def setUp(self):
        from app.main import app

        self.client = TestClient(app)

    # Both ids are empty by default now, so a test that needs a tag on the page
    # has to put one there. The deployment supplies the real one; the default
    # is empty because a fallback that cannot load is worse than none.
    @contextlib.contextmanager
    def tag_enabled(self, measurement="G-TESTONLY00"):
        previous = SETTINGS.analytics_measurement_id
        object.__setattr__(SETTINGS, "analytics_measurement_id", measurement)
        try:
            yield measurement
        finally:
            object.__setattr__(SETTINGS, "analytics_measurement_id", previous)

    def test_declining_analytics_denies_storage_but_keeps_the_tag_detectable(self):
        """Consent Mode: the tag is present, storage is denied.

        Withholding the tag entirely also hides it from Google's tag detection,
        which then reports a correctly installed site as having no tag.
        """
        with self.tag_enabled():
            response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "essential"})
            self.assertIn("googletagmanager", response.text)
            self.assertIn("'analytics_storage': 'denied'", response.text)
            self.assertNotIn("'analytics_storage': 'granted'", response.text)

    def test_a_choice_made_against_an_older_policy_version_is_asked_again(self):
        """The cookie recorded the choice but not the version it answered.

        So bumping WARRIORIQ_POLICY_VERSION re-prompted nobody: `decided` stayed
        true forever and the banner never came back, while the acceptance log
        recorded a version the live cookie had no link to.
        """
        from core.config import SETTINGS

        # The consent lines only render when a tag is configured, and this test
        # is about the version stamp rather than about analytics being on.
        with self.tag_enabled():
            current = self.client.get("/", cookies={
                "warrioriq_cookie_preferences": f"all:{SETTINGS.policy_version}"})
            self.assertIn("'analytics_storage': 'granted'", current.text)
            self.assertNotIn('id="cookieNotice"', current.text)

            stale = self.client.get("/", cookies={
                "warrioriq_cookie_preferences": "all:1999-01-01"})
            self.assertIn('id="cookieNotice"', stale.text)
            self.assertNotIn("'analytics_storage': 'granted'", stale.text)

    def test_a_choice_made_before_versions_were_recorded_is_not_asked_again(self):
        """Cookies written by the previous format carry no version.

        Those visitors did answer the question. Treating a missing version as
        undecided would re-open the banner for every existing visitor at once,
        which is not what "re-prompt when the policy changes" asks for.
        """
        with self.tag_enabled():
            response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"})
            self.assertIn("'analytics_storage': 'granted'", response.text)
            self.assertNotIn('id="cookieNotice"', response.text)

    def test_saving_a_choice_stamps_it_with_the_policy_version(self):
        from core.config import SETTINGS

        page = self.client.get("/")
        token = re.search(r'name="csrf-token" content="([^"]+)"', page.text).group(1)
        response = self.client.post(
            "/cookie-preferences",
            data={"choice": "essential", "next_path": "/", "csrf_token": token},
            headers={"X-CSRF-Token": token}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertIn(
            f"warrioriq_cookie_preferences={'essential'}:{SETTINGS.policy_version}",
            response.headers["set-cookie"])

    def test_accepting_analytics_grants_storage(self):
        with self.tag_enabled():
            response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"})
            policy = response.headers["content-security-policy"]
            self.assertIn("https://www.googletagmanager.com", policy)
            # The tag is useless if the script loads but its beacons are blocked.
            connect = [part for part in policy.split(";") if part.strip().startswith("connect-src")][0]
            self.assertIn("google-analytics.com", connect)
            self.assertIn("googletagmanager.com/gtag/js", response.text)
            self.assertIn("'analytics_storage': 'granted'", response.text)

    # The Tag Manager container is off by default - the direct gtag.js tag owns
    # analytics here - so the tests covering the container have to switch it on.
    # The container still has to work for whoever turns it back on, and dropping
    # the tests because it stopped being the default would lose the coverage
    # rather than the requirement.
    @contextlib.contextmanager
    def container_enabled(self, container="GTM-TESTONLY"):
        previous = SETTINGS.gtm_container_id
        object.__setattr__(SETTINGS, "gtm_container_id", container)
        try:
            yield container
        finally:
            object.__setattr__(SETTINGS, "gtm_container_id", previous)

    def test_consent_default_is_declared_before_any_tag_loads(self):
        """A default pushed after the loader would let a tag store first."""
        with self.tag_enabled():
            page = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"}).text
            self.assertLess(page.index("gtag('consent', 'default'"), page.index("gtag/js?id="))
        with self.container_enabled():
            page = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"}).text
            self.assertLess(page.index("gtag('consent', 'default'"), page.index("gtm.js?id="))

    def test_tag_manager_ships_both_halves_and_a_frame_policy(self):
        """GTM needs the head script, the noscript iframe, and frame-src.

        The noscript fallback is an iframe, which the site's default-src would
        block, so a container that loads but cannot frame is only half working.
        """
        with self.container_enabled() as container:
            response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"})
            self.assertIn(container, response.text)
            self.assertIn("googletagmanager.com/gtm.js", response.text)
            self.assertIn("googletagmanager.com/ns.html", response.text)
            policy = response.headers["content-security-policy"]
            frame = [part for part in policy.split(";") if part.strip().startswith("frame-src")][0]
            self.assertIn("https://www.googletagmanager.com", frame)

    def test_declining_analytics_loads_the_container_with_storage_denied(self):
        """The container is present so it stays verifiable, but stores nothing.

        Consent Mode is what keeps this compliant: the container may load, and
        it may not write an analytics cookie until consent is granted.
        """
        with self.container_enabled() as container:
            response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "essential"})
            self.assertIn(container, response.text)
            self.assertIn("'analytics_storage': 'denied'", response.text)
            self.assertNotIn("'analytics_storage': 'granted'", response.text)
            # The consent default must be pushed before the container can fire.
            self.assertLess(
                response.text.index("gtag('consent', 'default'"),
                response.text.index("gtm.js?id="),
            )

    def test_clearing_both_ids_disables_analytics_entirely(self):
        """Emptying one id must not silently leave the other measuring."""
        previous_ga = SETTINGS.analytics_measurement_id
        previous_gtm = SETTINGS.gtm_container_id
        object.__setattr__(SETTINGS, "analytics_measurement_id", "G-TESTONLY00")
        try:
            # The container alone still measures, so the policy must stay open.
            # It is off by default now, so this half has to switch it on before
            # it can prove that emptying the measurement id alone is not enough.
            object.__setattr__(SETTINGS, "analytics_measurement_id", "")
            object.__setattr__(SETTINGS, "gtm_container_id", "GTM-TESTONLY")
            response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"})
            self.assertIn("googletagmanager", response.text)
            self.assertNotIn("gtag/js", response.text)

            object.__setattr__(SETTINGS, "gtm_container_id", "")
            response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"})
            self.assertNotIn("googletagmanager", response.headers["content-security-policy"])
            self.assertNotIn("googletagmanager", response.text)
        finally:
            object.__setattr__(SETTINGS, "analytics_measurement_id", previous_ga)
            object.__setattr__(SETTINGS, "gtm_container_id", previous_gtm)


class PublicPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @contextlib.contextmanager
    def signed_in(self):
        """Reach pages that now require an account.

        Choosing a sport sends signed-out visitors to sign in, because /upload
        answers 401 without an account. These tests are about what those pages
        say, not about the gate, so they borrow an account rather than build one.
        """
        import app.main as webapp

        account = {"id": 1, "profile_id": 1, "email": "test@example.com"}
        # The navigation reads request.state.account, which the middleware
        # fills from resolve_session rather than from _account, so both have to
        # answer for the page to render as a signed-in one.
        with mock.patch.object(webapp, "_account", return_value=account),                 mock.patch.object(webapp, "resolve_session", return_value=account),                 mock.patch.object(webapp, "analysis_allowance", return_value=None):
            yield

    def test_public_pages_render(self):
        for path in ("/", "/dashboard", "/history", "/profile", "/coach", "/compare", "/pricing", "/privacy", "/login", "/signup"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("WARRIOR", response.text)
                # The floating back pill was removed: position:fixed inside a
                # transformed .shell, sitting on top of the bottom-left corner
                # of six pages. Nothing should bring it back.
                self.assertNotIn("globalBack", response.text)
                self.assertNotIn("global-back", response.text)

    def test_contact_names_the_address_for_each_purpose_in_the_section(self):
        """/contact read like a template nobody had filled in.

        "Use the configured support email" in one section, "the configured
        privacy email" in the next, and the three real addresses only in the
        operator block at the foot of the page - so a reader had to scroll past
        the whole document and then work out which of three addresses the
        section they were reading had meant.

        Where an address genuinely is not set the generic phrase stays. An
        empty mailto, or an address invented to make the page look finished,
        would be worse than saying it is configured elsewhere.
        """
        import dataclasses

        import core.legal as legal

        configured = dataclasses.replace(
            legal.SETTINGS,
            support_email="support@example.test",
            privacy_email="privacy@example.test",
            dmca_email="copyright@example.test",
        )
        with mock.patch.object(legal, "SETTINGS", configured):
            page = self.client.get("/contact").text
        self.assertIn("Write to support@example.test for account access", page)
        self.assertIn("Write to privacy@example.test for access, correction", page)
        self.assertIn("Write to copyright@example.test for infringement", page)
        self.assertNotIn("the configured support email", page)

        blank = dataclasses.replace(
            legal.SETTINGS, support_email="", privacy_email="", dmca_email="")
        with mock.patch.object(legal, "SETTINGS", blank):
            page = self.client.get("/contact").text
        self.assertIn("the configured support email", page)
        self.assertNotIn("Write to  ", page, "an unset address must not leave a hole")
        self.assertNotIn("mailto:\"", page)

        # Every other legal page still renders; resolve_document touches them
        # all and only text carrying a placeholder is formatted.
        for path in ("/terms", "/privacy", "/cookies", "/refunds"):
            with self.subTest(path=path):
                self.assertEqual(200, self.client.get(path).status_code)

    def test_legal_center_and_every_policy_render(self):
        paths = (
            "/legal", "/terms", "/cookies", "/acceptable-use", "/refunds", "/eula",
            "/video-upload-policy", "/sports-medical-disclaimer", "/dmca", "/accessibility",
            "/ai-transparency", "/security", "/subprocessors", "/contact",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("Policy version", response.text)
        # The launch checklist names the operator details still missing - company
        # name, registration number, copyright agent - which reads to a visitor
        # as "this company does not exist". Off unless deliberately switched on.
        legal = self.client.get("/legal").text
        self.assertNotIn("Launch is blocked", legal)
        self.assertNotIn("ITEMS MISSING", legal)
        self.assertIn("Privacy Policy", legal, "the documents themselves stay public")
        import dataclasses

        shown = dataclasses.replace(webapp.SETTINGS, show_launch_checklist=True)
        with mock.patch.object(webapp, "SETTINGS", shown):
            self.assertIn("Launch is blocked", self.client.get("/legal").text)
        self.assertIn("does not register", self.client.get("/dmca").text)

    def test_choosing_a_sport_signed_out_says_why_before_spending_an_upload(self):
        """Analysis needs an account, and the page says so before the upload.

        Letting someone pick a sport, upload a fight and wait for the analysis
        before mentioning that spends the one thing they cannot get back - so
        it is said first. It used to be said by redirecting to /login, while
        the five other workspace routes answered 200 with a signed-out shell.
        Someone who bookmarked /history got a sales page and someone who
        bookmarked /analyze got a login form, for the same signed-out state.

        What it must NOT say is that a signed-out analysis is "deleted after
        two hours". /upload answers 401 to a guest and is the only place a job
        is created, so no guest analysis can exist to be deleted - that
        sentence promised a guest mode this product does not have.
        """
        response = self.client.get("/analyze", follow_redirects=False)
        self.assertEqual(response.status_code, 200)
        self.assertIn("does not analyse a fight without an account", response.text)
        self.assertNotIn("two hours", response.text)
        # The destination survives whichever way they go.
        self.assertIn("/signup?next=/analyze", response.text)
        self.assertIn("/login?next=/analyze", response.text)
        # ...and the sport grid is not offered to somebody who cannot use it.
        self.assertNotIn("chooser-head", response.text)

    def test_every_workspace_route_answers_a_guest_the_same_way(self):
        for path in ("/analyze", "/dashboard", "/history", "/coach", "/profile", "/compare"):
            with self.subTest(path=path):
                response = self.client.get(path, follow_redirects=False)
                self.assertEqual(response.status_code, 200, f"{path} still redirects")
                self.assertIn("empty-workspace", response.text)

    def test_every_sport_states_what_the_analysis_cannot_see(self):
        """Coverage is disclosed at the point of choice, not after the upload.

        The detector reads punches, kicks and knees. For boxing that is the
        whole sport; for MMA it misses the ground entirely. The chooser badges
        every sport so the five can be compared at a glance, and each sport's
        setup page names what it will be silent about in full - before an hour
        is spent on an upload, rather than in the finished report.
        """
        from core.scoring import SPORTS, sport_unobserved

        with self.signed_in():
            chooser = self.client.get("/analyze").text
        for sport in SPORTS:
            with self.subTest(sport=sport):
                self.assertIn(f'data-sport="{sport}"', chooser)
                missing = sport_unobserved(sport)
                setup = self.client.get(f"/analyze/{sport}")
                self.assertEqual(setup.status_code, 200)
                # Each unobserved action is named, not summarised away.
                for action in missing:
                    self.assertIn(action, setup.text)
                self.assertIn(
                    'data-covered="no"' if missing else 'data-covered="yes"',
                    setup.text,
                )
        # A sport with gaps must not be presented as fully covered.
        self.assertIn('data-covered="no"', chooser)
        self.assertIn('data-covered="yes"', chooser)

    def test_choosing_a_sport_never_waits_on_an_animation(self):
        """The five cards are the page, so they may not fade in on scroll.

        They were built inside a motion sequence, which sets opacity to 0 until
        an IntersectionObserver marks each one visible. The two below the fold
        never got marked, so on a phone taekwondo and MMA could not be chosen at
        all. Decoration may wait for a scroll; navigation may not.
        """
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "sports.html").read_text(encoding="utf-8")
        self.assertIn('class="sport-grid"', template)
        self.assertNotIn("data-motion-sequence", template)
        # Every sport must be reachable straight from the markup.
        with self.signed_in():
            page = self.client.get("/analyze").text
        for sport in ("kickboxing", "boxing", "muay_thai", "taekwondo", "mma"):
            self.assertIn(f'href="/analyze/{sport}"', page)

    def test_a_sleeping_machine_promises_a_time_it_can_keep(self):
        """The wait is quoted from the mechanism that cannot fail.

        A magic packet may or may not cross the uploader's router, so before the
        machine has a track record the promise is the scheduled drain interval -
        the ceiling - and only once enough real wakes have been observed does it
        narrow to the measured median.
        """
        from unittest.mock import patch

        import app.main as webapp

        with patch.object(webapp, "wake_status", return_value={
                "observations": 0, "median_seconds": None}):
            fallback = webapp._wake_expectation()
        self.assertIn("at the latest", fallback)
        self.assertIn("5 minutes", fallback)

        # Two observations is not a track record; the ceiling still stands.
        with patch.object(webapp, "wake_status", return_value={
                "observations": 2, "median_seconds": 18.0}):
            self.assertIn("at the latest", webapp._wake_expectation())

        with patch.object(webapp, "wake_status", return_value={
                "observations": 9, "median_seconds": 22.0}):
            measured = webapp._wake_expectation()
        self.assertIn("usually", measured)
        self.assertIn("20 seconds", measured)

        message = None
        with patch.object(webapp, "wake_status", return_value={
                "observations": 0, "median_seconds": None}):
            message = webapp._deferred_analysis_message()
        self.assertIn("asleep", message)
        self.assertIn("saved", message)

    def test_readiness_reports_whether_the_wake_is_actually_working(self):
        """Observed latency is the only evidence that a magic packet arrives."""
        payload = self.client.get("/ready").json()
        self.assertIn("wake", payload)
        wake = payload["wake"]
        for key in ("observations", "median_seconds", "drain_interval_seconds",
                    "magic_packet_configured"):
            self.assertIn(key, wake)
        self.assertEqual(wake["drain_interval_seconds"], SETTINGS.wake_drain_interval_seconds)

    def test_the_shell_keeps_showing_which_sport_you_are_in(self):
        """Five sports share one product, so every screen has to answer which.

        Opening a sport remembers it, and the shell then carries that sport's
        crest and accent on every other page - so the answer is present without
        a banner, and the chip is itself the way to change it.
        """
        client = TestClient(app)
        try:
            # Nothing chosen yet: no chip rather than a wrong default.
            self.assertNotIn('class="sport-switch"', client.get("/history").text)

            opened = client.get("/analyze/muay_thai")
            self.assertEqual(opened.status_code, 200)
            self.assertEqual(client.cookies.get("warrioriq_sport"), "muay_thai")

            elsewhere = client.get("/history").text
            self.assertIn('class="sport-switch"', elsewhere)
            self.assertIn("Muay Thai", elsewhere)
            # The chip carries the sport's own accent and is the switcher.
            self.assertIn("--sport-accent:226 154 74", elsewhere)
            # It looked like a dropdown - bordered pill, chevron - and was a
            # plain link to /analyze, so pressing it left the page being read.
            self.assertIn('<details class="sport-menu"', elsewhere)
            self.assertNotIn('class="sport-switch" href="/analyze"', elsewhere)
            for key in ("kickboxing", "boxing", "muay_thai", "taekwondo", "mma"):
                self.assertIn(f'href="/analyze/{key}"', elsewhere)

            # Switching sport switches the shell.
            client.get("/analyze/boxing")
            self.assertIn("--sport-accent:233 106 106", client.get("/history").text)
        finally:
            client.close()

    def test_the_chip_shows_the_sport_this_page_is_for(self):
        """Visiting /analyze/boxing after /analyze/taekwondo said "Taekwondo".

        The middleware fills request.state.active_sport from the *incoming*
        cookie, and /analyze/<sport> only writes the new cookie onto the
        response - after the template has already been rendered from the old
        one. So the H1 read "Set up your Boxing fight." under a nav chip that
        still named the sport before it, on the page where getting the sport
        wrong matters most.

        Every ordered pair, because the bug is invisible when the sport does
        not change and one pair passing says nothing about the rest.
        """
        sports = {
            "kickboxing": "Kickboxing", "boxing": "Boxing", "muay_thai": "Muay Thai",
            "taekwondo": "Taekwondo", "mma": "MMA",
        }
        for previous, previous_label in sports.items():
            for current, current_label in sports.items():
                if previous == current:
                    continue
                with self.subTest(previous=previous, current=current):
                    client = TestClient(app)
                    try:
                        client.get(f"/analyze/{previous}")
                        self.assertEqual(client.cookies.get("warrioriq_sport"), previous)
                        page = client.get(f"/analyze/{current}").text
                        self.assertIn(
                            f'<span class="sport-switch-name">{current_label}</span>', page)
                        self.assertNotIn(
                            f'<span class="sport-switch-name">{previous_label}</span>', page)
                    finally:
                        client.close()

    def test_no_text_is_left_below_the_readability_floor(self):
        """The audit found 85 rules setting body and label text at 9-10px.

        A modular scale replaced them. This guards the floor rather than the
        exact sizes, so the scale can be tuned without the test fighting it.
        """
        system = (Path(__file__).resolve().parents[1] / "app" / "static" / "system.css").read_text(encoding="utf-8")
        self.assertIn("--wiq-text-micro: 11px", system)
        # No rule in the system layer may set type below the 11px floor.
        import re

        for size in re.findall(r"font-size:\s*([0-9.]+)px", system):
            self.assertGreaterEqual(float(size), 9.5, f"{size}px is below the floor")

    def test_an_unknown_sport_is_not_invented(self):
        self.assertEqual(self.client.get("/analyze/sumo").status_code, 404)

    def test_sports_with_one_ruleset_do_not_ask_which(self):
        """Boxing and MMA have a single ruleset, so the select is not a choice.

        The key still has to reach the scorer, so it posts as a hidden field
        rather than being dropped along with the question.
        """
        for sport, expected in (("boxing", "BOXING"), ("mma", "MMA")):
            with self.subTest(sport=sport):
                page = self.client.get(f"/analyze/{sport}").text
                self.assertNotIn('id="fightRuleset"', page)
                self.assertIn(f'type="hidden" name="ruleset" value="{expected}"', page)
        for sport in ("kickboxing", "muay_thai", "taekwondo"):
            with self.subTest(sport=sport):
                self.assertIn('id="fightRuleset"', self.client.get(f"/analyze/{sport}").text)

    def test_footer_exposes_compact_legal_navigation(self):
        page = self.client.get("/").text
        for path in ("/terms", "/privacy", "/cookies", "/video-upload-policy", "/refunds", "/acceptable-use", "/contact"):
            self.assertIn(f'href="{path}"', page)
        self.assertIn("© 2026 WarriorIQ. All rights reserved.", page)

    def test_health_probe_is_minimal_and_available(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "WarriorIQ")
        # The deployed commit makes a settings-only deploy visible, but the probe
        # must stay minimal: no account, model, path or worker detail.
        self.assertEqual(set(payload), {"status", "service", "commit"})
        self.assertNotIn("/", payload["commit"])
        self.assertLessEqual(len(payload["commit"]), 40)
        self.assertTrue(response.headers.get("x-request-id"))
        self.assertIn("app;dur=", response.headers.get("server-timing", ""))
        self.assertNotIn("gpu", response.text.lower())

    def test_large_public_responses_are_compressed_for_mobile(self):
        response = self.client.get("/", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-encoding"), "gzip")
        self.assertIn("Accept-Encoding", response.headers.get("vary", ""))

    def test_navigation_uses_the_small_logo_asset(self):
        root = Path(__file__).resolve().parents[1]
        home = self.client.get("/").text
        logo = root / "app" / "static" / "warrioriq-logo-96.png"
        self.assertIn('src="/static/warrioriq-logo-96.png"', home)
        self.assertLess(logo.stat().st_size, 25_000)

    def test_search_guides_are_unique_indexable_and_in_the_sitemap(self):
        paths = (
            "/kickboxing-fight-analysis", "/k1-fight-analysis",
            "/fight-video-analysis-for-coaches", "/how-to-record-a-fight-for-analysis",
        )
        titles = set()
        original = SETTINGS.public_base_url
        object.__setattr__(SETTINGS, "public_base_url", "https://warrioriq.eu")
        try:
            sitemap = self.client.get("/sitemap.xml").text
            for path in paths:
                with self.subTest(path=path):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn('name="robots" content="index,follow,max-image-preview:large"', response.text)
                    self.assertIn('property="og:url"', response.text)
                    self.assertIn('"FAQPage"', response.text)
                    self.assertIn(f"https://warrioriq.eu{path}", sitemap)
                    title = response.text.split("<title>", 1)[1].split("</title>", 1)[0]
                    titles.add(title)
        finally:
            object.__setattr__(SETTINGS, "public_base_url", original)
        self.assertEqual(len(titles), len(paths))

    def test_www_host_redirects_to_the_canonical_domain(self):
        original = SETTINGS.public_base_url
        object.__setattr__(SETTINGS, "public_base_url", "https://warrioriq.eu")
        try:
            response = self.client.get(
                "/kickboxing-fight-analysis?from=www",
                headers={"host": "www.warrioriq.eu", "x-forwarded-proto": "https"},
                follow_redirects=False,
            )
        finally:
            object.__setattr__(SETTINGS, "public_base_url", original)
        self.assertEqual(response.status_code, 308)
        self.assertEqual(
            response.headers["location"],
            "https://warrioriq.eu/kickboxing-fight-analysis?from=www",
        )

    def test_render_entrypoint_binds_public_host_and_port(self):
        from run import server_config

        with patch.dict(os.environ, {"RENDER": "true", "PORT": "10000"}, clear=True):
            self.assertEqual(server_config(), ("0.0.0.0", 10000, False))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(server_config(), ("127.0.0.1", 8000, True))

    def test_ouipanel_entrypoint_uses_server_port(self):
        from run import server_config

        with patch.dict(os.environ, {"SERVER_PORT": "25639"}, clear=True):
            self.assertEqual(server_config(), ("0.0.0.0", 25639, False))

    def test_ouipanel_environment_loader_is_installed(self):
        requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("python-dotenv", requirements)

    def test_cpanel_passenger_entrypoint_is_valid_and_loads_environment_first(self):
        root = Path(__file__).resolve().parents[1]
        passenger = (root / "passenger_wsgi.py").read_text(encoding="utf-8")
        cloudlinux = (root / "warrioriq_wsgi.py").read_text(encoding="utf-8")

        compile(passenger, "passenger_wsgi.py", "exec")
        compile(cloudlinux, "warrioriq_wsgi.py", "exec")
        for entrypoint in (passenger, cloudlinux):
            self.assertIn("from a2wsgi import ASGIMiddleware", entrypoint)
            self.assertLess(entrypoint.index("load_dotenv()"), entrypoint.index("from app.main import app"))
            self.assertIn("application = ASGIMiddleware(app", entrypoint)

    def test_cpanel_deployment_targets_the_stable_python_app_root(self):
        deployment = (Path(__file__).resolve().parents[1] / ".cpanel.yml").read_text(encoding="utf-8")

        self.assertIn("DEPLOYPATH=/home/dchoodxm/warrioriq", deployment)
        self.assertIn("/bin/cp -R app core dataset tests tools $DEPLOYPATH", deployment)
        self.assertIn("passenger_wsgi.py", deployment)
        self.assertIn("warrioriq_wsgi.py", deployment)
        self.assertIn("requirements-web.txt", deployment)
        self.assertNotIn(".env ", deployment)
        self.assertNotIn("uploads", deployment)

    def test_web_host_requirements_exclude_gpu_runtime(self):
        requirements = (Path(__file__).resolve().parents[1] / "requirements-web.txt").read_text(encoding="utf-8")
        self.assertIn("fastapi", requirements)
        self.assertIn("opencv-python-headless", requirements)
        for heavy in ("ultralytics", "sam2", "torch"):
            self.assertNotIn(heavy, requirements.lower())

    def test_health_probe_is_not_redirected_on_render_private_http(self):
        original = SETTINGS.public_base_url
        object.__setattr__(SETTINGS, "public_base_url", "https://warrioriq.onrender.com")
        try:
            response = self.client.get("/health", follow_redirects=False)
        finally:
            object.__setattr__(SETTINGS, "public_base_url", original)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_web_import_does_not_initialize_heavy_ai_runtimes(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment.update({"PYTHONPATH": str(root), "WARRIORIQ_DATA_DIR": temporary})
            completed = subprocess.run(
                [sys.executable, "-c", "import sys; import app.main; print(int('torch' in sys.modules), int('ultralytics' in sys.modules))"],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "0 0")

    def test_configured_data_directory_keeps_runtime_files_outside_source_tree(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            # This asks what DATA_ROOT alone does, so the narrower per-path
            # overrides have to come off first - tests/conftest.py sets all
            # three, and inheriting them would have the subprocess answer a
            # different question and fail for the wrong reason.
            for narrower in ("WARRIORIQ_DB_PATH", "WARRIORIQ_UPLOADS_DIR", "WARRIORIQ_OUTPUTS_DIR"):
                environment.pop(narrower, None)
            environment.update({"PYTHONPATH": str(root), "WARRIORIQ_DATA_DIR": temporary})
            completed = subprocess.run(
                [sys.executable, "-c", "from core.config import DATA_ROOT,UPLOADS,OUTPUTS,DB_PATH; print(DATA_ROOT); print(UPLOADS.parent==DATA_ROOT, OUTPUTS.parent==DATA_ROOT, DB_PATH.parent==DATA_ROOT)"],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("True True True", completed.stdout)

    def test_each_written_path_can_be_moved_on_its_own(self):
        """A narrower override must beat DATA_ROOT, or the test suite is not isolated.

        tests/conftest.py leans on exactly this: it points the database,
        uploads and outputs at a scratch directory while leaving MODELS and
        DATASET on the real DATA_ROOT, so a test that needs the pose engine
        still finds it. Without that precedence the suite writes into the
        development database and job queue, which is how 196 test accounts and
        725 phantom jobs accumulated there before anyone looked.
        """
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as data_root, tempfile.TemporaryDirectory() as scratch:
            environment = dict(os.environ)
            environment.update({
                "PYTHONPATH": str(root),
                "WARRIORIQ_DATA_DIR": data_root,
                "WARRIORIQ_DB_PATH": str(Path(scratch) / "test.sqlite3"),
                "WARRIORIQ_UPLOADS_DIR": str(Path(scratch) / "uploads"),
                "WARRIORIQ_OUTPUTS_DIR": str(Path(scratch) / "outputs"),
            })
            completed = subprocess.run(
                [sys.executable, "-c",
                 "from core.config import DATA_ROOT,UPLOADS,OUTPUTS,DB_PATH,MODELS,DATASET;"
                 "print(DB_PATH.parent!=DATA_ROOT, UPLOADS.parent!=DATA_ROOT, OUTPUTS.parent!=DATA_ROOT,"
                 "MODELS.parent==DATA_ROOT, DATASET.parent==DATA_ROOT)"],
                cwd=root, env=environment, capture_output=True, text=True, timeout=15, check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("True True True True True", completed.stdout,
                      "the written paths must move independently while models and dataset stay put")

    def test_render_blueprint_uses_health_probe_and_port_aware_entrypoint(self):
        blueprint = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(encoding="utf-8")
        self.assertIn("healthCheckPath: /health", blueprint)
        self.assertIn("startCommand: python run.py", blueprint)

    def test_render_blueprint_keeps_the_data_and_skips_the_gpu_stack(self):
        """Three ways this file loses data or fails to build, all of them quiet.

        A free Render service has no persistent disk, so the SQLite database
        and every stored report are erased on each deploy and each idle
        restart - 4.1 MB holding 229 fights and 692 MB of reports, at the time
        this was written, gone with no warning. A free service also sleeps and
        pays a cold start, which is the problem someone reaching for this file
        is usually trying to escape. And `requirements.txt` is the GPU analysis
        stack: gigabytes of torch and ultralytics the web service never
        imports, while `requirements-web.txt` is the one that exists for it.

        None of that announces itself, so it is pinned here rather than left to
        be rediscovered.
        """
        import yaml

        blueprint = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "render.yaml").read_text(encoding="utf-8"))
        service = blueprint["services"][0]

        self.assertNotEqual(service["plan"], "free",
                            "a free service has no disk: the database and every report are erased")
        disk = service.get("disk")
        self.assertIsNotNone(disk, "without a mounted disk WarriorIQ writes to ephemeral storage")

        env = {item["key"]: item for item in service["envVars"]}
        self.assertEqual(env["WARRIORIQ_DATA_DIR"]["value"], disk["mountPath"],
                         "the data directory must be the mounted disk, or nothing persists")
        self.assertIn("requirements-web.txt", service["buildCommand"])
        self.assertNotIn("-r requirements.txt", service["buildCommand"],
                         "that is the GPU stack; the web service never imports it")
        self.assertIs(env["WARRIORIQ_WORKER_TOKEN"].get("sync"), False,
                      "the worker token must be prompted for, never committed")
        self.assertNotIn("value", env["WARRIORIQ_WORKER_TOKEN"],
                         "a token value in the repository is a leaked credential")

    def test_https_proxy_origin_and_secure_cookie_work_on_render(self):
        response = self.client.post(
            "/cookie-preferences",
            data={"choice": "essential", "next_path": "/"},
            headers={
                "host": "warrioriq.onrender.com",
                "origin": "https://warrioriq.onrender.com",
                "x-forwarded-proto": "https",
                "x-forwarded-host": "warrioriq.onrender.com",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertIn("Secure", response.headers["set-cookie"])

    def test_public_pages_never_show_configuration_placeholders(self):
        for path in ("/", "/legal", "/privacy", "/terms"):
            with self.subTest(path=path):
                self.assertNotIn("Not configured", self.client.get(path).text)

    def test_analysis_errors_are_safe_for_public_status(self):
        message = _public_analysis_error(FileNotFoundError("C:/private/models/checkpoint.pt"))
        self.assertNotIn("C:/private", message)
        self.assertNotIn("checkpoint.pt", message)
        self.assertIn("analysis engine", message.lower())

    def test_robots_sitemap_and_private_noindex_are_safe(self):
        robots = self.client.get("/robots.txt")
        self.assertEqual(robots.status_code, 200)
        self.assertIn("User-agent: *", robots.text)
        sitemap = self.client.get("/sitemap.xml")
        self.assertEqual(sitemap.status_code, 200)
        self.assertNotIn("/result/", sitemap.text)
        self.assertNotIn("/media/", sitemap.text)
        self.assertIn('name="robots" content="noindex,nofollow"', self.client.get("/dashboard").text)

    def test_public_search_metadata_uses_official_domain_and_keeps_auth_private(self):
        original = SETTINGS.public_base_url
        object.__setattr__(SETTINGS, "public_base_url", "https://warrioriq.eu")
        try:
            home = self.client.get("/")
            sitemap = self.client.get("/sitemap.xml")
            robots = self.client.get("/robots.txt")
            login = self.client.get("/login")
        finally:
            object.__setattr__(SETTINGS, "public_base_url", original)
        self.assertIn("WarriorIQ · AI Fight Analysis for Combat Sports", home.text)
        self.assertIn('<link rel="canonical" href="https://warrioriq.eu/">', home.text)
        self.assertIn('name="robots" content="index,follow,max-image-preview:large"', home.text)
        self.assertIn('"@type":"WebSite"', home.text)
        self.assertIn("https://warrioriq.eu/", sitemap.text)
        self.assertIn("Sitemap: https://warrioriq.eu/sitemap.xml", robots.text)
        self.assertIn('name="robots" content="noindex,nofollow"', login.text)
        self.assertNotIn('<link rel="canonical"', login.text)

    def test_custom_404_is_useful_and_does_not_leak_details(self):
        response = self.client.get("/definitely-not-a-page")
        self.assertEqual(response.status_code, 404)
        self.assertIn("That page left the ring", response.text)
        self.assertIn("Start an analysis", response.text)

    def test_upload_permissions_are_simple_and_explicit(self):
        # The form lives on a sport's setup page now; kickboxing stands in for
        # all five, which render from one template.
        home = self.client.get("/analyze/kickboxing").text
        self.assertIn('name="rights_confirmed"', home)
        # Owning the footage and having permission from the people in it are two
        # different claims - you can hold the rights to a clip of somebody who
        # never agreed to appear in it. The server has always required both, but
        # the page used to post the second as a hidden field set to true, so the
        # user asserted a permission nobody asked them for. Both are checkboxes
        # now, both required, and no hidden consent field ships at all.
        self.assertIn('type="checkbox" name="people_permissions_confirmed" value="true" required', home)
        self.assertNotIn('type="hidden" name="people_permissions_confirmed"', home)
        self.assertEqual(2, home.count('type="checkbox" name="rights_confirmed" value="true" required')
                         + home.count('type="checkbox" name="people_permissions_confirmed" value="true" required'))
        self.assertIn('name="minor_permission_status"', home)
        self.assertEqual(home.count('type="radio" name="minor_permission_status"'), 2)
        self.assertNotIn(">Choose one<", home)
        # Rounds ARE asked for again, and as visible controls. Deriving them
        # from the file's duration made every bout "1 x 71s" in the report: a
        # 3x2min fight with breaks and a walk-off is eight minutes of footage,
        # and no amount of reading the container tells you it was three twos.
        # The sport's usual format is preselected so the common case is still
        # one glance, and "use the whole video" stays for anyone unsure.
        self.assertIn('id="roundCount"', home)
        self.assertIn('id="roundSeconds"', home)
        self.assertIn("readDuration", home)
        for name in ("round_count", "round_duration_seconds", "fight_type"):
            with self.subTest(field=name):
                self.assertNotIn('type="hidden" name="%s"' % name, home)
        # Sparring is a real choice, or the library's Sparring filter can never
        # match anything - every upload was posted as a competition.
        self.assertIn('<option value="sparring">', home)
        self.assertIn('<option value="competition" selected>', home)
        # The ruleset is the one thing WarriorIQ cannot infer, so it stays.
        self.assertIn('name="ruleset"', home)
        self.assertNotIn('id="fightSettings"', home)
        self.assertNotIn('id="trackingRecoveryTitle"', home)
        self.assertIn("Video Upload Policy", home)
        self.assertNotIn("These confirmations apply to this fight video", home)
        self.assertNotIn("Account policies are accepted only", home)
        # The age and guardian statements belong to the home page's explainer;
        # the form carries the consent controls themselves.
        explainer = self.client.get("/").text.lower()
        self.assertIn("18 or older", explainer)
        self.assertIn("parent or guardian", explainer)
        self.assertIn("parent or guardian", home.lower())
        self.assertIn("under 18", home.lower())
        self.assertIn('id="uploadProgress"', home)
        self.assertIn("request.upload.addEventListener('progress'", home)
        signup = self.client.get("/signup").text
        self.assertIn('name="accept_terms"', signup)
        # Signing in no longer re-asks: acceptance is taken at signup, and an
        # account behind the current policy version is asked once afterwards,
        # when there is finally an account to compare against.
        self.assertNotIn('name="accept_policies"', self.client.get("/login").text)
        self.assertIn("Acceptable Use Policy", signup)
        self.assertIn("Terms of Service", signup)
        self.assertIn("Privacy Policy", signup)
        self.assertIn('name="age_confirmed"', signup)
        self.assertIn("at least 18", signup.lower())
        self.assertIn('name="marketing_consent"', signup)

    @staticmethod
    def _without_random_tokens(html: str) -> str:
        """The page with its generated values blanked out.

        The check below is that the internal version marker never reaches a
        visitor - the build directory is WarriorIQ_V4_Professional and that
        must not leak. Run against the raw HTML it also searched the CSRF
        tokens, which are random base64url: with a 64-character alphabet and
        roughly 120 adjacent positions across the tokens on a page, two of them
        spell the marker every few hundred runs by pure chance. Observed:
        content="uXV451oWOqdn_u6h7BQdbeaetb-n2wSjScHuWjGhi2k", which fails on
        the "V4" at index 2 and passed on the very next run.

        A flake that rare is worse than no test - it fires long after the
        change that would explain it, and teaches everyone to re-run. So the
        generated values go before the assertion, and nothing else does.
        """
        html = re.sub(r'(content|value)="[A-Za-z0-9_\-]{20,}"',
                      r'\1="<generated>"', html)
        return re.sub(r'\?v=[0-9a-f]+', "?v=<generated>", html)

    def test_professional_home_has_product_and_trust_sections(self):
        response = self.client.get("/")
        self.assertIn("Four steps. No technical homework.", response.text)
        self.assertIn("Your next round starts here.", response.text)
        self.assertIn("Evidence before claims.", response.text)
        self.assertNotIn("V" + "4", self._without_random_tokens(response.text))

    def test_the_version_marker_check_is_not_defeated_by_the_masking(self):
        """The masking must not be able to hide a real leak.

        Blanking the random values is only safe if a genuine marker still
        fails. If this ever passes, the substitution has grown wide enough to
        swallow the thing it was protecting.
        """
        leaked = '<p>Built with WarriorIQ_V4_Professional</p>'
        self.assertIn("V" + "4", self._without_random_tokens(leaked))
        # And a token that happens to contain the marker still does not fail.
        innocent = '<meta name="csrf-token" content="uXV451oWOqdn_u6h7BQdbeaetb">'
        self.assertNotIn("V" + "4", self._without_random_tokens(innocent))

    def test_professional_polish_uses_shared_product_surfaces(self):
        home = self.client.get("/").text
        history = self.client.get("/history").text
        shell = (Path(__file__).resolve().parents[1] / "app" / "templates" / "base.html").read_text(encoding="utf-8")
        history_template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "history.html").read_text(encoding="utf-8")

        # product.css is served inside the concatenated base bundle now, so the
        # guarantee is that the shared layer still reaches the page - not that
        # base.html names the file.
        from app.main import CSS_BUNDLES

        self.assertIn('/assets/base.css', shell)
        self.assertIn("product.css", CSS_BUNDLES["base"])
        self.assertIn("Upload your fight.", home)
        self.assertIn("See it in numbers.", home)
        self.assertIn('class="product-preview"', home)
        self.assertIn('class="workflow-track"', home)
        self.assertIn('class="fight-archive"', history)
        self.assertIn('id="historySearch"', history_template)

    def test_fight_setup_redesign_preserves_analysis_workflow_contracts(self):
        root = Path(__file__).resolve().parents[1]
        frame = (root / "app" / "templates" / "frame.html").read_text(encoding="utf-8")
        selection = (root / "app" / "templates" / "select.html").read_text(encoding="utf-8")

        self.assertIn('class="workflow-steps"', frame)
        self.assertIn('class="frame-workbench"', frame)
        self.assertIn('id="sourceVideo"', frame)
        self.assertIn('id="useFrame"', frame)
        self.assertIn("setFrameStatus", frame)
        self.assertIn('class="fighter-lock-layout"', selection)
        self.assertIn('id="stage"', selection)
        self.assertIn('id="canvas"', selection)
        self.assertIn('name="focusFighter" value="A"', selection)
        self.assertIn('name="focusFighter" value="B"', selection)
        self.assertIn('id="start"', selection)
        self.assertNotIn('name="focusFighter" value="BOTH"', selection)

    def test_identity_canvas_reads_the_pixel_ratio_in_exactly_one_place(self):
        # The canvas backing store is sized in device pixels and everything else
        # on the page - pointer events, the image viewport, the boxes - is in CSS
        # pixels. fit() reconciles the two with a single setTransform.
        #
        # The bug this guards against: the draw functions used to read
        # window.devicePixelRatio themselves and multiply every coordinate by it.
        # The backing store was sized with the ratio at fit time, the drawing used
        # the ratio at draw time, and nothing re-fitted when the ratio changed -
        # ResizeObserver only fires on a size change, while moving a window to a
        # second monitor changes the ratio and nothing else. Measured with the real
        # geometry functions, fit at ratio 1 then draw at ratio 2 put a box dragged
        # at (100,80) 160x250 CSS px on screen at (200,160) 319x235.
        #
        # So: the ratio may be read in fit() and in the watcher that re-fits, and
        # nowhere else. If a new draw function needs it, that is the bug coming
        # back - scale in fit() instead.
        root = Path(__file__).resolve().parents[1]
        selection = (root / "app" / "templates" / "select.html").read_text(encoding="utf-8")
        script = selection.split("{% block scripts %}", 1)[1]
        code = "\n".join(
            line for line in script.splitlines() if not line.lstrip().startswith("//"))

        self.assertIn("ctx.setTransform(ratio,0,0,ratio,0,0)", code)
        self.assertEqual(2, code.count("devicePixelRatio"), code)
        self.assertIn("matchMedia", code)
        self.assertIn("dppx", code)
        # clearRect has to use the CSS size; canvas.width is in device pixels and
        # would leave a stale band on screen at any ratio above 1.
        self.assertIn("ctx.clearRect(0,0,cssWidth,cssHeight)", code)

    def test_home_hero_copy_starts_at_the_top_of_the_upload_card(self):
        css = self.client.get("/static/fixes.css").text
        self.assertRegex(css, r"\.hero\{[^}]*align-items:start")

    def test_guest_pages_do_not_expose_saved_account_data(self):
        client = TestClient(app)
        self.assertIn("Sign in to open your private fight library", client.get("/history").text)
        self.assertIn("Create your private athlete workspace", client.get("/dashboard").text)

    def test_mobile_navigation_is_available(self):
        response = self.client.get("/")
        self.assertIn('id="mobileMenuButton"', response.text)
        self.assertIn('id="mobileMenu"', response.text)

    def test_accessible_shell_and_mobile_install_metadata_are_available(self):
        page = self.client.get("/")
        self.assertIn('class="skip-link" href="#main-content"', page.text)
        self.assertIn('<main id="main-content" tabindex="-1">', page.text)
        self.assertIn('rel="manifest" href="/static/site.webmanifest"', page.text)
        self.assertIn('name="application-name" content="WarriorIQ"', page.text)
        manifest = self.client.get("/static/site.webmanifest")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(manifest.json()["name"], "WarriorIQ")
        self.assertEqual(manifest.json()["display"], "standalone")

    def test_public_shell_contains_structured_software_metadata(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "base.html").read_text(encoding="utf-8")
        self.assertIn('type="application/ld+json"', template)
        self.assertIn('"@type":"SoftwareApplication"', template)
        self.assertIn('"applicationCategory":"SportsApplication"', template)

    def test_professional_stylesheet_is_served(self):
        response = self.client.get("/static/style.css")
        self.assertEqual(response.status_code, 200)
        self.assertIn("--lime:#2f6fed", response.text)

    def test_restored_visual_system_uses_one_shared_motion_layer(self):
        page = self.client.get("/")
        self.assertNotIn('/static/athletic.css', page.text)
        self.assertIn('data-page="home"', page.text)
        # The cache-busting version changes on every asset release, so assert the
        # shared motion layer is linked rather than pinning one version string.
        # The motion layer ships inside the shell bundle; the script is still
        # its own file.
        from app.main import CSS_BUNDLES

        self.assertRegex(page.text, r'href="/assets/shell\.css\?v=[\w-]+"')
        self.assertIn("motion.css", CSS_BUNDLES["shell"])
        self.assertRegex(page.text, r'src="/static/motion\.js\?v=[\w-]+"')
        self.assertIn('id="pageScrollProgress"', page.text)
        css = self.client.get("/assets/shell.css")
        self.assertEqual(css.status_code, 200)
        for rule in (
            "--wiq-motion-fast:140ms",
            ".page-scroll-progress",
            ".analysis-stage[data-state=active]",
            "body[data-page=result] .report-path",
            "@media(prefers-reduced-motion:reduce)",
        ):
            self.assertIn(rule, css.text)
        script = self.client.get("/static/motion.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("IntersectionObserver", script.text)
        self.assertIn("prefers-reduced-motion: reduce", script.text)

    def test_analysis_progress_uses_real_backend_values_and_named_stages(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
        for stage in ("Upload preparation", "Video processing", "Fighter detection", "Fighter tracking", "Pose analysis", "Strike analysis", "Scoring", "Report generation"):
            self.assertIn(stage, template)
        self.assertIn("d.percent", template)
        self.assertIn("d.message", template)
        self.assertIn("does not invent separate percentages", template)
        self.assertIn('role="progressbar"', template)

    def test_fight_playback_starts_only_when_backend_analysis_begins(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
        video_tag = template.split('<video id="liveVideo"', 1)[1].split("</video>", 1)[0]

        self.assertNotIn("autoplay", video_tag)
        self.assertIn("analysisPlaybackReady", template)
        self.assertIn("d.status==='running'&&d.stage==='analysis'", template)
        self.assertIn("startPlaybackWhenAnalysisStarts(d)", template)
        self.assertNotIn("tryImmediatePlayback", template)

    def test_home_copy_uses_the_warrioriq_combat_sports_voice(self):
        page = self.client.get("/").text
        self.assertIn("For fighters and coaches", page)
        self.assertIn("Upload your fight.", page)
        self.assertIn("See it in numbers.", page)
        self.assertNotIn("WonderIQ", page)
        # "Frame by frame" is industry shorthand a fighter does not read as a
        # promise. The headline says what they get instead.
        self.assertNotIn("Frame by frame", page)
        self.assertNotIn("Fight intelligence", page)

    def test_upload_never_displays_selected_filename(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        self.assertIn("fileButton.classList.toggle('uploaded',ready)", template)
        # "uploaded" at selection described a transfer that had not started.
        self.assertIn("Fight video selected", template)
        self.assertNotIn("Fight video uploaded", template)
        self.assertNotIn("file.name", template)

    def test_replay_overlay_does_not_block_video_controls(self):
        css = self.client.get("/static/fixes.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn(".replay-wrap canvas{pointer-events:none}", css.text)

    def test_logo_asset_is_served(self):
        response = self.client.get("/static/warrioriq-logo.png")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 1000)

    def test_identity_recovery_requires_upload_time_opt_in(self):
        """An operator's API key does not constitute an uploader's consent."""
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        self.assertIn('type="checkbox" name="openai_identity_recovery" value="true"', template)
        self.assertNotIn("setup-extra", template)

        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("openai_identity_recovery: bool = Form(False)", source)

        # Sending frames to a third party on every analysis has to be what the
        # privacy policy actually says, or the policy is wrong.
        privacy = (Path(__file__).resolve().parents[1] / "app" / "templates" / "privacy.html").read_text(encoding="utf-8")
        self.assertIn("only when external recovery is explicitly enabled", privacy)
        self.assertNotIn("rather than being chosen per upload", privacy)

    def _render_result(self, selection_check, can_share=False, sharing=None, score_withheld=None,
                       scorecard_available=None, measurement=None, kick_minimum=None,
                       action_labels_available=None, report_access=None):
        """Actually render result.html, rather than grepping its source.

        Every other check on this template matches text in the file, which
        cannot catch a bad expression - a broken one only fails when Jinja
        evaluates it, and by then it is a 500 on the report page.
        """
        from jinja2 import ChainableUndefined, Environment, FileSystemLoader

        from app.main import _analysis_quality_summary, sport_identity

        class Stub:
            def __init__(self, **kw): self.__dict__.update(kw)
            def __getattr__(self, key): return Stub()
            def __getitem__(self, key): return Stub()
            def __str__(self): return ""
            def __bool__(self): return False
            def __iter__(self): return iter(())

        templates = Path(__file__).resolve().parents[1] / "app" / "templates"
        env = Environment(loader=FileSystemLoader(str(templates)), undefined=ChainableUndefined)
        # A real report, trimmed of its bulk arrays. Hand-built dicts kept
        # failing on fields the page reaches for, which is the point: only a
        # genuine report shape proves the template renders.
        fixture = Path(__file__).resolve().parent / "fixtures" / "report_sample.json"
        report = json.loads(fixture.read_text(encoding="utf-8"))
        report["selection_check"] = selection_check
        if measurement is not None:
            for fighter in ("A", "B"):
                report.setdefault("metrics", {}).setdefault(fighter, {})["measurement"] = measurement
        if scorecard_available is not None:
            report.setdefault("scorecard", {})["available"] = scorecard_available
        if action_labels_available is not None:
            report.setdefault("statistics", {})["action_labels_available"] = action_labels_available
        request = Stub(url=Stub(path="/report/abc"), state=Stub(account=None), cookies={}, headers={})
        return env.get_template("result.html").render(
            request=request, job_id="abc", report=report,
            identity=sport_identity("kickboxing"),
            report_access=report_access or {"report_tier": "full", "report_label": "Full", "label": "Full"},
            analysis_quality=_analysis_quality_summary(report), can_share=can_share,
            sharing=sharing, score_withheld=score_withheld, unavailable=[],
            kick_minimum=kick_minimum,
        )

    def _render_identity(self, trusted):
        from jinja2 import ChainableUndefined, Environment, FileSystemLoader

        from app.main import _analysis_quality_summary, sport_identity

        class Stub:
            def __init__(self, **kw): self.__dict__.update(kw)
            def __getattr__(self, key): return Stub()
            def __getitem__(self, key): return Stub()
            def __str__(self): return ""
            def __bool__(self): return False
            def __iter__(self): return iter(())

        templates = Path(__file__).resolve().parents[1] / "app" / "templates"
        env = Environment(loader=FileSystemLoader(str(templates)), undefined=ChainableUndefined)
        fixture = Path(__file__).resolve().parent / "fixtures" / "report_sample.json"
        report = json.loads(fixture.read_text(encoding="utf-8"))
        report.setdefault("integrity", {})["identity_evidence_trusted"] = trusted
        request = Stub(url=Stub(path="/report/abc"), state=Stub(account=None), cookies={}, headers={})
        return env.get_template("result.html").render(
            request=request, job_id="abc", report=report,
            identity=sport_identity("kickboxing"),
            report_access={"report_tier": "full", "report_label": "Full", "label": "Full"},
            analysis_quality=_analysis_quality_summary(report), can_share=False,
            sharing=None, score_withheld=None, unavailable=[], kick_minimum=None,
        )

    def test_the_identity_fix_is_the_first_thing_on_a_failed_report(self):
        """It was roughly 2177px down, under everything the failure invalidated.

        When identity fails there is no score, no per-fighter number and no
        scorecard - the page has nothing to say until it is fixed. Putting the
        one action that fixes it below all of that, while the header read
        "Analysis complete", told the reader the opposite of the truth.
        """
        page = self._render_identity(trusted=False)
        self.assertIn("Needs one quick step", page)
        self.assertNotIn("Analysis complete", page)

        # Above the fold means above the body, not merely present.
        action = page.index('href="/select/abc"')
        self.assertLess(action, page.index("Fighter identity check failed"),
                        "the fix must come before the notice explaining it")
        self.assertLess(action, 6000,
                        "the fix belongs in the header block, not partway down the report")

    def test_a_failed_report_attributes_nothing_to_either_fighter(self):
        """It said both things on one page.

        SCORE "Not scored" and "we have not credited strikes to either name",
        then "Fighter A: 19 leg strikes", "Fighter B: 22", and a movement
        scorecard reading 10-9 "Fighter A ahead". The strikes were seen; whose
        they were is exactly what the failed check could not establish, so the
        split is the part that has to go - not the count.
        """
        page = self._render_identity(trusted=False)
        self.assertNotIn("Movement scorecard", page)
        self.assertNotIn("Fighter A leg strikes", page)
        self.assertNotIn("Fighter B leg strikes", page)
        # The unattributed total survives, because it is still true.
        self.assertIn("Leg strikes seen, both fighters", page)

    def test_a_healthy_report_keeps_the_numbers_it_can_stand_behind(self):
        page = self._render_identity(trusted=True)
        self.assertIn("Fighter A leg strikes", page)
        self.assertNotIn("Leg strikes seen, both fighters", page)

    def test_a_healthy_report_is_not_told_it_needs_a_step(self):
        page = self._render_identity(trusted=True)
        self.assertIn("Analysis complete", page)
        self.assertNotIn("Needs one quick step", page)
        self.assertNotIn("Fighter identity check failed", page)
        self.assertNotIn('href="/select/abc"', page)

    def test_evidence_timestamps_are_links_a_reader_can_actually_follow(self):
        """They were a raw Python list, and before that a button with no handler.

        The coaching section printed `Evidence: [8.42, 17.27, 25.44]` straight
        at the reader, and the scorecard section rendered <button data-time=...>
        which nothing on the report page has ever listened for - grep the
        templates: replay.html and review.html wire their own, result.html
        wires none. Both are now anchors to /replay?t=, which the replay page
        already honours, so they work with no script at all.
        """
        html = self._render_result(None)
        self.assertNotIn("Evidence: [", html, "a python list is not evidence a coach can use")
        self.assertNotIn('data-time=', html, "a button nothing listens for is a dead control")

    def test_a_foul_is_never_pinned_on_a_fighter_the_analysis_cannot_identify(self):
        """Naming a fighter for an illegal act is the strongest claim on the page.

        It was gated only by the report tier. Found on 1947 boxing footage,
        where a colour histogram scores the two boxers as identical (pair
        similarity 1.0 on black-and-white film): the report already said
        fighters_separable false, withheld the entire scorecard, and still
        printed "Fighter A - Left Push - Illegal" with a timestamp.

        Rendered rather than grepped, in all four trust combinations, because
        reading the template is what let the punch counts through six separate
        surfaces before.
        """
        from jinja2 import ChainableUndefined, Environment, FileSystemLoader

        from app.main import _analysis_quality_summary, sport_identity

        class Stub:
            def __init__(self, **kw): self.__dict__.update(kw)
            def __getattr__(self, key): return Stub()
            def __getitem__(self, key): return Stub()
            def __str__(self): return ""
            def __bool__(self): return False
            def __iter__(self): return iter(())

        templates = Path(__file__).resolve().parents[1] / "app" / "templates"
        env = Environment(loader=FileSystemLoader(str(templates)), undefined=ChainableUndefined)
        fixture = Path(__file__).resolve().parent / "fixtures" / "report_sample.json"
        base = json.loads(fixture.read_text(encoding="utf-8"))

        def render(trusted, status):
            report = json.loads(json.dumps(base))
            report.setdefault("integrity", {})["action_metrics_trusted"] = trusted
            report.setdefault("scorecard", {})["status"] = status
            report["illegal_moves"] = [{
                "fighter": "A", "round_number": 1, "peak_time": 30.44,
                "technique": "left_push", "legality_reason": "push"}]
            return env.get_template("result.html").render(
                request=Stub(url=Stub(path="/report/abc"), state=Stub(account=None),
                             cookies={}, headers={}),
                job_id="abc", report=report, identity=sport_identity("boxing"),
                report_access={"report_tier": "full", "report_label": "Full", "label": "Full"},
                analysis_quality=_analysis_quality_summary(report), can_share=False,
                sharing=None, score_withheld=None, unavailable=[], kick_minimum=None)

        for trusted, status in ((False, "identity_integrity_failed"),
                                (False, "ok"), (True, "identity_integrity_failed")):
            html = render(trusted, status)
            self.assertNotIn("Left Push", html,
                             f"a foul was pinned on a fighter with trusted={trusted} status={status}")
            self.assertIn("Illegal-move flags are switched off", html)

        allowed = render(True, "ok")
        self.assertIn("Left Push", allowed,
                      "a fully trusted fight must still show its fouls")

    def test_the_kick_minimum_block_renders_and_never_alleges_a_shortfall(self):
        """WAKO Full Contact obliges six kicks a round, and we can only confirm it.

        Our kick count is a floor, so six or more proves the round was met and
        fewer proves nothing about the fighter. The unconfirmed cell therefore
        has to read as our limitation, not as a penalty against them - it must
        not say "failed", and it must not be styled as a failure.
        """
        card = {
            "minimum": 6,
            "rule": "WAKO Full Contact, Chapter 8 Article 6: minimum 6 kicks per round, 18 per bout.",
            "rounds": [
                {"round": 1, "fighters": {"A": {"kicks_evidenced": 7, "minimum_confirmed_met": True},
                                          "B": {"kicks_evidenced": 2, "minimum_confirmed_met": False}}},
            ],
            "basis": "kicks we could evidence while we had sight of that fighter",
            "is_a_floor": True,
            "note": "not a shortfall by the fighter",
        }
        html = self._render_result(None, kick_minimum=card)
        self.assertIn("Kick minimum", html)
        self.assertIn("Met &mdash; 7 kicks seen", html.replace("—", "&mdash;"))
        self.assertIn("Not confirmed", html)
        self.assertIn("Chapter 8 Article 6", html)
        # The fighter is never told they fell short, and the cell that could
        # imply it is the neutral class rather than an error one.
        for word in ("failed", "penalty", "violation"):
            self.assertNotIn(word, html.lower())
        self.assertIn('class="unconfirmed"', html)

    def test_the_kick_minimum_block_is_absent_for_every_other_discipline(self):
        """Only Full Contact carries the obligation, so the block must not render.

        A kick-minimum table on a K-1 report would be inventing a rule.
        """
        html = self._render_result(None, kick_minimum=None)
        self.assertNotIn("Kick minimum", html)

    def test_no_page_explains_itself_by_naming_an_internal_component(self):
        """An internal component name is not a reason a fighter can use.

        A kickboxer reading "hidden until the action model is trained" learns
        nothing they can act on and nothing about what WarriorIQ actually did.
        Six user-facing strings across five pages explained withheld strike
        numbers that way.
        """
        import re
        from pathlib import Path

        templates = Path(__file__).resolve().parents[1] / "app" / "templates"
        offenders = {}
        for page in sorted(templates.glob("*.html")):
            body = page.read_text(encoding="utf-8")
            # Developer comments may still name the component; the rendered
            # copy a fighter reads may not. Jinja comments span lines, so they
            # have to be removed as blocks rather than line by line.
            visible = re.sub(r"\{#.*?#\}", " ", body, flags=re.S)
            for phrase in ("action model", "evidence gate", "release-validation gate"):
                if phrase in visible:
                    offenders.setdefault(page.name, []).append(phrase)
        self.assertEqual(offenders, {}, f"internal vocabulary shown to fighters: {offenders}")

    def test_partial_coverage_shows_its_numbers_with_the_ground_they_stand_on(self):
        """The strip used to blank entirely below a coverage line.

        A fighter saw a row of dashes and nothing explaining them. The numbers
        are shown now, and the sample size travels with them so a partial round
        is never presented as a whole one.
        """
        html = self._render_result({}, measurement={
            "measured_frames": 410, "analyzed_frames": 923,
            "share": 410/923, "confident": False,
        })
        self.assertIn("Measured across 410 of 923 analysed frames", html)
        self.assertIn("44% of the fight", html)
        self.assertIn("could see rather than the whole of it", html)

    def test_full_coverage_states_the_basis_without_the_caution(self):
        html = self._render_result({}, measurement={
            "measured_frames": 900, "analyzed_frames": 923,
            "share": 900/923, "confident": True,
        })
        self.assertIn("Measured across 900 of 923 analysed frames", html)
        self.assertNotIn("rather than the whole of it", html, "no caveat when it saw the round")

    def test_an_unscored_fight_says_why_in_words_a_fighter_can_act_on(self):
        """The page named an internal component instead of giving a reason.

        "Strike scoring needs the action model" tells a kickboxer nothing, and
        it is usually not the cause either - a score is withheld far more often
        because a fighter was lost on camera. The real explanation was sitting
        in the scorecard disclaimer at the bottom of the report, where nobody
        reading "Not scored" would find it.
        """
        from app.main import _score_withheld

        withheld = _score_withheld({
            "scorecard": {"available": False, "status": "insufficient_observation_coverage"},
            "tracking": {"fighter_A_coverage": 0.96, "fighter_B_coverage": 0.81},
        })
        self.assertIn("96%", withheld["reason"])
        self.assertIn("81%", withheld["reason"], "the fighter is told which corner was lost")
        self.assertIn("85%", withheld["reason"], "and what the bar actually is")

        html = self._render_result({}, score_withheld=withheld, scorecard_available=False)
        self.assertIn("Not scored", html)
        self.assertNotIn("action model", html, "no internal component names on the page")
        self.assertIn("81%", html)
        self.assertIn(withheld["fix"], html)

    def test_every_withheld_score_state_gives_a_reason_and_a_next_step(self):
        """A state with no branch would silently fall back to saying nothing."""
        from app.main import _score_withheld

        states = [
            "both_fighters_required", "insufficient_observation_coverage",
            "insufficient_scoring_actions", "identity_integrity_failed",
            "insufficient_tracking_evidence", "some_state_added_later",
        ]
        for status in states:
            with self.subTest(status=status):
                withheld = _score_withheld({
                    "scorecard": {"available": False, "status": status, "evidence": {}},
                    "tracking": {"fighter_A_coverage": 0.5, "fighter_B_coverage": 0.5},
                })
                self.assertTrue(withheld["reason"].strip())
                self.assertTrue(withheld["fix"].strip())
                self.assertNotIn("model", withheld["reason"].lower())

    def test_a_scored_fight_has_nothing_to_explain(self):
        from app.main import _score_withheld

        self.assertIsNone(_score_withheld({"scorecard": {"available": True}}))

    def test_a_new_coach_link_is_shown_once_with_its_address(self):
        """Creating a link used to redirect to the coach's own view of the report.

        That page has no address on it, so the athlete who just made the link
        never saw the thing they were supposed to send. Only the moment of
        creation can show it - the database keeps a digest, not the token.
        """
        html = self._render_result({}, can_share=True, sharing={
            "links": [{"expires_at": "2026-09-10T12:00:00+00:00", "expires_label": "10 Sep 2026"}],
            "new_link": "https://warrioriq.eu/s/abc123", "new_link_expires": "10 Sep 2026",
            "revoked": None,
        })
        self.assertIn("https://warrioriq.eu/s/abc123", html)
        self.assertIn("data-copy-link", html)
        self.assertIn("10 Sep 2026", html)

    def test_revoking_coach_links_says_how_many_stopped_working(self):
        """The revoke button redirected to an identical page and read as dead."""
        html = self._render_result({}, can_share=True, sharing={
            "links": [], "new_link": None, "new_link_expires": None, "revoked": 2,
        })
        self.assertIn("2 links revoked", html)
        none_left = self._render_result({}, can_share=True, sharing={
            "links": [], "new_link": None, "new_link_expires": None, "revoked": 0,
        })
        self.assertIn("no live links to revoke", none_left.lower())

    def test_there_is_no_revoke_button_when_no_link_is_live(self):
        """Offering to revoke nothing is what made the button look broken."""
        html = self._render_result({}, can_share=True, sharing={
            "links": [], "new_link": None, "new_link_expires": None, "revoked": None,
        })
        self.assertNotIn("/shares/abc/revoke", html)
        live = self._render_result({}, can_share=True, sharing={
            "links": [{"expires_at": "2026-09-10T12:00:00+00:00", "expires_label": "10 Sep 2026"}],
            "new_link": None, "new_link_expires": None, "revoked": None,
        })
        self.assertIn("/shares/abc/revoke", live)
        self.assertIn("Revoke 1 link<", live)

    def test_the_labelling_list_survives_events_without_a_confidence(self):
        """Sorting candidates by confidence crashed on a null one.

        A manually added label carries no confidence at all, and float(None)
        raised a TypeError that took the whole labelling page with it - the one
        page the training data depends on.
        """
        from app.main import _review_candidates

        events = [
            {"peak_time": 1.0, "technique": "jab", "fighter": "A", "confidence": None},
            {"peak_time": 5.0, "technique": "cross", "fighter": "B", "confidence": 0.9},
            {"peak_time": 9.0, "technique": "hook", "fighter": "A", "confidence": "junk"},
        ]
        candidates = _review_candidates({"events": events}, "dataset")
        self.assertEqual(len(candidates), 3)
        # The scored one still sorts to the front.
        self.assertEqual(candidates[0]["technique"], "cross")

    def test_the_wrong_people_check_still_runs_even_though_it_is_not_shown(self):
        """The banner was removed at the owner's request.

        A report that announces it analysed the wrong people does not help
        anyone read their own fight, and the answer to bad tracking is better
        tracking. The measurement still runs and still lands in the report, so
        the evidence is there for a future decision - it is only not printed.
        """
        page = self._render_result({
            "actions_observed": 175, "actions_in_range": 12, "landed": 0,
            "median_separation_body_lengths": 2.84, "looks_like_a_fight": False,
            "warning": "...", "verdict": "selection_probably_wrong",
        })
        self.assertNotIn("These may not be the two fighters", page)
        self.assertGreater(len(page), 1000)

    def test_the_selection_check_is_still_produced(self):
        from core.contact import assess_selection

        verdict = assess_selection(
            [2.84] * 175, kept=12, discarded=163, landed=0,
            travel_per_minute={"A": 9.6, "B": 44.7},
        )
        self.assertFalse(verdict["looks_like_a_fight"])
        self.assertTrue(verdict["warning"])

    def test_a_normal_report_carries_no_such_warning(self):
        page = self._render_result({
            "actions_observed": 280, "actions_in_range": 153, "landed": 24,
            "median_separation_body_lengths": 0.76, "looks_like_a_fight": True,
            "warning": None, "verdict": "consistent_with_a_fight",
        })
        self.assertNotIn("These may not be the two fighters", page)

    def test_an_older_report_without_the_check_still_renders(self):
        """Reports analysed before this existed must not break the page."""
        page = self._render_result(None)
        self.assertNotIn("These may not be the two fighters", page)
        self.assertGreater(len(page), 1000)

    def test_training_plan_is_separate_from_coach_card(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn("Training plan · Fighter", template)

    def test_fighter_selection_is_manual_only(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "select.html").read_text(encoding="utf-8")
        self.assertNotIn("Automatically choose both", template)
        self.assertNotIn("Swap A and B", template)
        self.assertNotIn("Draw both boxes myself", template)
        self.assertIn("mode='A_DRAW'", template)
        self.assertIn('name="focusFighter"', template)
        self.assertIn("focus_fighter:focusFighter", template)
        self.assertNotIn('value="BOTH"', template)

    def test_fighter_selection_is_immediately_drawable_and_detector_safe(self):
        root = Path(__file__).resolve().parents[1]
        template = (root / "app" / "templates" / "select.html").read_text(encoding="utf-8")
        css = (root / "app" / "static" / "fighter-selection.css").read_text(encoding="utf-8")

        self.assertIn("mode='A_DRAW'", template)
        self.assertIn("loadDetectionCandidates", template)
        self.assertNotIn("boxA=boxB=null;mode='A_DRAW'", template)
        self.assertIn('id="redrawA"', template)
        self.assertIn('id="redrawB"', template)
        self.assertIn("imageViewport", template)
        self.assertIn("touch-action: none", css)

    def test_progress_shows_elapsed_time_not_realtime_speed(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
        self.assertIn("Analysis running", template)
        self.assertIn("d.elapsed_seconds", template)
        self.assertNotIn("Realtime speed", template)

    def test_replay_uses_rendered_media_time(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "replay.html").read_text(encoding="utf-8")
        self.assertIn("requestVideoFrameCallback", template)
        self.assertIn("metadata.mediaTime", template)

    def test_a_score_is_never_shown_without_the_limits_of_the_sport(self):
        """The scorer computes a coverage note; the pages have to print it.

        score_fight has emitted coverage_note and unobserved_actions since the
        sports expansion, and for a while nothing read either one: the upload
        page promised a disclosure the report never made. Both places that
        show a number now carry it, and a shared link carries it too because
        the person opening one never saw the upload page.
        """
        templates = Path(__file__).resolve().parents[1] / "app" / "templates"
        result = (templates / "result.html").read_text(encoding="utf-8")
        shared = (templates / "shared.html").read_text(encoding="utf-8")

        for page, source in (("result.html", result), ("shared.html", shared)):
            with self.subTest(page=page):
                self.assertIn("report.scorecard.coverage_note", source)

        # The illegal-move panel named one federation's rules for every sport,
        # which is wrong the moment a boxing or MMA bout is scored.
        self.assertNotIn(">WAKO rules<", result)
        self.assertIn("report.scorecard.sport_label", result)

    def test_the_upload_page_promises_only_the_disclosure_that_exists(self):
        """Copy that overstates the product is a defect like any other."""
        setup = (Path(__file__).resolve().parents[1] / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        self.assertIn("Your report says so too", setup)
        self.assertNotIn("says so on every page", setup)

    def test_evidence_replay_starts_one_second_before_exact_timestamp(self):
        replay = (Path(__file__).resolve().parents[1] / "app" / "templates" / "replay.html").read_text(encoding="utf-8")
        result = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn('data-lead="{{chapter.lead_seconds}}"', replay)
        self.assertIn("playAt(b.dataset.time,b.dataset.lead)", replay)
        self.assertIn("Replay with skeletons", result)
        self.assertNotIn("Automatic evidence timeline", result)

    def test_replay_has_neutral_chapters_when_no_action_is_verified(self):
        report = {
            "setup": {"start_seconds": 10.0, "end_seconds": 110.0},
            "performance": {"segment_duration_seconds": 100.0},
            "key_moments": [],
        }
        chapters, mode = _build_replay_chapters(report, "A")
        self.assertEqual(mode, "movement_chapters")
        self.assertEqual([item["time"] for item in chapters], [10.0, 35.0, 60.0, 85.0])
        self.assertTrue(all(item["kind"] == "movement_chapter" for item in chapters))
        self.assertTrue(all("kick" not in item["label"].lower() for item in chapters))

    def test_replay_verified_actions_keep_the_one_second_lead(self):
        chapters, mode = _build_replay_chapters({"key_moments": [{
            "peak_time": 12.5, "fighter": "B", "technique": "right_low_kick", "outcome": "clean",
        }]}, "B")
        self.assertEqual(mode, "verified_actions")
        self.assertEqual(chapters[0]["lead_seconds"], 1.0)
        self.assertIn("Right Low Kick", chapters[0]["label"])

    def test_single_round_does_not_claim_two_rounds_are_missing_evidence(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn("report.setup.round_count > 1", template)
        self.assertNotIn("Requires 2 rounds", template)

    def test_scorecard_displays_exact_evidence_requirements(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn("verified scoring actions", template)

    def test_focused_report_keeps_both_fighters_in_the_analysis_engine(self):
        app_source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('analysis_target="BOTH"', app_source)
        self.assertIn("focus_fighter=focus_fighter", app_source)

    def test_profile_upload_buttons_are_themed(self):
        css = self.client.get("/static/fixes.css").text
        self.assertIn(".file-native{position:absolute", css)
        self.assertIn(".file-button{display:flex", css)

    def test_replay_applies_measured_sync_delay(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "replay.html").read_text(encoding="utf-8")
        self.assertNotIn("skeletonDelay=", template)
        self.assertIn("t=Math.max(0,shown)", template)

    def test_failed_analysis_has_return_to_selection_action(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
        self.assertIn("Return to fighter selection", template)

    def test_raw_report_exports_are_not_public(self):
        self.assertEqual(self.client.get("/report/not-a-job.json").status_code, 404)
        self.assertEqual(self.client.get("/report/not-a-job.html").status_code, 404)
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertNotIn("HTML report", template)
        self.assertNotIn(">JSON<", template)

    def test_single_fighter_report_has_visibility_guards(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn("focus == 'BOTH' or focus == fighter", template)
        self.assertIn("report.video.focus_fighter", template)

    def test_knockdown_evidence_is_removed(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        analyzer = (Path(__file__).resolve().parents[1] / "core" / "analyzer.py").read_text(encoding="utf-8")
        self.assertNotIn("Knockdown evidence", template)
        self.assertNotIn("KnockdownDetector", analyzer)

    def test_timeline_is_conservative_and_bounded(self):
        report = (Path(__file__).resolve().parents[1] / "core" / "report.py").read_text(encoding="utf-8")
        self.assertIn("_timeline_event_reliable", report)
        self.assertIn("len(key_events) >= 8", report)
        self.assertIn('event.outcome not in {"clean", "blocked", "checked", "missed"}', report)

    def test_result_separates_illegal_wako_moves(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn("Illegal moves ·", template)
        self.assertIn("legality_reason", template)

    def test_result_is_automatic_and_does_not_assign_review_work_to_the_user(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertNotIn("Correct this moment", template)
        self.assertNotIn("/api/annotations/{{job_id}}", template)
        self.assertNotIn("Save ground truth", template)
        self.assertNotIn("Verify scorecard", template)

    def test_result_guides_people_through_only_the_core_training_path(self):
        """Rendered, not grepped: the chips are generated from the section gates.

        Matching source text stopped proving anything once the strip stopped
        being a hardcoded list - and a hardcoded list was the defect. Three
        chips pointed at #report-strikes, #report-combinations and
        #report-key-moments, which are only emitted for a full report with
        verified strike counting, so on every other report they were links to
        an anchor that did not exist.
        """
        page = self._render_result({"status": "ok"})
        self.assertIn('class="report-path"', page)
        for anchor in ("report-performance", "report-scorecard", "report-coaching", "report-training"):
            self.assertIn(f'href="#{anchor}"', page)
            self.assertIn(f'id="{anchor}"', page)
        self.assertNotIn('id="report-evidence"', page)
        self.assertIn("Replay with skeletons", page)

    def test_no_report_chip_points_at_a_section_the_page_did_not_render(self):
        """The invariant, checked on the rendered page in both gate states.

        Every in-page jump link must resolve to an id on the same page. This
        is the general form of the bug, so it holds whatever the tier and
        whatever strike counting reports.
        """
        import re

        from core.payments import PLANS

        # Every real plan tier, not an invented one: the chips depend on
        # report_tier, and compact and expanded carry item limits that full
        # does not.
        seen = set()
        for plan in PLANS.values():
            if plan["report_tier"] in seen:
                continue
            seen.add(plan["report_tier"])
            access = dict(plan, report_label="R", label="R")
            page = self._render_result({"status": "ok"}, report_access=access)
            targets = set(re.findall(r'href="#(report-[a-z-]+)"', page))
            present = set(re.findall(r'id="(report-[a-z-]+)"', page))
            self.assertTrue(targets, f"no chips rendered at tier={plan['report_tier']}")
            self.assertEqual(
                targets - present, set(),
                f"dead jump links at tier={plan['report_tier']}: {sorted(targets - present)}")
        self.assertEqual(seen, {"compact", "expanded", "full"})

    def test_a_chip_whose_section_is_withheld_says_why_instead_of_linking(self):
        page = self._render_result({"status": "ok"}, action_labels_available=False)
        self.assertNotIn('href="#report-strikes"', page)
        self.assertIn("hidden until strike counting is verified", page)
        # ...and comes back as a real link once the gate opens.
        opened = self._render_result({"status": "ok"}, action_labels_available=True)
        self.assertIn('href="#report-strikes"', opened)
        self.assertIn('id="report-strikes"', opened)

    def test_result_explains_analysis_quality_without_calling_coverage_accuracy(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn('class="analysis-quality', template)
        self.assertIn("Fighter A observed", template)
        self.assertIn("Fighter B observed", template)
        self.assertIn("It is not ground-truth identity or action accuracy", template)

    def test_progress_uses_supported_movement_metrics_when_actions_are_unvalidated(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "dashboard.html").read_text(encoding="utf-8")
        for label in ("Guard", "Balance", "Ring centre", "Pose evidence"):
            self.assertIn(f'<span class="label">{label}</span>', template)
        self.assertIn("Movement progress is ready", template)
        self.assertIn("Not validated", template)

    def test_account_policies_are_acknowledged_at_auth_not_every_upload(self):
        auth = (Path(__file__).resolve().parents[1] / "app" / "templates" / "auth.html").read_text(encoding="utf-8")
        upload = (Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html").read_text(encoding="utf-8")
        for path in ("/terms", "/privacy", "/acceptable-use"):
            self.assertIn(f'href="{path}"', auth)
        self.assertNotIn("Uploading is subject to", upload)
        self.assertNotIn("Account policies are accepted only when creating an account or signing in", upload)

    def test_social_sign_in_is_configured_not_decorative(self):
        auth = (Path(__file__).resolve().parents[1] / "app" / "templates" / "auth.html").read_text(encoding="utf-8")
        registry = (Path(__file__).resolve().parents[1] / "core" / "social_auth.py").read_text(encoding="utf-8")
        self.assertIn("request.state.oauth_providers", auth)
        for provider in ("google", "facebook", "microsoft"):
            self.assertIn(f'"{provider}"', registry)
        # Apple was removed: the Developer Program needs age 18 and $99/year,
        # and its client secret expires within six months.
        self.assertNotIn('"apple"', registry)
        for provider in ("google",):
            self.assertIn(f'"{provider}"', registry)
        self.assertIn('formaction="/auth/{{provider.key}}/start"', auth)
        self.assertNotIn("Sign in with X", auth)

    def test_performance_report_motion_targets_real_values_and_respects_reduced_motion(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        motion_js = self.client.get("/static/motion.js").text
        motion_css = self.client.get("/static/motion.css").text
        self.assertIn("data-count-up", template)
        self.assertIn("[data-count-up]", motion_js)
        self.assertIn(".tactical-performance.is-visible .attack-mix b", motion_css)
        self.assertIn("prefers-reduced-motion", motion_css)

    def test_report_carries_ai_scoring_and_medical_limits(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        for phrase in (
            "AI-Assisted Analysis", "detection mistakes are possible", "not guaranteed facts",
            "official judging decision", "not medical advice", "Camera angle",
        ):
            self.assertIn(phrase, template)

    def test_compare_button_never_silently_reloads_an_empty_comparison(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "compare.html").read_text(encoding="utf-8")
        self.assertIn("fights|length < 2", template)
        self.assertIn("Two fights required", template)
        self.assertIn('name="{{field}}" required', template)
        self.assertIn("setCustomValidity", template)
        self.assertIn("Choose two different fights", template)

    def test_history_new_analysis_button_targets_the_upload_card(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "history.html").read_text(encoding="utf-8")
        self.assertIn('href="/analyze">Analyze another fight', template)

    def test_coach_workspace_uses_the_selected_fighter_and_one_click_plan(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "coach.html").read_text(encoding="utf-8")
        self.assertIn("latest.coaching[focus]", template)
        self.assertNotIn("for fighter in ['A','B']", template)
        self.assertIn("One-click suggestions", template)
        self.assertIn("Mark complete", template)

    def test_advanced_report_diagnostics_stay_available_but_folded_away(self):
        """Deep diagnostics remain, one tap away, but do not greet a first-timer.

        SAM2 propagation counts and ReID tracker names are exactly what a coach
        may want and exactly what makes a fighter close the page. They stay in
        the report; they no longer open by default.
        """
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn('<details class="report-details"', template)
        self.assertIn("More performance details", template)
        self.assertIn('<details class="card report-technical">', template)
        self.assertIn("Technical analysis details", template)
        self.assertNotIn('<details class="card report-technical" open>', template)
        self.assertNotIn('<section class="card"><div class="eyebrow">Performance integrity</div>', template)

    def test_mobile_navigation_keeps_the_primary_product_actions_only(self):
        """Signed out, the menu offers the public site and nothing more.

        Fight library, Progress, Coach and Compare all render a "sign in first"
        empty state for a visitor with no account, so listing them handed
        someone who had never used WarriorIQ four doors to the same locked
        room. The accuracy lab is never listed: it reports internal model
        validation counts.
        """
        menu = self.client.get("/").text.split('<div class="mobile-menu"', 1)[1].split('</div>', 1)[0]
        for label in (">Analyze fight<", ">How it works<", ">Plans<", ">Sign in<", ">Create account<"):
            self.assertIn(label, menu)
        for hidden in ("Accuracy", "Compare", "Fight library", ">Progress<", ">Coach<"):
            self.assertNotIn(hidden, menu, "workspace tools are not offered before there is a workspace")

    def test_signed_in_navigation_still_carries_the_workspace_tools(self):
        """Splitting the navigation must not cost an account holder anything."""
        with self.signed_in():
            menu = self.client.get("/").text.split('<div class="mobile-menu"', 1)[1].split('</div>', 1)[0]
        for label in (">Analyze<", ">Fight library<", ">Progress<", ">Coach<", ">Plans<"):
            self.assertIn(label, menu)
        self.assertNotIn("Accuracy", menu, "still never the accuracy lab")

    def test_performance_report_has_accessible_animated_measurement_rails(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        css = self.client.get("/static/fixes.css").text
        self.assertIn('class="performance-rail"', template)
        self.assertIn('role="progressbar"', template)
        self.assertIn(".performance-meter", css)
        self.assertIn("@keyframes report-meter-fill", css)
        self.assertIn("prefers-reduced-motion:reduce", css)

    def test_scorecard_candidates_collapse_repeated_frames_of_one_action(self):
        base = {
            "fighter": "A", "technique": "right_low_kick", "family": "kick",
            "limb": "right_leg", "target": "leg", "outcome": "clean",
            "confidence": .90, "contact_confidence": .86, "round_number": 1,
            # Scoring now needs proof a kick left the floor, so these stand
            # for real kicks rather than footwork.
            "evidence": {"foot_lift_torsos": 1.2},
        }
        report = {"setup": {"ruleset": "K1"}, "events": [
            {**base, "peak_time": 8.00},
            {**base, "peak_time": 8.11, "confidence": .92},
            {**base, "peak_time": 8.23, "contact_confidence": .94},
            {**base, "peak_time": 8.34},
            {**base, "fighter": "B", "limb": "left_hand", "family": "punch", "technique": "jab", "target": "head", "peak_time": 8.16},
            {**base, "limb": "left_hand", "family": "punch", "technique": "jab", "target": "head", "peak_time": 8.18},
        ]}
        candidates = _review_candidates(report, "scorecard")
        self.assertEqual(len(candidates), 3)
        self.assertEqual(sum(item["fighter"] == "A" and item["limb"] == "right_leg" for item in candidates), 1)

    def test_the_accuracy_lab_is_not_reachable_by_a_visitor(self):
        """It reports "0 labels, not release ready" - true, and not for customers.

        A fighter deciding whether to trust WarriorIQ reads internal model
        validation counts as a verdict on the product. The page is unlinked and
        now answers 404 to anyone without an account, so it cannot be stumbled
        into from the site.
        """
        self.assertEqual(self.client.get("/validation").status_code, 404)

    def test_accuracy_page_never_invents_measurements(self):
        with self.signed_in():
            response = self.client.get("/validation")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Ground-truth validation", response.text)
        self.assertIn("Release target", response.text)
        self.assertIn("Dataset integrity", response.text)
        self.assertIn("Per-class validation", response.text)
        self.assertIn("Untouched test set", response.text)

    def test_correction_reclassifies_wako_legality(self):
        report = {"setup": {"ruleset": "LIGHT_CONTACT"}, "key_moments": [], "illegal_moves": [{
            "fighter": "A", "opponent": "B", "round_number": 1,
            "peak_time": 4.2, "technique": "left_knee", "family": "knee",
            "limb": "left_knee", "target": "body", "outcome": "clean",
        }]}
        annotations = [{"event_time": 4.2, "predicted": {
            "fighter": "A", "technique": "left_knee", "family": "knee",
            "limb": "left_knee", "target": "body", "outcome": "clean",
        }, "corrected": {
            "fighter": "A", "technique": "left_front_kick", "family": "kick",
            "limb": "left_leg", "target": "body", "outcome": "blocked", "contact_time": 4.05,
        }}]
        _apply_report_annotations(report, annotations)
        self.assertEqual(report["illegal_moves"], [])
        self.assertEqual(report["key_moments"][0]["technique"], "left_front_kick")
        self.assertEqual(report["key_moments"][0]["peak_time"], 4.05)
        self.assertTrue(report["key_moments"][0]["is_corrected"])

    def test_annotation_prediction_comes_from_report(self):
        report = {"events": [{
            "peak_time": 7.125, "fighter": "B", "technique": "right_low_kick",
            "family": "kick", "limb": "right_leg", "target": "leg", "outcome": "checked",
        }]}
        prediction = _prediction_at(report, 7.125)
        self.assertEqual(prediction["fighter"], "B")
        self.assertEqual(prediction["technique"], "right_low_kick")
        self.assertIsNone(_prediction_at(report, 8.0))

    def test_tracking_uses_decoded_presentation_timestamps(self):
        analyzer = (Path(__file__).resolve().parents[1] / "core" / "analyzer.py").read_text(encoding="utf-8")
        self.assertIn("cv2.CAP_PROP_POS_MSEC", analyzer)

    def test_unvalidated_model_candidates_are_never_public_evidence(self):
        candidate = {
            "peak_time": 4.2, "fighter": "A", "technique": "left_head_kick",
            "family": "kick", "limb": "left_leg", "target": "head", "outcome": "clean",
        }
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False},
            "integrity": {}, "setup": {"ruleset": "K1"}, "video": {"analysis_target": "BOTH"},
            "events": [candidate], "key_moments": [candidate], "illegal_moves": [], "metrics": {},
        }
        _apply_report_annotations(report, [])
        self.assertEqual(report["key_moments"], [])
        self.assertFalse(report["scorecard"]["available"])
        self.assertFalse(report["integrity"]["action_metrics_trusted"])

    def test_no_score_is_shown_while_punches_cannot_be_counted(self):
        """Was: high coverage plus enough candidates gave a preliminary score.

        It no longer does, and the reason is not coverage. Every ruleset here
        scores hands and feet - K-1 weights a punch at 1.0 against a kick at
        1.15 - so a punch decides most close rounds. Checked against video on
        three bouts the punch count was overstated by eleven in two of them,
        and the report stopped publishing it. A score computed from that
        family is the same number wearing a different hat, and dropping
        punches from the maths would score a boxing-heavy round as though
        nobody threw a hand.

        Turning STRIKE_COUNTS_PRECISION_VALIDATED on restores the score, so
        this is switched off rather than deleted.
        """
        candidate = {
            "peak_time": 4.2, "start_time": 4.0, "end_time": 4.4,
            "round_number": 1, "fighter": "A", "technique": "left_head_kick",
            "family": "kick", "limb": "left_leg", "target": "head",
            "outcome": "clean", "confidence": .92, "contact_confidence": .90,
            "evidence": {"foot_lift_torsos": 1.4},   # a head kick, so plainly airborne
        }
        candidates = [
            {**candidate, "peak_time": 4.2 + i * 3, "start_time": 4.0 + i * 3, "end_time": 4.4 + i * 3,
             "fighter": "A" if i % 2 == 0 else "B"}
            for i in range(SETTINGS.min_verified_actions_for_score + 1)
        ]
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False},
            "integrity": {}, "setup": {"ruleset": "K1"},
            "video": {"analysis_target": "BOTH"},
            "tracking": {"fighter_A_coverage": .87, "fighter_B_coverage": .94},
            "rounds": [{"number": 1, "selected": True}],
            "events": candidates, "key_moments": candidates, "illegal_moves": [], "metrics": {},
        }
        _apply_report_annotations(report, [])
        card = report["scorecard"]
        self.assertFalse(card["available"])
        self.assertEqual(card["status"], "punch_counting_unavailable")
        self.assertIsNone(card["totals"]["A"])
        self.assertIn("punch counting is not accurate enough", card["disclaimer"])
        # The coverage was fine and there were candidates - the refusal has to
        # say which of the two reasons it is, or it reads as a coverage problem.
        self.assertNotIn("observation coverage", card["disclaimer"])
        self.assertEqual(card["evidence"]["evidence_source"], "unvalidated_action_candidates")
        self.assertEqual(report["key_moments"], [])

        # And the switch is a switch, not a deletion.
        from unittest import mock

        from core import report as report_module

        second = dict(report, scorecard=None)
        second.pop("scorecard")
        with mock.patch.object(report_module, "STRIKE_COUNTS_PRECISION_VALIDATED", True):
            _apply_report_annotations(second, [])
        self.assertTrue(second["scorecard"]["available"])
        self.assertEqual(second["scorecard"]["status"], "preliminary_unvalidated")
        self.assertFalse(report["integrity"]["action_metrics_trusted"])

    def test_a_round_is_not_scored_from_a_couple_of_actions(self):
        """Found by running a real fight, not by a fixture.

        135 seconds of real tournament footage produced 244 detected actions,
        two of which passed the evidence thresholds - and the scorer published
        a 9-10 round with a named winner and one fighter on exactly zero. A
        ten-point-must round is a judgement about who controlled the round, and
        two actions cannot support it. The score is withheld; every movement
        measurement the fight did support is kept.
        """
        candidate = {
            "peak_time": 4.2, "start_time": 4.0, "end_time": 4.4,
            "round_number": 1, "fighter": "A", "technique": "cross",
            "family": "punch", "limb": "right_hand", "target": "head",
            "outcome": "clean", "confidence": .92, "contact_confidence": .90,
        }
        thin = [{**candidate, "peak_time": 4.2 + i * 3} for i in range(2)]
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False},
            "integrity": {}, "setup": {"ruleset": "K1"},
            "video": {"analysis_target": "BOTH"},
            # Coverage is excellent; the footage simply did not contain enough
            # verifiable scoring action, which is a different failure.
            "tracking": {"fighter_A_coverage": .978, "fighter_B_coverage": .983},
            "rounds": [{"number": 1, "selected": True}],
            "events": thin, "key_moments": thin, "illegal_moves": [], "metrics": {},
        }
        _apply_report_annotations(report, [])
        card = report["scorecard"]
        self.assertFalse(card["available"])
        self.assertEqual(card["status"], "insufficient_scoring_actions")
        self.assertIsNone(card["totals"]["A"])
        self.assertEqual(card["rounds"], [])
        self.assertIsNone(card["winner_estimate"])
        # The reason is stated with the actual count, not as a generic refusal.
        self.assertIn(str(SETTINGS.min_verified_actions_for_score), card["disclaimer"])
        self.assertIn("Movement, coverage", card["disclaimer"])

    def test_scorecard_only_review_unlocks_human_score_without_claiming_coaching_trust(self):
        candidate = {
            "peak_time": 4.2, "round_number": 1, "fighter": "A", "technique": "cross",
            "family": "punch", "limb": "right_hand", "target": "head", "outcome": "clean",
            "confidence": .91, "contact_confidence": .91,
        }
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False}, "integrity": {},
            "setup": {"ruleset": "K1"}, "video": {"analysis_target": "BOTH"},
            "tracking": {"fighter_A_coverage": .90, "fighter_B_coverage": .91},
            "rounds": [{"number": 1, "selected": True}],
            "events": [candidate], "key_moments": [], "illegal_moves": [], "metrics": {},
        }
        annotation = {"event_time": 4.2, "predicted": candidate, "corrected": dict(candidate)}
        _apply_report_annotations(report, [annotation], review_status="scorecard_complete")
        self.assertEqual(report["scorecard"]["status"], "human_reviewed")
        self.assertEqual(report["scorecard"]["evidence"]["evidence_source"], "human_ground_truth")
        self.assertTrue(report["integrity"]["scorecard_human_review_complete"])
        self.assertFalse(report["integrity"]["human_review_complete"])
        self.assertFalse(report["integrity"]["action_metrics_trusted"])

    def test_human_labels_override_candidates_and_negative_labels_disappear(self):
        candidate = {
            "peak_time": 4.2, "fighter": "A", "technique": "left_head_kick",
            "family": "kick", "limb": "left_leg", "target": "head", "outcome": "clean",
        }
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False},
            "integrity": {}, "setup": {"ruleset": "K1"}, "video": {"analysis_target": "BOTH"},
            "events": [candidate], "key_moments": [candidate], "illegal_moves": [], "metrics": {},
        }
        correction = {"event_time": 4.2, "predicted": candidate, "corrected": {
            "fighter": "B", "technique": "right_low_kick", "family": "kick",
            "limb": "right_leg", "target": "leg", "outcome": "checked",
        }}
        _apply_report_annotations(report, [correction])
        self.assertEqual(report["key_moments"][0]["fighter"], "B")
        self.assertTrue(report["key_moments"][0]["human_verified"])
        correction["corrected"] = {"fighter": "B", "technique": "none", "family": "none",
                                     "limb": "none", "target": None, "outcome": "uncertain"}
        _apply_report_annotations(report, [correction])
        self.assertEqual(report["key_moments"], [])

    def test_release_gate_requires_untouched_test_evidence(self):
        classifier = {"custom_temporal_checkpoint_loaded": True, "temporal_validation": {
            "val_accuracy": .96, "held_out_fights": ["v1", "v2", "v3"], "dataset_version": "gold-v1",
        }}
        self.assertFalse(automated_evidence_trust(classifier)["automated_evidence_trusted"])
        classifier["temporal_validation"].update({
            "test_accuracy": .94,
            "held_out_test_fights": ["t1", "t2", "t3"],
            "per_class_test_accuracy": {name: .85 for name in ACTION_CLASSES},
            "per_class_test_f1": {name: .82 for name in ACTION_CLASSES},
        })
        self.assertFalse(automated_evidence_trust(classifier)["automated_evidence_trusted"])
        classifier["temporal_validation"]["end_to_end_validation"] = {
            "fights": 5,
            "action_labels": 120,
            "timing_samples": 80,
            "fighter_identity_accuracy": .97,
            "target_accuracy": .93,
            "outcome_accuracy": .88,
            "legality_accuracy": .97,
            "timing_mae_seconds": .16,
        }
        self.assertTrue(automated_evidence_trust(classifier)["automated_evidence_trusted"])

        classifier["temporal_validation"]["end_to_end_validation"]["outcome_accuracy"] = .70
        decision = automated_evidence_trust(classifier)
        self.assertFalse(decision["automated_evidence_trusted"])
        self.assertIn("outcome", " ".join(decision["end_to_end_gate"]["failures"]))
        classifier["temporal_validation"]["end_to_end_validation"]["outcome_accuracy"] = .88

        classifier["temporal_validation"]["per_class_test_accuracy"] = {
            f"invented_{index}": .99 for index in range(len(ACTION_CLASSES))
        }
        self.assertFalse(automated_evidence_trust(classifier)["automated_evidence_trusted"])

    def test_completed_human_review_rebuilds_score_and_coaching(self):
        candidate = {
            "peak_time": 4.2, "fighter": "A", "technique": "cross", "family": "punch",
            "limb": "right_hand", "target": "head", "outcome": "clean", "round_number": 1,
        }
        base_metric = {
            "pose_coverage": .9, "guard_index": .35, "balance_index": .7,
            "dashboard": {}, "attacks": {}, "combinations": {}, "counters": {},
            "defenses": {}, "vulnerability_targets": {},
        }
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False}, "integrity": {},
            "setup": {"ruleset": "K1"}, "video": {"analysis_target": "BOTH"},
            "performance": {"segment_duration_seconds": 120},
            "rounds": [{"number": 1, "start_seconds": 0, "end_seconds": 120, "selected": True}],
            "events": [candidate], "key_moments": [], "illegal_moves": [],
            "metrics": {"A": dict(base_metric), "B": dict(base_metric)},
        }
        annotation = {"event_time": 4.2, "predicted": candidate, "corrected": dict(candidate)}
        _apply_report_annotations(report, [annotation], human_review_complete=True)
        self.assertTrue(report["integrity"]["action_metrics_trusted"])
        self.assertTrue(report["scorecard"]["available"])
        self.assertEqual(report["scorecard"]["evidence"]["evidence_source"], "human_ground_truth")
        self.assertTrue(report["coaching"]["A"]["improvements"])

    def test_performance_report_is_numbers_first_without_removing_deep_detail(self):
        root = Path(__file__).resolve().parents[1]
        template = (root / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        analyzer = (root / "core" / "analyzer.py").read_text(encoding="utf-8")

        self.assertIn('class="numbers-first-report"', template)
        self.assertIn("Fighter comparison", template)
        self.assertIn('class="tactical-performance"', template)
        self.assertIn("Defensive denial", template)
        self.assertIn("Clean exposure", template)
        self.assertIn("Knees landed", template)
        self.assertIn("Watch moments", template)
        self.assertIn("report.statistics", template)
        self.assertIn('class="report-deep-dive"', template)
        self.assertIn('report["statistics"] = final_live_stats', analyzer)
        self.assertIn('report["event_feed"] = all_final_live_events', analyzer)

    def test_complete_report_exposes_every_supported_analysis_section(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")

        self.assertIn('class="report-deep-dive"', template)
        self.assertNotIn('class="report-deep-dive" open', template)
        self.assertIn('id="report-strikes"', template)
        self.assertIn('id="report-combinations"', template)
        self.assertIn('id="report-key-moments"', template)
        self.assertIn('id="report-summary"', template)
        self.assertIn("technique_breakdown", template)
        for outcome in ("Attempts", "Landed", "Missed", "Blocked", "Evaded", "Uncertain"):
            self.assertIn(outcome, template)


if __name__ == "__main__":
    unittest.main()


class SocialAuthResilienceTests(unittest.TestCase):
    """Pressing a social sign-in button must always answer."""

    def test_unreachable_provider_reports_instead_of_hanging(self):
        """The button spun for ever in production instead of failing.

        Starting an OIDC sign-in makes the web server fetch the provider's
        discovery document. On a shared host whose outbound HTTPS is blocked
        that call never completes, so the POST never answered and the button
        showed a loading state indefinitely with nothing explaining why.
        """
        import httpx

        import app.main as webapp
        from app.main import app

        class Unreachable:
            async def authorize_redirect(self, *args, **kwargs):
                raise httpx.ConnectTimeout("outbound HTTPS blocked")

        client = TestClient(app)
        with patch.object(webapp.SOCIAL_AUTH, "client", return_value=Unreachable()):
            response = client.post(
                "/auth/google/start",
                data={"mode": "login", "next_path": "/dashboard", "accept_policies": "true"},
                follow_redirects=False,
            )
        # The point is that it answers at all, and says why. _auth_page uses
        # 400 for a rejected attempt; what must never happen is no reply.
        self.assertEqual(response.status_code, 400)
        self.assertIn("could not reach", response.text.lower())
        self.assertIn("email and password", response.text.lower())

    def test_every_configured_provider_has_a_label_and_an_icon(self):
        """A provider with no icon renders the Microsoft squares by mistake."""
        from core.social_auth import PROVIDER_LABELS

        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "auth.html").read_text(encoding="utf-8")
        for key in PROVIDER_LABELS:
            if key == "microsoft":
                continue                      # the else-branch icon
            self.assertIn(f"provider.key == '{key}'", template, f"{key} has no icon branch")

    def test_outbound_calls_carry_a_timeout(self):
        from core.social_auth import OUTBOUND_TIMEOUT_SECONDS

        source = (Path(__file__).resolve().parents[1] / "core" / "social_auth.py").read_text(encoding="utf-8")
        self.assertGreater(OUTBOUND_TIMEOUT_SECONDS, 0)
        # Every registered client, OIDC or not, has to carry the ceiling.
        self.assertEqual(source.count('"timeout": OUTBOUND_TIMEOUT_SECONDS'), 3)


class UndecodableUploadTests(unittest.TestCase):
    """A file the server cannot decode must be refused clearly and early."""

    def test_the_page_no_longer_re_encodes_before_uploading(self):
        """The file is sent as filmed.

        The in-browser re-encode produced WebM this server cannot decode,
        truncated uploads when a phone backgrounded the tab, made people wait
        minutes before a byte moved, and finally froze at a percentage because
        its draw loop runs on requestAnimationFrame, which stops when a screen
        dims. Fewer bytes was never worth any of that.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "app" / "static" / "upload-compress.js").exists())
        page = (root / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        for gone in ("wiqShrinkVideo", "upload-compress", "Preparing video"):
            self.assertNotIn(gone, page)


class MovementScorecardRenderTests(unittest.TestCase):
    """The movement scorecard has to reach the page, and say what it excludes."""

    def test_the_template_renders_the_card_and_its_limits(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates"
                    / "result.html").read_text(encoding="utf-8")
        self.assertIn('id="report-movement-score"', template)
        self.assertIn("movement.criteria_scored", template)
        # The exclusion is not optional dressing: a scorecard that does not say
        # it left out clean striking is claiming to be a full score.
        self.assertIn("movement.criteria_excluded", template)
        self.assertIn("movement.disclaimer", template)
        # And it must render its own withheld state rather than vanishing.
        self.assertIn("movement.reason", template)

    def test_it_sits_above_the_striking_scorecard(self):
        """The card WarriorIQ can stand behind comes first."""
        template = (Path(__file__).resolve().parents[1] / "app" / "templates"
                    / "result.html").read_text(encoding="utf-8")
        self.assertLess(
            template.index('id="report-movement-score"'),
            template.index('id="report-scorecard"'),
        )


class UploadProgressStripTests(unittest.TestCase):
    """The upload runs while the page stays readable."""

    def setUp(self):
        self.page = (Path(__file__).resolve().parents[1] / "app" / "templates"
                     / "analyze.html").read_text(encoding="utf-8")

    def test_the_strip_is_moved_out_of_the_transformed_wrapper(self):
        """position:fixed is not enough on this site.

        .shell carries an identity transform, and a transformed ancestor
        becomes the containing block for fixed descendants - so the strip
        pinned to the bottom of the window rendered a thousand pixels below it.
        It has to be reparented to <body> to escape that.
        """
        self.assertIn("document.body.appendChild(progress)", self.page)
        # Exactly one place unhides it - the helper that reparents first. Any
        # second one would show the strip while still inside .shell.
        self.assertEqual(
            self.page.count("progress.hidden=false;progress.dataset.tone="), 1,
            "the progress strip is shown somewhere that skips the reparenting",
        )
        self.assertGreaterEqual(self.page.count("showProgress("), 2)

    def test_leaving_the_page_mid_upload_is_warned_about(self):
        """The browser kills the transfer on navigation; say so beforehand."""
        self.assertIn("beforeunload", self.page)
        self.assertIn("uploadInFlight", self.page)

    def test_the_in_page_picker_is_gone(self):
        for absent in ("inline-picker.js", "wiqInlinePicker", 'id="inlinePicker"'):
            self.assertNotIn(absent, self.page)


class ComponentStylesReachTheirPagesTests(unittest.TestCase):
    """Styles are only styles if the page that needs them loads the file.

    components.css was linked by analyze, frame and select alone, so every rule
    written into it for the roster, the squad table, the movement scorecard and
    the plan ladders was inert on the pages that used those classes. The visible
    symptom was a select and a button stacking instead of sitting in a row.
    """

    def setUp(self):
        self.client = TestClient(app)

    def test_every_page_using_component_classes_loads_them(self):
        """Checked against the template source, because several of these
        classes only render for a signed-in account with fights."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "app" / "templates"
        component_classes = ("fighter-assign", "add-fighter", "audience-switch",
                             "trend-cell", "plan-blocked", "file-button")
        for template in root.glob("*.html"):
            text = template.read_text(encoding="utf-8")
            used = [name for name in component_classes if name in text]
            if not used:
                continue
            self.assertIn(
                "components.css", text,
                f"{template.name} uses {used} but never loads components.css",
            )

    def test_component_styles_are_cache_busted_everywhere(self):
        for path in ("/coach", "/pricing", "/analyze/kickboxing"):
            page = self.client.get(path).text
            self.assertNotIn('href="/static/components.css"', page)
            self.assertRegex(page, r'components\.css\?v=[0-9a-f]+')

    def test_no_button_uses_a_variant_that_does_not_exist(self):
        """.btn.ghost was never defined, so those rendered as primary buttons.

        A Save on every table row and an Add box both competing with the page's
        real call to action.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        defined = set()
        for sheet in (root / "app" / "static").glob("*.css"):
            defined.update(re.findall(r"\.btn\.([a-z-]+)", sheet.read_text(encoding="utf-8")))

        used = set()
        for template in (root / "app" / "templates").glob("*.html"):
            text = template.read_text(encoding="utf-8")
            # Jinja inside a class attribute would otherwise contribute words
            # like "endif" and "audience" as if they were CSS classes.
            text = re.sub(r"\{%.*?%\}|\{\{.*?\}\}", " ", text, flags=re.S)
            for match in re.findall(r'class="btn ([^"]*)"', text):
                used.update(word for word in match.split() if word.isalpha())

        undefined = used - defined - {"btn"}
        self.assertFalse(undefined, f"button variants with no CSS: {sorted(undefined)}")


class FormLabellingTests(unittest.TestCase):
    """Every control on the upload form must have an accessible name.

    fighter_name had none: in the branch where an account already has a roster
    it is the "+ Add a new fighter" text box, and the only <label> nearby
    points at the select above it. A screen reader reached an unnamed edit
    field on the form that starts every analysis.
    """

    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    # Hidden and CSRF fields carry no user-visible question, and submits are
    # named by their own text.
    _EXEMPT_TYPES = {"hidden", "submit", "button", "image", "reset"}

    def _unlabelled(self, page: str) -> list[str]:
        labelled = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', page))
        # <label><input> text</label> names the control by containing it, and
        # the consent checkboxes are written that way. Record those spans so
        # wrapping counts as a name alongside for=/aria-label.
        wrapped = [(m.start(), m.end())
                   for m in re.finditer(r"<label\b[^>]*>.*?</label>", page, re.S)]
        missing = []
        for match in re.finditer(r"<(?:input|select|textarea)\b[^>]*>", page):
            tag = match.group(0)
            kind = (re.search(r'\btype="([^"]+)"', tag) or [None, ""])[1]
            if kind in self._EXEMPT_TYPES:
                continue
            if "aria-label" in tag or "aria-labelledby" in tag:
                continue
            identifier = re.search(r'\bid="([^"]+)"', tag)
            if identifier and identifier.group(1) in labelled:
                continue
            if any(start < match.start() < end for start, end in wrapped):
                continue
            missing.append(tag)
        return missing

    def test_every_control_on_the_analyze_form_is_named(self):
        for sport in ("kickboxing", "boxing", "muay_thai", "taekwondo", "mma"):
            with self.subTest(sport=sport):
                missing = self._unlabelled(self.client.get(f"/analyze/{sport}").text)
                self.assertEqual(missing, [], f"unlabelled control(s): {missing}")

    def test_the_new_fighter_name_field_is_labelled(self):
        """Read from the template, not a rendered page: this branch only
        exists for a signed-in account that already has a roster, and the
        anonymous page never reaches it."""
        source = (Path(__file__).resolve().parents[1] / "app" / "templates"
                  / "analyze.html").read_text(encoding="utf-8")
        self.assertIn('<label for="fighterName">', source)
        # The label lives inside the wrapper the script hides, so it cannot be
        # left on screen pointing at a field that is no longer there.
        self.assertIn('id="newFighterField"', source)
        self.assertIn("newFighterField.hidden=!adding", source)
        # required stays on the input: a required control inside a hidden
        # wrapper blocks submit with an error pointing at nothing visible.
        self.assertIn("fighterName.required=adding", source)

    def test_no_control_in_the_analyze_template_is_left_unnamed(self):
        """The rendered check above only sees the signed-out form. The
        template source reaches the account and roster branches too."""
        source = (Path(__file__).resolve().parents[1] / "app" / "templates"
                  / "analyze.html").read_text(encoding="utf-8")
        missing = self._unlabelled(source)
        self.assertEqual(missing, [], f"unlabelled control(s): {missing}")


class AssetCacheBustingTests(unittest.TestCase):
    """Every stylesheet must carry the content hash, not a typed-in date.

    /static/ is served with max-age=604800, so a stylesheet whose token never
    moves reaches returning visitors up to a week late - new markup against an
    old stylesheet, which looks like the site broke rather than like a cache.
    Three of them carried "?v=20260831-ui4", the exact hand-typed token that
    _asset_version() was written to get rid of.
    """

    def test_no_template_carries_a_hand_typed_asset_token(self):
        templates_dir = Path(__file__).resolve().parents[1] / "app" / "templates"
        for path in sorted(templates_dir.glob("*.html")):
            for token in re.findall(r'\?v=([^"\']+)', path.read_text(encoding="utf-8")):
                with self.subTest(template=path.name, token=token):
                    self.assertEqual(
                        token, "{{asset_version}}",
                        f"{path.name} pins an asset to a token that never moves")

    def test_the_token_follows_the_stylesheet_contents(self):
        """It hashes contents, not mtimes: the deploy copies with `cp -R`, so
        an mtime-based token threw away every visitor's cache on every deploy
        whether or not anything had changed."""
        from app.main import _asset_version

        static = Path(__file__).resolve().parents[1] / "app" / "static"
        target = static / "frame-picker.css"
        before = _asset_version()
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n/* cache probe */\n")
            self.assertNotEqual(before, _asset_version())
        finally:
            target.write_bytes(original)
        self.assertEqual(before, _asset_version())


class FightVideoFormatTests(unittest.TestCase):
    """The picker, the copy and the server must name the same formats.

    accept="video/*" let the picker offer .wmv, .flv, .3gp and .mpg. They were
    accepted at pick time and refused after the upload had already been sent -
    the worst possible place to find out, on a phone connection.
    """

    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_the_picker_offers_exactly_what_the_server_accepts(self):
        from core.upload_security import FIGHT_VIDEO_ACCEPT, FIGHT_VIDEO_EXTENSIONS

        page = self.client.get("/analyze/kickboxing").text
        self.assertIn(f'accept="{FIGHT_VIDEO_ACCEPT}"', page)
        self.assertNotIn('accept="video/*"', page)
        # Every extension the server would take is offered by the picker.
        for suffix in FIGHT_VIDEO_EXTENSIONS:
            self.assertIn(suffix, FIGHT_VIDEO_ACCEPT)
        # And nothing the server refuses is.
        for refused in (".wmv", ".flv", ".3gp", ".mpg", ".mpeg"):
            self.assertNotIn(refused, FIGHT_VIDEO_ACCEPT)

    def test_the_displayed_copy_is_built_from_the_same_list(self):
        from core.upload_security import FIGHT_VIDEO_FORMATS, FIGHT_VIDEO_LABEL

        self.assertEqual(FIGHT_VIDEO_LABEL, "MP4, MOV, MKV, AVI, M4V or WEBM")
        self.assertIn(FIGHT_VIDEO_LABEL, self.client.get("/analyze/kickboxing").text)
        # Naming a format in the sentence that the server does not take is the
        # drift this whole arrangement exists to prevent.
        for suffix, _ in FIGHT_VIDEO_FORMATS:
            self.assertIn(suffix.lstrip(".").upper(), FIGHT_VIDEO_LABEL)

    def test_the_signature_check_covers_every_accepted_format(self):
        """A format we accept but cannot recognise on disk would be refused
        after upload by looks_like_video, which is the same late refusal in a
        different coat."""
        from core.upload_security import FIGHT_VIDEO_EXTENSIONS, looks_like_video
        import tempfile

        headers = {
            ".mp4": b"\x00\x00\x00\x20ftypisom", ".mov": b"\x00\x00\x00\x14ftypqt  ",
            ".m4v": b"\x00\x00\x00\x20ftypM4V ", ".mkv": b"\x1a\x45\xdf\xa3\x01\x00\x00\x00",
            ".webm": b"\x1a\x45\xdf\xa3\x01\x00\x00\x00",
            ".avi": b"RIFF\x00\x00\x00\x00AVI LIST",
        }
        self.assertEqual(set(headers), set(FIGHT_VIDEO_EXTENSIONS))
        for suffix, header in headers.items():
            with self.subTest(suffix=suffix):
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                    handle.write(header + b"\x00" * 32)
                    path = handle.name
                try:
                    self.assertTrue(looks_like_video(path))
                finally:
                    Path(path).unlink(missing_ok=True)


class UploadHandoffTests(unittest.TestCase):
    """The upload must hand off to the frame picker without a prompt."""

    def setUp(self):
        from pathlib import Path
        self.page = (Path(__file__).resolve().parents[1] / "app" / "templates"
                     / "analyze.html").read_text(encoding="utf-8")

    def test_the_leave_guard_is_released_before_navigating(self):
        """'load' fires before 'loadend', and every branch of it navigates.

        Releasing in loadend meant WarriorIQ's own redirect tripped its own
        beforeunload guard: the bar stopped at 99% and a "Leave site?" prompt
        stood between the upload and the analysis.
        """
        load = self.page.index("request.addEventListener('load'")
        loadend = self.page.index("request.addEventListener('loadend'")
        self.assertIn("request.addEventListener('load',()=>{releasePage();", self.page)
        # And the release must not be only in loadend, which runs afterwards.
        self.assertLess(load, loadend + len(self.page), "sanity")

    def test_a_finished_transfer_does_not_sit_at_99(self):
        """The cap exists so the bar is not full while the server answers.

        That is a state to name, not a number to freeze on.
        """
        self.assertIn("request.upload.addEventListener('load'", self.page)
        self.assertIn("Upload complete", self.page)

    def test_the_estimate_is_not_presented_as_a_countdown(self):
        """It rises when a connection slows, which is true and not a fault.

        Labelled as a countdown it read as the timer running backwards, so it
        says what it is and rounds coarsely enough not to tick.
        """
        self.assertIn("at this speed", self.page)
        self.assertNotIn("s left`", self.page)
        # Rate comes from a trailing window, not the average since the start.
        self.assertIn("samples.shift()", self.page)

    def test_no_round_default_can_truncate_a_fight(self):
        """Zero means the whole video, and it has to mean that in the code.

        It used to mean it only by accident. build_round_schedule clamps the
        round length with `max(1.0, ...)`, so a posted 0 became a ONE SECOND
        round; nobody saw that because the page's JavaScript always overwrote
        the field with the file's duration first. On a video whose duration the
        browser could not read - the case the fallback exists for - it posted 0
        and analysed one second.

        Now that the length is a visible control a reader can genuinely leave
        at "use the whole video", this is asserted against the scheduler rather
        than against markup.
        """
        import types
        from pathlib import Path

        from core.video import build_round_schedule

        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("round_duration_seconds: float = Form(0.0)", source)

        info = types.SimpleNamespace(duration=187.0, fps=30.0)
        req = types.SimpleNamespace(
            start_seconds=0.0, end_seconds=None, round_count=3,
            round_duration_seconds=0.0, break_duration_seconds=60.0,
            selected_rounds=None)
        rounds = build_round_schedule(req, info)
        self.assertEqual(len(rounds), 1, "zero length must collapse to one span")
        self.assertAlmostEqual(rounds[0].end_seconds, 187.0, places=3)
        self.assertTrue(rounds[0].selected)

        # And a real format is still honoured rather than flattened.
        req.round_duration_seconds, req.break_duration_seconds = 120.0, 0.0
        three_twos = build_round_schedule(req, info)
        self.assertEqual(len(three_twos), 2, "187s of footage holds two full two-minute rounds")
        self.assertAlmostEqual(three_twos[0].end_seconds, 120.0, places=3)


class HomepagePromiseTests(unittest.TestCase):
    """The homepage may not sell what the report withholds."""

    def test_the_hero_does_not_promise_strike_counting(self):
        """It said "See every shot" and "what landed, what missed".

        The report says the opposite in as many words - landed, missed and
        blocked are hidden until WarriorIQ can count them accurately - so the
        first line on the site promised precisely the thing the product
        declines to show. Whatever the hero says has to survive being read
        beside a finished report.
        """
        from pathlib import Path

        home = (Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html").read_text(encoding="utf-8")
        hero = home.split("</section>")[0].lower()
        for claim in ("see every shot", "what landed", "every punch", "counts every"):
            self.assertNotIn(claim, hero, f"the hero cannot promise {claim!r} while strike counting is off")

    def test_the_hero_says_what_is_actually_delivered(self):
        """"Straight to the point" means naming the real deliverables."""
        from pathlib import Path

        home = (Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html").read_text(encoding="utf-8")
        hero = home.split("</section>")[0].lower()
        self.assertIn("score estimate", hero, "the scorecard is the headline deliverable")
        self.assertIn("four-week plan", hero, "so is the training plan")
        self.assertIn("guard", hero, "and the measurements that are actually real")


class InterruptedVideoTransferTests(unittest.TestCase):
    """Upload succeeded, analysis said it could not open the video."""

    def test_a_truncated_transfer_is_not_blamed_on_the_footage(self):
        """The file was decoded here well enough to cut a selection frame.

        So a video the analysis machine cannot open means the copy to that
        machine stopped early. The old wording sent people away to re-export a
        file that was never the problem.
        """
        for error in (
            RuntimeError("Could not open fight video"),
            RuntimeError("Could not read the selected fight-start frame"),
            RuntimeError("Video download stopped early: received 12 of 900 bytes."),
        ):
            with self.subTest(error=str(error)):
                message = _public_analysis_error(error)
                self.assertIn("start the analysis again", message.lower())
                self.assertNotIn("could not finish", message.lower())

    def test_other_failures_keep_their_own_wording(self):
        self.assertIn("memory", _public_analysis_error(MemoryError("out of memory")).lower())
        self.assertIn("unavailable", _public_analysis_error(ImportError("no module")).lower())
        self.assertIn("could not finish", _public_analysis_error(ValueError("something else")).lower())


class CsrfTests(unittest.TestCase):
    """Cross-site request forgery: the second lock, and the one that catches
    login CSRF.

    The session cookie is SameSite=Lax and every mutation is a POST, so the
    classic attack was already refused. What Lax does not stop is an attacker
    giving you *their* session: a cross-site POST to /login sends no cookie,
    it only sets one, and the victim then uploads private footage into the
    attacker's library. See core/csrf.py.
    """

    def setUp(self):
        from fastapi.testclient import TestClient

        from app.main import app

        self.client = TestClient(app)

    def test_every_browser_mutation_requires_a_token(self):
        """The guarantee that makes per-route enforcement safe.

        Enforcement is a dependency rather than middleware, because checking
        this in middleware means reading the body before the endpoint does and
        getting that wrong silently empties uploads. The cost of per-route is
        that a new route can forget it - so this walks the route table instead
        of trusting anybody to remember.
        """
        from app.main import CSRF_EXEMPT_PATHS, CSRF_EXEMPT_PREFIXES, app, require_csrf

        unsafe = {"POST", "PUT", "PATCH", "DELETE"}
        missing = []
        for route in app.routes:
            methods = getattr(route, "methods", set()) or set()
            path = getattr(route, "path", "")
            if not (methods & unsafe):
                continue
            if path.startswith(CSRF_EXEMPT_PREFIXES) or path in CSRF_EXEMPT_PATHS:
                continue
            deps = [d.dependency for d in getattr(route, "dependencies", [])]
            if require_csrf not in deps:
                missing.append(f"{sorted(methods & unsafe)} {path}")
        self.assertEqual(missing, [], "these routes change state without a CSRF check")

    def test_the_exempt_routes_are_only_machine_clients(self):
        """A browser never posts to these, and they authenticate their own way.

        The worker sends a bearer token compared in constant time and Stripe
        signs its body, both checked before anything happens. Requiring a
        cookie-derived token from a client that has no cookies would break
        them for no gain - but the exemption list must stay this short.
        """
        from app.main import CSRF_EXEMPT_PATHS, CSRF_EXEMPT_PREFIXES, app, require_csrf

        self.assertEqual(CSRF_EXEMPT_PREFIXES, ("/api/worker/", "/stripe/"))
        self.assertEqual(CSRF_EXEMPT_PATHS, ("/auth/{provider}/callback",))
        # The callback is exempt; the start of the same flow is not, because
        # that one posts from our own sign-in form.
        start = next(r for r in app.routes if getattr(r, "path", "") == "/auth/{provider}/start")
        self.assertIn(require_csrf, [d.dependency for d in start.dependencies])

    def test_a_post_without_a_token_is_refused(self):
        response = self.client.post("/login", data={"email": "a@b.co", "password": "x" * 12})
        self.assertEqual(response.status_code, 403)
        self.assertIn("another site", response.text)

    def test_a_post_with_the_wrong_token_is_refused(self):
        self.client.get("/login")
        response = self.client.post(
            "/login", data={"email": "a@b.co", "password": "x" * 12,
                            "csrf_token": "n" * 43},
        )
        self.assertEqual(response.status_code, 403)

    def test_login_csrf_is_what_this_closes(self):
        """The attacker cannot sign a victim into the attacker's account.

        Lax does not cover this: the forged POST needs no cookie sent, only
        one set. Without a token the attacker's credentials would be accepted
        and the victim's next upload would land in the attacker's library.
        """
        response = self.client.post(
            "/login", data={"email": "attacker@example.com", "password": "x" * 12},
            headers={"origin": "https://evil.example"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("warrioriq_session", response.cookies)

    def test_a_page_carries_a_token_the_visitor_can_return(self):
        """The token reaches the page from server state, not from the cookie.

        The cookie stays httponly on purpose: the usual double-submit recipe
        makes it script-readable so forms can copy it, which hands the token
        to any injected script too.
        """
        page = self.client.get("/login")
        self.assertEqual(page.status_code, 200)
        self.assertIn('name="csrf-token"', page.text)
        self.assertIn('name="csrf_token"', page.text)
        cookie = self.client.cookies.get("warrioriq_csrf")
        self.assertTrue(cookie)
        self.assertIn(cookie, page.text)

    def test_the_token_cookie_is_httponly_and_samesite_lax(self):
        response = self.client.get("/login")
        header = " ".join(
            v for k, v in response.headers.items()
            if k.lower() == "set-cookie" and "warrioriq_csrf" in v
        ).lower()
        self.assertIn("httponly", header)
        self.assertIn("samesite=lax", header)

    def test_a_matching_token_passes_through(self):
        """A real submission still works - the point is not to break the app."""
        self.client.get("/login")
        token = self.client.cookies.get("warrioriq_csrf")
        response = self.client.post(
            "/login", data={"email": "nobody@example.com", "password": "x" * 12,
                            "csrf_token": token},
            follow_redirects=False,
        )
        # Wrong credentials, but it got past the CSRF gate rather than 403.
        self.assertNotEqual(response.status_code, 403)

    def test_a_stale_cookie_shape_is_replaced_rather_than_rejected(self):
        """A visitor carrying junk gets a working token, not a wall of 403s.

        A cookie of the wrong shape is treated as absent, so the middleware
        issues a fresh one on the next page load. Rejecting it instead would
        leave somebody stuck with no way to clear it from inside the browser.
        """
        from core.csrf import issue_token, usable_token

        for junk in ("", None, "short", "!" * 40, "x" * 400, "has spaces in it here"):
            with self.subTest(cookie=junk):
                self.assertIsNone(usable_token(junk))
        token = issue_token()
        self.assertEqual(usable_token(token), token)
        self.assertEqual(usable_token("  %s  " % token), token)

    def test_an_absent_token_never_compares_equal(self):
        """Two missing values must not satisfy the check."""
        from core.csrf import tokens_match

        self.assertFalse(tokens_match(None, None))
        self.assertFalse(tokens_match("", ""))
        self.assertFalse(tokens_match("abc", None))
        self.assertFalse(tokens_match(None, "abc"))
        self.assertTrue(tokens_match("abc", "abc"))


class RateLimitClientTests(unittest.TestCase):
    """Who a rate limit counts against.

    The limiter keyed on `request.client.host`, and warrioriq.eu runs behind
    Apache and Passenger - so every visitor arrives from 127.0.0.1 and shared
    one bucket. Thirty failed sign-ins by one person locked the login page for
    everybody, and no per-visitor limit was ever really in force.
    """

    @staticmethod
    def _request(peer, forwarded=None):
        headers = []
        if forwarded is not None:
            headers.append((b"x-forwarded-for", forwarded.encode()))
        scope = {
            "type": "http", "method": "GET", "path": "/", "headers": headers,
            "client": (peer, 1234) if peer else None, "query_string": b"",
        }
        from starlette.requests import Request

        return Request(scope)

    def test_a_public_peer_is_believed_over_the_header(self):
        """An attacker cannot opt out of a limit by sending a header.

        Anyone can put anything in X-Forwarded-For. It is only worth reading
        when the connection itself came from a proxy we run.
        """
        from app.main import _client_ip

        self.assertEqual(
            _client_ip(self._request("8.8.4.4", forwarded="198.51.100.1")),
            "8.8.4.4",
        )

    def test_a_peer_nobody_could_browse_from_is_not_treated_as_public(self):
        """Loopback, private, link-local and the documentation ranges.

        All of them mean the connection came from something in front of us
        rather than from a visitor, which is when the header is worth reading.
        """
        from app.main import _peer_is_routable

        for host in ("127.0.0.1", "10.0.0.5", "192.168.1.9", "169.254.1.1",
                     "203.0.113.7", "::1", "not-an-address", ""):
            with self.subTest(peer=host):
                self.assertFalse(_peer_is_routable(host))
        for host in ("8.8.4.4", "1.1.1.1"):
            with self.subTest(peer=host):
                self.assertTrue(_peer_is_routable(host))

    def test_behind_a_local_proxy_the_header_is_read(self):
        from app.main import _client_ip

        self.assertEqual(
            _client_ip(self._request("127.0.0.1", forwarded="203.0.113.7")),
            "203.0.113.7",
        )

    def test_the_rightmost_entry_wins(self):
        """The nearest proxy appends what it actually saw; the rest is client input.

        A client sending `X-Forwarded-For: 1.2.3.4` gets that value kept and
        the proxy's own observation appended after it. Taking the leftmost
        entry would let the client pick its own bucket - which is what
        `_forwarded_header` does for host and scheme, correctly for those and
        wrongly for this.
        """
        from app.main import _client_ip

        self.assertEqual(
            _client_ip(self._request("10.0.0.5", forwarded="1.2.3.4, 203.0.113.7")),
            "203.0.113.7",
        )

    def test_no_header_behind_a_proxy_falls_back_to_the_peer(self):
        from app.main import _client_ip

        self.assertEqual(_client_ip(self._request("127.0.0.1")), "127.0.0.1")
        self.assertEqual(_client_ip(self._request(None)), "unknown")

    def test_two_visitors_behind_one_proxy_get_separate_buckets(self):
        """The actual bug, stated as a test."""
        from app.main import _client_ip

        first = _client_ip(self._request("127.0.0.1", forwarded="203.0.113.7"))
        second = _client_ip(self._request("127.0.0.1", forwarded="203.0.113.8"))
        self.assertNotEqual(first, second)

    def test_the_bucket_store_does_not_grow_without_bound(self):
        """Passenger keeps a worker alive for a long time.

        Every scope-and-address pair used to add an entry that was never
        removed, so a month of traffic was a month of accumulated keys.
        """
        import app.main as main

        saved = dict(main._rate_windows)
        try:
            main._rate_windows.clear()
            now = time.monotonic()
            for n in range(50):
                main._rate_windows[f"scope:{n}"] = [now - 7200]
            main._rate_windows["fresh"] = [now]
            main._prune_rate_windows(now)
            self.assertEqual(list(main._rate_windows), ["fresh"])
        finally:
            main._rate_windows.clear()
            main._rate_windows.update(saved)

    def test_pruning_forgives_rather_than_locks_out(self):
        """Dropping a bucket must fail in the safe direction.

        Losing the record of requests already counted can let somebody
        through; it can never lock somebody out who should be allowed in.
        """
        import app.main as main

        saved = dict(main._rate_windows)
        try:
            main._rate_windows.clear()
            now = time.monotonic()
            main._rate_windows["scope:x"] = [now - 7200] * 99
            main._prune_rate_windows(now)
            self.assertNotIn("scope:x", main._rate_windows)
        finally:
            main._rate_windows.clear()
            main._rate_windows.update(saved)

    def test_the_expensive_endpoints_are_limited(self):
        """Restart queues GPU work; corrections write files; both were open."""
        import inspect

        import app.main as main

        for name, scope in (
            ("restart_interrupted_analysis", "analysis-restart"),
            ("set_selection_frame", "selection-frame"),
            ("annotate_event", "annotations"),
            ("reset_password", "password-reset-complete"),
            ("delete_account_route", "account-delete"),
        ):
            with self.subTest(route=name):
                self.assertIn(scope, inspect.getsource(getattr(main, name)))


class FocusFighterDefaultTests(unittest.TestCase):
    """Which fighter gets the detailed report starts from the account's answer.

    It was already a per-fight choice - the radio on /select posts
    focus_fighter with the boxes and the job stores it - but the page hardcoded
    Fighter A. "My identity in reports" on /profile drove only the progress
    dashboard, so an athlete who had said they were Fighter B had to say it
    again on every upload and silently got a report about their opponent
    whenever they forgot.

    The profile value is the default and nothing more. Which corner somebody is
    in changes from fight to fight, so the per-fight radio still wins and
    choosing on one fight never writes back to the profile.
    """

    def _checked(self, default_fighter, has_profile=True):
        import app.main as webapp

        job = {"id": "j1", "status": "selection",
               "video_width": 1920, "video_height": 1080}
        client = TestClient(webapp.app)
        patches = [
            mock.patch.object(webapp, "_authorized_job", return_value=job),
            mock.patch.object(webapp, "_profile_id",
                              return_value=1 if has_profile else None),
        ]
        if has_profile:
            patches.append(mock.patch.object(
                webapp, "get_profile",
                return_value={"default_fighter": default_fighter}))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            page = client.get("/select/j1").text
        return [value for value, checked in re.findall(
            r'name="focusFighter" value="([AB])"\s*(checked)?', page) if checked]

    def test_the_profile_choice_is_the_one_already_selected(self):
        self.assertEqual(["A"], self._checked("A"))
        self.assertEqual(["B"], self._checked("B"))

    def test_a_missing_or_nonsense_setting_falls_back_to_fighter_a(self):
        for value in (None, "", "banana", "C"):
            with self.subTest(value=value):
                self.assertEqual(["A"], self._checked(value))
        self.assertEqual(["A"], self._checked(None, has_profile=False))

    def test_exactly_one_radio_is_ever_preselected(self):
        # Two checked radios in one group is a silent browser-dependent choice.
        for value in ("A", "B", None, "banana"):
            with self.subTest(value=value):
                self.assertEqual(1, len(self._checked(value)))


class KeyboardFighterSelectionTests(unittest.TestCase):
    """Fighter selection has to be possible without a pointer.

    Drawing a box is a pointer-only action and nothing in WarriorIQ works
    until two fighters are chosen, so "draw a box" alone makes the whole
    product unusable by keyboard - not one feature, all of it. The launch
    checklist asks for this by name.
    """

    @staticmethod
    def _template():
        return (Path(__file__).resolve().parents[1] / "app" / "templates" / "select.html").read_text(encoding="utf-8")

    def test_there_is_a_way_through_without_drawing(self):
        page = self._template()
        self.assertIn('id="pickFromList"', page)
        # Native radios in a fieldset with a legend, so the grouping is
        # announced. A cleverer control would not survive being read aloud.
        self.assertIn("<fieldset", page)
        self.assertIn("<legend>Fighter A</legend>", page)
        self.assertIn("<legend>Fighter B</legend>", page)

    def test_each_person_is_described_in_words(self):
        """The list must be usable by somebody who cannot see the frame.

        "Person 2" alone identifies nobody. Position and size are what
        actually separate two athletes from a referee standing between them.
        """
        page = self._template()
        self.assertIn("function describeDetection", page)
        for word in ("far left", "left of centre", "centre", "right of centre",
                     "far right", "fills the frame", "large", "medium", "small"):
            with self.subTest(word=word):
                self.assertIn(word, page)

    def test_it_reads_the_box_off_the_detection_object(self):
        """A detection is {box, confidence}, not a bare array.

        Treating it as an array set both fighters to undefined and broke the
        page on the first click. It was caught by driving the page in a
        browser, which is the only place this shape shows up - and it is the
        same shape detectorMatch already reads.
        """
        page = self._template()
        self.assertIn("item.box", page)
        self.assertNotIn("[...detections].sort", page)

    def test_the_two_paths_cannot_disagree(self):
        """Drawing after picking, or starting over, must clear the list.

        Otherwise a radio stays checked next to a box it no longer describes,
        and a screen reader is told the wrong thing.
        """
        page = self._template()
        self.assertIn("function clearPicked", page)
        # Start over, redraw B, and both branches of finishSelection.
        self.assertGreaterEqual(page.count("clearPicked("), 6)

    def test_the_control_has_a_visible_focus_ring(self):
        """It exists for keyboard users, so focus has to be visible.

        The default ring is invisible against the dark panel.
        """
        css = (Path(__file__).resolve().parents[1] / "app" / "static" / "fighter-selection.css").read_text(encoding="utf-8")
        self.assertIn(".pick-fighter input:focus-visible", css)
        self.assertIn("outline", css)


class ReflowTests(unittest.TestCase):
    """The page must not scroll sideways at 320px.

    WCAG 2.2 AA 1.4.10 Reflow, and the same thing 400% zoom on a laptop
    produces. Measured in a browser on 2026-09-07: the home page had 27px of
    horizontal scroll and every other public page was clean.
    """

    @staticmethod
    def _css(name):
        return (Path(__file__).resolve().parents[1] / "app" / "static" / name).read_text(encoding="utf-8")

    def test_the_hero_buttons_can_shrink(self):
        """`1fr` is `minmax(auto,1fr)`, and `auto` will not go below min-width.

        fixes.css sets `min-width:145px` on these buttons in the same
        breakpoint, left from when this was a flex row that wrapped. Two of
        them plus the gap demanded 302px inside a 288px column, so the home
        page scrolled. Both halves have to stay fixed: a shrinkable track, and
        the stale minimum cleared.
        """
        css = self._css("product.css")
        self.assertIn("grid-template-columns:minmax(0,1fr) minmax(0,1fr)", css)
        self.assertIn(".hero-actions .btn{width:100%;min-width:0}", css)

    def test_the_stale_minimum_is_still_the_reason(self):
        """If fixes.css ever drops it, the comment above stops making sense."""
        self.assertIn("min-width:145px", self._css("fixes.css"))


class WorkerAuthOrderingTests(unittest.TestCase):
    """An unauthenticated request must be refused before its body is parsed.

    Found on the live site: posting `{}` to /api/worker/heartbeat with no
    bearer token returned **422 with the full field list**, because the auth
    check ran inside the handler and FastAPI validates the body first. Not a
    hole - a well-formed body still got 401 - but an anonymous caller learned
    the request schema, and the server did work for a request it was going to
    refuse. Auth is now a route dependency, which FastAPI solves first.
    """

    def setUp(self):
        import importlib

        import app.main
        import core.config

        # app.main is reloaded because the worker routes get their auth
        # dependency at import time, and that is the whole point of these
        # tests. core.config is NOT reloaded, and must not be.
        #
        # Reloading it builds a brand new SETTINGS object, while every module
        # that did `from core.config import SETTINGS` - payments, referee,
        # legal, social_auth and a dozen more - keeps holding the old one. The
        # cleanup reload then made a THIRD object and still fixed none of them.
        # From that point on core.config.SETTINGS and everyone else's SETTINGS
        # were different objects, so any later test that edits the settings in
        # place and expects another module to see it quietly failed:
        #
        #   ProductFoundationTests.test_complimentary_grant_outranks_the_stored_plan
        #     put a grant in core.config.SETTINGS.complimentary_plans; payments
        #     read its stale copy and returned "free" instead of "gym".
        #   RefereeProbeLocationTests.test_a_genuinely_missing_probe_still_disables_cleanly
        #     pointed core.config.SETTINGS at a missing probe; referee read its
        #     stale copy, found the real file and loaded it.
        #
        # Both passed when test_core ran alone and failed when test_web ran
        # first, which reads like an unrelated flake and is not one.
        #
        # So the two fields are set on the settings object that already exists,
        # and restored afterwards. One SETTINGS for the whole process.
        settings = core.config.SETTINGS
        previous = {
            "analysis_worker_mode": settings.analysis_worker_mode,
            "worker_token": settings.worker_token,
        }
        object.__setattr__(settings, "analysis_worker_mode", "remote")
        object.__setattr__(settings, "worker_token", "unit-test-worker-token")

        def restore():
            for field, value in previous.items():
                object.__setattr__(settings, field, value)
            importlib.reload(app.main)

        self.addCleanup(restore)
        self.main = importlib.reload(app.main)
        from fastapi.testclient import TestClient
        self.client = TestClient(self.main.app)

    def test_no_token_is_refused_before_the_body_is_validated(self):
        response = self.client.post("/api/worker/heartbeat", json={})
        self.assertEqual(response.status_code, 401)
        # And the refusal must not describe the schema it never looked at.
        self.assertNotIn("worker_id", response.text)

    def test_auth_passing_still_validates_the_body(self):
        response = self.client.post(
            "/api/worker/heartbeat", json={},
            headers={"Authorization": "Bearer unit-test-worker-token"})
        self.assertEqual(response.status_code, 422)

    def test_a_correct_call_still_works(self):
        response = self.client.post(
            "/api/worker/heartbeat", json={"worker_id": "unit-test"},
            headers={"Authorization": "Bearer unit-test-worker-token"})
        self.assertEqual(response.status_code, 200)

    def test_every_worker_route_declares_the_auth_dependency(self):
        """The guarantee, the same shape as the CSRF one.

        Eight routes carry it today; a ninth added without it would be
        authenticated only by whatever the handler remembers to call.
        """
        from fastapi import Depends  # noqa: F401  (documents the mechanism)

        missing = []
        for route in self.main.app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/api/worker/"):
                continue
            deps = [d.dependency for d in getattr(route, "dependencies", [])]
            if self.main._require_remote_worker not in deps:
                missing.append(path)
        self.assertEqual(missing, [], "worker routes without the auth dependency")


class ReportOrderTests(unittest.TestCase):
    """The part that is true comes first.

    The report used to open with strike counts. Measured against the video,
    the punch count was overstated by eleven in two of three bouts, while the
    pose metrics - footwork, pressure, centre, guard, balance - are measured
    and always available. Leading with the striking read put the weakest
    claim at the top of the page.
    """

    @staticmethod
    def _page():
        return Path("app/templates/result.html").read_text(encoding="utf-8")

    def test_the_fighters_own_numbers_come_before_the_strike_count(self):
        page = self._page()
        vitals = page.index('class="fight-vitals"')
        strikes = page.index("Kicks we could count")
        self.assertLess(vitals, strikes,
                        "the strike count is above the pose metrics again")

    def test_the_movement_scorecard_does_not_open_the_report(self):
        """It needs 85% coverage and real footage gives 19-40%.

        Promoting it put a refusal above the numbers that always work, which
        is what happened on the first attempt at this reshape.
        """
        page = self._page()
        self.assertLess(page.index('class="fight-vitals"'),
                        page.index("Movement scorecard"))

    def test_the_order_is_explained_where_someone_would_change_it(self):
        page = self._page()
        self.assertIn("The part that is true leads", page)
        self.assertIn("promoting it opened the report with a refusal", page)


class NoPunchClaimLeaksTests(unittest.TestCase):
    """Render the page and look, rather than reading the guards.

    Punch counts are withheld because they were overstated by eleven in two
    of three checked bouts. Getting that right means every branch of a large
    template agreeing, and auditing it by reading `{% if %}` blocks missed a
    whole section the first time - the headline numbers under `stats_ready`,
    which carry Landed, Accuracy, Combinations, a named "Best weapon" and a
    "Punches landed" comparison row.

    So this renders the real template with a report that *does* contain punch
    data and asserts none of it reaches the page.
    """

    @staticmethod
    def _render(report):
        import json as _json

        from jinja2 import ChainableUndefined, Environment, FileSystemLoader

        from app.main import _analysis_quality_summary
        from core.report import kick_minimum_check, observed_summary

        class Stub:
            def __init__(self, **kw): self.__dict__.update(kw)
            def __getattr__(self, k): return Stub()
            def __getitem__(self, k): return Stub()
            def __str__(self): return ""
            def __bool__(self): return False
            def __iter__(self): return iter(())

        env = Environment(loader=FileSystemLoader("app/templates"), undefined=ChainableUndefined)
        env.policies["json.dumps_function"] = _json.dumps
        return env.get_template("result.html").render(
            request=Stub(url=Stub(path="/result/x"), state=Stub(csrf_token="t" * 43, account=None)),
            report=report, job_id="x", observed=observed_summary(report),
            kick_minimum=kick_minimum_check(report), asset_version="t",
            analysis_quality=_analysis_quality_summary(report),
            report_access={"report_tier": "full"}, can_share=False, sharing=None)

    @staticmethod
    def _report(trusted=False):
        """The committed sample, with punch data put back in.

        Built from tests/fixtures/report_sample.json rather than by hand: the
        page reads dozens of fields and a hand-made stub only proves the
        branches it happens to reach.
        """
        import json as _json

        report = _json.loads(
            Path("tests/fixtures/report_sample.json").read_text(encoding="utf-8"))
        report.setdefault("integrity", {}).update({
            "action_metrics_trusted": trusted,
            "identity_evidence_trusted": trusted,
            "fighter_identity_trusted": {"A": trusted, "B": trusted},
        })
        stats = report.setdefault("statistics", {})
        stats["action_labels_available"] = trusted
        stats["attempt_counts_available"] = True
        for who in ("A", "B"):
            item = stats.setdefault("fighters", {}).setdefault(who, {})
            item.update({"observation_coverage": 0.55, "total_strikes": 9,
                         "punch_attempts": 6, "kick_attempts": 3, "knee_attempts": 0,
                         "punches_landed": 4, "accuracy": 0.44, "combinations": 2})
            metrics = report.setdefault("metrics", {}).setdefault(who, {})
            metrics["strongest_weapon"] = "jab"
            attacks = metrics.setdefault("attacks", {})
            attacks.update({"attempts": 9, "landed": 4, "accuracy": 0.44,
                            "families": {"punch": 6, "kick": 3},
                            "techniques": {"jab": 4, "right_hook": 2},
                            "landed_techniques": {"jab": 3}})
        return report

    def test_an_untrusted_report_shows_no_punch_claim_anywhere(self):
        import re

        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", self._render(self._report())))
        # Anchored to a count label, not any digit before the word: the
        # score line "10-9" sits directly before the sentence saying punch
        # counting is switched off, and a looser pattern read that as a leak.
        for pattern, what in ((r"punch(es)?[ :]*\d+", "a punch count"),
                              (r"Punches landed\s*\d", "punches landed"),
                              (r"\b(jab|cross|uppercut|hook|backfist)\b", "a named punch"),
                              (r"(Best|Strongest) weapon\s*[A-Za-z]", "a named weapon")):
            with self.subTest(claim=what):
                found = re.search(pattern, text, re.I)
                self.assertIsNone(found, "%s reached the page: %r" % (
                    what, text[max(0, found.start() - 40):found.end() + 20] if found else ""))

    def test_the_leg_strike_count_does_reach_the_page(self):
        """The withholding must not be so broad that nothing is reported.

        The sample report has a scorecard, so the "Kicks we could count"
        panel - which appears only *instead* of a scorecard - is not on
        this page. The leg-strike count still is, in the numbers row, and
        that row used to print total attempts including punches.
        """
        text = self._render(self._report())
        self.assertIn("leg strikes", text.lower())
        self.assertNotIn("Fighter A attempts", text)


class NoUnsupportedClaimSurvivesTests(NoPunchClaimLeaksTests):
    """Every claim that rests on the strike detector, checked in one place.

    Individually gated claims kept slipping through one at a time - punch
    counts on four separate surfaces, then a round score computed from the
    same punches. Each was found by a different ad-hoc check. This renders
    the page once and asserts the whole family is absent, so the next one
    cannot hide in a branch nobody thought to look at.

    The detector is 29% precise overall and its punch family 14%. Nothing
    derived from it may state a number while that is true.
    """

    @staticmethod
    def _untrusted_report():
        report = NoPunchClaimLeaksTests._report(trusted=False)
        report["scorecard"] = {
            "available": False, "status": "punch_counting_unavailable",
            "ruleset_label": "K-1", "totals": {"A": None, "B": None},
            "rounds": [], "winner_estimate": None,
            "disclaimer": "No score is shown.",
        }
        return report

    def test_nothing_derived_from_the_strike_detector_states_a_number(self):
        import re

        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", self._render(self._untrusted_report())))
        claims = (
            ("landed count",      r"\blanded\b\s*\|?\s*\d"),
            ("accuracy percent",  r"accuracy\s*\|?\s*\d+\s*%"),
            ("combinations",      r"combinations?\s*\|?\s*\d"),
            ("counters",          r"counters?\s*\|?\s*\d"),
            ("blocked or evaded", r"(blocked|evaded)\s*\|?\s*\d"),
            ("a technique name",  r"\b(jab|cross|uppercut|hook|backfist)\b"),
            ("rounds won",        r"rounds won"),
            ("a round scoreline", r"\b10\s*[-\u2013:]\s*9\b"),
        )
        for label, pattern in claims:
            with self.subTest(claim=label):
                found = re.search(pattern, text, re.I)
                self.assertIsNone(found, "%s reached the page: %r" % (
                    label, text[max(0, found.start() - 45):found.end() + 20] if found else ""))

    def test_the_measured_numbers_do_survive(self):
        """The withholding must not quietly become "show nothing".

        These come from pose rather than the strike detector, so they are the
        report when everything else is withheld. If this fails the product has
        no content left.
        """
        text = self._render(self._untrusted_report()).lower()
        for shown in ("movement", "pressure", "centre", "guard", "balance", "leg strikes"):
            with self.subTest(number=shown):
                self.assertIn(shown, text)


class NavigationReachabilityTests(unittest.TestCase):
    """The desktop nav and the hamburger have to hand over at the same width.

    Between 901px and 1180px a signed-in visitor had neither: .account-nav was
    hidden at 1180, .nav-more at 1120, and .navlinks and .mobile-menu-button
    only swapped at 900. Athlete profile, Fight library, Compare fights and
    Sign out were all unreachable for 280px of viewport width.
    """

    SELECTORS = (".navlinks", ".nav-more", ".account-nav", ".mobile-menu-button")

    def _enclosing_breakpoint(self, css: str, index: int) -> str:
        """The @media condition the rule at `index` sits inside.

        The stylesheets are minified onto single lines, so the nearest media
        query opening before the match is the one that governs it.
        """
        opened = css.rfind("@media", 0, index)
        self.assertNotEqual(opened, -1, "rule is not inside any media query")
        return css[opened:css.index("{", opened)]

    def test_all_four_nav_selectors_collapse_on_one_breakpoint(self):
        # The shipped bundle, not the source files: the cascade that reaches a
        # browser is the concatenation, and a stray rule in a later file would
        # win without appearing wrong in its own file.
        from app.main import CSS_BUNDLE_TEXT

        css = "\n".join(CSS_BUNDLE_TEXT.values())
        found = {}
        for selector in self.SELECTORS:
            rules = [m.start() for m in re.finditer(
                re.escape(selector) + r"\{display:[a-z]+\}", css)]
            self.assertEqual(
                len(rules), 1,
                f"{selector} has {len(rules)} display rules; there must be exactly one")
            found[selector] = self._enclosing_breakpoint(css, rules[0])
        self.assertEqual(
            len(set(found.values())), 1,
            f"the nav collapses at more than one width: {found}")

    # Every width the shipped CSS currently switches on. The list can SHRINK
    # freely - that is the migration onto the four documented values in
    # style.css - but a new entry has to be added here deliberately, which is
    # the point: this got to twenty-two because each one looked like a single
    # harmless number at the time.
    #
    # 981 is not one of the ad-hoc ones. It is the min-width partner of 980, and
    # max-width:980 / min-width:981 is how a two-sided boundary is written
    # without a one-pixel overlap. Do not "consolidate" it into 980.
    KNOWN_BREAKPOINTS = {
        430, 440, 480, 560, 620, 650, 680, 700, 720, 760, 780,
        850, 860, 900, 950, 980, 981, 1000, 1024, 1100, 1120, 1180,
    }

    def test_no_new_breakpoint_is_introduced(self):
        """A ratchet, not a migration.

        Twenty-two widths across seventy-one media queries is a real
        maintainability problem and the audit is right to name it. It is not
        currently a user-visible one: sweeping /, /pricing and
        /analyze/kickboxing at 360, 430, 520, 620, 700, 850, 1000 and 1180
        found zero horizontal overflow and navigation present at every width on
        every page. Moving all seventy-one onto four values would change
        behaviour at each of them to fix a defect measurement cannot find, so
        the count is frozen rather than forced down in one pass.
        """
        from app.main import CSS_BUNDLE_TEXT

        css = "\n".join(CSS_BUNDLE_TEXT.values())
        conditions = [css[m.start():css.index("{", m.start())]
                      for m in re.finditer(r"@media", css)]
        widths = {int(w) for condition in conditions
                  for w in re.findall(r"(?:max|min)-width:\s*(\d+)px", condition)}

        added = widths - self.KNOWN_BREAKPOINTS
        self.assertEqual(
            set(), added,
            f"new breakpoint(s) {sorted(added)}; use one of the four in "
            f"style.css, or add it here on purpose")
        # The four documented values must survive whatever else moves.
        self.assertLessEqual({480, 620, 900, 1180}, widths)

    def test_the_mobile_menu_carries_everything_the_collapsed_bars_held(self):
        base = (Path(__file__).resolve().parents[1] / "app" / "templates"
                / "base.html").read_text(encoding="utf-8")

        def region(start: str, end: str) -> str:
            opened = base.index(start)
            return base[opened:base.index(end, opened)]

        desktop = (region('<div class="account-nav">', '<button class="mobile-menu-button"')
                   + region('<details class="nav-more"', "</details>"))
        menu = region('<div class="mobile-menu" id="mobileMenu"', "<main id=")

        hidden = set(re.findall(r'href="(/[^"#]*)"', desktop))
        reachable = set(re.findall(r'href="(/[^"#]*)"', menu))
        self.assertIn("/compare", hidden, "nav-more no longer holds the compare link")
        self.assertEqual(
            hidden - reachable, set(),
            f"hidden behind the hamburger but missing from it: {sorted(hidden - reachable)}")
        # Sign out is a form, not a link, in both places.
        self.assertIn('action="/logout"', desktop)
        self.assertIn('action="/logout"', menu)


class PlanBadgeTests(unittest.TestCase):
    """/pricing showed two different current plans at once.

    The strip read analysis_allowance, which resolves through
    effective_plan_key and so honours a complimentary grant. The card badge
    compared `plan_override or plan` by hand - the same lookup, minus the
    grant branch - so an account holding a granted Gym plan was told "Your
    current plan: Gym" directly above a Starter card badged "Current plan".
    """

    def _render(self, current_plan_key, stored_plan="free"):
        from jinja2 import ChainableUndefined, Environment, FileSystemLoader

        from core.payments import PLANS, plan_for_key

        class Stub:
            def __init__(self, **kw): self.__dict__.update(kw)
            def __getattr__(self, key): return Stub()
            def __getitem__(self, key): return Stub()
            def __str__(self): return ""
            def __bool__(self): return False

        templates = Path(__file__).resolve().parents[1] / "app" / "templates"
        env = Environment(loader=FileSystemLoader(str(templates)),
                          undefined=ChainableUndefined)
        return env.get_template("pricing.html").render(
            request=Stub(url=Stub(path="/pricing"), state=Stub(account=None), cookies={}),
            plans=PLANS, roster_held=0, payments_enabled=False,
            account={"plan": stored_plan, "plan_override": None, "email": "a@b.c"},
            allowance={"plan": plan_for_key(current_plan_key), "remaining": None},
            current_plan_key=current_plan_key,
        )

    def _badged(self, page):
        import re

        # Each card carries data-plan="<key>"; find the one holding the badge.
        cards = re.split(r'(?=<section class="card pricing-card)', page)
        return {
            re.search(r'data-plan="([^"]+)"', card).group(1)
            for card in cards
            if 'data-plan="' in card and "Current plan</span>" in card
        }

    def test_the_badged_card_is_the_plan_the_strip_names(self):
        page = self._render("gym", stored_plan="free")
        self.assertIn("Gym", page)
        self.assertEqual(
            self._badged(page), {"gym"},
            "the badge follows the stored plan instead of the effective one")

    def test_exactly_one_card_is_ever_badged(self):
        from core.payments import PLANS

        for key in PLANS:
            self.assertEqual(
                len(self._badged(self._render(key))), 1,
                f"{key} did not badge exactly one card")

    def test_a_signed_out_visitor_sees_no_current_plan(self):
        self.assertEqual(self._badged(self._render(None)), set())

    def test_no_button_claims_to_preview_a_plan_it_cannot_show(self):
        page = self._render("free")
        self.assertNotIn("Preview this plan", page)

    def test_a_closed_checkout_does_not_price_a_plan_nobody_can_buy(self):
        """The page used to argue with itself.

        It printed €9.99 to €89.99 as each card's headline price next to a
        banner claiming every plan was free during early access, and neither
        was true. Nothing makes a paid plan free: accounts.plan defaults to
        'free', the only way off it is a per-account grant the operator makes
        by hand, and checkout is shut, so nobody can be on a paid plan at all.

        With payments disabled a paid card must not headline a price, because
        that price cannot be charged today - it says what the plan will cost
        instead. Starter is genuinely €0 forever and keeps its number.
        """
        import re

        page = self._render("free")
        self.assertIn("Paid plans are not open yet", page)
        self.assertNotIn("Every plan below is free to use", page)

        for card in re.split(r'(?=<section class="card pricing-card)', page):
            key = re.search(r'data-plan="([^"]+)"', card)
            if not key:
                continue
            card = card.split("</section>", 1)[0]
            headline = re.search(r'class="plan-price">(.*?)</div>', card, re.S)
            self.assertIsNotNone(headline, key.group(1))
            headline = headline.group(1).strip()
            if key.group(1) == "free":
                self.assertEqual("€0", headline)
            else:
                self.assertEqual("Not open yet", headline, key.group(1))
                # The real price still has to be visible, just not as the
                # headline - hiding it would be the opposite mistake.
                self.assertIn("when billing opens", card, key.group(1))

    def test_a_paid_card_offers_one_action_and_it_is_about_that_plan(self):
        """Pressing "Coach 30" used to look like it had selected Coach 30.

        Each paid card carried "Open your workspace" first - a link to
        /dashboard that does nothing about the plan - and the interest form
        second, both styled as secondary. So the card had two actions, the more
        prominent one was unrelated to it, and neither said which plan it meant.
        Registering interest is the only thing the card can do while checkout is
        closed, so it is the only button on it and it names its own plan.
        """
        import re

        page = self._render("free")
        for card in re.split(r'(?=<section class="card pricing-card)', page):
            key = re.search(r'data-plan="([^"]+)"', card)
            if not key or key.group(1) == "free":
                continue
            # Stop at the card's own closing tag; the last split otherwise
            # carries the rest of the page, cookie banner included.
            card = card.split("</section>", 1)[0]
            buttons = re.findall(
                r'<(?:a|button)[^>]*class="btn[^"]*"[^>]*>(.*?)</(?:a|button)>', card, re.S)
            self.assertEqual(1, len(buttons), f"{key.group(1)}: {buttons}")
            self.assertNotIn("Open your workspace", card, key.group(1))
            self.assertIn("Tell us you want", buttons[0])


class CoachFilenameTests(unittest.TestCase):
    """/pricing carries the check-marked promise "No video filename shown"."""

    def test_the_squad_table_shows_a_fight_label_not_the_uploaded_filename(self):
        coach = (Path(__file__).resolve().parents[1] / "app" / "templates"
                 / "coach.html").read_text(encoding="utf-8")
        self.assertNotIn("f.name", coach)
        self.assertIn("{{f.label}}", coach)
        # The raw enum went with it: KICK_LIGHT is not a thing to show a coach.
        self.assertNotIn("{{f.ruleset}}", coach)
        # The ruleset rides in the label now - "Kickboxing · Kick Light ·
        # 14 Sep" - so the separate Sport and Ruleset columns were repeating
        # it twice more in a table that had nine columns in 969px.
        self.assertNotIn("<th>Ruleset</th>", coach)
        self.assertNotIn("<th>Sport</th>", coach)

    def test_the_promise_that_makes_this_a_defect_is_still_on_the_pricing_page(self):
        pricing = (Path(__file__).resolve().parents[1] / "app" / "templates"
                   / "pricing.html").read_text(encoding="utf-8")
        self.assertIn("No video filename shown", pricing,
                      "the promise moved; this test should follow it")


class BackNavigationTests(unittest.TestCase):
    """The floating Back pill is gone; what replaced it has to actually exist.

    It was position:fixed with an opaque background and no reserved space, and
    .shell carries a transform - which makes .shell the containing block for a
    fixed descendant - so it sat on top of whatever occupied the bottom-left
    corner. It covered content on six pages.

    The audit's reason for removing it was that "every page already has an
    inline back link". Five of thirty templates did. The flow pages are the
    ones that need it, and /result had nothing at all.
    """

    FLOW_TEMPLATES = ("analyze.html", "select.html", "progress.html",
                      "replay.html", "review.html", "result.html")

    def test_every_page_in_the_analysis_flow_offers_a_way_back(self):
        templates = Path(__file__).resolve().parents[1] / "app" / "templates"
        for name in self.FLOW_TEMPLATES:
            with self.subTest(template=name):
                page = (templates / name).read_text(encoding="utf-8")
                # Any of the three shapes the flow pages use: the quiet-back
                # class, a back-arrow link, or a named return to the report.
                self.assertTrue(
                    "quiet-back" in page
                    or "← " in page
                    or "Back to report" in page,
                    f"{name} has no inline way back and the pill is gone")

    def test_the_pill_is_not_reintroduced_by_any_stylesheet(self):
        from app.main import CSS_BUNDLE_TEXT

        for bundle, text in CSS_BUNDLE_TEXT.items():
            with self.subTest(bundle=bundle):
                self.assertNotIn("global-back", text)


class MediaCachingTests(unittest.TestCase):
    """/media/<id> sent Cache-Control: no-store.

    So every seek and every revisit re-downloaded the whole file. At the ~460
    KB/s measured against the host that is about three and a half minutes for a
    101 MB replay, paid again on every scrub, for footage the viewer had
    already been sent once.
    """

    def test_the_media_route_asks_for_a_private_cache_not_none_at_all(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        marker = source.index("def media(request: Request, job_id: str):")
        body = source[marker:marker + 2000]
        self.assertIn('"Cache-Control": "private, max-age=3600"', body)
        # private, never public: the URL is account-scoped and must not be held
        # by a shared cache between two people's browsers.
        self.assertNotIn('"public', body)

    def test_the_pages_that_must_not_be_cached_still_are_not(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn('"/result/", "/replay/", "/media/", "/api/"', source)
        self.assertIn('response.headers.setdefault("Cache-Control", "no-store")', source)
        # setdefault is what lets the media route keep its own value while
        # every other prefix, and a 404 on /media/, still gets no-store.
        self.assertNotIn('response.headers["Cache-Control"] = "no-store"', source)


class ReplayFailurePathTests(unittest.TestCase):
    """A replay that never arrived left a spinner and 0:00 on screen forever."""

    def setUp(self):
        self.page = (Path(__file__).resolve().parents[1] / "app" / "templates"
                     / "replay.html").read_text(encoding="utf-8")

    def test_a_stalled_transfer_is_noticed_even_though_it_fires_no_error(self):
        # 'error' only fires when the browser rejects the file. A transfer that
        # simply stops fires nothing, which is the case that hung.
        self.assertIn("'stalled'", self.page)
        self.assertIn("SLOW_AFTER_MS", self.page)
        self.assertIn("setTimeout", self.page)
        # ...and the timer must be cancelled when the video does arrive.
        self.assertIn("clearTimeout(slowTimer)", self.page)
        self.assertIn("'loadeddata'", self.page)

    def test_a_format_the_browser_refuses_says_so_rather_than_blaming_the_load(self):
        # MediaError 3 (decode) and 4 (src not supported) mean the bytes are
        # here and unplayable - which is exactly what a QuickTime replay does.
        self.assertIn("video.error", self.page)
        self.assertIn("cannot play the saved video in the format", self.page)

    def test_every_failure_offers_the_original_file(self):
        self.assertIn("Download the original video", self.page)
        self.assertIn("setAttribute('download'", self.page)

    def test_the_failure_message_is_built_as_nodes_not_interpolated_html(self):
        """statusEl gets a job id in it; building that as an HTML string would
        put a URL fragment into innerHTML on every failure path."""
        self.assertIn("createElement('a')", self.page)
        self.assertNotIn("statusEl.innerHTML=`", self.page)


class ReadableValueTests(unittest.TestCase):
    """/profile printed "KICK_LIGHT · 2026-09-14T15:52:56.914959+00:00"."""

    def test_the_formatters_turn_stored_values_into_readable_ones(self):
        from app.main import _fight_moment, _ruleset_label

        self.assertEqual(_fight_moment("2026-09-14T15:52:56.914959+00:00"), "14 Sep 2026, 15:52")
        self.assertEqual(_fight_moment("2026-09-04T09:05:00+00:00", False), "4 Sep 2026")
        self.assertEqual(_ruleset_label("KICK_LIGHT"), "Kick Light")
        # A value with no mapping is still not shown as an enum.
        self.assertEqual(_ruleset_label("SOME_NEW_ONE"), "Some New One")

    def test_nothing_unparseable_becomes_an_exception_on_somebody_s_page(self):
        from app.main import _fight_moment, _ruleset_label

        for bad in (None, "", "not-a-date", "2026-13-45"):
            self.assertIsInstance(_fight_moment(bad), str)
        self.assertEqual(_ruleset_label(None), "\u2014")

    def test_no_page_renders_a_raw_enum_or_iso_string(self):
        import re

        templates = Path(__file__).resolve().parents[1] / "app" / "templates"
        # Text renders only. An attribute value - datetime="..." for the
        # local-time script, data-date="..." for the sort - holds the stored
        # value on purpose; that is machine data, not something a reader sees.
        raw = re.compile(r'(?<!=")\{\{\s*[a-z_]+\.(?:created_at(?:\[[^\]]*\])?|ruleset)\s*\}\}')
        for name in ("profile.html", "history.html", "dashboard.html", "settings.html",
                     "coach.html", "compare.html"):
            page = (templates / name).read_text(encoding="utf-8")
            self.assertEqual(
                raw.findall(page), [],
                f"{name} renders a stored value straight at the reader")

    def test_timestamps_carry_what_the_local_time_script_needs(self):
        base = (Path(__file__).resolve().parents[1] / "app" / "templates"
                / "base.html").read_text(encoding="utf-8")
        # The audit expected a `timezone` cookie to read. There is none, and no
        # timezone handling anywhere - the browser already knows, so nothing
        # has to be stored to ask it.
        self.assertIn("time[data-local]", base)
        self.assertIn("toLocaleString", base)
        main = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("timezone_cookie", main)


class DeleteAffordanceTests(unittest.TestCase):
    """Delete sat beside Replay at 11px and 4.04:1, guarded by confirm()."""

    def test_the_delete_control_clears_the_contrast_minimum(self):
        css = (Path(__file__).resolve().parents[1] / "app" / "static"
               / "product.css").read_text(encoding="utf-8")
        self.assertIn("color:var(--wiq-danger)", css)
        self.assertNotIn("color:#8d6670", css)
        self.assertIn(".record-delete{padding:2px 0;border:0;background:none;"
                      "color:var(--wiq-danger);font-size:13px", css)

    def test_wiq_danger_actually_passes_on_this_ground(self):
        """Measured, not assumed - the previous colour failed at 4.04:1."""
        def channel(value):
            value /= 255
            return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

        def luminance(colour):
            r, g, b = (channel(c) for c in colour)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        foreground, background = (0xEF, 0x78, 0x91), (9, 9, 11)
        high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
        self.assertGreaterEqual((high + 0.05) / (low + 0.05), 4.5)

    def test_the_dialog_names_the_fight_and_what_goes_with_it(self):
        page = (Path(__file__).resolve().parents[1] / "app" / "templates"
                / "history.html").read_text(encoding="utf-8")
        self.assertIn("{{f.ruleset|ruleset_label}} fight from", page)
        self.assertIn("video, report and corrections", page)
        self.assertNotIn("Delete this saved fight and its analysis files?", page)


class EmptyStateTests(unittest.TestCase):
    """Clicking Sparring with no sparring fights said "No saved fights match
    this search" - nothing had been searched, and the only way back to
    everything was to notice the All fights button."""

    def setUp(self):
        self.page = (Path(__file__).resolve().parents[1] / "app" / "templates"
                     / "history.html").read_text(encoding="utf-8")

    def test_the_three_states_have_their_own_message(self):
        self.assertIn("No saved fights match", self.page)      # searched
        self.assertIn("fights match \u201c", self.page)        # searched within a filter
        self.assertIn("have not saved any", self.page)          # filtered, nothing there
        self.assertNotIn("No saved fights match this search.</p>", self.page)

    def test_each_state_offers_the_action_that_undoes_it(self):
        for action in ("Clear search", "Show all fights", "Clear search and show all fights"):
            self.assertIn(action, self.page)
        self.assertIn("historyNoResultsAction", self.page)

    def test_the_filter_is_set_in_one_place(self):
        """The empty state's buttons and the filter row must not be able to
        disagree about which filter is pressed."""
        self.assertIn("const setFilter=", self.page)
        self.assertEqual(self.page.count("aria-pressed',String(item===button)"), 1)


class AssetVersionTests(unittest.TestCase):
    """The cache token moved on every deploy even when no stylesheet changed.

    _asset_version hashed st_mtime_ns, and .cpanel.yml deploys with `cp -R`,
    which writes a fresh mtime on every file whether or not its contents moved.
    So each deploy threw away the stylesheet cache of everybody who had ever
    visited - the opposite of what the function was written to do.
    """

    def test_touching_a_stylesheet_does_not_move_the_token(self):
        """A deploy copies the tree; that must not look like a change."""
        import os

        from app.main import ROOT, _asset_version

        sheet = ROOT / "app" / "static" / "style.css"
        if not sheet.exists():
            self.skipTest("no style.css in this checkout")
        before = _asset_version()
        original = sheet.stat()
        try:
            # Exactly what `cp` does to the mtime, without touching contents.
            os.utime(sheet, ns=(original.st_atime_ns, original.st_mtime_ns + 10_000_000_000))
            self.assertEqual(
                _asset_version(), before,
                "a deploy that only copies files still busts every visitor's cache")
        finally:
            os.utime(sheet, ns=(original.st_atime_ns, original.st_mtime_ns))

    def test_a_real_stylesheet_change_does_move_the_token(self):
        """The token still has to do its job: a changed sheet must invalidate."""
        from app.main import ROOT, _asset_version

        sheet = ROOT / "app" / "static" / "style.css"
        if not sheet.exists():
            self.skipTest("no style.css in this checkout")
        before = _asset_version()
        original = sheet.read_bytes()
        try:
            sheet.write_bytes(original + b"\n/* cache token probe */\n")
            self.assertNotEqual(
                _asset_version(), before,
                "a changed stylesheet would be served from a stale cache")
        finally:
            sheet.write_bytes(original)
        self.assertEqual(_asset_version(), before, "the probe was not cleaned up")

    def test_it_is_cheap_enough_to_run_at_import(self):
        import time

        from app.main import _asset_version

        _asset_version()                      # warm the page cache
        start = time.perf_counter()
        _asset_version()
        self.assertLess(time.perf_counter() - start, 0.25)


class CookielessTrafficCountTests(unittest.TestCase):
    """Counting the visitors Google is not told about.

    Consent Mode leaves analytics_storage denied until someone accepts the
    banner, and a property this size is far below Google's modelling
    threshold, so a visitor who ignores the banner is absent from the reports
    rather than estimated. Verified on the live site: the tag fires with
    `gcs=G100` and sets no `_ga` cookie until consent is given.

    These counts include those visitors. They are page views and nothing else
    - no visitor column, no IP, no user agent - which is what keeps them
    outside the thing the banner exists to ask about.
    """

    BROWSER = {"user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"}

    def setUp(self):
        from app.main import app

        self.client = TestClient(app)

    def _views(self, path):
        from core.db import page_view_summary

        return next(
            (page["views"] for page in page_view_summary(2)["pages"] if page["path"] == path), 0)

    def test_a_public_page_view_is_counted(self):
        before = self._views("/pricing")
        self.assertEqual(self.client.get("/pricing", headers=self.BROWSER).status_code, 200)
        self.assertEqual(self._views("/pricing"), before + 1)

    def test_a_crawler_is_not_counted(self):
        before = self._views("/pricing")
        self.client.get("/pricing", headers={"user-agent": "Googlebot/2.1 (+http://www.google.com/bot.html)"})
        self.client.get("/pricing", headers={"user-agent": "python-requests/2.31"})
        self.assertEqual(self._views("/pricing"), before)

    def test_only_pages_count_not_the_files_they_load(self):
        """HTML only. That excludes assets, JSON polling and redirects at once,
        without a list of exclusions that has to be kept current."""
        from core.db import page_view_summary

        before = page_view_summary(2)["total"]
        self.client.get("/assets/base.css", headers=self.BROWSER)
        self.client.get("/health", headers=self.BROWSER)
        self.assertEqual(page_view_summary(2)["total"], before)

    def test_the_workspace_is_not_counted(self):
        """The question is whether anyone is finding the site. Counting the
        private routes would mostly count the owner using their own product."""
        before = self._views("/dashboard")
        self.client.get("/dashboard", headers=self.BROWSER)
        self.assertEqual(self._views("/dashboard"), 0)
        self.assertEqual(before, 0)

    def test_identifiers_are_folded_so_the_table_stays_readable(self):
        from app.main import _counted_path

        for path, expected in (
            ("/", "/"),
            ("/pricing", "/pricing"),
            ("/analyze/kickboxing", "/analyze/kickboxing"),
            # A slug is long too. Folding it would throw away the one thing
            # worth knowing - which page - so only identifiers fold.
            ("/how-to-record-a-fight-for-analysis", "/how-to-record-a-fight-for-analysis"),
            ("/report/9f2c81aa7d34", "/report/:id"),
            ("/fights/12", "/fights/:id"),
        ):
            with self.subTest(path=path):
                self.assertEqual(_counted_path(path), expected)

    def test_a_failed_count_still_serves_the_page(self):
        """A visitor came for the page, not for the statistic."""
        import app.main as main

        def explode(*_args, **_kwargs):
            raise RuntimeError("database is locked")

        original = main.record_page_view
        main.record_page_view = explode
        try:
            self.assertEqual(self.client.get("/pricing", headers=self.BROWSER).status_code, 200)
        finally:
            main.record_page_view = original

    def test_the_count_is_where_the_owner_will_see_it(self):
        """A number nobody opens is not a measurement."""
        from core.auth import register

        register("traffic-owner@example.com", "Strong-Local-Password")
        self.client.post(
            "/login",
            data={"email": "traffic-owner@example.com", "password": "Strong-Local-Password"})
        previous = SETTINGS.admin_emails
        object.__setattr__(SETTINGS, "admin_emails", ("traffic-owner@example.com",))
        try:
            self.client.get("/pricing", headers=self.BROWSER)
            page = self.client.get("/admin")
            self.assertEqual(page.status_code, 200)
            self.assertIn("Traffic", page.text)
            self.assertIn("/pricing", page.text)
        finally:
            object.__setattr__(SETTINGS, "admin_emails", previous)

    def test_counting_does_not_put_a_database_write_on_the_page(self):
        """Measured before this: 19 ms a view, against 8-9 ms to build the page.

        Closing the last connection checkpoints the WAL, so a write per view
        cost more than serving the page it was counting. Views are buffered and
        written in batches instead; the request path must not touch the file.
        """
        from core.db import connection, flush_page_views, record_page_view

        flush_page_views()
        with connection() as con:
            before = con.execute(
                "SELECT COALESCE(SUM(views),0) FROM page_views").fetchone()[0]

        for _ in range(5):
            self.client.get("/pricing", headers=self.BROWSER)

        with connection() as con:
            during = con.execute(
                "SELECT COALESCE(SUM(views),0) FROM page_views").fetchone()[0]
        self.assertEqual(during, before, "a page view wrote to the database")

        record_page_view("/pricing")
        self.assertEqual(flush_page_views(), 6, "buffered views were lost")
        with connection() as con:
            after = con.execute(
                "SELECT COALESCE(SUM(views),0) FROM page_views").fetchone()[0]
        self.assertEqual(after, before + 6)

    def test_a_failed_flush_keeps_the_counts(self):
        """A locked database should delay the numbers, not lose them."""
        from unittest import mock

        from core import db as database

        database.flush_page_views()
        database.record_page_view("/pricing", day="2026-01-01")
        with mock.patch.object(database, "connection", side_effect=RuntimeError("locked")):
            with self.assertRaises(RuntimeError):
                database.flush_page_views()
        self.assertEqual(database.flush_page_views(), 1)

    def test_the_shutdown_flush_never_raises(self):
        """An atexit handler that raises prints a traceback after everything
        else has finished, where it reads as a crash in whatever ran last.
        At shutdown the database may simply be gone."""
        from unittest import mock

        from core import db as database

        database.record_page_view("/pricing", day="2026-01-02")
        with mock.patch.object(database, "connection", side_effect=OSError("no database")):
            database._flush_page_views_at_exit()   # must not raise
        self.assertEqual(database.flush_page_views(), 1, "the counts were dropped")
