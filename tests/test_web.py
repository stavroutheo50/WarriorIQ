from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

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

    def test_declining_analytics_denies_storage_but_keeps_the_tag_detectable(self):
        """Consent Mode: the tag is present, storage is denied.

        Withholding the tag entirely also hides it from Google's tag detection,
        which then reports a correctly installed site as having no tag.
        """
        response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "essential"})
        self.assertIn("googletagmanager", response.text)
        self.assertIn("'analytics_storage': 'denied'", response.text)
        self.assertNotIn("'analytics_storage': 'granted'", response.text)

    def test_accepting_analytics_grants_storage(self):
        response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"})
        policy = response.headers["content-security-policy"]
        self.assertIn("https://www.googletagmanager.com", policy)
        # The tag is useless if the script loads but its beacons are blocked.
        connect = [part for part in policy.split(";") if part.strip().startswith("connect-src")][0]
        self.assertIn("google-analytics.com", connect)
        self.assertIn("googletagmanager.com/gtag/js", response.text)
        self.assertIn("'analytics_storage': 'granted'", response.text)

    def test_consent_default_is_declared_before_any_tag_loads(self):
        """A default pushed after the loader would let a tag store first."""
        page = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"}).text
        self.assertLess(page.index("gtag('consent', 'default'"), page.index("gtm.js?id="))
        self.assertLess(page.index("gtag('consent', 'default'"), page.index("gtag/js?id="))

    def test_tag_manager_ships_both_halves_and_a_frame_policy(self):
        """GTM needs the head script, the noscript iframe, and frame-src.

        The noscript fallback is an iframe, which the site's default-src would
        block, so a container that loads but cannot frame is only half working.
        """
        response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "all"})
        self.assertIn(SETTINGS.gtm_container_id, response.text)
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
        response = self.client.get("/", cookies={"warrioriq_cookie_preferences": "essential"})
        self.assertIn(SETTINGS.gtm_container_id, response.text)
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
        try:
            # The container alone still measures, so the policy must stay open.
            object.__setattr__(SETTINGS, "analytics_measurement_id", "")
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

    def test_public_pages_render(self):
        for path in ("/", "/dashboard", "/history", "/profile", "/coach", "/compare", "/validation", "/pricing", "/privacy", "/login", "/signup"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("WARRIOR", response.text)
                self.assertIn("globalBack", response.text)

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
        self.assertIn("Launch is blocked", self.client.get("/legal").text)
        self.assertIn("does not register", self.client.get("/dmca").text)

    def test_every_sport_states_what_the_analysis_cannot_see(self):
        """Coverage is disclosed at the point of choice, not after the upload.

        The detector reads punches, kicks and knees. For boxing that is the
        whole sport; for MMA it misses the ground entirely. The chooser badges
        every sport so the five can be compared at a glance, and each sport's
        setup page names what it will be silent about in full - before an hour
        is spent on an upload, rather than in the finished report.
        """
        from core.scoring import SPORTS, sport_unobserved

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
            # The chip carries the sport's own accent and leads to the switcher.
            self.assertIn("--sport-accent:226 154 74", elsewhere)
            self.assertIn('class="sport-switch" href="/analyze"', elsewhere)

            # Switching sport switches the shell.
            client.get("/analyze/boxing")
            self.assertIn("--sport-accent:233 106 106", client.get("/history").text)
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

    def test_render_blueprint_uses_health_probe_and_port_aware_entrypoint(self):
        blueprint = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(encoding="utf-8")
        self.assertIn("healthCheckPath: /health", blueprint)
        self.assertIn("startCommand: python run.py", blueprint)

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
        self.assertIn('type="hidden" name="people_permissions_confirmed" value="true"', home)
        self.assertIn('name="minor_permission_status"', home)
        self.assertEqual(home.count('type="radio" name="minor_permission_status"'), 2)
        self.assertNotIn(">Choose one<", home)
        # Rounds are no longer asked for. They still post, derived from the
        # video's real duration, so per-round scoring keeps working without
        # putting two more questions in front of a first-time user.
        self.assertIn('id="roundCount"', home)
        self.assertIn('id="roundSeconds"', home)
        self.assertIn("readDuration", home)
        self.assertIn('name="fight_type" value="competition"', home)
        # The ruleset is the one thing WarriorIQ cannot infer, so it stays.
        self.assertIn('name="ruleset"', home)
        self.assertNotIn('id="fightSettings"', home)
        self.assertIn('id="trackingRecoveryTitle"', home)
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
        self.assertIn('name="accept_policies"', self.client.get("/login").text)
        self.assertIn("Terms of Service", signup)
        self.assertIn("Privacy Policy", signup)
        self.assertIn('name="age_confirmed"', signup)
        self.assertIn("at least 18", signup.lower())
        self.assertIn('name="marketing_consent"', signup)

    def test_professional_home_has_product_and_trust_sections(self):
        response = self.client.get("/")
        self.assertIn("Four steps. No technical homework.", response.text)
        self.assertIn("Your next round starts here.", response.text)
        self.assertIn("Evidence before claims.", response.text)
        self.assertNotIn("V" + "4", response.text)

    def test_professional_polish_uses_shared_product_surfaces(self):
        home = self.client.get("/").text
        history = self.client.get("/history").text
        shell = (Path(__file__).resolve().parents[1] / "app" / "templates" / "base.html").read_text(encoding="utf-8")
        history_template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "history.html").read_text(encoding="utf-8")

        self.assertIn('/static/product.css', shell)
        self.assertIn("See every shot.", home)
        self.assertIn("Know what to fix.", home)
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
        self.assertRegex(page.text, r'href="/static/motion\.css\?v=[\w-]+"')
        self.assertRegex(page.text, r'src="/static/motion\.js\?v=[\w-]+"')
        self.assertIn('id="pageScrollProgress"', page.text)
        css = self.client.get("/static/motion.css")
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
        self.assertIn("See every shot.", page)
        self.assertIn("Know what to fix.", page)
        self.assertNotIn("WonderIQ", page)
        # "Frame by frame" is industry shorthand a fighter does not read as a
        # promise. The headline says what they get instead.
        self.assertNotIn("Frame by frame", page)
        self.assertNotIn("Fight intelligence", page)

    def test_upload_never_displays_selected_filename(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        self.assertIn("fileButton.classList.toggle('uploaded',ready)", template)
        self.assertIn("Fight video uploaded", template)
        self.assertNotIn("file.name", template)

    def test_replay_overlay_does_not_block_video_controls(self):
        css = self.client.get("/static/fixes.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn(".replay-wrap canvas{pointer-events:none}", css.text)

    def test_logo_asset_is_served(self):
        response = self.client.get("/static/warrioriq-logo.png")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 1000)

    def test_openai_identity_recovery_requires_explicit_opt_in(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        self.assertIn('name="openai_identity_recovery"', template)
        self.assertNotIn('name="openai_identity_recovery" value="true" checked', template)

    def _render_result(self, selection_check):
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
        request = Stub(url=Stub(path="/report/abc"), state=Stub(account=None), cookies={}, headers={})
        return env.get_template("result.html").render(
            request=request, job_id="abc", report=report,
            identity=sport_identity("kickboxing"),
            report_access={"report_tier": "full", "report_label": "Full", "label": "Full"},
            analysis_quality=_analysis_quality_summary(report), can_share=False, unavailable=[],
        )

    def test_the_report_warns_when_the_wrong_two_people_were_picked(self):
        """The measured coach mis-pick on 2.mp4: 175 actions, none landed."""
        page = self._render_result({
            "actions_observed": 175, "actions_in_range": 12, "landed": 0,
            "median_separation_body_lengths": 2.84, "looks_like_a_fight": False,
            "warning": "...", "verdict": "selection_probably_wrong",
        })
        self.assertIn("These may not be the two fighters", page)
        self.assertIn("0 of 175 moves landed", page)
        self.assertIn("2.8 body lengths", page)

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
        self.assertIn("skeletonDelay=.14", template)

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
        self.assertIn("Nothing for you to do", template)

    def test_result_guides_people_through_only_the_core_training_path(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn('class="report-path"', template)
        for anchor in ("report-performance", "report-scorecard", "report-coaching", "report-training"):
            self.assertIn(f'href="#{anchor}"', template)
            self.assertIn(f'id="{anchor}"', template)
        self.assertNotIn('id="report-evidence"', template)
        self.assertIn("Replay with skeletons", template)

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
        page = self.client.get("/").text
        menu = page.split('<div class="mobile-menu"', 1)[1].split('</div>', 1)[0]
        for label in (">Analyze<", ">Progress<", ">Coach<", ">Plans<"):
            self.assertIn(label, menu)
        self.assertNotIn("Accuracy", menu)
        self.assertNotIn("Compare", menu)

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

    def test_accuracy_page_never_invents_measurements(self):
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

    def test_high_coverage_unvalidated_actions_get_preliminary_not_verified_score(self):
        candidate = {
            "peak_time": 4.2, "start_time": 4.0, "end_time": 4.4,
            "round_number": 1, "fighter": "A", "technique": "left_head_kick",
            "family": "kick", "limb": "left_leg", "target": "head",
            "outcome": "clean", "confidence": .92, "contact_confidence": .90,
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
        self.assertTrue(report["scorecard"]["available"])
        self.assertEqual(report["scorecard"]["status"], "preliminary_unvalidated")
        self.assertEqual(report["scorecard"]["evidence"]["evidence_source"], "unvalidated_action_candidates")
        self.assertNotIn("verified_scoring_actions", report["scorecard"]["evidence"])
        self.assertEqual(report["key_moments"], [])
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
