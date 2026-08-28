from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import _apply_report_annotations, _build_replay_chapters, _prediction_at, _review_candidates, app
from core.evidence_trust import automated_evidence_trust


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

    def test_footer_exposes_compact_legal_navigation(self):
        page = self.client.get("/").text
        for path in ("/terms", "/privacy", "/cookies", "/video-upload-policy", "/refunds", "/acceptable-use", "/contact"):
            self.assertIn(f'href="{path}"', page)
        self.assertIn("© 2026 WarriorIQ. All rights reserved.", page)

    def test_health_probe_is_minimal_and_available(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "WarriorIQ"})
        self.assertNotIn("gpu", response.text.lower())

    def test_robots_sitemap_and_private_noindex_are_safe(self):
        robots = self.client.get("/robots.txt")
        self.assertEqual(robots.status_code, 200)
        self.assertIn("User-agent: *", robots.text)
        sitemap = self.client.get("/sitemap.xml")
        self.assertEqual(sitemap.status_code, 200)
        self.assertNotIn("/result/", sitemap.text)
        self.assertNotIn("/media/", sitemap.text)
        self.assertIn('name="robots" content="noindex,nofollow"', self.client.get("/dashboard").text)

    def test_custom_404_is_useful_and_does_not_leak_details(self):
        response = self.client.get("/definitely-not-a-page")
        self.assertEqual(response.status_code, 404)
        self.assertIn("That page left the ring", response.text)
        self.assertIn("Start an analysis", response.text)

    def test_adult_accounts_and_minor_video_permissions_are_separate(self):
        home = self.client.get("/").text
        self.assertIn('name="rights_confirmed"', home)
        self.assertIn('name="people_permissions_confirmed"', home)
        self.assertIn('name="minor_permission_status"', home)
        self.assertIn("Video Upload Policy", home)
        self.assertNotIn("These confirmations apply to this fight video", home)
        self.assertNotIn("Account policies are accepted only", home)
        self.assertIn("18 or older", home.lower())
        self.assertIn("parent or guardian", home.lower())
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
        self.assertIn("Your fight.", home)
        self.assertIn("Frame by frame.", home)
        self.assertIn('class="product-preview"', home)
        self.assertIn('class="workflow-track"', home)
        self.assertIn('class="fight-archive"', history)
        self.assertIn('id="historySearch"', history_template)

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
        self.assertIn('href="/static/motion.css?v=20260828-2"', page.text)
        self.assertIn('src="/static/motion.js?v=20260828-2"', page.text)
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

    def test_home_copy_uses_the_warrioriq_combat_sports_voice(self):
        page = self.client.get("/").text
        self.assertIn("Fight intelligence for athletes and coaches", page)
        self.assertIn("See the fight.", page)
        self.assertIn("Train what matters.", page)
        self.assertNotIn("WonderIQ", page)

    def test_upload_never_displays_selected_filename(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html").read_text(encoding="utf-8")
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
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('name="openai_identity_recovery"', template)
        self.assertNotIn('name="openai_identity_recovery" value="true" checked', template)

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

    def test_progress_shows_elapsed_time_not_realtime_speed(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
        self.assertIn("Analysis running", template)
        self.assertIn("d.elapsed_seconds", template)
        self.assertNotIn("Realtime speed", template)

    def test_replay_uses_rendered_media_time(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "replay.html").read_text(encoding="utf-8")
        self.assertIn("requestVideoFrameCallback", template)
        self.assertIn("metadata.mediaTime", template)

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
        self.assertIn("Fully automatic", template)

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
        for label in ("Latest guard position", "Latest balance", "Latest center position", "Average pose evidence"):
            self.assertIn(label, template)
        self.assertIn("Movement progress is ready", template)
        self.assertIn("Not validated", template)

    def test_account_policies_are_acknowledged_at_auth_not_every_upload(self):
        auth = (Path(__file__).resolve().parents[1] / "app" / "templates" / "auth.html").read_text(encoding="utf-8")
        upload = (Path(__file__).resolve().parents[1] / "app" / "templates" / "index.html").read_text(encoding="utf-8")
        for path in ("/terms", "/privacy", "/acceptable-use"):
            self.assertIn(f'href="{path}"', auth)
        self.assertNotIn("Uploading is subject to", upload)
        self.assertNotIn("Account policies are accepted only when creating an account or signing in", upload)

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
        self.assertIn('href="/#analyze">Analyze another fight', template)

    def test_coach_workspace_uses_the_selected_fighter_and_one_click_plan(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "coach.html").read_text(encoding="utf-8")
        self.assertIn("latest.coaching[focus]", template)
        self.assertNotIn("for fighter in ['A','B']", template)
        self.assertIn("One-click suggestions", template)
        self.assertIn("Mark complete", template)

    def test_advanced_report_diagnostics_are_optional_not_front_and_center(self):
        template = (Path(__file__).resolve().parents[1] / "app" / "templates" / "result.html").read_text(encoding="utf-8")
        self.assertIn('<details class="report-details">', template)
        self.assertIn("More performance details", template)
        self.assertIn('<details class="card report-technical">', template)
        self.assertIn("Technical analysis details", template)
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
            "limb": "left_leg", "target": "body", "outcome": "blocked",
        }}]
        _apply_report_annotations(report, annotations)
        self.assertEqual(report["illegal_moves"], [])
        self.assertEqual(report["key_moments"][0]["technique"], "left_front_kick")
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
        report = {
            "classifier": {"custom_temporal_checkpoint_loaded": False},
            "integrity": {}, "setup": {"ruleset": "K1"},
            "video": {"analysis_target": "BOTH"},
            "tracking": {"fighter_A_coverage": .87, "fighter_B_coverage": .94},
            "rounds": [{"number": 1, "selected": True}],
            "events": [candidate], "key_moments": [candidate], "illegal_moves": [], "metrics": {},
        }
        _apply_report_annotations(report, [])
        self.assertTrue(report["scorecard"]["available"])
        self.assertEqual(report["scorecard"]["status"], "preliminary_unvalidated")
        self.assertEqual(report["scorecard"]["evidence"]["evidence_source"], "unvalidated_action_candidates")
        self.assertNotIn("verified_scoring_actions", report["scorecard"]["evidence"])
        self.assertEqual(report["key_moments"], [])
        self.assertFalse(report["integrity"]["action_metrics_trusted"])

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
            "per_class_test_accuracy": {f"class_{index}": .85 for index in range(18)},
        })
        self.assertTrue(automated_evidence_trust(classifier)["automated_evidence_trusted"])

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
        self.assertIn("Watch moments", template)
        self.assertIn("report.statistics", template)
        self.assertIn('class="report-deep-dive"', template)
        self.assertIn('report["statistics"] = final_live_stats', analyzer)
        self.assertIn('report["event_feed"] = all_final_live_events', analyzer)


if __name__ == "__main__":
    unittest.main()
