import io
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi.responses import HTMLResponse

from app import state
from app import main as webapp
from app.main import GUEST_COOKIE, app
from core.analyzer import _live_event_payload, _provisional_stats
from core.config import SETTINGS


class DurableAnalysisStateTests(TestCase):
    def test_new_run_clears_stale_results_and_rejects_older_worker_updates(self):
        with TemporaryDirectory() as temporary:
            with patch.object(state, "OUTPUTS", Path(temporary)):
                state.create_job("generation-job", {
                    "owner_key": "guest:test",
                    "status": "complete",
                    "percent": 100.0,
                    "video_path": "fight.mp4",
                    "live_events": [{"id": "old-event"}],
                    "provisional_stats": {"fighters": {"A": {"attempts": 99}}},
                    "latest_observation": {"time_seconds": 88.0},
                    "report": {"fight": "old"},
                })

                first_run = state.prepare_job_run("generation-job", {})
                self.assertTrue(state.start_job_run("generation-job", "worker-old", first_run))
                self.assertTrue(state.update_job_for_worker(
                    "generation-job", "worker-old", first_run,
                    {"live_events": [{"id": "first-run-event"}], "percent": 20.0},
                ))

                second_run = state.prepare_job_run("generation-job", {})
                self.assertNotEqual(first_run, second_run)
                self.assertFalse(state.update_job_for_worker(
                    "generation-job", "worker-old", first_run,
                    {"status": "complete", "percent": 100.0, "report": {"fight": "stale"}},
                ))
                current = state.get_job("generation-job")
                self.assertEqual(current["analysis_run_id"], second_run)
                self.assertEqual(current["status"], "queued")
                self.assertEqual(current["percent"], 0.0)
                self.assertEqual(current["live_events"], [])
                self.assertEqual(current["provisional_stats"], {})
                self.assertIsNone(current["latest_observation"])
                self.assertIsNone(current.get("report"))
                state.delete_job("generation-job")

    def test_job_listing_refreshes_state_written_by_another_process(self):
        with TemporaryDirectory() as temporary:
            with patch.object(state, "OUTPUTS", Path(temporary)):
                state.create_job("shared-job", {
                    "owner_key": "guest:test",
                    "status": "queued",
                    "video_path": "fight.mp4",
                })
                path = Path(temporary) / "shared-job" / "analysis-session.json"
                persisted = json.loads(path.read_text(encoding="utf-8"))
                persisted.update({"status": "complete", "percent": 100.0})
                path.write_text(json.dumps(persisted), encoding="utf-8")

                listed = dict(state.list_jobs())["shared-job"]
                self.assertEqual(listed["status"], "complete")
                self.assertEqual(listed["percent"], 100.0)
                state.delete_job("shared-job")

    def test_running_session_is_restored_as_interrupted_without_losing_setup(self):
        with TemporaryDirectory() as temporary:
            with patch.object(state, "OUTPUTS", Path(temporary)):
                state.create_job("durable-job", {
                    "owner_key": "guest:test",
                    "status": "running",
                    "video_path": "fight.mp4",
                    "fighter_a_box": [1, 2, 3, 4],
                    "fighter_b_box": [5, 6, 7, 8],
                    "percent": 42.0,
                })
                state._jobs.clear()
                restored = state.get_job("durable-job")

                self.assertEqual(restored["status"], "interrupted")
                self.assertEqual(restored["percent"], 42.0)
                self.assertEqual(restored["fighter_a_box"], [1, 2, 3, 4])
                self.assertTrue((Path(temporary) / "durable-job" / "analysis-session.json").exists())
                state.delete_job("durable-job")

    def test_queued_session_survives_restart_and_can_be_claimed_once(self):
        with TemporaryDirectory() as temporary:
            with patch.object(state, "OUTPUTS", Path(temporary)):
                state.create_job("queued-job", {
                    "owner_key": "guest:test",
                    "status": "queued",
                    "video_path": "fight.mp4",
                })
                state._jobs.clear()
                restored = state.get_job("queued-job")
                self.assertEqual(restored["status"], "queued")

                claimed = state.claim_next_job("gpu-worker-test")
                self.assertIsNotNone(claimed)
                self.assertEqual(claimed[0], "queued-job")
                self.assertEqual(claimed[1]["status"], "running")
                self.assertIsNone(state.claim_next_job("second-worker"))
                state.delete_job("queued-job")

    def test_worker_outlives_faults_that_are_not_connection_errors(self):
        """A long-lived worker must not die on a surprise.

        Only RemoteWorkerError was handled, so any other exception ended the
        process and left the queue unattended until someone noticed and
        restarted it by hand -- which is exactly what happened in production.
        """
        import worker as worker_module
        from core.worker_client import RemoteWorkerError

        seen = {"n": 0}

        def flaky(client):
            seen["n"] += 1
            if seen["n"] == 1:
                raise ValueError("not a connection error")
            if seen["n"] == 2:
                raise RemoteWorkerError("server returned 500")
            if seen["n"] == 3:
                raise KeyError("malformed payload")
            raise SystemExit(0)

        with patch.object(worker_module, "retry_heartbeat", flaky), \
             patch.object(worker_module.time, "sleep", lambda seconds: None):
            with self.assertRaises(SystemExit):
                worker_module.run_remote_worker("test-worker")

        # Three distinct faults, and the loop was still polling after each.
        self.assertEqual(seen["n"], 4)

    def test_wake_hook_never_blocks_or_breaks_the_upload(self):
        """The fight is already queued durably, so a failed wake must be harmless."""
        from core.worker_client import wake_remote_worker

        # No hook configured is the normal PC-worker case.
        self.assertFalse(wake_remote_worker("", "token", "job-1"))

        # An unreachable or failing endpoint must return False, never raise.
        self.assertFalse(
            wake_remote_worker("http://127.0.0.1:9/wake", "token", "job-1", timeout=1.0)
        )

        sent = {}

        class _Response:
            status = 202
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        def _capture(request, timeout=None):
            sent["url"] = request.full_url
            sent["auth"] = request.get_header("Authorization")
            sent["body"] = json.loads(request.data)
            return _Response()

        with patch("urllib.request.urlopen", _capture):
            self.assertTrue(wake_remote_worker("https://gpu.example/wake", "secret", "job-9"))
        self.assertEqual(sent["url"], "https://gpu.example/wake")
        self.assertEqual(sent["auth"], "Bearer secret")
        self.assertEqual(sent["body"], {"job_id": "job-9"})

    def test_fight_is_queued_while_the_analysis_machine_is_offline(self):
        """A durable queue lets a fight wait for the GPU machine to reconnect.

        Refusing here is what made the public site look broken whenever the
        analysis machine was switched off.
        """
        from app.main import _analysis_queue_decision

        previous_mode = SETTINGS.analysis_worker_mode
        previous_deferred = SETTINGS.accept_deferred_analysis
        try:
            object.__setattr__(SETTINGS, "accept_deferred_analysis", True)
            object.__setattr__(SETTINGS, "analysis_worker_mode", "remote")

            offline = {"available": False, "reason": "worker_heartbeat_missing"}
            with patch("app.main.worker_status", return_value=offline):
                self.assertEqual(
                    _analysis_queue_decision(), {"accepted": True, "deferred": True}
                )

            # Nothing would ever claim these, so they must still be refused.
            for reason in ("analysis_dependencies_missing", "worker_token_missing", "worker_mode_invalid"):
                with patch("app.main.worker_status", return_value={"available": False, "reason": reason}):
                    self.assertFalse(_analysis_queue_decision()["accepted"], reason)

            # An in-process server has no detached worker to wait for.
            object.__setattr__(SETTINGS, "analysis_worker_mode", "inprocess")
            with patch("app.main.worker_status", return_value=offline):
                self.assertFalse(_analysis_queue_decision()["accepted"])

            # The operator can switch deferred acceptance off entirely.
            object.__setattr__(SETTINGS, "analysis_worker_mode", "remote")
            object.__setattr__(SETTINGS, "accept_deferred_analysis", False)
            with patch("app.main.worker_status", return_value=offline):
                self.assertFalse(_analysis_queue_decision()["accepted"])
        finally:
            object.__setattr__(SETTINGS, "analysis_worker_mode", previous_mode)
            object.__setattr__(SETTINGS, "accept_deferred_analysis", previous_deferred)

    def test_unpersisted_queue_fails_loudly_when_a_detached_worker_must_claim(self):
        """A detached worker finds queued work only on disk.

        When the session cannot be written the analysis would otherwise sit at
        "Queued" forever, so queueing must raise instead of stranding the fight.
        """
        previous_mode = SETTINGS.analysis_worker_mode
        with TemporaryDirectory() as temporary:
            try:
                with patch.object(state, "OUTPUTS", Path(temporary)):
                    state.create_job("strand-job", {"owner_key": "guest:test", "video_path": "fight.mp4"})
                    with patch.object(state, "_write_session", return_value=False):
                        object.__setattr__(SETTINGS, "analysis_worker_mode", "remote")
                        with self.assertRaises(state.AnalysisStateNotPersisted):
                            state.prepare_job_run("strand-job", {})

                        # In-process runs keep the job in memory, so a failed
                        # write must not block a local PyCharm analysis.
                        object.__setattr__(SETTINGS, "analysis_worker_mode", "inprocess")
                        self.assertTrue(state.prepare_job_run("strand-job", {}))
                    state.delete_job("strand-job")
            finally:
                object.__setattr__(SETTINGS, "analysis_worker_mode", previous_mode)

    def test_session_staging_name_stays_close_to_the_final_name(self):
        """A long staging suffix pushed the temp file past the Windows path limit."""
        with TemporaryDirectory() as temporary:
            with patch.object(state, "OUTPUTS", Path(temporary)):
                state.create_job("path-job", {"owner_key": "guest:test", "video_path": "fight.mp4"})
                final = state._session_path("path-job")
                self.assertTrue(final.is_file())
                staged = final.with_name(f"{final.name}.{'a' * 8}.tmp")
                self.assertLessEqual(len(str(staged)) - len(str(final)), 16)
                state.delete_job("path-job")

    def test_remote_gpu_worker_claims_downloads_updates_and_publishes_one_generation(self):
        client = TestClient(app)
        previous_mode = SETTINGS.analysis_worker_mode
        previous_token = SETTINGS.worker_token
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            uploads = root / "uploads"
            outputs = root / "outputs"
            uploads.mkdir()
            outputs.mkdir()
            video = uploads / "remote-job.mp4"
            video.write_bytes(b"synthetic-video")
            object.__setattr__(SETTINGS, "analysis_worker_mode", "remote")
            object.__setattr__(SETTINGS, "worker_token", "test-worker-token")
            headers = {"Authorization": "Bearer test-worker-token"}
            try:
                with patch.object(state, "OUTPUTS", outputs), patch.object(webapp, "OUTPUTS", outputs), patch.object(webapp, "UPLOADS", uploads):
                    state.create_job("remote-job", {
                        "owner_key": "guest:test", "status": "selecting",
                        "video_path": str(video), "fighter_a_box": [1, 2, 30, 80],
                        "fighter_b_box": [40, 2, 70, 80], "focus_fighter": "A",
                        "fight_type": "competition", "ruleset": "K1", "round_count": 1,
                        "round_duration_seconds": 120.0, "break_duration_seconds": 60.0,
                        "start_seconds": 0.0, "end_seconds": 20.0, "persist_result": False,
                    })
                    run_id = state.prepare_job_run("remote-job", {})

                    unauthorized = client.post("/api/worker/claim", json={"worker_id": "gpu-test"})
                    self.assertEqual(unauthorized.status_code, 401)
                    claimed = client.post(
                        "/api/worker/claim", headers=headers, json={"worker_id": "gpu-test"},
                    )
                    self.assertEqual(claimed.status_code, 200)
                    job = claimed.json()["job"]
                    self.assertEqual(job["job_id"], "remote-job")
                    self.assertEqual(job["analysis_run_id"], run_id)
                    self.assertNotIn("video_path", job)

                    downloaded = client.get(
                        "/api/worker/jobs/remote-job/video",
                        params={"worker_id": "gpu-test", "analysis_run_id": run_id},
                        headers=headers,
                    )
                    self.assertEqual(downloaded.status_code, 200)
                    self.assertEqual(downloaded.content, b"synthetic-video")
                    progressed = client.post(
                        "/api/worker/jobs/remote-job/progress", headers=headers,
                        json={"worker_id": "gpu-test", "analysis_run_id": run_id,
                              "patch": {"percent": 45, "message": "Tracking fighters", "status": "complete"}},
                    )
                    self.assertEqual(progressed.status_code, 200)
                    self.assertEqual(state.get_job("remote-job")["status"], "running")

                    report = {key: {} for key in webapp._WORKER_REPORT_KEYS}
                    report["scorecard"] = {"totals": {"A": None, "B": None}}
                    archive = io.BytesIO()
                    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                        bundle.writestr("report.json", json.dumps(report))
                        bundle.writestr("tracking.jsonl", '{"fighter":"A"}\n')
                        bundle.writestr("events.json", "[]")
                    completed = client.post(
                        "/api/worker/jobs/remote-job/complete", headers=headers,
                        data={"worker_id": "gpu-test", "analysis_run_id": run_id},
                        files={"archive": ("worker-result.zip", archive.getvalue(), "application/zip")},
                    )
                    self.assertEqual(completed.status_code, 201)
                    self.assertEqual(state.get_job("remote-job")["status"], "complete")
                    self.assertTrue((outputs / "remote-job" / "report.json").is_file())
                    self.assertTrue((outputs / "remote-job" / "tracking.jsonl").is_file())
                    repeated = client.post(
                        "/api/worker/jobs/remote-job/complete", headers=headers,
                        data={"worker_id": "gpu-test", "analysis_run_id": run_id},
                        files={"archive": ("worker-result.zip", archive.getvalue(), "application/zip")},
                    )
                    self.assertEqual(repeated.status_code, 201)
                    self.assertTrue(repeated.json()["already_complete"])
            finally:
                state.delete_job("remote-job")
                object.__setattr__(SETTINGS, "analysis_worker_mode", previous_mode)
                object.__setattr__(SETTINGS, "worker_token", previous_token)
                client.close()

    def test_readiness_probe_checks_runtime_dependencies(self):
        client = TestClient(app)
        try:
            response = client.get("/ready")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ready"])
            self.assertEqual(
                set(payload["components"]), {"database", "storage", "analysis_worker"},
            )
        finally:
            client.close()

    def test_inprocess_worker_is_not_ready_when_the_analysis_engine_is_missing(self):
        real_find_spec = state.importlib.util.find_spec
        with patch.object(
            state.importlib.util,
            "find_spec",
            side_effect=lambda name: None if name == "ultralytics" else real_find_spec(name),
        ):
            status = state.worker_status()
        self.assertFalse(status["available"])
        self.assertEqual(status["reason"], "analysis_dependencies_missing")
        self.assertIn("ultralytics", status["missing_dependencies"])

    def test_live_template_is_video_first_and_never_claims_unvalidated_actions(self):
        template = (
            Path(__file__).resolve().parents[1] / "app" / "templates" / "progress.html"
        ).read_text(encoding="utf-8")

        self.assertIn('id="liveVideo"', template)
        self.assertIn('id="eventFeed"', template)
        self.assertIn('id="analysisMarkers"', template)
        self.assertIn("action model has not passed WarriorIQ’s release gate", template)
        self.assertIn("Restart preserved analysis", template)
        self.assertIn("Leave analysis running", template)
        self.assertIn('id="currentAnalysis"', template)
        self.assertIn("event.time_seconds)-1", template)
        self.assertIn("feedPinned", template)
        self.assertIn("markerElementById", template)
        self.assertNotIn('autoplay muted', template)
        self.assertIn('preload="metadata"', template)
        self.assertIn("startPlaybackWhenAnalysisStarts", template)
        self.assertIn("scheduleReportTransition", template)
        self.assertIn("location.assign(resultUrl)", template)
        self.assertIn('id="analysisWarmup"', template)
        self.assertIn("warmupVisible", template)
        self.assertIn("Math.min(99.9,backendPercent)", template)
        fixes = (
            Path(__file__).resolve().parents[1] / "app" / "static" / "fixes.css"
        ).read_text(encoding="utf-8")
        self.assertIn(".live-event-empty[hidden]{display:none}", fixes)
        self.assertIn(".analysis-warmup[hidden]{display:none}", fixes)
        self.assertIn("@media(prefers-reduced-motion:reduce)", fixes)

    def test_report_generation_has_real_late_pipeline_progress(self):
        analyzer = (
            Path(__file__).resolve().parents[1] / "core" / "analyzer.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"Building performance report", 98.3', analyzer)
        self.assertIn('"Finalizing coaching priorities", 99.2', analyzer)
        self.assertIn('"Saving completed report", 99.7', analyzer)
        self.assertIn('stage="report"', analyzer)

    def test_current_analysis_pointer_advances_to_results_instead_of_disappearing(self):
        client = TestClient(app)
        job_id = "current-analysis-nav"
        try:
            client.get("/")
            guest_id = client.cookies.get(GUEST_COOKIE)
            client.cookies.set("warrioriq_active_analysis", job_id)
            state.create_job(job_id, {
                "owner_key": f"guest:{guest_id}",
                "status": "running",
                "percent": 44.0,
                "video_path": "fight.mp4",
            })

            running = client.get("/dashboard").text
            self.assertIn(f'href="/progress/{job_id}"', running)
            self.assertIn("Analyzing 44%", running)

            state.update_job(job_id, {"status": "complete", "percent": 100.0})
            completed = client.get("/dashboard").text
            self.assertIn(f'href="/result/{job_id}"', completed)
            self.assertIn("Results ready", completed)
            active = client.get("/api/active-analysis").json()
            self.assertEqual(active["status"], "complete")
            self.assertEqual(active["url"], f"/result/{job_id}")
        finally:
            state.delete_job(job_id)
            client.close()

    def test_stale_completed_cookie_never_overrides_a_processing_fight(self):
        client = TestClient(app)
        completed_id = "completed-analysis-cookie"
        active_id = "new-processing-analysis"
        try:
            client.get("/")
            guest_id = client.cookies.get(GUEST_COOKIE)
            owner_key = f"guest:{guest_id}"
            state.create_job(completed_id, {
                "owner_key": owner_key,
                "status": "complete",
                "percent": 100.0,
                "video_path": "completed.mp4",
            })
            state.create_job(active_id, {
                "owner_key": owner_key,
                "status": "running",
                "percent": 37.0,
                "video_path": "active.mp4",
            })
            client.cookies.set("warrioriq_active_analysis", completed_id)

            dashboard = client.get("/dashboard").text
            self.assertIn(f'href="/progress/{active_id}"', dashboard)
            self.assertIn("Analyzing 37%", dashboard)
            self.assertNotIn(f'href="/result/{completed_id}"', dashboard)

            navigation = client.get("/api/active-analysis").json()
            self.assertEqual(navigation["active_analysis_id"], active_id)
            self.assertEqual(navigation["last_completed_analysis_id"], completed_id)
            self.assertEqual(navigation["url"], f"/progress/{active_id}")
        finally:
            state.delete_job(completed_id)
            state.delete_job(active_id)
            client.close()

    def test_opening_a_completed_report_clears_the_ready_notification(self):
        client = TestClient(app)
        job_id = "completed-report-seen"
        rendered_state = {}
        with TemporaryDirectory() as temporary, patch.object(state, "OUTPUTS", Path(temporary)), patch.object(webapp, "OUTPUTS", Path(temporary)):
            try:
                client.get("/")
                guest_id = client.cookies.get(GUEST_COOKIE)
                client.cookies.set("warrioriq_active_analysis", job_id)
                state.create_job(job_id, {
                    "owner_key": f"guest:{guest_id}",
                    "status": "complete",
                    "percent": 100.0,
                    "video_path": "fight.mp4",
                })
                report_dir = Path(temporary) / job_id
                report_dir.mkdir(exist_ok=True)
                (report_dir / "report.json").write_text(
                    '{"events":[],"key_moments":[],"tracking":{"fighter_A_coverage":1,"fighter_B_coverage":1},"video":{"analysis_target":"BOTH"},"scorecard":{"available":false}}',
                    encoding="utf-8",
                )

                def fake_template_response(*, request, name, context):
                    rendered_state["active_analysis"] = request.state.active_analysis
                    return HTMLResponse("report")

                with (
                    patch.object(webapp.templates, "TemplateResponse", side_effect=fake_template_response),
                    patch.object(webapp, "_apply_report_annotations"),
                    patch.object(webapp, "refresh_identity_integrity"),
                    patch.object(webapp, "_analysis_quality_summary", return_value={}),
                ):
                    response = client.get(f"/result/{job_id}")

                self.assertEqual(response.status_code, 200)
                self.assertIsNone(rendered_state["active_analysis"])
                cookie_headers = "\n".join(response.headers.get_list("set-cookie"))
                self.assertIn("warrioriq_last_completed_analysis=", cookie_headers)
                self.assertIn("Max-Age=0", cookie_headers)
            finally:
                state.delete_job(job_id)
                client.close()

    def test_live_statistics_are_split_by_strike_family_and_outcome(self):
        events = [
            {"fighter": "A", "family": "punch", "outcome": "clean", "time_seconds": 2.0},
            {"fighter": "A", "family": "punch", "outcome": "missed", "time_seconds": 4.0},
            {"fighter": "A", "family": "kick", "outcome": "blocked", "time_seconds": 6.0},
            {"fighter": "B", "family": "kick", "outcome": "clean", "time_seconds": 3.0},
        ]

        stats = _provisional_stats(events, {"A": 8, "B": 7}, 10, True)["fighters"]

        self.assertEqual(stats["A"]["punch_attempts"], 2)
        self.assertEqual(stats["A"]["punches_landed"], 1)
        self.assertEqual(stats["A"]["punches_missed"], 1)
        self.assertEqual(stats["A"]["kick_attempts"], 1)
        self.assertEqual(stats["A"]["kicks_blocked"], 1)
        self.assertEqual(stats["A"]["total_strikes"], 3)
        self.assertAlmostEqual(stats["A"]["accuracy"], 1 / 3)
        self.assertAlmostEqual(stats["B"]["observation_coverage"], .7)

    def test_outcomes_balance_attempts_and_distinguish_evaded(self):
        events = [
            {"id": "a1", "fighter": "A", "family": "punch", "outcome": "landed", "time_seconds": 1.0},
            {"id": "a2", "fighter": "A", "family": "punch", "outcome": "missed", "time_seconds": 2.0},
            {"id": "a3", "fighter": "A", "family": "punch", "outcome": "blocked", "time_seconds": 3.0},
            {"id": "a4", "fighter": "A", "family": "punch", "outcome": "evaded", "time_seconds": 4.0},
            {"id": "a5", "fighter": "A", "family": "kick", "outcome": "uncertain", "time_seconds": 5.0},
        ]

        fighter = _provisional_stats(events, {"A": 10, "B": 10}, 10, True, 10.0)["fighters"]["A"]

        self.assertEqual(fighter["attempts"], 5)
        self.assertEqual(fighter["landed"], 1)
        self.assertEqual(fighter["missed"], 1)
        self.assertEqual(fighter["blocked"], 1)
        self.assertEqual(fighter["evaded"], 1)
        self.assertEqual(fighter["uncertain"], 1)
        self.assertEqual(
            fighter["attempts"],
            fighter["landed"] + fighter["missed"] + fighter["blocked"] + fighter["evaded"] + fighter["uncertain"],
        )
        self.assertIsNone(fighter["accuracy"])
        self.assertAlmostEqual(fighter["punch_accuracy"], .25)
        self.assertIsNone(fighter["kick_accuracy"])

    def test_combination_requires_an_uninterrupted_same_fighter_sequence(self):
        events = [
            {"id": "a1", "fighter": "A", "family": "punch", "technique": "jab", "outcome": "landed", "time_seconds": 1.0, "round_number": 1},
            {"id": "a2", "fighter": "A", "family": "punch", "technique": "cross", "outcome": "missed", "time_seconds": 1.6, "round_number": 1},
            {"id": "b1", "fighter": "B", "family": "punch", "technique": "jab", "outcome": "missed", "time_seconds": 2.0, "round_number": 1},
            {"id": "a3", "fighter": "A", "family": "kick", "technique": "right_low_kick", "outcome": "landed", "time_seconds": 2.4, "round_number": 1},
            {"id": "a4", "fighter": "A", "family": "punch", "technique": "jab", "outcome": "landed", "time_seconds": 4.2, "round_number": 1},
            {"id": "a5", "fighter": "A", "family": "punch", "technique": "cross", "outcome": "landed", "time_seconds": 4.8, "round_number": 1},
            {"id": "a6", "fighter": "A", "family": "kick", "technique": "left_low_kick", "outcome": "blocked", "time_seconds": 5.4, "round_number": 1},
        ]

        fighter = _provisional_stats(events, {"A": 10, "B": 10}, 10, True, 10.0)["fighters"]["A"]

        self.assertEqual(fighter["combinations"], 2)
        self.assertEqual(fighter["longest_combination"], 3)
        self.assertEqual(fighter["combination_sequences"][0]["techniques"], ["jab", "cross"])
        self.assertEqual(fighter["combination_sequences"][1]["techniques"], ["jab", "cross", "left_low_kick"])

    def test_complete_statistics_include_techniques_combinations_and_round_outcomes(self):
        events = [
            {"id": "a1", "fighter": "A", "family": "punch", "technique": "jab", "outcome": "landed", "time_seconds": 1.0, "round_number": 1},
            {"id": "a2", "fighter": "A", "family": "punch", "technique": "jab", "outcome": "missed", "time_seconds": 1.8, "round_number": 1},
            {"id": "a3", "fighter": "A", "family": "kick", "technique": "right_low_kick", "outcome": "blocked", "time_seconds": 4.0, "round_number": 1},
        ]

        stats = _provisional_stats(events, {"A": 20, "B": 18}, 20, True, 60.0)
        fighter = stats["fighters"]["A"]
        jab = fighter["technique_breakdown"]["jab"]

        self.assertEqual(jab["attempts"], 2)
        self.assertEqual(jab["landed"], 1)
        self.assertEqual(jab["missed"], 1)
        self.assertEqual(fighter["failed_combinations"], 1)
        self.assertEqual(fighter["successful_combinations"], 0)
        self.assertEqual(fighter["combination_success_rate"], 0.0)
        round_a = stats["rounds"][0]["fighters"]["A"]
        self.assertEqual(round_a["blocked"], 1)
        self.assertEqual(round_a["families"]["punch"]["attempts"], 2)
        self.assertEqual(round_a["families"]["kick"]["attempts"], 1)

    def test_tactical_performance_is_derived_from_supported_fight_events(self):
        events = [
            {"id": "a1", "fighter": "A", "family": "punch", "technique": "jab", "outcome": "landed", "time_seconds": 1.0, "round_number": 1},
            {"id": "b1", "fighter": "B", "family": "punch", "technique": "cross", "outcome": "landed", "time_seconds": 1.4, "round_number": 1},
            {"id": "a2", "fighter": "A", "family": "kick", "technique": "right_low_kick", "outcome": "blocked", "time_seconds": 2.0, "round_number": 1},
            {"id": "b2", "fighter": "B", "family": "kick", "technique": "left_body_kick", "outcome": "evaded", "time_seconds": 3.0, "round_number": 1},
            {"id": "a3", "fighter": "A", "family": "knee", "technique": "right_knee", "outcome": "landed", "time_seconds": 61.0, "round_number": 2},
            {"id": "b3", "fighter": "B", "family": "knee", "technique": "left_knee", "outcome": "blocked", "time_seconds": 61.5, "round_number": 2},
            {"id": "a4", "fighter": "A", "family": "punch", "technique": "cross", "outcome": "missed", "time_seconds": 62.0, "round_number": 2},
            {"id": "a5", "fighter": "A", "family": "kick", "technique": "left_body_kick", "outcome": "landed", "time_seconds": 63.0, "round_number": 2},
            {"id": "b4", "fighter": "B", "family": "punch", "technique": "jab", "outcome": "missed", "time_seconds": 64.0, "round_number": 2},
        ]

        stats = _provisional_stats(events, {"A": 100, "B": 100}, 100, True, 120.0)
        fighter = stats["fighters"]["A"]

        self.assertEqual(fighter["knee_attempts"], 1)
        self.assertEqual(fighter["knees_landed"], 1)
        self.assertAlmostEqual(fighter["initiative_share"], 5 / 9)
        self.assertEqual(fighter["attack_mix"], {"punch": .4, "kick": .4, "knee": .2})
        self.assertAlmostEqual(fighter["defensive_denial_rate"], .5)
        self.assertAlmostEqual(fighter["clean_exposure_rate"], .25)
        self.assertEqual(fighter["round_profile"]["peak_landed_round"], 2)
        self.assertEqual(fighter["round_profile"]["peak_round_landed"], 2)
        self.assertEqual(fighter["round_profile"]["attempt_change"], 1)
        self.assertEqual(stats["comparison"]["combined_attempts"], 9)
        self.assertAlmostEqual(stats["comparison"]["initiative_margin"], 1 / 9)
        self.assertEqual(stats["comparison"]["landed_margin"], 2)

    def test_unvalidated_live_pipeline_emits_only_generic_observed_attempts(self):
        candidate = SimpleNamespace(
            fighter="A", round_number=1, peak_time=12.4, peak_frame=372,
            technique="left_head_kick", family="kick", limb="left_leg",
            target="head", outcome="clean", confidence=.94, contact_confidence=.91,
            attempted=True,
            metadata={"attacker_identity_confidence": .93, "opponent_identity_confidence": .90},
        )
        weak_candidate = SimpleNamespace(
            fighter="B", round_number=1, peak_time=13.0, peak_frame=390,
            technique="jab", family="punch", limb="left_hand", target="head",
            outcome="missed", confidence=.45, contact_confidence=.20, attempted=True,
            metadata={"attacker_identity_confidence": .91, "opponent_identity_confidence": .91},
        )

        observed = _live_event_payload([candidate, weak_candidate], "K1", False, limit=None)

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["fighter"], "A")
        self.assertEqual(observed[0]["family"], "kick")
        self.assertEqual(observed[0]["verification"], "observed")
        self.assertIsNone(observed[0]["technique"])
        self.assertIsNone(observed[0]["target"])
        self.assertEqual(observed[0]["outcome"], "unclassified")

        stats = _provisional_stats(observed, {"A": 9, "B": 8}, 10, False, 30.0)
        self.assertFalse(stats["action_labels_available"])
        self.assertTrue(stats["attempt_counts_available"])
        self.assertEqual(stats["fighters"]["A"]["kick_attempts"], 1)
        self.assertEqual(stats["fighters"]["A"]["total_strikes"], 1)
        self.assertIsNone(stats["fighters"]["A"]["kicks_landed"])
        self.assertIsNone(stats["fighters"]["A"]["accuracy"])

    def test_live_status_is_owner_scoped_and_excludes_private_job_fields(self):
        client = TestClient(app)
        other_client = TestClient(app)
        job_id = "live-status-test"
        try:
            client.get("/")
            guest_id = client.cookies.get(GUEST_COOKIE)
            state.create_job(job_id, {
                "owner_key": f"guest:{guest_id}",
                "status": "running",
                "percent": 37.5,
                "message": "Analyzing fight",
                "video_path": "private-video-path.mp4",
                "original_name": "private-filename.mp4",
                "video_duration": 120.0,
                "processed_video_seconds": 24.0,
                "live_event_mode": "movement_only",
                "live_events": [],
                "provisional_stats": {"action_labels_available": False},
            })

            response = client.get(f"/api/status/{job_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["percent"], 37.5)
            self.assertEqual(payload["video_url"], f"/media/{job_id}")
            self.assertNotIn("video_path", payload)
            self.assertNotIn("original_name", payload)
            self.assertNotIn("owner_key", payload)
            self.assertEqual(other_client.get(f"/api/status/{job_id}").status_code, 404)
            self.assertIn("Watch WarriorIQ read the fight", client.get(f"/progress/{job_id}").text)
        finally:
            state.delete_job(job_id)
            client.close()
            other_client.close()
