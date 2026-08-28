from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import state
from app.main import GUEST_COOKIE, app
from core.analyzer import _live_event_payload, _provisional_stats


class DurableAnalysisStateTests(TestCase):
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
        self.assertIn('autoplay muted', template)
        self.assertIn('preload="auto"', template)
        self.assertIn("scheduleReportTransition", template)
        self.assertIn("location.assign(resultUrl)", template)
        fixes = (
            Path(__file__).resolve().parents[1] / "app" / "static" / "fixes.css"
        ).read_text(encoding="utf-8")
        self.assertIn(".live-event-empty[hidden]{display:none}", fixes)

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
