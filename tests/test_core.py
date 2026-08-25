from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from core.action import ActionEngine, ActiveLimb
from core.annotations import _sample, accuracy_summary
from core.auth import hash_password, normalize_email, verify_password
from core.contact import classify_contact
from core.coaching import build_pose_coaching, build_training_plan
from core.config import SETTINGS
from core.identity import IdentityManager
from core.metrics import MetricsAccumulator
from core.pose_tracker import find_initial_people
from core.progress_insights import build_progress
from core.quality_guardian import quality_summary
from core.report import refresh_identity_integrity
from core.scoring import deduplicate_scoring_events, event_legality, is_legal_event, score_fight
from core.sam_recovery import nearest_guidance, sam_sampling_stride
from core.types import PersonObservation, StrikeEvent


def _person(track_id: int, x1: float, x2: float, appearance_index: int) -> PersonObservation:
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = np.linspace(x1 + 4, x2 - 4, 17)
    kp[:, 1] = np.linspace(100, 290, 17)
    appearance = np.zeros((64,), dtype=np.float32)
    appearance[appearance_index] = 1.0
    return PersonObservation(
        track_id=track_id,
        box=np.asarray([x1, 80, x2, 310], dtype=np.float32),
        confidence=0.95,
        keypoints=kp,
        keypoint_conf=np.ones((17,), dtype=np.float32),
        appearance=appearance,
    )


def _body_keypoints():
    kp = [[0.0, 0.0] for _ in range(17)]
    kp[0] = [200, 100]
    kp[1], kp[2], kp[3], kp[4] = [195, 98], [205, 98], [190, 102], [210, 102]
    kp[5], kp[6] = [185, 140], [215, 140]
    kp[9], kp[10] = [180, 155], [220, 155]
    kp[11], kp[12] = [190, 220], [210, 220]
    kp[13], kp[14], kp[15], kp[16] = [190, 270], [210, 270], [190, 320], [210, 320]
    return kp


class ProductFoundationTests(unittest.TestCase):
    @staticmethod
    def _pose_metrics(guard: float, balance: float, center: float) -> dict:
        return {
            "pose_coverage": .9,
            "guard_index": guard,
            "balance_index": balance,
            "ring_center_control": center,
            "attacks": {},
            "combinations": {},
        }

    def test_unvalidated_actions_still_produce_fighter_specific_pose_plan(self):
        own_a = self._pose_metrics(.22, .81, .63)
        own_b = self._pose_metrics(.58, .44, .76)
        coaching_a = build_pose_coaching("A", own_a)
        coaching_b = build_pose_coaching("B", own_b)
        plan_a = build_training_plan(coaching_a, "A", own_a)
        plan_b = build_training_plan(coaching_b, "B", own_b)

        self.assertEqual(coaching_a["evidence_type"], "pose_only")
        self.assertIn("Guard position 22.0%", coaching_a["improvements"][0]["title"])
        self.assertIn("Post-action balance 44.0%", coaching_b["improvements"][0]["title"])
        self.assertTrue(plan_a)
        self.assertTrue(plan_b)
        self.assertNotEqual(plan_a, plan_b)
        self.assertIn("22.0%", plan_a[0]["goal"])
        self.assertIn("30.0%", plan_a[0]["goal"])

    def test_legacy_low_overlap_referee_track_is_invalidated_despite_high_coverage(self):
        metric = self._pose_metrics(.4, .7, .6)
        report = {
            "video": {"analysis_target": "BOTH"},
            "tracking": {
                "fighter_A_coverage": .969,
                "fighter_B_coverage": .947,
                "fighter_A_seed_source": "pose_detector",
                "fighter_B_seed_source": "pose_detector",
                "initial_iou_A": .1277,
                "initial_iou_B": .8413,
            },
            "integrity": {"action_metrics_trusted": True},
            "scorecard": {"available": True, "totals": {"A": 40, "B": 3}},
            "key_moments": [{"fighter": "A"}],
            "illegal_moves": [{"fighter": "A"}],
            "metrics": {"A": dict(metric), "B": dict(metric)},
            "coaching": {"A": {}, "B": {}},
            "training_plan": {"A": [], "B": []},
        }

        refresh_identity_integrity(report)

        self.assertFalse(report["integrity"]["identity_evidence_trusted"])
        self.assertFalse(report["integrity"]["action_metrics_trusted"])
        self.assertEqual(report["scorecard"]["status"], "identity_integrity_failed")
        self.assertEqual(report["key_moments"], [])
        self.assertEqual(report["coaching"]["A"]["improvements"], [])
        self.assertEqual(report["coaching"]["B"]["evidence_type"], "pose_only")

    def test_annotation_sample_preserves_both_identity_confidences(self):
        observation = {"box": [10, 20, 50, 100], "keypoints": _body_keypoints(), "keypoint_conf": [1.0] * 17}
        record = {
            "source_frame": 12, "time_seconds": 1.25, "round_number": 1,
            "fighter_A": {"identity_confidence": .83, "observation": observation},
            "fighter_B": {"identity_confidence": .71, "observation": observation},
        }
        sample = _sample(record, "A")
        self.assertIsNotNone(sample)
        self.assertEqual(sample.identity_confidence, .83)
        self.assertEqual(sample.opponent_identity_confidence, .71)

    def test_passwords_are_salted_and_verifiable(self):
        first = hash_password("Correct-Horse-2026")
        second = hash_password("Correct-Horse-2026")
        self.assertNotEqual(first, second)
        self.assertTrue(verify_password("Correct-Horse-2026", first))
        self.assertFalse(verify_password("wrong-password", first))
        self.assertEqual(normalize_email(" Athlete@Example.COM "), "athlete@example.com")

    def test_quality_guardian_reports_measured_limitations(self):
        result = quality_summary(width=640, height=360, fps=15, brightness=22, sharpness=18)
        self.assertEqual(result["status"], "review")
        self.assertLess(result["score"], 70)
        self.assertTrue(any("light" in note.lower() for note in result["notes"]))
        self.assertTrue(any("motion" in note.lower() for note in result["notes"]))

    def test_progress_uses_real_reports_and_never_invents_missing_metrics(self):
        reports = [
            {"created_at": "2026-08-01T00:00:00+00:00", "job_id": "one", "report": {
                "video": {"analysis_target": "BOTH"}, "setup": {"ruleset": "K1"},
                "tracking": {"fighter_A_coverage": .88},
                "metrics": {"A": {"pose_coverage": .88, "attacks": {"accuracy": None, "attempts": 8},
                                  "dashboard": {"activity_attempts_per_minute": 4.0, "combinations_per_minute": 1.0}}},
                "coaching": {"A": {"improvements": []}}, "training_plan": {"A": []},
            }},
            {"created_at": "2026-08-10T00:00:00+00:00", "job_id": "two", "report": {
                "video": {"analysis_target": "BOTH"}, "setup": {"ruleset": "K1"},
                "tracking": {"fighter_A_coverage": .93},
                "metrics": {"A": {"pose_coverage": .93, "attacks": {"accuracy": .62, "attempts": 12},
                                  "dashboard": {"activity_attempts_per_minute": 6.0, "combinations_per_minute": 1.5}}},
                "coaching": {"A": {"improvements": [{"title": "Guard recovery", "detail": "Hands return late."}]}},
                "training_plan": {"A": [{"focus": "Guard recovery", "work": "Three technical rounds", "goal": "Return both hands"}]},
            }},
            {"created_at": "2026-08-12T00:00:00+00:00", "job_id": "three", "report": {
                "video": {"analysis_target": "B"}, "setup": {"ruleset": "K1"},
                "metrics": {"A": {"pose_coverage": .99, "attacks": {"accuracy": .99, "attempts": 30},
                                  "dashboard": {"activity_attempts_per_minute": 15.0, "combinations_per_minute": 6.0}}},
                "coaching": {"A": {"improvements": [{"title": "Wrong fighter", "detail": "Must remain hidden."}]}},
                "training_plan": {"A": []},
            }},
        ]
        result = build_progress(reports, "A")
        self.assertEqual(result["fight_count"], 2)
        self.assertEqual(result["latest"]["accuracy"], .62)
        self.assertIsNone(result["trends"]["accuracy"])
        self.assertEqual(result["trends"]["activity"], 2.0)
        self.assertEqual(result["focus"][0]["title"], "Guard recovery")

    def test_progress_keeps_pose_baselines_but_hides_untrusted_action_candidates(self):
        records = [{"created_at": "2026-08-10T00:00:00+00:00", "job_id": "pose", "report": {
            "video": {"analysis_target": "BOTH", "focus_fighter": "A"},
            "setup": {"ruleset": "K1"},
            "integrity": {"action_metrics_trusted": False},
            "metrics": {"A": {
                "pose_coverage": .91, "guard_index": .64, "balance_index": .58,
                "ring_center_control": .47, "footwork_body_lengths_per_second": .32,
                "attacks": {"accuracy": .99, "attempts": 99},
                "dashboard": {"activity_attempts_per_minute": 30.0, "combinations_per_minute": 12.0},
            }},
            "coaching": {"A": {"improvements": []}}, "training_plan": {"A": []},
        }}]
        progress = build_progress(records, "A")
        self.assertEqual(progress["latest"]["guard"], .64)
        self.assertEqual(progress["latest"]["balance"], .58)
        self.assertIsNone(progress["latest"]["accuracy"])
        self.assertIsNone(progress["latest"]["attempts"])
        self.assertEqual(progress["action_fight_count"], 0)


class IdentityTests(unittest.TestCase):
    def test_default_tracker_enables_reid(self):
        tracker = Path(SETTINGS.tracker)
        self.assertTrue(tracker.is_file())
        self.assertIn("with_reid: true", tracker.read_text(encoding="utf-8"))

    def test_track_id_change_keeps_warrioriq_identity(self):
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        a, b = manager.update([_person(7, 106, 206, 1), _person(2, 394, 494, 2)], 1)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(a.track_id, 7)
        self.assertEqual(b.track_id, 2)
        self.assertNotEqual(a.track_id, b.track_id)

    def test_tracker_id_swap_does_not_swap_fighters(self):
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        # The external tracker exchanged its numeric IDs at a crossing, while
        # clothing appearance still identifies the actual combatants.
        a, b = manager.update([_person(2, 110, 210, 1), _person(1, 390, 490, 2)], 1)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(a.track_id, 2)
        self.assertEqual(b.track_id, 1)
        self.assertLess(float(a.box[0]), float(b.box[0]))

    def test_sam_guidance_resolves_positionally_ambiguous_crossing(self):
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        left = _person(11, 160, 260, 3)
        right = _person(12, 340, 440, 4)
        a, b = manager.update(
            [left, right],
            1,
            sam_guidance={"A": right.box.copy(), "B": left.box.copy()},
        )
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(a.track_id, 12)
        self.assertEqual(b.track_id, 11)

    def test_sam_guidance_reopens_motion_gate_after_long_disappearance(self):
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        a_returned = _person(21, 700, 800, 1)
        b_returned = _person(22, 900, 1000, 2)
        a, b = manager.update(
            [a_returned, b_returned],
            100,
            sam_guidance={"A": a_returned.box.copy(), "B": b_returned.box.copy()},
        )
        self.assertEqual(a.track_id, 21)
        self.assertEqual(b.track_id, 22)

    def test_sam_guidance_does_not_bridge_large_video_gap(self):
        tracks = {100: {"A": np.asarray([1, 2, 3, 4], dtype=np.float32)}}
        self.assertIsNotNone(nearest_guidance(tracks, 103, tolerance=3))
        self.assertIsNone(nearest_guidance(tracks, 104, tolerance=3))

    def test_sam_sampling_obeys_absolute_work_budget(self):
        total_frames = 30 * 120
        stride = sam_sampling_stride(30.0, total_frames)
        sampled = int(np.ceil(total_frames / stride))
        self.assertLessEqual(sampled, SETTINGS.sam_continuous_max_frames)
        self.assertGreaterEqual(SETTINGS.sam_continuous_chunk_frames, SETTINGS.sam_continuous_max_frames)

    def test_ai_assignment_cannot_use_one_person_twice(self):
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        a, b = manager.apply_ai_assignment([_person(3, 110, 210, 1), _person(4, 390, 490, 2)], 0, 0, 1, 0.9)
        self.assertIsNone(a)
        self.assertIsNone(b)

    def test_far_coach_is_not_guessed_as_missing_fighter(self):
        manager = IdentityManager(_person(1, 100, 200, 1), _person(2, 400, 500, 2), 0)
        a, b = manager.update([_person(9, 800, 900, 4), _person(2, 395, 495, 2)], 1)
        self.assertIsNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(b.track_id, 2)

    def test_initial_lock_accepts_overlapping_detector_box(self):
        people = [_person(1, 125, 205, 1), _person(2, 425, 505, 2)]
        # Detector boxes can tighten or expand slightly around a selection,
        # but must still overlap the person the user actually selected.
        a, b, _, _ = find_initial_people([120, 80, 200, 310], [420, 80, 500, 310], people)
        self.assertEqual(a.track_id, 1)
        self.assertEqual(b.track_id, 2)

    def test_initial_lock_rejects_low_overlap_referee_from_failed_analysis(self):
        frame = np.zeros((144, 320, 3), dtype=np.uint8)
        manual_a = [252.0197, 30.4453, 282.7586, 105.6416]
        manual_b = [198.0296, 35.5261, 231.9212, 119.3598]
        nearby_referee = PersonObservation(
            track_id=9,
            box=np.asarray([230.2289, 26.2259, 257.3365, 103.7073], dtype=np.float32),
            confidence=0.94,
        )
        fighter_b = PersonObservation(
            track_id=2,
            box=np.asarray([197.1639, 40.3924, 229.2620, 118.7541], dtype=np.float32),
            confidence=0.95,
        )

        a, b, a_score, b_score = find_initial_people(
            manual_a,
            manual_b,
            [nearby_referee, fighter_b],
            frame,
        )

        self.assertIsNone(a.track_id)
        np.testing.assert_allclose(a.box, manual_a, rtol=0, atol=1e-4)
        self.assertEqual(a_score, 0.0)
        self.assertEqual(b.track_id, 2)
        self.assertGreater(b_score, 0.8)

    def test_initial_lock_uses_manual_anchor_when_detector_misses_b(self):
        frame = np.zeros((400, 800, 3), dtype=np.uint8)
        a, b, a_score, b_score = find_initial_people(
            [100, 80, 200, 310],
            [500, 80, 600, 310],
            [_person(1, 100, 200, 1)],
            frame,
        )
        self.assertEqual(a.track_id, 1)
        self.assertIsNone(b.track_id)
        self.assertEqual(b_score, 0.0)
        self.assertEqual(b.box.tolist(), [500.0, 80.0, 600.0, 310.0])


class ContactTests(unittest.TestCase):
    def test_contact_uses_temporal_support(self):
        opponent = _body_keypoints()

        def attacker(wrist_x):
            kp = _body_keypoints()
            kp[10] = [wrist_x, 105]
            return kp

        event = StrikeEvent(
            fighter="A",
            opponent="B",
            round_number=1,
            start_frame=1,
            peak_frame=2,
            end_frame=3,
            start_time=0.1,
            peak_time=0.2,
            end_time=0.3,
            technique="cross",
            family="punch",
            limb="right_hand",
            confidence=0.85,
            evidence={
                "contact_samples": [
                    {"frame": 1, "time": 0.1, "attacker_keypoints": attacker(170), "opponent_keypoints": opponent, "opponent_box": [175, 90, 225, 330]},
                    {"frame": 2, "time": 0.2, "attacker_keypoints": attacker(198), "opponent_keypoints": opponent, "opponent_box": [175, 90, 225, 330]},
                    {"frame": 3, "time": 0.3, "attacker_keypoints": attacker(203), "opponent_keypoints": opponent, "opponent_box": [175, 90, 225, 330]},
                ]
            },
        )
        result = classify_contact(event)
        self.assertEqual(result.target, "head")
        self.assertIn(result.outcome, {"clean", "blocked", "likely_landed"})
        self.assertGreaterEqual(result.evidence["contact_support_frames"], 2)

    def test_punch_in_leg_zone_is_not_scored(self):
        opponent = _body_keypoints()
        attacker = _body_keypoints()
        attacker[10] = [205, 275]
        event = StrikeEvent(
            "A", "B", 1, 1, 2, 3, 0.1, 0.2, 0.3,
            "cross", "punch", "right_hand", confidence=0.9,
            evidence={"contact_samples": [
                {"frame": 2, "time": 0.2, "attacker_keypoints": attacker,
                 "opponent_keypoints": opponent, "opponent_box": [175, 90, 225, 330]},
            ]},
        )
        result = classify_contact(event)
        self.assertEqual(result.outcome, "uncertain")
        self.assertFalse(result.landed)
        self.assertIsNone(result.target)

    def test_contact_frame_drives_time_side_and_kick_height(self):
        opponent = _body_keypoints()

        def attacker(ankle_x):
            kp = _body_keypoints()
            kp[16] = [ankle_x, 220]
            return kp

        event = StrikeEvent(
            "A", "B", 1, 1, 2, 4, 0.1, 0.1, 0.4,
            "left_low_kick", "kick", "right_leg", confidence=0.92,
            evidence={"contact_samples": [
                {"frame": 3, "time": 0.25, "attacker_keypoints": attacker(150),
                 "opponent_keypoints": opponent, "opponent_box": [175, 90, 225, 330]},
                {"frame": 4, "time": 0.30, "attacker_keypoints": attacker(190),
                 "opponent_keypoints": opponent, "opponent_box": [175, 90, 225, 330]},
            ]},
        )
        result = classify_contact(event)
        self.assertEqual(result.target, "body")
        self.assertEqual(result.technique, "right_body_kick")
        self.assertEqual(result.peak_frame, 4)
        self.assertAlmostEqual(result.peak_time, 0.30)
        self.assertAlmostEqual(result.evidence["action_peak_time"], 0.1)


class ActionStateTests(unittest.TestCase):
    def test_side_correction_does_not_leave_wrong_active_limb_alive(self):
        engine = ActionEngine()
        state = engine.states["A"]
        start = _person(1, 100, 300, 1)
        opponent = _person(2, 400, 600, 2)
        start.keypoints = np.asarray(_body_keypoints(), dtype=np.float32)
        peak = _person(1, 100, 300, 1)
        peak.keypoints = start.keypoints.copy()
        peak.keypoints[15] += np.asarray([1, 0], dtype=np.float32)
        peak.keypoints[16] += np.asarray([110, 0], dtype=np.float32)
        start_sample = engine._make_sample(1, 0.1, 1, start, opponent, 0.95, 0.95)
        peak_sample = engine._make_sample(2, 0.2, 1, peak, opponent, 0.95, 0.95)
        state.samples.append(start_sample)
        state.active["left_leg"] = ActiveLimb(
            fighter="A", limb="left_leg", family="kick",
            start_sample=start_sample, peak_sample=peak_sample,
            max_speed=SETTINGS.min_strike_speed_body_lengths_per_s * 2,
            start_extension=0.1, peak_extension=0.8,
            frames_active=max(6, SETTINGS.action_window),
        )

        events = engine.update("A", 3, 0.3, 1, peak, opponent, 0.95, 0.95)
        self.assertNotIn("left_leg", state.active)
        self.assertTrue(any(event.limb == "right_leg" for event in events))


class ScoringTests(unittest.TestCase):
    def _event(self, family="punch", target="head", technique="cross"):
        return StrikeEvent(
            "A", "B", 1, 1, 2, 3, 0.1, 0.2, 0.3,
            technique, family, "right_hand",
            outcome="clean", landed=True, target=target,
            confidence=0.9, contact_confidence=0.9,
        )

    def test_knee_legal_in_k1_not_full_contact(self):
        knee = self._event("knee", "body", "right_knee")
        self.assertTrue(is_legal_event(knee, "K1"))
        self.assertFalse(is_legal_event(knee, "FULL_CONTACT"))

    def test_wako_backfist_matrix(self):
        ordinary = self._event("punch", "head", "backfist")
        spinning = self._event("punch", "head", "spinning_backfist")
        self.assertTrue(is_legal_event(ordinary, "POINT_FIGHTING"))
        self.assertFalse(is_legal_event(ordinary, "LIGHT_CONTACT"))
        self.assertTrue(is_legal_event(spinning, "K1"))
        self.assertFalse(is_legal_event(spinning, "LOW_KICK"))

    def test_kick_light_knee_is_reported_illegal(self):
        knee = self._event("knee", "body", "right_knee")
        legal, reason = event_legality(knee, "KICK_LIGHT")
        self.assertFalse(legal)
        self.assertIn("not legal in Kick Light", reason)

    def test_front_kick_to_leg_is_illegal_even_when_low_kicks_are_allowed(self):
        kick = self._event("kick", "leg", "right_front_kick")
        self.assertFalse(is_legal_event(kick, "KICK_LIGHT"))
        round_kick = self._event("kick", "leg", "right_low_kick")
        self.assertTrue(is_legal_event(round_kick, "KICK_LIGHT"))

    def test_wako_low_kick_and_knee_matrix(self):
        low = self._event("kick", "leg", "right_low_kick")
        knee = self._event("knee", "body", "right_knee")
        for ruleset in ("POINT_FIGHTING", "LIGHT_CONTACT", "FULL_CONTACT"):
            self.assertFalse(is_legal_event(low, ruleset), ruleset)
        for ruleset in ("KICK_LIGHT", "LOW_KICK", "K1"):
            self.assertTrue(is_legal_event(low, ruleset), ruleset)
        for ruleset in ("POINT_FIGHTING", "LIGHT_CONTACT", "KICK_LIGHT", "FULL_CONTACT", "LOW_KICK"):
            self.assertFalse(is_legal_event(knee, ruleset), ruleset)
        self.assertTrue(is_legal_event(knee, "K1"))

    def test_ring_score_estimate(self):
        result = score_fight([self._event()], "K1", [1], [])
        self.assertGreater(result["totals"]["A"], result["totals"]["B"])

    def test_low_confidence_contact_does_not_score(self):
        event = self._event()
        event.contact_confidence = 0.3
        result = score_fight([event], "KICK_LIGHT", [1], [])
        self.assertEqual(result["totals"], {"A": 0, "B": 0})

    def test_simultaneous_classifier_candidates_count_as_one_action(self):
        first = self._event("punch", "head", "jab")
        second = self._event("kick", "head", "right_head_kick")
        second.peak_time = first.peak_time
        result = score_fight([first, second], "KICK_LIGHT", [1], [])
        self.assertEqual(result["verified_actions_counted"], 1)
        self.assertEqual(result["duplicate_action_candidates_removed"], 1)
        self.assertEqual(result["totals"]["A"], 1)

    def test_one_strike_repeated_across_classifier_frames_is_counted_once(self):
        events = []
        for peak in (4.00, 4.16, 4.31, 4.47):
            event = self._event("kick", "leg", "right_low_kick")
            event.peak_time = peak
            event.limb = "right_leg"
            events.append(event)
        kept, removed = deduplicate_scoring_events(events)
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed, 3)

    def test_fast_opposite_limb_combination_is_not_deduplicated(self):
        right = self._event("punch", "head", "cross")
        right.limb = "right_hand"
        right.peak_time = 4.00
        left = self._event("punch", "head", "jab")
        left.limb = "left_hand"
        left.peak_time = 4.18
        kept, removed = deduplicate_scoring_events([right, left])
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, 0)

    def test_unreliable_tracking_suppresses_score(self):
        result = score_fight([self._event()], "K1", [1], [], reliable=False)
        self.assertFalse(result["available"])
        self.assertEqual(result["totals"], {"A": None, "B": None})


class MetricsTests(unittest.TestCase):
    def test_empty_metrics_do_not_crash(self):
        metrics = MetricsAccumulator(1920, 1080).finalize([], [], 120.0)
        self.assertIn("dashboard", metrics["A"])
        self.assertIsNone(metrics["A"]["dashboard"]["defense_response_rate"])


class AnnotationAccuracyTests(unittest.TestCase):
    def test_empty_ground_truth_has_no_fake_accuracy(self):
        summary = accuracy_summary([])
        self.assertEqual(summary["annotations"], 0)
        self.assertFalse(summary["train_ready"])
        self.assertTrue(all(item["accuracy"] is None for item in summary["metrics"].values()))

    def test_corrections_measure_fields_independently(self):
        item = {"job_id": "fight1", "ruleset": "KICK_LIGHT", "predicted": {
            "fighter": "A", "technique": "left_low_kick", "target": "leg", "outcome": "clean", "family": "kick", "limb": "left_leg"
        }, "corrected": {
            "fighter": "A", "technique": "right_low_kick", "target": "leg", "outcome": "blocked", "family": "kick", "limb": "right_leg"
        }}
        summary = accuracy_summary([item])
        self.assertEqual(summary["metrics"]["fighter_identity"]["accuracy"], 1.0)
        self.assertEqual(summary["metrics"]["limb_side"]["accuracy"], 0.0)
        self.assertEqual(summary["metrics"]["outcome"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
