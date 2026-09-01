from __future__ import annotations

import unittest
import sqlite3
import json
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from core.action import ActionEngine, ActiveLimb
from core.annotations import _sample, accuracy_summary
from core.auth import hash_password, normalize_email, verify_password
from core.contact import classify_contact
from core.coaching import build_pose_coaching, build_training_plan
from core.config import SETTINGS
from core.identity import IdentityManager
from core.metrics import MetricsAccumulator
from core.model_validation import audit_sequence_directory, classification_metrics
from core.pose_tracker import find_initial_people
from core.progress_insights import build_progress
from core.quality_guardian import quality_summary
from core.report import refresh_identity_integrity
from core.release_validation import assess_end_to_end_validation, end_to_end_metadata
from core.scoring import deduplicate_scoring_events, event_legality, is_legal_event, score_fight
from core.sam_recovery import nearest_guidance, sam_sampling_stride
from core.types import PersonObservation, StrikeEvent
from core.temporal_model import ACTION_CLASSES
from tools import backup_runtime


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

    def _strike(self, family, technique, target, fighter="A"):
        from core.types import StrikeEvent

        return StrikeEvent(
            fighter=fighter, opponent="B" if fighter == "A" else "A", round_number=1,
            start_frame=0, peak_frame=1, end_frame=2, start_time=0.0, peak_time=0.1, end_time=0.2,
            technique=technique, family=family, limb="x", outcome="clean", target=target,
            confidence=1.0, contact_confidence=1.0, model_source="test",
        )

    def test_each_sport_enforces_its_own_legal_techniques(self):
        """A ruleset that shares one legality table scores every sport as kickboxing."""
        from core.scoring import event_legality

        # Boxing: hands only.
        self.assertFalse(event_legality(self._strike("kick", "right_round_kick", "body"), "BOXING")[0])
        self.assertFalse(event_legality(self._strike("knee", "right_knee", "body"), "BOXING")[0])
        self.assertTrue(event_legality(self._strike("punch", "jab", "head"), "BOXING")[0])

        # WT taekwondo: kicks decide it, and a punch to the head is a penalty.
        self.assertFalse(event_legality(self._strike("punch", "cross", "head"), "WT_TAEKWONDO")[0])
        self.assertTrue(event_legality(self._strike("punch", "cross", "body"), "WT_TAEKWONDO")[0])
        self.assertTrue(event_legality(self._strike("kick", "right_round_kick", "head"), "WT_TAEKWONDO")[0])

        # Muay Thai and MMA permit the full observed striking set.
        for ruleset in ("MUAY_THAI", "MMA"):
            self.assertTrue(event_legality(self._strike("knee", "right_knee", "body"), ruleset)[0], ruleset)
            self.assertTrue(event_legality(self._strike("kick", "left_round_kick", "leg"), ruleset)[0], ruleset)

    def test_taekwondo_scores_kicks_above_punches(self):
        """One shared weighting would score a taekwondo round as kickboxing."""
        from core.scoring import RULESETS, _effective_value

        kick = self._strike("kick", "right_round_kick", "head")
        punch = self._strike("punch", "cross", "body")
        tkd, boxing = RULESETS["WT_TAEKWONDO"], RULESETS["BOXING"]
        self.assertGreater(_effective_value(kick, tkd), 2 * _effective_value(punch, tkd))
        # Boxing has no other family to weigh a punch against.
        self.assertGreater(_effective_value(punch, boxing), 0)

    def test_a_sport_declares_the_actions_it_cannot_observe(self):
        """The detector sees punches, kicks and knees. Nothing else.

        A striking read of an MMA round is not a read of the round, so the
        report has to say which one it is giving you rather than presenting a
        partial count as complete.
        """
        from core.scoring import score_fight, unobserved_actions

        self.assertEqual(unobserved_actions("BOXING"), ())
        for ruleset, expected in (
            ("MMA", "takedowns"), ("MUAY_THAI", "elbow"), ("WT_TAEKWONDO", "electronic"),
        ):
            actions = " ".join(unobserved_actions(ruleset))
            self.assertIn(expected, actions, ruleset)
            card = score_fight([], ruleset, [1])
            self.assertTrue(card["coverage_note"], ruleset)
            self.assertIn("cannot see", card["coverage_note"])
        # A fully observed sport must not manufacture a caveat.
        self.assertEqual(score_fight([], "BOXING", [1])["coverage_note"], "")

    def test_every_sport_resolves_from_what_a_person_would_type(self):
        from core.scoring import RULESETS, SPORTS, normalize_ruleset

        for typed, expected in (
            ("boxing", "BOXING"), ("Muay Thai", "MUAY_THAI"), ("thai", "MUAY_THAI"),
            ("wtf", "WT_TAEKWONDO"), ("WTF Taekwondo", "WT_TAEKWONDO"), ("tkd", "WT_TAEKWONDO"),
            ("mma", "MMA"), ("k-1", "K1"),
        ):
            self.assertEqual(normalize_ruleset(typed), expected, typed)
        # Every ruleset belongs to exactly one sport, and every listed one exists.
        listed = [key for keys in SPORTS.values() for key in keys]
        self.assertEqual(sorted(listed), sorted(RULESETS))
        self.assertEqual(len(listed), len(set(listed)))

    def test_withdrawn_coach_plan_does_not_downgrade_existing_subscribers(self):
        """An unknown plan key falls back to free, which would strip paid access."""
        from core.payments import PLANS, plan_for_key

        self.assertNotIn("coach", PLANS)
        self.assertEqual(plan_for_key("coach")["label"], "Athlete Pro")
        self.assertEqual(plan_for_key("Coach")["label"], "Athlete Pro")
        # A genuinely unknown key must still land on free, not on the alias.
        self.assertEqual(plan_for_key("nonsense")["label"], "Starter")

    def test_complimentary_grant_outranks_the_stored_plan(self):
        from core.config import SETTINGS
        from core.payments import effective_plan_key

        granted = next(iter(SETTINGS.complimentary_plans), None)
        self.assertIsNotNone(granted, "expected at least one complimentary grant")
        self.assertEqual(effective_plan_key("free", None, granted), "gym")
        # Case and spacing in the stored email must not defeat the grant.
        self.assertEqual(effective_plan_key("free", None, f"  {granted.upper()} "), "gym")
        # Everyone else keeps exactly what they hold.
        self.assertEqual(effective_plan_key("free", None, "other@example.com"), "free")
        self.assertEqual(effective_plan_key("free", "athlete", "other@example.com"), "athlete")

    def test_pasted_secrets_survive_panel_wrapper_characters(self):
        """A wrapped secret must authenticate, not fail as a wrong password.

        Copying a token between a hosting panel and a .env file routinely picks
        up angle brackets left from a <placeholder> or defensive quotes. The
        wrapper is never part of the secret, but it produces a bare 401 that is
        indistinguishable from a genuinely wrong token.
        """
        import os

        from core.config import env_secret

        for raw in ("<tok3n>", '"tok3n"', "'tok3n'", "  <tok3n>  ", "<<tok3n>>", "tok3n"):
            os.environ["WIQ_TEST_SECRET"] = raw
            self.assertEqual(env_secret("WIQ_TEST_SECRET"), "tok3n", raw)
        # An empty or wrapper-only value must stay empty, never become a wrapper.
        for raw in ("", "<>", '""'):
            os.environ["WIQ_TEST_SECRET"] = raw
            self.assertEqual(env_secret("WIQ_TEST_SECRET"), "", raw)
        os.environ.pop("WIQ_TEST_SECRET", None)

    def test_small_sources_are_analysed_at_a_larger_inference_size(self):
        """Low-resolution footage carries very small fighters.

        Measured on a real 480x220 WAKO bout: at the standard 640 the detector
        found 1.6 people per frame and missed both athletes, resolving only the
        referee. At 1280 it found 5.9 for 31% more time.
        """
        from core.config import SETTINGS
        from core.pose_tracker import inference_size

        self.assertEqual(inference_size(480, 220), SETTINGS.low_resolution_imgsz)
        self.assertEqual(inference_size(854, 480), SETTINGS.low_resolution_imgsz)
        # Large sources are downscaled anyway and keep the tuned default.
        self.assertEqual(inference_size(1920, 1080), SETTINGS.default_imgsz)
        self.assertEqual(inference_size(1280, 720), SETTINGS.default_imgsz)
        # An unknown size must not silently upscale every analysis.
        self.assertEqual(inference_size(0, 0), SETTINGS.default_imgsz)

    def test_quality_controller_recovers_to_this_sources_own_size(self):
        """Recovery must not cap a low-resolution fight back at the global default."""
        from core.config import SETTINGS
        from core.pose_tracker import QualityController

        controller = QualityController(30.0, 480, 220)
        self.assertEqual(controller.base_imgsz, SETTINGS.low_resolution_imgsz)
        self.assertEqual(controller.imgsz, SETTINGS.low_resolution_imgsz)

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
        item = {"job_id": "fight1", "event_time": 12.4, "ruleset": "KICK_LIGHT", "predicted": {
            "fighter": "A", "technique": "left_low_kick", "target": "leg", "outcome": "clean", "family": "kick", "limb": "left_leg"
        }, "corrected": {
            "fighter": "A", "technique": "right_low_kick", "target": "leg", "outcome": "blocked", "family": "kick", "limb": "right_leg", "contact_time": 12.15,
        }}
        summary = accuracy_summary([item])
        self.assertEqual(summary["metrics"]["fighter_identity"]["accuracy"], 1.0)
        self.assertEqual(summary["metrics"]["limb_side"]["accuracy"], 0.0)
        self.assertEqual(summary["metrics"]["outcome"]["accuracy"], 0.0)
        self.assertEqual(summary["timing"]["samples"], 1)
        self.assertAlmostEqual(summary["timing"]["mean_absolute_error_seconds"], 0.25)
        self.assertEqual(summary["timing"]["within_250ms"], 1)

    def test_classification_metrics_expose_false_alarms_and_missed_classes(self):
        metrics = classification_metrics(
            ["none", "cross", "cross", "jab"],
            ["cross", "cross", "jab", "jab"],
            classes=ACTION_CLASSES,
        )

        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["false_alarms"], 1)
        self.assertEqual(metrics["missed_actions"], 0)
        self.assertAlmostEqual(metrics["per_class"]["cross"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["cross"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["cross"]["f1"], 0.5)

    def test_dataset_audit_rejects_bad_shapes_and_reports_real_class_coverage(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez_compressed(
                root / "fight-1__valid.npz",
                x=np.zeros((SETTINGS.action_window, 102), dtype=np.float32),
                y=np.int64(ACTION_CLASSES.index("cross")),
                fight_id=np.asarray("fight-1"),
            )
            np.savez_compressed(
                root / "fight-2__invalid.npz",
                x=np.zeros((SETTINGS.action_window, 99), dtype=np.float32),
                y=np.int64(ACTION_CLASSES.index("jab")),
                fight_id=np.asarray("fight-2"),
            )

            audit = audit_sequence_directory(root)

        self.assertEqual(audit["files"], 2)
        self.assertEqual(audit["valid_sequences"], 1)
        self.assertEqual(audit["invalid_sequences"], 1)
        self.assertEqual(audit["fights"], 1)
        self.assertEqual(audit["class_support"]["cross"], 1)
        self.assertEqual(audit["covered_classes"], 1)
        self.assertIn("102 features", audit["issues"][0]["reason"])

    def test_end_to_end_gate_requires_every_fight_fact_dimension(self):
        summary = accuracy_summary([])
        empty = assess_end_to_end_validation(end_to_end_metadata(summary))
        self.assertFalse(empty["passed"])
        self.assertTrue(any("fighter_identity_accuracy" in item for item in empty["failures"]))

        complete = assess_end_to_end_validation({
            "fights": 5,
            "action_labels": 100,
            "timing_samples": 50,
            "fighter_identity_accuracy": .95,
            "target_accuracy": .90,
            "outcome_accuracy": .85,
            "legality_accuracy": .95,
            "timing_mae_seconds": .25,
        })
        self.assertTrue(complete["passed"])


class BackupRuntimeTests(unittest.TestCase):
    def test_online_backup_writes_an_integrity_checked_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.sqlite3"
            with closing(sqlite3.connect(source)) as con:
                con.execute("CREATE TABLE proof(value TEXT NOT NULL)")
                con.execute("INSERT INTO proof(value) VALUES('warrioriq')")
                con.commit()
            destination = root / "backups"
            with patch.object(backup_runtime, "DB_PATH", source):
                result = backup_runtime.backup_database(destination)

            backup = Path(result["path"])
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            with closing(sqlite3.connect(backup)) as con:
                self.assertEqual(con.execute("SELECT value FROM proof").fetchone()[0], "warrioriq")
            self.assertEqual(manifest["integrity_check"], "ok")
            self.assertEqual(len(manifest["sha256"]), 64)
            self.assertFalse(manifest["contains_original_videos"])


if __name__ == "__main__":
    unittest.main()


class SportSpecificCoachingTests(unittest.TestCase):
    """A weapon mix is only meaningful against what the ruleset pays for."""

    @staticmethod
    def _strike(family, seconds, fighter="A"):
        from core.types import StrikeEvent

        return StrikeEvent(
            fighter=fighter, opponent="B", round_number=1, start_frame=0, peak_frame=1,
            end_frame=2, start_time=seconds, peak_time=seconds, end_time=seconds + .1,
            technique="cross", family=family, limb="lead", outcome="clean", landed=True,
        )

    def test_reward_shares_come_from_the_ruleset_not_a_second_opinion(self):
        """Scoring weights have one home; the coaching reads them, not a copy."""
        from core.scoring import RULESETS
        from core.sport_profiles import reward_shares

        for key in ("BOXING", "WT_TAEKWONDO", "MUAY_THAI", "MMA"):
            with self.subTest(ruleset=key):
                shares = reward_shares(key)
                self.assertAlmostEqual(sum(shares.values()), 1.0, places=6)
                # A family the ruleset forbids carries none of its value.
                profile = RULESETS[key]
                if not profile.allow_kick:
                    self.assertNotIn("kick", shares)
                if not profile.allow_knee:
                    self.assertNotIn("knee", shares)

        # Taekwondo is decided on kicks, and the split has to say so.
        self.assertGreater(reward_shares("WT_TAEKWONDO")["kick"], .7)
        # ITF pays the hands more than WT does, because head punches score.
        self.assertGreater(reward_shares("ITF_TAEKWONDO")["punch"],
                           reward_shares("WT_TAEKWONDO")["punch"])

    def test_a_boxer_in_a_kicking_sport_is_told_what_it_cost(self):
        from core.sport_profiles import build_sport_coaching

        events = [self._strike("punch", i * 3.0) for i in range(15)]
        events += [self._strike("kick", 52.0), self._strike("kick", 61.0)]
        metrics = {"A": {"attacks": {"families": {"punch": 15, "kick": 2}}}}
        result = build_sport_coaching("A", metrics, events, "WT_TAEKWONDO")

        self.assertFalse(result["insufficient_evidence"])
        kick = [o for o in result["observations"] if "kick" in o["title"]]
        self.assertTrue(kick, "under-used kick should be reported in taekwondo")
        self.assertEqual(kick[0]["tone"], "improvement")
        # The finding is linked to the footage that produced it.
        self.assertEqual(kick[0]["evidence_times"], [52.0, 61.0])

    def test_an_action_the_sport_does_not_score_is_named_on_one_occurrence(self):
        """A kick in boxing is not a style choice, so no threshold applies."""
        from core.sport_profiles import build_sport_coaching

        events = [self._strike("punch", i * 3.0) for i in range(15)] + [self._strike("kick", 52.0)]
        metrics = {"A": {"attacks": {"families": {"punch": 15, "kick": 1}}}}
        result = build_sport_coaching("A", metrics, events, "BOXING")

        illegal = [o for o in result["observations"] if "do not score" in o["title"]]
        self.assertTrue(illegal)
        self.assertIn("Kicks", illegal[0]["title"])
        self.assertEqual(illegal[0]["evidence_times"], [52.0])

    def test_too_few_attempts_reads_no_tendency_at_all(self):
        """Three attempts is not a weapon mix, and pretending otherwise lies."""
        from core.sport_profiles import build_sport_coaching

        events = [self._strike("punch", 1.0), self._strike("punch", 4.0), self._strike("punch", 9.0)]
        metrics = {"A": {"attacks": {"families": {"punch": 3}}}}
        result = build_sport_coaching("A", metrics, events, "WT_TAEKWONDO")

        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual([o for o in result["observations"] if "Under-using" in o["title"]], [])

    def test_each_sport_has_a_distinct_identity(self):
        """Five sports that look identical are one sport with five labels."""
        from core.scoring import SPORTS
        from core.sport_profiles import SPORT_IDENTITIES, family_plural, sport_identity

        self.assertEqual(set(SPORT_IDENTITIES), set(SPORTS))
        accents = [i.accent for i in SPORT_IDENTITIES.values()]
        marks = [i.mark for i in SPORT_IDENTITIES.values()]
        self.assertEqual(len(set(accents)), len(accents), "accents must be distinct")
        self.assertEqual(len(set(marks)), len(marks), "crests must be distinct")
        for sport, identity in SPORT_IDENTITIES.items():
            with self.subTest(sport=sport):
                self.assertEqual(sport_identity(sport).key, sport)
                self.assertTrue(identity.decided_by.endswith("."))
        # "Punchs" is not a word, and the report says it a dozen times a page.
        self.assertEqual(family_plural("punch"), "Punches")


class EngagementRangeTests(unittest.TestCase):
    """A strike is only an attempt at someone who could be hit.

    Found by running four real fights: the two tracked fighters were a median
    2.78 body lengths apart when an action fired, and only 18% of detections
    happened inside 1.5. One fight logged a cross while the opponent was a
    28x48px figure 9.5 body lengths away at the far edge of frame. Those cannot
    land, so every one of them counted as a miss and dragged the landed rate to
    2% - against 11% among strikes actually thrown in range.
    """

    @staticmethod
    def _event(attacker_box, opponent_box):
        from core.types import StrikeEvent

        event = StrikeEvent(
            fighter="A", opponent="B", round_number=1, start_frame=0, peak_frame=1,
            end_frame=2, start_time=1.0, peak_time=1.0, end_time=1.1,
            technique="cross", family="punch", limb="right_hand",
        )
        event.evidence["peak_attacker_box"] = attacker_box
        event.evidence["peak_opponent_box"] = opponent_box
        return event

    def test_a_strike_across_the_ring_is_not_an_attempt(self):
        from core.contact import opponent_separation, thrown_at_opponent

        # The real case: opponent 28x48px at the far edge, attacker at the near
        # edge. Separation is many body lengths; nothing can reach.
        far = self._event([20.0, 40.0, 48.0, 88.0], [371.5, 39.8, 399.2, 87.6])
        self.assertGreater(opponent_separation(far), 5.0)
        self.assertFalse(thrown_at_opponent(far))

    def test_a_strike_in_range_is_kept(self):
        from core.contact import opponent_separation, thrown_at_opponent

        close = self._event([189.0, 42.0, 216.0, 119.0], [232.0, 27.0, 267.0, 104.0])
        self.assertLess(opponent_separation(close), 1.5)
        self.assertTrue(thrown_at_opponent(close))

    def test_an_event_without_boxes_is_never_discarded(self):
        """Missing evidence is not evidence of absence: keep the action."""
        from core.contact import opponent_separation, thrown_at_opponent

        bare = self._event(None, None)
        self.assertIsNone(opponent_separation(bare))
        self.assertTrue(thrown_at_opponent(bare))

    def test_the_limit_is_generous_enough_for_a_real_kick(self):
        """A kick reaches roughly 1.5-2 body lengths; the gate must clear that."""
        from core.config import SETTINGS

        self.assertGreaterEqual(SETTINGS.max_engagement_body_lengths, 2.0)

    def test_no_correctly_picked_fight_is_ever_accused(self):
        """The three real fights, each seeded with its actual two fighters.

        Median separation varies hugely with camera distance - 0.76, 2.55 and
        1.90 body lengths - which is exactly why distance alone cannot decide
        this. None of them may be accused.
        """
        from core.contact import assess_selection

        for name, median, kept, discarded, landed in (
            ("2.mp4", 0.76, 153, 127, 24),
            ("3.mp4", 2.55, 68, 354, 2),
            ("0-02-05", 1.90, 48, 230, 4),
        ):
            with self.subTest(fight=name):
                verdict = assess_selection(
                    [median] * (kept + discarded), kept, discarded, landed
                )
                self.assertTrue(verdict["looks_like_a_fight"])
                self.assertIsNone(verdict["warning"])

    def test_a_fighter_paired_with_a_ringside_coach_is_caught(self):
        """The real failure that started this: on 2.mp4 the automatic pick took
        a crouching coach as fighter B. The pair sat 2.84 body lengths apart for
        the whole bout and not one of 175 actions landed."""
        from core.contact import assess_selection

        verdict = assess_selection([2.84] * 175, kept=12, discarded=163, landed=0)
        self.assertFalse(verdict["looks_like_a_fight"])
        self.assertEqual(verdict["verdict"], "selection_probably_wrong")
        self.assertIn("body lengths apart", verdict["warning"])

    def test_picking_the_referee_is_a_known_miss(self):
        """Documents a real limitation rather than pretending it is covered.

        The referee stands between the fighters, so strikes aimed at the
        opponent pass close to him and register as contact - on 3.mp4 the
        referee pick produced more apparent scoring than the real fighters.
        Distance plus nothing-landed cannot see that, and this guard stays
        silent rather than guess.
        """
        from core.contact import assess_selection

        verdict = assess_selection([3.25] * 342, kept=73, discarded=269, landed=7)
        self.assertTrue(verdict["looks_like_a_fight"])

    def test_a_quiet_fight_is_never_accused(self):
        """Too few actions is not evidence that the wrong people were picked."""
        from core.contact import assess_selection

        verdict = assess_selection([3.5] * 4, kept=1, discarded=3, landed=0)
        self.assertTrue(verdict["looks_like_a_fight"])
        self.assertEqual(verdict["verdict"], "not_enough_to_judge")


class FederationPointTableTests(unittest.TestCase):
    """Where a federation publishes a point value, WarriorIQ uses that value.

    ITF taekwondo scores a hand technique to any legal target 1, a kick to the
    mid section 2 and a kick to the high section 3. WT scores a punch to the
    trunk 1, a kick to the trunk 2 and a kick to the head 3, and forbids punches
    to the head entirely. Both were scored off one shared table that gave a body
    kick 1 and a head kick 2, which is neither federation's rules.
    """

    @staticmethod
    def _land(family, target, seconds, fighter="A"):
        from core.types import StrikeEvent

        return StrikeEvent(
            fighter=fighter, opponent="B" if fighter == "A" else "A", round_number=1,
            start_frame=0, peak_frame=1, end_frame=2, start_time=seconds,
            peak_time=seconds, end_time=seconds + .1, technique="x", family=family,
            limb="lead", outcome="clean", landed=True, target=target,
            confidence=.9, contact_confidence=.9,
        )

    def test_itf_scores_its_own_published_table(self):
        from core.scoring import score_fight

        card = score_fight([
            self._land("kick", "head", 1),    # 3
            self._land("kick", "body", 3),    # 2
            self._land("punch", "body", 5),   # 1
            self._land("punch", "head", 7),   # 1 - hands to the head are legal
        ], "ITF_TAEKWONDO", [1])
        self.assertEqual(card["rounds"][0]["fighter_A"], 7)

    def test_wt_scores_its_own_table_and_ignores_the_illegal_head_punch(self):
        from core.scoring import score_fight

        card = score_fight([
            self._land("kick", "head", 1),    # 3
            self._land("kick", "body", 3),    # 2
            self._land("punch", "body", 5),   # 1
            self._land("punch", "head", 7),   # 0 - illegal in WT
        ], "WT_TAEKWONDO", [1])
        self.assertEqual(card["rounds"][0]["fighter_A"], 6)

    def test_a_kick_to_the_leg_scores_nothing_in_either_federation(self):
        """Neither federation lists the legs as a scoring target."""
        from core.scoring import score_fight

        for ruleset in ("ITF_TAEKWONDO", "WT_TAEKWONDO"):
            with self.subTest(ruleset=ruleset):
                card = score_fight([self._land("kick", "leg", 1)], ruleset, [1])
                self.assertEqual(card["rounds"][0]["fighter_A"], 0)

    def test_sports_judged_round_by_round_keep_no_table(self):
        """A ten-point-must sport has no per-strike value to publish.

        Muay Thai, kickboxing, boxing and MMA are judged on the round, so a
        fixed table would be inventing a rule the federation does not have.
        """
        from core.scoring import RULESETS

        for key in ("K1", "BOXING", "MUAY_THAI", "MMA"):
            with self.subTest(ruleset=key):
                self.assertEqual(RULESETS[key].point_table, ())
