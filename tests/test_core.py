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
        # Labels are plainer now, and each fighter's weakest number is found by
        # ranking against the band these metrics sit in rather than comparing a
        # guard percentage against a balance percentage.
        self.assertIn("Guard 22%", coaching_a["improvements"][0]["title"])
        self.assertIn("Balance 44%", coaching_b["improvements"][0]["title"])
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
            # A real kick takes a foot off the floor; scoring now requires that
            # evidence, so a fixture standing for a real kick carries it.
            evidence={"foot_lift_torsos": 1.2},
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
        """The grant mechanism, not whichever address happens to be configured.

        This used to read the first entry of the live setting, which passed only
        because the owner's own email was hardcoded as the default - publishing a
        personal address in a public repository. Grants now come from the
        server's environment, so the test supplies its own.
        """
        from core.config import SETTINGS
        from core.payments import effective_plan_key

        # SETTINGS is frozen, so the mapping is edited in place and restored.
        granted = "granted@example.com"
        original = dict(SETTINGS.complimentary_plans)
        SETTINGS.complimentary_plans.clear()
        SETTINGS.complimentary_plans[granted] = "gym"
        try:
            self._assert_grant_behaviour(effective_plan_key, granted)
        finally:
            SETTINGS.complimentary_plans.clear()
            SETTINGS.complimentary_plans.update(original)

    def _assert_grant_behaviour(self, effective_plan_key, granted):
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
            evidence={"foot_lift_torsos": 1.2},
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

    def test_the_plan_targets_the_number_its_drill_was_chosen_for(self):
        """Goals used to be picked by matching words in the drill's name.

        There was no branch for pressure or footwork, so the two dimensions
        that separate fighters most got a generic "improve on your measured
        baseline" line. Each drill now carries its own metric.
        """
        from core.coaching import build_pose_coaching, build_training_plan

        a = dict(guard_index=0.125, balance_index=0.756, ring_center_control=0.505,
                 pressure_index=0.117, footwork_body_lengths_per_second=1.174)
        b = dict(guard_index=0.236, balance_index=0.715, ring_center_control=0.548,
                 pressure_index=-0.057, footwork_body_lengths_per_second=1.044)
        plan_b = build_training_plan(build_pose_coaching("B", b, a), "B", b)
        goals = " ".join(block["goal"] for block in plan_b)

        # B is behind on pressure and footwork, so both must get real targets.
        self.assertIn("pressure", goals)
        self.assertIn("body lengths a second", goals)
        self.assertNotIn("measured baseline", goals)
        # And each goal names what the opponent managed, as the benchmark.
        for block in plan_b:
            self.assertIn("Your opponent was at", block["goal"])

    def test_the_plan_states_each_number_once(self):
        """The goal appended the drill's rationale, which repeated the same
        figure in a second format: "from 22.0% toward 30.0% ... Measured at 22%"."""
        from core.coaching import build_pose_coaching, build_training_plan

        own = dict(guard_index=0.22, balance_index=0.81, ring_center_control=0.63)
        plan = build_training_plan(build_pose_coaching("A", own), "A", own)
        self.assertTrue(plan)
        self.assertNotIn("Measured at", plan[0]["goal"])

    def test_rounds_are_read_from_the_fight_not_typed_in(self):
        """Two rounds with a break between them are found without being told."""
        from core.round_detect import RoundDetector

        detector = RoundDetector()
        for second in range(0, 60):
            detector.observe(second, 1.0)        # engaged
        for second in range(60, 100):
            detector.observe(second, 6.0)        # corners, far apart
        for second in range(100, 160):
            detector.observe(second, 1.2)        # engaged again
        summary = detector.summary()
        self.assertEqual(summary["rounds_detected"], 2)
        self.assertAlmostEqual(summary["rounds"][0]["end_seconds"], 60.0, places=0)
        self.assertAlmostEqual(summary["rounds"][1]["start_seconds"], 100.0, places=0)

    def test_a_continuous_fight_is_not_carved_into_rounds(self):
        """Saying nothing is a real answer.

        A single continuous round and a video the detector could not read look
        identical from here, and both are served correctly by analysing the
        whole thing as one round.
        """
        from core.round_detect import RoundDetector

        detector = RoundDetector()
        for second in range(0, 180):
            detector.observe(second, 1.0)
        self.assertIsNone(detector.summary()["rounds_detected"])

    def test_resets_inside_a_round_are_not_mistaken_for_breaks(self):
        """Fighters break off constantly. A break is long, and it holds."""
        from core.round_detect import RoundDetector

        detector = RoundDetector()
        for second in range(0, 180):
            detector.observe(second, 1.0 if second % 7 else 5.0)
        self.assertIsNone(detector.summary()["rounds_detected"])

    def test_a_fight_nobody_could_be_located_in_claims_nothing(self):
        from core.round_detect import RoundDetector

        detector = RoundDetector()
        for second in range(0, 180):
            detector.observe(second, None)
        self.assertIsNone(detector.summary()["rounds_detected"])

    def test_the_worker_reloads_itself_without_needing_a_supervisor(self):
        """It restarts in place rather than exiting.

        Exiting assumed something was supervising the process. When nothing
        was, the worker vanished the moment the code changed and a queued fight
        sat with nobody to claim it - which is exactly what happened. Re-exec
        needs no supervisor and cannot leave a hole, so this checks the
        mechanism actually replaces the process on this platform.
        """
        import subprocess
        import sys
        import tempfile
        import textwrap
        from pathlib import Path

        import worker

        source = Path(worker.__file__).read_text(encoding="utf-8")
        self.assertIn("os.execv", source, "the worker must restart in place")
        self.assertNotIn("exiting so a restart picks it up", source)

        script = textwrap.dedent("""
            import os, sys
            generation = int(sys.argv[1]) if len(sys.argv) > 1 else 0
            if generation < 2:
                os.execv(sys.executable, [sys.executable, __file__, str(generation + 1)])
            print("chain completed")
        """)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "execv_probe.py"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(path)], capture_output=True, text=True, timeout=60,
            )
        self.assertIn("chain completed", result.stdout)

    def test_the_worker_notices_when_its_own_code_changes(self):
        """The analysis runs on the GPU box, not the web server.

        Deploying the website changes nothing about how a fight is analysed, so
        a worker started before a fix keeps running the old code silently. One
        ran five hours across fourteen commits: every analysis fix of the day
        sat on disk loaded by nothing, and the fights it was meant to fix came
        back unchanged.
        """
        import io as _io
        import time

        import worker

        before = worker._source_fingerprint()
        self.assertEqual(before, worker._source_fingerprint(), "must be stable")
        self.assertFalse(worker._code_changed_since(before))

        path = "core/metrics.py"
        original = _io.open(path, encoding="utf-8").read()
        try:
            _io.open(path, "w", encoding="utf-8").write(
                original + chr(10) + "# touched by a test" + chr(10)
            )
            time.sleep(0.05)
            self.assertTrue(worker._code_changed_since(before))
        finally:
            _io.open(path, "w", encoding="utf-8").write(original)

    def test_a_fighter_is_never_handed_to_someone_standing_still(self):
        """The reported failure: one fighter tracked, the other lost to a coach.

        Coaches, the referee and the officials all stand near the action and all
        look plausible for a frame. What none of them do is cover ground.
        Measured across every bout so far fighters run 24-65 body lengths a
        minute; a man at the mat edge managed 9.6.
        """
        import numpy as np

        from core.identity import IdentityManager
        from core.types import PersonObservation

        def person(track_id, x, y):
            return PersonObservation(
                track_id=track_id,
                box=np.asarray([x, y, x + 30, y + 80], dtype=np.float32),
                confidence=0.9,
            )

        manager = IdentityManager(person(1, 100, 100), person(2, 300, 100), 0, source_fps=30.0)
        # A bystander who has been standing in one spot for four seconds.
        for frame in range(0, 120, 3):
            manager._remember_positions([person(9, 200.0, 100.0)], frame)

        travel = manager._recent_travel(9, 30.0)
        self.assertIsNotNone(travel)
        self.assertLess(travel, 1.0, "a stationary person should register almost no travel")

    def test_a_track_seen_only_briefly_is_never_refused(self):
        """Too short a look is not evidence of standing still."""
        import numpy as np

        from core.identity import IdentityManager
        from core.types import PersonObservation

        def person(track_id, x):
            return PersonObservation(
                track_id=track_id,
                box=np.asarray([x, 100, x + 30, 180], dtype=np.float32),
                confidence=0.9,
            )

        manager = IdentityManager(person(1, 100), person(2, 300), 0, source_fps=30.0)
        for frame in range(0, 15, 3):
            manager._remember_positions([person(9, 200.0)], frame)
        self.assertIsNone(manager._recent_travel(9, 30.0))

    def test_the_veto_can_never_drop_the_fighter_already_held(self):
        """One-directional by design: it blocks a switch, never a hold.

        A fighter resting between exchanges must not be given away because they
        stopped moving for a few seconds.
        """
        from core.config import SETTINGS

        self.assertLess(
            SETTINGS.min_switch_travel_per_minute,
            SETTINGS.min_fighter_travel_per_minute,
            "refusing a switch should be more cautious than reporting a bystander",
        )

    def test_coaching_differs_between_the_two_fighters_in_a_bout(self):
        """Every fighter used to get the same advice, in every fight.

        Strength and weakness were picked by comparing raw values across
        different metrics. On real footage guard sits near 0.15 and balance
        near 0.70, so balance was always the strength and guard always a
        weakness - four fighters across two bouts all received "your strength
        is post-action balance, work on guard and ring-centre". Ranking is now
        against the opponent's same number.
        """
        from core.coaching import build_pose_coaching

        a = dict(guard_index=0.125, balance_index=0.756, ring_center_control=0.505,
                 pressure_index=0.117, footwork_body_lengths_per_second=1.174)
        b = dict(guard_index=0.236, balance_index=0.715, ring_center_control=0.548,
                 pressure_index=-0.057, footwork_body_lengths_per_second=1.044)
        coach_a = build_pose_coaching("A", a, b)
        coach_b = build_pose_coaching("B", b, a)

        self.assertNotEqual(coach_a["strengths"][0]["title"], coach_b["strengths"][0]["title"])
        # A walked forward and B gave ground, so that is A's strength.
        self.assertIn("Walking them down", coach_a["strengths"][0]["title"])
        # B kept a better guard than A, so that is B's.
        self.assertIn("Guard", coach_b["strengths"][0]["title"])

    def test_nobody_is_told_to_fix_something_they_are_winning(self):
        """A fighter ahead on everything but one thing gets one thing to fix."""
        from core.coaching import build_pose_coaching

        ahead = dict(guard_index=0.50, balance_index=0.80, ring_center_control=0.70,
                     pressure_index=0.30, footwork_body_lengths_per_second=1.50)
        behind = dict(guard_index=0.20, balance_index=0.60, ring_center_control=0.40,
                      pressure_index=0.40, footwork_body_lengths_per_second=1.00)
        coaching = build_pose_coaching("A", ahead, behind)
        for improvement in coaching["improvements"]:
            self.assertNotIn("better than your opponent", improvement["detail"])

    def test_a_fighter_ahead_everywhere_is_told_so(self):
        from core.coaching import build_pose_coaching

        ahead = dict(guard_index=0.50, balance_index=0.80, ring_center_control=0.70,
                     pressure_index=0.30, footwork_body_lengths_per_second=1.50)
        behind = dict(guard_index=0.20, balance_index=0.60, ring_center_control=0.40,
                      pressure_index=0.10, footwork_body_lengths_per_second=1.00)
        coaching = build_pose_coaching("A", ahead, behind)
        self.assertEqual(len(coaching["improvements"]), 1)
        self.assertIn("Nothing behind your opponent", coaching["improvements"][0]["title"])

    def test_centre_control_measures_the_fight_not_the_camera(self):
        """Two fighters standing together must not both score as in control.

        The old measure took distance from the middle of the picture. A
        tournament camera covers a whole hall, so the mat is a fraction of the
        frame and frame-centre is not ring-centre: on one real bout it gave the
        two fighters 0.769 and 0.771, which describes the camera rather than
        either of them.
        """
        from core.metrics import MetricsAccumulator

        metrics = MetricsAccumulator(width=1920, height=1080)
        # Both fighters work in the same small area, far from the frame centre.
        for step in range(40):
            metrics.positions["A"].append(np.asarray([300.0 + step, 900.0], dtype=np.float32))
            metrics.positions["B"].append(np.asarray([340.0 - step, 900.0], dtype=np.float32))
        a = metrics._center_control("A")
        b = metrics._center_control("B")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        # Judged against the spread of the fight, both are near its middle.
        for value in (a, b):
            self.assertGreater(value, 0.3)

    def test_a_fighter_worked_to_the_outside_scores_lower(self):
        from core.metrics import MetricsAccumulator

        metrics = MetricsAccumulator(width=1920, height=1080)
        for step in range(60):
            # A holds one spot; B circles far out around it.
            metrics.positions["A"].append(np.asarray([500.0, 500.0], dtype=np.float32))
            angle = step / 60.0 * 6.28318
            metrics.positions["B"].append(
                np.asarray([500.0 + 400.0 * np.cos(angle), 500.0 + 400.0 * np.sin(angle)],
                           dtype=np.float32)
            )
        self.assertGreater(metrics._center_control("A"), metrics._center_control("B"))

    def test_declared_rounds_never_cost_you_footage(self):
        """A nine-minute bout entered as 3 x 2 min lost three of its minutes.

        The schedule stopped at 3 x 120s and the tail was dropped in silence -
        no note in the report, no warning on the page. Round numbers describe
        the fight's shape; they must not decide how much of the upload is worth
        looking at.
        """
        from core.types import AnalysisRequest, VideoInfo
        from core.video import build_round_schedule

        for duration, rounds_entered, seconds_each in (
            (540.0, 3, 120.0),   # nine-minute fight, standard three-round entry
            (270.0, 2, 120.0),
            (900.0, 5, 120.0),
            (94.0, 1, 94.0),
        ):
            with self.subTest(duration=duration):
                req = AnalysisRequest(
                    video_path="x", fighter_a_box=[0, 0, 1, 1], fighter_b_box=[0, 0, 1, 1],
                    fight_type="competition", ruleset="K1", start_seconds=0.0,
                    round_count=rounds_entered, round_duration_seconds=seconds_each,
                    break_duration_seconds=0.0, selected_rounds=None, end_seconds=None,
                    analysis_target="BOTH", focus_fighter="A", job_id="j",
                )
                info = VideoInfo(path="x", width=480, height=220, fps=30.0,
                                 duration=duration, frame_count=int(duration * 30))
                schedule = build_round_schedule(req, info)
                covered = sum(r.end_seconds - r.start_seconds for r in schedule if r.selected)
                self.assertAlmostEqual(covered, duration, places=3)

    def test_asking_for_one_round_of_five_still_gets_exactly_that(self):
        """Covering the whole video must not override a deliberate choice."""
        from core.types import AnalysisRequest, VideoInfo
        from core.video import build_round_schedule

        req = AnalysisRequest(
            video_path="x", fighter_a_box=[0, 0, 1, 1], fighter_b_box=[0, 0, 1, 1],
            fight_type="competition", ruleset="K1", start_seconds=0.0, round_count=5,
            round_duration_seconds=120.0, break_duration_seconds=0.0,
            selected_rounds=[2], end_seconds=None, analysis_target="BOTH",
            focus_fighter="A", job_id="j",
        )
        info = VideoInfo(path="x", width=480, height=220, fps=30.0,
                         duration=600.0, frame_count=18000)
        schedule = build_round_schedule(req, info)
        selected = [r for r in schedule if r.selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].number, 2)
        self.assertAlmostEqual(selected[0].end_seconds - selected[0].start_seconds, 120.0, places=3)

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

    def test_watching_the_wrong_fighter_is_caught_by_the_followed_share(self):
        """Measured end to end by running analyze() twice on the same fight.

        Scores are per fighter, because the failure is asymmetric: seeding one
        real fighter and one coach still follows half the fight. Asking "was
        either box on this fighter?" scored a bad run 0.936 against 0.994 for a
        good one and separated nothing. Per fighter, the one nobody watched
        drops to 0.283.
        """
        from core.contact import assess_selection

        verdict = assess_selection(
            [1.2] * 200, kept=48, discarded=190, landed=4,
            travel_per_minute={"A": 26.0, "B": 24.0},
            observed_fighters={
                "centres": ((255.0, 74.0), (272.0, 65.0)),
                "travel_per_minute": (31.0, 26.0),
                "share_within_range": 1.0,
                "followed_share": (0.698, 0.283),
                "disagrees_with_selection": True,
            },
        )
        self.assertFalse(verdict["looks_like_a_fight"])
        self.assertEqual(verdict["verdict"], "another_pair_did_the_fighting")

    def test_correctly_seeded_fights_survive_the_followed_share(self):
        """Both real fights, measured. The lower of the two is 0.713."""
        from core.config import SETTINGS
        from core.contact import assess_selection
        from core.fighter_suggest import analysis_missed_the_fight

        for name, shares in (("2.mp4", (0.992, 0.995)), ("0-02-05", (0.713, 0.978))):
            with self.subTest(fight=name):
                self.assertFalse(analysis_missed_the_fight(shares))
                verdict = assess_selection(
                    [0.8] * 200, kept=150, discarded=120, landed=20,
                    travel_per_minute={"A": 31.0, "B": 26.0},
                    observed_fighters={
                        "centres": ((250.0, 70.0), (270.0, 68.0)),
                        "travel_per_minute": (31.0, 26.0),
                        "share_within_range": 1.0,
                        "followed_share": shares,
                        "disagrees_with_selection": analysis_missed_the_fight(shares),
                    },
                )
                self.assertTrue(verdict["looks_like_a_fight"])
        # Margin on both sides of the measured spread, not a fitted number.
        self.assertLess(SETTINGS.min_followed_share, 0.713)
        self.assertGreater(SETTINGS.min_followed_share, 0.283)

    def test_no_pair_found_means_no_accusation(self):
        from core.fighter_suggest import analysis_missed_the_fight

        self.assertFalse(analysis_missed_the_fight(None))

    def test_someone_standing_at_ringside_is_caught_even_when_close(self):
        """A live job (69511cc2c1cc) tracked two people at the edge of the mat.

        The separation test could not see it: the pair stood 0.48 body lengths
        apart, closer than the correctly picked fighters on 2.mp4. What gave it
        away was movement - "fighter A" covered 41 body lengths in 255 seconds,
        about 9.6 a minute, while every correctly tracked person measured so far
        covers 24 to 65. Someone who barely moves for four minutes is not
        fighting, however close they are standing to anyone else.
        """
        from core.contact import assess_selection

        verdict = assess_selection(
            [0.48] * 300, kept=24, discarded=50, landed=2,
            travel_per_minute={"A": 9.6, "B": 44.7},
        )
        self.assertFalse(verdict["looks_like_a_fight"])
        self.assertEqual(verdict["verdict"], "fighter_barely_moved")
        self.assertIn("barely moved", verdict["warning"])

    def test_real_fighters_are_never_called_stationary(self):
        """Every tracked pair measured on real footage, correct picks included."""
        from core.contact import assess_selection

        for name, a, b in (
            ("2.mp4 correct", 31.0, 26.0),
            ("3.mp4", 54.0, 34.0),
            ("0-02-05", 26.0, 28.0),
            ("slowest real pair seen", 25.0, 32.0),
        ):
            with self.subTest(run=name):
                verdict = assess_selection(
                    [0.8] * 200, kept=100, discarded=100, landed=10,
                    travel_per_minute={"A": a, "B": b},
                )
                self.assertTrue(verdict["looks_like_a_fight"])

    def test_a_clip_too_short_to_judge_movement_is_left_alone(self):
        """Under twenty seconds of tracking, _Travel reports nothing at all."""
        from core.contact import assess_selection

        verdict = assess_selection(
            [0.8] * 200, kept=100, discarded=100, landed=10,
            travel_per_minute={"A": None, "B": None},
        )
        self.assertTrue(verdict["looks_like_a_fight"])

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
            evidence={"foot_lift_torsos": 1.2},   # a real kick left the floor
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


def test_attempt_gate_sits_inside_the_producible_confidence_range():
    """A gate above what the confidence formula emits hides every attempt.

    This happened. The gate was a hard-coded 0.86 while the rule-based
    confidence ran from 0.30 to 0.94 with a median near 0.50, so six real
    fights produced 309 detections and five visible attempts - reported from
    the field as "zero attempts for both fighters".
    """
    from core.action import CONFIDENCE_CEILING, CONFIDENCE_FLOOR
    from core.analyzer import ATTEMPT_CONFIDENCE

    span = CONFIDENCE_CEILING - CONFIDENCE_FLOOR
    assert CONFIDENCE_FLOOR < ATTEMPT_CONFIDENCE < CONFIDENCE_CEILING
    # A strike carrying half the available evidence must survive the gate; a
    # threshold above that is rejecting ordinary technique, not noise.
    assert ATTEMPT_CONFIDENCE <= CONFIDENCE_FLOOR + 0.5 * span


def test_detected_rounds_can_carry_every_round_number_for_scoring():
    """A detected structure has to reach scoring, not just the report.

    RoundDetector found the rounds for a while before anything used them: the
    schedule still came from the setup page, so a ten-round bout entered as one
    span scored as round 1 throughout. The analyser now rebuilds the schedule
    from what was detected, and report.py derives its round_numbers from that,
    so this checks the shape that hand-off depends on.
    """
    from core.round_detect import RoundDetector
    from core.types import RoundSpec
    from core.video import round_at_time

    detector = RoundDetector()
    for second in range(0, 60):
        detector.observe(second, 1.0)          # round 1
    for second in range(60, 100):
        detector.observe(second, 6.0)          # break: apart and staying apart
    for second in range(100, 160):
        detector.observe(second, 1.2)          # round 2
    detected = detector.rounds()
    self_rounds = [RoundSpec(r.number, r.start_seconds, r.end_seconds, True) for r in detected]

    # Every round is selected, so per-round scoring sees the whole fight.
    assert [r.number for r in self_rounds if r.selected] == [1, 2]
    # A strike after the break belongs to round 2, not to round 1.
    assert round_at_time(self_rounds, 30.0).number == 1
    assert round_at_time(self_rounds, 140.0).number == 2
    # Nothing is scored inside the break itself.
    assert round_at_time(self_rounds, 80.0) is None


def test_per_round_evidence_follows_the_detected_rounds():
    """Two numbers worked out at different times must still agree.

    Round numbers reach MetricsAccumulator.update while the fight is being
    watched, from the schedule it started with. The real structure is only
    known afterwards. Before rebucketing, a fight detected as two rounds
    showed two rounds on the scorecard and one in the evidence table.
    """
    from core.metrics import MetricsAccumulator
    from core.types import RoundSpec

    metrics = MetricsAccumulator(640, 360)
    for second in range(0, 200):
        for fighter in ("A", "B"):
            # Round number 1 throughout, exactly as the loop would supply it.
            metrics.update(fighter, float(second), 1, None, None)

    metrics.rebucket_rounds([RoundSpec(1, 0.0, 60.0, True), RoundSpec(2, 100.0, 180.0, True)])
    assert sorted(metrics.round_frames) == [1, 2]
    assert metrics.round_frames[1]["A"] == 60         # 0-59s
    assert metrics.round_frames[2]["A"] == 80         # 100-179s
    # The break and the tail past the last round belong to no round at all.
    assert sum(v["A"] for v in metrics.round_frames.values()) == 140


def test_selection_frame_skips_the_static_opening(tmp_path):
    """The picker must not hand back frame 0 when nothing is happening there.

    Frame 0 of a fight is the round start: the referee stands between the two
    fighters with both arms out, the biggest and most confident person on the
    mat, and a selection box drawn there lands on the referee. Measured on real
    footage, seeding from frame 0 tracked the referee for a whole bout.
    """
    import cv2
    import numpy as np

    from core.video import get_video_info, pick_selection_frame

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    assert writer.isOpened(), "no mp4v encoder available to build the fixture"
    # 30 still frames, then movement in the middle of the mat. The still part
    # stays inside the window the picker searches, so this is a fair test of
    # it preferring action over the opening.
    for index in range(900):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        if index >= 30:
            x = 120 + (index % 20) * 3
            cv2.rectangle(frame, (x, 100), (x + 30, 170), (230, 230, 230), -1)
        writer.write(frame)
    writer.release()

    info = get_video_info(path)
    picked = pick_selection_frame(path, info)
    assert picked > 0, "picker returned the static opening frame"
    assert picked >= 30, f"picked frame {picked} is still inside the static opening"
    # And it stays inside the budget, because everything before it is dropped.
    assert picked / info.fps <= 20.0


def _kick_sample(shoulder_angle_deg, supporting_foot_y, frame=0):
    """A minimal pose: shoulders at a given angle, both ankles placed."""
    import numpy as np

    from core.action import Sample

    kp = np.zeros((17, 2), dtype=np.float32)
    half = 20.0
    radians = np.radians(shoulder_angle_deg)
    centre = np.array([100.0, 100.0], dtype=np.float32)
    offset = np.array([np.cos(radians), np.sin(radians)], dtype=np.float32) * half
    kp[5] = centre - offset          # left shoulder
    kp[6] = centre + offset          # right shoulder
    # Hips far enough below the shoulders that _body_length clears the
    # minimum height airtime needs; a small figure is deliberately unreadable.
    kp[11] = [90.0, 190.0]
    kp[12] = [110.0, 190.0]
    kp[15] = [95.0, supporting_foot_y]
    kp[16] = [105.0, supporting_foot_y]
    return Sample(
        frame=frame, time=frame / 30.0, round_number=1,
        box=np.array([80.0, 60.0, 120.0, 200.0], dtype=np.float32),
        keypoints=kp, conf=None, opponent_box=None, opponent_keypoints=None,
        opponent_conf=None, identity_confidence=1.0, opponent_identity_confidence=1.0,
    )


def test_a_turning_kick_is_told_apart_from_a_square_one():
    from core.action import _is_spinning

    # A kick thrown square: the torso opens a little and stops.
    assert not _is_spinning([0.0, 6.0, 13.0, 19.0, 25.0]), "a square kick was read as a spin"
    # A real turn passes through the angles on the way round.
    assert _is_spinning([0.0, 30.0, 62.0, 95.0, 128.0, 158.0]), "a turning kick was not detected"


def test_a_swapped_shoulder_label_is_not_a_spin():
    """The failure this replaced, and the reason spin is measured per frame.

    Pose estimators swap left and right shoulders when someone faces away from
    the camera, which is most of a taekwondo exchange, and that flips the
    shoulder vector 180 degrees in one frame. Comparing only the first and last
    frame of an action called 13 of 38 kicks in a real bout "turning".
    """
    from core.action import _is_spinning

    # Barely moving, then one frame where the labels swap, then barely moving.
    assert not _is_spinning([0.0, 4.0, 184.0, 188.0, 191.0])
    # And the same swap in the middle of a genuinely square kick.
    assert not _is_spinning([10.0, 14.0, 196.0, 200.0, 203.0, 206.0])


def test_a_jump_needs_the_supporting_foot_to_leave_the_floor():
    """One foot always leaves the floor on a kick; that is not a jump."""
    from core.action import _is_jumping

    grounded = (_kick_sample(0.0, 290.0), _kick_sample(0.0, 289.0, frame=6))
    airborne = (_kick_sample(0.0, 290.0), _kick_sample(0.0, 230.0, frame=6))
    assert not _is_jumping(*grounded)
    assert _is_jumping(*airborne)


def test_world_taekwondo_pays_more_for_a_turning_kick():
    """WT scores a turning head kick 5 and a square one 3."""
    from core.scoring import RULESETS, _table_points
    from core.types import StrikeEvent

    def kick(target, spinning):
        return StrikeEvent(
            fighter="A", opponent="B", round_number=1,
            start_frame=0, peak_frame=1, end_frame=2,
            start_time=0.0, peak_time=0.1, end_time=0.2,
            technique="right_round_kick", family="kick", limb="right_leg",
            confidence=0.9, outcome="clean", target=target,
            evidence={"spinning": spinning},
        )

    profile = RULESETS["WT_TAEKWONDO"]
    assert _table_points(kick("head", False), profile) == 3
    assert _table_points(kick("head", True), profile) == 5
    assert _table_points(kick("body", False), profile) == 2
    assert _table_points(kick("body", True), profile) == 4
    # A ruleset with no turning distinction is untouched by the spin flag.
    k1 = RULESETS["K1"]
    assert _table_points(kick("head", True), k1) == _table_points(kick("head", False), k1)


def test_airtime_is_not_claimed_on_a_distant_fighter():
    """Tournament footage is the case this has to get right.

    On 3.mp4 the athletes stand about eighty pixels tall, so the airborne
    threshold works out at nine pixels - ankle jitter. Every airborne call at
    that size was wrong, and one of them was a fighter walking past the referee
    between rounds. Too far away has to mean "cannot tell", not "jumped".
    """
    import numpy as np

    from core.action import _is_jumping, Sample

    def tiny(foot_y, frame=0):
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[5], kp[6] = [95.0, 100.0], [115.0, 100.0]
        kp[11], kp[12] = [98.0, 118.0], [112.0, 118.0]   # ~80px body length
        kp[15], kp[16] = [100.0, foot_y], [110.0, foot_y]
        return Sample(
            frame=frame, time=frame / 30.0, round_number=1,
            box=np.array([90.0, 90.0, 120.0, 170.0], dtype=np.float32),
            keypoints=kp, conf=None, opponent_box=None, opponent_keypoints=None,
            opponent_conf=None, identity_confidence=1.0, opponent_identity_confidence=1.0,
        )

    # A lift that would clear the ratio on a close-up fighter, at a distance.
    assert not _is_jumping(tiny(170.0), tiny(150.0, frame=6))


def test_both_confidence_gates_track_the_formula_that_feeds_them():
    """Two gates were fitted to a confidence formula that later changed.

    The attempt gate was 0.86 and the scoring gate 0.72, both set when the
    rule-based confidence saturated near 1.0. After it was rebuilt to run
    0.30-0.94 with a median near 0.50, the first hid all but 5 of 309
    detections and the second left a kickboxing bout with zero verified
    scoring actions - its four landed strikes peaked at 0.70.

    So both are now shares of the range the formula can actually produce, and
    this checks the two properties that matter: each sits inside the range,
    and scoring stays the stronger claim.
    """
    from core.action import CONFIDENCE_CEILING, CONFIDENCE_FLOOR
    from core.analyzer import ATTEMPT_CONFIDENCE
    from core.scoring import SCORING_CONFIDENCE

    span = CONFIDENCE_CEILING - CONFIDENCE_FLOOR
    for gate in (ATTEMPT_CONFIDENCE, SCORING_CONFIDENCE):
        assert CONFIDENCE_FLOOR < gate < CONFIDENCE_CEILING
    # Showing a score demands more than saying a strike was thrown.
    assert SCORING_CONFIDENCE > ATTEMPT_CONFIDENCE
    # And neither may drift back above what ordinary technique produces.
    assert SCORING_CONFIDENCE <= CONFIDENCE_FLOOR + 0.6 * span


def test_a_step_is_not_a_scored_kick():
    """Footwork was reaching the scorecard as landed kicks.

    On a taekwondo bout every one of 33 detected kicks had the kicking foot
    level with or below the standing foot, and nine were recorded as landing.
    The cause is upstream: across 487 fighter-frames the pose model never
    placed a foot more than 0.6 torsos above the other, with high ankle
    confidence, while the video plainly shows kicks. Until that changes, a
    kick with no lift behind it cannot be scored.
    """
    from core.scoring import MIN_SCORING_FOOT_LIFT, is_verified_scoring_event
    from core.types import StrikeEvent

    def kick(lift):
        return StrikeEvent(
            fighter="A", opponent="B", round_number=1, start_frame=0, peak_frame=1,
            end_frame=2, start_time=0.0, peak_time=0.1, end_time=0.2,
            technique="right_round_kick", family="kick", limb="right_leg",
            confidence=0.90, contact_confidence=0.90, outcome="clean", target="body",
            evidence={} if lift is None else {"foot_lift_torsos": lift},
        )

    assert not is_verified_scoring_event(kick(-0.01), "K1"), "a step scored as a kick"
    assert not is_verified_scoring_event(kick(0.14), "K1"), "a shuffle scored as a kick"
    assert not is_verified_scoring_event(kick(None), "K1"), "unmeasurable counted as proof"
    assert is_verified_scoring_event(kick(MIN_SCORING_FOOT_LIFT + 0.4), "K1")

    # Punches are untouched: a punch has no foot to lift.
    punch = StrikeEvent(
        fighter="A", opponent="B", round_number=1, start_frame=0, peak_frame=1,
        end_frame=2, start_time=0.0, peak_time=0.1, end_time=0.2, technique="cross",
        family="punch", limb="right_hand", confidence=0.90, contact_confidence=0.90,
        outcome="clean", target="body", evidence={},
    )
    assert is_verified_scoring_event(punch, "K1")


def test_close_readings_do_not_become_a_wipeout():
    """The scaling bug this replaced turned any difference into 100%/0%.

    Normalising two fighters against each other - subtracting whichever was
    lower - forces the loser to exactly zero. Two fighters who pressed forward
    0.02 and -0.03 came out as a 100%/0% aggression split, which reads on a
    scorecard as total dominance and means almost nothing.
    """
    from core.generalship import _share

    close = _share(0.02, -0.03, -1.0, 1.0)
    assert close is not None
    assert 0.49 < close[0] < 0.53, f"a near-tie scored {close}"
    assert abs(close[0] + close[1] - 1.0) < 1e-9

    # A real gap separates them clearly - but still not to a wipeout, because
    # pressing forward 0.6 against giving ground 0.4 is one-sided, not total.
    clear = _share(0.60, -0.40, -1.0, 1.0)
    assert 0.70 < clear[0] < 0.85, f"a one-sided round scored {clear}"


def test_a_round_nobody_controlled_is_called_even():
    """Refusing to name a winner is a real answer for a close round."""
    from core.generalship import judge_round

    level = {f: [0.01] * 80 for f in ("A", "B")}
    centre = {f: [0.5] * 80 for f in ("A", "B")}
    territory = {f: [0.0] * 4 for f in ("A", "B")}
    judged = judge_round(1, level, centre, territory)
    assert judged is not None
    assert judged.winner is None
    assert judged.score("A") == judged.score("B") == 10


def test_a_dominant_round_is_scored_ten_nine():
    from core.generalship import judge_round

    pressure = {"A": [0.55] * 80, "B": [-0.35] * 80}
    centre = {"A": [0.80] * 80, "B": [0.25] * 80}
    territory = {"A": [0.30] * 6, "B": [-0.30] * 6}
    judged = judge_round(2, pressure, centre, territory)
    assert judged.winner == "A"
    assert judged.score("A") == 10 and judged.score("B") == 9
    assert judged.aggression["A"] > judged.aggression["B"]


def test_movement_judging_is_withheld_when_tracking_is_poor():
    """Same rule as the striking scorecard: no score from a fight it did not watch."""
    from core.generalship import judge_fight

    class _Metrics:
        timed_pressure: list = []
        timed_positions: list = []

    card = judge_fight(_Metrics(), [], {"A": 0.60, "B": 0.95}, 0.85)
    assert card["available"] is False
    assert card["status"] == "insufficient_tracking"
    assert "60%" in card["reason"]


def test_progress_is_not_claimed_from_a_badly_tracked_fight():
    """"You got worse" is a serious thing to tell an athlete.

    A fight WarriorIQ could not watch properly is not evidence of anything, so
    it is skipped as a comparison point rather than compared against.
    """
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from core.squad import compare_with_previous

    def report(coverage, pressure):
        return {
            "video": {"focus_fighter": "A"},
            "tracking": {"fighter_A_coverage": coverage, "fighter_B_coverage": coverage},
            "metrics": {"A": {"pressure_index": pressure, "ring_center_control": 0.5,
                              "footwork_body_lengths_per_second": 1.0}},
        }

    with TemporaryDirectory() as tmp:
        paths = []
        for name, body in (("now", report(0.95, 0.30)), ("bad", report(0.40, 0.90))):
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(body), encoding="utf-8")
            paths.append(path)
        fights = [
            {"job_id": "now", "original_name": "now.mp4", "report_path": str(paths[0]),
             "ruleset": "K1", "created_at": "2026-09-03T10:00:00"},
            {"job_id": "bad", "original_name": "bad.mp4", "report_path": str(paths[1]),
             "ruleset": "K1", "created_at": "2026-09-02T10:00:00"},
        ]
        current = json.loads(paths[0].read_text(encoding="utf-8"))
        assert compare_with_previous(current, fights, "now") == {"available": False}
