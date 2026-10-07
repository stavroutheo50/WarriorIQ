"""Is this a fight at all? Pose articulation, facing and engagement.

QA, 2026-10-07, /result/9138ca9b38a7: two photo cut-outs sliding left and
right got "Strong observation evidence", guard-drop timestamps, strengths and
a four-week plan. Every fight-presence window passed - both found, full
length, close, the boxes moving - because nothing asked whether the bodies
moved like bodies.
"""

from __future__ import annotations

import gzip
import json
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from core import fight_presence as fp
from core.report import not_a_fight, refresh_identity_integrity, share_card
from core.types import PersonObservation

TRACK = Path(__file__).resolve().parents[1] / "dataset" / "regression" / "kicklight_stavrou_ceschia" / "track.jsonl.gz"


def _rows():
    with gzip.open(TRACK, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _obs(entry):
    if not entry:
        return None
    return PersonObservation(track_id=entry.get("track_id"), box=np.asarray(entry["box"], dtype=np.float32),
                             confidence=float(entry["confidence"]),
                             keypoints=np.asarray(entry["keypoints"], dtype=np.float32),
                             keypoint_conf=np.asarray(entry["keypoint_conf"], dtype=np.float32))


def _verdict(frames) -> dict:
    """frames: (seconds, observation A, observation B)."""
    presence = fp.FightPresence(frames[0][0], frames[-1][0] + 0.1)
    for seconds, a, b in frames:
        presence.observe(seconds, a, b)
    return presence.summary()["plausibility"]


def _real_frames():
    return [(row["time_seconds"], _obs(row["fighter_A"]["observation"]), _obs(row["fighter_B"]["observation"]))
            for row in _rows()]


def _still_pose(side: str) -> dict:
    """One real, fully visible pose of this fighter, to cut out like a photo."""
    for row in _rows():
        entry = row[f"fighter_{side}"]["observation"]
        if entry and min(entry["keypoint_conf"][i] for i in (0, 5, 6, 11, 12, 9, 10, 15, 16)) >= 0.3:
            return entry
    raise AssertionError("no fully visible pose in the track")


def _slide(entry: dict, dx: float, rng) -> PersonObservation:
    """The same pose moved sideways, with the pose model's pixel jitter."""
    points = np.asarray(entry["keypoints"], dtype=np.float32) + np.array([dx, 0.0], dtype=np.float32)
    points = points + rng.normal(0.0, 1.0, points.shape).astype(np.float32)
    box = np.asarray(entry["box"], dtype=np.float32) + np.array([dx, 0, dx, 0], dtype=np.float32)
    return PersonObservation(track_id=1, box=box, confidence=0.9, keypoints=points,
                             keypoint_conf=np.asarray(entry["keypoint_conf"], dtype=np.float32))


def _cutouts(seconds: float = 20.0, fps: float = 6.0):
    rng = np.random.default_rng(7)
    a, b = _still_pose("A"), _still_pose("B")
    frames = []
    for index in range(int(seconds * fps)):
        t = index / fps
        frames.append((t, _slide(a, 40 * np.sin(t * 1.3), rng), _slide(b, -40 * np.sin(t * 1.1), rng)))
    return frames


class VerdictTests(unittest.TestCase):
    def test_a_real_fight_is_plausible(self):
        """The one real two-fighter track in the repository must never be
        called "not a fight": that would be a false claim."""
        verdict = _verdict(_real_frames())
        self.assertTrue(verdict["plausible"], verdict)
        self.assertGreater(min(verdict["signals"]["articulation"].values()), 2 * fp.MIN_ARTICULATION)

    def test_photo_cutouts_sliding_are_not_a_fight(self):
        frames = _cutouts()
        verdict = _verdict(frames)
        self.assertFalse(verdict["plausible"])
        self.assertIn("rigid", verdict["reasons"])
        self.assertIn("cut-outs", verdict["text"][0])
        # Every fight-presence window still passed: the old gate alone let it through.
        presence = fp.FightPresence(0.0, frames[-1][0] + 0.1)
        for seconds, a, b in frames:
            presence.observe(seconds, a, b)
        self.assertTrue(presence.summary()["sufficient"])

    def test_two_people_facing_away_from_each_other(self):
        frames = []
        for seconds, a, b in _real_frames():
            turned = []
            for obs in (a, b):
                if obs is None:
                    turned.append(None)
                    continue
                points = obs.keypoints.copy()
                shoulders_x = (points[5][0] + points[6][0]) / 2.0
                other = b if obs is a else a
                if other is not None:
                    other_x = (other.box[0] + other.box[2]) / 2.0
                    away = -1.0 if other_x >= shoulders_x else 1.0
                    for index in range(5):
                        points[index][0] = shoulders_x + away * (abs(points[index][0] - shoulders_x) + 8.0)
                turned.append(PersonObservation(track_id=obs.track_id, box=obs.box, confidence=obs.confidence,
                                                keypoints=points, keypoint_conf=obs.keypoint_conf))
            frames.append((seconds, turned[0], turned[1]))
        verdict = _verdict(frames)
        self.assertEqual(verdict["reasons"], ["facing_away"])

    def test_low_resolution_jitter_is_not_taken_for_movement(self):
        """A cut-out on an 18 px torso with 2 px of pose jitter per frame still
        reads as rigid: limbs are smoothed over time before they are compared."""
        rng = np.random.default_rng(3)
        a, b = _still_pose("A"), _still_pose("B")
        frames = []
        for index in range(120):
            t = index / 6.0
            pair = []
            for entry, dx in ((a, 40 * np.sin(t)), (b, -40 * np.sin(t))):
                obs = _slide(entry, dx, rng)
                obs.keypoints = (np.asarray(entry["keypoints"], dtype=np.float32) + np.array([dx, 0], dtype=np.float32)
                                 + rng.normal(0.0, 2.0, (17, 2)).astype(np.float32))
                pair.append(obs)
            frames.append((t, pair[0], pair[1]))
        self.assertIn("rigid", _verdict(frames)["reasons"])

    def test_too_little_to_judge_decides_nothing(self):
        verdict = _verdict(_cutouts(seconds=2.0))
        self.assertTrue(verdict["plausible"])
        self.assertIsNone(verdict["signals"]["articulation"]["A"])


def _report(verdict: dict) -> dict:
    metrics = {side: {"guard_index": 0.4, "balance_index": 0.7, "guard_definition": "hands_up_share/1",
                      "footwork_body_lengths_per_second": 0.8, "pose_coverage": 0.95,
                      "numbers": {"hands_up_share": 0.4, "longest_hands_down_seconds": 3.0, "off_balance_count": 1,
                                  "seen_seconds": 20.0},
                      "moments": {"guard_index": {"low": [3.0], "high": [9.0]}, "pressure_index": {"low": [1.0]}}}
               for side in ("A", "B")}
    return {
        "video": {"focus_fighter": "A", "analysis_target": "BOTH", "plausibility": verdict},
        "setup": {"ruleset": "K1"},
        "integrity": {"identity_evidence_trusted": True},
        "tracking": {"fighter_A_seed_source": "pose_detector", "fighter_B_seed_source": "pose_detector",
                     "initial_iou_A": 0.8, "initial_iou_B": 0.8, "fighter_A_coverage": 0.95,
                     "fighter_B_coverage": 0.95, "fighters_separable": True, "identity_confusions": 0},
        "metrics": {**metrics, "round_pose_coverage": {"1": {"A": 0.9, "B": 0.9}}},
        "coaching": {"A": {"strengths": [{"title": "Guard: ahead"}], "improvements": [], "drills": []}},
        "training_plan": {"A": [{"focus": "Block 1"}]},
        "key_moments": [{"t": 3.0}],
        "scorecard": {"available": True, "totals": {"A": 10, "B": 9}},
        "statistics": {"fighters": {"A": {}, "B": {}}},
    }


class ReportGateTests(unittest.TestCase):
    def setUp(self):
        self.rigid = _verdict(_cutouts())

    def test_coaching_plan_score_and_moments_are_withheld_with_the_reason(self):
        report = refresh_identity_integrity(_report(self.rigid))
        for side in ("A", "B"):
            coaching = report["coaching"][side]
            self.assertEqual((coaching["strengths"], coaching["improvements"], coaching["drills"]), ([], [], []))
            self.assertIn("not recognised as a fight", coaching["note"])
            self.assertIn("cut-outs", coaching["note"])
            self.assertEqual(report["training_plan"][side], [])
            self.assertEqual(report["training_progression"][side], [])
        self.assertEqual(report["key_moments"], [])
        self.assertFalse(report["scorecard"]["available"])
        self.assertEqual(report["integrity"]["coaching_evidence_mode"], "withheld_not_a_fight")
        self.assertIsNone(share_card(report))

    def test_measured_movement_stays_and_frozen_poses_lose_guard_and_balance(self):
        report = refresh_identity_integrity(_report(self.rigid))
        own = report["metrics"]["A"]
        self.assertEqual(own["footwork_body_lengths_per_second"], 0.8)
        self.assertIsNone(own["guard_index"])
        self.assertIsNone(own["balance_index"])
        self.assertIsNone(own["numbers"]["longest_hands_down_seconds"])
        self.assertNotIn("guard_index", own["moments"])
        self.assertIn("pressure_index", own["moments"])
        self.assertIn("poses never changed", own["guard_note"])
        # Only the fighters' own metrics; the per-round coverage table beside
        # them is not a fighter (it broke the report page when touched).
        self.assertEqual(report["metrics"]["round_pose_coverage"], {"1": {"A": 0.9, "B": 0.9}})

    def test_real_bodies_facing_away_keep_their_guard(self):
        verdict = {"plausible": False, "reasons": ["facing_away"], "text": [fp.PLAUSIBILITY_TEXT["facing_away"]]}
        report = refresh_identity_integrity(_report(verdict))
        self.assertEqual(report["metrics"]["A"]["guard_index"], 0.4)
        self.assertEqual(report["training_plan"]["A"], [])

    def test_a_plausible_fight_is_untouched(self):
        report = refresh_identity_integrity(_report(_verdict(_real_frames())))
        self.assertIsNone(not_a_fight(report))
        self.assertNotEqual(report["integrity"].get("coaching_evidence_mode"), "withheld_not_a_fight")

    def test_page_state_and_evidence_label(self):
        from app.main import _analysis_quality_summary, _numbers_state

        report = refresh_identity_integrity(_report(self.rigid))
        state = _numbers_state(report)
        self.assertEqual(state["state"], "movement_only")
        self.assertIn("coaching, the training plan", state["message"])
        label = _analysis_quality_summary(report)["label"]
        self.assertEqual(label, "Not recognised as a fight")


class SavedReportTests(unittest.TestCase):
    def test_old_report_gets_the_verdict_from_its_saved_track(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracking.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for seconds, a, b in _cutouts():
                    handle.write(json.dumps({"time_seconds": seconds, "fighter_A": {"observation": {
                        "box": a.box.tolist(), "confidence": 0.9, "keypoints": a.keypoints.tolist(),
                        "keypoint_conf": a.keypoint_conf.tolist()}}, "fighter_B": {"observation": {
                        "box": b.box.tolist(), "confidence": 0.9, "keypoints": b.keypoints.tolist(),
                        "keypoint_conf": b.keypoint_conf.tolist()}}}) + "\n")
            report = _report(None)
            report["video"].pop("plausibility")
            fp.attach_plausibility(report, path)
            self.assertFalse(report["video"]["plausibility"]["plausible"])
            refresh_identity_integrity(report)
            self.assertEqual(report["training_plan"]["A"], [])
            # A report with no saved track is left as it was.
            untouched = _report(None)
            untouched["video"].pop("plausibility")
            fp.attach_plausibility(untouched, Path(folder) / "missing.jsonl")
            self.assertNotIn("plausibility", untouched["video"])

    def test_the_repository_track_replays_as_a_fight(self):
        verdict = fp.plausibility_from_tracking(TRACK)
        self.assertTrue(verdict["plausible"])


if __name__ == "__main__":
    unittest.main()
