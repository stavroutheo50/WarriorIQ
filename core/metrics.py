from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from core.config import SETTINGS
from core.types import DefenseEvent, PersonObservation, StrikeEvent

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


def _p(kp, idx):
    if kp is None or len(kp) <= idx:
        return None
    p = np.asarray(kp[idx], dtype=np.float32)[:2]
    return p if p[0] > 0 and p[1] > 0 else None


def _center(box):
    if box is None:
        return None
    x1, y1, x2, y2 = map(float, box)
    return np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def _body(obs: PersonObservation | None) -> float:
    if obs is None:
        return 100.0
    kp = obs.keypoints
    ls, rs, lh, rh = (_p(kp, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
    if all(x is not None for x in (ls, rs, lh, rh)):
        torso = float(np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2))
        if torso > 5:
            return torso * 2.15
    return max(20.0, float(obs.box[3] - obs.box[1]))


class MetricsAccumulator:
    def __init__(self, width: int, height: int):
        self.width = max(1, width)
        self.height = max(1, height)
        self.frames = {"A": 0, "B": 0}
        self.visible = {"A": 0, "B": 0}
        self.last_center = {"A": None, "B": None}
        self.last_time = {"A": None, "B": None}
        self.movement = {"A": 0.0, "B": 0.0}
        self.pressure_samples = {"A": [], "B": []}
        # Where each fighter actually stood, so the middle of the fight can be
        # worked out from the fight rather than assumed to be the middle of the
        # camera frame.
        self.positions = {"A": [], "B": []}
        self.guard_samples = {"A": [], "B": []}
        self.balance_samples = {"A": [], "B": []}
        self.round_frames = defaultdict(lambda: {"A": 0, "B": 0})
        self.round_visible = defaultdict(lambda: {"A": 0, "B": 0})
        # When each fighter was and was not visible, so per-round evidence can
        # be rebuilt once the real round structure is known. See rebucket_rounds.
        self._presence: list[tuple[float, str, bool]] = []

    def rebucket_rounds(self, rounds) -> None:
        """Re-assign per-round pose evidence after the rounds are detected.

        The round number passed to update() comes from the schedule the fight
        started with, and the real structure is only known once the whole fight
        has been watched. Without rebuilding here, a fight detected as two
        rounds reported two rounds on the scorecard and one in the evidence
        table, which reads as a bug in the report rather than a difference in
        when two numbers were worked out.

        Samples inside a detected break belong to no round and are dropped:
        coverage during a break says nothing about how well the fight was seen.
        """
        from core.video import round_at_time

        self.round_frames = defaultdict(lambda: {"A": 0, "B": 0})
        self.round_visible = defaultdict(lambda: {"A": 0, "B": 0})
        for seconds, fighter, visible in self._presence:
            spec = round_at_time(rounds, seconds)
            if spec is None:
                continue
            self.round_frames[spec.number][fighter] += 1
            if visible:
                self.round_visible[spec.number][fighter] += 1

    def update(self, fighter: str, seconds: float, round_number: int | None, obs: PersonObservation | None, opponent: PersonObservation | None):
        self.frames[fighter] += 1
        self._presence.append((float(seconds), fighter, obs is not None))
        if round_number is not None:
            self.round_frames[round_number][fighter] += 1
        if obs is None:
            return
        self.visible[fighter] += 1
        if round_number is not None:
            self.round_visible[round_number][fighter] += 1

        center = _center(obs.box)
        opp_center = _center(opponent.box) if opponent is not None else None
        body = _body(obs)
        previous_center = self.last_center[fighter]
        previous_time = self.last_time[fighter]
        if previous_center is not None and previous_time is not None and seconds > previous_time:
            delta = center - previous_center
            self.movement[fighter] += float(np.linalg.norm(delta)) / body
            if opp_center is not None:
                to_opp = opp_center - previous_center
                nd, nv = float(np.linalg.norm(to_opp)), float(np.linalg.norm(delta))
                if nd > 1e-6 and nv > 1e-6:
                    self.pressure_samples[fighter].append(float(np.dot(delta / nv, to_opp / nd)))

        self.positions[fighter].append(np.asarray(center, dtype=np.float32))

        kp = obs.keypoints
        nose, lw, rw = _p(kp, NOSE), _p(kp, L_WRIST), _p(kp, R_WRIST)
        if nose is not None and (lw is not None or rw is not None):
            distances = []
            for wrist in (lw, rw):
                if wrist is not None:
                    distances.append(float(np.linalg.norm(wrist - nose)) / body)
            if distances:
                self.guard_samples[fighter].append(1.0 - min(1.0, min(distances) / 0.48))

        ls, rs, lh, rh, la, ra = (_p(kp, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE))
        if all(x is not None for x in (ls, rs, lh, rh)):
            shoulder_tilt = abs(float(ls[1] - rs[1])) / body
            hip_tilt = abs(float(lh[1] - rh[1])) / body
            base = float(np.linalg.norm(la - ra)) / body if la is not None and ra is not None else 0.22
            base_score = 1.0 - min(1.0, abs(base - 0.28) / 0.35)
            tilt_score = 1.0 - min(1.0, (shoulder_tilt + hip_tilt) / 0.35)
            self.balance_samples[fighter].append(0.55 * tilt_score + 0.45 * base_score)

        self.last_center[fighter] = center
        self.last_time[fighter] = seconds

    def _center_control(self, fighter: str) -> float | None:
        """How much of the fight this fighter spent in the middle of it.

        The middle is taken from where the two fighters actually went, not from
        the centre of the picture. A tournament camera covers a whole hall, so
        the mat is only part of the frame and frame-centre is not ring-centre:
        measured that way both fighters on one bout scored 0.769 and 0.771,
        which says nothing about either of them.

        Scored against the spread of the fight itself, so it means the same on
        a tight ring shot and a wide hall shot.
        """
        mine = self.positions.get(fighter) or []
        everyone = [point for side in ("A", "B") for point in (self.positions.get(side) or [])]
        if not mine or len(everyone) < 10:
            return None
        middle = np.mean(np.stack(everyone), axis=0)
        spread = float(np.mean(np.linalg.norm(np.stack(everyone) - middle, axis=1)))
        if spread < 1e-6:
            return None
        distances = np.linalg.norm(np.stack(mine) - middle, axis=1)
        # 1.0 is dead centre of the action; 0.0 is a full spread away from it.
        return float(np.mean(np.clip(1.0 - distances / (spread * 2.0), 0.0, 1.0)))

    @staticmethod
    def _attack_stats(fighter: str, events: list[StrikeEvent]) -> dict:
        own = [e for e in events if e.fighter == fighter]
        attempts = len(own)
        landed = sum(e.outcome in {"clean", "likely_landed"} for e in own)
        blocked = sum(e.outcome == "blocked" for e in own)
        checked = sum(e.outcome == "checked" for e in own)
        missed = sum(e.outcome == "missed" for e in own)
        uncertain = sum(e.outcome == "uncertain" for e in own)
        techniques = Counter(e.technique for e in own)
        landed_techniques = Counter(e.technique for e in own if e.outcome in {"clean", "likely_landed"})
        targets = Counter(e.target or "unknown" for e in own if e.outcome in {"clean", "likely_landed"})
        families = Counter(e.family for e in own)
        accuracy = landed / attempts if attempts else None
        return {
            "attempts": attempts,
            "landed": landed,
            "missed": missed,
            "blocked": blocked,
            "checked": checked,
            "uncertain": uncertain,
            "accuracy": accuracy,
            "techniques": dict(techniques),
            "landed_techniques": dict(landed_techniques),
            "targets_landed": dict(targets),
            "families": dict(families),
        }

    @staticmethod
    def _combination_stats(fighter: str, events: list[StrikeEvent]) -> dict:
        own = sorted([e for e in events if e.fighter == fighter], key=lambda e: e.peak_time)
        combos = []
        current = []
        for event in own:
            if not current or event.peak_time - current[-1].peak_time <= 1.20:
                current.append(event)
            else:
                if len(current) >= 2:
                    combos.append(current)
                current = [event]
        if len(current) >= 2:
            combos.append(current)
        return {
            "count": len(combos),
            "max_length": max((len(c) for c in combos), default=0),
            "evidence": [[round(e.peak_time, 3) for e in combo] for combo in combos[:20]],
        }

    @staticmethod
    def _counter_stats(fighter: str, events: list[StrikeEvent]) -> dict:
        own = [e for e in events if e.fighter == fighter]
        opp = [e for e in events if e.fighter != fighter]
        counters = []
        for event in own:
            recent = [o for o in opp if 0.0 <= event.start_time - o.end_time <= 1.0]
            if recent:
                counters.append(event)
        return {"count": len(counters), "times": [round(e.peak_time, 3) for e in counters[:30]]}

    def finalize(self, events: list[StrikeEvent], defenses: list[DefenseEvent], segment_duration: float) -> dict:
        result = {}
        for fighter in ("A", "B"):
            coverage = self.visible[fighter] / max(1, self.frames[fighter])
            attack = self._attack_stats(fighter, events)
            combos = self._combination_stats(fighter, events)
            counters = self._counter_stats(fighter, events)
            defense_counts = Counter(d.defense for d in defenses if d.fighter == fighter)
            total_defenses = sum(defense_counts.values())
            advanced_available = coverage >= SETTINGS.min_pose_coverage_for_metric

            if advanced_available:
                footwork = self.movement[fighter] / max(1.0, segment_duration)
                pressure = float(np.mean(self.pressure_samples[fighter])) if self.pressure_samples[fighter] else None
                ring_control = self._center_control(fighter)
                guard = float(np.mean(self.guard_samples[fighter])) if self.guard_samples[fighter] else None
                balance = float(np.mean(self.balance_samples[fighter])) if self.balance_samples[fighter] else None
            else:
                footwork = pressure = ring_control = guard = balance = None

            opponent_landed = [e for e in events if e.fighter != fighter and e.outcome in {"clean", "likely_landed"}]
            vulnerability_targets = Counter(e.target or "unknown" for e in opponent_landed)
            vulnerability_techniques = Counter(e.technique for e in opponent_landed)

            strongest = None
            if attack["landed_techniques"]:
                strongest = max(attack["landed_techniques"], key=attack["landed_techniques"].get)

            own_events = [e for e in events if e.fighter == fighter]
            opp_events = [e for e in events if e.fighter != fighter]
            attempts_per_minute = attack["attempts"] / max(1e-6, segment_duration / 60.0)
            combinations_per_minute = combos["count"] / max(1e-6, segment_duration / 60.0)
            defense_rate = total_defenses / len(opp_events) if opp_events else None
            landed_quality = [
                max(0.0, min(1.0, e.confidence * 0.55 + e.contact_confidence * 0.45))
                for e in own_events if e.outcome in {"clean", "likely_landed"}
            ]
            technique_execution = float(np.mean(landed_quality)) if landed_quality else None
            round_attempt_rates = []
            round_ids = sorted({e.round_number for e in own_events if e.round_number is not None})
            for rid in round_ids:
                count = sum(1 for e in own_events if e.round_number == rid)
                round_attempt_rates.append(float(count))
            consistency = None
            if len(round_attempt_rates) >= 2 and np.mean(round_attempt_rates) > 0:
                consistency = float(max(0.0, min(1.0, 1.0 - np.std(round_attempt_rates) / np.mean(round_attempt_rates))))

            result[fighter] = {
                "pose_coverage": coverage,
                "advanced_metrics_available": advanced_available,
                "attacks": attack,
                "combinations": combos,
                "counters": counters,
                "defenses": dict(defense_counts),
                "strongest_weapon": strongest,
                "vulnerability_targets": dict(vulnerability_targets),
                "vulnerability_techniques": dict(vulnerability_techniques),
                "footwork_body_lengths_per_second": footwork,
                "pressure_index": pressure,
                "ring_center_control": ring_control,
                "guard_index": guard,
                "balance_index": balance,
                "dashboard": {
                    "technique_execution_confidence": technique_execution,
                    "defense_response_rate": defense_rate,
                    "footwork_body_lengths_per_second": footwork,
                    "accuracy": attack["accuracy"],
                    "activity_attempts_per_minute": attempts_per_minute,
                    "ring_center_control": ring_control,
                    "combinations_per_minute": combinations_per_minute,
                    "round_to_round_consistency": consistency,
                    "availability_note": "Movement-derived values are unavailable when pose coverage is below the configured evidence threshold." if not advanced_available else None,
                },
                "availability": {
                    "accuracy": {
                        "available": attack["accuracy"] is not None,
                        "reason": None if attack["accuracy"] is not None else "No verified attack attempts were detected.",
                        "samples": attack["attempts"],
                    },
                    "movement": {
                        "available": advanced_available,
                        "reason": None if advanced_available else f"Pose coverage was {coverage*100:.1f}%; at least {SETTINGS.min_pose_coverage_for_metric*100:.0f}% is required.",
                        "samples": self.visible[fighter],
                    },
                    "guard": {
                        "available": guard is not None,
                        "reason": None if guard is not None else ("The wrist and head keypoints were not visible often enough." if advanced_available else f"Pose coverage was {coverage*100:.1f}%; at least {SETTINGS.min_pose_coverage_for_metric*100:.0f}% is required."),
                        "samples": len(self.guard_samples[fighter]),
                    },
                    "balance": {
                        "available": balance is not None,
                        "reason": None if balance is not None else ("The shoulders, hips and stance keypoints were not visible often enough." if advanced_available else f"Pose coverage was {coverage*100:.1f}%; at least {SETTINGS.min_pose_coverage_for_metric*100:.0f}% is required."),
                        "samples": len(self.balance_samples[fighter]),
                    },
                    "defense_response": {
                        "available": defense_rate is not None,
                        "reason": None if defense_rate is not None else "No verified opponent attack events were available for a response-rate denominator.",
                        "samples": len(opp_events),
                    },
                    "technique_execution": {
                        "available": technique_execution is not None,
                        "reason": None if technique_execution is not None else "No verified clean or likely-landed techniques were available.",
                        "samples": len(landed_quality),
                    },
                    "round_consistency": {
                        "available": consistency is not None,
                        "reason": None if consistency is not None else "At least two analyzed rounds containing verified attacks are required.",
                        "samples": len(round_attempt_rates),
                    },
                },
            }

        result["round_pose_coverage"] = {
            str(r): {
                f: self.round_visible[r][f] / max(1, self.round_frames[r][f])
                for f in ("A", "B")
            }
            for r in sorted(self.round_frames)
        }
        return result
