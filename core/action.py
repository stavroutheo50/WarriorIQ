from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import acos, degrees

import numpy as np

from core.config import SETTINGS
from core.temporal_model import TemporalModel
from core.types import PersonObservation, StrikeEvent

# COCO 17-keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class Sample:
    frame: int
    time: float
    round_number: int | None
    box: np.ndarray
    keypoints: np.ndarray
    conf: np.ndarray | None
    opponent_box: np.ndarray | None
    opponent_keypoints: np.ndarray | None
    opponent_conf: np.ndarray | None
    identity_confidence: float
    opponent_identity_confidence: float


@dataclass
class ActiveLimb:
    fighter: str
    limb: str
    family: str
    start_sample: Sample
    peak_sample: Sample
    max_speed: float
    start_extension: float
    peak_extension: float
    frames_active: int = 1


def _point(kp: np.ndarray | None, idx: int) -> np.ndarray | None:
    if kp is None or len(kp) <= idx:
        return None
    p = np.asarray(kp[idx], dtype=np.float32)
    if p.size < 2 or float(p[0]) <= 0 or float(p[1]) <= 0:
        return None
    return p[:2]


def _body_length(kp: np.ndarray | None, box: np.ndarray | None) -> float:
    if kp is not None:
        ls, rs, lh, rh = (_point(kp, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP))
        if all(p is not None for p in (ls, rs, lh, rh)):
            shoulders = (ls + rs) / 2.0
            hips = (lh + rh) / 2.0
            torso = float(np.linalg.norm(shoulders - hips))
            if torso > 5:
                return torso * 2.15
    if box is not None:
        x1, y1, x2, y2 = map(float, box)
        return max(20.0, y2 - y1)
    return 100.0


def _center(box) -> np.ndarray | None:
    if box is None:
        return None
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def _angle(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float | None:
    if a is None or b is None or c is None:
        return None
    u, v = a - b, c - b
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return None
    cosine = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return degrees(acos(cosine))


def _toward_opponent_velocity(p_now, p_prev, opponent_box, dt: float, body: float) -> tuple[float, float]:
    if p_now is None or p_prev is None or dt <= 0 or body <= 0:
        return 0.0, 0.0
    velocity = (p_now - p_prev) / dt
    speed = float(np.linalg.norm(velocity) / body)
    opp = _center(opponent_box)
    if opp is None:
        return speed, 0.5
    direction = opp - p_prev
    nd = float(np.linalg.norm(direction))
    nv = float(np.linalg.norm(velocity))
    if nd < 1e-6 or nv < 1e-6:
        return speed, 0.0
    toward = float(np.dot(velocity / nv, direction / nd))
    return speed, toward


def _limb_indices(limb: str):
    if limb == "left_hand":
        return L_SHOULDER, L_ELBOW, L_WRIST
    if limb == "right_hand":
        return R_SHOULDER, R_ELBOW, R_WRIST
    if limb == "left_leg":
        return L_HIP, L_KNEE, L_ANKLE
    if limb == "right_leg":
        return R_HIP, R_KNEE, R_ANKLE
    if limb == "left_knee":
        return L_HIP, L_KNEE, L_ANKLE
    return R_HIP, R_KNEE, R_ANKLE


def _extension(sample: Sample, limb: str) -> float:
    root_i, joint_i, end_i = _limb_indices(limb)
    root, end = _point(sample.keypoints, root_i), _point(sample.keypoints, end_i)
    if root is None or end is None:
        return 0.0
    return float(np.linalg.norm(end - root) / _body_length(sample.keypoints, sample.box))


def _lead_hand(sample: Sample) -> str:
    """Estimate lead side from which shoulder is closer to opponent."""
    opp = _center(sample.opponent_box)
    ls, rs = _point(sample.keypoints, L_SHOULDER), _point(sample.keypoints, R_SHOULDER)
    if opp is None or ls is None or rs is None:
        return "left_hand"
    return "left_hand" if np.linalg.norm(ls - opp) <= np.linalg.norm(rs - opp) else "right_hand"


def _classify_punch(start: Sample, peak: Sample, limb: str) -> str:
    _, elbow_i, wrist_i = _limb_indices(limb)
    shoulder_i = L_SHOULDER if limb == "left_hand" else R_SHOULDER
    start_wrist, peak_wrist = _point(start.keypoints, wrist_i), _point(peak.keypoints, wrist_i)
    if start_wrist is None or peak_wrist is None:
        return "jab" if limb == _lead_hand(peak) else "cross"
    delta = peak_wrist - start_wrist
    body = _body_length(peak.keypoints, peak.box)
    upward = -float(delta[1]) / body
    horizontal = abs(float(delta[0])) / body
    elbow_angle = _angle(_point(peak.keypoints, shoulder_i), _point(peak.keypoints, elbow_i), peak_wrist)
    side = "left" if limb == "left_hand" else "right"
    if upward > 0.12 and upward > horizontal * 0.55:
        return f"{side}_uppercut"
    if elbow_angle is not None and elbow_angle < 128 and horizontal > 0.08:
        return f"{side}_hook"
    return "jab" if limb == _lead_hand(peak) else "cross"


def _classify_kick(start: Sample, peak: Sample, limb: str) -> str:
    hip_i, knee_i, ankle_i = _limb_indices(limb)
    start_ankle, peak_ankle = _point(start.keypoints, ankle_i), _point(peak.keypoints, ankle_i)
    body = _body_length(peak.keypoints, peak.box)
    side = "left" if limb.startswith("left") else "right"
    if start_ankle is None or peak_ankle is None:
        return f"{side}_round_kick"
    delta = peak_ankle - start_ankle
    horizontal = abs(float(delta[0])) / body
    vertical = abs(float(delta[1])) / body
    knee_angle = _angle(_point(peak.keypoints, hip_i), _point(peak.keypoints, knee_i), peak_ankle)
    # Straighter extension and mostly opponent-directed path -> front/push kick.
    if knee_angle is not None and knee_angle > 145 and horizontal >= vertical * 0.70:
        opp = _center(peak.opponent_box)
        forward_alignment = 0.0
        if opp is not None:
            direction = opp - start_ankle
            nd, nv = float(np.linalg.norm(direction)), float(np.linalg.norm(delta))
            if nd > 1e-6 and nv > 1e-6:
                forward_alignment = float(np.dot(delta / nv, direction / nd))
        extension = _extension(peak, limb)
        if forward_alignment > 0.82 and extension > 0.62:
            return f"{side}_push_kick"
        return f"{side}_front_kick"
    return f"{side}_round_kick"


def _classify_knee(peak: Sample, limb: str) -> str:
    return "left_knee" if limb.startswith("left") else "right_knee"


def _feature_vector(sample: Sample, previous: Sample | None) -> np.ndarray:
    """102-dim vector: 17*(x,y,conf) + 17*(vx,vy,speed)."""
    kp = sample.keypoints
    box = sample.box
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    body = _body_length(kp, box)
    values: list[float] = []
    velocities: list[float] = []
    dt = sample.time - previous.time if previous is not None else 0.0
    for i in range(17):
        p = _point(kp, i)
        conf = float(sample.conf[i]) if sample.conf is not None and i < len(sample.conf) else (1.0 if p is not None else 0.0)
        if p is None:
            values += [0.0, 0.0, 0.0]
            velocities += [0.0, 0.0, 0.0]
            continue
        values += [(float(p[0]) - x1) / bw, (float(p[1]) - y1) / bh, conf]
        prev_p = _point(previous.keypoints, i) if previous is not None else None
        if prev_p is None or dt <= 0:
            velocities += [0.0, 0.0, 0.0]
        else:
            v = (p - prev_p) / dt / body
            velocities += [float(v[0]), float(v[1]), float(np.linalg.norm(v))]
    return np.asarray(values + velocities, dtype=np.float32)


class FighterActionState:
    def __init__(self):
        self.samples: deque[Sample] = deque(maxlen=max(SETTINGS.action_window * 2, 24))
        self.features: deque[np.ndarray] = deque(maxlen=SETTINGS.action_window)
        self.active: dict[str, ActiveLimb] = {}
        self.last_event_time: dict[str, float] = {}
        self.last_family_event_time: dict[str, float] = {}


class ActionEngine:
    """Multi-frame strike candidate detector.

    A strike requires acceleration/extension toward the opponent, a peak, and
    retraction. No single frame can create an event. A trained WarriorIQ
    temporal checkpoint, when available, can override the heuristic technique
    label while the same temporal/contact safeguards remain in force.
    """

    LIMBS = (
        ("left_hand", "punch", L_WRIST),
        ("right_hand", "punch", R_WRIST),
        ("left_leg", "kick", L_ANKLE),
        ("right_leg", "kick", R_ANKLE),
        ("left_knee", "knee", L_KNEE),
        ("right_knee", "knee", R_KNEE),
    )

    def __init__(self):
        self.states = {"A": FighterActionState(), "B": FighterActionState()}
        self.temporal = TemporalModel()

    def _make_sample(
        self,
        frame: int,
        seconds: float,
        round_number: int | None,
        fighter: PersonObservation,
        opponent: PersonObservation | None,
        identity_confidence: float,
        opponent_identity_confidence: float,
    ) -> Sample:
        return Sample(
            frame=frame,
            time=seconds,
            round_number=round_number,
            box=fighter.box.copy(),
            keypoints=fighter.keypoints.copy(),
            conf=None if fighter.keypoint_conf is None else fighter.keypoint_conf.copy(),
            opponent_box=None if opponent is None else opponent.box.copy(),
            opponent_keypoints=None if opponent is None or opponent.keypoints is None else opponent.keypoints.copy(),
            opponent_conf=None if opponent is None or opponent.keypoint_conf is None else opponent.keypoint_conf.copy(),
            identity_confidence=float(max(0.0, min(1.0, identity_confidence))),
            opponent_identity_confidence=float(max(0.0, min(1.0, opponent_identity_confidence))),
        )

    def update(
        self,
        fighter_name: str,
        frame: int,
        seconds: float,
        round_number: int | None,
        fighter: PersonObservation | None,
        opponent: PersonObservation | None,
        identity_confidence: float = 1.0,
        opponent_identity_confidence: float = 1.0,
    ) -> list[StrikeEvent]:
        if fighter is None or fighter.keypoints is None or len(fighter.keypoints) < 17:
            return []
        state = self.states[fighter_name]
        sample = self._make_sample(
            frame,
            seconds,
            round_number,
            fighter,
            opponent,
            identity_confidence,
            opponent_identity_confidence,
        )
        previous = state.samples[-1] if state.samples else None
        state.samples.append(sample)
        state.features.append(_feature_vector(sample, previous))
        if previous is None:
            return []

        events: list[StrikeEvent] = []
        body = _body_length(sample.keypoints, sample.box)
        dt = max(1e-4, sample.time - previous.time)

        for limb, family, endpoint_idx in self.LIMBS:
            # Keep the dictionary key immutable. Pose-side confirmation may
            # change the event limb, but it must never pop/update another
            # active candidate and leave this one contaminating later events.
            active_key = limb
            p_now, p_prev = _point(sample.keypoints, endpoint_idx), _point(previous.keypoints, endpoint_idx)
            speed, toward = _toward_opponent_velocity(p_now, p_prev, sample.opponent_box, dt, body)
            ext = _extension(sample, limb)
            prev_ext = _extension(previous, limb)
            active = state.active.get(limb)

            # Knee strikes use knee speed, but require a bent leg so normal
            # walking/ring movement is not promoted to a strike.
            if family == "knee":
                hip_i, knee_i, ankle_i = _limb_indices(limb)
                knee_angle = _angle(_point(sample.keypoints, hip_i), _point(sample.keypoints, knee_i), _point(sample.keypoints, ankle_i))
                knee_shape_ok = knee_angle is not None and knee_angle < 125
            else:
                knee_shape_ok = True

            start_condition = (
                speed >= SETTINGS.min_strike_speed_body_lengths_per_s
                and toward >= 0.12
                and ext - prev_ext >= SETTINGS.min_extension_gain * 0.35
                and knee_shape_ok
            )

            if active is None:
                last_time = state.last_event_time.get(limb, -999.0)
                if start_condition and seconds - last_time >= SETTINGS.min_event_gap_seconds:
                    state.active[limb] = ActiveLimb(
                        fighter=fighter_name,
                        limb=limb,
                        family=family,
                        start_sample=previous,
                        peak_sample=sample,
                        max_speed=speed,
                        start_extension=prev_ext,
                        peak_extension=ext,
                    )
                continue

            active.frames_active += 1
            active.max_speed = max(active.max_speed, speed)
            if ext > active.peak_extension:
                active.peak_extension = ext
                active.peak_sample = sample

            extension_gain = active.peak_extension - active.start_extension
            retracting = ext < active.peak_extension - max(0.035, SETTINGS.min_extension_gain * 0.45)
            timed_out = active.frames_active >= max(6, SETTINGS.action_window)

            if (retracting and active.frames_active >= 3) or timed_out:
                start_sample = active.start_sample
                peak_sample = active.peak_sample

                valid = (
                    active.frames_active >= 3
                    and active.max_speed >= SETTINGS.min_strike_speed_body_lengths_per_s
                    and extension_gain >= SETTINGS.min_extension_gain
                )
                if family == "knee":
                    # Knees naturally have less endpoint extension.
                    valid = active.frames_active >= 3 and active.max_speed >= SETTINGS.min_strike_speed_body_lengths_per_s * 0.82

                if valid:
                    # Pose models occasionally swap left/right ankle labels for
                    # a frame during crossings. Confirm the named leg against
                    # which ankle actually travelled farther over the action.
                    event_limb = limb
                    if family in {"kick", "knee"}:
                        left_idx = L_KNEE if family == "knee" else L_ANKLE
                        right_idx = R_KNEE if family == "knee" else R_ANKLE
                        left_start, left_peak = _point(start_sample.keypoints, left_idx), _point(peak_sample.keypoints, left_idx)
                        right_start, right_peak = _point(start_sample.keypoints, right_idx), _point(peak_sample.keypoints, right_idx)
                        left_travel = 0.0 if left_start is None or left_peak is None else float(np.linalg.norm(left_peak - left_start))
                        right_travel = 0.0 if right_start is None or right_peak is None else float(np.linalg.norm(right_peak - right_start))
                        if max(left_travel, right_travel) > 1.25 * max(1.0, min(left_travel, right_travel)):
                            side = "left" if left_travel > right_travel else "right"
                            event_limb = f"{side}_{'knee' if family == 'knee' else 'leg'}"

                    if family == "punch":
                        technique = _classify_punch(start_sample, peak_sample, event_limb)
                    elif family == "kick":
                        technique = _classify_kick(start_sample, peak_sample, event_limb)
                    else:
                        technique = _classify_knee(peak_sample, event_limb)

                    model_source = "temporal_rules"
                    confidence = min(0.94, 0.45 + 0.22 * min(2.0, active.max_speed) + 0.35 * min(0.45, extension_gain))

                    # Optional trained model gets a vote only when a complete
                    # sequence is available and its confidence is strong.
                    if self.temporal.available and len(state.features) >= SETTINGS.action_window:
                        sequence = np.stack(list(state.features)[-SETTINGS.action_window :], axis=0)
                        prediction = self.temporal.predict(sequence)
                        if prediction is not None:
                            label, model_conf = prediction
                            # Only allow same-family overrides; this stops a
                            # noisy model from turning a kick candidate into a punch.
                            same_family = (
                                family == "punch" and any(x in label for x in ("jab", "cross", "hook", "uppercut", "backfist"))
                            ) or (family == "kick" and "kick" in label) or (family == "knee" and "knee" in label)
                            event_side = "left" if event_limb.startswith("left") else "right"
                            label_side = "left" if label.startswith("left_") else "right" if label.startswith("right_") else None
                            side_consistent = label_side is None or label_side == event_side
                            if same_family and side_consistent:
                                technique = label
                                confidence = max(confidence, model_conf)
                                model_source = "warrioriq_temporal_model"

                    opponent_name = "B" if fighter_name == "A" else "A"
                    # Keep a compact temporal contact trajectory separate from
                    # technique classification. Contact/target logic later
                    # evaluates several samples around the strike instead of
                    # declaring contact from one peak frame.
                    contact_samples = []
                    for s in state.samples:
                        if start_sample.frame <= s.frame <= sample.frame:
                            contact_samples.append({
                                "frame": int(s.frame),
                                "time": float(s.time),
                                "attacker_keypoints": s.keypoints.tolist(),
                                "attacker_conf": None if s.conf is None else s.conf.tolist(),
                                "attacker_identity_confidence": float(s.identity_confidence),
                                "opponent_box": None if s.opponent_box is None else s.opponent_box.tolist(),
                                "opponent_keypoints": None if s.opponent_keypoints is None else s.opponent_keypoints.tolist(),
                                "opponent_conf": None if s.opponent_conf is None else s.opponent_conf.tolist(),
                                "opponent_identity_confidence": float(s.opponent_identity_confidence),
                            })

                    event = StrikeEvent(
                        fighter=fighter_name,
                        opponent=opponent_name,
                        round_number=peak_sample.round_number,
                        start_frame=start_sample.frame,
                        peak_frame=peak_sample.frame,
                        end_frame=sample.frame,
                        start_time=start_sample.time,
                        peak_time=peak_sample.time,
                        end_time=sample.time,
                        technique=technique,
                        family=family,
                        limb=event_limb,
                        confidence=float(max(0.0, min(1.0, confidence))),
                        model_source=model_source,
                        evidence={
                            "peak_attacker_box": peak_sample.box.tolist(),
                            "peak_attacker_keypoints": peak_sample.keypoints.tolist(),
                            "peak_attacker_conf": None if peak_sample.conf is None else peak_sample.conf.tolist(),
                            "peak_opponent_box": None if peak_sample.opponent_box is None else peak_sample.opponent_box.tolist(),
                            "peak_opponent_keypoints": None if peak_sample.opponent_keypoints is None else peak_sample.opponent_keypoints.tolist(),
                            "peak_opponent_conf": None if peak_sample.opponent_conf is None else peak_sample.opponent_conf.tolist(),
                            "contact_samples": contact_samples,
                            "max_speed_body_lengths_per_s": float(active.max_speed),
                            "extension_gain": float(extension_gain),
                        },
                    )
                    events.append(event)
                    state.last_event_time[active_key] = seconds

                state.active.pop(active_key, None)

        # A single technique can trigger several limb candidates as joints
        # retract on adjacent frames. Keep only the strongest same-family
        # candidate inside one physical-action window.
        accepted: list[StrikeEvent] = []
        for event in sorted(events, key=lambda item: item.confidence, reverse=True):
            last_family = state.last_family_event_time.get(event.family, -999.0)
            if event.peak_time - last_family < 0.30:
                continue
            accepted.append(event)
            state.last_family_event_time[event.family] = event.peak_time
        return sorted(accepted, key=lambda item: item.peak_time)
