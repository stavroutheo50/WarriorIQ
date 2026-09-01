from __future__ import annotations

import numpy as np

from core.action import (
    L_ANKLE,
    L_KNEE,
    L_WRIST,
    R_ANKLE,
    R_KNEE,
    R_WRIST,
)
from core.config import SETTINGS
from core.types import StrikeEvent

# COCO points
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def _valid(kp: np.ndarray | None, idx: int):
    if kp is None or len(kp) <= idx:
        return None
    p = np.asarray(kp[idx], dtype=np.float32)[:2]
    if p[0] <= 0 or p[1] <= 0:
        return None
    return p


def _mean_points(kp, indices):
    points = [_valid(kp, i) for i in indices]
    points = [p for p in points if p is not None]
    return np.mean(points, axis=0) if points else None


def _body_length(kp: np.ndarray | None, box) -> float:
    if kp is not None:
        shoulders = _mean_points(kp, [L_SHOULDER, R_SHOULDER])
        hips = _mean_points(kp, [L_HIP, R_HIP])
        if shoulders is not None and hips is not None:
            torso = float(np.linalg.norm(shoulders - hips))
            if torso > 5:
                return torso * 2.15
    if box is not None:
        x1, y1, x2, y2 = map(float, box)
        return max(20.0, y2 - y1)
    return 100.0


def _target_points(kp: np.ndarray | None) -> dict[str, list[np.ndarray]]:
    if kp is None:
        return {"head": [], "body": [], "leg": []}
    head = [_valid(kp, i) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)]
    body = [_valid(kp, i) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    legs = [_valid(kp, i) for i in (13, 14, 15, 16)]
    return {
        "head": [p for p in head if p is not None],
        "body": [p for p in body if p is not None],
        "leg": [p for p in legs if p is not None],
    }


def _distance_to_group(point: np.ndarray, points: list[np.ndarray]) -> float:
    if not points:
        return 9999.0
    return min(float(np.linalg.norm(point - p)) for p in points)


def _endpoint_index(event: StrikeEvent) -> int:
    if event.limb == "left_hand":
        return L_WRIST
    if event.limb == "right_hand":
        return R_WRIST
    if event.limb == "left_leg":
        return L_ANKLE
    if event.limb == "right_leg":
        return R_ANKLE
    if event.limb == "left_knee":
        return L_KNEE
    return R_KNEE


def _block_distance(endpoint: np.ndarray, defender_kp: np.ndarray | None, body: float) -> float:
    wrists = [_valid(defender_kp, L_WRIST), _valid(defender_kp, R_WRIST)]
    wrists = [p for p in wrists if p is not None]
    if not wrists:
        return 999.0
    return min(float(np.linalg.norm(endpoint - p)) / body for p in wrists)


def _is_leg_check(defender_kp: np.ndarray | None, body: float) -> bool:
    if defender_kp is None:
        return False
    lk, rk = _valid(defender_kp, 13), _valid(defender_kp, 14)
    la, ra = _valid(defender_kp, L_ANKLE), _valid(defender_kp, R_ANKLE)
    if any(p is None for p in (lk, rk, la, ra)):
        return False
    threshold = max(4.0, body * 0.08)
    left_raised = float(rk[1] - lk[1]) > threshold and float(ra[1] - la[1]) > threshold
    right_raised = float(lk[1] - rk[1]) > threshold and float(la[1] - ra[1]) > threshold
    return bool(left_raised or right_raised)


def _evaluate_snapshot(event: StrikeEvent, attacker_raw, defender_raw, defender_box):
    if attacker_raw is None or defender_raw is None:
        return None
    attacker_kp = np.asarray(attacker_raw, dtype=np.float32)
    defender_kp = np.asarray(defender_raw, dtype=np.float32)
    endpoint = _valid(attacker_kp, _endpoint_index(event))
    if endpoint is None:
        return None
    body = _body_length(defender_kp, defender_box)
    groups = _target_points(defender_kp)
    distances = {name: _distance_to_group(endpoint, points) / body for name, points in groups.items()}
    target = min(distances, key=distances.get)
    return {
        "target": target,
        "distances": distances,
        "endpoint": endpoint,
        "defender_kp": defender_kp,
        "body": body,
        "defender_box": defender_box,
        "block_distance": _block_distance(endpoint, defender_kp, body),
    }


def opponent_separation(event: StrikeEvent) -> float | None:
    """How far apart the two fighters were, in the opponent's body lengths.

    A strike is only an attempt at someone if they were reachable. Separation
    is measured centre to centre from the boxes already stored on the event, so
    this needs no extra inference.
    """
    evidence = event.evidence or {}
    attacker, opponent = evidence.get("peak_attacker_box"), evidence.get("peak_opponent_box")
    if not attacker or not opponent:
        return None
    ax = (float(attacker[0]) + float(attacker[2])) / 2.0
    ay = (float(attacker[1]) + float(attacker[3])) / 2.0
    ox = (float(opponent[0]) + float(opponent[2])) / 2.0
    oy = (float(opponent[1]) + float(opponent[3])) / 2.0
    body = max(20.0, float(opponent[3]) - float(opponent[1]))
    return float(np.hypot(ax - ox, ay - oy)) / body


def thrown_at_opponent(event: StrikeEvent) -> bool:
    """False when this was not an attempt at the opponent.

    Two separate ways an action fails to be one, and both are needed. The
    fighters can be far enough apart that nothing could reach; or they can be
    close while this particular action finishes nowhere near a legal target,
    which is what stepping, checking and feinting look like to the detector.
    """
    separation = opponent_separation(event)
    if separation is not None and separation > SETTINGS.max_engagement_body_lengths:
        return False
    reach = (event.evidence or {}).get("contact_distance_body_lengths")
    if isinstance(reach, (int, float)):
        return float(reach) <= SETTINGS.max_strike_reach_body_lengths
    # No measurement is not evidence of a miss; keep the action.
    return True


def assess_selection(
    separations: list[float], kept: int, discarded: int, landed: int
) -> dict:
    """Were the two people picked actually the two fighters?

    Two fighters trade. A referee, a coach or someone in the crowd does not, so
    whoever was picked by mistake stays at a distance all bout and nothing ever
    lands on them. Both halves are needed. Distance alone does not separate the
    cases: measured across three fights, a correctly picked pair ranged up to
    2.55 body lengths apart while a mistaken pair started at 2.71, which is far
    too thin a margin to accuse anyone on.

    So this fires only on the unmistakable version - far apart *and* nothing
    landed at all - and says nothing otherwise. It will miss real mistakes.
    That is the intended trade: telling someone their real fight was analysed
    on the wrong people is worse than staying quiet.
    """
    total = kept + discarded
    result = {
        "actions_observed": total,
        "actions_in_range": kept,
        "landed": landed,
        "median_separation_body_lengths": None,
        "looks_like_a_fight": True,
        "warning": None,
    }
    if not separations or total < SETTINGS.min_actions_to_judge_selection:
        result["verdict"] = "not_enough_to_judge"
        return result

    median = float(np.median(separations))
    result["median_separation_body_lengths"] = round(median, 2)
    too_far = median > SETTINGS.max_median_separation_body_lengths
    if not (too_far and landed == 0):
        result["verdict"] = "consistent_with_a_fight"
        return result

    result["looks_like_a_fight"] = False
    result["verdict"] = "selection_probably_wrong"
    result["warning"] = (
        f"Nothing landed in {total} actions, and these two stayed about "
        f"{median:.1f} body lengths apart the whole video. One of them is "
        "probably not a fighter - the referee and the coaches stand close to "
        "the action and are easy to pick by mistake."
    )
    return result


def classify_contact(event: StrikeEvent) -> StrikeEvent:
    """Classify strike outcome from a short temporal contact trajectory.

    Technique recognition and contact remain separate. A clean contact requires
    temporal support near the defender target, rather than a single-frame
    proximity coincidence. When evidence is weaker, WarriorIQ downgrades to
    likely_landed or uncertain instead of inventing certainty.
    """
    evidence = event.evidence
    evaluations = []

    for sample in evidence.get("contact_samples") or []:
        item = _evaluate_snapshot(
            event,
            sample.get("attacker_keypoints"),
            sample.get("opponent_keypoints"),
            sample.get("opponent_box"),
        )
        if item is not None:
            item["frame"] = sample.get("frame")
            item["time"] = sample.get("time")
            item["attacker_conf"] = sample.get("attacker_conf")
            item["attacker_identity_confidence"] = float(sample.get("attacker_identity_confidence", 1.0))
            item["opponent_identity_confidence"] = float(sample.get("opponent_identity_confidence", 1.0))
            evaluations.append(item)

    # Backward-compatible fallback for reports/events created without the
    # trajectory field.
    if not evaluations:
        item = _evaluate_snapshot(
            event,
            evidence.get("peak_attacker_keypoints"),
            evidence.get("peak_opponent_keypoints"),
            evidence.get("peak_opponent_box"),
        )
        if item is not None:
            item["frame"] = event.peak_frame
            item["time"] = event.peak_time
            item["attacker_conf"] = evidence.get("peak_attacker_conf")
            item["attacker_identity_confidence"] = 1.0
            item["opponent_identity_confidence"] = 1.0
            evaluations.append(item)

    if not evaluations:
        event.outcome = "uncertain"
        event.landed = False
        event.contact_confidence = 0.0
        return event

    # Pick the target whose normalized distance reaches the smallest value over
    # the trajectory, then measure temporal support for that same target.
    best_target = None
    best_distance = 9999.0
    best_eval = None
    for item in evaluations:
        for target, distance in item["distances"].items():
            if distance < best_distance:
                best_target = target
                best_distance = float(distance)
                best_eval = item

    if best_target is None or best_eval is None:
        event.outcome = "uncertain"
        event.landed = False
        event.contact_confidence = 0.0
        return event

    # Target height is more stable from the full defender box than from one
    # noisy wrist/ankle keypoint. The closest-keypoint distance still decides
    # whether contact occurred; this zone decides high/body/low terminology.
    defender_box = best_eval.get("defender_box")
    if defender_box is not None:
        _, y1, _, y2 = map(float, defender_box)
        height = max(1.0, y2 - y1)
        vertical = (float(best_eval["endpoint"][1]) - y1) / height
        if vertical <= 0.28:
            best_target = "head"
        elif vertical >= 0.56:
            best_target = "leg"
        else:
            best_target = "body"

    # Once the target zone is known, bind every later decision to the sample
    # closest to that same zone. Previously the target could be changed after
    # selecting a frame, leaving the timestamp, distance and block decision
    # sourced from different instants.
    best_eval = min(evaluations, key=lambda item: float(item["distances"].get(best_target, 9999.0)))
    best_distance = float(best_eval["distances"].get(best_target, 9999.0))
    target_distances = [float(item["distances"].get(best_target, 9999.0)) for item in evaluations]
    exact_hits = sum(d <= SETTINGS.contact_threshold_body_lengths for d in target_distances)
    support_hits = sum(d <= SETTINGS.likely_contact_threshold_body_lengths for d in target_distances)
    required = max(1, int(SETTINGS.contact_confirmation_frames))
    temporal_confirmed = exact_hits >= 1 and support_hits >= required

    event.target = best_target
    event.evidence["contact_distance_body_lengths"] = float(best_distance)
    event.evidence["contact_target_distances_over_time"] = target_distances
    event.evidence["contact_exact_frames"] = int(exact_hits)
    event.evidence["contact_support_frames"] = int(support_hits)
    event.evidence["contact_temporally_confirmed"] = bool(temporal_confirmed)
    event.evidence["contact_best_frame"] = best_eval.get("frame")
    event.evidence["contact_best_time"] = best_eval.get("time")
    event.evidence["contact_attacker_conf"] = best_eval.get("attacker_conf")
    event.evidence["target_distances"] = {k: float(v) for k, v in best_eval["distances"].items()}

    block_dist = float(best_eval["block_distance"])
    event.evidence["defender_guard_distance"] = block_dist

    # A glove appearing closest to a leg during a crossing/occlusion is not a
    # legal punch target and must never become a scored timeline moment.
    if event.family == "punch" and best_target == "leg":
        event.target = None
        event.outcome = "uncertain"
        event.landed = False
        event.contact_confidence = 0.0
        event.evidence["rejected_contact_reason"] = "punch_endpoint_in_leg_zone"
        return event

    if best_distance <= SETTINGS.contact_threshold_body_lengths and temporal_confirmed:
        if event.family == "kick" and best_target == "leg" and _is_leg_check(best_eval["defender_kp"], best_eval["body"]):
            event.outcome = "checked"
            event.landed = False
            event.contact_confidence = min(0.98, 0.80 + 0.04 * min(3, support_hits))
        elif best_target in {"head", "body"} and block_dist <= SETTINGS.block_proximity_body_lengths:
            event.outcome = "blocked"
            event.landed = False
            event.contact_confidence = min(0.97, 0.78 + 0.04 * min(3, support_hits))
        else:
            event.outcome = "clean"
            event.landed = True
            event.contact_confidence = min(0.99, 0.82 + 0.04 * min(3, support_hits))
    elif best_distance <= SETTINGS.likely_contact_threshold_body_lengths:
        # One sampled frame can miss the exact instant of contact at 10-15 FPS.
        # Preserve that uncertainty instead of calling it definitively clean.
        event.outcome = "likely_landed"
        event.landed = True
        span = SETTINGS.likely_contact_threshold_body_lengths - SETTINGS.contact_threshold_body_lengths
        distance_penalty = max(0.0, best_distance - SETTINGS.contact_threshold_body_lengths) / max(0.01, span)
        event.contact_confidence = max(0.45, min(0.78, 0.72 - 0.20 * distance_penalty + 0.02 * support_hits))
    else:
        event.outcome = "missed"
        event.landed = False
        event.contact_confidence = min(0.92, 0.52 + min(0.40, best_distance - SETTINGS.likely_contact_threshold_body_lengths))

    # Keep side, kick height and target internally consistent. The tracked limb
    # is authoritative for left/right; the verified contact zone is
    # authoritative for low/body/head.
    if event.family == "kick" and any(token in event.technique for token in ("round_kick", "low_kick", "body_kick", "head_kick")):
        side = "left" if event.limb.startswith("left") else "right"
        if best_target == "leg":
            event.technique = f"{side}_low_kick"
        elif best_target == "body":
            event.technique = f"{side}_body_kick"
        elif best_target == "head":
            event.technique = f"{side}_head_kick"

    # The public event time is the verified impact/closest-approach sample,
    # not the earlier maximum-extension sample. Preserve the motion peak for
    # diagnostics and training exports.
    contact_frame = best_eval.get("frame")
    contact_time = best_eval.get("time")
    event.evidence.setdefault("action_peak_frame", int(event.peak_frame))
    event.evidence.setdefault("action_peak_time", float(event.peak_time))
    if contact_frame is not None:
        event.peak_frame = int(contact_frame)
    if contact_time is not None:
        event.peak_time = float(contact_time)
    event.metadata["attacker_identity_confidence"] = float(best_eval.get("attacker_identity_confidence", 1.0))
    event.metadata["opponent_identity_confidence"] = float(best_eval.get("opponent_identity_confidence", 1.0))

    return event
