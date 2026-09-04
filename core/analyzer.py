from __future__ import annotations

import json
import hashlib
import math
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from core.action import ActionEngine
from core.config import OUTPUTS, SETTINGS
from core.fighter_suggest import FighterFinder, analysis_missed_the_fight
from core.generalship import judge_fight
from core.round_detect import RoundDetector
from core.contact import (
    assess_selection,
    classify_contact,
    opponent_separation,
    thrown_at_opponent,
)
from core.db import save_fight
from core.defense import DefenseEngine
from core.evidence_trust import automated_evidence_trust
from core.fight_stats import normalize_outcome, summarize_fight_events
from core.action import CONFIDENCE_CEILING, CONFIDENCE_FLOOR

# How much of the confidence range an attempt must clear to be shown.
#
# An attempt is the low-information tier: this fighter, at this second, threw a
# punch or a kick. No contact, no target, no scoring - those stay withheld
# until the action classifier is validated. So the bar is deliberately low, and
# only has to reject a trigger that fired with essentially no evidence behind
# it. Precision belongs on the verified tier, which _live_event_reliable gates
# separately and far harder.
#
# Expressed as a share of the range rather than a constant, because the last
# constant here was calibrated against an older formula, survived a rescale of
# it, and left six real fights showing 5 attempts out of 309 detections.
ATTEMPT_CONFIDENCE = CONFIDENCE_FLOOR + 0.25 * (CONFIDENCE_CEILING - CONFIDENCE_FLOOR)

from core.identity import IdentityManager
from core.metrics import MetricsAccumulator
from core.pose_tracker import PoseTracker, QualityController, find_initial_people
from core.report import build_report, write_report
from core.sam_recovery import SamRecovery, nearest_guidance, sam_sampling_stride
from core.openai_identity import OpenAIIdentityReferee
from core.scoring import is_legal_event, normalize_ruleset
from core.types import AnalysisProgress, AnalysisRequest, PersonObservation, PoseFrame, RoundSpec
from core.video import build_round_schedule, get_video_info, requested_segment_end, round_at_time

ProgressCallback = Callable[[dict], None]


def _live_event_reliable(event, ruleset: str) -> bool:
    return (
        is_legal_event(event, ruleset)
        and event.outcome in {"clean", "blocked", "checked", "missed"}
        and event.target in {"head", "body", "leg"}
        and float(event.confidence) >= 0.84
        and float(event.contact_confidence) >= 0.86
        and float(event.metadata.get("attacker_identity_confidence", 1.0)) >= 0.70
        and float(event.metadata.get("opponent_identity_confidence", 1.0)) >= 0.70
    )


def _live_attempt_reliable(event) -> bool:
    """Return only identity-safe temporal attempts for the provisional live view.

    This is deliberately a lower information tier than verified fight evidence:
    it supports fighter, timestamp and broad punch/kick family only. Contact,
    target, technique side and scoring remain withheld until the release gate
    validates the complete action classifier.
    """
    return (
        bool(getattr(event, "attempted", True))
        and getattr(event, "family", None) in {"punch", "kick"}
        and math.isfinite(float(getattr(event, "peak_time", -1.0)))
        and float(getattr(event, "peak_time", -1.0)) >= 0.0
        and float(getattr(event, "confidence", 0.0)) >= ATTEMPT_CONFIDENCE
        and float(event.metadata.get("attacker_identity_confidence", 1.0)) >= 0.76
        and float(event.metadata.get("opponent_identity_confidence", 1.0)) >= 0.76
    )


def _live_event_payload(events: list, ruleset: str, trusted: bool, limit: int | None = 160) -> list[dict]:
    reliable = sorted(
        (
            event for event in events
            if _live_attempt_reliable(event)
        ),
        key=lambda item: item.peak_time,
    )
    deduplicated = []
    for event in reliable:
        duplicate_index = next((
            index for index, kept in enumerate(deduplicated)
            if event.fighter == kept.fighter
            and (event.limb or event.family) == (kept.limb or kept.family)
            and abs(event.peak_time - kept.peak_time) <= 0.48
        ), None)
        if duplicate_index is None:
            deduplicated.append(event)
        elif (event.contact_confidence, event.confidence) > (
            deduplicated[duplicate_index].contact_confidence,
            deduplicated[duplicate_index].confidence,
        ):
            deduplicated[duplicate_index] = event
    payload = []
    for event in deduplicated:
        outcome_reliable = trusted and _live_event_reliable(event, ruleset)
        outcome = "uncertain"
        if outcome_reliable:
            outcome = normalize_outcome(event.outcome)
            defense = str(event.metadata.get("defense") or "")
            defense_confidence = float(event.metadata.get("defense_confidence", 0.0) or 0.0)
            if event.outcome == "missed" and defense_confidence >= 0.70:
                if defense in {"slip", "evade"}:
                    outcome = "evaded"
                elif defense == "parry":
                    outcome = "blocked"
        payload.append({
            "id": f"{event.fighter}-{event.peak_frame}-{event.technique if trusted else event.family}",
            "kind": "strike",
            "fighter": event.fighter,
            "round_number": event.round_number,
            "start_time": float(getattr(event, "start_time", event.peak_time)),
            "time_seconds": float(event.peak_time),
            "end_time": float(getattr(event, "end_time", event.peak_time)),
            "technique": event.technique if trusted else None,
            "family": event.family,
            "limb": event.limb if trusted else None,
            "target": event.target if outcome_reliable else None,
            "outcome": outcome if trusted else "unclassified",
            "confidence": float(
                min(event.confidence, event.contact_confidence) if outcome_reliable else event.confidence
            ),
            "verification": "verified" if outcome_reliable else "supported" if trusted else "observed",
        })
    return payload if limit is None else payload[-max(1, int(limit)):]


def _live_event_diagnostics(events: list, ruleset: str, trusted: bool, emitted: list[dict]) -> dict:
    return {
        "candidate_events_seen": len(events),
        "identity_safe_attempts": sum(_live_attempt_reliable(event) for event in events),
        "verified_events": sum(_live_event_reliable(event, ruleset) for event in events),
        "events_emitted": len(emitted),
        "event_mode": "validated_actions" if trusted else "observed_attempts",
    }


def _provisional_stats(
    live_events: list[dict], found: dict, analyzed_frames: int, trusted: bool,
    processed_seconds: float | None = None,
) -> dict:
    return summarize_fight_events(
        live_events, found, analyzed_frames, trusted, processed_seconds,
    )


def _live_keypoints(observation, width: int, height: int) -> list[list[float] | None] | None:
    """The fighter's joints for the live overlay, as fractions of the frame.

    The progress page drew a bounding box, which says where a fighter is but
    nothing about what they are doing - and a box around two people standing
    close together looks identical whether tracking is right or wrong. The
    skeleton makes a bad lock obvious while the analysis is still running.
    """
    points = getattr(observation, "keypoints", None)
    if points is None or not width or not height:
        return None
    scores = getattr(observation, "keypoint_conf", None)
    live: list[list[float] | None] = []
    for index, point in enumerate(points[:, :2]):
        score = 1.0
        if scores is not None and index < len(scores):
            score = float(scores[index])
        if score < _MIN_LIVE_KEYPOINT_CONF:
            live.append(None)
            continue
        live.append([float(point[0]) / width, float(point[1]) / height])
    return live


# A live keypoint is drawn or it is not; there is no half-confident joint on a
# moving overlay. Below this the point is sent as null so the page skips the
# limb rather than drawing an arm through the floor.
_MIN_LIVE_KEYPOINT_CONF = 0.30


def _latest_observation(seconds: float, width: int, height: int, fighter_a, fighter_b, manager) -> dict:
    def item(observation, confidence: float) -> dict:
        box = None
        keypoints = None
        if observation is not None:
            x1, y1, x2, y2 = (float(value) for value in observation.box)
            box = [x1 / width, y1 / height, x2 / width, y2 / height]
            keypoints = _live_keypoints(observation, width, height)
        return {
            "visible": observation is not None, "box": box, "keypoints": keypoints,
            "identity_confidence": float(confidence),
        }

    return {
        "time_seconds": float(seconds),
        "fighters": {
            "A": item(fighter_a, manager.a.identity_confidence),
            "B": item(fighter_b, manager.b.identity_confidence),
        },
    }


@lru_cache(maxsize=1)
def get_pose_tracker() -> PoseTracker:
    """One GPU model instance for the local WarriorIQ server."""
    return PoseTracker()


def _validate_request(req: AnalysisRequest, duration: float) -> None:
    req.analysis_target = (req.analysis_target or "BOTH").upper()
    if req.analysis_target not in {"A", "B", "BOTH"}:
        raise ValueError("analysis_target must be A, B or BOTH")
    req.focus_fighter = req.focus_fighter.upper() if req.focus_fighter else None
    if req.focus_fighter not in {None, "A", "B"}:
        raise ValueError("focus_fighter must be A or B")
    req.fight_type = (req.fight_type or "competition").lower()
    if req.fight_type not in {"competition", "sparring"}:
        raise ValueError("fight_type must be competition or sparring")
    req.ruleset = normalize_ruleset(req.ruleset)
    if len(req.fighter_a_box) != 4 or len(req.fighter_b_box) != 4:
        raise ValueError("Each fighter selection must contain four coordinates")
    req.start_seconds = max(0.0, min(float(req.start_seconds), max(0.0, duration - 0.001)))
    req.round_count = max(1, min(20, int(req.round_count)))
    req.round_duration_seconds = max(10.0, float(req.round_duration_seconds))
    req.break_duration_seconds = max(0.0, float(req.break_duration_seconds))
    if req.selected_rounds:
        req.selected_rounds = sorted({int(x) for x in req.selected_rounds if 1 <= int(x) <= req.round_count})
    if req.end_seconds is not None:
        req.end_seconds = max(req.start_seconds, min(float(req.end_seconds), duration))


def _serializable_observation(obs: PersonObservation | None) -> dict | None:
    if obs is None:
        return None
    return {
        "track_id": obs.track_id,
        "box": [float(x) for x in obs.box],
        "confidence": float(obs.confidence),
        "keypoints": None if obs.keypoints is None else [[float(x), float(y)] for x, y in obs.keypoints[:, :2]],
        "keypoint_conf": None if obs.keypoint_conf is None else [float(x) for x in obs.keypoint_conf],
    }


def _pose_frame(source_frame: int, seconds: float, round_number: int | None, fighter: str, obs: PersonObservation | None, identity_confidence: float) -> PoseFrame:
    return PoseFrame(
        source_frame=source_frame,
        time_seconds=seconds,
        round_number=round_number,
        fighter=fighter,
        box=None if obs is None else [float(x) for x in obs.box],
        keypoints=None if obs is None or obs.keypoints is None else [[float(x), float(y)] for x, y in obs.keypoints[:, :2]],
        keypoint_conf=None if obs is None or obs.keypoint_conf is None else [float(x) for x in obs.keypoint_conf],
        identity_confidence=float(identity_confidence),
        visible=obs is not None,
    )


def _buffer_since_last_seen(buffer: deque, last_seen_source_frame: int) -> list[np.ndarray]:
    if not buffer:
        return []
    items = list(buffer)
    start = 0
    for i, (frame_number, _) in enumerate(items):
        if frame_number <= last_seen_source_frame:
            start = i
    return [frame for _, frame in items[start:]]


class _Travel:
    """How far a tracked person actually went, in their own body lengths.

    Measured in body lengths rather than pixels so it means the same thing
    whether the camera is at the ring apron or the back of a sports hall.
    """

    __slots__ = ("_last", "_distance", "_heights", "_first_time", "_last_time")

    def __init__(self) -> None:
        self._last: tuple[float, float] | None = None
        self._distance = 0.0
        self._heights: list[float] = []
        self._first_time: float | None = None
        self._last_time: float | None = None

    def add(self, seconds: float, observation) -> None:
        box = getattr(observation, "box", None) if observation is not None else None
        if box is None:
            self._last = None          # a gap is not travel; do not bridge it
            return
        box = [float(value) for value in box]
        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        height = max(1.0, box[3] - box[1])
        self._heights.append(height)
        if self._first_time is None:
            self._first_time = seconds
        self._last_time = seconds
        if self._last is not None:
            self._distance += math.hypot(centre[0] - self._last[0], centre[1] - self._last[1])
        self._last = centre

    def body_lengths_per_minute(self) -> float | None:
        if not self._heights or self._first_time is None or self._last_time is None:
            return None
        span = self._last_time - self._first_time
        if span < 20.0:                # too short a look to judge anyone by
            return None
        median_height = sorted(self._heights)[len(self._heights) // 2]
        return round(self._distance / median_height / (span / 60.0), 1)


def _pair_separation(fighter_a, fighter_b) -> float | None:
    """How far apart the two fighters are, in body lengths, this frame."""
    box_a = getattr(fighter_a, "box", None) if fighter_a is not None else None
    box_b = getattr(fighter_b, "box", None) if fighter_b is not None else None
    if box_a is None or box_b is None:
        return None
    ax = (float(box_a[0]) + float(box_a[2])) / 2.0
    ay = (float(box_a[1]) + float(box_a[3])) / 2.0
    bx = (float(box_b[0]) + float(box_b[2])) / 2.0
    by = (float(box_b[1]) + float(box_b[3])) / 2.0
    body = max(20.0, (float(box_a[3]) - float(box_a[1]) + float(box_b[3]) - float(box_b[1])) / 2.0)
    return math.hypot(ax - bx, ay - by) / body


def _observed_fighter_mismatch(finder) -> dict | None:
    """Who actually fought, and whether the analysis was watching them.

    Returns None when there is nothing to say - too little footage, or nobody
    in frame cleared the movement floor. Saying nothing is the right answer far
    more often than guessing.
    """
    best = finder.best_pair()
    if not best:
        return None
    followed = best.get("followed_share")
    return {
        "centres": best["centres"],
        "travel_per_minute": best["travel"],
        "share_within_range": best["close_share"],
        "followed_share": followed,
        "disagrees_with_selection": analysis_missed_the_fight(followed),
    }


def analyze(req: AnalysisRequest, progress_callback: ProgressCallback | None = None) -> dict:
    info = get_video_info(req.video_path)
    _validate_request(req, info.duration)
    rounds = build_round_schedule(req, info)
    segment_end_seconds = requested_segment_end(req, info, rounds)
    segment_duration = max(0.001, segment_end_seconds - req.start_seconds)
    start_frame = int(round(req.start_seconds * info.fps))
    end_frame = min(info.frame_count, int(math.ceil(segment_end_seconds * info.fps)))

    # Repeated analyses must begin from the same pseudo-random state. CUDA
    # kernels may still have hardware-specific behavior, so the report records
    # the complete sampling/seed signature rather than promising bit identity.
    np.random.seed(0)
    cv2.setRNGSeed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    job_dir = OUTPUTS / req.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = job_dir / "tracking.jsonl"
    events_path = job_dir / "events.json"

    # ETA follows the same weighted phase progress as the visible progress bar.
    # The old calculation used "processed video seconds", which reaches the end
    # during SAM and then starts again during pose analysis, causing huge jumps.
    estimated_total_seconds: float | None = None
    live_action_trusted = False

    def progress(
        message: str,
        percent: float,
        elapsed: float,
        processed: float,
        manager=None,
        current_round=None,
        quality=None,
        *,
        stage: str | None = None,
        live_events_snapshot: list[dict] | None = None,
        stats: dict | None = None,
        observation: dict | None = None,
    ):
        nonlocal estimated_total_seconds
        if progress_callback is None:
            return
        speed = processed / elapsed if elapsed > 0.0 else 0.0
        bounded_percent = float(max(0.0, min(100.0, percent)))
        if bounded_percent >= 100.0:
            eta = 0.0
        elif bounded_percent >= 2.0 and elapsed > 0.0:
            projected_total = elapsed / (bounded_percent / 100.0)
            if estimated_total_seconds is None:
                estimated_total_seconds = projected_total
            else:
                # A conservative EWMA absorbs short model warmups and frame
                # complexity changes without presenting impossible minute jumps.
                estimated_total_seconds = 0.82 * estimated_total_seconds + 0.18 * projected_total
            estimated_total_seconds = max(elapsed, estimated_total_seconds)
            eta = max(0.0, estimated_total_seconds - elapsed)
        else:
            eta = None
        payload = AnalysisProgress(
            percent=bounded_percent,
            message=message,
            elapsed_seconds=float(elapsed),
            # SAM2 has its own pass over the whole clip. The public analysis
            # cursor must advance only when the action/pose pass has processed
            # that video time, otherwise the live timeline would get ahead of
            # the evidence actually available to the user.
            processed_video_seconds=float(processed if stage in {"analysis", "report", "complete"} else 0.0),
            speed=float(speed),
            eta_seconds=float(max(0.0, eta)) if eta is not None else None,
            fighter_a_confidence=0.0 if manager is None else float(manager.a.identity_confidence),
            fighter_b_confidence=0.0 if manager is None else float(manager.b.identity_confidence),
            current_round=current_round,
            quality_mode="balanced" if quality is None else quality.mode,
            stage=stage or ("complete" if bounded_percent >= 100 else "preparing"),
            video_duration_seconds=float(info.duration),
            live_event_mode="validated_actions" if live_action_trusted else "observed_attempts",
            live_events=live_events_snapshot or [],
            provisional_stats=stats or {},
            latest_observation=observation,
        )
        progress_callback(payload.to_dict())

    # Honest wall timer: model load/warmup is part of the user's wait.
    wall_start = time.perf_counter()
    progress("Loading GPU models", 0.0, 0.0, 0.0)

    pose_tracker = get_pose_tracker()
    sam_recovery = SamRecovery()
    action_engine = ActionEngine()
    defense_engine = DefenseEngine()
    metrics = MetricsAccumulator(info.width, info.height)
    quality = QualityController(info.fps, info.width, info.height)
    classifier = {
        "action_classifier": "warrioriq_temporal_model" if action_engine.temporal.available else "multi_frame_temporal_rules",
        "custom_temporal_checkpoint_loaded": bool(action_engine.temporal.available),
        "temporal_architecture": action_engine.temporal.architecture,
        "temporal_validation": action_engine.temporal.validation,
        "contact_classifier": "pose_geometry_temporal_contact",
        "max_engagement_body_lengths": SETTINGS.max_engagement_body_lengths,
        "uncertainty_policy": "No single-frame strike events, temporal support for contact, and no identity reassignment when recovery evidence is ambiguous.",
    }
    live_action_trusted = bool(automated_evidence_trust(classifier)["automated_evidence_trusted"])

    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open fight video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("Could not read the selected fight-start frame")

    # Warm model before tracker initialization. This also downloads the model on
    # first use if Ultralytics has not cached it yet.
    pose_tracker.warmup(first_frame)
    # The model is cached across jobs for speed; tracker identities are not.
    pose_tracker.reset_tracking()

    # Start BoT-SORT on exactly the same frame the user used for A/B selection.
    first_people = pose_tracker.track(first_frame, quality.imgsz)
    initial_a, initial_b, iou_a, iou_b = find_initial_people(
        np.asarray(req.fighter_a_box, dtype=np.float32),
        np.asarray(req.fighter_b_box, dtype=np.float32),
        first_people,
        first_frame,
    )
    manager = IdentityManager(initial_a, initial_b, start_frame, source_fps=info.fps)
    canonical_a_box = [float(value) for value in initial_a.box]
    canonical_b_box = [float(value) for value in initial_b.box]
    identity_referee = OpenAIIdentityReferee(req.openai_identity_recovery, first_frame, canonical_a_box, canonical_b_box)

    progress("Following both fighters with SAM2", 1.0, time.perf_counter() - wall_start, 0.0, manager, None, quality, stage="tracking")
    sam_tracks = sam_recovery.track_segment(
        req.video_path,
        start_frame,
        end_frame,
        info.fps,
        canonical_a_box,
        canonical_b_box,
        progress_callback=lambda completed, total: progress(
            "Following both fighters with SAM2",
            2.0 + 33.0 * completed / max(1, total),
            time.perf_counter() - wall_start,
            segment_duration * completed / max(1, total),
            manager,
            None,
            quality,
            stage="tracking",
        ),
    )
    sam_was_available = sam_recovery.available
    sam_recovery.release()
    # SAM2 reads the segment independently. Resume the pose pass immediately
    # after the already-consumed selection frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 1)
    pose_pass_start = time.perf_counter()

    # Continuous SAM guidance already provides the recovery path. Avoid a
    # full-resolution copy of every decoded frame when that path is available;
    # the fallback buffer is retained unchanged when continuous guidance is not.
    fallback_buffer_enabled = bool(SETTINGS.sam_recovery_enabled and not sam_tracks)
    frame_buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=max(SETTINGS.sam_buffer_frames + 4, 24))
    if fallback_buffer_enabled:
        frame_buffer.append((start_frame, first_frame.copy()))
    ai_history: deque[np.ndarray] = deque(maxlen=7)
    # OpenAI recovery is opt-in. Resizing one frame per second for a disabled
    # feature added CPU work without contributing to local analysis quality.
    if identity_referee.enabled:
        ai_history.append(cv2.resize(first_frame, (640, max(1, round(first_frame.shape[0] * 640 / first_frame.shape[1])))))
    next_ai_history_frame = start_frame + max(1, round(info.fps))
    next_ai_audit_seconds = req.start_seconds + SETTINGS.openai_identity_audit_seconds

    events = []
    defenses = []
    analyzed_frames = 0
    active_analyzed_frames = 0
    found = {"A": 0, "B": 0}
    missing = {"A": 0, "B": 0}
    sam_guided = {"A": 0, "B": 0}
    guided_pose_recoveries = {"A": 0, "B": 0}
    coverage_windows: dict[int, dict[str, int]] = {}
    sam_stride = sam_sampling_stride(info.fps, end_frame - start_frame)
    current_frame = start_frame
    last_progress_emit = 0
    # The start frame is already analyzed above. Schedule the next expensive
    # inference at the adaptive tracking stride instead of analyzing the very
    # next source frame again.
    next_inference_frame = start_frame + max(1, quality.stride)
    current_imgsz = quality.imgsz
    decoded_seconds = req.start_seconds

    # Since the first frame has already been consumed, process it through the
    # same downstream path before entering the read loop.
    pending = [(start_frame, first_frame, first_people, initial_a, initial_b)]

    out_of_range_actions = 0
    observed_separations: list[float] = []
    travel = {"A": _Travel(), "B": _Travel()}
    fighter_finder = FighterFinder()
    round_detector = RoundDetector()
    tracking_file = tracking_path.open("w", encoding="utf-8") if SETTINGS.save_tracking_jsonl else None

    try:
        while current_frame < end_frame:
            if pending:
                source_frame, frame, people, fighter_a, fighter_b = pending.pop(0)
            else:
                source_frame = current_frame + 1
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                current_frame = source_frame
                pts_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                decoded_seconds = pts_ms / 1000.0 if pts_ms > 0.0 else source_frame / info.fps
                if fallback_buffer_enabled:
                    frame_buffer.append((source_frame, frame.copy()))
                if identity_referee.enabled and source_frame >= next_ai_history_frame:
                    ai_history.append(cv2.resize(frame, (640, max(1, round(frame.shape[0] * 640 / frame.shape[1])))))
                    next_ai_history_frame = source_frame + max(1, round(info.fps))

                if source_frame < next_inference_frame:
                    continue

                seconds = decoded_seconds
                spec = round_at_time(rounds, seconds)
                active_selected_round = bool(spec and spec.selected)
                base_stride = quality.stride
                # Preserve identity through breaks/non-selected rounds at about
                # 5 FPS without spending full action-analysis budget.
                inference_stride = base_stride if active_selected_round else max(base_stride, round(info.fps / 5.0))
                next_inference_frame = source_frame + max(1, inference_stride)

                people = pose_tracker.track(frame, current_imgsz)
                guidance = nearest_guidance(sam_tracks, source_frame, sam_stride)
                focused = pose_tracker.recover_from_guidance(frame, guidance, people)
                for observation in focused:
                    if observation.track_id == -1001:
                        guided_pose_recoveries["A"] += 1
                    elif observation.track_id == -1002:
                        guided_pose_recoveries["B"] += 1
                people.extend(focused)
                fighter_a, fighter_b = manager.update(people, source_frame, sam_guidance=guidance)
                if guidance is not None:
                    if fighter_a is not None and guidance.get("A") is not None:
                        sam_guided["A"] += 1
                    if fighter_b is not None and guidance.get("B") is not None:
                        sam_guided["B"] += 1

                # Rare, short-window recovery. Never accept a SAM box directly
                # as identity; it must still agree with a current detector person.
                elapsed_now = time.perf_counter() - wall_start
                processed_now = max(0.0, seconds - req.start_seconds)
                speed_now = processed_now / elapsed_now if elapsed_now > 0 else 0.0
                recovery_allowed = quality.mode != "deadline" or speed_now >= 0.90

                for state, name in ((manager.a, "A"), (manager.b, "B")):
                    if recovery_allowed and not sam_tracks and manager.needs_recovery(state, analyzed_frames):
                        state.last_sam_attempt_analyzed_frame = analyzed_frames
                        buffered = _buffer_since_last_seen(frame_buffer, state.last_seen_source_frame)
                        recovered_box = sam_recovery.recover(buffered, state.last_box)
                        recovered_obs = manager.apply_external_recovery(state, recovered_box, people, source_frame)
                        if recovered_obs is not None:
                            if name == "A":
                                fighter_a = recovered_obs
                            else:
                                fighter_b = recovered_obs

                # When local tracking and SAM remain unresolved, ask the
                # optional OpenAI visual referee to jointly identify A/B.
                if fighter_a is None or fighter_b is None or (identity_referee.enabled and seconds >= next_ai_audit_seconds):
                    decision = identity_referee.recover(list(ai_history), frame, people, seconds)
                    if seconds >= next_ai_audit_seconds:
                        next_ai_audit_seconds = seconds + SETTINGS.openai_identity_audit_seconds
                    if decision is not None:
                        recovered_a, recovered_b = manager.apply_ai_assignment(
                            people,
                            int(decision["fighter_a_candidate"]),
                            int(decision["fighter_b_candidate"]),
                            source_frame,
                            float(decision["confidence"]),
                        )
                        fighter_a = recovered_a or fighter_a
                        fighter_b = recovered_b or fighter_b

            seconds = req.start_seconds if source_frame == start_frame else decoded_seconds
            spec = round_at_time(rounds, seconds)
            round_number = spec.number if spec else None
            active_selected_round = bool(spec and spec.selected)

            # First frame did not run manager.update because the initial lock is
            # itself the update.
            if source_frame == start_frame:
                manager.a.identity_confidence = 1.0
                manager.b.identity_confidence = 1.0

            analyzed_frames += 1
            if fighter_a is not None:
                found["A"] += 1
            else:
                missing["A"] += 1
            if fighter_b is not None:
                found["B"] += 1
            else:
                missing["B"] += 1
            window_start = int(max(0.0, seconds - req.start_seconds) // 10) * 10
            window = coverage_windows.setdefault(window_start, {"analyzed": 0, "A": 0, "B": 0})
            window["analyzed"] += 1
            window["A"] += int(fighter_a is not None)
            window["B"] += int(fighter_b is not None)

            defense_engine.update_pose("A", source_frame, seconds, fighter_a)
            defense_engine.update_pose("B", source_frame, seconds, fighter_b)

            if active_selected_round:
                active_analyzed_frames += 1
                metrics.update("A", seconds, round_number, fighter_a, fighter_b)
                metrics.update("B", seconds, round_number, fighter_b, fighter_a)

                new_events = []
                if req.analysis_target in {"A", "BOTH"}:
                    new_events.extend(action_engine.update(
                        "A", source_frame, seconds, round_number, fighter_a, fighter_b,
                        manager.a.identity_confidence, manager.b.identity_confidence,
                    ))
                if req.analysis_target in {"B", "BOTH"}:
                    new_events.extend(action_engine.update(
                        "B", source_frame, seconds, round_number, fighter_b, fighter_a,
                        manager.b.identity_confidence, manager.a.identity_confidence,
                    ))

                for event in new_events:
                    event = classify_contact(event)
                    # Recorded before the range gate, so the record covers every
                    # action seen - including the ones thrown at nobody, which
                    # are exactly the evidence that the wrong person was picked.
                    separation = opponent_separation(event)
                    if separation is not None:
                        observed_separations.append(separation)
                    if not thrown_at_opponent(event):
                        out_of_range_actions += 1
                        continue
                    events.append(event)
                    defense = defense_engine.classify(event)
                    if defense is not None:
                        event.metadata["defense"] = defense.defense
                        event.metadata["defense_confidence"] = float(defense.confidence)
                        defenses.append(defense)

            # Fighters move. Someone at ringside does not, and that is the
            # difference a separation test cannot see when the wrong two
            # people happen to be standing next to each other.
            travel["A"].add(seconds, fighter_a)
            travel["B"].add(seconds, fighter_b)
            fighter_finder.observe(seconds, people)
            fighter_finder.observe_selected(fighter_a, fighter_b)
            round_detector.observe(seconds, _pair_separation(fighter_a, fighter_b))

            if tracking_file is not None:
                record = {
                    "source_frame": source_frame,
                    "time_seconds": seconds,
                    "round_number": round_number,
                    "selected_round": active_selected_round,
                    "fighter_A": {
                        "identity_confidence": manager.a.identity_confidence,
                        "warrioriq_identity": "A",
                        "current_track_id": manager.a.current_track_id,
                        "observation": _serializable_observation(fighter_a),
                    },
                    "fighter_B": {
                        "identity_confidence": manager.b.identity_confidence,
                        "warrioriq_identity": "B",
                        "current_track_id": manager.b.current_track_id,
                        "observation": _serializable_observation(fighter_b),
                    },
                }
                tracking_file.write(json.dumps(record) + "\n")

            processed_seconds = min(segment_duration, max(0.0, seconds - req.start_seconds))
            elapsed = time.perf_counter() - wall_start
            # Adapt pose inference to pose-pass throughput. The SAM2 primary
            # pass is already complete and must not force lower pose quality.
            quality.maybe_adjust(analyzed_frames, processed_seconds, time.perf_counter() - pose_pass_start)
            current_imgsz = quality.imgsz

            if analyzed_frames - last_progress_emit >= SETTINGS.progress_interval_frames:
                last_progress_emit = analyzed_frames
                percent = 35.0 + 63.0 * processed_seconds / segment_duration
                all_live_event_data = _live_event_payload(events, req.ruleset, live_action_trusted, limit=None)
                live_event_data = all_live_event_data[-160:]
                live_stats = _provisional_stats(
                    all_live_event_data, found, analyzed_frames, live_action_trusted, processed_seconds,
                )
                live_stats["diagnostics"] = _live_event_diagnostics(
                    events, req.ruleset, live_action_trusted, all_live_event_data,
                )
                progress(
                    "Analyzing fight",
                    percent,
                    elapsed,
                    processed_seconds,
                    manager,
                    round_number,
                    quality,
                    stage="analysis",
                    live_events_snapshot=live_event_data,
                    stats=live_stats,
                    observation=_latest_observation(seconds, info.width, info.height, fighter_a, fighter_b, manager),
                )

            if source_frame >= end_frame - 1:
                break

    finally:
        cap.release()
        if tracking_file is not None:
            tracking_file.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    analysis_seconds = time.perf_counter() - wall_start
    realtime_speed = segment_duration / analysis_seconds if analysis_seconds > 0 else 0.0
    within_budget = analysis_seconds <= segment_duration

    # Round structure comes from the fight, not from the setup page.
    #
    # The page guessed "two-minute rounds, one minute between", so a 3:17 bout
    # became a single 2:00 round and the remaining 77 seconds only survived
    # because build_round_schedule extends the last round to the end of the
    # footage. A guess also cannot know that this fight ran three-minute
    # rounds, or ten of them, or that it stopped for an injury or for the
    # referee to give a count. RoundDetector reads that out of the video: a
    # sustained stretch where the fighters are apart and staying apart is a
    # break, and everything between breaks is a round. Whatever it finds
    # replaces the schedule here, so round numbers, per-round scoring and the
    # scorecard all describe the fight that actually happened.
    #
    # Only when the whole video was asked for. Someone who requested round 2 of
    # 5 meant it, and re-cutting the fight underneath them would be wrong.
    detected_round_spans = round_detector.rounds()
    rounds_from_footage = bool(detected_round_spans) and all(spec.selected for spec in rounds)
    if rounds_from_footage:
        rounds = [
            RoundSpec(item.number, item.start_seconds, item.end_seconds, True)
            for item in detected_round_spans
        ]
        for event in events:
            spec = round_at_time(rounds, float(event.peak_time))
            if spec is not None:
                event.round_number = spec.number
        # Per-round pose evidence was bucketed against the old schedule while
        # the loop ran, so rebuild it against the rounds that were found.
        metrics.rebucket_rounds(rounds)

    all_final_live_events = _live_event_payload(events, req.ruleset, live_action_trusted, limit=None)
    final_live_stats = _provisional_stats(
        all_final_live_events, found, analyzed_frames, live_action_trusted, segment_duration,
    )
    final_live_stats["diagnostics"] = _live_event_diagnostics(
        events, req.ruleset, live_action_trusted, all_final_live_events,
    )
    final_live_events = all_final_live_events[-160:]
    public_event_ids = {item["id"] for item in all_final_live_events}
    report_events = (
        [
            event for event in events
            if f"{event.fighter}-{event.peak_frame}-{event.technique}" in public_event_ids
        ]
        if live_action_trusted else events
    )
    classifier["actions_discarded_out_of_range"] = out_of_range_actions
    metric_data = metrics.finalize(report_events, defenses, segment_duration)
    signature_payload = {
        "video_segment": [start_frame, end_frame, round(info.fps, 6)],
        "canonical_boxes": [canonical_a_box, canonical_b_box],
        "pose_model": pose_tracker.model_path,
        "tracker": SETTINGS.tracker,
        "imgsz": SETTINGS.default_imgsz,
        "target_fps": SETTINGS.target_tracking_fps,
        "adaptive_quality": SETTINGS.adaptive_quality,
        "sam_fps": SETTINGS.sam_continuous_fps,
        "seed": 0,
    }
    tracking = {
        "metric_definition": "Observation coverage: accepted fighter observations divided by analyzed frames. This is not ground-truth identity accuracy.",
        "selection_source_frame": start_frame,
        "selection_source_seconds": start_frame / info.fps,
        "requested_fighter_A_box": [float(value) for value in req.fighter_a_box],
        "requested_fighter_B_box": [float(value) for value in req.fighter_b_box],
        "canonical_fighter_A_box": canonical_a_box,
        "canonical_fighter_B_box": canonical_b_box,
        "fighter_A_seed_source": "pose_detector" if initial_a.track_id is not None else "manual_anchor",
        "fighter_B_seed_source": "pose_detector" if initial_b.track_id is not None else "manual_anchor",
        "analysis_signature": hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16],
        "coverage_windows": [
            {
                "start_seconds": start,
                "end_seconds": min(start + 10, segment_duration),
                "fighter_A_coverage": values["A"] / max(1, values["analyzed"]),
                "fighter_B_coverage": values["B"] / max(1, values["analyzed"]),
                "analyzed_frames": values["analyzed"],
            }
            for start, values in sorted(coverage_windows.items())
        ],
        "initial_iou_A": iou_a,
        "initial_iou_B": iou_b,
        "analyzed_frames": analyzed_frames,
        "active_round_analyzed_frames": active_analyzed_frames,
        "fighter_A_coverage": found["A"] / max(1, analyzed_frames),
        "fighter_B_coverage": found["B"] / max(1, analyzed_frames),
        "fighter_A_missing_frames": missing["A"],
        "fighter_B_missing_frames": missing["B"],
        "fighter_A_recoveries": manager.a.recovery_count,
        "fighter_B_recoveries": manager.b.recovery_count,
        "fighter_A_sam_recoveries": manager.a.sam_recovery_count,
        "fighter_B_sam_recoveries": manager.b.sam_recovery_count,
        "sam_continuous_enabled": SETTINGS.sam_continuous_enabled,
        "sam_continuous_frames": sam_recovery.continuous_frames,
        "fighter_A_sam_guided_frames": sam_guided["A"],
        "fighter_B_sam_guided_frames": sam_guided["B"],
        "fighter_A_guided_pose_recoveries": guided_pose_recoveries["A"],
        "fighter_B_guided_pose_recoveries": guided_pose_recoveries["B"],
        "sam_continuous_failure_reason": sam_recovery.continuous_failure_reason,
        "fighter_A_rejected_switches": manager.a.switches_rejected,
        "fighter_B_rejected_switches": manager.b.switches_rejected,
        "sam_available": sam_was_available,
        "sam_failure_reason": sam_recovery.failure_reason,
        "openai_identity_enabled": identity_referee.enabled,
        "openai_identity_attempts": identity_referee.attempts,
        "openai_identity_recoveries": identity_referee.recoveries,
        "openai_identity_failure_reason": identity_referee.failure_reason,
    }
    performance = {
        "segment_duration_seconds": segment_duration,
        "analysis_seconds": analysis_seconds,
        "realtime_speed": realtime_speed,
        "within_video_length_budget": within_budget,
        "final_analysis_fps": quality.effective_fps,
        "final_imgsz": quality.imgsz,
        "quality_mode": quality.mode,
        "pose_model": pose_tracker.model_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "unused_frame_copy_avoidance": not fallback_buffer_enabled,
        "external_identity_history_enabled": identity_referee.enabled,
    }
    progress(
        "Building performance report", 98.3, analysis_seconds, segment_duration, manager, None, quality,
        stage="report", live_events_snapshot=final_live_events, stats=final_live_stats,
        observation=_latest_observation(segment_end_seconds, info.width, info.height, fighter_a, fighter_b, manager),
    )
    original_name = req.original_name or Path(req.video_path).name
    report = build_report(
        req=req,
        original_name=original_name,
        rounds=rounds,
        events=report_events,
        defenses=defenses,
        metrics=metric_data,
        tracking=tracking,
        performance=performance,
        classifier=classifier,
    )
    report["detected_rounds"] = round_detector.summary() | {"applied": rounds_from_footage}

    # A scorecard for the criteria movement can evidence. Kept separate from
    # report["scorecard"] on purpose: that one scores strikes and is withheld
    # because strikes cannot be detected reliably, while this one scores
    # aggression, generalship and territory and says exactly what it leaves out.
    report["movement_scorecard"] = judge_fight(
        metrics, rounds,
        {f: float(report["tracking"].get(f"fighter_{f}_coverage", 0.0)) for f in ("A", "B")},
        SETTINGS.min_tracking_coverage_for_score,
    )
    report["selection_check"] = assess_selection(
        observed_separations,
        len(report_events),
        out_of_range_actions,
        landed=sum(
            1 for event in report_events
            if getattr(event, "outcome", None) in {"clean", "likely_landed"}
        ),
        travel_per_minute={
            fighter: travel[fighter].body_lengths_per_minute() for fighter in ("A", "B")
        },
        observed_fighters=_observed_fighter_mismatch(fighter_finder),
    )
    progress(
        "Finalizing coaching priorities", 99.2, time.perf_counter() - wall_start, segment_duration,
        manager, None, quality, stage="report", live_events_snapshot=final_live_events,
        stats=final_live_stats,
        observation=_latest_observation(segment_end_seconds, info.width, info.height, fighter_a, fighter_b, manager),
    )
    # Freeze the exact customer-facing event stream once. Live completion,
    # saved report totals, round summaries, evidence buttons, and progress
    # history all consume this same snapshot so their numbers cannot drift.
    report["statistics"] = final_live_stats
    report["event_feed"] = all_final_live_events
    if live_action_trusted:
        for fighter in ("A", "B"):
            public = final_live_stats["fighters"][fighter]
            attacks = report["metrics"][fighter]["attacks"]
            attacks.update({
                "attempts": public["attempts"],
                "landed": public["landed"],
                "missed": public["missed"],
                "blocked": public["blocked"],
                "checked": 0,
                "uncertain": public["uncertain"],
                "accuracy": public["accuracy"],
            })
            report["metrics"][fighter]["combinations"].update({
                "count": public["combinations"],
                "max_length": public["longest_combination"],
                "evidence": [
                    sequence["techniques"] for sequence in public["combination_sequences"]
                ],
            })
            report["metrics"][fighter]["strongest_weapon"] = (
                public["best_weapon"]["technique"] if public["best_weapon"] else None
            )
            dashboard = report["metrics"][fighter]["dashboard"]
            dashboard["accuracy"] = public["accuracy"]
            dashboard["activity_attempts_per_minute"] = public["activity_rate"]
            dashboard["combinations_per_minute"] = (
                float(public["combinations"]) / max(1e-6, segment_duration / 60.0)
            )
    # The customer-facing duration includes report construction, not only the
    # pose pass. Keep the saved report, progress screen and history summary on
    # the same wall-clock definition.
    analysis_seconds = time.perf_counter() - wall_start
    realtime_speed = segment_duration / analysis_seconds if analysis_seconds > 0 else 0.0
    within_budget = analysis_seconds <= segment_duration
    report["performance"].update({
        "analysis_seconds": analysis_seconds,
        "realtime_speed": realtime_speed,
        "within_video_length_budget": within_budget,
    })
    json_path, html_path = write_report(job_dir, report)
    events_path.write_text(json.dumps([e.to_dict() for e in events], indent=2), encoding="utf-8")
    progress(
        "Saving completed report", 99.7, time.perf_counter() - wall_start, segment_duration,
        manager, None, quality, stage="report", live_events_snapshot=final_live_events,
        stats=final_live_stats,
        observation=_latest_observation(segment_end_seconds, info.width, info.height, fighter_a, fighter_b, manager),
    )

    summary = {
        "winner_estimate": report["scorecard"]["winner_estimate"],
        "score_totals": report["scorecard"]["totals"],
        "analysis_seconds": analysis_seconds,
        "video_seconds": segment_duration,
        "within_budget": within_budget,
        "fighter_A_coverage": tracking["fighter_A_coverage"],
        "fighter_B_coverage": tracking["fighter_B_coverage"],
        # The Progress page needs only these compact, final fields. Keeping the
        # snapshot in SQLite avoids reopening and rebuilding every full report
        # whenever an athlete visits the dashboard.
        "progress_report": {
            "video": report.get("video", {}),
            "setup": report.get("setup", {}),
            "integrity": report.get("integrity", {}),
            "metrics": report.get("metrics", {}),
            "statistics": report.get("statistics", {}),
            "coaching": report.get("coaching", {}),
            "training_plan": report.get("training_plan", {}),
        },
    }
    if req.persist_result:
        save_fight(
            job_id=req.job_id,
            profile_id=req.profile_id,
            original_name=original_name,
            video_path=req.video_path,
            report_path=str(json_path),
            fight_type=req.fight_type,
            ruleset=req.ruleset,
            analysis_target=req.focus_fighter or req.analysis_target,
            summary=summary,
            fighter_id=req.fighter_id,
            video_delete_after=(
                datetime.now(timezone.utc) + timedelta(days=SETTINGS.saved_video_retention_days)
            ).isoformat(),
        )

    progress(
        "Complete", 100.0, time.perf_counter() - wall_start, segment_duration, manager, None, quality,
        stage="complete",
        live_events_snapshot=final_live_events,
        stats=final_live_stats,
        observation=_latest_observation(segment_end_seconds, info.width, info.height, fighter_a, fighter_b, manager),
    )
    return report
