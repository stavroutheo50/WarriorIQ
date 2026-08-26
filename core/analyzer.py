from __future__ import annotations

import json
import hashlib
import math
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from core.action import ActionEngine
from core.config import OUTPUTS, SETTINGS
from core.contact import classify_contact
from core.db import save_fight
from core.defense import DefenseEngine
from core.evidence_trust import automated_evidence_trust
from core.identity import IdentityManager
from core.metrics import MetricsAccumulator
from core.pose_tracker import PoseTracker, QualityController, find_initial_people
from core.report import build_report, write_report
from core.sam_recovery import SamRecovery, nearest_guidance, sam_sampling_stride
from core.openai_identity import OpenAIIdentityReferee
from core.scoring import is_legal_event, normalize_ruleset
from core.types import AnalysisProgress, AnalysisRequest, PersonObservation, PoseFrame
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
        and float(getattr(event, "confidence", 0.0)) >= 0.86
        and float(event.metadata.get("attacker_identity_confidence", 1.0)) >= 0.76
        and float(event.metadata.get("opponent_identity_confidence", 1.0)) >= 0.76
    )


def _live_event_payload(events: list, ruleset: str, trusted: bool, limit: int | None = 160) -> list[dict]:
    reliable = sorted(
        (
            event for event in events
            if (_live_event_reliable(event, ruleset) if trusted else _live_attempt_reliable(event))
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
    payload = [{
        "id": f"{event.fighter}-{event.peak_frame}-{event.technique if trusted else event.family}",
        "fighter": event.fighter,
        "round_number": event.round_number,
        "time_seconds": float(event.peak_time),
        "technique": event.technique if trusted else None,
        "family": event.family,
        "limb": event.limb if trusted else None,
        "target": event.target if trusted else None,
        "outcome": event.outcome if trusted else "unclassified",
        "confidence": float(min(event.confidence, event.contact_confidence) if trusted else event.confidence),
        "verification": "verified" if trusted else "observed",
    } for event in deduplicated]
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
    coverage = {
        fighter: (float(found[fighter]) / analyzed_frames if analyzed_frames else 0.0)
        for fighter in ("A", "B")
    }
    fighters = {}
    for fighter in ("A", "B"):
        own = [event for event in live_events if event["fighter"] == fighter]
        punches = [event for event in own if event.get("family") == "punch"]
        kicks = [event for event in own if event.get("family") == "kick"]
        landed = sum(event["outcome"] == "clean" for event in own)
        blocked = sum(event["outcome"] in {"blocked", "checked"} for event in own)
        missed = sum(event["outcome"] == "missed" for event in own)
        combinations = sum(
            1 for previous, current in zip(own, own[1:])
            if 0.10 <= current["time_seconds"] - previous["time_seconds"] <= 1.20
        )
        fighters[fighter] = {
            "attempts": len(own),
            "clean": landed if trusted else None,
            "blocked_or_checked": blocked if trusted else None,
            "missed": missed if trusted else None,
            "combinations": combinations if trusted else None,
            "punch_attempts": len(punches),
            "punches_landed": sum(event["outcome"] == "clean" for event in punches) if trusted else None,
            "punches_missed": sum(event["outcome"] == "missed" for event in punches) if trusted else None,
            "punches_blocked": sum(event["outcome"] in {"blocked", "checked"} for event in punches) if trusted else None,
            "kick_attempts": len(kicks),
            "kicks_landed": sum(event["outcome"] == "clean" for event in kicks) if trusted else None,
            "kicks_missed": sum(event["outcome"] == "missed" for event in kicks) if trusted else None,
            "kicks_blocked": sum(event["outcome"] in {"blocked", "checked"} for event in kicks) if trusted else None,
            "total_strikes": len(own),
            "accuracy": (float(landed) / len(own)) if trusted and own else (0.0 if trusted else None),
            "activity_rate": (
                float(len(own)) / (float(processed_seconds) / 60.0)
                if processed_seconds and processed_seconds > 0 else 0.0
            ),
            "observation_coverage": coverage[fighter],
        }
    return {
        "action_labels_available": trusted,
        "attempt_counts_available": True,
        "event_mode": "validated_actions" if trusted else "observed_attempts",
        "fighters": fighters,
    }


def _latest_observation(seconds: float, width: int, height: int, fighter_a, fighter_b, manager) -> dict:
    def item(observation, confidence: float) -> dict:
        box = None
        if observation is not None:
            x1, y1, x2, y2 = (float(value) for value in observation.box)
            box = [x1 / width, y1 / height, x2 / width, y2 / height]
        return {"visible": observation is not None, "box": box, "identity_confidence": float(confidence)}

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
            processed_video_seconds=float(processed if stage in {"analysis", "complete"} else 0.0),
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
    quality = QualityController(info.fps)
    classifier = {
        "action_classifier": "warrioriq_temporal_model" if action_engine.temporal.available else "multi_frame_temporal_rules",
        "custom_temporal_checkpoint_loaded": bool(action_engine.temporal.available),
        "temporal_architecture": action_engine.temporal.architecture,
        "temporal_validation": action_engine.temporal.validation,
        "contact_classifier": "pose_geometry_temporal_contact",
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
    manager = IdentityManager(initial_a, initial_b, start_frame)
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

    frame_buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=max(SETTINGS.sam_buffer_frames + 4, 24))
    frame_buffer.append((start_frame, first_frame.copy()))
    ai_history: deque[np.ndarray] = deque(maxlen=7)
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
                frame_buffer.append((source_frame, frame.copy()))
                if source_frame >= next_ai_history_frame:
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
                    events.append(event)
                    defense = defense_engine.classify(event)
                    if defense is not None:
                        defenses.append(defense)

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

    metric_data = metrics.finalize(events, defenses, segment_duration)
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
    }
    original_name = req.original_name or Path(req.video_path).name
    report = build_report(
        req=req,
        original_name=original_name,
        rounds=rounds,
        events=events,
        defenses=defenses,
        metrics=metric_data,
        tracking=tracking,
        performance=performance,
        classifier=classifier,
    )
    json_path, html_path = write_report(job_dir, report)
    events_path.write_text(json.dumps([e.to_dict() for e in events], indent=2), encoding="utf-8")

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
        )

    all_final_live_events = _live_event_payload(events, req.ruleset, live_action_trusted, limit=None)
    final_live_events = all_final_live_events[-160:]
    final_live_stats = _provisional_stats(
        all_final_live_events, found, analyzed_frames, live_action_trusted, segment_duration,
    )
    final_live_stats["diagnostics"] = _live_event_diagnostics(
        events, req.ruleset, live_action_trusted, all_final_live_events,
    )
    progress(
        "Complete", 100.0, analysis_seconds, segment_duration, manager, None, quality,
        stage="complete",
        live_events_snapshot=final_live_events,
        stats=final_live_stats,
        observation=_latest_observation(segment_end_seconds, info.width, info.height, fighter_a, fighter_b, manager),
    )
    return report
