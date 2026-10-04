from __future__ import annotations

import json
import hashlib
import logging
import math
import shutil
import tempfile
import time
from dataclasses import replace
from threading import Lock, RLock
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
from core.frame_feed import FrameFeed
from core.generalship import judge_fight
from core.fight_presence import FightPresence
from core.kit import kit_similarity
from core.ground import DownWatch
from core.round_detect import RoundDetector
from core.contact import (
    assess_selection,
    classify_contact,
    opponent_separation,
    resolve_simultaneous_attribution,
    thrown_at_opponent,
)
from core.db import save_fight
from core.defense import DefenseEngine
from core.evidence_trust import automated_evidence_trust
from core.fight_numbers import output_numbers
from core.fight_stats import normalize_outcome, summarize_fight_events
from core.action import CONFIDENCE_CEILING, CONFIDENCE_FLOOR
from core import backtrack as _backtrack
from core.build_info import stamp as build_stamp

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
#
# A tenth of the range, down from a quarter. Scored against a competitor's
# answers on the labelled Kick Light bout (tools/benchmark_labelled_fight.py),
# the quarter bar kept 8 real strikes and 10 non-events and threw away 7 real
# strikes: confidence from the rule-based detector does not separate the two
# (real 0.40-0.81, non-events 0.41-0.77). A tenth still rejects a trigger with
# essentially nothing behind it, which is all this gate is for.
ATTEMPT_CONFIDENCE = CONFIDENCE_FLOOR + 0.10 * (CONFIDENCE_CEILING - CONFIDENCE_FLOOR)

LOGGER = logging.getLogger("warrioriq.analysis")

# What this pipeline needs resident, measured on 2026-09-16 on an 8 GB RTX 5060:
# the TensorRT pose engine takes 3.85 GB (3.5 GB of it the execution context),
# SAM2's weights 0.30 GB, and SAM2 propagation peaks at 1.56 GB reserved. Below
# roughly that total the card is already spoken for, and on Windows the driver
# answers by spilling to system RAM over PCIe rather than failing - which is
# silent, and is what turned a 45-second SAM2 pass into 28 minutes.
GPU_HEADROOM_WANTED_GB = 5.5

# What an analysis holds in host memory *besides* the SAM2 frame buffer:
# the pose model and its TensorRT context, SAM2's weights, torch, and the
# decode working set. Measured by sampling RSS through a real pass on an
# RTX 5060 - 0.51 GB bare, 1.16 GB with the pose model, 1.48 GB with SAM2
# loaded, peaking at 5.33 GB against a 4.22 GB buffer, so 1.11 GB beside it.
# Rounded up for slack.
#
# This replaces a threshold of twice the buffer, which wanted 8.44 GB for a
# run that peaks at 5.33 GB. It fired on an analysis that then completed a
# 94-second clip in 103 seconds at full speed - a warning that cries wolf on
# a healthy run is worse than none, because it teaches the reader to ignore
# the one that matters.
ANALYSIS_WORKING_SET_GB = 1.5


def _sam_clip_buffer_bytes(segment_frames: int = 0, source_fps: float = 0.0) -> int:
    """What SAM2 is about to allocate in system RAM for one chunk.

    _DecodedClip holds `capacity x 3 x image_size x image_size` float32, and
    offload_video_to_cpu keeps it on the host rather than the card. SAM2's
    image_size is 1024, so this is the number that decides whether the machine
    swaps.

    The capacity is `min(chunk_frames, expected - saved)`, not chunk_frames -
    a short segment never fills a chunk. Estimating from chunk_frames alone
    overstated the buffer badly on exactly the footage where it matters:

         30s segment   needs  60 frames   0.70 GiB   estimate said 4.22
         70s segment   needs 140 frames   1.64 GiB   estimate said 4.22
        302s segment   needs 349 frames   4.09 GiB   estimate said 4.22

    A 70-second round was told to close other applications while 2.92 GB was
    free and it needed 1.64. The comment on ANALYSIS_WORKING_SET_GB already
    says why that is the worst kind of bug here - "a warning that cries wolf on
    a healthy run is worse than none, because it teaches the reader to ignore
    the one that matters" - and this was doing it again from the other side.

    Falls back to the old chunk-sized estimate when the segment is not known,
    which is the conservative direction.
    """
    frames = max(1, int(SETTINGS.sam_continuous_chunk_frames))
    if segment_frames > 0 and source_fps > 0:
        from core.sam_recovery import sam_sampling_stride

        stride = sam_sampling_stride(source_fps, int(segment_frames))
        expected = (int(segment_frames) + stride - 1) // max(1, stride)
        frames = max(1, min(frames, expected))
    return frames * 3 * 1024 * 1024 * 4


def _log_host_memory(segment_frames: int = 0, source_fps: float = 0.0) -> None:
    """Say how much system memory was free before this analysis started.

    The GPU line below could not explain a run that was thirty times slower
    than the same code on the same card with the card almost idle. The host is
    the other half: SAM2's frame buffer is gigabytes of ordinary RAM, and a
    machine that starts swapping to find it behaves exactly like a slow GPU,
    reports nothing, and cannot be told apart afterwards.
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        available = memory.available / 1024 ** 3
        buffer_gb = _sam_clip_buffer_bytes(segment_frames, source_fps) / 1024 ** 3
        wanted = buffer_gb + ANALYSIS_WORKING_SET_GB
        LOGGER.info(
            "analysis_host_memory available_gb=%.2f total_gb=%.2f sam_clip_buffer_gb=%.2f "
            "wanted_gb=%.2f",
            available, memory.total / 1024 ** 3, buffer_gb, wanted)
        if available < wanted:
            # Label the numbers as what they are. This printed `wanted` - the
            # buffer PLUS the working set - into a slot named
            # sam_clip_buffer_gb, so the log overstated the buffer by the
            # working set on every run that tripped it.
            LOGGER.warning(
                "analysis_host_memory_tight available_gb=%.2f sam_clip_buffer_gb=%.2f "
                "wanted_gb=%.2f - this analysis may push the machine into "
                "swapping, which looks like a slow GPU and is not one. Close "
                "other applications, or lower WARRIORIQ_SAM_CHUNK_FRAMES.",
                available, buffer_gb, wanted)
    except Exception as exc:                 # noqa: BLE001 - diagnostics only
        LOGGER.info("analysis_host_memory_unavailable error=%s", type(exc).__name__)


def _log_gpu_state() -> None:
    """Say what the card looked like before this analysis touched it.

    A run that is thirty times slower than the same code on the same hardware
    looks identical, from the outside, to a run that is simply heavy. Nothing
    recorded the difference, so an hour went into ruling out the code. Two
    numbers at the start settle it next time.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            LOGGER.warning("analysis_gpu_absent cuda_available=false")
            return
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        free_gb, total_gb = free / 1024 ** 3, total / 1024 ** 3
        LOGGER.info(
            "analysis_gpu_state device=%r free_gb=%.2f total_gb=%.2f in_use_gb=%.2f",
            torch.cuda.get_device_name(index), free_gb, total_gb, total_gb - free_gb)
        if free_gb < GPU_HEADROOM_WANTED_GB:
            LOGGER.warning(
                "analysis_gpu_contended free_gb=%.2f wanted_gb=%.2f in_use_gb=%.2f "
                "- another process is holding this card; expect the analysis to "
                "run far slower than normal",
                free_gb, GPU_HEADROOM_WANTED_GB, total_gb - free_gb)
    except Exception as exc:                 # never let diagnostics stop a run
        LOGGER.info("analysis_gpu_state_unavailable error=%s", type(exc).__name__)


# Where the progress bar stops describing setup and starts describing the
# per-frame pass. Below this is model loading and the SAM2 identity pass;
# from here to 98 is the action and pose pass, which is the only part whose
# rate says anything about how long the rest will take.
ANALYSIS_PHASE_START = 35.0
ANALYSIS_PHASE_SPAN = 63.0
# The bar also moves at least this often during the frame pass, whatever the
# frame count. On a machine without a GPU, 40 analysed frames took minutes,
# and the bar sat still long enough to look frozen.
PROGRESS_MAX_SILENCE_SECONDS = 5.0

from core.identity import IdentityManager, fighter_pair_similarity
from core.metrics import MetricsAccumulator
from core.preflight import Preflight
from core.preflight import probe as probe_video
from core.pose_smoothing import JointGate
from core.pose_tracker import PoseTracker, QualityController, find_initial_people
from core.report import build_report, identity_tracking, write_report
from core.rtm_pose import refine as refine_fighter_pose
from core.edgetam_recovery import build_recovery
from core.sam_recovery import nearest_guidance, sam_sampling_stride
from core.openai_identity import OpenAIIdentityReferee
from core.scoring import (
    RULESETS, collapse_simultaneous_labels, is_legal_event, normalize_ruleset,
    sport_counted_families, sport_of,
)

_PLURAL_FAMILY = {"punch": "punches", "kick": "kicks", "knee": "knees"}
from core.types import AnalysisProgress, AnalysisRequest, PersonObservation, PoseFrame, RoundSpec
from core.video import (
    SourceTimestampClock, build_round_schedule, decodable_copy, get_video_info, opencv_decodes,
    requested_segment_end, round_at_time,
)

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


# How close the punching fist must come to one of the opponent's gloves, in
# body lengths, for a "missed" punch to count as thrown: it reached the guard.
#
# Measured 2026-09-30, every "missed" punch checked by eye:
#
#     phone sparring (V16, V17, VID)  within 0.42: 7 real, 1 real but filed
#                                     under the other fighter, 1 unclear;
#                                     beyond: 1 real at 0.45, the rest (14)
#                                     a fighter standing in guard
#     Kick Light (labelled)           within 0.42: the 1 real one (0.42);
#                                     beyond: all 11 fakes, nearest 0.45
#
# A jab into the guard is scored "missed" because no clean contact was seen,
# and on handheld phone sparring those were thrown out wholesale.
MISSED_PUNCH_REACHED_GUARD = 0.42


def _punch_thrown_at_nothing(event) -> bool:
    """A punch whose hand never came near the opponent.

    Labelled by a competitor on the Kick Light bout, 11 punches the analysis
    called "missed" were proposed and not one was a real strike: they were a
    lead hand reaching out at distance, measuring or feinting, which is not a
    strike in any ruleset here. Missed kicks are left in - 3 of 5 were real.
    A missed punch whose fist reached the opponent's guard is kept: see
    MISSED_PUNCH_REACHED_GUARD. The event itself is kept either way; it is
    only not counted as an attempt.
    """
    if getattr(event, "family", None) != "punch" or getattr(event, "outcome", None) != "missed":
        return False
    guard = (getattr(event, "evidence", None) or {}).get("defender_guard_distance")
    return not (guard is not None and float(guard) <= MISSED_PUNCH_REACHED_GUARD)


def _live_attempt_reliable(event, ruleset: str | None = None) -> bool:
    """Return only identity-safe temporal attempts for the provisional live view.

    This is deliberately a lower information tier than verified fight evidence:
    it supports fighter, timestamp and broad punch/kick family only. Contact,
    target, technique side and scoring remain withheld until the release gate
    validates the complete action classifier.
    """
    return _attempt_drop_reason(event, ruleset) is None


def _live_event_payload(events: list, ruleset: str, trusted: bool, limit: int | None = 160) -> list[dict]:
    reliable = sorted(
        (
            event for event in events
            if _live_attempt_reliable(event, ruleset)
        ),
        key=lambda item: item.peak_time,
    )
    # One fighter, one instant, one action. The detector emits mutually
    # exclusive alternatives at a single frame, and the loop below groups by
    # limb-or-family - so a moment labelled both a punch and a kick used to
    # survive as two entries here while the scorecard counted it once. Same
    # rule as scoring now, from the same function.
    reliable, _simultaneous = collapse_simultaneous_labels(reliable)
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


def _fallback_buffer_needed(sam_tracks: dict, sam_was_available: bool | None) -> bool:
    """Whether to keep a converted copy of every decoded frame for SAM2 rescues.

    Continuous SAM guidance already provides the recovery path, so the buffer
    is only for when it produced nothing. And only when SAM2 could still run:
    if it failed to load - no NVIDIA GPU, the commonest reason there are no
    tracks at all - every rescue returns nothing, and the buffer was a
    full-resolution YUV-to-BGR conversion and copy of every frame of the
    video for it, undoing the grab()/retrieve() split in the frame loop.
    Measured over 1200 frames of 1080p60 phone footage that conversion is
    most of 5.28 s against 2.05 s for the decode alone. `None` means SAM2 was
    never tried (continuous sweep switched off) and may yet load.
    """
    return bool(SETTINGS.sam_recovery_enabled and not sam_tracks and sam_was_available is not False)


def _attempt_drop_reason(event, ruleset: str | None = None) -> str | None:
    """Why _live_attempt_reliable turned a candidate down, or None if it did not.

    The same checks in the same order, named, so a fight that shows no strikes
    can say what happened to the ones the detector proposed.
    """
    if not bool(getattr(event, "attempted", True)):
        return "not_a_strike"
    # Knees included. They were excluded here while knee_attempts was still
    # computed downstream, so the report carried a knee column that could
    # never be anything but zero while real knees were discarded - five of
    # twelve events on one bout. At this tier the claim is only "a limb was
    # thrown", and a knee is evidenced exactly as a punch is.
    family = getattr(event, "family", None)
    if family not in {"punch", "kick", "knee"}:
        return "not_a_strike"
    # A sport with no kicks at all has no leg strikes to find: in boxing every
    # kick or knee the detector proposes is footwork. On 30 s of handheld
    # boxing sparring (dataset/regression/identity_phone, V16) 7 of the 9
    # counted strikes were kicks and knees, and none of them were, checked by
    # eye. Knees that are merely illegal - Kick Light, K1 variants - are still
    # thrown and still counted; this is only for a sport without kicks.
    if ruleset and family in {"kick", "knee"} and not RULESETS[normalize_ruleset(ruleset)].allow_kick:
        return "not_in_this_sport"
    # And more generally, only what the report for this sport counts
    # (core/sport_policy.py): a knee in taekwondo was listed live as an attempt
    # and then left out of every number in the report.
    if ruleset and _PLURAL_FAMILY.get(family) not in sport_counted_families(
            sport_of(normalize_ruleset(ruleset))):
        return "not_in_this_sport"
    peak = float(getattr(event, "peak_time", -1.0))
    if not math.isfinite(peak) or peak < 0.0:
        return "not_a_strike"
    if float(getattr(event, "confidence", 0.0)) < ATTEMPT_CONFIDENCE:
        return "too_faint"
    if _punch_thrown_at_nothing(event):
        return "thrown_at_nothing"
    if (float(event.metadata.get("attacker_identity_confidence", 1.0)) < 0.76
            or float(event.metadata.get("opponent_identity_confidence", 1.0)) < 0.76):
        return "unsure_who_was_who"
    return None


def _live_event_diagnostics(events: list, ruleset: str, trusted: bool, emitted: list[dict]) -> dict:
    dropped: dict[str, int] = {}
    for event in events:
        reason = _attempt_drop_reason(event, ruleset)
        if reason is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
    return {
        # What happened to every strike the detector proposed. See
        # _attempt_drop_reason; the result page shows it when nothing counted.
        "dropped": dropped,
        "candidate_events_seen": len(events),
        "identity_safe_attempts": sum(_live_attempt_reliable(event, ruleset) for event in events),
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


_POSE_MODEL_LOCK = Lock()
_ANALYSIS_LOCK = RLock()


@lru_cache(maxsize=1)
def _cached_pose_tracker() -> PoseTracker:
    return PoseTracker()


def get_pose_tracker() -> PoseTracker:
    """Reuse model weights without racing first-time model initialization."""
    with _POSE_MODEL_LOCK:
        return _cached_pose_tracker()


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


def _frame_breakdown(steps: dict[str, float], total: float, frames: int) -> dict[str, float] | None:
    """Seconds per analysed frame for each named step, plus what is left."""
    if not frames:
        return None
    named = {name: round(seconds / frames, 4) for name, seconds in steps.items()}
    named["other"] = round(max(0.0, total - sum(steps.values())) / frames, 4)
    return named


class SeedCheckFailed(RuntimeError):
    """The forward pass reached the selection frame with A and B not where the
    person drew them, so the identities carried back before it are wrong."""


def _for_metrics(obs: PersonObservation | None) -> PersonObservation | None:
    """What the metrics read from an observation, without the heavy parts.

    Metrics are now measured after the fight is classified, so every analysed
    frame's pair is kept until then; the appearance vectors are not needed and
    would be most of the memory.
    """
    if obs is None:
        return None
    return PersonObservation(
        track_id=obs.track_id, box=np.asarray(obs.box, dtype=np.float32).copy(),
        confidence=float(obs.confidence),
        keypoints=None if obs.keypoints is None else np.asarray(obs.keypoints, dtype=np.float32).copy(),
        keypoint_conf=None if obs.keypoint_conf is None else np.asarray(obs.keypoint_conf, dtype=np.float32).copy(),
    )


def _seed_frame_kit(video_path: str, frame_index: int, box_a, box_b) -> dict | None:
    """Kit similarity on the frame the person chose, with their boxes."""
    capture = cv2.VideoCapture(video_path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    return kit_similarity(frame, box_a, box_b) if ok and frame is not None else None


def _measure_footage(video_path: str, pose_tracker, start_seconds: float, end_seconds: float | None):
    """Measure the recording before choosing how to look at it. Never fatal."""
    try:
        return probe_video(video_path, pose_tracker.model, start_seconds=start_seconds, end_seconds=end_seconds)
    except Exception as error:  # noqa: BLE001 - a probe must never block a paid run
        preflight = Preflight()
        preflight.warnings.append("The video could not be measured before analysis: %s" % error)
        return preflight


def _analysed_span(info, start_seconds: float, end_seconds: float, seed_seconds: float,
                   handoff: dict | None, reason: str | None) -> dict:
    """What part of the video the report describes, stated in seconds."""
    excluded_before = max(0.0, float(start_seconds))
    return {
        "start_seconds": round(float(start_seconds), 3),
        "end_seconds": round(float(end_seconds), 3),
        "video_duration_seconds": round(float(info.duration), 3),
        "selection_seconds": round(float(seed_seconds), 3),
        # Whole seconds, because that is the resolution the report prints at:
        # a span that starts 0.4 s in still reads 0:00, and it is not called
        # "the whole video" unless it really does start inside that first
        # second and run to the end.
        "whole_video": excluded_before < 1.0 and float(info.duration) - float(end_seconds) < 1.0,
        "excluded_start_seconds": round(excluded_before, 3),
        "excluded_reason": reason if excluded_before >= 1.0 else None,
        "excluded_reason_text": (_backtrack.REASONS.get(reason) if reason and excluded_before >= 1.0 else None),
        "backtrack": handoff,
    }


class UnreadableVideo(RuntimeError):
    """No decoder on the analysis machine can read this video's frames."""


def analyze(req: AnalysisRequest, progress_callback: ProgressCallback | None = None) -> dict:
    # BoT-SORT state belongs to one run for its entire lifetime. Per-frame
    # locks would still let a second job reset identities between frames.
    with _ANALYSIS_LOCK:
        if opencv_decodes(req.video_path):
            return _analyze_from_seed(req, progress_callback)
        # OpenCV cannot read this file at all - an AV1 WebM is the usual one:
        # it opens, reports its frames and decodes none. Rather than fail a
        # real fight, analyse an H.264 copy with the same frames and timing,
        # made here and deleted afterwards.
        if progress_callback is not None:
            progress_callback(AnalysisProgress(
                percent=0.2, message="Converting the video so it can be read",
                elapsed_seconds=0.0, processed_video_seconds=0.0, speed=0.0, eta_seconds=None,
                stage="preparing").to_dict())
        scratch = Path(tempfile.mkdtemp(prefix=".decodable-", dir=str(Path(req.output_dir).parent)
                                        if req.output_dir else None))
        try:
            copy = decodable_copy(req.video_path, scratch / "decodable.mp4")
            if copy is None or not opencv_decodes(copy):
                raise UnreadableVideo(
                    "The analysis machine cannot decode this video's format, and no converter is "
                    "installed on it.")
            LOGGER.info("analysing_decodable_copy original=%s", Path(req.video_path).name)
            return _analyze_from_seed(
                replace(req, video_path=str(copy),
                        original_name=req.original_name or Path(req.video_path).name),
                progress_callback)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _analyze_from_seed(req: AnalysisRequest, progress_callback: ProgressCallback | None) -> dict:
    """Analyse from the requested start, seeding identity at the chosen frame.

    The person draws the two boxes on one frame. Everything before that frame
    is reached by following both fighters backwards from it (core/backtrack.py)
    and the result is checked when the forward pass arrives back at it.

    The analysis always starts at the requested start (0:00 unless a later one
    was asked for). It used to restart at the chosen frame whenever the
    backward pass stopped short or the check failed, which silently cut the
    opening off the report (QA, 2026-10-04: 0:08 of a 0:10 clip analysed as
    "Analysis complete"). Now, when the fighters cannot be followed all the
    way back, the forward pass still starts at the beginning and the check at
    the chosen frame decides whether its identities can be trusted; if not,
    the report says so rather than leaving footage out.
    """
    wrapper_started = time.perf_counter()
    seed_seconds = req.selection_seconds
    if seed_seconds is None:
        return _analyze(req, progress_callback)
    info = get_video_info(req.video_path)
    fps = float(info.fps) if info.fps > 0 else 30.0
    requested_start = max(0.0, min(float(req.start_seconds), max(0.0, info.duration - 0.001)))
    seed_seconds = max(requested_start, min(float(seed_seconds), max(0.0, info.duration - 0.001)))
    seed_frame = int(round(seed_seconds * fps))
    start_frame = int(round(requested_start * fps))
    if seed_frame - start_frame < _backtrack.MIN_BACKTRACK_SECONDS * fps:
        return _analyze(replace(req, start_seconds=requested_start, selection_seconds=None),
                        progress_callback, seed_seconds=seed_seconds)

    def report_progress(fraction: float) -> None:
        if progress_callback is None:
            return
        progress_callback(AnalysisProgress(
            percent=0.5 + 0.5 * max(0.0, min(1.0, fraction)),
            message="Following both fighters back to the start of the video",
            elapsed_seconds=0.0, processed_video_seconds=0.0, speed=0.0, eta_seconds=None,
            stage="tracking", video_duration_seconds=float(info.duration),
            analysed_from_seconds=requested_start,
        ).to_dict())

    report_progress(0.0)
    pose_tracker = get_pose_tracker()
    seed_kit = _seed_frame_kit(req.video_path, seed_frame, req.fighter_a_box, req.fighter_b_box)
    # The backward pass has to see people the way the forward pass will, so it
    # uses the same measured inference size - and the forward pass reuses the
    # measurement instead of taking it again.
    measured = _measure_footage(req.video_path, pose_tracker, requested_start, None)
    imgsz = QualityController(
        info.fps, info.width, info.height,
        measured_imgsz=measured.recommended_inference_size if measured.measured else None).imgsz
    try:
        handoff = _backtrack.backtrack(
            req.video_path, pose_tracker, fps, start_frame, seed_frame,
            req.fighter_a_box, req.fighter_b_box,
            imgsz,
            find_initial=find_initial_people,
            workdir=Path(req.output_dir) if req.output_dir else None,
            progress=report_progress,
        )
    except Exception as error:                                      # noqa: BLE001
        # The backward pass is an addition; it must never cost the analysis.
        LOGGER.warning("backtrack_failed error=%s detail=%s", type(error).__name__, str(error)[:200])
        handoff = _backtrack.Handoff(seed_frame=seed_frame, requested_start_frame=start_frame,
                                     frame=seed_frame, reason="unavailable")
    handoff_record = handoff.as_dict(fps)
    if handoff.moved:
        # Seeded where the backward pass got to. That is the requested start
        # whenever it succeeded; when it stopped short the forward pass still
        # begins there, at the start, with the boxes from the earliest frame
        # both fighters were followed to as its best guess.
        start_boxes = (list(handoff.a_box), list(handoff.b_box))
        forward_start = handoff.frame / fps if handoff.reason is None else requested_start
    else:
        start_boxes = (list(req.fighter_a_box), list(req.fighter_b_box))
        forward_start = requested_start
    forward = replace(
        req, start_seconds=forward_start, selection_seconds=None,
        fighter_a_box=start_boxes[0], fighter_b_box=start_boxes[1])
    seed_check = {"frame": seed_frame, "a_box": list(req.fighter_a_box), "b_box": list(req.fighter_b_box),
                  "window_frames": int(round(1.5 * fps))}
    seed_pair = (handoff.seed_a, handoff.seed_b) if handoff.seed_a is not None and handoff.seed_b is not None else None
    # The reason is kept for the record only: no footage is excluded any more,
    # so it is not passed on as an exclusion.
    return _analyze(forward, progress_callback, seed_seconds=seed_seconds,
                    handoff=handoff_record, seed_check=seed_check,
                    seed_pair=seed_pair, measured_footage=measured, seed_kit=seed_kit,
                    budget_spent_before=time.perf_counter() - wrapper_started)


def _analyze(req: AnalysisRequest, progress_callback: ProgressCallback | None = None, *,
             seed_seconds: float | None = None, handoff: dict | None = None,
             excluded_reason: str | None = None, seed_check: dict | None = None,
             seed_pair: tuple | None = None, measured_footage=None,
             seed_kit: dict | None = None, budget_spent_before: float = 0.0) -> dict:
    info = get_video_info(req.video_path)
    _validate_request(req, info.duration)
    if seed_seconds is None:
        seed_seconds = req.start_seconds
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

    # How much of the card was already taken before this run allocated anything.
    #
    # Windows falls back to system RAM over PCIe rather than erroring when a
    # CUDA allocation will not fit, which produces a large silent slowdown with
    # no message anywhere. An analysis that took 28 minutes to reach 30% was
    # eventually traced to exactly this, after five other causes had been ruled
    # out by measurement, and the note written at the time said the one thing
    # that would have answered it immediately was a free-VRAM reading at the
    # start of the run. Nothing recorded one. This does.
    #
    # It is a diagnostic, not a gate: a busy card is the user's business and
    # refusing to run would be worse than running slowly. But when a report
    # shows a missed budget, this says in one number whether the code was slow
    # or the machine was busy.
    vram_free_at_start = None
    if torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            vram_free_at_start = {
                "free_gb": round(free_bytes / 2 ** 30, 2),
                "total_gb": round(total_bytes / 2 ** 30, 2),
                "in_use_fraction": round(1.0 - free_bytes / max(1, total_bytes), 3),
            }
        except Exception:                                        # noqa: BLE001
            vram_free_at_start = None

    job_dir = Path(req.output_dir) if req.output_dir else OUTPUTS / req.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = job_dir / "tracking.jsonl"
    events_path = job_dir / "events.json"

    # ETA follows the same weighted phase progress as the visible progress bar.
    # The old calculation used "processed video seconds", which reaches the end
    # during SAM and then starts again during pose analysis, causing huge jumps.
    #
    # Weighting by percent removed the jumps and left a subtler error: percent
    # covers two unrelated workloads. Everything below ANALYSIS_PHASE_START is
    # model loading and the SAM2 identity pass; everything above it is the
    # per-frame action and pose pass. Dividing elapsed by overall percent
    # therefore predicted the second from the rate of the first. Observed on a
    # 1:10 clip: 28m17s at 30.3%, still inside the SAM2 pass, reported as
    # "ETA 62m44s" - a number extrapolated from a workload that was about to
    # end. The EWMA below cannot rescue that; its comment calls the warmup
    # short, and a full SAM2 pass over the clip is not short.
    #
    # So: no estimate at all until the frame pass has started and measured
    # itself, then extrapolate from that phase alone. "Calculating..." for the
    # warmup is worth more than a confident wrong number.
    estimated_total_seconds: float | None = None
    analysis_phase_started_at: float | None = None
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
        nonlocal estimated_total_seconds, analysis_phase_started_at
        if progress_callback is None:
            return
        speed = processed / elapsed if elapsed > 0.0 else 0.0
        bounded_percent = float(max(0.0, min(100.0, percent)))
        if analysis_phase_started_at is None and bounded_percent >= ANALYSIS_PHASE_START:
            analysis_phase_started_at = elapsed
        phase_elapsed = (
            elapsed - analysis_phase_started_at if analysis_phase_started_at is not None else 0.0)
        phase_done = bounded_percent - ANALYSIS_PHASE_START
        if bounded_percent >= 100.0:
            eta = 0.0
        # Both guards matter on the first emit after the boundary, where a
        # fraction of a percent over a fraction of a second projects hours.
        elif phase_done >= 1.0 and phase_elapsed >= 2.0:
            projected_total = elapsed + (100.0 - bounded_percent) / phase_done * phase_elapsed
            if estimated_total_seconds is None:
                estimated_total_seconds = projected_total
            else:
                # A conservative EWMA absorbs frame complexity changes without
                # presenting impossible minute jumps.
                estimated_total_seconds = 0.82 * estimated_total_seconds + 0.18 * projected_total
            estimated_total_seconds = max(elapsed, estimated_total_seconds)
            eta = max(0.0, estimated_total_seconds - elapsed)
        else:
            # Warmup and the SAM2 pass. Nothing here measures the frame pass.
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
            analysed_from_seconds=float(req.start_seconds),
            live_event_mode="validated_actions" if live_action_trusted else "observed_attempts",
            live_events=live_events_snapshot or [],
            provisional_stats=stats or {},
            latest_observation=observation,
        )
        progress_callback(payload.to_dict())

    # Honest wall timer: model load/warmup is part of the user's wait.
    wall_start = time.perf_counter()
    _log_gpu_state()
    _log_host_memory(max(0, end_frame - start_frame), float(info.fps))
    # Not "GPU models": the machine running this may have no GPU.
    progress("Loading analysis models", 0.0, 0.0, 0.0)

    pose_tracker = get_pose_tracker()
    # SAM2 unless WARRIORIQ_SAM_BACKEND says otherwise. Both backends return
    # the same thing from track_segment and mean the same thing by it.
    sam_recovery = build_recovery()
    action_engine = ActionEngine()
    defense_engine = DefenseEngine()
    joint_gate = JointGate() if SETTINGS.pose_gate_enabled else None
    metrics = MetricsAccumulator(info.width, info.height)
    # Measure the footage before choosing how to look at it. A second of
    # sampling buys the inference size, and the same numbers tell the person who
    # filmed it what to do differently. Never fatal: if the probe cannot read
    # the file the analysis proceeds on the old resolution rule and says so.
    preflight = measured_footage if measured_footage is not None else _measure_footage(
        req.video_path, pose_tracker, req.start_seconds, segment_end_seconds)
    quality = QualityController(
        info.fps, info.width, info.height,
        measured_imgsz=preflight.recommended_inference_size if preflight.measured else None)
    classifier = {
        "action_classifier": "warrioriq_temporal_model" if action_engine.temporal.available else "multi_frame_temporal_rules",
        "custom_temporal_checkpoint_loaded": bool(action_engine.temporal.available),
        "temporal_architecture": action_engine.temporal.architecture,
        "temporal_validation": action_engine.temporal.validation,
        "temporal_runtime": action_engine.temporal.diagnostics(),
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
    source_clock = SourceTimestampClock(info.fps)
    first_seconds = source_clock.seconds(start_frame, float(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # Warm model before tracker initialization. This also downloads the model on
    # first use if Ultralytics has not cached it yet.
    pose_tracker.warmup(first_frame)
    # The model is cached across jobs for speed; tracker identities are not.
    pose_tracker.reset_tracking()

    # Watch the seconds before the round for their motion history alone, so
    # the identity guards are not blind for the opening of every analysis.
    # See IdentityManager.prime_track_history. Nothing here is scored: these
    # frames are outside the round and never reach metrics, events or output.
    warm_samples: list[tuple[int, list]] = []
    warm_frames = int(round(SETTINGS.identity_warmup_seconds * info.fps))
    warm_start = max(0, start_frame - warm_frames)
    if warm_frames > 0 and warm_start < start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm_start)
        warm_index = warm_start
        while warm_index < start_frame:
            warm_ok, warm_frame = cap.read()
            if not warm_ok or warm_frame is None:
                break
            if (warm_index - warm_start) % max(1, quality.stride) == 0:
                warm_samples.append((warm_index, pose_tracker.track(warm_frame, quality.imgsz)))
            warm_index += 1
        # Back to the frame the user actually selected on.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        warm_ok, rewound = cap.read()
        if not warm_ok or rewound is None:
            cap.release()
            raise RuntimeError("Could not re-read the fight-start frame after warm-up")
        first_frame = rewound

    # Start BoT-SORT on exactly the same frame the user used for A/B selection.
    first_people = pose_tracker.track(first_frame, quality.imgsz)
    initial_a, initial_b, iou_a, iou_b = find_initial_people(
        np.asarray(req.fighter_a_box, dtype=np.float32),
        np.asarray(req.fighter_b_box, dtype=np.float32),
        first_people,
        first_frame,
    )
    manager = IdentityManager(initial_a, initial_b, start_frame, source_fps=info.fps)
    if seed_pair is not None:
        # Seeded earlier than the frame the person chose, after following the
        # fighters back from it. The identity anchors are still taken from the
        # chosen frame - "what the fighter looked like where the box was
        # drawn" - because that is the frame picked to show them clearly, and
        # anchoring on a clinch at the hand-off cost a real bout most of one
        # fighter's coverage.
        for state, seed in ((manager.a, seed_pair[0]), (manager.b, seed_pair[1])):
            manager.adopt_anchor(state, seed)
    # Can these two be told apart in this video at all? Asked once, at the
    # start, because no amount of work downstream recovers from "no".
    # On the frame the person chose when the fighters were followed back from
    # it: that is the frame they picked to show the two apart.
    # The kit comparison is made on the frame the person chose, with the boxes
    # they drew - the same frame and boxes the selection page warned about, so
    # the report cannot disagree with the warning. See core/kit.py.
    kit = seed_kit if seed_kit is not None else kit_similarity(
        first_frame, req.fighter_a_box, req.fighter_b_box)
    histogram_similarity = fighter_pair_similarity(*(seed_pair or (initial_a, initial_b)))
    pair_similarity = kit["similarity"] if kit is not None else histogram_similarity
    manager.prime_track_history(warm_samples)
    canonical_a_box = [float(value) for value in initial_a.box]
    canonical_b_box = [float(value) for value in initial_b.box]
    identity_referee = OpenAIIdentityReferee(req.openai_identity_recovery, first_frame, canonical_a_box, canonical_b_box)

    progress("Following both fighters", 1.0, time.perf_counter() - wall_start, 0.0, manager, None, quality, stage="tracking")
    sweep_start = time.perf_counter()
    sam_tracks = sam_recovery.track_segment(
        req.video_path,
        start_frame,
        end_frame,
        info.fps,
        canonical_a_box,
        canonical_b_box,
        progress_callback=lambda completed, total: progress(
            "Following both fighters",
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
    sam_sweep_seconds = time.perf_counter() - sweep_start
    # The bar reserves 0-35% for SAM2. Without a GPU SAM2 is skipped at once,
    # and the bar used to sit at 1% until the first frame-pass update, then
    # jump past 50%. Mark the start of the frame pass as soon as it begins.
    progress("Analyzing fight", ANALYSIS_PHASE_START, time.perf_counter() - wall_start, 0.0,
             manager, None, quality, stage="analysis")
    # SAM2 reads the segment independently. Resume the pose pass immediately
    # after the already-consumed selection frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 1)
    pose_pass_start = time.perf_counter()
    # Where each analysed frame's time goes, so the next speed-up is chosen by
    # measurement rather than guessed at. Pose model and appearance are timed
    # inside PoseTracker.track; "other" is whatever the named steps leave.
    pose_tracker.timing = {"pose_model": 0.0, "appearance": 0.0}
    step_seconds = {"reading_video": 0.0, "missing_fighter_search": 0.0,
                    "identity": 0.0, "joint_refinement": 0.0}

    fallback_buffer_enabled = _fallback_buffer_needed(sam_tracks, sam_was_available)
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
    # Recoveries from the tracker's own prediction rather than the sweep's,
    # counted apart so the two can be told from each other in the report.
    expected_pose_recoveries = {"A": 0, "B": 0}
    coverage_windows: dict[int, dict[str, int]] = {}
    sam_stride = sam_sampling_stride(info.fps, end_frame - start_frame)
    current_frame = start_frame
    last_progress_emit = 0
    last_progress_emit_at = time.perf_counter()
    # The start frame is already analyzed above. Schedule the next expensive
    # inference at the adaptive tracking stride instead of analyzing the very
    # next source frame again.
    next_inference_frame = start_frame + max(1, quality.stride)
    current_imgsz = quality.imgsz
    decoded_seconds = first_seconds

    # Since the first frame has already been consumed, process it through the
    # same downstream path before entering the read loop.
    pending = [(start_frame, first_frame, first_people, initial_a, initial_b)]

    out_of_range_actions = 0
    observed_separations: list[float] = []
    travel = {"A": _Travel(), "B": _Travel()}
    fighter_finder = FighterFinder()
    down_watch = DownWatch()
    round_detector = RoundDetector()
    # Is this stretch two people fighting? Decided per window from the frames
    # below; metrics and strike counts are then taken from fight footage only.
    # See core/fight_presence.py.
    presence = FightPresence(req.start_seconds, segment_end_seconds)
    metric_frames: list[tuple[float, int | None, PersonObservation | None, PersonObservation | None]] = []
    tracking_file = tracking_path.open("w", encoding="utf-8") if SETTINGS.save_tracking_jsonl else None
    # Decoding runs a step ahead on a helper thread; see core/frame_feed.py.
    feed = FrameFeed(
        cap, first_source_frame=start_frame + 1, next_inference_frame=next_inference_frame,
        retrieve_all=fallback_buffer_enabled,
        history_step=max(1, round(info.fps)) if identity_referee.enabled else None,
        next_history_frame=next_ai_history_frame, ahead=SETTINGS.decode_ahead)

    try:
        while current_frame < end_frame:
            if pending:
                source_frame, frame, people, fighter_a, fighter_b = pending.pop(0)
            else:
                source_frame = current_frame + 1
                # Decode, but do not pay to *convert* a frame nobody wants.
                #
                # At the 6 fps floor on 60 fps phone footage this loop walks
                # nine frames for every one it analyses, and it used to call
                # read() on all of them - which decodes the packet and then
                # converts YUV to BGR into a fresh 1920x1080x3 buffer that is
                # dropped on the next line. grab() stops after the decode;
                # retrieve() does the conversion, and is only worth calling for
                # a frame something is going to look at.
                #
                # Measured over 1200 frames, stride 9:
                #
                #     phone 1080p60     read 5.28 s -> grab 2.05 s   -61%
                #     fight 1 480x220   read 0.19 s -> grab 0.13 s   negligible
                #
                # So this is a high-resolution win and nearly nothing on the
                # old broadcast captures, which is the right shape: the cost it
                # removes is per pixel, and SD frames have few.
                read_started = time.perf_counter()
                fed = feed.next()
                if fed is None:
                    break
                if fed.source_frame != source_frame:
                    raise RuntimeError("frame feed out of step with the analysis loop")
                current_frame = source_frame
                pts_ms = fed.pts_ms
                decoded_seconds = source_clock.seconds(source_frame, pts_ms)
                # Three things can want this frame, and the cheap check has to
                # consider all of them or a feature silently stops being fed.
                wants_history = bool(
                    identity_referee.enabled and source_frame >= next_ai_history_frame)
                wants_inference = source_frame >= next_inference_frame
                if not (fallback_buffer_enabled or wants_history or wants_inference):
                    step_seconds["reading_video"] += time.perf_counter() - read_started
                    continue
                frame = fed.frame
                step_seconds["reading_video"] += time.perf_counter() - read_started
                if frame is None:
                    break
                if fallback_buffer_enabled:
                    frame_buffer.append((source_frame, frame.copy()))
                if wants_history:
                    ai_history.append(cv2.resize(frame, (640, max(1, round(frame.shape[0] * 640 / frame.shape[1])))))
                    next_ai_history_frame = source_frame + max(1, round(info.fps))

                if not wants_inference:
                    continue

                seconds = decoded_seconds
                spec = round_at_time(rounds, seconds)
                active_selected_round = bool(spec and spec.selected)
                base_stride = quality.stride
                # Preserve identity through breaks/non-selected rounds at about
                # 5 FPS without spending full action-analysis budget.
                inference_stride = base_stride if active_selected_round else max(base_stride, round(info.fps / 5.0))
                next_inference_frame = source_frame + max(1, inference_stride)
                feed.allow_through(next_inference_frame)

                people = pose_tracker.track(frame, current_imgsz)
                guidance = nearest_guidance(sam_tracks, source_frame, sam_stride)
                search_started = time.perf_counter()
                focused = pose_tracker.recover_from_guidance(frame, guidance, people)
                for observation in focused:
                    if observation.track_id == -1001:
                        guided_pose_recoveries["A"] += 1
                    elif observation.track_id == -1002:
                        guided_pose_recoveries["B"] += 1
                people.extend(focused)
                # And where the sweep said nothing, look where the tracker
                # expects them anyway.
                #
                # The sweep only covers a fraction of the frames - 64 of 620 on
                # one fight - so for most of a bout the crop was never taken.
                # Meanwhile the detector was missing fighter B on 279 of 713
                # frames of fight 1, and 238 of those had nobody detected where
                # B was expected: a fighter too small in the network input to
                # find, with a perfectly good prediction of where to look.
                #
                # recover_from_guidance already refuses any fighter an existing
                # detection overlaps, so this costs a crop only on the frames
                # where somebody is actually missing.
                expected = pose_tracker.recover_from_guidance(
                    frame, manager.expected_boxes(), people)
                for observation in expected:
                    if observation.track_id == -1001:
                        expected_pose_recoveries["A"] += 1
                    elif observation.track_id == -1002:
                        expected_pose_recoveries["B"] += 1
                people.extend(expected)
                identity_started = time.perf_counter()
                step_seconds["missing_fighter_search"] += identity_started - search_started
                fighter_a, fighter_b = manager.update(people, source_frame, sam_guidance=guidance)
                joints_started = time.perf_counter()
                step_seconds["identity"] += joints_started - identity_started
                # Joints only, and only for the two fighters, only after
                # identity has already chosen them. See core/rtm_pose.py.
                refine_fighter_pose(frame, [fighter_a, fighter_b])
                step_seconds["joint_refinement"] += time.perf_counter() - joints_started
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

            seconds = first_seconds if source_frame == start_frame else decoded_seconds
            spec = round_at_time(rounds, seconds)
            round_number = spec.number if spec else None
            active_selected_round = bool(spec and spec.selected)

            # First frame did not run manager.update because the initial lock is
            # itself the update.
            if source_frame == start_frame:
                manager.a.identity_confidence = 1.0
                manager.b.identity_confidence = 1.0

            # Arriving back at the frame the person picked, after following the
            # fighters from earlier in the video: A has to be on the box they
            # drew for A, and B on B's. Anything else means the identities
            # carried back before it are not the ones they chose.
            if seed_check is not None and not seed_check.get("done") and source_frame >= seed_check["frame"]:
                verdict = _backtrack.seed_verdict(
                    fighter_a, fighter_b, seed_check["a_box"], seed_check["b_box"])
                seed_check.setdefault("verdicts", []).append(verdict)
                # Recorded, never a reason to stop: the footage before the
                # chosen frame stays in the report, and a failed check marks
                # its identities as unconfirmed instead (see
                # identity_seed_confirmed below).
                if verdict == "match":
                    seed_check["done"] = True
                elif verdict == "swapped":
                    seed_check["done"] = True
                    seed_check["failed"] = "swapped"
                elif source_frame - seed_check["frame"] >= seed_check["window_frames"]:
                    seed_check["done"] = True
                    if "partial" not in seed_check["verdicts"]:
                        seed_check["failed"] = "unconfirmed"

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

            # Between identity and everything that measures. The metrics, the
            # action engine and the defence engine all read these keypoints, so
            # one collapsed skeleton corrupts guard, balance, centre and travel
            # at once; identity above keeps the raw observation, because its
            # job is deciding who this is rather than measuring them.
            if joint_gate is not None:
                joint_gate.apply("A", seconds, fighter_a)
                joint_gate.apply("B", seconds, fighter_b)

            defense_engine.update_pose("A", source_frame, seconds, fighter_a)
            defense_engine.update_pose("B", source_frame, seconds, fighter_b)

            presence.observe(seconds, fighter_a, fighter_b, people)
            if active_selected_round:
                active_analyzed_frames += 1
                # Measured once the whole fight is classified, so footage that
                # is not two people fighting never reaches a metric.
                metric_frames.append((seconds, round_number, _for_metrics(fighter_a), _for_metrics(fighter_b)))

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
            else:
                # Do not join a pre-break extension to post-break retraction.
                action_engine.interrupt("A")
                action_engine.interrupt("B")

            if live_action_trusted and not action_engine.temporal.available:
                live_action_trusted = False
                classifier["custom_temporal_checkpoint_loaded"] = False
            classifier["temporal_runtime"] = action_engine.temporal.diagnostics()

            # Fighters move. Someone at ringside does not, and that is the
            # difference a separation test cannot see when the wrong two
            # people happen to be standing next to each other.
            travel["A"].add(seconds, fighter_a)
            travel["B"].add(seconds, fighter_b)
            fighter_finder.observe(seconds, people)
            fighter_finder.observe_selected(fighter_a, fighter_b)
            down_watch.observe(seconds, people, fighter_a, fighter_b)
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
            # One planned decision rather than a controller chasing machine
            # load: measure what a frame really costs here, then fix the stride
            # for the rest of the run so the analysis fits inside the video's
            # own length. Deliberately separate from maybe_adjust, which is off
            # by default because continuous adaptation makes identical fights
            # follow different frame paths.
            if SETTINGS.hard_realtime_budget:
                quality.plan_for_budget(
                    analyzed_frames, processed_seconds,
                    time.perf_counter() - pose_pass_start, segment_duration,
                    overhead_seconds=(pose_pass_start - wall_start) + float(budget_spent_before))
            current_imgsz = quality.imgsz

            if (analyzed_frames - last_progress_emit >= SETTINGS.progress_interval_frames
                    or (analyzed_frames > last_progress_emit
                        and time.perf_counter() - last_progress_emit_at >= PROGRESS_MAX_SILENCE_SECONDS)):
                last_progress_emit = analyzed_frames
                last_progress_emit_at = time.perf_counter()
                percent = ANALYSIS_PHASE_START + ANALYSIS_PHASE_SPAN * processed_seconds / segment_duration
                # Only strikes from windows already judged to be fight footage:
                # a strike shown live and then dropped from the report would be
                # the live page contradicting the report again.
                decided = presence.decided_until(seconds)
                live_segments = presence.segments()
                live_candidates = [
                    event for event in events
                    if float(event.peak_time) < decided
                    and presence.is_fight(float(event.peak_time), live_segments)]
                all_live_event_data = _live_event_payload(live_candidates, req.ruleset, live_action_trusted, limit=None)
                live_event_data = all_live_event_data[-160:]
                live_stats = _provisional_stats(
                    all_live_event_data, found, analyzed_frames, live_action_trusted, processed_seconds,
                )
                live_stats["diagnostics"] = _live_event_diagnostics(
                    live_candidates, req.ruleset, live_action_trusted, all_live_event_data,
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
        feed.close()
        cap.release()
        if tracking_file is not None:
            tracking_file.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if seed_check is not None and not seed_check.get("done"):
        if "partial" not in seed_check.get("verdicts", []):
            seed_check["failed"] = "never_reached"

    # Fight footage only, from here on. Interviews, graphics and a selected
    # pair that never engaged are left out of every measurement and count.
    fight_footage = presence.summary()
    fight_segments = [tuple(span) for span in fight_footage["segments"]]
    previous_kept = None
    for frame_seconds, frame_round, obs_a, obs_b in metric_frames:
        if not presence.is_fight(frame_seconds, fight_segments):
            previous_kept = None
            continue
        if previous_kept is None:
            # A new stretch of fight footage: no movement across the gap.
            for side in ("A", "B"):
                metrics.last_center[side] = None
                metrics.last_time[side] = None
        metrics.update("A", frame_seconds, frame_round, obs_a, obs_b)
        metrics.update("B", frame_seconds, frame_round, obs_b, obs_a)
        previous_kept = frame_seconds
    events_before = len(events)
    events = [event for event in events if presence.is_fight(float(event.peak_time), fight_segments)]
    defenses = [d for d in defenses if presence.is_fight(float(d.time_seconds), fight_segments)]
    fight_footage["actions_left_out"] = events_before - len(events)
    # Rates are per minute of fight footage, not per minute of video.
    measured_duration = max(1.0, float(fight_footage["fight_seconds"])) if fight_footage["fight_seconds"] > 0 else segment_duration

    # The whole wait, including the backward identity pass and any rerun.
    analysis_seconds = time.perf_counter() - wall_start + float(budget_spent_before)
    frame_pass_seconds = time.perf_counter() - pose_pass_start
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
    # Both fighters move when one of them is hit. Decide which of the two was
    # the aggressor before anything downstream counts, scores or reports them.
    events, defender_actions = resolve_simultaneous_attribution(events)

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
        # And the setup says what was found, not what the form posted: the
        # upload page no longer asks for a format, so everything that reads
        # setup.round_count (the fight library, the round-consistency metric)
        # would otherwise say "1 round" for every fight.
        req.round_count = len(rounds)
        req.round_duration_seconds = round(
            sum(spec.end_seconds - spec.start_seconds for spec in rounds) / len(rounds), 1)

    all_final_live_events = _live_event_payload(events, req.ruleset, live_action_trusted, limit=None)
    final_live_stats = _provisional_stats(
        all_final_live_events, found, analyzed_frames, live_action_trusted, measured_duration,
    )
    final_live_stats["diagnostics"] = _live_event_diagnostics(
        events, req.ruleset, live_action_trusted, all_final_live_events,
    )
    # When each fighter threw, from the same counted strikes as the totals.
    # See core/fight_numbers.py.
    round_bounds = [
        (spec.number, max(spec.start_seconds, req.start_seconds), min(spec.end_seconds, segment_end_seconds))
        for spec in rounds
        if spec.selected and min(spec.end_seconds, segment_end_seconds) > max(spec.start_seconds, req.start_seconds)
    ] or [(1, req.start_seconds, segment_end_seconds)]
    for side in ("A", "B"):
        mine = [float(item["time_seconds"]) for item in all_final_live_events if item.get("fighter") == side]
        theirs = [float(item["time_seconds"]) for item in all_final_live_events if item.get("fighter") not in (side, None)]
        final_live_stats.setdefault("fighters", {}).setdefault(side, {})["output"] = output_numbers(mine, theirs, round_bounds)
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
    # Kicks and knees not emitted because the leg that would have thrown them
    # was not visible (waist-up framing, legs out of shot).
    classifier["leg_actions_discarded_legs_not_visible"] = {
        fighter: int(state.legs_hidden_discards) for fighter, state in action_engine.states.items()}
    # Always against the final rounds (detected or scheduled), so the plain
    # numbers can be given per round. Idempotent after a detection above.
    metrics.rebucket_rounds(rounds)
    metric_data = metrics.finalize(report_events, defenses, measured_duration)
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
        # How often the pose model returned a joint the body could not have
        # reached. A quality signal rather than a performance one: a rising
        # share is the detector degrading on footage it finds hard, and that is
        # worth seeing before the metrics built on those joints start drifting.
        "pose_gate": None if joint_gate is None else joint_gate.summary(),
        "metric_definition": "Observation coverage: accepted fighter observations divided by analyzed frames. This is not ground-truth identity accuracy.",
        # Where the forward pass was seeded. Equal to the frame the person
        # picked unless the fighters were followed back from it, in which
        # case identity_seed says where they picked and how far back it went.
        "selection_source_frame": start_frame,
        "selection_source_seconds": start_frame / info.fps,
        "identity_seed": {
            "selection_seconds": round(float(seed_seconds), 3),
            "seed_check": None if seed_check is None else {
                "verdicts": list(seed_check.get("verdicts", []))[:12],
                "confirmed": not seed_check.get("failed"),
                "failed": seed_check.get("failed"),
            },
            "backtrack": handoff,
        },
        # False when the forward pass reached the frame the person picked with
        # A and B not on the boxes they drew. The identity gate reads it
        # (core/report.py identity_ready_by_fighter), so such a report is
        # shown as unverified rather than as the chosen fighter's numbers.
        "identity_seed_confirmed": None if seed_check is None else not seed_check.get("failed"),
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
        "fighter_A_track_handoffs": manager.track_handoffs["A"],
        "fighter_B_track_handoffs": manager.track_handoffs["B"],
        "fighter_A_handoffs_per_minute": round(manager.track_handoffs["A"] / (segment_duration / 60.0), 2),
        "fighter_B_handoffs_per_minute": round(manager.track_handoffs["B"] / (segment_duration / 60.0), 2),
        "fighter_A_suspicious_handoffs_per_minute": round(
            manager.suspicious_handoffs["A"] / (segment_duration / 60.0), 2),
        "fighter_B_suspicious_handoffs_per_minute": round(
            manager.suspicious_handoffs["B"] / (segment_duration / 60.0), 2),
        "identity_swaps_corrected": int(manager.swaps_corrected),
        "fighter_A_sam_recoveries": manager.a.sam_recovery_count,
        "fighter_B_sam_recoveries": manager.b.sam_recovery_count,
        "sam_continuous_enabled": SETTINGS.sam_continuous_enabled,
        "sam_continuous_frames": sam_recovery.continuous_frames,
        "fighter_A_sam_guided_frames": sam_guided["A"],
        "fighter_B_sam_guided_frames": sam_guided["B"],
        "fighter_A_guided_pose_recoveries": guided_pose_recoveries["A"],
        "fighter_B_guided_pose_recoveries": guided_pose_recoveries["B"],
        # Found by cropping to where the tracker expected them, on frames the
        # whole-picture pass missed them and the sweep had nothing to say.
        "fighter_A_expected_pose_recoveries": expected_pose_recoveries["A"],
        "fighter_B_expected_pose_recoveries": expected_pose_recoveries["B"],
        "sam_continuous_failure_reason": sam_recovery.continuous_failure_reason,
        "fighter_A_rejected_switches": manager.a.switches_rejected,
        "fighter_B_rejected_switches": manager.b.switches_rejected,
        # Which guard refused, not merely how many times. A total cannot tell an
        # appearance gate turning away the real fighter from a motion gate
        # correctly refusing a spectator, and those want opposite fixes.
        "rejected_switch_reasons": dict(manager.rejections),
        # Counted per lost fighter rather than per candidate, so it says which
        # gate is actually costing coverage. Read this one, not the line above.
        "blocked_recovery_reasons": dict(manager.blocked_recovery),
        # The same reasons, split by fighter, with the number of frames each
        # of them was missing as the denominator.
        #
        # The pooled dict above cannot answer why one corner is tracked worse
        # than the other, and it is: red measured 64-74% against blue's 82-88%
        # on every fight sampled, never once close. A cause that is most of
        # A's losses and none of B's is indistinguishable, pooled, from one
        # shared evenly - and the two call for opposite fixes.
        "blocked_recovery_by_fighter": {
            side: dict(reasons)
            for side, reasons in manager.blocked_recovery_by_fighter.items()
        },
        "missing_frames_by_fighter": dict(manager.missing_frames_by_fighter),
        # What the recording itself looks like, and what the person holding the
        # camera could do differently. Kept beside the tracking numbers because
        # it is usually the explanation for them.
        "recording": preflight.as_dict(),
        "furniture_tracks_readmitted": manager.forgiven_furniture,
        # How alike the two chosen fighters are, and how often the analysis
        # could not tell which was which. Reported whether or not they cross
        # the line, because the number is the evidence for the verdict.
        "fighter_pair_similarity": None if pair_similarity is None else float(pair_similarity),
        "pair_similarity_method": "kit_regions_lab_v1" if kit is not None else "torso_hs_histogram",
        "kit_similarity": kit,
        "fighter_pair_histogram_similarity": (
            None if histogram_similarity is None else float(histogram_similarity)),
        "fighters_separable": (
            None if pair_similarity is None
            else bool(pair_similarity < (SETTINGS.max_kit_similarity if kit is not None
                                         else SETTINGS.max_fighter_pair_similarity))),
        # Moments the identity manager could not tell which was which, per
        # minute. With look-alike kit this - not the kit itself - decides
        # whether identity held (core/report.py identity_ready_by_fighter).
        "identity_confusions_per_minute": round(
            int(manager.confusions) / max(1e-6, segment_duration / 60.0), 2),
        "identity_confusions": int(manager.confusions),
        # Actions credited to the fighter who was being hit rather than the one
        # hitting, and removed. See core.contact.resolve_simultaneous_attribution.
        "defender_actions_dropped": int(defender_actions),
        "last_identity_confusion_frame": manager.last_confusion_frame,
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
        # What the budget governor decided, and whether it expected to succeed.
        # Published because sampling less is a real cost to the analysis and the
        # reader is entitled to know it was paid: "budget_met_expected" false
        # means this machine cannot analyse this footage in real time without
        # going below min_tracking_fps, and the stride stopped at that floor.
        "realtime_budget_enforced": SETTINGS.hard_realtime_budget,
        "budget_plan": quality.budget_reason,
        # Where the frame cost behind the plan came from. Two runs of one video
        # that planned different strides are explained by this field and by
        # nothing else in the report.
        "budget_cost_source": quality.budget_cost_source,
        # What the plan assumed a frame costs, and what one actually cost during
        # calibration. Equal on a machine behaving as its profile describes.
        # When observed is far above planned, the stored profile is stale for
        # this run - something else is using the card, or the workload is not
        # what the profile was measured on - and budget_plan says
        # "machine_slower_than_profile" rather than claiming a budget was met.
        # Without these two numbers that conclusion is unfalsifiable.
        "budget_cost_planned_seconds": quality.budget_cost_planned,
        "budget_cost_observed_seconds": quality.budget_cost_observed,
        "budget_met_expected": quality.budget_expected_met,
        "planned_stride": quality.planned_stride,
        "final_analysis_fps": quality.effective_fps,
        # Where the time went. The frame rate above is set by what one frame
        # costs, and without these a slow run could only be guessed at: the
        # SAM2 sweep that follows both fighters before the frame pass, and the
        # frame pass itself (detection, pose, identity, actions) per analysed
        # frame. Loading models and building the report are the rest of
        # analysis_seconds.
        "sam_sweep_seconds": round(sam_sweep_seconds, 2),
        "frame_pass_seconds": round(frame_pass_seconds, 2),
        "frame_pass_seconds_per_frame": (round(frame_pass_seconds / analyzed_frames, 4)
                                         if analyzed_frames else None),
        # The frame pass split by step, in seconds per analysed frame.
        "frame_pass_breakdown_per_frame": _frame_breakdown(
            {**pose_tracker.timing, **step_seconds}, frame_pass_seconds, analyzed_frames),
        "final_imgsz": quality.imgsz,
        "quality_mode": quality.mode,
        "pose_model": pose_tracker.model_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        # What else was on the card before this run started. See the comment at
        # the top of analyze(): a slow run and a busy card are indistinguishable
        # in every other field here.
        "vram_free_at_start": vram_free_at_start,
        "unused_frame_copy_avoidance": not fallback_buffer_enabled,
        "external_identity_history_enabled": identity_referee.enabled,
        "timestamp_source": "decoded_presentation_timestamps",
        "timestamp_fallback_frames": source_clock.fallback_frames,
    }
    progress(
        "Building performance report", 98.3, analysis_seconds, segment_duration, manager, None, quality,
        stage="report", live_events_snapshot=final_live_events, stats=final_live_stats,
        observation=_latest_observation(seconds, info.width, info.height, fighter_a, fighter_b, manager),
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
    # Which part of the video this report describes. Every surface that states
    # a duration reads it from here, so none of them can call a span "the
    # fight" without saying what was left out.
    report.setdefault("video", {})["analysed_span"] = _analysed_span(
        info, req.start_seconds, segment_end_seconds, seed_seconds, handoff, excluded_reason)
    # Which analysis code made this, so a result from an out-of-date worker
    # can be recognised and refused by the web app (core/build_info.py).
    report["analysis_build"] = build_stamp()
    report["video"]["fight_footage"] = fight_footage
    report.setdefault("integrity", {})["fight_footage_sufficient"] = bool(fight_footage["sufficient"])

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
    # Moments someone went down, against the rounds as finally found. Not
    # attributed to a fighter and not scored; see core/ground.py.
    report["went_down"] = down_watch.summary(rounds)
    progress(
        "Finalizing coaching priorities", 99.2, time.perf_counter() - wall_start, segment_duration,
        manager, None, quality, stage="report", live_events_snapshot=final_live_events,
        stats=final_live_stats,
        observation=_latest_observation(seconds, info.width, info.height, fighter_a, fighter_b, manager),
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
                float(public["combinations"]) / max(1e-6, measured_duration / 60.0)
            )
    # The customer-facing duration includes report construction, not only the
    # pose pass. Keep the saved report, progress screen and history summary on
    # the same wall-clock definition.
    # The whole wait, including the backward identity pass and any rerun.
    analysis_seconds = time.perf_counter() - wall_start + float(budget_spent_before)
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
        observation=_latest_observation(seconds, info.width, info.height, fighter_a, fighter_b, manager),
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
            "tracking": identity_tracking(report.get("tracking", {})),
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
        "Finalizing report" if req.output_dir else "Complete",
        99.9 if req.output_dir else 100.0,
        time.perf_counter() - wall_start, segment_duration, manager, None, quality,
        stage="report" if req.output_dir else "complete",
        live_events_snapshot=final_live_events,
        stats=final_live_stats,
        observation=_latest_observation(seconds, info.width, info.height, fighter_a, fighter_b, manager),
    )
    return report
