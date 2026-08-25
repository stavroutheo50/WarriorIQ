from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


@dataclass
class RoundSpec:
    number: int
    start_seconds: float
    end_seconds: float
    selected: bool = True


@dataclass
class AnalysisRequest:
    video_path: str
    fighter_a_box: list[float]
    fighter_b_box: list[float]
    original_name: str | None = None
    analysis_target: str = "BOTH"
    focus_fighter: str | None = None
    fight_type: str = "competition"
    ruleset: str = "K1"
    start_seconds: float = 0.0
    round_count: int = 3
    round_duration_seconds: float = 120.0
    break_duration_seconds: float = 60.0
    selected_rounds: list[int] | None = None
    end_seconds: float | None = None
    job_id: str = "cli"
    profile_id: int = 1
    persist_result: bool = True
    openai_identity_recovery: bool = False


@dataclass
class PersonObservation:
    track_id: int | None
    box: np.ndarray
    confidence: float
    keypoints: np.ndarray | None = None
    keypoint_conf: np.ndarray | None = None
    appearance: np.ndarray | None = None
    pose_signature: np.ndarray | None = None


@dataclass
class FighterState:
    name: str
    current_track_id: int | None = None
    last_box: np.ndarray | None = None
    prev_box: np.ndarray | None = None
    velocity: np.ndarray | None = None
    last_keypoints: np.ndarray | None = None
    appearance: np.ndarray | None = None
    pose_signature: np.ndarray | None = None
    identity_confidence: float = 0.0
    missing_frames: int = 0
    last_seen_source_frame: int = 0
    recovery_count: int = 0
    sam_recovery_count: int = 0
    last_sam_attempt_analyzed_frame: int = -10_000
    switches_rejected: int = 0


@dataclass
class PoseFrame:
    source_frame: int
    time_seconds: float
    round_number: int | None
    fighter: str
    box: list[float] | None
    keypoints: list[list[float]] | None
    keypoint_conf: list[float] | None
    identity_confidence: float
    visible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrikeEvent:
    fighter: str
    opponent: str
    round_number: int | None
    start_frame: int
    peak_frame: int
    end_frame: int
    start_time: float
    peak_time: float
    end_time: float
    technique: str
    family: str
    limb: str
    attempted: bool = True
    outcome: str = "uncertain"  # clean, blocked, checked, missed, likely_landed, uncertain
    landed: bool = False
    target: str | None = None  # head, body, leg
    confidence: float = 0.0
    contact_confidence: float = 0.0
    model_source: str = "temporal_rules"
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefenseEvent:
    fighter: str
    opponent: str
    round_number: int | None
    time_seconds: float
    source_frame: int
    defense: str  # block, parry, slip, evade, check, guard
    confidence: float
    against_technique: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnockdownEvent:
    fighter: str
    caused_by: str | None
    round_number: int | None
    source_frame: int
    time_seconds: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisProgress:
    percent: float
    message: str
    elapsed_seconds: float
    processed_video_seconds: float
    speed: float
    eta_seconds: float | None
    fighter_a_confidence: float = 0.0
    fighter_b_confidence: float = 0.0
    current_round: int | None = None
    quality_mode: str = "balanced"
    stage: str = "preparing"
    video_duration_seconds: float = 0.0
    live_event_mode: str = "withheld"
    live_events: list[dict[str, Any]] = field(default_factory=list)
    provisional_stats: dict[str, Any] = field(default_factory=dict)
    latest_observation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
