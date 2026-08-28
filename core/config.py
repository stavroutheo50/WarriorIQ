from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
DATASET = ROOT / "dataset"
DB_PATH = ROOT / "warrioriq.sqlite3"
ULTRALYTICS_CONFIG = ROOT / ".ultralytics"
HUGGINGFACE_CACHE = ROOT / ".huggingface"

# Keep Ultralytics settings inside the project. This avoids roaming-profile
# permission failures and keeps the local build's configuration self-contained.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG))
os.environ.setdefault("HF_HOME", str(HUGGINGFACE_CACHE))

for path in (UPLOADS, OUTPUTS, MODELS, DATASET, ULTRALYTICS_CONFIG, HUGGINGFACE_CACHE):
    path.mkdir(parents=True, exist_ok=True)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------
    # Core vision stack
    # ------------------------------------------------------------
    # YOLO26m Pose is the default compromise for an RTX 5060 8 GB and
    # the hard <= 1x video-duration budget. If a TensorRT engine exists,
    # WarriorIQ automatically prefers it.
    pose_model_pt: str = os.getenv("WARRIORIQ_POSE_MODEL", "yolo26m-pose.pt")
    pose_model_engine: str = os.getenv(
        "WARRIORIQ_POSE_ENGINE",
        str(MODELS / "yolo26m-pose.engine"),
    )
    tracker: str = os.getenv("WARRIORIQ_TRACKER", str(MODELS / "warrioriq_botsort.yaml"))
    # "auto" selects CUDA when available and CPU everywhere else.
    device: str = os.getenv("WARRIORIQ_DEVICE", "auto")
    default_imgsz: int = int(os.getenv("WARRIORIQ_IMGSZ", "640"))
    min_imgsz: int = int(os.getenv("WARRIORIQ_MIN_IMGSZ", "512"))
    detection_conf: float = float(os.getenv("WARRIORIQ_DET_CONF", "0.20"))

    # ------------------------------------------------------------
    # Performance target
    # ------------------------------------------------------------
    target_tracking_fps: float = float(os.getenv("WARRIORIQ_TARGET_FPS", "15"))
    min_tracking_fps: float = float(os.getenv("WARRIORIQ_MIN_FPS", "10"))
    max_tracking_fps: float = float(os.getenv("WARRIORIQ_MAX_FPS", "30"))
    hard_realtime_budget: bool = env_bool("WARRIORIQ_HARD_REALTIME", True)
    # Fixed sampling is the reproducible default. Adaptive sampling depends on
    # momentary machine load and can make identical fights follow different
    # frame paths; it remains available as an explicit speed opt-in.
    adaptive_quality: bool = env_bool("WARRIORIQ_ADAPTIVE_QUALITY", False)
    progress_interval_frames: int = 40

    # ------------------------------------------------------------
    # Identity manager
    # ------------------------------------------------------------
    # A manual fighter selection must substantially overlap a detector box
    # before that detector identity is allowed to replace the user's box.
    # Low-overlap, center-near candidates are commonly referees.
    min_initial_iou: float = 0.35
    track_id_bonus: float = 0.34
    # Fighter motion between sampled frames is often larger than generic
    # pedestrian motion. These values still reject distant bystanders, while
    # allowing a fighter to be recovered after a tracker-ID reset.
    min_reid_score: float = 0.48
    min_reid_margin: float = 0.05
    max_normalized_jump: float = 1.65
    max_missing_analyzed_frames: int = 20
    missing_before_recovery: int = 4
    appearance_ema: float = 0.85
    pose_ema: float = 0.82

    # ------------------------------------------------------------
    # SAM recovery
    # ------------------------------------------------------------
    # SAM2 propagates the two user-selected identities through the segment.
    # YOLO pose/ReID must still confirm a person before metrics are accepted.
    sam_recovery_enabled: bool = env_bool("WARRIORIQ_SAM_RECOVERY", True)
    sam_continuous_enabled: bool = env_bool("WARRIORIQ_SAM_CONTINUOUS", True)
    sam_continuous_fps: float = float(os.getenv("WARRIORIQ_SAM_FPS", "4"))
    sam_continuous_max_frames: int = int(os.getenv("WARRIORIQ_SAM_MAX_FRAMES", "360"))
    # Keep the bounded two-minute guidance pass in one memory state. Short
    # chunk resets were faster but could drift when reseeded during a crossing.
    sam_continuous_chunk_frames: int = int(os.getenv("WARRIORIQ_SAM_CHUNK_FRAMES", "360"))
    sam_model_id: str = os.getenv("WARRIORIQ_SAM_MODEL", "facebook/sam2.1-hiera-small")
    sam_buffer_frames: int = int(os.getenv("WARRIORIQ_SAM_BUFFER", "18"))
    sam_cooldown_analyzed_frames: int = 24
    openai_identity_model: str = os.getenv("WARRIORIQ_OPENAI_MODEL", "gpt-5.6-terra")
    openai_identity_min_confidence: float = float(os.getenv("WARRIORIQ_OPENAI_ID_MIN_CONF", "0.82"))
    openai_identity_cooldown_seconds: float = 10.0
    openai_identity_audit_seconds: float = 15.0

    # ------------------------------------------------------------
    # Temporal action recognition
    # ------------------------------------------------------------
    temporal_checkpoint: str = os.getenv(
        "WARRIORIQ_TEMPORAL_MODEL",
        str(MODELS / "warrioriq_temporal_best.pt"),
    )
    action_window: int = 12
    min_event_gap_seconds: float = 0.22
    min_strike_speed_body_lengths_per_s: float = 0.90
    min_extension_gain: float = 0.07
    temporal_probability_threshold: float = 0.60

    # ------------------------------------------------------------
    # Contact / outcome
    # ------------------------------------------------------------
    contact_threshold_body_lengths: float = 0.24
    likely_contact_threshold_body_lengths: float = 0.33
    contact_confirmation_frames: int = 2
    block_proximity_body_lengths: float = 0.18

    # ------------------------------------------------------------
    # Reports / evidence
    # ------------------------------------------------------------
    min_pose_coverage_for_metric: float = 0.60
    # A score remains explicitly estimated, but 85% verified coverage for
    # both fighters is sufficient when each counted action also passes the
    # stricter technique/contact evidence thresholds in scoring.py.
    min_tracking_coverage_for_score: float = float(os.getenv("WARRIORIQ_SCORE_COVERAGE", "0.85"))
    evidence_pre_seconds: float = 1.0
    evidence_post_seconds: float = 1.0
    save_tracking_jsonl: bool = True
    save_debug_video: bool = False

    # ------------------------------------------------------------
    # Product
    # ------------------------------------------------------------
    brand_name: str = "WarriorIQ"
    version: str = "1.0"
    default_profile_name: str = "My Athlete"
    payments_enabled: bool = env_bool("WARRIORIQ_PAYMENTS", False)

    # ------------------------------------------------------------
    # Public-launch and legal identity
    # ------------------------------------------------------------
    # These values intentionally have no made-up defaults. Paid checkout is
    # blocked until the real operator has supplied every launch-critical item.
    policy_version: str = os.getenv("WARRIORIQ_POLICY_VERSION", "2026-08-24")
    public_base_url: str = os.getenv("WARRIORIQ_PUBLIC_BASE_URL", "").rstrip("/")
    operator_name: str = os.getenv("WARRIORIQ_OPERATOR_NAME", "").strip()
    operator_address: str = os.getenv("WARRIORIQ_OPERATOR_ADDRESS", "").strip()
    operator_registration: str = os.getenv("WARRIORIQ_OPERATOR_REGISTRATION", "").strip()
    operator_vat: str = os.getenv("WARRIORIQ_OPERATOR_VAT", "").strip()
    governing_country: str = os.getenv("WARRIORIQ_GOVERNING_COUNTRY", "").strip()
    support_email: str = os.getenv("WARRIORIQ_SUPPORT_EMAIL", "").strip()
    privacy_email: str = os.getenv("WARRIORIQ_PRIVACY_EMAIL", "").strip()
    dmca_email: str = os.getenv("WARRIORIQ_DMCA_EMAIL", "").strip()
    dmca_agent_name: str = os.getenv("WARRIORIQ_DMCA_AGENT_NAME", "").strip()
    minimum_account_age: int = int(os.getenv("WARRIORIQ_MINIMUM_AGE", "18"))
    saved_video_retention_days: int = max(1, int(os.getenv("WARRIORIQ_VIDEO_RETENTION_DAYS", "30")))
    failed_upload_retention_hours: int = max(1, int(os.getenv("WARRIORIQ_FAILED_UPLOAD_RETENTION_HOURS", "24")))
    admin_emails: tuple[str, ...] = tuple(
        email.strip().lower()
        for email in os.getenv("WARRIORIQ_ADMIN_EMAILS", "").split(",")
        if email.strip()
    )
    email_provider: str = os.getenv("WARRIORIQ_EMAIL_PROVIDER", "").strip()


SETTINGS = Settings()

# Rule labels shown everywhere in the product.
RULESET_LABELS = {
    "K1": "K-1",
    "LOW_KICK": "Low Kick",
    "FULL_CONTACT": "Full Contact",
    "POINT_FIGHTING": "Point Fighting",
    "LIGHT_CONTACT": "Light Contact",
    "KICK_LIGHT": "Kick Light",
}

FIGHT_TYPES = ("competition", "sparring")
ANALYSIS_TARGETS = ("A", "B", "BOTH")
