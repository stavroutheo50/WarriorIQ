from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("WARRIORIQ_DATA_DIR", str(ROOT))).expanduser().resolve()
UPLOADS = DATA_ROOT / "uploads"
OUTPUTS = DATA_ROOT / "outputs"
MODELS = DATA_ROOT / "models"
DATASET = DATA_ROOT / "dataset"
DB_PATH = DATA_ROOT / "warrioriq.sqlite3"
ULTRALYTICS_CONFIG = DATA_ROOT / ".ultralytics"
HUGGINGFACE_CACHE = DATA_ROOT / ".huggingface"

# Keep Ultralytics settings inside the project. This avoids roaming-profile
# permission failures and keeps the local build's configuration self-contained.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG))
os.environ.setdefault("HF_HOME", str(HUGGINGFACE_CACHE))

for path in (UPLOADS, OUTPUTS, MODELS, DATASET, ULTRALYTICS_CONFIG, HUGGINGFACE_CACHE):
    path.mkdir(parents=True, exist_ok=True)


def env_secret(name: str, default: str = "") -> str:
    """Read a shared secret, tolerating how hosting panels mangle pasted values.

    A secret is usually copied by hand from a control panel into a .env file or
    back again, and it commonly arrives wrapped: angle brackets left over from a
    placeholder like <token>, or quotes added to be safe. The wrapper is never
    part of the secret, but it makes authentication fail with a bare 401 that
    looks identical to a wrong password, so it is stripped here rather than
    debugged again at every call site.
    """
    value = os.getenv(name, default).strip()
    while len(value) >= 2 and value[0] in "<\"'" and value[-1] in ">\"'":
        value = value[1:-1].strip()
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


IS_RENDER = env_bool("RENDER", False)


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
    # Fighters in a wide or low-resolution recording occupy very few pixels.
    # Inferring a 480px-wide video at 640 barely upscales it and the athletes
    # stay under the size the detector resolves reliably, so a small source is
    # analysed at a larger inference size instead.
    low_resolution_edge: int = int(os.getenv("WARRIORIQ_LOW_RES_EDGE", "960"))
    low_resolution_imgsz: int = int(os.getenv("WARRIORIQ_LOW_RES_IMGSZ", "1280"))
    detection_conf: float = float(os.getenv("WARRIORIQ_DET_CONF", "0.20"))
    # Fighter drawing is fully manual. Candidate detection is only a visual
    # advisory, so free-tier web instances can skip loading YOLO on this page.
    selection_detection_enabled: bool = env_bool("WARRIORIQ_SELECTION_DETECTION", not IS_RENDER)

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
    sam_recovery_enabled: bool = env_bool("WARRIORIQ_SAM_RECOVERY", not IS_RENDER)
    # Continuous mode propagates SAM2 masks across the whole segment before the
    # main pass, which doubles the work for guidance the tracker usually does
    # not need. Measured on a real 480x220 WAKO bout, same 60-second segment:
    #
    #   continuous on   188.8s   coverage A 0.997  B 0.916
    #   continuous off   62.6s   coverage A 0.985  B 0.987
    #
    # Identity trust and the initial-lock check were satisfied either way, and
    # fighter B was tracked better without it. Recovery stays enabled, so the
    # fallback buffer in analyzer.py still engages when continuous tracks are
    # absent: the safety net for hard footage remains, only the unconditional
    # second pass is gone. Set this true to force the exhaustive pass on footage
    # where identities genuinely swap.
    sam_continuous_enabled: bool = env_bool("WARRIORIQ_SAM_CONTINUOUS", False)
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
    # A ten-point-must round is a judgement about who controlled a round, and
    # two detected actions cannot support one. Measured on real 480x220
    # tournament footage: 244 actions detected, 2 of them verified, and the
    # scorer still published a 9-10 round with a named winner. The score is
    # withheld below this; every movement measurement is kept.
    min_verified_actions_for_score: int = int(os.getenv("WARRIORIQ_MIN_SCORING_ACTIONS", "6"))
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

    # Social sign-in is opt-in per provider. A provider is exposed only when
    # both credentials and a stable state-signing secret are configured.
    oauth_state_secret: str = os.getenv("WARRIORIQ_OAUTH_STATE_SECRET", "").strip()
    google_client_id: str = os.getenv("WARRIORIQ_GOOGLE_CLIENT_ID", "").strip()
    google_client_secret: str = os.getenv("WARRIORIQ_GOOGLE_CLIENT_SECRET", "").strip()
    apple_client_id: str = os.getenv("WARRIORIQ_APPLE_CLIENT_ID", "").strip()
    apple_client_secret: str = os.getenv("WARRIORIQ_APPLE_CLIENT_SECRET", "").strip()
    facebook_client_id: str = os.getenv("WARRIORIQ_FACEBOOK_CLIENT_ID", "").strip()
    facebook_client_secret: str = os.getenv("WARRIORIQ_FACEBOOK_CLIENT_SECRET", "").strip()
    microsoft_client_id: str = os.getenv("WARRIORIQ_MICROSOFT_CLIENT_ID", "").strip()
    microsoft_client_secret: str = os.getenv("WARRIORIQ_MICROSOFT_CLIENT_SECRET", "").strip()

    # Analysis jobs can run inside the web process for a local/PyCharm build,
    # or be claimed by a separate GPU worker in production.  External mode is
    # deliberately opt-in so existing local installs keep working unchanged.
    analysis_worker_mode: str = os.getenv("WARRIORIQ_WORKER_MODE", "inprocess").strip().lower()
    worker_poll_seconds: float = max(0.2, float(os.getenv("WARRIORIQ_WORKER_POLL_SECONDS", "1.0")))
    worker_lease_seconds: int = max(30, int(os.getenv("WARRIORIQ_WORKER_LEASE_SECONDS", "180")))
    worker_stale_seconds: int = max(60, int(os.getenv("WARRIORIQ_WORKER_STALE_SECONDS", "300")))
    # Remote mode lets a GPU machine claim jobs over HTTPS instead of requiring
    # the web server and worker to share a filesystem. The token must be the
    # same high-entropy secret on both machines and is never sent to browsers.
    # A detached worker keeps the queue durable, so a fight can be accepted while
    # the analysis machine is switched off and claimed when it next connects.
    # Turn this off to refuse uploads whenever no worker is currently online.
    accept_deferred_analysis: bool = env_bool("WARRIORIQ_ACCEPT_DEFERRED_ANALYSIS", True)
    # Optional wake hook for a scale-to-zero GPU. When a fight is queued with no
    # worker online, the web process pings this URL so a serverless GPU starts,
    # drains the queue and shuts down again. Polling a GPU would bill idle time;
    # waking one on demand bills only the analysis itself.
    worker_wake_url: str = os.getenv("WARRIORIQ_WORKER_WAKE_URL", "").strip()
    # Wake-on-LAN for an analysis machine that sleeps between fights. The queue
    # cannot reach a sleeping PC and a sleeping PC cannot poll, so the web
    # process sends a magic packet the network card still listens for. Needs the
    # router to forward this UDP port to the LAN broadcast address. Purely an
    # accelerator: the scheduled drain still collects the fight if it fails.
    wol_mac: str = os.getenv("WARRIORIQ_WOL_MAC", "").strip()
    wol_host: str = os.getenv("WARRIORIQ_WOL_HOST", "").strip()
    wol_port: int = int(os.getenv("WARRIORIQ_WOL_PORT", "9"))
    # How often deploy/drain-queue.ps1 is scheduled to wake the machine. This is
    # the guaranteed ceiling on how long a fight waits when the magic packet
    # never arrives, and it is what the uploader is promised before any real
    # wake latency has been observed.
    wake_drain_interval_seconds: int = int(os.getenv("WARRIORIQ_DRAIN_INTERVAL", "300"))
    worker_remote_url: str = os.getenv("WARRIORIQ_WORKER_REMOTE_URL", "").rstrip("/")
    worker_token: str = env_secret("WARRIORIQ_WORKER_TOKEN")
    worker_artifact_max_bytes: int = max(
        10 * 1024 * 1024,
        int(os.getenv("WARRIORIQ_WORKER_ARTIFACT_MAX_BYTES", str(256 * 1024 * 1024))),
    )
    minimum_free_storage_gb: float = max(0.25, float(os.getenv("WARRIORIQ_MIN_FREE_STORAGE_GB", "2")))
    max_fight_bytes: int = max(
        50 * 1024 * 1024,
        int(os.getenv("WARRIORIQ_MAX_FIGHT_BYTES", str(2 * 1024 * 1024 * 1024))),
    )
    max_video_duration_seconds: int = max(60, int(os.getenv("WARRIORIQ_MAX_VIDEO_SECONDS", "10800")))
    max_video_pixels: int = max(640 * 360, int(os.getenv("WARRIORIQ_MAX_VIDEO_PIXELS", str(3840 * 2160))))
    malware_scan_command: str = os.getenv("WARRIORIQ_MALWARE_SCAN_COMMAND", "").strip()
    malware_scan_required: bool = env_bool("WARRIORIQ_MALWARE_SCAN_REQUIRED", False)

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
    # Complimentary plan grants as "email:plan_key" pairs, comma separated.
    # Recorded here rather than as a row edit on the live database so the grant
    # is visible, reviewable and survives a restore.
    complimentary_plans: dict[str, str] = field(default_factory=lambda: {
        email.strip().lower(): plan.strip().lower()
        for entry in os.getenv(
            "WARRIORIQ_COMPLIMENTARY_PLANS", "stavroutheo50@gmail.com:gym",
        ).split(",")
        if entry.strip() and ":" in entry
        for email, plan in [entry.split(":", 1)]
    })
    admin_emails: tuple[str, ...] = tuple(
        email.strip().lower()
        for email in os.getenv("WARRIORIQ_ADMIN_EMAILS", "").split(",")
        if email.strip()
    )
    # Google Analytics measurement ID (G-XXXXXXXXXX). The tag is rendered only
    # for visitors who accept analytics cookies; leaving this empty disables
    # analytics entirely and keeps the strict Content-Security-Policy.
    analytics_measurement_id: str = os.getenv("WARRIORIQ_ANALYTICS_ID", "G-5V5Q4H30LD").strip()
    # Google Tag Manager container. GTM loads whatever tags the container holds,
    # so if a GA4 tag inside it uses the same measurement ID as
    # WARRIORIQ_ANALYTICS_ID, every page view is counted twice. Run one or the
    # other: empty this to use the direct tag, or empty the measurement ID to
    # let the container own analytics.
    gtm_container_id: str = os.getenv("WARRIORIQ_GTM_ID", "GTM-PFCW27J2").strip()
    email_provider: str = os.getenv("WARRIORIQ_EMAIL_PROVIDER", "").strip()
    require_email_verification: bool = env_bool("WARRIORIQ_REQUIRE_EMAIL_VERIFICATION", False)


SETTINGS = Settings()

# Rule labels shown everywhere in the product.
# Grouped by sport for the upload form. The key order here is the order a
# visitor sees, so the discipline most people arrive for stays first.
RULESET_SPORTS = {
    "kickboxing": "Kickboxing",
    "boxing": "Boxing",
    "muay_thai": "Muay Thai",
    "taekwondo": "Taekwondo",
    "mma": "MMA",
}

RULESET_LABELS = {
    "K1": "K-1",
    "LOW_KICK": "Low Kick",
    "FULL_CONTACT": "Full Contact",
    "POINT_FIGHTING": "Point Fighting",
    "LIGHT_CONTACT": "Light Contact",
    "KICK_LIGHT": "Kick Light",
    "BOXING": "Boxing",
    "MUAY_THAI": "Full rules (elbows allowed)",
    "MUAY_THAI_NO_ELBOWS": "No elbows",
    "ITF_TAEKWONDO": "ITF · International Taekwon-Do Federation",
    "WT_TAEKWONDO": "WT · World Taekwondo (Olympic)",
    "MMA": "MMA (standing exchanges)",
}

RULESET_SHORT = {
    "ITF_TAEKWONDO": "ITF",
    "WT_TAEKWONDO": "WT",
    "MUAY_THAI": "Full rules",
    "MUAY_THAI_NO_ELBOWS": "No elbows",
    "MMA": "Standing exchanges",
}

FIGHT_TYPES = ("competition", "sparring")
ANALYSIS_TARGETS = ("A", "B", "BOTH")
