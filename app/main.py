from __future__ import annotations

import json
import html
import hmac
import logging
import math
import shutil
import threading
import time
import uuid
import os
import secrets
import zipfile
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.base_client.errors import OAuthError

from app.state import (
    AnalysisRunLost, AnalysisStateNotPersisted, claim_next_job, create_job, delete_job,
    finalize_job_from_worker, get_job, list_jobs, prepare_job_run, record_worker_heartbeat,
    start_job_run, update_job, update_job_for_worker, worker_status,
)
from core.auth import (
    authenticate, end_session, hash_password, issue_session, register, resolve_session,
    session_token, token_digest, valid_email, valid_password,
)
from core.config import DATASET, OUTPUTS, ROOT, RULESET_LABELS, RULESET_SHORT, RULESET_SPORTS, SETTINGS, UPLOADS
from core.annotations import accuracy_summary, export_sequence
from core.model_validation import audit_dataset_split
from core.release_validation import assess_end_to_end_validation, end_to_end_metadata
from core.db import (
    add_assignment, analysis_allowance, apply_checkout_event, consume_email_verification_token,
    consume_password_reset_token,
    create_moderation_report, create_oauth_account, delete_account, delete_fight,
    delete_legal_acceptances_for_resource, get_account, get_account_by_email,
    get_account_for_oauth_identity, get_annotations, get_fight, get_fight_review, get_profile,
    get_report_share, init_db, list_accounts, list_all_fight_storage, list_annotations, list_assignments,
    list_expired_fight_videos, list_fights, list_legal_acceptances, list_moderation_reports,
    list_oauth_identities,
    list_outbound_messages, list_security_events, list_subscription_actions, mark_fight_video_deleted,
    mark_email_verified, mark_outbound_message_sent,
    queue_outbound_message, record_account_signup_acceptance, record_legal_acceptance,
    record_security_event, record_subscription_action, release_analysis, reserve_analysis,
    resolve_moderation_report, revoke_account_sessions, revoke_report_shares, save_annotation,
    save_email_verification_token, save_fight, save_password_reset_token, save_report_share,
    set_account_status, set_annotation_sequence,
    set_fight_review_status, toggle_assignment, update_cookie_preferences,
    update_marketing_consent, update_password_hash, update_profile,
)
from core.evidence_trust import report_evidence_trust
from core.coaching import build_coaching, build_training_plan
from core.payments import PLANS, cancel_subscription_at_period_end, create_checkout, effective_plan_key, plan_for_key, verify_webhook
from core.legal import LEGAL_DOCUMENTS, launch_readiness
from core.notifications import send_transactional_email
from core.progress_insights import build_progress
from core.quality_guardian import inspect_video_quality
from core.upload_security import scan_upload
from core.report import build_preliminary_scorecard, refresh_identity_integrity
from core.retention import (
    GUEST_RETENTION_HOURS, cleanup_abandoned_processing_files, cleanup_expired_guest_jobs,
    guest_job_valid, mark_guest_job,
)
from core.scoring import SPORTS, deduplicate_scoring_events, event_legality, is_verified_scoring_event, normalize_ruleset, score_fight, sport_unobserved
from core.social_auth import SOCIAL_AUTH
from core.types import AnalysisRequest, StrikeEvent
from core.video import get_video_info, read_frame

app = FastAPI(title="WarriorIQ")
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
_oauth_cookie_secure = SETTINGS.public_base_url.lower().startswith("https://")
app.add_middleware(
    SessionMiddleware,
    secret_key=SETTINGS.oauth_state_secret or secrets.token_urlsafe(48),
    session_cookie="warrioriq_oauth",
    max_age=600,
    same_site="none" if _oauth_cookie_secure else "lax",
    https_only=_oauth_cookie_secure,
)
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))
executor = ThreadPoolExecutor(max_workers=1)
init_db()

SESSION_COOKIE = "warrioriq_session"
GUEST_COOKIE = "warrioriq_guest"
ACTIVE_ANALYSIS_COOKIE = "warrioriq_active_analysis"
LAST_COMPLETED_ANALYSIS_COOKIE = "warrioriq_last_completed_analysis"
COOKIE_PREFERENCES_COOKIE = "warrioriq_cookie_preferences"
_last_guest_cleanup = 0.0
_last_saved_video_cleanup = 0.0
_rate_windows: dict[str, list[float]] = {}
MAX_FIGHT_BYTES = SETTINGS.max_fight_bytes
MAX_PROFILE_PHOTO_BYTES = 15 * 1024 * 1024
MAX_PROFILE_VIDEO_BYTES = 500 * 1024 * 1024
_progress_report_cache: dict[str, tuple[int, dict]] = {}
LOGGER = logging.getLogger("warrioriq")

PUBLIC_INDEX_ROUTES = (
    "/", "/pricing", "/privacy", "/legal", "/terms", "/cookies",
    "/acceptable-use", "/refunds", "/video-upload-policy", "/sports-medical-disclaimer",
    "/eula", "/dmca", "/accessibility", "/ai-transparency", "/security",
    "/subprocessors", "/contact", "/kickboxing-fight-analysis", "/k1-fight-analysis",
    "/fight-video-analysis-for-coaches", "/how-to-record-a-fight-for-analysis",
)
PRIVATE_ROUTE_PREFIXES = (
    "/api/", "/frame/", "/select/", "/progress/", "/result/", "/replay/", "/review/",
    "/media/", "/fighter-portrait/", "/selection-image/", "/dashboard", "/history",
    "/compare", "/coach", "/profile", "/validation", "/s/", "/share/", "/shares/",
    "/account/", "/settings/", "/admin", "/checkout/", "/stripe/", "/purchase/",
    "/auth/",
)

SEARCH_GUIDES = {
    "kickboxing-fight-analysis": {
        "title": "Kickboxing Fight Analysis for Athletes & Coaches | WarriorIQ",
        "description": "Learn how WarriorIQ turns kickboxing fight video into evidence-linked replay, measured performance insights and practical training priorities.",
        "eyebrow": "Kickboxing fight analysis",
        "heading": "Turn a full fight into a clearer next session.",
        "intro": "WarriorIQ helps kickboxers and coaches review what the footage can actually support. It follows both selected fighters, separates measured observations from uncertain action labels, and links useful findings back to the video.",
        "sections": [
            {"title": "What a useful fight review should answer", "body": "A good review should show where the athlete was effective, where position or timing broke down, and which moments deserve another look. WarriorIQ keeps fighter identity, round context and evidence coverage visible so a number never appears without context.", "items": ["Movement, guard and balance observations", "Ruleset-aware supported actions", "Round-by-round performance context", "Evidence replay and fighter-specific training priorities"]},
            {"title": "What happens when the footage is unclear", "body": "Fast exchanges, camera movement, obstructions and low light can limit any video model. WarriorIQ does not invent strikes to fill a report. Unsupported claims are withheld or presented as review candidates, while tracking and pose coverage remain visible."},
            {"title": "Built for training, not official judging", "body": "The scorecard is an evidence-gated training estimate. It is designed to help athletes and coaches structure review; it does not replace licensed officials or the governing rules of an event."},
        ],
        "faqs": [
            {"question": "Can WarriorIQ analyse sparring as well as competition footage?", "answer": "Yes. Choose the video type before upload so the report keeps the session context clear."},
            {"question": "Does WarriorIQ analyse both fighters?", "answer": "Yes. Both fighters are tracked for identity and fight context, while the selected focus fighter receives the deeper coaching report and training plan."},
        ],
        "related": [("K-1 fight analysis", "/k1-fight-analysis"), ("Record better analysis footage", "/how-to-record-a-fight-for-analysis")],
    },
    "k1-fight-analysis": {
        "title": "K-1 Fight Analysis and Video Review | WarriorIQ",
        "description": "Review K-1 fight video with ruleset-aware evidence, fighter tracking, round context, coaching priorities and replayable key moments.",
        "eyebrow": "K-1 video review",
        "heading": "Review a K-1 fight with the ruleset in view.",
        "intro": "K-1 review needs more than a generic strike counter. WarriorIQ keeps punches, kicks and permitted knee actions in the selected ruleset context while suppressing unsupported contact and scoring claims.",
        "sections": [
            {"title": "Ruleset-aware evidence", "body": "Select K-1 before analysis so legality checks and report wording use the correct style. The event promoter or federation remains the authority for the exact rules used in a particular bout.", "items": ["Separate fighter attribution", "Outcome labels only when supported", "Round and interruption context", "Illegal-action review kept separate from supported scoring evidence"]},
            {"title": "A replay your coach can use", "body": "Supported moments link back to the contact time and replay begins just before the event, giving the coach enough context to see the setup, defensive response and exit."},
            {"title": "Honest limits", "body": "A high tracking percentage is observation coverage, not proof that every action label is correct. WarriorIQ shows the difference and withholds a score when the available evidence is not strong enough."},
        ],
        "faqs": [
            {"question": "Does the analysis replace a K-1 judge?", "answer": "No. It is a training and video-review tool, not an official judging system."},
            {"question": "What camera angle works best?", "answer": "Use a stable, elevated ringside angle that keeps both fighters fully visible with minimal obstruction."},
        ],
        "related": [("Kickboxing fight analysis", "/kickboxing-fight-analysis"), ("Fight analysis for coaches", "/fight-video-analysis-for-coaches")],
    },
    "fight-video-analysis-for-coaches": {
        "title": "Fight Video Analysis for Kickboxing Coaches | WarriorIQ",
        "description": "Use fight video to build evidence-linked coaching priorities, athlete-specific training plans and practical work between kickboxing sessions.",
        "eyebrow": "For kickboxing coaches",
        "heading": "Spend review time on the moments that change training.",
        "intro": "WarriorIQ organises a fight into evidence, measured observations and coaching priorities so coaches can move from a long video to a focused conversation without pretending uncertain detections are facts.",
        "sections": [
            {"title": "From report to session plan", "body": "The focused athlete receives coaching priorities and a training plan derived from that fight's supported weaknesses. The plan is not copied between fighters and should be adapted by the coach to the athlete's level, health and competition calendar.", "items": ["Evidence-linked review moments", "Fighter-specific strengths and priorities", "Practical drill prescriptions", "Saved-fight progress context"]},
            {"title": "Keep the athlete in context", "body": "Both fighters are followed because pressure, defence and positioning depend on the opponent. The selected athlete receives the detailed report; the opponent remains contextual rather than receiving an unnecessary duplicate plan."},
            {"title": "Share carefully", "body": "Saved reports are private by default. Supported plans can share time-limited report links on eligible plans, while video permissions and athlete privacy remain the uploader's responsibility."},
        ],
        "faqs": [
            {"question": "Can I compare an athlete across fights?", "answer": "Saved analyses can contribute to progress views when the same athlete profile is used and the underlying observations are available."},
            {"question": "Will every report contain a scorecard?", "answer": "No. A scorecard is withheld when both fighters were not analysed or the evidence gates are not met."},
        ],
        "related": [("Kickboxing fight analysis", "/kickboxing-fight-analysis"), ("Record better analysis footage", "/how-to-record-a-fight-for-analysis")],
    },
    "how-to-record-a-fight-for-analysis": {
        "title": "How to Record a Kickboxing Fight for Video Analysis | WarriorIQ",
        "description": "Record clearer kickboxing footage for fighter tracking and fight analysis with practical advice on angle, lighting, framing and video quality.",
        "eyebrow": "Better footage guide",
        "heading": "Give fight analysis a clear view of both athletes.",
        "intro": "The best analysis starts before upload. A stable view of both full bodies makes fighter identity, footwork, guard and contact timing easier to observe throughout the fight.",
        "sections": [
            {"title": "Use one stable, wide angle", "body": "Place the camera high enough to see the floor around both fighters and far enough back to keep heads, gloves and feet in frame. Avoid digital zoom and rapid panning.", "items": ["Keep both full bodies visible", "Use landscape orientation", "Prefer 1080p at 30 fps or higher", "Avoid filming through ropes, spectators or the referee when possible"]},
            {"title": "Light and focus matter", "body": "Fast strikes need short exposure and clear focus. Use the brightest practical venue position, clean the lens and tap to focus on the ring before recording."},
            {"title": "Choose a clear selection frame", "body": "After upload, pick a frame where Fighter A and Fighter B are separated and fully visible. Draw each box tightly around the complete athlete, not a referee or corner person."},
        ],
        "faqs": [
            {"question": "Can I upload phone video?", "answer": "Yes. MP4 and MOV phone recordings are supported when the resolution, duration and file size stay within the upload limits."},
            {"question": "Should I crop the video first?", "answer": "Only if the crop keeps both fighters visible for the entire analysed segment. Cutting out feet or exits can weaken tracking and movement evidence."},
        ],
        "related": [("Kickboxing fight analysis", "/kickboxing-fight-analysis"), ("K-1 fight analysis", "/k1-fight-analysis")],
    },
}


class StartPayload(BaseModel):
    fighter_a_box: list[float]
    fighter_b_box: list[float]
    focus_fighter: str | None = None
    analysis_target: str | None = None


class DeletePayload(BaseModel):
    confirm: bool = False


class SelectionFramePayload(BaseModel):
    seconds: float


class AnnotationPayload(BaseModel):
    event_time: float
    contact_time: float | None = None
    predicted: dict
    fighter: str
    technique: str
    target: str
    outcome: str
    manual: bool = False


class WorkerIdentityPayload(BaseModel):
    worker_id: str


class WorkerProgressPayload(WorkerIdentityPayload):
    analysis_run_id: str
    patch: dict


class WorkerFailurePayload(WorkerIdentityPayload):
    analysis_run_id: str
    error_code: str = "analysis_failed"


def _analyze(req: AnalysisRequest, progress_callback):
    """Import the heavy vision stack only after a user starts analysis."""
    from core.analyzer import analyze

    return analyze(req, progress_callback)


def _get_pose_tracker():
    """Lazy model access for optional selection-page candidate detection."""
    from core.analyzer import get_pose_tracker

    return get_pose_tracker()


def _forwarded_header(request: Request, name: str, fallback: str) -> str:
    return request.headers.get(name, fallback).split(",", 1)[0].strip()


def _external_scheme(request: Request) -> str:
    return _forwarded_header(request, "x-forwarded-proto", request.url.scheme).lower()


def _external_origin(request: Request) -> str:
    scheme = _external_scheme(request)
    host = _forwarded_header(request, "x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{scheme}://{host}".lower()


def _request_is_secure(request: Request) -> bool:
    return _external_scheme(request) == "https"


def _public_analysis_error(exc: Exception) -> str:
    """Return a useful status without leaking model names or server paths."""
    if isinstance(exc, (FileNotFoundError, ImportError, ModuleNotFoundError)):
        return "The analysis engine is unavailable on this server. Your upload and fighter selections are preserved."
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return "This analysis exceeded the server's available memory. Your upload and fighter selections are preserved."
    return "WarriorIQ could not finish this analysis. Your upload and fighter selections are preserved so you can try again."


def _account(request: Request) -> dict | None:
    return getattr(request.state, "account", None)


def _profile_id(request: Request) -> int | None:
    account = _account(request)
    return int(account["profile_id"]) if account else None


def _request_plan(request: Request) -> dict:
    account = _account(request)
    if not account:
        return plan_for_key("free")
    return plan_for_key(effective_plan_key(
        account.get("plan"), account.get("plan_override"), account.get("email"),
    ))


def _owner_key(request: Request) -> str:
    account = _account(request)
    return f"account:{account['id']}" if account else f"guest:{request.state.guest_id}"


def _active_job_for_owner(owner_key: str) -> dict | None:
    jobs = [
        {"job_id": job_id, **job}
        for job_id, job in list_jobs()
        if job.get("owner_key") == owner_key and job.get("status") in {"queued", "running", "interrupted"}
    ]
    if not jobs:
        return None
    jobs.sort(key=lambda job: (
        {"running": 3, "queued": 2, "interrupted": 1}.get(job.get("status"), 0),
        float(job.get("updated_at_epoch", 0) or 0),
        float(job.get("percent", 0)),
    ), reverse=True)
    return jobs[0]


def _owned_job(owner_key: str, job_id: str | None, statuses: set[str]) -> dict | None:
    if not job_id:
        return None
    job = get_job(job_id)
    if not job or job.get("owner_key") != owner_key or job.get("status") not in statuses:
        return None
    return {"job_id": job_id, **job}


def _analysis_navigation_state(
    owner_key: str,
    active_job_id: str | None,
    completed_job_id: str | None,
) -> dict:
    """Keep the processing pointer and completed-result pointer independent.

    A stale cookie that names a completed fight must never hide a currently
    processing fight. The completed pointer is retained only as the exact
    result destination once no processing job owns the top-bar position.
    """
    processing_statuses = {"queued", "running", "interrupted"}
    active = _owned_job(owner_key, active_job_id, processing_statuses)
    if active is None:
        active = _active_job_for_owner(owner_key)

    completed = _owned_job(owner_key, completed_job_id, {"complete"})
    if completed is None:
        # Backward-compatible migration for browsers that only have the older
        # active-analysis cookie after that exact job completed.
        completed = _owned_job(owner_key, active_job_id, {"complete"})

    return {
        "active": active,
        "last_completed": completed,
        "display": active or completed,
    }


def _analysis_navigation_job(owner_key: str, preferred_job_id: str | None) -> dict | None:
    """Compatibility wrapper for callers that need only the displayed job."""
    return _analysis_navigation_state(owner_key, preferred_job_id, None)["display"]


def _analysis_navigation_url(job: dict) -> str:
    job_id = job["job_id"]
    return f"/result/{job_id}" if job.get("status") == "complete" else f"/progress/{job_id}"


def _safe_next(value: str | None, fallback: str = "/dashboard") -> str:
    value = (value or "").strip()
    return value if value.startswith("/") and not value.startswith("//") else fallback


def _is_admin(request: Request) -> bool:
    account = _account(request)
    return bool(account and account.get("email", "").lower() in SETTINGS.admin_emails)


def _enforce_rate_limit(request: Request, scope: str, limit: int, window_seconds: int) -> None:
    """Small single-process safety limit; production should add an edge/shared limiter too."""
    now = time.monotonic()
    client = request.client.host if request.client else "unknown"
    key = f"{scope}:{client}"
    recent = [stamp for stamp in _rate_windows.get(key, []) if now - stamp < window_seconds]
    if len(recent) >= limit:
        record_security_event(
            "rate_limit_exceeded", severity="warning", resource_type="route", resource_id=scope,
            metadata={"client": client},
        )
        raise HTTPException(429, "Too many requests. Wait a little and try again.")
    recent.append(now)
    _rate_windows[key] = recent


def _cookie_preferences(request: Request) -> dict:
    raw = request.cookies.get(COOKIE_PREFERENCES_COOKIE, "")
    if raw == "all":
        return {"decided": True, "analytics": True, "marketing": True}
    if raw == "custom-analytics":
        return {"decided": True, "analytics": True, "marketing": False}
    if raw == "custom-marketing":
        return {"decided": True, "analytics": False, "marketing": True}
    if raw == "essential":
        return {"decided": True, "analytics": False, "marketing": False}
    return {"decided": False, "analytics": False, "marketing": False}


def _queue_transactional_notice(
    account_id: int,
    message_type: str,
    recipient: str,
    subject: str,
    body: str,
    payload: dict,
) -> int:
    message_id = queue_outbound_message(account_id, message_type, recipient, payload)
    try:
        if send_transactional_email(recipient, subject, body):
            mark_outbound_message_sent(message_id)
    except Exception as exc:
        # The durable queued record lets the production delivery worker retry.
        LOGGER.warning(
            "transactional_notice_deferred message_id=%s type=%s error=%s",
            message_id, message_type, type(exc).__name__,
        )
    return message_id


def _authorized_job(request: Request, job_id: str) -> dict | None:
    job = get_job(job_id)
    if job:
        return job if job.get("owner_key") == _owner_key(request) else None
    fight = get_fight(job_id)
    profile_id = _profile_id(request)
    if fight and profile_id is not None and int(fight["profile_id"]) == profile_id:
        return fight
    if not _account(request) and guest_job_valid(job_id, request.state.guest_id):
        return {"job_id": job_id, "guest": True}
    return None


def _reports_for_profile(profile_id: int) -> list[dict]:
    records = []
    fights = list_fights(profile_id)
    newest_legacy_ids = {fight["job_id"] for fight in fights[:2]}
    for fight in reversed(fights):
        compact = (fight.get("summary") or {}).get("progress_report")
        if isinstance(compact, dict):
            records.append({"job_id": fight["job_id"], "created_at": fight["created_at"], "report": compact})
            continue
        if fight["job_id"] not in newest_legacy_ids:
            summary = fight.get("summary") or {}
            # Legacy rows predate compact progress snapshots. Their stored
            # observation coverage is still real, so keep it in history while
            # loading full detail only for the two newest legacy fights needed
            # for the current value and trend.
            target = fight.get("analysis_target") or "BOTH"
            minimal = {
                "video": {"analysis_target": target, "focus_fighter": target if target in {"A", "B"} else None},
                "setup": {"ruleset": fight.get("ruleset", "K1")},
                "integrity": {"action_metrics_trusted": False},
                "metrics": {
                    "A": {"pose_coverage": summary.get("fighter_A_coverage")},
                    "B": {"pose_coverage": summary.get("fighter_B_coverage")},
                },
                "coaching": {}, "training_plan": {},
            }
            records.append({"job_id": fight["job_id"], "created_at": fight["created_at"], "report": minimal})
            continue
        path = Path(fight["report_path"])
        if not path.exists():
            continue
        try:
            modified = path.stat().st_mtime_ns
            cached = _progress_report_cache.get(str(path))
            if cached and cached[0] == modified:
                report = cached[1]
            else:
                full_report = json.loads(path.read_text(encoding="utf-8"))
                report = {
                    key: full_report.get(key, {})
                    for key in ("video", "setup", "integrity", "metrics", "coaching", "training_plan")
                }
                _progress_report_cache[str(path)] = (modified, report)
        except (OSError, json.JSONDecodeError):
            continue
        records.append({"job_id": fight["job_id"], "created_at": fight["created_at"], "report": report})
    return records


def _analysis_quality_summary(report: dict) -> dict:
    """Summarize evidence quality without pretending coverage is accuracy."""
    video = report.get("video", {})
    focus = video.get("focus_fighter") or video.get("analysis_target", "BOTH")
    if focus not in {"A", "B"}:
        focus = "A"
    tracking = report.get("tracking", {})
    metrics = report.get("metrics", {})
    integrity = report.get("integrity", {})
    coverage = {
        fighter: max(0.0, min(1.0, float(tracking.get(f"fighter_{fighter}_coverage", 0.0) or 0.0)))
        for fighter in ("A", "B")
    }
    pose = max(0.0, min(1.0, float(metrics.get(focus, {}).get("pose_coverage", 0.0) or 0.0)))
    identities = integrity.get("fighter_identity_trusted", {})
    stable = all(bool(identities.get(fighter, tracking.get(f"fighter_{fighter}_initial_lock_safe", False))) for fighter in ("A", "B"))
    minimum_coverage = min(coverage.values())
    if not stable:
        label, tone = "Needs another fighter selection", "bad"
    elif minimum_coverage >= .85 and pose >= .80:
        label, tone = "Strong observation evidence", "strong"
    elif minimum_coverage >= .65 and pose >= .55:
        label, tone = "Good observation evidence", "good"
    else:
        label, tone = "Partial observation evidence", "review"
    return {
        "focus": focus,
        "label": label,
        "tone": tone,
        "coverage": coverage,
        "pose_coverage": pose,
        "identity_stable": stable,
        "action_trusted": bool(integrity.get("action_metrics_trusted", False)),
    }


def _save_upload_limited(upload: UploadFile, destination: Path, limit: int) -> None:
    free_before = shutil.disk_usage(destination.parent).free
    reserve = int(SETTINGS.minimum_free_storage_gb * 1024**3)
    if free_before <= reserve:
        raise HTTPException(507, "Fight uploads are temporarily paused while storage capacity is restored.")
    total = 0
    try:
        with destination.open("wb") as handle:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, "The uploaded file exceeds the maximum allowed size.")
                if free_before - total <= reserve:
                    raise HTTPException(507, "This upload would exceed WarriorIQ's private storage safety reserve.")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


@app.middleware("http")
async def viewer_context(request: Request, call_next):
    global _last_guest_cleanup, _last_saved_video_cleanup
    request_started = time.perf_counter()
    request_id = request.headers.get("x-request-id", "").strip()[:64] or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    forwarded_scheme = _external_scheme(request)
    public_host = urlsplit(SETTINGS.public_base_url).netloc.lower()
    forwarded_host = _forwarded_header(
        request, "x-forwarded-host", request.headers.get("host", request.url.netloc)
    ).lower()
    if public_host and forwarded_host == f"www.{public_host}":
        target = f"{SETTINGS.public_base_url}{request.url.path}"
        if request.url.query:
            target += f"?{request.url.query}"
        return RedirectResponse(target, status_code=308)
    # Render may call the health probe over its private HTTP network. Keep that
    # endpoint directly reachable while redirecting public browser traffic.
    if request.url.path != "/health" and SETTINGS.public_base_url.startswith("https://") and forwarded_scheme != "https":
        target = f"{SETTINGS.public_base_url}{request.url.path}"
        if request.url.query:
            target += f"?{request.url.query}"
        return RedirectResponse(target, status_code=308)
    request.state.account = resolve_session(request.cookies.get(SESSION_COOKIE))
    guest_id = request.cookies.get(GUEST_COOKIE)
    new_guest = not guest_id or len(guest_id) < 24 or len(guest_id) > 96
    request.state.guest_id = session_token() if new_guest else guest_id
    request.state.analysis_navigation = _analysis_navigation_state(
        _owner_key(request),
        request.cookies.get(ACTIVE_ANALYSIS_COOKIE),
        request.cookies.get(LAST_COMPLETED_ANALYSIS_COOKIE),
    )
    request.state.active_analysis = request.state.analysis_navigation["display"]
    request.state.launch = launch_readiness()
    request.state.minimum_account_age = SETTINGS.minimum_account_age
    request.state.oauth_providers = SOCIAL_AUTH.provider_buttons
    request.state.cookie_preferences = _cookie_preferences(request)
    request.state.analytics_measurement_id = SETTINGS.analytics_measurement_id
    request.state.gtm_container_id = SETTINGS.gtm_container_id
    request.state.is_admin = _is_admin(request)
    request.state.noindex = (
        not SETTINGS.public_base_url
        or request.url.path.startswith(PRIVATE_ROUTE_PREFIXES)
        or request.url.path not in PUBLIC_INDEX_ROUTES
    )
    request.state.canonical_url = (
        f"{SETTINGS.public_base_url}{request.url.path}"
        if SETTINGS.public_base_url and request.url.path in PUBLIC_INDEX_ROUTES else ""
    )
    request.state.site_url = SETTINGS.public_base_url
    request.state.social_image_url = (
        f"{SETTINGS.public_base_url}/static/warrioriq-logo.png" if SETTINGS.public_base_url else ""
    )
    oauth_callback = request.url.path.startswith("/auth/") and request.url.path.endswith("/callback")
    if (
        request.method in {"POST", "PUT", "PATCH", "DELETE"}
        and request.url.path != "/stripe/webhook"
        and not oauth_callback
    ):
        expected_origin = _external_origin(request)
        source = request.headers.get("origin") or request.headers.get("referer")
        if source:
            parsed = urlsplit(source)
            source_origin = f"{parsed.scheme}://{parsed.netloc}".lower()
            if source_origin != expected_origin:
                return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
        if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
            return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
    now = time.monotonic()
    if now - _last_guest_cleanup > 600:
        protected = {
            job_id for job_id, job in list_jobs()
            if job.get("status") in {"selecting", "queued", "running"}
        }
        for job_id in cleanup_expired_guest_jobs(protected):
            delete_legal_acceptances_for_resource(job_id)
            delete_job(job_id)
        _last_guest_cleanup = now
    if now - _last_saved_video_cleanup > 3600:
        for fight in list_expired_fight_videos():
            video = Path(fight.get("video_path") or "missing").resolve()
            if video.parent == UPLOADS.resolve():
                video.unlink(missing_ok=True)
                mark_fight_video_deleted(fight["job_id"], int(fight["profile_id"]))
                record_security_event(
                    "video_retention_deleted", account_id=None, resource_type="fight",
                    resource_id=fight["job_id"], metadata={"scheduled": True},
                )
        protected = {
            job_id for job_id, job in list_jobs()
            if job.get("status") in {"selecting", "queued", "running"}
        }
        saved = {item["job_id"] for item in list_all_fight_storage()}
        for abandoned_job_id in cleanup_abandoned_processing_files(
            protected, saved, older_than_hours=SETTINGS.failed_upload_retention_hours,
        ):
            delete_legal_acceptances_for_resource(abandoned_job_id)
            delete_job(abandoned_job_id)
            record_security_event(
                "abandoned_processing_files_deleted", resource_type="fight", resource_id=abandoned_job_id,
            )
        _last_saved_video_cleanup = now
    response = await call_next(request)
    duration_ms = (time.perf_counter() - request_started) * 1000.0
    response.headers.setdefault("X-Request-ID", request_id)
    response.headers.setdefault("Server-Timing", f"app;dur={duration_ms:.1f}")
    LOGGER.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
    # The analytics tag is only rendered once a visitor accepts analytics
    # cookies, so the policy only names Google's hosts for those visitors.
    # Without this the browser blocks googletagmanager.com outright and no
    # measurement ever reaches Google, however the tag is configured.
    # Consent Mode loads the tag on every page and denies storage until the
    # visitor accepts, so the policy has to permit Google's hosts whenever a tag
    # is configured. Consent controls what may be stored, not whether the script
    # is reachable; gating the policy on consent hid the tag from Google's own
    # detection and made a correct install look absent.
    analytics_allowed = bool(SETTINGS.analytics_measurement_id or SETTINGS.gtm_container_id)
    script_src = "'self' 'unsafe-inline'"
    connect_src = "'self'"
    img_src = "'self' data:"
    # Tag Manager's noscript fallback is an iframe, which default-src would
    # block, so frame-src is only widened when a container is actually loaded.
    frame_src = "'self'"
    if analytics_allowed:
        script_src += " https://www.googletagmanager.com"
        connect_src += " https://*.google-analytics.com https://*.analytics.google.com https://*.googletagmanager.com"
        img_src += " https://*.google-analytics.com https://*.googletagmanager.com"
        if SETTINGS.gtm_container_id:
            frame_src += " https://www.googletagmanager.com"
    response.headers.setdefault(
        "Content-Security-Policy",
        f"default-src 'self' data:; script-src {script_src}; style-src 'self' 'unsafe-inline'; "
        f"img-src {img_src}; media-src 'self'; connect-src {connect_src}; frame-src {frame_src}; "
        "frame-ancestors 'none'; form-action 'self'",
    )
    if request.url.path.startswith(("/result/", "/replay/", "/media/", "/api/", "/profile", "/history", "/dashboard", "/coach", "/s/")):
        response.headers.setdefault("Cache-Control", "no-store")
    elif request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "public, max-age=604800")
    if _request_is_secure(request):
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if new_guest:
        response.set_cookie(
            GUEST_COOKIE, request.state.guest_id, max_age=60 * 60 * 24,
            httponly=True, samesite="lax", secure=_request_is_secure(request),
        )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_error_page(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith(("/api/", "/stripe/")) or "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
    title = {
        400: "That request needs attention",
        403: "This area is private",
        404: "That page left the ring",
        410: "This link has expired",
        413: "That file is too large",
        429: "Analysis limit reached",
        503: "This feature is not launch-ready",
        507: "Storage is temporarily full",
    }.get(exc.status_code, "WarriorIQ could not complete that request")
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={"request": request, "status_code": exc.status_code, "error_title": title, "error_detail": str(exc.detail or "")},
        status_code=exc.status_code,
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unexpected_error_page(request: Request, exc: Exception):
    LOGGER.exception("Unhandled request failure", exc_info=exc)
    detail = "WarriorIQ could not complete this request. Please try again."
    if request.url.path.startswith(("/api/", "/stripe/")) or "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"detail": detail}, status_code=500)
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "request": request,
            "status_code": 500,
            "error_title": "WarriorIQ hit a technical problem",
            "error_detail": detail,
        },
        status_code=500,
    )


ANNOTATION_TECHNIQUES = (
    "none",
    "jab", "cross", "left_hook", "right_hook", "left_uppercut", "right_uppercut",
    "backfist", "spinning_backfist", "left_low_kick", "right_low_kick",
    "left_body_kick", "right_body_kick", "left_head_kick", "right_head_kick",
    "left_front_kick", "right_front_kick", "left_push_kick", "right_push_kick",
    "left_knee", "right_knee",
)


def _event_prediction(event: dict) -> dict:
    return {
        "fighter": event.get("fighter", "A"),
        "technique": event.get("technique", "none"),
        "target": event.get("target"),
        "outcome": event.get("outcome", "uncertain"),
        "family": event.get("family", "punch"),
        "limb": event.get("limb", "hand"),
    }


def _prediction_at(report: dict, event_time: float) -> dict | None:
    candidates = report.get("events") or (report.get("key_moments", []) + report.get("illegal_moves", []))
    if not candidates:
        return None
    closest = min(candidates, key=lambda event: abs(float(event.get("peak_time", -1)) - event_time))
    if abs(float(closest.get("peak_time", -1)) - event_time) > 0.02:
        return None
    return _event_prediction(closest)


def _strike_from_dict(event: dict) -> StrikeEvent:
    fighter = str(event.get("fighter", "A")).upper()
    return StrikeEvent(
        fighter=fighter,
        opponent="B" if fighter == "A" else "A",
        round_number=event.get("round_number"),
        start_frame=int(event.get("start_frame", 0)),
        peak_frame=int(event.get("peak_frame", 0)),
        end_frame=int(event.get("end_frame", 0)),
        start_time=float(event.get("start_time", event.get("peak_time", 0))),
        peak_time=float(event.get("peak_time", 0)),
        end_time=float(event.get("end_time", event.get("peak_time", 0))),
        technique=str(event.get("technique", "none")),
        family=str(event.get("family", "punch")),
        limb=str(event.get("limb", "hand")),
        outcome=str(event.get("outcome", "uncertain")),
        target=event.get("target"),
        confidence=1.0 if event.get("human_verified") else float(event.get("confidence", 0)),
        contact_confidence=1.0 if event.get("human_verified") else float(event.get("contact_confidence", 0)),
        model_source="human_ground_truth" if event.get("human_verified") else str(event.get("model_source", "temporal_rules")),
    )


def _round_number_at(report: dict, event_time: float) -> int | None:
    for item in report.get("rounds", []):
        if float(item.get("start_seconds", 0)) <= event_time <= float(item.get("end_seconds", 0)):
            return item.get("number")
    return None


def _confirmed_metrics(report: dict, events: list[StrikeEvent]) -> dict:
    """Replace action-derived fields with human labels while retaining pose measurements."""
    metrics = deepcopy(report.get("metrics", {}))
    duration_minutes = max(1 / 60, float(report.get("performance", {}).get("segment_duration_seconds", 0)) / 60)
    for fighter in ("A", "B"):
        own = metrics.setdefault(fighter, {})
        fighter_events = [event for event in events if event.fighter == fighter]
        landed = [event for event in fighter_events if event.outcome == "clean"]
        attempts = len(fighter_events)
        techniques: dict[str, int] = {}
        families: dict[str, int] = {}
        targets: dict[str, int] = {}
        for event in fighter_events:
            techniques[event.technique] = techniques.get(event.technique, 0) + 1
            families[event.family] = families.get(event.family, 0) + 1
        for event in landed:
            if event.target:
                targets[event.target] = targets.get(event.target, 0) + 1
        own["attacks"] = {
            "attempts": attempts,
            "landed": len(landed),
            "clean": len(landed),
            "likely_landed": 0,
            "blocked": sum(event.outcome == "blocked" for event in fighter_events),
            "checked": sum(event.outcome == "checked" for event in fighter_events),
            "missed": sum(event.outcome == "missed" for event in fighter_events),
            "uncertain": sum(event.outcome == "uncertain" for event in fighter_events),
            "accuracy": len(landed) / attempts if attempts else None,
            "techniques": techniques,
            "families": families,
            "targets_landed": targets,
        }
        own["strongest_weapon"] = max(
            (event.technique for event in landed),
            key=lambda technique: sum(item.technique == technique for item in landed),
            default=None,
        )
        ordered = sorted(fighter_events, key=lambda event: event.peak_time)
        combo_times = [ordered[index].peak_time for index in range(1, len(ordered)) if ordered[index].peak_time - ordered[index - 1].peak_time <= 1.25]
        own["combinations"] = {"count": len(combo_times), "max_length": 2 if combo_times else 0, "times": combo_times[:12]}
        own["counters"] = {"count": 0, "times": []}
        own["defenses"] = {}
        vulnerabilities: dict[str, int] = {}
        for event in events:
            if event.fighter != fighter and event.outcome == "clean" and event.target:
                vulnerabilities[event.target] = vulnerabilities.get(event.target, 0) + 1
        own["vulnerability_targets"] = vulnerabilities
        dashboard = own.setdefault("dashboard", {})
        dashboard["activity_attempts_per_minute"] = attempts / duration_minutes
        dashboard["combinations_per_minute"] = len(combo_times) / duration_minutes
        dashboard["technique_execution_confidence"] = len(landed) / attempts if attempts else None
        dashboard["defense_response_rate"] = None
    return metrics


def _withhold_unverified_action_report(report: dict, reason: str) -> None:
    ruleset = report.get("setup", {}).get("ruleset", "K1")
    round_numbers = [int(item["number"]) for item in report.get("rounds", []) if item.get("selected", True)]
    candidate_events = [_strike_from_dict(item) for item in report.get("events", [])]
    report["scorecard"] = build_preliminary_scorecard(
        candidate_events,
        ruleset,
        round_numbers,
        report.get("tracking", {}),
        report.get("video", {}).get("analysis_target", "BOTH"),
    )
    empty = {"strengths": [], "improvements": [], "drills": [], "note": reason}
    report["coaching"] = {"A": dict(empty), "B": dict(empty)}
    report["training_plan"] = {"A": [], "B": []}
    for fighter in ("A", "B"):
        own = report.get("metrics", {}).get(fighter)
        if not own:
            continue
        own["attacks"] = {
            "attempts": 0, "landed": 0, "clean": 0, "likely_landed": 0,
            "blocked": 0, "checked": 0, "missed": 0, "uncertain": 0,
            "accuracy": None, "techniques": {}, "families": {}, "targets_landed": {},
        }
        own["strongest_weapon"] = None
        own["combinations"] = {"count": 0, "max_length": 0, "times": []}
        own["counters"] = {"count": 0, "times": []}
        own["defenses"] = {}
        own["vulnerability_targets"] = {}
        dashboard = own.setdefault("dashboard", {})
        dashboard.update({
            "technique_execution_confidence": None,
            "defense_response_rate": None,
            "activity_attempts_per_minute": 0.0,
            "combinations_per_minute": 0.0,
        })


def _apply_human_scorecard(report: dict, confirmed_events: list[StrikeEvent]) -> None:
    if report.get("video", {}).get("analysis_target", "BOTH") != "BOTH":
        report.setdefault("scorecard", {}).update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": "both_fighters_required",
            "disclaimer": "To receive an estimated scorecard, choose Analyze both fighters. A one-fighter analysis does not count the opponent's points.",
        })
        return
    ruleset = report.get("setup", {}).get("ruleset", "K1")
    round_numbers = [int(item["number"]) for item in report.get("rounds", []) if item.get("selected", True)]
    report["scorecard"] = score_fight(confirmed_events, ruleset, round_numbers, [], reliable=True)
    report["scorecard"].update({
        "status": "human_reviewed",
        "disclaimer": "Human-reviewed training estimate based only on actions confirmed while watching the video; it is not an official judges' score.",
        "evidence": {
            "verified_scoring_actions": int(report["scorecard"].get("verified_actions_counted", len(confirmed_events))),
            "fighter_A_tracking_coverage": float(report.get("tracking", {}).get("fighter_A_coverage", 0)),
            "fighter_B_tracking_coverage": float(report.get("tracking", {}).get("fighter_B_coverage", 0)),
            "evidence_source": "human_ground_truth",
        },
    })


def _apply_report_annotations(
    report: dict,
    annotations: list[dict],
    human_review_complete: bool = False,
    review_status: str | None = None,
) -> None:
    """Expose only validated-model or human-confirmed actions as fight evidence."""
    trust = report_evidence_trust(report)
    automated_trusted = bool(trust["automated_evidence_trusted"])
    integrity = report.setdefault("integrity", {})
    integrity.update(trust)
    status = review_status or ("complete" if human_review_complete else "in_progress")
    full_review_complete = status == "complete"
    scorecard_review_complete = status in {"scorecard_complete", "complete"}
    integrity["human_review_complete"] = full_review_complete
    integrity["scorecard_human_review_complete"] = scorecard_review_complete
    integrity["review_status"] = status

    displayed: dict[str, dict] = {}
    if automated_trusted:
        for source in (report.get("key_moments", []), report.get("illegal_moves", [])):
            for event in source:
                key = f"{float(event.get('peak_time', 0)):.3f}"
                item = dict(event)
                item["original_prediction"] = _event_prediction(event)
                item["evidence_source"] = "validated_model"
                displayed[key] = item

    raw_events = report.get("events", [])
    for annotation in annotations:
        event_time = float(annotation["event_time"])
        key = f"{event_time:.3f}"
        corrected_time = float(annotation["corrected"].get("contact_time", event_time))
        corrected_key = f"{corrected_time:.3f}"
        closest = min(raw_events, key=lambda event: abs(float(event.get("peak_time", -999)) - event_time), default=None)
        if closest is None or abs(float(closest.get("peak_time", -999)) - event_time) > 0.04:
            closest = {
                "fighter": annotation["corrected"].get("fighter", "A"),
                "round_number": _round_number_at(report, event_time),
                "start_frame": 0, "peak_frame": 0, "end_frame": 0,
                "start_time": event_time, "peak_time": event_time, "end_time": event_time,
            }
        item = dict(closest)
        original_peak_time = float(item.get("peak_time", event_time))
        time_shift = corrected_time - original_peak_time
        item["original_prediction"] = annotation["predicted"]
        item.update(annotation["corrected"])
        item["peak_time"] = corrected_time
        item["start_time"] = max(0.0, float(item.get("start_time", original_peak_time)) + time_shift)
        item["end_time"] = max(corrected_time, float(item.get("end_time", original_peak_time)) + time_shift)
        item["round_number"] = _round_number_at(report, corrected_time) or item.get("round_number")
        item["human_verified"] = True
        item["evidence_source"] = "human_ground_truth"
        item["is_corrected"] = annotation["predicted"] != annotation["corrected"]
        item["confidence"] = 1.0
        item["contact_confidence"] = 1.0
        if item.get("technique") == "none":
            displayed.pop(key, None)
        else:
            displayed.pop(key, None)
            displayed[corrected_key] = item

    legal_moments: list[dict] = []
    illegal_moments: list[dict] = []
    ruleset = report.get("setup", {}).get("ruleset", "K1")
    for item in displayed.values():
        legal, reason = event_legality(_strike_from_dict(item), ruleset)
        if legal:
            item.pop("legality_reason", None)
            legal_moments.append(item)
        else:
            item["legality_reason"] = reason
            illegal_moments.append(item)
    report["key_moments"] = sorted(legal_moments, key=lambda item: float(item.get("peak_time", 0)))
    report["illegal_moves"] = sorted(illegal_moments, key=lambda item: float(item.get("peak_time", 0)))
    integrity["human_confirmed_events"] = sum(item.get("human_verified", False) for item in displayed.values())

    if automated_trusted:
        integrity["action_metrics_trusted"] = True
        return

    reason = trust["action_evidence_reason"]
    integrity["action_metrics_trusted"] = False
    _withhold_unverified_action_report(report, reason)
    confirmed_events = [_strike_from_dict(item) for item in displayed.values() if item.get("human_verified")]
    if scorecard_review_complete:
        _apply_human_scorecard(report, confirmed_events)
    if not full_review_complete:
        return

    report["metrics"] = _confirmed_metrics(report, confirmed_events)
    report["coaching"] = {
        fighter: build_coaching(fighter, report["metrics"], confirmed_events)
        for fighter in ("A", "B")
    }
    report["training_plan"] = {
        fighter: build_training_plan(report["coaching"][fighter], fighter, report["metrics"][fighter])
        for fighter in ("A", "B")
    }
    integrity["action_metrics_trusted"] = True
    _apply_human_scorecard(report, confirmed_events)


def _save_fighter_portrait(job_id: str, fighter: str, box: list[float]) -> None:
    """Save the exact A/B selection as a small identity-reference portrait."""
    selection = cv2.imread(str(OUTPUTS / job_id / "selection.jpg"))
    if selection is None:
        return
    h, w = selection.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return
    crop = selection[y1:y2, x1:x2]
    if crop.size:
        cv2.imwrite(str(OUTPUTS / job_id / f"fighter_{fighter.upper()}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _parse_rounds(text: str, count: int) -> list[int] | None:
    text = (text or "").strip()
    if not text or text.upper() == "ALL":
        return None
    values = []
    for part in text.split(","):
        try:
            number = int(part.strip())
        except ValueError:
            continue
        if 1 <= number <= count:
            values.append(number)
    return sorted(set(values)) or None


def _analysis_request(job_id: str, job: dict, fighter_a_box: list[float], fighter_b_box: list[float], focus_fighter: str) -> AnalysisRequest:
    return AnalysisRequest(
        video_path=job["video_path"],
        fighter_a_box=fighter_a_box,
        fighter_b_box=fighter_b_box,
        original_name=job.get("original_name"),
        analysis_target="BOTH",
        focus_fighter=focus_fighter,
        fight_type=job["fight_type"],
        ruleset=job["ruleset"],
        start_seconds=job["start_seconds"],
        round_count=job["round_count"],
        round_duration_seconds=job["round_duration_seconds"],
        break_duration_seconds=job["break_duration_seconds"],
        selected_rounds=job.get("selected_rounds"),
        end_seconds=job.get("end_seconds"),
        job_id=job_id,
        profile_id=job.get("profile_id", 1),
        persist_result=bool(job.get("persist_result", False)),
        openai_identity_recovery=bool(job.get("openai_identity_recovery", False)),
    )


def _run_job(job_id: str, req: AnalysisRequest, analysis_run_id: str):
    worker_id = f"inprocess-{os.getpid()}-{analysis_run_id[:8]}"
    try:
        def cb(patch: dict):
            if not update_job_for_worker(job_id, worker_id, analysis_run_id, patch):
                raise AnalysisRunLost(f"Analysis run {analysis_run_id} no longer owns {job_id}")

        if not start_job_run(job_id, worker_id, analysis_run_id):
            LOGGER.warning("analysis_run_not_started job_id=%s run_id=%s", job_id, analysis_run_id)
            return
        report = _analyze(req, cb)
        if not update_job_for_worker(job_id, worker_id, analysis_run_id, {
            "status": "complete", "report": report, "percent": 100.0,
            "message": "Complete", "worker_lease_expires_epoch": None,
        }, renew_lease=False):
            LOGGER.warning("analysis_completion_discarded job_id=%s run_id=%s", job_id, analysis_run_id)
    except AnalysisRunLost:
        # A newer run or recovery now owns this job. Never overwrite its state
        # or refund its already-reserved account allowance.
        LOGGER.warning("analysis_run_ownership_lost job_id=%s run_id=%s", job_id, analysis_run_id)
    except Exception as exc:
        job = get_job(job_id)
        still_owns_run = bool(job and job.get("analysis_run_id") == analysis_run_id)
        if still_owns_run and job.get("usage_reserved") and job.get("account_id"):
            release_analysis(int(job["account_id"]), job_id)
            update_job(job_id, {"usage_reserved": False})
        LOGGER.exception("Analysis job %s failed", job_id, exc_info=exc)
        if still_owns_run:
            update_job(job_id, {
                "status": "error", "message": _public_analysis_error(exc),
                "worker_lease_expires_epoch": None,
            })


DEFERRED_ANALYSIS_MESSAGE = (
    "Waiting for the analysis machine to connect. Your fight is saved and starts automatically."
)


def _analysis_queue_decision() -> dict:
    """Decide whether a fight can be queued now, later, or not at all.

    A detached worker keeps queued work on disk, so a fight can wait for the
    analysis machine to reconnect instead of being refused. A server that is
    misconfigured or missing the vision stack would never process the fight,
    so those cases must still be refused rather than queued forever.
    """
    status = worker_status()
    if status.get("available"):
        return {"accepted": True, "deferred": False}
    deferrable = (
        SETTINGS.accept_deferred_analysis
        and SETTINGS.analysis_worker_mode in {"external", "remote"}
        and status.get("reason") == "worker_heartbeat_missing"
    )
    return {"accepted": deferrable, "deferred": deferrable}


def _require_analysis_capacity() -> dict:
    decision = _analysis_queue_decision()
    if not decision["accepted"]:
        raise HTTPException(
            503,
            "The fight-analysis worker is temporarily unavailable. Your video and fighter selection are preserved.",
        )
    return decision


def _wake_analysis_worker(job_id: str) -> None:
    """Rouse the analysis machine in the background, never blocking the upload.

    Two mechanisms, either or both: a webhook that starts a scale-to-zero GPU,
    and a Wake-on-LAN packet for a machine that sleeps between fights. Both are
    accelerators only -- the fight is queued durably either way.
    """
    if not SETTINGS.worker_wake_url and not (SETTINGS.wol_mac and SETTINGS.wol_host):
        return

    def run() -> None:
        from core.worker_client import send_magic_packet, wake_remote_worker

        if SETTINGS.wol_mac and SETTINGS.wol_host:
            # Sent twice: a card waking from a cold sleep routinely misses the
            # first packet, and a duplicate costs 102 bytes.
            sent = any(
                send_magic_packet(SETTINGS.wol_mac, SETTINGS.wol_host, SETTINGS.wol_port)
                for _ in range(2)
            )
            LOGGER.info("analysis_worker_wol job_id=%s sent=%s", job_id, sent)
        if SETTINGS.worker_wake_url:
            woken = wake_remote_worker(SETTINGS.worker_wake_url, SETTINGS.worker_token, job_id)
            LOGGER.info("analysis_worker_wake job_id=%s delivered=%s", job_id, woken)

    threading.Thread(target=run, name=f"wiq-wake-{job_id}", daemon=True).start()


def _analysis_started_response(request: Request, job_id: str, deferred: bool = False) -> JSONResponse:
    response = JSONResponse({
        "ok": True,
        "progress_url": f"/progress/{job_id}",
        "deferred": deferred,
        **({"notice": DEFERRED_ANALYSIS_MESSAGE} if deferred else {}),
    })
    response.set_cookie(
        ACTIVE_ANALYSIS_COOKIE, job_id, max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", secure=_request_is_secure(request),
    )
    return response


def _auth_page(request: Request, mode: str, error: str = "", next_path: str = "/dashboard"):
    return templates.TemplateResponse(
        request=request,
        name="auth.html",
        context={"request": request, "mode": mode, "error": error, "next_path": _safe_next(next_path)},
        status_code=400 if error else 200,
    )


def _oauth_redirect_uri(request: Request, provider: str) -> str:
    base = SETTINGS.public_base_url or _external_origin(request)
    return f"{base}/auth/{provider}/callback"


async def _oauth_callback_state(request: Request) -> str:
    if request.method == "GET":
        return str(request.query_params.get("state") or "")
    form = await request.form()
    return str(form.get("state") or "")


def _social_auth_error(request: Request, intent: dict | None, message: str):
    intent = intent or {}
    return _auth_page(
        request,
        str(intent.get("mode") or "login"),
        message,
        str(intent.get("next_path") or "/dashboard"),
    )


@app.post("/auth/{provider}/start")
async def social_auth_start(
    request: Request,
    provider: str,
    mode: str = Form("login"),
    next_path: str = Form("/dashboard"),
    accept_terms: bool = Form(False),
    age_confirmed: bool = Form(False),
    accept_policies: bool = Form(False),
    marketing_consent: bool = Form(False),
):
    _enforce_rate_limit(request, "social-auth-start", 30, 300)
    if _account(request):
        return RedirectResponse(_safe_next(next_path), status_code=303)
    if mode not in {"signup", "login"}:
        raise HTTPException(400, "Choose sign in or account creation.")
    client = SOCIAL_AUTH.client(provider)
    if not client:
        raise HTTPException(404, "This sign-in provider is not configured.")
    if mode == "signup" and (not accept_terms or not age_confirmed):
        return _auth_page(
            request,
            "signup",
            f"Confirm that you are at least {SETTINGS.minimum_account_age} and accept the Terms of Service and Privacy Policy.",
            next_path,
        )
    if mode == "login" and not accept_policies:
        return _auth_page(
            request,
            "login",
            "Confirm the Terms, Privacy Policy, and Acceptable Use Policy to sign in.",
            next_path,
        )
    authorize_options = {"response_mode": "form_post"} if provider == "apple" else {}
    response = await client.authorize_redirect(
        request, _oauth_redirect_uri(request, provider), **authorize_options,
    )
    state = str((parse_qs(urlsplit(response.headers.get("location", "")).query).get("state") or [""])[0])
    if not state:
        LOGGER.error("social_auth_state_missing provider=%s", provider)
        return _auth_page(request, mode, "Secure sign-in could not start. Please try again.", next_path)
    request.session[f"wiq_social_intent:{state}"] = {
        "provider": provider,
        "mode": mode,
        "next_path": _safe_next(next_path),
        "marketing_consent": bool(marketing_consent),
        "created_at_epoch": time.time(),
    }
    return response


@app.api_route("/auth/{provider}/callback", methods=["GET", "POST"])
async def social_auth_callback(request: Request, provider: str):
    _enforce_rate_limit(request, "social-auth-callback", 40, 300)
    state = await _oauth_callback_state(request)
    intent = request.session.pop(f"wiq_social_intent:{state}", None) if state else None
    if (
        not isinstance(intent, dict)
        or intent.get("provider") != provider
        or time.time() - float(intent.get("created_at_epoch", 0.0) or 0.0) > 600
    ):
        return _social_auth_error(
            request, intent, "This secure sign-in attempt expired or was already used. Please start again."
        )
    client = SOCIAL_AUTH.client(provider)
    if not client:
        return _social_auth_error(request, intent, "This sign-in provider is not available.")
    try:
        token = await client.authorize_access_token(request)
        identity = await SOCIAL_AUTH.identity_from_token(provider, client, token)
    except (OAuthError, ValueError) as exc:
        LOGGER.warning("social_auth_rejected provider=%s error=%s", provider, type(exc).__name__)
        return _social_auth_error(request, intent, "The identity provider could not verify this sign-in.")
    except Exception as exc:
        LOGGER.warning("social_auth_unavailable provider=%s error=%s", provider, type(exc).__name__)
        return _social_auth_error(request, intent, "Secure sign-in is temporarily unavailable. Please try again.")

    account = get_account_for_oauth_identity(identity.provider, identity.subject)
    created = False
    if not account:
        if intent["mode"] != "signup":
            return _social_auth_error(
                request, intent,
                f"No WarriorIQ account is connected to this {provider.title()} identity yet. Choose Create account first.",
            )
        if not identity.email or not valid_email(identity.email):
            return _social_auth_error(
                request, intent,
                "The identity provider did not share a usable email address. Allow email access or use email signup.",
            )
        try:
            account = create_oauth_account(
                identity.provider,
                identity.subject,
                identity.email,
                hash_password(session_token() + session_token()),
                identity.display_name,
            )
        except ValueError as exc:
            return _social_auth_error(request, intent, str(exc))
        created = True
        record_account_signup_acceptance(
            int(account["id"]),
            terms_version=SETTINGS.policy_version,
            privacy_version=SETTINGS.policy_version,
            marketing_consent=bool(intent.get("marketing_consent")),
        )
        for kind, status in (
            ("terms_acceptance", "accepted"),
            ("privacy_acknowledgement", "accepted"),
            ("age_18_plus_confirmation", "accepted"),
            ("marketing_consent", "accepted" if intent.get("marketing_consent") else "declined"),
        ):
            record_legal_acceptance(
                kind,
                SETTINGS.policy_version,
                profile_id=int(account["profile_id"]),
                metadata={
                    "source": "social_signup",
                    "provider": identity.provider,
                    "enabled": bool(intent.get("marketing_consent")) if kind == "marketing_consent" else True,
                },
                current_status=status,
            )
        record_security_event(
            "account_created",
            account_id=int(account["id"]),
            metadata={"policy_version": SETTINGS.policy_version, "provider": identity.provider},
        )
    else:
        record_legal_acceptance(
            "account_signin_policies",
            SETTINGS.policy_version,
            profile_id=int(account["profile_id"]),
            metadata={"source": "social_login", "provider": identity.provider},
        )
    if identity.email_verified and not account.get("email_verified_at"):
        mark_email_verified(int(account["id"]))
        account = get_account(int(account["id"])) or account
    if SETTINGS.require_email_verification and not account.get("email_verified_at"):
        delivered = _send_verification_email(request, account)
        record_security_event(
            "email_verification_requested", account_id=int(account["id"]),
            metadata={"email_delivery": "sent" if delivered else "unavailable", "provider": identity.provider},
        )
        return RedirectResponse("/verify-email", status_code=303)
    record_security_event(
        "social_login_succeeded",
        account_id=int(account["id"]),
        metadata={"provider": identity.provider, "created": created},
    )
    response = RedirectResponse(_safe_next(str(intent.get("next_path"))), status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        issue_session(int(account["id"])),
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="lax",
        secure=_request_is_secure(request),
    )
    return response


def _send_verification_email(request: Request, account: dict) -> bool:
    token = session_token()
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    save_email_verification_token(int(account["id"]), token_digest(token), expires)
    base = SETTINGS.public_base_url or str(request.base_url).rstrip("/")
    verify_url = f"{base}/verify-email/{token}"
    try:
        return send_transactional_email(
            account["email"], "Verify your WarriorIQ email",
            f"Verify your private WarriorIQ workspace within 24 hours:\n\n{verify_url}\n\nIf you did not create this account, ignore this message.",
        )
    except Exception:
        return False


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, next: str = "/dashboard"):
    if _account(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return _auth_page(request, "signup", next_path=next)


@app.post("/signup")
def signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next_path: str = Form("/dashboard"),
    accept_terms: bool = Form(False),
    age_confirmed: bool = Form(False),
    marketing_consent: bool = Form(False),
):
    _enforce_rate_limit(request, "signup", 20, 300)
    if not accept_terms or not age_confirmed:
        return _auth_page(
            request, "signup",
            f"Confirm that you are at least {SETTINGS.minimum_account_age} and accept the Terms of Service and Privacy Policy.",
            next_path,
        )
    try:
        account = register(email, password)
    except ValueError as exc:
        return _auth_page(request, "signup", str(exc), next_path)
    record_account_signup_acceptance(
        int(account["id"]), terms_version=SETTINGS.policy_version,
        privacy_version=SETTINGS.policy_version, marketing_consent=bool(marketing_consent),
    )
    for kind, status in (
        ("terms_acceptance", "accepted"),
        ("privacy_acknowledgement", "accepted"),
        ("age_18_plus_confirmation", "accepted"),
        ("marketing_consent", "accepted" if marketing_consent else "declined"),
    ):
        record_legal_acceptance(
            kind, SETTINGS.policy_version, profile_id=int(account["profile_id"]),
            metadata={"source": "signup", "enabled": bool(marketing_consent) if kind == "marketing_consent" else True},
            current_status=status,
        )
    record_security_event("account_created", account_id=int(account["id"]), metadata={"policy_version": SETTINGS.policy_version})
    if SETTINGS.require_email_verification:
        delivered = _send_verification_email(request, account)
        record_security_event(
            "email_verification_requested", account_id=int(account["id"]),
            metadata={"email_delivery": "sent" if delivered else "unavailable"},
        )
        return RedirectResponse("/verify-email", status_code=303)
    mark_email_verified(int(account["id"]))
    response = RedirectResponse(_safe_next(next_path), status_code=303)
    response.set_cookie(
        SESSION_COOKIE, issue_session(int(account["id"])), max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", secure=_request_is_secure(request),
    )
    return response


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/dashboard"):
    if _account(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return _auth_page(request, "login", next_path=next)


@app.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next_path: str = Form("/dashboard"),
    accept_policies: bool = Form(False),
):
    _enforce_rate_limit(request, "login", 30, 300)
    if not accept_policies:
        return _auth_page(
            request, "login",
            "Confirm the Terms, Privacy Policy, and Acceptable Use Policy to sign in.",
            next_path,
        )
    account = authenticate(email, password)
    if not account:
        record_security_event("login_failed", severity="warning", metadata={"email_hash": token_digest(email.strip().lower())[:16]})
        return _auth_page(request, "login", "The email or password is incorrect.", next_path)
    if SETTINGS.require_email_verification and not account.get("email_verified_at"):
        return _auth_page(
            request, "login",
            "Verify your email before signing in. You can request a fresh verification link below.",
            next_path,
        )
    record_legal_acceptance(
        "account_signin_policies", SETTINGS.policy_version,
        profile_id=int(account["profile_id"]),
        metadata={"source": "login"},
    )
    response = RedirectResponse(_safe_next(next_path), status_code=303)
    record_security_event("login_succeeded", account_id=int(account["id"]))
    response.set_cookie(
        SESSION_COOKIE, issue_session(int(account["id"])), max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", secure=_request_is_secure(request),
    )
    return response


@app.get("/verify-email", response_class=HTMLResponse)
def verify_email_page(request: Request, message: str = ""):
    return templates.TemplateResponse(
        request=request, name="verify_email.html",
        context={"request": request, "message": message},
    )


@app.get("/verify-email/{token}")
def verify_email_token(request: Request, token: str):
    account_id = consume_email_verification_token(token_digest(token))
    if account_id is None:
        raise HTTPException(410, "This email-verification link is invalid or expired.")
    record_security_event("email_verified", account_id=account_id)
    return RedirectResponse("/login?verified=1", status_code=303)


@app.post("/verify-email", response_class=HTMLResponse)
def resend_verification_email(request: Request, email: str = Form(...)):
    _enforce_rate_limit(request, "email-verification", 5, 3600)
    account = get_account_by_email(email.strip().lower()) if valid_email(email) else None
    if account and not account.get("email_verified_at"):
        delivered = _send_verification_email(request, account)
        record_security_event(
            "email_verification_resent", account_id=int(account["id"]),
            metadata={"email_delivery": "sent" if delivered else "unavailable"},
        )
    return verify_email_page(
        request,
        "If an unverified account exists, a fresh verification link has been sent.",
    )


@app.post("/logout")
def logout(request: Request):
    end_session(request.cookies.get(SESSION_COOKIE))
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="password_reset.html",
        context={"request": request, "token": "", "message": ""},
    )


@app.post("/forgot-password", response_class=HTMLResponse)
def request_password_reset(request: Request, email: str = Form(...)):
    _enforce_rate_limit(request, "password-reset", 8, 900)
    account = get_account_by_email(email.strip().lower()) if valid_email(email) else None
    if account and account.get("account_status", "active") == "active":
        token = session_token()
        expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        save_password_reset_token(int(account["id"]), token_digest(token), expires)
        reset_url = f"{str(request.base_url).rstrip('/')}/reset-password/{token}"
        try:
            delivered = send_transactional_email(
                account["email"], "Reset your WarriorIQ password",
                f"Use this one-time link within 30 minutes:\n\n{reset_url}\n\nIf you did not request this, ignore this message.",
            )
        except Exception:
            delivered = False
        record_security_event(
            "password_reset_requested", account_id=int(account["id"]),
            metadata={"email_delivery": "sent" if delivered else "unavailable"},
        )
    return templates.TemplateResponse(
        request=request, name="password_reset.html",
        context={
            "request": request, "token": "",
            "message": "If an eligible account exists, a time-limited reset link has been queued for its email provider.",
        },
    )


@app.get("/reset-password/{token}", response_class=HTMLResponse)
def reset_password_page(request: Request, token: str):
    return templates.TemplateResponse(
        request=request, name="password_reset.html",
        context={"request": request, "token": token, "message": ""},
    )


@app.post("/reset-password/{token}")
def reset_password(request: Request, token: str, password: str = Form(...)):
    if not valid_password(password):
        raise HTTPException(400, "Password must contain between 10 and 1,024 characters.")
    account_id = consume_password_reset_token(token_digest(token))
    if account_id is None or not update_password_hash(account_id, hash_password(password)):
        raise HTTPException(410, "This password-reset link is invalid or expired.")
    record_security_event("password_reset_completed", account_id=account_id)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    profile_id = _profile_id(request)
    profile = get_profile(profile_id) if profile_id is not None else None
    account = _account(request)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "sports": RULESET_SPORTS,
            "sport_unobserved": {sport: sport_unobserved(sport) for sport in SPORTS},
            "profile": profile,
            "version": SETTINGS.version,
            "allowance": analysis_allowance(int(account["id"])) if account else None,
        },
    )


def _sport_context(request: Request, sport: str) -> dict:
    """Everything a single sport's setup page needs to describe itself."""
    account = _account(request)
    keys = SPORTS[sport]
    return {
        "request": request,
        "sport": sport,
        "sport_label": RULESET_SPORTS[sport],
        "sports": RULESET_SPORTS,
        # Boxing and MMA each have exactly one ruleset, so asking which one is a
        # question with a single answer. The page drops the field and posts the
        # key instead of showing a select the reader cannot get wrong.
        "rulesets": [(key, RULESET_LABELS[key]) for key in keys],
        "only_ruleset": keys[0] if len(keys) == 1 else None,
        "unobserved": sport_unobserved(sport),
        "version": SETTINGS.version,
        "allowance": analysis_allowance(int(account["id"])) if account else None,
    }


def _ruleset_summary(sport: str) -> list[str]:
    """Short ruleset names for a chooser card.

    Kickboxing has six disciplines and taekwondo two federations with long
    formal names; a card five across cannot carry either in full. The complete
    names are on the sport's own page, where the choice is actually made.
    """
    keys = SPORTS[sport]
    if len(keys) == 1 and RULESET_LABELS[keys[0]] == RULESET_SPORTS[sport]:
        # Boxing's one ruleset is named after the sport, so listing it says
        # nothing. Say what is true of the sport instead.
        return ["One unified ruleset"]
    names = [RULESET_SHORT.get(key, RULESET_LABELS[key]) for key in keys]
    if len(names) > 3:
        names = names[:3] + [f"+{len(names) - 3} more"]
    return names


@app.get("/analyze", response_class=HTMLResponse)
def choose_sport(request: Request):
    """The five sports, as the first real decision in the flow.

    Rules, legal targets and what the analysis can observe all follow from the
    sport, so it is asked first and asked on its own rather than as one field
    among ten on a form the reader has already started filling in.
    """
    account = _account(request)
    return templates.TemplateResponse(
        request=request,
        name="sports.html",
        context={
            "request": request,
            # An exhausted allowance is worth knowing before a sport is picked
            # and a video chosen, not two pages later.
            "allowance": analysis_allowance(int(account["id"])) if account else None,
            "sports": RULESET_SPORTS,
            # Boxing's single ruleset is named after the sport, so listing it
            # tells the reader nothing; say what is actually true instead.
            "sport_rulesets": {s: _ruleset_summary(s) for s in SPORTS},
            "sport_unobserved": {sport: sport_unobserved(sport) for sport in SPORTS},
            "version": SETTINGS.version,
        },
    )


@app.get("/analyze/{sport}", response_class=HTMLResponse)
def sport_setup(request: Request, sport: str):
    key = (sport or "").strip().lower()
    if key not in SPORTS:
        raise HTTPException(status_code=404, detail="Unknown sport")
    return templates.TemplateResponse(
        request=request, name="analyze.html", context=_sport_context(request, key),
    )


@app.post("/upload")
async def upload(
    request: Request,
    video: UploadFile = File(...),
    fight_type: str = Form("competition"),
    analysis_target: str = Form("BOTH"),
    ruleset: str = Form("K1"),
    start_seconds: float = Form(0.0),
    end_seconds: str = Form(""),
    round_count: int = Form(3),
    round_duration_seconds: float = Form(120.0),
    break_duration_seconds: float = Form(60.0),
    selected_rounds: str = Form("ALL"),
    openai_identity_recovery: bool = Form(False),
    rights_confirmed: bool = Form(False),
    people_permissions_confirmed: bool = Form(False),
    minor_permission_status: str = Form(""),
):
    _enforce_rate_limit(request, "fight-upload", 12, 600)
    minor_permission_status = minor_permission_status.strip().lower()
    if (
        not rights_confirmed
        or not people_permissions_confirmed
        or minor_permission_status not in {"no_minors", "guardian_authorized"}
    ):
        raise HTTPException(
            400,
            "Confirm your footage rights, permission for people shown, and the minor/guardian status before upload.",
        )
    account = _account(request)
    if not account:
        # Analysis is account-only. Fight footage carries identifiable athletes
        # and a per-plan allowance, neither of which can be attached to an
        # anonymous browser session; the 401 lets the upload form send the
        # visitor to sign-in without losing what they filled in.
        raise HTTPException(401, "Create a free account or sign in to analyse a fight.")
    allowance = analysis_allowance(int(account["id"]))
    if allowance["remaining"] == 0:
        raise HTTPException(429, f"Your {allowance['plan']['label']} allowance is used for this period. It will reset automatically.")
    if not video.filename:
        raise HTTPException(400, "Choose a fight video.")
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(video.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}:
        raise HTTPException(400, "Unsupported video format.")

    video_path = UPLOADS / f"{job_id}{suffix}"
    # UploadFile uses a spooled file. Keep the blocking disk copy outside the
    # event loop so one large phone upload cannot freeze every other request.
    await run_in_threadpool(_save_upload_limited, video, video_path, MAX_FIGHT_BYTES)

    scan = await run_in_threadpool(scan_upload, video_path)
    if not scan["clean"]:
        video_path.unlink(missing_ok=True)
        if scan["status"] == "infected":
            record_security_event("malware_upload_blocked", severity="warning")
            raise HTTPException(400, "This file did not pass the upload safety scan.")
        raise HTTPException(503, "Fight uploads are paused because the safety scanner is unavailable.")

    try:
        info = await run_in_threadpool(get_video_info, video_path)
        if info.duration > SETTINGS.max_video_duration_seconds:
            raise HTTPException(413, "This video is longer than the configured analysis limit.")
        if info.width * info.height > SETTINGS.max_video_pixels:
            raise HTTPException(413, "This video's resolution exceeds the configured processing limit.")
        quality = await run_in_threadpool(inspect_video_quality, video_path, info)
    except Exception:
        video_path.unlink(missing_ok=True)
        raise
    start = max(0.0, min(float(start_seconds), max(0.0, info.duration - 0.001)))
    end = None if not end_seconds.strip() else max(start, min(float(end_seconds), info.duration))
    count = max(1, min(20, int(round_count)))
    selection_frame = int(round(start * info.fps))
    job_dir = OUTPUTS / job_id
    selection_path = job_dir / "selection.jpg"
    try:
        frame = await run_in_threadpool(read_frame, video_path, selection_frame)
        job_dir.mkdir(parents=True, exist_ok=True)
        if not await run_in_threadpool(cv2.imwrite, str(selection_path), frame):
            raise OSError("OpenCV could not save the fighter-selection frame")
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        shutil.rmtree(job_dir, ignore_errors=True)
        LOGGER.warning("upload_selection_frame_failed job_id=%s error=%s", job_id, type(exc).__name__)
        raise HTTPException(422, "WarriorIQ could not prepare this video's fighter-selection frame.") from exc

    profile_id = int(account["profile_id"]) if account else 0
    if not account:
        mark_guest_job(job_id, request.state.guest_id, str(video_path))

    create_job(
        job_id,
        {
            "video_path": str(video_path),
            "original_name": video.filename,
            "fight_type": fight_type.lower(),
            "analysis_target": analysis_target.upper(),
            "ruleset": normalize_ruleset(ruleset),
            "start_seconds": start,
            "end_seconds": end,
            "round_count": count,
            "round_duration_seconds": float(round_duration_seconds),
            "break_duration_seconds": float(break_duration_seconds),
            "selected_rounds": _parse_rounds(selected_rounds, count),
            "video_width": info.width,
            "video_height": info.height,
            "video_duration": info.duration,
            "selection_frame": selection_frame,
            "profile_id": profile_id,
            "account_id": int(account["id"]) if account else None,
            "persist_result": bool(account),
            "owner_key": _owner_key(request),
            "quality": quality,
            "upload_scan_status": scan["status"],
            "openai_identity_recovery": bool(openai_identity_recovery),
        },
    )
    acceptance_owner = {"profile_id": int(account["profile_id"])} if account else {"guest_id": request.state.guest_id}
    record_legal_acceptance(
        "fight_video_upload_permission", SETTINGS.policy_version,
        resource_id=job_id,
        metadata={
            "rights_confirmed": True,
            "people_permissions_confirmed": True,
            "minor_permission_status": minor_permission_status,
            "ruleset": normalize_ruleset(ruleset),
            "external_ai_enabled": bool(openai_identity_recovery),
            "private_by_default": True,
        },
        **acceptance_owner,
    )
    record_security_event(
        "fight_video_uploaded", account_id=int(account["id"]) if account else None,
        resource_type="fight", resource_id=job_id,
        metadata={"guest": not bool(account), "minor_permission_status": minor_permission_status},
    )
    if openai_identity_recovery:
        record_legal_acceptance(
            "external_ai_frame_processing", SETTINGS.policy_version,
            resource_id=job_id,
            metadata={"provider": "OpenAI", "purpose": "fighter_identity_recovery"},
            **acceptance_owner,
        )
    next_url = f"/frame/{job_id}"
    if "application/json" in request.headers.get("accept", ""):
        response = JSONResponse({"job_id": job_id, "next_url": next_url}, status_code=201)
    else:
        response = RedirectResponse(next_url, status_code=303)
    response.set_cookie(
        ACTIVE_ANALYSIS_COOKIE, job_id, max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", secure=_request_is_secure(request),
    )
    return response


@app.get("/frame/{job_id}", response_class=HTMLResponse)
def frame_page(request: Request, job_id: str):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    return templates.TemplateResponse(request=request, name="frame.html", context={"request": request, "job_id": job_id, "job": job})


@app.get("/select/{job_id}", response_class=HTMLResponse)
def select_page(request: Request, job_id: str):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    return templates.TemplateResponse(request=request, name="select.html", context={"request": request, "job_id": job_id, "job": job})


@app.get("/selection-image/{job_id}")
def selection_image(request: Request, job_id: str):
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    path = OUTPUTS / job_id / "selection.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/selection-frame/{job_id}")
def set_selection_frame(request: Request, job_id: str, payload: SelectionFramePayload):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    seconds = max(0.0, min(float(payload.seconds), max(0.0, float(job["video_duration"]) - 0.001)))
    info = get_video_info(job["video_path"])
    frame_number = int(round(seconds * info.fps))
    frame = read_frame(job["video_path"], frame_number)
    if not cv2.imwrite(str(OUTPUTS / job_id / "selection.jpg"), frame):
        raise HTTPException(500, "Could not save the selected fighter frame.")
    update_job(job_id, {"selection_frame": frame_number, "start_seconds": seconds})
    return {"ok": True, "seconds": seconds, "frame": frame_number}


@app.get("/api/detect/{job_id}")
def detect_people(request: Request, job_id: str):
    """Return optional person candidates without blocking manual selection."""
    job = _authorized_job(request, job_id)
    path = OUTPUTS / job_id / "selection.jpg"
    if not job or not path.exists():
        raise HTTPException(404)
    if not SETTINGS.selection_detection_enabled:
        return {
            "people": [], "width": job["video_width"], "height": job["video_height"],
            "availability": "manual_only",
        }
    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(500, "Could not read selection image")
    try:
        tracker = _get_pose_tracker()
        tracker.warmup(frame)
        results = tracker.model.predict(
            frame,
            device=tracker.device,
            imgsz=SETTINGS.default_imgsz,
            conf=SETTINGS.detection_conf,
            classes=[0],
            verbose=False,
        )
    except Exception as exc:
        LOGGER.warning("Selection candidate detection unavailable: %s", type(exc).__name__)
        return {
            "people": [], "width": job["video_width"], "height": job["video_height"],
            "availability": "manual_only",
        }
    boxes = []
    result = results[0]
    if result.boxes is not None:
        for box, conf in zip(result.boxes.xyxy.detach().cpu().numpy(), result.boxes.conf.detach().cpu().numpy()):
            boxes.append({"box": [float(x) for x in box], "confidence": float(conf)})
    return {
        "people": boxes, "width": job["video_width"], "height": job["video_height"],
        "availability": "candidates_ready",
    }


def _validated_fighter_box(box: list[float], width: float, height: float, label: str) -> list[float]:
    if len(box) != 4 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in box
    ):
        raise HTTPException(400, f"{label} needs four valid coordinates.")
    x1, y1, x2, y2 = (float(value) for value in box)
    tolerance = 1e-3
    if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
        raise HTTPException(400, f"Keep the {label} box inside the video frame.")
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(400, f"Draw the {label} box from one corner to the opposite corner.")
    minimum_width = max(16.0, width * 0.025)
    minimum_height = max(32.0, height * 0.10)
    if x2 - x1 < minimum_width or y2 - y1 < minimum_height:
        raise HTTPException(400, f"Draw a larger full-body box around {label}.")
    return [x1, y1, x2, y2]


_WORKER_PROGRESS_FIELDS = {
    "percent", "message", "stage", "elapsed_seconds", "eta_seconds",
    "processed_video_seconds", "video_duration_seconds", "fighter_a_confidence",
    "fighter_b_confidence", "current_round", "quality_mode", "live_event_mode",
    "live_events", "provisional_stats", "latest_observation",
}
_WORKER_ARCHIVE_FILES = {"report.json", "tracking.jsonl", "events.json"}
_WORKER_REPORT_KEYS = {
    "video", "setup", "performance", "tracking", "classifier", "metrics",
    "scorecard", "coaching", "training_plan", "integrity", "statistics",
}


def _require_remote_worker(request: Request) -> None:
    if SETTINGS.analysis_worker_mode != "remote" or not SETTINGS.worker_token:
        raise HTTPException(503, "Remote analysis workers are not configured.")
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, SETTINGS.worker_token):
        raise HTTPException(401, "Worker authentication failed.", headers={"WWW-Authenticate": "Bearer"})


def _validated_worker_id(value: str) -> str:
    worker_id = str(value or "").strip()
    if not 3 <= len(worker_id) <= 128 or any(
        not (character.isalnum() or character in "-_.") for character in worker_id
    ):
        raise HTTPException(400, "Invalid worker identity.")
    return worker_id


def _owned_worker_job(job_id: str, worker_id: str, analysis_run_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Analysis job not found.")
    if (
        job.get("status") != "running"
        or job.get("worker_id") != worker_id
        or job.get("analysis_run_id") != analysis_run_id
    ):
        raise HTTPException(409, "This worker no longer owns the analysis generation.")
    return job


def _remote_job_payload(job_id: str, job: dict) -> dict:
    return {
        "job_id": job_id,
        "analysis_run_id": str(job.get("analysis_run_id") or ""),
        "video_extension": Path(str(job.get("video_path") or "fight.mp4")).suffix.lower() or ".mp4",
        "fighter_a_box": list(job["fighter_a_box"]),
        "fighter_b_box": list(job["fighter_b_box"]),
        "analysis_target": "BOTH",
        "focus_fighter": job.get("focus_fighter") or "A",
        "fight_type": job["fight_type"],
        "ruleset": job["ruleset"],
        "start_seconds": float(job.get("start_seconds", 0.0)),
        "end_seconds": job.get("end_seconds"),
        "round_count": int(job.get("round_count", 1)),
        "round_duration_seconds": float(job.get("round_duration_seconds", 120.0)),
        "break_duration_seconds": float(job.get("break_duration_seconds", 60.0)),
        "selected_rounds": job.get("selected_rounds"),
        "openai_identity_recovery": bool(job.get("openai_identity_recovery", False)),
    }


def _prepare_worker_archive(archive_path: Path, job_dir: Path, analysis_run_id: str) -> tuple[dict, dict[str, Path]]:
    staged: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            files = [info for info in bundle.infolist() if not info.is_dir()]
            names = [info.filename for info in files]
            if len(names) != len(set(names)) or set(names) - _WORKER_ARCHIVE_FILES:
                raise ValueError("The worker archive contains unsupported files.")
            if not {"report.json", "tracking.jsonl"}.issubset(names):
                raise ValueError("The worker archive is missing the report or skeleton tracking data.")
            if sum(info.file_size for info in files) > SETTINGS.worker_artifact_max_bytes:
                raise ValueError("The worker archive expands beyond the configured safety limit.")
            report = json.loads(bundle.read("report.json"))
            if not isinstance(report, dict) or not _WORKER_REPORT_KEYS.issubset(report):
                raise ValueError("The worker report is incomplete.")
            for name in ("tracking.jsonl", "events.json"):
                if name not in names:
                    continue
                destination = job_dir / f"{name}.{analysis_run_id}.worker"
                with bundle.open(name) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                staged[name] = destination
        return report, staged
    except Exception:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise


def _save_remote_fight(job_id: str, job: dict, report: dict) -> None:
    if not job.get("persist_result"):
        return
    performance = report.get("performance", {})
    tracking = report.get("tracking", {})
    scorecard = report.get("scorecard", {})
    summary = {
        "winner_estimate": scorecard.get("winner_estimate"),
        "score_totals": scorecard.get("totals", {"A": None, "B": None}),
        "analysis_seconds": performance.get("analysis_seconds"),
        "video_seconds": performance.get("segment_duration_seconds"),
        "within_budget": performance.get("within_video_length_budget"),
        "fighter_A_coverage": tracking.get("fighter_A_coverage", 0.0),
        "fighter_B_coverage": tracking.get("fighter_B_coverage", 0.0),
        "progress_report": {
            key: report.get(key, {})
            for key in ("video", "setup", "integrity", "metrics", "statistics", "coaching", "training_plan")
        },
    }
    save_fight(
        job_id=job_id,
        profile_id=int(job.get("profile_id", 0)),
        original_name=str(job.get("original_name") or "Fight video"),
        video_path=str(job["video_path"]),
        report_path=str(OUTPUTS / job_id / "report.json"),
        fight_type=str(job["fight_type"]),
        ruleset=str(job["ruleset"]),
        analysis_target=str(job.get("focus_fighter") or "A"),
        summary=summary,
        video_delete_after=(
            datetime.now(timezone.utc) + timedelta(days=SETTINGS.saved_video_retention_days)
        ).isoformat(),
    )


@app.post("/api/worker/heartbeat")
def remote_worker_heartbeat(request: Request, payload: WorkerIdentityPayload):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(payload.worker_id)
    record_worker_heartbeat(worker_id)
    return {"ok": True}


@app.post("/api/worker/claim")
def remote_worker_claim(request: Request, payload: WorkerIdentityPayload):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(payload.worker_id)
    record_worker_heartbeat(worker_id)
    claimed = claim_next_job(worker_id)
    if not claimed:
        return {"job": None}
    job_id, job = claimed
    record_worker_heartbeat(worker_id, job_id)
    return {"job": _remote_job_payload(job_id, job)}


@app.get("/api/worker/jobs/{job_id}/video")
def remote_worker_video(request: Request, job_id: str, worker_id: str, analysis_run_id: str):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(worker_id)
    job = _owned_worker_job(job_id, worker_id, analysis_run_id)
    video_path = Path(str(job.get("video_path") or ""))
    if not video_path.is_file() or video_path.parent.resolve() != UPLOADS.resolve():
        raise HTTPException(404, "Fight video not found.")
    record_worker_heartbeat(worker_id, job_id)
    return FileResponse(video_path, filename=f"{job_id}{video_path.suffix.lower()}", media_type="video/mp4")


@app.post("/api/worker/jobs/{job_id}/progress")
def remote_worker_progress(request: Request, job_id: str, payload: WorkerProgressPayload):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(payload.worker_id)
    _owned_worker_job(job_id, worker_id, payload.analysis_run_id)
    if len(json.dumps(payload.patch, separators=(",", ":"))) > 2 * 1024 * 1024:
        raise HTTPException(413, "Worker progress payload is too large.")
    patch = {key: value for key, value in payload.patch.items() if key in _WORKER_PROGRESS_FIELDS}
    if "percent" in patch:
        patch["percent"] = max(0.0, min(99.9, float(patch["percent"])))
    if not update_job_for_worker(job_id, worker_id, payload.analysis_run_id, patch):
        raise HTTPException(409, "This worker no longer owns the analysis generation.")
    record_worker_heartbeat(worker_id, job_id)
    return {"ok": True}


@app.post("/api/worker/jobs/{job_id}/complete", status_code=201)
async def remote_worker_complete(
    request: Request,
    job_id: str,
    archive: UploadFile = File(...),
    worker_id: str = Form(...),
    analysis_run_id: str = Form(...),
):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(worker_id)
    existing = get_job(job_id)
    if (
        existing
        and existing.get("status") == "complete"
        and existing.get("worker_id") == worker_id
        and existing.get("analysis_run_id") == analysis_run_id
        and (OUTPUTS / job_id / "report.json").is_file()
    ):
        # A worker may retry after the web server committed the result but the
        # success response was lost. Treat that exact generation as complete;
        # never re-run it or turn a successful analysis into an error.
        record_worker_heartbeat(worker_id)
        return {"ok": True, "job_id": job_id, "already_complete": True}
    job = _owned_worker_job(job_id, worker_id, analysis_run_id)
    job_dir = OUTPUTS / job_id
    archive_path = job_dir / f"worker-result.{analysis_run_id}.zip"
    try:
        await run_in_threadpool(
            _save_upload_limited, archive, archive_path, SETTINGS.worker_artifact_max_bytes,
        )
        try:
            report, staged = await run_in_threadpool(
                _prepare_worker_archive, archive_path, job_dir, analysis_run_id,
            )
        except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise HTTPException(400, str(exc) or "Invalid worker artifact archive.") from exc
        if not finalize_job_from_worker(job_id, worker_id, analysis_run_id, report, staged):
            for path in staged.values():
                path.unlink(missing_ok=True)
            raise HTTPException(409, "This worker no longer owns the analysis generation.")
        try:
            _save_remote_fight(job_id, job, report)
        except Exception as exc:
            LOGGER.exception("remote_worker_fight_persist_failed job_id=%s", job_id, exc_info=exc)
        record_worker_heartbeat(worker_id)
        record_security_event(
            "remote_analysis_completed", account_id=job.get("account_id"),
            resource_type="fight", resource_id=job_id,
        )
        return {"ok": True, "job_id": job_id}
    finally:
        archive_path.unlink(missing_ok=True)


@app.post("/api/worker/jobs/{job_id}/failed")
def remote_worker_failed(request: Request, job_id: str, payload: WorkerFailurePayload):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(payload.worker_id)
    job = _owned_worker_job(job_id, worker_id, payload.analysis_run_id)
    if job.get("usage_reserved") and job.get("account_id"):
        release_analysis(int(job["account_id"]), job_id)
        update_job(job_id, {"usage_reserved": False})
    if not update_job_for_worker(job_id, worker_id, payload.analysis_run_id, {
        "status": "error",
        "message": "WarriorIQ could not finish this analysis. Your upload and fighter selections are preserved so you can try again.",
        "worker_lease_expires_epoch": None,
        "worker_error_code": str(payload.error_code or "analysis_failed")[:80],
    }, renew_lease=False):
        raise HTTPException(409, "This worker no longer owns the analysis generation.")
    record_worker_heartbeat(worker_id)
    return {"ok": True}


@app.post("/api/start/{job_id}")
def start(request: Request, job_id: str, payload: StartPayload):
    _enforce_rate_limit(request, "analysis-start", 12, 600)
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    if job.get("status") in {"queued", "running"}:
        return _analysis_started_response(request, job_id)
    capacity = _require_analysis_capacity()
    video_width = float(job.get("video_width") or 0)
    video_height = float(job.get("video_height") or 0)
    if video_width <= 0 or video_height <= 0:
        selection = cv2.imread(str(OUTPUTS / job_id / "selection.jpg"))
        if selection is None:
            raise HTTPException(409, "The selected frame is unavailable. Choose another frame.")
        video_height, video_width = selection.shape[:2]
    fighter_a_box = _validated_fighter_box(payload.fighter_a_box, video_width, video_height, "Fighter A")
    fighter_b_box = _validated_fighter_box(payload.fighter_b_box, video_width, video_height, "Fighter B")
    shared = (
        max(0.0, min(fighter_a_box[2], fighter_b_box[2]) - max(fighter_a_box[0], fighter_b_box[0]))
        * max(0.0, min(fighter_a_box[3], fighter_b_box[3]) - max(fighter_a_box[1], fighter_b_box[1]))
    )
    smallest = min(
        (fighter_a_box[2] - fighter_a_box[0]) * (fighter_a_box[3] - fighter_a_box[1]),
        (fighter_b_box[2] - fighter_b_box[0]) * (fighter_b_box[3] - fighter_b_box[1]),
    )
    if shared / max(1.0, smallest) >= 0.28:
        raise HTTPException(400, "Draw a separate fighter in each box; the two selections overlap too much.")
    # A/B is the report focus, not a tracking shortcut. WarriorIQ always
    # analyzes both selected fighters so identity context and the scorecard do
    # not disappear when the user asks for a detailed report on one athlete.
    focus_fighter = (payload.focus_fighter or payload.analysis_target or "A").upper()
    if focus_fighter not in {"A", "B"}:
        raise HTTPException(400, "Choose Fighter A or Fighter B for the detailed report.")

    _save_fighter_portrait(job_id, "A", fighter_a_box)
    _save_fighter_portrait(job_id, "B", fighter_b_box)

    req = _analysis_request(job_id, job, fighter_a_box, fighter_b_box, focus_fighter)
    if job.get("account_id") and not job.get("usage_reserved"):
        if not reserve_analysis(int(job["account_id"]), job_id):
            plan = _request_plan(request)
            raise HTTPException(429, f"Your {plan['label']} plan includes {plan['limit_label'].lower()}. Your allowance will reset automatically.")
        update_job(job_id, {"usage_reserved": True})
    try:
        analysis_run_id = prepare_job_run(job_id, {
            "analysis_target": "BOTH", "focus_fighter": focus_fighter,
            "fighter_a_box": fighter_a_box, "fighter_b_box": fighter_b_box,
            **({"message": DEFERRED_ANALYSIS_MESSAGE} if capacity["deferred"] else {}),
        })
    except AnalysisStateNotPersisted as exc:
        if job.get("account_id") and get_job(job_id).get("usage_reserved"):
            release_analysis(int(job["account_id"]), job_id)
            update_job(job_id, {"usage_reserved": False})
        LOGGER.error("analysis_queue_not_persisted job_id=%s", job_id, exc_info=exc)
        raise HTTPException(
            503,
            "WarriorIQ could not queue this analysis. Your video and fighter selection are preserved.",
        ) from exc
    if SETTINGS.analysis_worker_mode == "inprocess":
        executor.submit(_run_job, job_id, req, analysis_run_id)
    else:
        _wake_analysis_worker(job_id)
    return _analysis_started_response(request, job_id, capacity["deferred"])


@app.post("/api/restart/{job_id}")
def restart_interrupted_analysis(request: Request, job_id: str):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    if job.get("status") in {"queued", "running"}:
        return _analysis_started_response(request, job_id)
    capacity = _require_analysis_capacity()
    if job.get("status") != "interrupted":
        raise HTTPException(409, "Only an analysis interrupted by a server restart can be resumed here.")
    fighter_a_box = job.get("fighter_a_box")
    fighter_b_box = job.get("fighter_b_box")
    focus_fighter = job.get("focus_fighter") or "A"
    if not isinstance(fighter_a_box, list) or len(fighter_a_box) != 4 or not isinstance(fighter_b_box, list) or len(fighter_b_box) != 4:
        raise HTTPException(409, "The saved session predates resumable analysis. Return to fighter selection once; your video is still available.")
    req = _analysis_request(job_id, job, fighter_a_box, fighter_b_box, focus_fighter)
    try:
        analysis_run_id = prepare_job_run(job_id, {
            "message": DEFERRED_ANALYSIS_MESSAGE if capacity["deferred"] else "Restarting the preserved analysis session",
        })
    except AnalysisStateNotPersisted as exc:
        LOGGER.error("analysis_queue_not_persisted job_id=%s", job_id, exc_info=exc)
        raise HTTPException(
            503,
            "WarriorIQ could not queue this analysis. Your video and fighter selection are preserved.",
        ) from exc
    if SETTINGS.analysis_worker_mode == "inprocess":
        executor.submit(_run_job, job_id, req, analysis_run_id)
    else:
        _wake_analysis_worker(job_id)
    return _analysis_started_response(request, job_id, capacity["deferred"])


@app.get("/progress/{job_id}", response_class=HTMLResponse)
def progress_page(request: Request, job_id: str):
    authorized = _authorized_job(request, job_id)
    if not authorized:
        raise HTTPException(404)
    job = get_job(job_id) or authorized
    return templates.TemplateResponse(request=request, name="progress.html", context={
        "request": request, "job_id": job_id, "initial_status": _public_job_status(job_id, job),
    })


def _public_job_status(job_id: str, job: dict) -> dict:
    public_fields = {
        "job_id", "status", "percent", "message", "stage", "elapsed_seconds", "eta_seconds",
        "processed_video_seconds", "video_duration_seconds", "fighter_a_confidence",
        "fighter_b_confidence", "current_round", "quality_mode", "live_event_mode",
        "live_events", "provisional_stats", "latest_observation", "focus_fighter",
    }
    payload = {key: value for key, value in job.items() if key in public_fields}
    payload.setdefault("job_id", job_id)
    payload.setdefault("video_duration_seconds", job.get("video_duration", 0.0))
    start_seconds = float(job.get("start_seconds", 0.0) or 0.0)
    full_duration = float(job.get("video_duration", payload.get("video_duration_seconds", 0.0)) or 0.0)
    scheduled_duration = (
        float(job.get("round_count", 1) or 1) * float(job.get("round_duration_seconds", full_duration) or full_duration)
        + max(0, int(job.get("round_count", 1) or 1) - 1) * float(job.get("break_duration_seconds", 0.0) or 0.0)
    )
    end_seconds = float(job.get("end_seconds") or min(full_duration, start_seconds + scheduled_duration))
    payload["analysis_start_seconds"] = start_seconds
    payload["analysis_duration_seconds"] = max(0.0, end_seconds - start_seconds)
    payload["video_url"] = f"/media/{job_id}"
    payload["result_url"] = f"/result/{job_id}" if job.get("status") == "complete" else None
    payload["restart_url"] = f"/api/restart/{job_id}" if job.get("status") == "interrupted" else None
    return payload


@app.get("/api/status/{job_id}")
def status(request: Request, response: Response, job_id: str):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    if job.get("status") == "complete":
        response.set_cookie(
            LAST_COMPLETED_ANALYSIS_COOKIE, job_id, max_age=60 * 60 * 24 * 30,
            httponly=True, samesite="lax", secure=_request_is_secure(request),
        )
        if request.cookies.get(ACTIVE_ANALYSIS_COOKIE) == job_id:
            response.delete_cookie(ACTIVE_ANALYSIS_COOKIE, httponly=True, samesite="lax")
    return _public_job_status(job_id, job)


@app.get("/api/active-analysis")
def active_analysis(request: Request):
    navigation = _analysis_navigation_state(
        _owner_key(request),
        request.cookies.get(ACTIVE_ANALYSIS_COOKIE),
        request.cookies.get(LAST_COMPLETED_ANALYSIS_COOKIE),
    )
    job = navigation["display"]
    if not job:
        return {
            "active": False,
            "processing": False,
            "active_analysis_id": None,
            "last_completed_analysis_id": None,
        }
    return {
        "active": True,
        "processing": navigation["active"] is not None,
        "active_analysis_id": navigation["active"]["job_id"] if navigation["active"] else None,
        "last_completed_analysis_id": (
            navigation["last_completed"]["job_id"] if navigation["last_completed"] else None
        ),
        "job_id": job["job_id"],
        "status": job.get("status"),
        "percent": float(job.get("percent", 0.0)),
        "url": _analysis_navigation_url(job),
    }


@app.get("/result/{job_id}", response_class=HTMLResponse)
def result_page(request: Request, job_id: str):
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    path = OUTPUTS / job_id / "report.json"
    if not path.exists():
        raise HTTPException(404)
    report = json.loads(path.read_text(encoding="utf-8"))
    if "key_moments" not in report:
        report["key_moments"] = [e for e in report.get("events", []) if e.get("outcome") in {"clean", "likely_landed"} and float(e.get("confidence", 0)) >= .72 and float(e.get("contact_confidence", 0)) >= .62][:18]
    coverage_ok = min(float(report.get("tracking", {}).get("fighter_A_coverage", 0)), float(report.get("tracking", {}).get("fighter_B_coverage", 0))) >= SETTINGS.min_tracking_coverage_for_score
    report.setdefault("scorecard", {})["available"] = bool(report.get("scorecard", {}).get("available", coverage_ok) and coverage_ok)
    if report.get("video", {}).get("analysis_target", "BOTH") != "BOTH":
        report["scorecard"]["available"] = False
        report["scorecard"]["disclaimer"] = "To receive an estimated scorecard, choose Analyze both fighters. A one-fighter analysis does not count the opponent's points."
    # Customer reports are fully automatic. Human annotations remain isolated
    # in the model-validation lab and never become required report work.
    _apply_report_annotations(report, [])
    refresh_identity_integrity(report)
    report_access = _request_plan(request)
    # Opening the exact completed report acknowledges its one-time notification.
    # Without this reset, the green "Results ready" chip was written back on
    # every report view and could appear indefinitely before a new analysis.
    displayed = request.state.active_analysis
    if displayed and displayed.get("job_id") == job_id and displayed.get("status") == "complete":
        request.state.analysis_navigation["last_completed"] = None
        request.state.analysis_navigation["display"] = request.state.analysis_navigation.get("active")
        request.state.active_analysis = request.state.analysis_navigation.get("active")
    response = templates.TemplateResponse(request=request, name="result.html", context={
        "request": request, "job_id": job_id, "report": report,
        "report_access": report_access,
        "analysis_quality": _analysis_quality_summary(report),
        "can_share": bool(_account(request) and report_access.get("can_share")),
    })
    response.delete_cookie(LAST_COMPLETED_ANALYSIS_COOKIE, httponly=True, samesite="lax")
    if request.cookies.get(ACTIVE_ANALYSIS_COOKIE) == job_id:
        response.delete_cookie(ACTIVE_ANALYSIS_COOKIE, httponly=True, samesite="lax")
    return response


@app.post("/api/annotations/{job_id}")
def annotate_event(request: Request, job_id: str, payload: AnnotationPayload):
    if not _account(request) or not _authorized_job(request, job_id) or not _request_plan(request).get("can_correct"):
        raise HTTPException(403, "Evidence corrections are available with a complete-report plan.")
    report_path = OUTPUTS / job_id / "report.json"
    if not report_path.exists():
        raise HTTPException(404, "Fight report not found")
    fighter = payload.fighter.upper()
    target = payload.target.lower()
    outcome = payload.outcome.lower()
    technique = payload.technique.lower().strip().replace(" ", "_")
    contact_time = payload.event_time if payload.contact_time is None else payload.contact_time
    if (
        not math.isfinite(payload.event_time) or payload.event_time < 0
        or not math.isfinite(contact_time) or contact_time < 0
        or fighter not in {"A", "B"}
        or target not in {"head", "body", "leg", "none"}
        or outcome not in {"clean", "blocked", "checked", "missed", "uncertain"}
        or technique not in ANNOTATION_TECHNIQUES
    ):
        raise HTTPException(400, "Invalid correction")
    if technique == "none":
        family, limb, target, outcome = "none", "none", "none", "uncertain"
    else:
        family = "knee" if "knee" in technique else "kick" if "kick" in technique else "punch"
        side = "left" if technique.startswith("left_") else "right" if technique.startswith("right_") else ""
        limb = f"{side + '_' if side else ''}{'knee' if family == 'knee' else 'leg' if family == 'kick' else 'hand'}"
    corrected = {"fighter": fighter, "technique": technique, "target": None if target == "none" else target,
                 "outcome": outcome, "family": family, "limb": limb, "contact_time": float(contact_time)}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    refresh_identity_integrity(report)
    if not report.get("integrity", {}).get("identity_evidence_trusted", True):
        raise HTTPException(409, "Choose the fighters again before correcting action evidence.")
    predicted = _prediction_at(report, payload.event_time)
    if predicted is None and not payload.manual:
        raise HTTPException(404, "No analyzed action exists at that time")
    segment_end = float(report.get("setup", {}).get("end_seconds") or (
        float(report.get("setup", {}).get("start_seconds", 0))
        + float(report.get("performance", {}).get("segment_duration_seconds", 0))
    ))
    if contact_time > segment_end + 0.05:
        raise HTTPException(400, "The exact contact time is outside the analyzed segment")
    if predicted is None:
        if payload.event_time > segment_end + 0.05:
            raise HTTPException(400, "The label time is outside the analyzed segment")
        predicted = {"fighter": fighter, "technique": "none", "target": None,
                     "outcome": "uncertain", "family": "none", "limb": "none"}
    annotation_id = save_annotation(job_id, payload.event_time, report.get("setup", {}).get("ruleset", "K1"), predicted, corrected)
    profile = get_profile(_profile_id(request))
    training_consent = bool(profile and profile.get("allow_model_training"))
    sequence_path = export_sequence(job_id, annotation_id, corrected, contact_time) if training_consent else None
    set_annotation_sequence(annotation_id, sequence_path)
    return {
        "ok": True, "annotation_id": annotation_id,
        "sequence_exported": bool(sequence_path), "training_consent": training_consent,
        "corrected": corrected,
    }


def _review_candidates(report: dict, mode: str = "dataset") -> list[dict]:
    """Collapse burst duplicates while retaining every distinct action hypothesis."""
    if mode == "scorecard":
        ruleset = report.get("setup", {}).get("ruleset", "K1")
        scoring_events: list[StrikeEvent] = []
        for item in report.get("events", []):
            try:
                event_time = float(item.get("peak_time"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(event_time) or event_time < 0 or item.get("technique") == "none":
                continue
            event = _strike_from_dict(item)
            if is_verified_scoring_event(event, ruleset):
                scoring_events.append(event)
        deduplicated, _ = deduplicate_scoring_events(scoring_events)
        return [event.to_dict() for event in deduplicated]

    selected: list[dict] = []
    for event in sorted(report.get("events", []), key=lambda item: float(item.get("peak_time", 0))):
        try:
            event_time = float(event.get("peak_time"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(event_time) or event_time < 0 or event.get("technique") == "none":
            continue
        item = dict(event)
        if selected and item.get("fighter") == selected[-1].get("fighter") and event_time - float(selected[-1].get("peak_time", 0)) < 0.30:
            current_score = float(item.get("confidence", 0)) + float(item.get("contact_confidence", 0))
            kept_score = float(selected[-1].get("confidence", 0)) + float(selected[-1].get("contact_confidence", 0))
            if current_score > kept_score:
                selected[-1] = item
            continue
        selected.append(item)
    return selected


@app.get("/review/{job_id}", response_class=HTMLResponse)
def review_evidence_page(request: Request, job_id: str, page: int = 1, mode: str = "scorecard"):
    if not _account(request) or not _authorized_job(request, job_id) or not _request_plan(request).get("can_correct"):
        raise HTTPException(403, "Evidence review is available with a complete-report plan.")
    report_path = OUTPUTS / job_id / "report.json"
    if not report_path.exists():
        raise HTTPException(404)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    refresh_identity_integrity(report)
    if not report.get("integrity", {}).get("identity_evidence_trusted", True):
        return RedirectResponse(f"/select/{job_id}", status_code=303)
    mode = "dataset" if mode == "dataset" else "scorecard"
    all_candidates = _review_candidates(report, mode)
    annotations = get_annotations(job_id)
    annotation_map = {f"{float(item['event_time']):.3f}": item for item in annotations}
    per_page = 40
    page_count = max(1, math.ceil(len(all_candidates) / per_page))
    current_page = max(1, min(int(page), page_count))
    start = (current_page - 1) * per_page
    reviewed_total = len({key for key in annotation_map if any(abs(float(key) - float(item.get('peak_time', -999))) <= .02 for item in all_candidates)})
    return templates.TemplateResponse(request=request, name="review.html", context={
        "request": request,
        "job_id": job_id,
        "report": report,
        "candidates": all_candidates[start:start + per_page],
        "candidate_total": len(all_candidates),
        "reviewed_total": reviewed_total,
        "remaining_total": max(0, len(all_candidates) - reviewed_total),
        "annotations": annotation_map,
        "annotation_techniques": ANNOTATION_TECHNIQUES,
        "page": current_page,
        "page_count": page_count,
        "mode": mode,
        "review": get_fight_review(job_id),
    })


@app.post("/review/{job_id}/complete")
def complete_evidence_review(
    request: Request,
    job_id: str,
    complete: bool = Form(False),
    mode: str = Form("dataset"),
):
    account = _account(request)
    fight = get_fight(job_id)
    if not account or not fight or int(fight["profile_id"]) != int(account["profile_id"]) or not _request_plan(request).get("can_correct"):
        raise HTTPException(403)
    if complete:
        mode = "dataset" if mode == "dataset" else "scorecard"
        report_path = OUTPUTS / job_id / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        refresh_identity_integrity(report)
        if not report.get("integrity", {}).get("identity_evidence_trusted", True):
            return RedirectResponse(f"/select/{job_id}", status_code=303)
        candidates = _review_candidates(report, mode)
        reviewed_times = [float(item["event_time"]) for item in get_annotations(job_id)]
        remaining = [item for item in candidates if not any(abs(float(item["peak_time"]) - value) <= .02 for value in reviewed_times)]
        if remaining:
            raise HTTPException(409, f"Review the remaining {len(remaining)} candidates before completing the fight.")
    current_status = get_fight_review(job_id).get("status", "in_progress")
    if not complete:
        next_status = "in_progress"
    elif mode == "dataset" or current_status == "complete":
        next_status = "complete"
    else:
        next_status = "scorecard_complete"
    set_fight_review_status(job_id, int(account["profile_id"]), next_status)
    return RedirectResponse(f"/result/{job_id}", status_code=303)


@app.get("/validation", response_class=HTMLResponse)
def validation_page(request: Request):
    profile_id = _profile_id(request)
    owned_jobs = {fight["job_id"] for fight in list_fights(profile_id)} if profile_id is not None else set()
    annotations = [item for item in list_annotations() if item["job_id"] in owned_jobs]
    dataset = audit_dataset_split(DATASET / "sequences", DATASET / "untouched_test")
    summary = accuracy_summary(annotations)
    end_to_end = assess_end_to_end_validation(end_to_end_metadata(summary))
    return templates.TemplateResponse(request=request, name="validation.html", context={
        "request": request,
        "summary": summary,
        "annotations": annotations,
        "dataset": dataset,
        "end_to_end": end_to_end,
    })


def _build_replay_chapters(
    report: dict,
    focus: str,
    fighter_filter: str | None = None,
    family_filter: str | None = None,
    outcome_filter: str | None = None,
) -> tuple[list[dict], str]:
    """Build useful replay navigation without turning candidates into facts."""
    focus = focus if focus in {"A", "B", "BOTH"} else "BOTH"
    fighter_filter = fighter_filter if fighter_filter in {"A", "B"} else None
    family_filter = family_filter if family_filter in {"punch", "kick"} else None
    outcome_filter = outcome_filter if outcome_filter in {"landed", "missed", "blocked", "evaded"} else None
    filtered = any((fighter_filter, family_filter, outcome_filter))
    source_events = report.get("event_feed", []) if filtered else report.get("key_moments", [])
    verified = []
    for event in source_events:
        if filtered and event.get("verification") != "verified":
            continue
        if fighter_filter and event.get("fighter") != fighter_filter:
            continue
        if family_filter and event.get("family") != family_filter:
            continue
        if outcome_filter and event.get("outcome") != outcome_filter:
            continue
        if not fighter_filter and focus != "BOTH" and event.get("fighter") != focus:
            continue
        try:
            event_time = float(event.get("time_seconds", event.get("peak_time")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(event_time) or event_time < 0:
            continue
        technique = str(event.get("technique") or "verified action").replace("_", " ").title()
        outcome = str(event.get("outcome") or "verified").replace("_", " ").title()
        verified.append({
            "time": round(event_time, 3),
            "lead_seconds": 1.0,
            "label": f"{event_time:.2f}s · Fighter {event.get('fighter', focus)} · {technique}",
            "detail": outcome,
            "kind": "verified_action",
        })
    if verified:
        return verified[:8], "verified_actions"

    setup = report.get("setup", {})
    performance = report.get("performance", {})
    start = float(setup.get("start_seconds", 0.0) or 0.0)
    end_value = setup.get("end_seconds")
    end = float(end_value) if isinstance(end_value, (int, float)) else start + float(performance.get("segment_duration_seconds", 0.0) or 0.0)
    if not math.isfinite(end) or end <= start:
        end = start + 1.0

    chapters: list[dict] = []
    labels = ("Opening", "Early section", "Middle section", "Closing section")
    fractions = (0.0, .25, .50, .75)
    for label, fraction in zip(labels, fractions):
        moment = start + (end - start) * fraction
        if chapters and moment - chapters[-1]["time"] < 2.0:
            continue
        chapters.append({
            "time": round(moment, 3),
            "lead_seconds": 0.0,
            "label": f"{label} · {moment:.1f}s",
            "detail": f"Fighter {focus} skeleton chapter" if focus != "BOTH" else "Skeleton replay chapter",
            "kind": "movement_chapter",
        })
    return chapters, "movement_chapters"


@app.get("/replay/{job_id}", response_class=HTMLResponse)
def replay_page(
    request: Request,
    job_id: str,
    fighter: str | None = None,
    family: str | None = None,
    outcome: str | None = None,
):
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    path = OUTPUTS / job_id / "report.json"
    if not path.exists():
        raise HTTPException(404)
    report = json.loads(path.read_text(encoding="utf-8"))
    if "key_moments" not in report:
        report["key_moments"] = [e for e in report.get("events", []) if e.get("outcome") in {"clean", "likely_landed"} and float(e.get("confidence", 0)) >= .72 and float(e.get("contact_confidence", 0)) >= .62][:18]
    _apply_report_annotations(report, [])
    refresh_identity_integrity(report)
    identity_safe = bool(report.get("integrity", {}).get("identity_evidence_trusted", True))
    focus = report.get("video", {}).get("focus_fighter") or report.get("video", {}).get("analysis_target", "BOTH")
    replay_chapters, replay_mode = _build_replay_chapters(
        report, focus,
        fighter.upper() if fighter else None,
        family.lower() if family else None,
        outcome.lower() if outcome else None,
    )
    return templates.TemplateResponse(
        request=request,
        name="replay.html",
        context={
            "request": request, "job_id": job_id, "report": report,
            "identity_safe": identity_safe,
            "replay_chapters": replay_chapters,
            "replay_mode": replay_mode,
            "evidence_filter": " · ".join(
                value.replace("_", " ").title()
                for value in (fighter, family, outcome) if value
            ),
        },
    )


@app.get("/api/tracking/{job_id}")
def tracking_data(request: Request, job_id: str):
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    path = OUTPUTS / job_id / "tracking.jsonl"
    if not path.exists():
        raise HTTPException(404)
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return JSONResponse({"frames": frames})


@app.get("/fighter-portrait/{job_id}/{fighter}")
def fighter_portrait(request: Request, job_id: str, fighter: str):
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    fighter = fighter.upper()
    if fighter not in {"A", "B"}:
        raise HTTPException(404)
    path = OUTPUTS / job_id / f"fighter_{fighter}.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/media/{job_id}")
def media(request: Request, job_id: str):
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    job = get_job(job_id)
    if job:
        path = Path(job["video_path"])
    else:
        fight = get_fight(job_id)
        path = Path(fight["video_path"]) if fight else Path("missing")
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request):
    profile_id = _profile_id(request)
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "request": request,
            "profile": get_profile(profile_id) if profile_id is not None else None,
            "fights": list_fights(profile_id) if profile_id is not None else [],
        },
    )


@app.get("/settings")
def settings_root(request: Request):
    return RedirectResponse("/settings/privacy", status_code=303)


@app.get("/settings/{section}", response_class=HTMLResponse)
def settings_page(request: Request, section: str, notice: str = ""):
    account = _account(request)
    if not account:
        return RedirectResponse(f"/login?next=/settings/{section}", status_code=303)
    if section not in {"privacy", "billing"}:
        raise HTTPException(404)
    account = get_account(int(account["id"])) or account
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "request": request,
            "section": section,
            "notice": notice,
            "account": account,
            "profile": get_profile(int(account["profile_id"])) or {},
            "fights": list_fights(int(account["profile_id"])),
            "video_retention_days": SETTINGS.saved_video_retention_days,
            "plan": plan_for_key(account.get("plan_override") or account.get("plan")),
            "subscription_actions": list_subscription_actions(int(account["id"])),
        },
    )


@app.post("/settings/marketing")
def save_marketing_preference(request: Request, enabled: bool = Form(False)):
    account = _account(request)
    if not account:
        raise HTTPException(403)
    update_marketing_consent(int(account["id"]), bool(enabled))
    record_legal_acceptance(
        "marketing_consent", SETTINGS.policy_version, profile_id=int(account["profile_id"]),
        metadata={"enabled": bool(enabled), "source": "privacy_settings"},
        current_status="accepted" if enabled else "withdrawn",
    )
    return RedirectResponse("/settings/privacy?notice=Marketing+preference+saved", status_code=303)


@app.post("/settings/sessions/revoke")
def revoke_other_sessions(request: Request):
    account = _account(request)
    if not account:
        raise HTTPException(403)
    current = request.cookies.get(SESSION_COOKIE)
    revoke_account_sessions(int(account["id"]), token_digest(current) if current else None)
    record_security_event("other_sessions_revoked", account_id=int(account["id"]))
    return RedirectResponse("/settings/privacy?notice=Other+sessions+signed+out", status_code=303)


@app.post("/settings/videos/{job_id}/delete")
def delete_original_video(request: Request, job_id: str):
    account = _account(request)
    fight = get_fight(job_id)
    if not account or not fight or int(fight["profile_id"]) != int(account["profile_id"]):
        raise HTTPException(404)
    video = Path(fight.get("video_path") or "missing").resolve()
    if video.parent == UPLOADS.resolve():
        video.unlink(missing_ok=True)
    mark_fight_video_deleted(job_id, int(account["profile_id"]))
    record_security_event(
        "fight_video_deleted", account_id=int(account["id"]), resource_type="fight", resource_id=job_id,
    )
    return RedirectResponse("/settings/privacy?notice=Original+video+deleted", status_code=303)


@app.post("/settings/billing/cancel")
def cancel_subscription(request: Request):
    session_account = _account(request)
    if not session_account:
        raise HTTPException(403)
    account = get_account(int(session_account["id"])) or session_account
    subscription_id = account.get("stripe_subscription_id")
    if not subscription_id:
        raise HTTPException(400, "No connected paid subscription can be cancelled.")
    try:
        provider = cancel_subscription_at_period_end(str(subscription_id))
    except Exception as exc:
        raise HTTPException(503, f"Cancellation was not confirmed by the payment provider: {exc}")
    if not provider.get("cancel_at_period_end"):
        raise HTTPException(503, "The payment provider did not confirm cancellation.")
    action = record_subscription_action(
        int(account["id"]), "cancel", "scheduled",
        effective_at=account.get("subscription_period_end"), provider_reference=str(subscription_id),
        metadata={"provider_status": provider.get("status")},
    )
    _queue_transactional_notice(
        int(account["id"]), "subscription_cancellation_confirmation", account["email"],
        "Your WarriorIQ subscription cancellation",
        f"Cancellation was scheduled on {action['requested_at']}. Access ends at {account.get('subscription_period_end') or 'the confirmed billing-period end'}. No further renewal should be charged after that date.",
        {"requested_at": action["requested_at"], "access_ends": account.get("subscription_period_end")},
    )
    return RedirectResponse("/settings/billing?notice=Cancellation+scheduled", status_code=303)


@app.post("/settings/billing/withdraw")
def request_contract_withdrawal(request: Request, confirm: bool = Form(False)):
    account = _account(request)
    if not account or not confirm:
        raise HTTPException(400, "Confirm the withdrawal request.")
    full_account = get_account(int(account["id"])) or account
    if not full_account.get("stripe_subscription_id"):
        raise HTTPException(400, "No connected purchase is available for withdrawal review.")
    action = record_subscription_action(
        int(account["id"]), "eu_withdrawal", "pending_review",
        provider_reference=full_account.get("stripe_subscription_id"),
        metadata={"eligibility_not_determined": True, "policy_version": SETTINGS.policy_version},
    )
    _queue_transactional_notice(
        int(account["id"]), "withdrawal_request_confirmation", account["email"],
        "WarriorIQ withdrawal request received",
        f"Your withdrawal request was received on {action['requested_at']} and is pending eligibility and payment review. This is separate from normal subscription cancellation.",
        {"requested_at": action["requested_at"], "status": "pending_review"},
    )
    return RedirectResponse("/settings/billing?notice=Withdrawal+request+recorded+for+review", status_code=303)


@app.post("/profile")
async def save_profile(
    request: Request,
    display_name: str = Form(...),
    notes: str = Form(""),
    default_fighter: str = Form("A"),
    allow_model_training: bool = Form(False),
    photo: UploadFile | None = File(None),
    profile_video: UploadFile | None = File(None),
):
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/profile", status_code=303)
    current_profile = get_profile(profile_id) or {}
    default_fighter = default_fighter.upper()
    if default_fighter not in {"A", "B"}:
        raise HTTPException(400, "Default fighter must be A or B.")
    photo_path = None
    profile_video_path = None
    if photo and photo.filename:
        folder = ROOT / "app" / "static" / "profile"
        folder.mkdir(parents=True, exist_ok=True)
        suffix = Path(photo.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(400, "Profile photo must be JPG, PNG or WEBP.")
        file_path = folder / f"profile_{profile_id}{suffix}"
        await run_in_threadpool(_save_upload_limited, photo, file_path, MAX_PROFILE_PHOTO_BYTES)
        if await run_in_threadpool(cv2.imread, str(file_path)) is None:
            file_path.unlink(missing_ok=True)
            raise HTTPException(400, "The profile photo could not be decoded as a safe image.")
        photo_path = f"/static/profile/{file_path.name}"
        if current_profile.get("photo_path") != photo_path:
            _remove_profile_file(current_profile.get("photo_path"))
    if profile_video and profile_video.filename:
        folder = ROOT / "app" / "static" / "profile"
        folder.mkdir(parents=True, exist_ok=True)
        suffix = Path(profile_video.filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".m4v", ".webm"}:
            raise HTTPException(400, "Profile video must be MP4, MOV, M4V or WEBM.")
        file_path = folder / f"profile_video_{profile_id}{suffix}"
        await run_in_threadpool(_save_upload_limited, profile_video, file_path, MAX_PROFILE_VIDEO_BYTES)
        try:
            await run_in_threadpool(get_video_info, file_path)
        except Exception:
            file_path.unlink(missing_ok=True)
            raise HTTPException(400, "The profile video could not be decoded as a supported video.")
        profile_video_path = f"/static/profile/{file_path.name}"
        if current_profile.get("video_path") != profile_video_path:
            _remove_profile_file(current_profile.get("video_path"))
    update_profile(
        profile_id, display_name.strip()[:80] or SETTINGS.default_profile_name,
        photo_path, profile_video_path, notes.strip()[:2000], default_fighter, allow_model_training,
    )
    record_legal_acceptance(
        "ai_training_consent", SETTINGS.policy_version,
        profile_id=profile_id,
        metadata={"enabled": bool(allow_model_training), "source": "profile"},
        current_status="accepted" if allow_model_training else "withdrawn",
    )
    return RedirectResponse("/profile", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    profile_id = _profile_id(request)
    if profile_id is None:
        return templates.TemplateResponse(
            request=request, name="dashboard.html",
            context={"request": request, "profile": None, "progress": None, "assignments": []},
        )
    profile = get_profile(profile_id) or {}
    progress = build_progress(_reports_for_profile(profile_id), profile.get("default_fighter", "A"))
    return templates.TemplateResponse(
        request=request, name="dashboard.html",
        context={
            "request": request, "profile": profile, "progress": progress,
            "assignments": list_assignments(profile_id),
        },
    )


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    profile_id = _profile_id(request)
    fights = list_fights(profile_id) if profile_id is not None else []
    return templates.TemplateResponse(
        request=request, name="history.html",
        context={"request": request, "fights": fights, "signed_in": profile_id is not None},
    )


def _remove_fight_files(fight: dict) -> None:
    job_dir = (OUTPUTS / fight["job_id"]).resolve()
    if job_dir.parent == OUTPUTS.resolve() and job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    video = Path(fight["video_path"]).resolve()
    if video.parent == UPLOADS.resolve():
        video.unlink(missing_ok=True)


def _remove_profile_file(value: str | None) -> None:
    if not value or not value.startswith("/static/profile/"):
        return
    path = (ROOT / "app" / value.lstrip("/")).resolve()
    if path.parent == (ROOT / "app" / "static" / "profile").resolve():
        path.unlink(missing_ok=True)


@app.post("/delete/{job_id}")
def delete_fight_route(request: Request, job_id: str):
    profile_id = _profile_id(request)
    existing = get_fight(job_id)
    if profile_id is None or not existing or int(existing["profile_id"]) != profile_id:
        raise HTTPException(404)
    fight = delete_fight(job_id)
    if fight:
        _remove_fight_files(fight)
        account = _account(request)
        record_security_event(
            "fight_analysis_deleted", account_id=int(account["id"]) if account else None,
            resource_type="fight", resource_id=job_id,
        )
    return RedirectResponse("/history", status_code=303)


@app.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request, a: str = "", b: str = ""):
    profile_id = _profile_id(request)
    fights = list_fights(profile_id) if profile_id is not None else []
    allowed = {fight["job_id"] for fight in fights}
    reports = []
    for job_id in (a, b):
        path = OUTPUTS / job_id / "report.json"
        report = json.loads(path.read_text(encoding="utf-8")) if job_id in allowed and path.exists() else None
        if report is not None:
            _apply_report_annotations(report, [])
            refresh_identity_integrity(report)
        reports.append(report)
    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={"request": request, "fights": fights, "a": a, "b": b, "reports": reports, "signed_in": profile_id is not None},
    )


@app.get("/coach", response_class=HTMLResponse)
def coach_page(request: Request):
    profile_id = _profile_id(request)
    profile = get_profile(profile_id) if profile_id is not None else None
    fights = list_fights(profile_id) if profile_id is not None else []
    latest = None
    focus = (profile or {}).get("default_fighter", "A")
    suggested_assignments: list[dict] = []
    if fights:
        path = Path(fights[0]["report_path"])
        if path.exists():
            latest = json.loads(path.read_text(encoding="utf-8"))
            _apply_report_annotations(latest, [])
            refresh_identity_integrity(latest)
            focus = latest.get("video", {}).get("focus_fighter") or latest.get("video", {}).get("analysis_target", focus)
            if focus not in {"A", "B"}:
                focus = (profile or {}).get("default_fighter", "A")
            suggested_assignments = list(latest.get("training_plan", {}).get(focus, []))[:3]
    return templates.TemplateResponse(
        request=request, name="coach.html",
        context={
            "request": request, "fights": fights, "latest": latest,
            "assignments": list_assignments(profile_id) if profile_id is not None else [],
            "signed_in": profile_id is not None,
            "focus": focus,
            "suggested_assignments": suggested_assignments,
        },
    )


@app.post("/coach/assignments")
def create_coach_assignment(
    request: Request,
    title: str = Form(...),
    detail: str = Form(""),
    next_path: str = Form("/coach#assignments"),
):
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/coach", status_code=303)
    title, detail = title.strip()[:100], detail.strip()[:600]
    if not title:
        raise HTTPException(400, "Assignment title is required.")
    add_assignment(profile_id, title, detail)
    return RedirectResponse(_safe_next(next_path, "/coach#assignments"), status_code=303)


@app.post("/coach/assignments/{assignment_id}/toggle")
def update_coach_assignment(
    request: Request,
    assignment_id: int,
    next_path: str = Form("/coach#assignments"),
):
    profile_id = _profile_id(request)
    if profile_id is None or not toggle_assignment(assignment_id, profile_id):
        raise HTTPException(404)
    return RedirectResponse(_safe_next(next_path, "/coach#assignments"), status_code=303)


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="privacy.html",
        context={
            "request": request, "guest_retention_hours": GUEST_RETENTION_HOURS,
            "video_retention_days": SETTINGS.saved_video_retention_days,
            "minimum_account_age": SETTINGS.minimum_account_age,
            "launch": launch_readiness(), "policy_version": SETTINGS.policy_version,
        },
    )


@app.post("/cookie-preferences")
def save_cookie_preferences(
    request: Request,
    choice: str = Form(...),
    next_path: str = Form("/"),
    analytics: bool = Form(False),
    marketing: bool = Form(False),
):
    if choice == "all":
        analytics = marketing = True
        value = "all"
    elif choice == "essential":
        analytics = marketing = False
        value = "essential"
    elif choice == "custom":
        value = "custom-analytics" if analytics and not marketing else "custom-marketing" if marketing and not analytics else "all" if analytics else "essential"
    else:
        raise HTTPException(400, "Choose a valid cookie preference.")
    account = _account(request)
    owner = {"profile_id": int(account["profile_id"])} if account else {"guest_id": request.state.guest_id}
    if account:
        update_cookie_preferences(int(account["id"]), analytics=analytics, marketing=marketing)
    record_legal_acceptance(
        "cookie_preferences", SETTINGS.policy_version, metadata={"analytics": analytics, "marketing": marketing},
        current_status="accepted" if analytics or marketing else "declined", **owner,
    )
    response = RedirectResponse(_safe_next(next_path, "/"), status_code=303)
    response.set_cookie(
        COOKIE_PREFERENCES_COOKIE, value, max_age=60 * 60 * 24 * 365,
        httponly=True, samesite="lax", secure=_request_is_secure(request),
    )
    return response


@app.get("/legal", response_class=HTMLResponse)
def legal_center(request: Request):
    return templates.TemplateResponse(
        request=request, name="legal.html",
        context={
            "request": request, "documents": LEGAL_DOCUMENTS,
            "launch": launch_readiness(), "policy_version": SETTINGS.policy_version,
        },
    )


@app.get("/terms", response_class=HTMLResponse)
@app.get("/cookies", response_class=HTMLResponse)
@app.get("/acceptable-use", response_class=HTMLResponse)
@app.get("/refunds", response_class=HTMLResponse)
@app.get("/video-upload-policy", response_class=HTMLResponse)
@app.get("/sports-medical-disclaimer", response_class=HTMLResponse)
@app.get("/eula", response_class=HTMLResponse)
@app.get("/dmca", response_class=HTMLResponse)
@app.get("/accessibility", response_class=HTMLResponse)
@app.get("/ai-transparency", response_class=HTMLResponse)
@app.get("/security", response_class=HTMLResponse)
@app.get("/subprocessors", response_class=HTMLResponse)
@app.get("/contact", response_class=HTMLResponse)
def legal_document(request: Request):
    slug = request.url.path.strip("/")
    document = LEGAL_DOCUMENTS.get(slug)
    if document is None:
        raise HTTPException(404)
    return templates.TemplateResponse(
        request=request, name="legal_document.html",
        context={
            "request": request, "document": document, "slug": slug,
            "launch": launch_readiness(), "policy_version": SETTINGS.policy_version,
        },
    )


@app.get("/copyright-report", response_class=HTMLResponse)
def copyright_report_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="copyright_report.html",
        context={"request": request, "submitted": False, "reference": None},
    )


@app.post("/copyright-report", response_class=HTMLResponse)
def submit_copyright_report(
    request: Request,
    email: str = Form(...),
    details: str = Form(...),
    resource_id: str = Form(""),
    good_faith: bool = Form(False),
):
    _enforce_rate_limit(request, "copyright-report", 6, 3600)
    if not valid_email(email) or len(details.strip()) < 40 or not good_faith:
        raise HTTPException(400, "Provide a valid email, a detailed good-faith report and the required confirmation.")
    report_id = create_moderation_report("copyright", email.strip().lower(), details.strip(), resource_id.strip())
    record_security_event("copyright_report_received", resource_type="moderation_report", resource_id=str(report_id))
    return templates.TemplateResponse(
        request=request, name="copyright_report.html",
        context={"request": request, "submitted": True, "reference": report_id},
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, q: str = ""):
    if not _is_admin(request):
        raise HTTPException(404)
    account = _account(request)
    record_security_event("admin_area_viewed", account_id=int(account["id"]), resource_type="admin")
    return templates.TemplateResponse(
        request=request, name="admin.html",
        context={
            "request": request, "query": q[:200], "users": list_accounts(q),
            "reports": list_moderation_reports(), "security_events": list_security_events(),
        },
    )


@app.post("/admin/accounts/{account_id}/status")
def admin_account_status(request: Request, account_id: int, status: str = Form(...)):
    if not _is_admin(request):
        raise HTTPException(404)
    actor = _account(request)
    if int(actor["id"]) == int(account_id) and status != "active":
        raise HTTPException(400, "An administrator cannot suspend their current account.")
    if not set_account_status(account_id, status):
        raise HTTPException(404)
    record_security_event(
        "admin_account_status_changed", account_id=int(actor["id"]), severity="warning",
        resource_type="account", resource_id=str(account_id), metadata={"status": status},
    )
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/reports/{report_id}/resolve")
def admin_resolve_report(request: Request, report_id: int):
    if not _is_admin(request):
        raise HTTPException(404)
    if not resolve_moderation_report(report_id):
        raise HTTPException(404)
    actor = _account(request)
    record_security_event(
        "admin_moderation_report_resolved", account_id=int(actor["id"]),
        resource_type="moderation_report", resource_id=str(report_id),
    )
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/videos/delete")
def admin_delete_prohibited_video(request: Request, job_id: str = Form(...)):
    if not _is_admin(request):
        raise HTTPException(404)
    fight = delete_fight(job_id.strip())
    if not fight:
        raise HTTPException(404, "No saved analysis matches that ID.")
    _remove_fight_files(fight)
    actor = _account(request)
    record_security_event(
        "admin_prohibited_content_deleted", account_id=int(actor["id"]), severity="warning",
        resource_type="fight", resource_id=job_id.strip(),
    )
    return RedirectResponse("/admin", status_code=303)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve the icon from the site root.

    Search engines request /favicon.ico directly rather than reading the page's
    <link rel="icon">, so a 404 here is why WarriorIQ showed a blank icon in
    search results even though the logo was declared in the template.
    """
    return FileResponse(
        ROOT / "app" / "static" / "favicon.ico",
        media_type="image/x-icon",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    if not SETTINGS.public_base_url:
        return PlainTextResponse("User-agent: *\nDisallow: /\n")
    disallowed = "\n".join(f"Disallow: {prefix}" for prefix in PRIVATE_ROUTE_PREFIXES)
    return PlainTextResponse(
        f"User-agent: *\nAllow: /\n{disallowed}\nSitemap: {SETTINGS.public_base_url}/sitemap.xml\n"
    )


def _deployed_commit() -> str:
    """Read the commit stamped into the app root at deploy time.

    Environment variables and code are deployed by separate cPanel steps, so a
    restart can pick up new settings while still running old code. Reporting
    the deployed commit makes that mismatch visible instead of leaving it to be
    inferred from which routes happen to 404.
    """
    try:
        return (ROOT / "DEPLOYED_COMMIT").read_text(encoding="utf-8").strip()[:40] or "unknown"
    except OSError:
        return "unknown"


@app.get("/health", include_in_schema=False)
def health_check():
    """Minimal deployment probe with no account, model or filesystem details."""
    return {"status": "ok", "service": "WarriorIQ", "commit": _deployed_commit()}


@app.get("/ready", include_in_schema=False)
def readiness_check():
    """Operational readiness: database, private storage and analysis worker."""
    from core.readiness import operational_readiness

    readiness = operational_readiness(worker_status())
    return JSONResponse(readiness, status_code=200 if readiness["ready"] else 503)


@app.get("/sitemap.xml")
def sitemap_xml():
    base = SETTINGS.public_base_url
    routes = (
        "/", "/pricing", "/privacy", "/legal", "/terms", "/cookies", "/acceptable-use",
        "/refunds", "/video-upload-policy", "/sports-medical-disclaimer", "/eula", "/dmca",
        "/accessibility", "/ai-transparency", "/security", "/subprocessors", "/contact",
        "/kickboxing-fight-analysis", "/k1-fight-analysis",
        "/fight-video-analysis-for-coaches", "/how-to-record-a-fight-for-analysis",
    )
    urls = "" if not base else "".join(
        f"<url><loc>{html.escape(base + path)}</loc></url>" for path in routes
    )
    return Response(
        f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>',
        media_type="application/xml",
    )


@app.get("/kickboxing-fight-analysis", response_class=HTMLResponse)
@app.get("/k1-fight-analysis", response_class=HTMLResponse)
@app.get("/fight-video-analysis-for-coaches", response_class=HTMLResponse)
@app.get("/how-to-record-a-fight-for-analysis", response_class=HTMLResponse)
def search_guide_page(request: Request):
    slug = request.url.path.strip("/")
    page = SEARCH_GUIDES.get(slug)
    if page is None:
        raise HTTPException(404)
    schema = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Article",
                "headline": page["heading"],
                "description": page["description"],
                "mainEntityOfPage": f"{SETTINGS.public_base_url}/{slug}",
                "publisher": {"@type": "Organization", "name": "WarriorIQ"},
            },
            {
                "@type": "FAQPage",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": item["question"],
                        "acceptedAnswer": {"@type": "Answer", "text": item["answer"]},
                    }
                    for item in page["faqs"]
                ],
            },
        ],
    }
    return templates.TemplateResponse(
        request=request,
        name="search_guide.html",
        context={"request": request, "page": page, "schema": schema},
    )


@app.get("/pricing", response_class=HTMLResponse)
def pricing_page(request: Request):
    account = _account(request)
    return templates.TemplateResponse(
        request=request,
        name="pricing.html",
        context={
            "request": request, "plans": PLANS,
            "payments_enabled": SETTINGS.payments_enabled, "account": account,
            "allowance": analysis_allowance(int(account["id"])) if account else None,
        },
    )


@app.post("/share/{job_id}")
def share_report(request: Request, job_id: str):
    _enforce_rate_limit(request, "report-share", 20, 3600)
    profile_id = _profile_id(request)
    fight = get_fight(job_id)
    if profile_id is None or not _request_plan(request).get("can_share"):
        raise HTTPException(403, "Private report sharing is available on Athlete, Pro, Coach and Gym plans.")
    if not fight or int(fight["profile_id"]) != profile_id:
        raise HTTPException(404)
    token = session_token()
    expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    save_report_share(job_id, profile_id, token_digest(token), expires)
    return RedirectResponse(f"/s/{token}", status_code=303)


@app.get("/s/{token}", response_class=HTMLResponse)
def shared_report(request: Request, token: str):
    share = get_report_share(token_digest(token))
    if not share:
        raise HTTPException(404)
    path = OUTPUTS / share["job_id"] / "report.json"
    if not path.exists():
        raise HTTPException(404)
    report = json.loads(path.read_text(encoding="utf-8"))
    _apply_report_annotations(report, [])
    refresh_identity_integrity(report)
    return templates.TemplateResponse(
        request=request, name="shared.html",
        context={"request": request, "report": report, "expires_at": share["expires_at"]},
    )


@app.post("/shares/{job_id}/revoke")
def revoke_shares(request: Request, job_id: str):
    profile_id = _profile_id(request)
    fight = get_fight(job_id)
    if profile_id is None or not fight or int(fight["profile_id"]) != profile_id:
        raise HTTPException(404)
    revoke_report_shares(job_id, profile_id)
    return RedirectResponse(f"/result/{job_id}", status_code=303)


@app.post("/account/export")
def export_account_data(request: Request, password: str = Form(...)):
    _enforce_rate_limit(request, "account-export", 5, 3600)
    account = _account(request)
    if not account or not authenticate(account["email"], password):
        raise HTTPException(400, "Enter the current account password to export your data.")
    profile_id = int(account["profile_id"])
    profile = dict(get_profile(profile_id) or {})
    profile["has_photo"] = bool(profile.pop("photo_path", None))
    profile["has_profile_video"] = bool(profile.pop("video_path", None))
    fights = []
    annotations = {}
    for saved_fight in list_fights(profile_id):
        fight = dict(saved_fight)
        fight.pop("video_path", None)
        fight.pop("report_path", None)
        fights.append(fight)
        safe_annotations = []
        for saved_annotation in get_annotations(fight["job_id"]):
            annotation = dict(saved_annotation)
            annotation["training_sequence_exported"] = bool(annotation.pop("sequence_path", None))
            safe_annotations.append(annotation)
        annotations[fight["job_id"]] = safe_annotations
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "policy_version": SETTINGS.policy_version,
        "account": {
            "email": account["email"], "plan": account.get("plan"),
            "plan_override": account.get("plan_override"), "created_at": account.get("created_at"),
            "account_status": account.get("account_status"),
            "terms_version": account.get("terms_version"),
            "privacy_version": account.get("privacy_version"),
            "policies_accepted_at": account.get("policies_accepted_at"),
            "marketing_consent": bool(account.get("marketing_consent")),
            "marketing_consent_at": account.get("marketing_consent_at"),
            "cookie_preferences": {
                "analytics": bool(account.get("cookie_analytics")),
                "marketing": bool(account.get("cookie_marketing")),
            },
            "subscription_status": account.get("subscription_status"),
            "subscription_period_end": account.get("subscription_period_end"),
        },
        "profile": profile,
        "fights": fights,
        "annotations": annotations,
        "coach_assignments": list_assignments(profile_id),
        "connected_sign_in_identities": list_oauth_identities(int(account["id"])),
        "legal_acceptances": list_legal_acceptances(profile_id=profile_id),
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": 'attachment; filename="warrioriq-account-data.json"', "Cache-Control": "no-store"},
    )


@app.post("/account/delete")
def delete_account_route(request: Request, password: str = Form(...), confirmation: str = Form(...)):
    account = _account(request)
    if not account:
        raise HTTPException(403)
    if confirmation.strip().upper() != "DELETE" or not authenticate(account["email"], password):
        raise HTTPException(400, "Enter DELETE and your current password to remove the account.")
    if any(
        job.get("owner_key") == f"account:{account['id']}" and job.get("status") in {"queued", "running"}
        for _, job in list_jobs()
    ):
        raise HTTPException(409, "Wait for the running analysis to finish before deleting the account.")
    if account.get("stripe_subscription_id"):
        try:
            cancellation = cancel_subscription_at_period_end(str(account["stripe_subscription_id"]))
        except Exception as exc:
            raise HTTPException(
                503,
                f"Account deletion is paused because subscription cancellation was not confirmed: {exc}",
            )
        if not cancellation.get("cancel_at_period_end"):
            raise HTTPException(503, "Account deletion is paused because subscription cancellation was not confirmed.")
    record_security_event("account_deletion_requested", account_id=int(account["id"]), severity="warning")
    removed = delete_account(int(account["id"]))
    if removed:
        for fight in removed["fights"]:
            _remove_fight_files(fight)
        profile = removed.get("profile") or {}
        for field in ("photo_path", "video_path"):
            _remove_profile_file(profile.get(field))
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.post("/checkout/{plan_key}")
def checkout(request: Request, plan_key: str, billing_acceptance: bool = Form(False)):
    account = _account(request)
    if not account:
        return RedirectResponse(f"/login?next=/pricing", status_code=303)
    if not billing_acceptance:
        raise HTTPException(400, "Confirm the recurring price, renewal and cancellation terms before checkout.")
    from core.readiness import release_readiness

    readiness = release_readiness(worker_status())
    if not readiness["release_ready"]:
        raise HTTPException(503, "Paid checkout is blocked until WarriorIQ's real operator identity, email verification, storage, and analysis worker are ready.")
    if plan_key not in PLANS or plan_key == "free":
        raise HTTPException(400, "Choose a valid paid plan.")
    record_legal_acceptance(
        "recurring_billing_terms", SETTINGS.policy_version,
        profile_id=int(account["profile_id"]), resource_id=plan_key,
        metadata={"price": PLANS[plan_key]["price"], "period": PLANS[plan_key]["period"]},
    )
    base = str(request.base_url).rstrip("/")
    try:
        url = create_checkout(
            plan_key, f"{base}/purchase/confirmation?session_id={{CHECKOUT_SESSION_ID}}", f"{base}/pricing?cancelled=1",
            int(account["id"]), account["email"],
        )
    except Exception as exc:
        raise HTTPException(400, str(exc))
    return RedirectResponse(url, status_code=303)


@app.get("/purchase/confirmation", response_class=HTMLResponse)
def purchase_confirmation(request: Request):
    account = _account(request)
    if not account:
        return RedirectResponse("/login?next=/purchase/confirmation", status_code=303)
    receipt = next(
        (message for message in list_outbound_messages(int(account["id"])) if message["message_type"] == "purchase_confirmation"),
        None,
    )
    return templates.TemplateResponse(
        request=request, name="purchase_confirmation.html",
        context={"request": request, "account": account, "receipt": receipt},
    )


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    try:
        event = verify_webhook(payload, request.headers.get("stripe-signature", ""))
    except Exception as exc:
        raise HTTPException(400, str(exc))
    if event.get("type") == "checkout.session.completed":
        session = event["data"]["object"]
        metadata = session.get("metadata") or {}
        account_id = metadata.get("warrioriq_account_id")
        plan_key = metadata.get("warrioriq_plan")
        plan = PLANS.get(plan_key)
        if account_id and plan and session.get("payment_status") in {"paid", "no_payment_required"}:
            recorded = apply_checkout_event(
                str(event.get("id", "")), str(event.get("type", "")), int(account_id),
                plan_key, int(plan.get("credits", 0)),
                customer_id=str(session.get("customer") or "") or None,
                subscription_id=str(session.get("subscription") or "") or None,
                subscription_status="active",
            )
            if recorded:
                account = get_account(int(account_id)) or {}
                currency = str(session.get("currency") or "eur").upper()
                amount = int(session.get("amount_total") or 0) / 100
                tax = int((session.get("total_details") or {}).get("amount_tax") or 0) / 100
                payment_date = datetime.fromtimestamp(int(session.get("created") or time.time()), tz=timezone.utc).isoformat()
                receipt_payload = {
                        "plan": plan["label"], "amount": f"{amount:.2f} {currency}",
                        "tax": f"{tax:.2f} {currency}", "payment_date": payment_date,
                        "renewal": f"Automatically renews {plan['period']} until cancelled",
                        "cancellation_path": "/settings/billing", "terms_path": "/terms",
                        "refunds_path": "/refunds",
                    }
                _queue_transactional_notice(
                    int(account_id), "purchase_confirmation", account.get("email", ""),
                    f"WarriorIQ {plan['label']} purchase confirmation",
                    f"Plan: {plan['label']}\nAmount: {amount:.2f} {currency}\nTax: {tax:.2f} {currency}\nPayment date: {payment_date}\nRenewal: automatically renews {plan['period']} until cancelled.\nCancel: /settings/billing\nRefunds and withdrawal: /refunds\nTerms: /terms",
                    receipt_payload,
                )
                record_security_event("purchase_confirmed", account_id=int(account_id), resource_type="plan", resource_id=plan_key)
    return {"received": True}
