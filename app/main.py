from __future__ import annotations

import json
import html
import math
import shutil
import time
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.state import create_job, delete_job, get_job, list_jobs, update_job
from core.analyzer import analyze, get_pose_tracker
from core.auth import authenticate, end_session, issue_session, register, resolve_session, session_token, token_digest
from core.config import OUTPUTS, ROOT, RULESET_LABELS, SETTINGS, UPLOADS
from core.annotations import accuracy_summary, export_sequence
from core.db import (
    add_assignment, analysis_allowance, apply_checkout_event, delete_account, delete_fight,
    delete_legal_acceptances_for_resource, get_annotations, get_fight,
    get_fight_review, get_profile, get_report_share, init_db, list_annotations, list_assignments,
    list_fights, list_legal_acceptances, record_legal_acceptance, release_analysis, reserve_analysis,
    revoke_report_shares, save_annotation, save_report_share,
    set_annotation_sequence, set_fight_review_status, toggle_assignment, update_profile,
)
from core.evidence_trust import report_evidence_trust
from core.coaching import build_coaching, build_training_plan
from core.payments import PLANS, create_checkout, plan_for_key, verify_webhook
from core.legal import LEGAL_DOCUMENTS, launch_readiness
from core.progress_insights import build_progress
from core.quality_guardian import inspect_video_quality
from core.report import build_preliminary_scorecard, refresh_identity_integrity
from core.retention import GUEST_RETENTION_HOURS, cleanup_expired_guest_jobs, guest_job_valid, mark_guest_job
from core.scoring import deduplicate_scoring_events, event_legality, is_verified_scoring_event, normalize_ruleset, score_fight
from core.types import AnalysisRequest, StrikeEvent
from core.video import get_video_info, read_frame

app = FastAPI(title="WarriorIQ")
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))
executor = ThreadPoolExecutor(max_workers=1)
init_db()

SESSION_COOKIE = "warrioriq_session"
GUEST_COOKIE = "warrioriq_guest"
ACTIVE_ANALYSIS_COOKIE = "warrioriq_active_analysis"
_last_guest_cleanup = 0.0
MAX_FIGHT_BYTES = 2 * 1024 * 1024 * 1024
MAX_PROFILE_PHOTO_BYTES = 15 * 1024 * 1024
MAX_PROFILE_VIDEO_BYTES = 500 * 1024 * 1024
_progress_report_cache: dict[str, tuple[int, dict]] = {}

PUBLIC_INDEX_ROUTES = (
    "/", "/pricing", "/privacy", "/legal", "/terms", "/cookies",
    "/acceptable-use", "/refunds", "/eula", "/dmca", "/accessibility",
    "/ai-transparency", "/security", "/subprocessors", "/contact", "/login", "/signup",
)
PRIVATE_ROUTE_PREFIXES = (
    "/api/", "/frame/", "/select/", "/progress/", "/result/", "/replay/", "/review/",
    "/media/", "/fighter-portrait/", "/selection-image/", "/dashboard", "/history",
    "/compare", "/coach", "/profile", "/validation", "/s/", "/share/", "/shares/",
    "/account/", "/checkout/", "/stripe/",
)


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
    predicted: dict
    fighter: str
    technique: str
    target: str
    outcome: str
    manual: bool = False


def _account(request: Request) -> dict | None:
    return getattr(request.state, "account", None)


def _profile_id(request: Request) -> int | None:
    account = _account(request)
    return int(account["profile_id"]) if account else None


def _request_plan(request: Request) -> dict:
    account = _account(request)
    return plan_for_key((account.get("plan_override") or account.get("plan")) if account else "free")


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


def _analysis_navigation_job(owner_key: str, preferred_job_id: str | None) -> dict | None:
    """Resolve the exact session selected by this browser before using a safe owner fallback."""
    if preferred_job_id:
        preferred = get_job(preferred_job_id)
        if preferred and preferred.get("owner_key") == owner_key and preferred.get("status") in {
            "queued", "running", "interrupted", "complete", "error",
        }:
            return {"job_id": preferred_job_id, **preferred}
    return _active_job_for_owner(owner_key)


def _analysis_navigation_url(job: dict) -> str:
    job_id = job["job_id"]
    return f"/result/{job_id}" if job.get("status") == "complete" else f"/progress/{job_id}"


def _safe_next(value: str | None, fallback: str = "/dashboard") -> str:
    value = (value or "").strip()
    return value if value.startswith("/") and not value.startswith("//") else fallback


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
    total = 0
    try:
        with destination.open("wb") as handle:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, "The uploaded file is larger than this local build allows.")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


@app.middleware("http")
async def viewer_context(request: Request, call_next):
    global _last_guest_cleanup
    request.state.account = resolve_session(request.cookies.get(SESSION_COOKIE))
    guest_id = request.cookies.get(GUEST_COOKIE)
    new_guest = not guest_id or len(guest_id) < 24 or len(guest_id) > 96
    request.state.guest_id = session_token() if new_guest else guest_id
    request.state.active_analysis = _analysis_navigation_job(
        _owner_key(request), request.cookies.get(ACTIVE_ANALYSIS_COOKIE),
    )
    request.state.launch = launch_readiness()
    request.state.noindex = (
        not SETTINGS.public_base_url
        or request.url.path.startswith(PRIVATE_ROUTE_PREFIXES)
        or request.url.path not in PUBLIC_INDEX_ROUTES
    )
    request.state.canonical_url = (
        f"{SETTINGS.public_base_url}{request.url.path}"
        if SETTINGS.public_base_url and request.url.path in PUBLIC_INDEX_ROUTES else ""
    )
    request.state.social_image_url = (
        f"{SETTINGS.public_base_url}/static/warrioriq-logo.png" if SETTINGS.public_base_url else ""
    )
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path != "/stripe/webhook":
        expected_origin = f"{request.url.scheme}://{request.url.netloc}"
        source = request.headers.get("origin") or request.headers.get("referer")
        if source:
            parsed = urlsplit(source)
            source_origin = f"{parsed.scheme}://{parsed.netloc}"
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
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self' data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'",
    )
    if request.url.path.startswith(("/result/", "/replay/", "/media/", "/api/", "/profile", "/history", "/dashboard", "/coach", "/s/")):
        response.headers.setdefault("Cache-Control", "no-store")
    elif request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "public, max-age=604800")
    if request.url.scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if new_guest:
        response.set_cookie(
            GUEST_COOKIE, request.state.guest_id, max_age=60 * 60 * 24,
            httponly=True, samesite="lax", secure=request.url.scheme == "https",
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
    }.get(exc.status_code, "WarriorIQ could not complete that request")
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={"request": request, "status_code": exc.status_code, "error_title": title, "error_detail": str(exc.detail or "")},
        status_code=exc.status_code,
        headers=exc.headers,
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
        closest = min(raw_events, key=lambda event: abs(float(event.get("peak_time", -999)) - event_time), default=None)
        if closest is None or abs(float(closest.get("peak_time", -999)) - event_time) > 0.04:
            closest = {
                "fighter": annotation["corrected"].get("fighter", "A"),
                "round_number": _round_number_at(report, event_time),
                "start_frame": 0, "peak_frame": 0, "end_frame": 0,
                "start_time": event_time, "peak_time": event_time, "end_time": event_time,
            }
        item = dict(closest)
        item["original_prediction"] = annotation["predicted"]
        item.update(annotation["corrected"])
        item["peak_time"] = event_time
        item["start_time"] = float(item.get("start_time", event_time))
        item["end_time"] = float(item.get("end_time", event_time))
        item["round_number"] = item.get("round_number") or _round_number_at(report, event_time)
        item["human_verified"] = True
        item["evidence_source"] = "human_ground_truth"
        item["is_corrected"] = annotation["predicted"] != annotation["corrected"]
        item["confidence"] = 1.0
        item["contact_confidence"] = 1.0
        if item.get("technique") == "none":
            displayed.pop(key, None)
        else:
            displayed[key] = item

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


def _run_job(job_id: str, req: AnalysisRequest):
    try:
        def cb(patch: dict):
            update_job(job_id, patch)

        update_job(job_id, {"status": "running", "message": "Starting GPU analysis"})
        report = analyze(req, cb)
        update_job(job_id, {"status": "complete", "report": report, "percent": 100.0, "message": "Complete"})
    except Exception as exc:
        job = get_job(job_id)
        if job and job.get("usage_reserved") and job.get("account_id"):
            release_analysis(int(job["account_id"]), job_id)
            update_job(job_id, {"usage_reserved": False})
        update_job(job_id, {"status": "error", "message": f"{type(exc).__name__}: {exc}"})


def _analysis_started_response(request: Request, job_id: str) -> JSONResponse:
    response = JSONResponse({"ok": True, "progress_url": f"/progress/{job_id}"})
    response.set_cookie(
        ACTIVE_ANALYSIS_COOKIE, job_id, max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", secure=request.url.scheme == "https",
    )
    return response


def _auth_page(request: Request, mode: str, error: str = "", next_path: str = "/dashboard"):
    return templates.TemplateResponse(
        request=request,
        name="auth.html",
        context={"request": request, "mode": mode, "error": error, "next_path": _safe_next(next_path)},
        status_code=400 if error else 200,
    )


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
    account_manager_confirmed: bool = Form(False),
):
    if not accept_terms or not account_manager_confirmed:
        return _auth_page(
            request, "signup",
            "Confirm that you can legally manage this account and accept the Terms, Privacy Policy, and Acceptable Use Policy.",
            next_path,
        )
    try:
        account = register(email, password)
    except ValueError as exc:
        return _auth_page(request, "signup", str(exc), next_path)
    record_legal_acceptance(
        "account_terms_privacy_acceptable_use_manager", SETTINGS.policy_version,
        profile_id=int(account["profile_id"]),
        metadata={"guardian_managed_child_access": True, "source": "signup"},
    )
    response = RedirectResponse(_safe_next(next_path), status_code=303)
    response.set_cookie(
        SESSION_COOKIE, issue_session(int(account["id"])), max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", secure=request.url.scheme == "https",
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
    if not accept_policies:
        return _auth_page(
            request, "login",
            "Confirm the Terms, Privacy Policy, and Acceptable Use Policy to sign in.",
            next_path,
        )
    account = authenticate(email, password)
    if not account:
        return _auth_page(request, "login", "The email or password is incorrect.", next_path)
    record_legal_acceptance(
        "account_signin_policies", SETTINGS.policy_version,
        profile_id=int(account["profile_id"]),
        metadata={"source": "login"},
    )
    response = RedirectResponse(_safe_next(next_path), status_code=303)
    response.set_cookie(
        SESSION_COOKIE, issue_session(int(account["id"])), max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", secure=request.url.scheme == "https",
    )
    return response


@app.post("/logout")
def logout(request: Request):
    end_session(request.cookies.get(SESSION_COOKIE))
    response = RedirectResponse("/", status_code=303)
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
            "rulesets": RULESET_LABELS,
            "profile": profile,
            "version": SETTINGS.version,
            "allowance": analysis_allowance(int(account["id"])) if account else None,
        },
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
    guardian_authorized_upload: bool = Form(False),
):
    if not rights_confirmed or not guardian_authorized_upload:
        raise HTTPException(
            400,
            "Confirm that you are authorised to use the footage and that a parent or legal guardian manages any upload involving a child.",
        )
    account = _account(request)
    if account:
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
    _save_upload_limited(video, video_path, MAX_FIGHT_BYTES)

    try:
        info = get_video_info(video_path)
        quality = inspect_video_quality(video_path, info)
    except Exception:
        video_path.unlink(missing_ok=True)
        raise
    start = max(0.0, min(float(start_seconds), max(0.0, info.duration - 0.001)))
    end = None if not end_seconds.strip() else max(start, min(float(end_seconds), info.duration))
    count = max(1, min(20, int(round_count)))
    selection_frame = int(round(start * info.fps))
    frame = read_frame(video_path, selection_frame)

    job_dir = OUTPUTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    selection_path = job_dir / "selection.jpg"
    cv2.imwrite(str(selection_path), frame)

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
            "openai_identity_recovery": bool(openai_identity_recovery),
        },
    )
    acceptance_owner = {"profile_id": int(account["profile_id"])} if account else {"guest_id": request.state.guest_id}
    record_legal_acceptance(
        "fight_upload_rights", SETTINGS.policy_version,
        resource_id=job_id,
        metadata={"ruleset": normalize_ruleset(ruleset), "external_ai_enabled": bool(openai_identity_recovery)},
        **acceptance_owner,
    )
    if openai_identity_recovery:
        record_legal_acceptance(
            "external_ai_frame_processing", SETTINGS.policy_version,
            resource_id=job_id,
            metadata={"provider": "OpenAI", "purpose": "fighter_identity_recovery"},
            **acceptance_owner,
        )
    response = RedirectResponse(f"/frame/{job_id}", status_code=303)
    response.set_cookie(
        ACTIVE_ANALYSIS_COOKIE, job_id, max_age=60 * 60 * 24 * 30,
        httponly=True, samesite="lax", secure=request.url.scheme == "https",
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
    """Pre-warm YOLO and return clickable person boxes for fighter selection."""
    job = _authorized_job(request, job_id)
    path = OUTPUTS / job_id / "selection.jpg"
    if not job or not path.exists():
        raise HTTPException(404)
    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(500, "Could not read selection image")
    tracker = get_pose_tracker()
    tracker.warmup(frame)
    results = tracker.model.predict(
        frame,
        device=tracker.device,
        imgsz=SETTINGS.default_imgsz,
        conf=SETTINGS.detection_conf,
        classes=[0],
        verbose=False,
    )
    boxes = []
    result = results[0]
    if result.boxes is not None:
        for box, conf in zip(result.boxes.xyxy.detach().cpu().numpy(), result.boxes.conf.detach().cpu().numpy()):
            boxes.append({"box": [float(x) for x in box], "confidence": float(conf)})
    return {"people": boxes, "width": job["video_width"], "height": job["video_height"]}


@app.post("/api/start/{job_id}")
def start(request: Request, job_id: str, payload: StartPayload):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    if job.get("status") in {"queued", "running"}:
        return _analysis_started_response(request, job_id)
    if len(payload.fighter_a_box) != 4 or len(payload.fighter_b_box) != 4:
        raise HTTPException(400, "Each fighter selection needs four coordinates.")
    # A/B is the report focus, not a tracking shortcut. WarriorIQ always
    # analyzes both selected fighters so identity context and the scorecard do
    # not disappear when the user asks for a detailed report on one athlete.
    focus_fighter = (payload.focus_fighter or payload.analysis_target or "A").upper()
    if focus_fighter not in {"A", "B"}:
        raise HTTPException(400, "Choose Fighter A or Fighter B for the detailed report.")

    _save_fighter_portrait(job_id, "A", payload.fighter_a_box)
    _save_fighter_portrait(job_id, "B", payload.fighter_b_box)

    req = _analysis_request(job_id, job, payload.fighter_a_box, payload.fighter_b_box, focus_fighter)
    if job.get("account_id") and not job.get("usage_reserved"):
        if not reserve_analysis(int(job["account_id"]), job_id):
            plan = _request_plan(request)
            raise HTTPException(429, f"Your {plan['label']} plan includes {plan['limit_label'].lower()}. Your allowance will reset automatically.")
        update_job(job_id, {"usage_reserved": True})
    update_job(job_id, {
        "status": "queued", "message": "Queued for fight analysis", "percent": 0.0,
        "analysis_target": "BOTH", "focus_fighter": focus_fighter,
        "fighter_a_box": payload.fighter_a_box, "fighter_b_box": payload.fighter_b_box,
    })
    executor.submit(_run_job, job_id, req)
    return _analysis_started_response(request, job_id)


@app.post("/api/restart/{job_id}")
def restart_interrupted_analysis(request: Request, job_id: str):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    if job.get("status") in {"queued", "running"}:
        return _analysis_started_response(request, job_id)
    if job.get("status") != "interrupted":
        raise HTTPException(409, "Only an analysis interrupted by a server restart can be resumed here.")
    fighter_a_box = job.get("fighter_a_box")
    fighter_b_box = job.get("fighter_b_box")
    focus_fighter = job.get("focus_fighter") or "A"
    if not isinstance(fighter_a_box, list) or len(fighter_a_box) != 4 or not isinstance(fighter_b_box, list) or len(fighter_b_box) != 4:
        raise HTTPException(409, "The saved session predates resumable analysis. Return to fighter selection once; your video is still available.")
    req = _analysis_request(job_id, job, fighter_a_box, fighter_b_box, focus_fighter)
    update_job(job_id, {
        "status": "queued", "message": "Restarting the preserved analysis session", "percent": 0.0,
        "eta_seconds": None, "live_events": [], "provisional_stats": {}, "latest_observation": None,
    })
    executor.submit(_run_job, job_id, req)
    return _analysis_started_response(request, job_id)


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
def status(request: Request, job_id: str):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    return _public_job_status(job_id, job)


@app.get("/api/active-analysis")
def active_analysis(request: Request):
    job = _analysis_navigation_job(
        _owner_key(request), request.cookies.get(ACTIVE_ANALYSIS_COOKIE),
    )
    if not job:
        return {"active": False}
    return {
        "active": True,
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
    return templates.TemplateResponse(request=request, name="result.html", context={
        "request": request, "job_id": job_id, "report": report,
        "report_access": report_access,
        "analysis_quality": _analysis_quality_summary(report),
        "can_share": bool(_account(request) and report_access.get("can_share")),
    })


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
    if (
        not math.isfinite(payload.event_time) or payload.event_time < 0
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
                 "outcome": outcome, "family": family, "limb": limb}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    refresh_identity_integrity(report)
    if not report.get("integrity", {}).get("identity_evidence_trusted", True):
        raise HTTPException(409, "Choose the fighters again before correcting action evidence.")
    predicted = _prediction_at(report, payload.event_time)
    if predicted is None and not payload.manual:
        raise HTTPException(404, "No analyzed action exists at that time")
    if predicted is None:
        segment_end = float(report.get("setup", {}).get("end_seconds") or (
            float(report.get("setup", {}).get("start_seconds", 0))
            + float(report.get("performance", {}).get("segment_duration_seconds", 0))
        ))
        if payload.event_time > segment_end + 0.05:
            raise HTTPException(400, "The label time is outside the analyzed segment")
        predicted = {"fighter": fighter, "technique": "none", "target": None,
                     "outcome": "uncertain", "family": "none", "limb": "none"}
    annotation_id = save_annotation(job_id, payload.event_time, report.get("setup", {}).get("ruleset", "K1"), predicted, corrected)
    profile = get_profile(_profile_id(request))
    training_consent = bool(profile and profile.get("allow_model_training"))
    sequence_path = export_sequence(job_id, annotation_id, corrected, payload.event_time) if training_consent else None
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
    return templates.TemplateResponse(request=request, name="validation.html", context={"request": request, "summary": accuracy_summary(annotations), "annotations": annotations})


def _build_replay_chapters(report: dict, focus: str) -> tuple[list[dict], str]:
    """Build useful replay navigation without turning candidates into facts."""
    focus = focus if focus in {"A", "B", "BOTH"} else "BOTH"
    verified = []
    for event in report.get("key_moments", []):
        if focus != "BOTH" and event.get("fighter") != focus:
            continue
        try:
            event_time = float(event.get("peak_time"))
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
def replay_page(request: Request, job_id: str):
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
    replay_chapters, replay_mode = _build_replay_chapters(report, focus)
    return templates.TemplateResponse(
        request=request,
        name="replay.html",
        context={
            "request": request, "job_id": job_id, "report": report,
            "identity_safe": identity_safe,
            "replay_chapters": replay_chapters,
            "replay_mode": replay_mode,
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
        _save_upload_limited(photo, file_path, MAX_PROFILE_PHOTO_BYTES)
        if cv2.imread(str(file_path)) is None:
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
        _save_upload_limited(profile_video, file_path, MAX_PROFILE_VIDEO_BYTES)
        try:
            get_video_info(file_path)
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
        "model_training_opt_in" if allow_model_training else "model_training_opt_out",
        SETTINGS.policy_version,
        profile_id=profile_id,
        metadata={"enabled": bool(allow_model_training), "source": "profile"},
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
            "launch": launch_readiness(), "policy_version": SETTINGS.policy_version,
        },
    )


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


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    if not SETTINGS.public_base_url:
        return PlainTextResponse("User-agent: *\nDisallow: /\n")
    disallowed = "\n".join(f"Disallow: {prefix}" for prefix in PRIVATE_ROUTE_PREFIXES)
    return PlainTextResponse(
        f"User-agent: *\nAllow: /\n{disallowed}\nSitemap: {SETTINGS.public_base_url}/sitemap.xml\n"
    )


@app.get("/health", include_in_schema=False)
def health_check():
    """Minimal deployment probe with no account, model or filesystem details."""
    return {"status": "ok", "service": "WarriorIQ"}


@app.get("/sitemap.xml")
def sitemap_xml():
    base = SETTINGS.public_base_url
    routes = (
        "/", "/pricing", "/privacy", "/legal", "/terms", "/cookies", "/acceptable-use",
        "/refunds", "/eula", "/dmca", "/accessibility", "/ai-transparency", "/security", "/subprocessors", "/contact",
    )
    urls = "" if not base else "".join(
        f"<url><loc>{html.escape(base + path)}</loc></url>" for path in routes
    )
    return Response(
        f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>',
        media_type="application/xml",
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
        },
        "profile": profile,
        "fights": fights,
        "annotations": annotations,
        "coach_assignments": list_assignments(profile_id),
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
    readiness = launch_readiness()
    if not readiness["ready"]:
        raise HTTPException(503, "Paid checkout is blocked until the real operator and legal contact details are configured.")
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
            plan_key, f"{base}/pricing?paid=1", f"{base}/pricing?cancelled=1",
            int(account["id"]), account["email"],
        )
    except Exception as exc:
        raise HTTPException(400, str(exc))
    return RedirectResponse(url, status_code=303)


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
            apply_checkout_event(
                str(event.get("id", "")), str(event.get("type", "")), int(account_id),
                plan_key, int(plan.get("credits", 0)),
            )
    return {"received": True}
