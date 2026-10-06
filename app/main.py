from __future__ import annotations

import io
import ipaddress
import json
import html
import hmac
import logging
import re
from logging.handlers import RotatingFileHandler
import math
import shutil
import threading
import time
import uuid
import os
import hashlib
import secrets
import zipfile
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, HTTPException,
                     Request, UploadFile)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
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
    analysis_run_directory, completed_artifact_directory, persist_completed_job,
    finalize_job_from_worker, get_job, list_jobs, prepare_job_run, record_worker_heartbeat,
    start_job_run, state_generation, update_job, update_job_for_worker, wake_status, worker_status,
    worker_status_heartbeat_age,
)
from core.auth import (
    authenticate, end_session, hash_password, issue_session, normalize_email, register, resolve_session,
    session_token, token_digest, valid_email, valid_password,
)
from core.identity import fighter_pair_similarity
from core.kit import alike_reason as kit_alike_reason, kit_similarity
from core.csrf import (
    issue_token as issue_csrf_token, tokens_match as csrf_tokens_match,
    usable_token as usable_csrf_token,
)
from core.config import (
    DATA_ROOT, DATASET, OUTPUTS, ROOT, RULESET_LABELS, RULESET_SHORT, RULESET_SPORTS, SETTINGS, UPLOADS,
)
from core.annotations import accuracy_summary, export_sequence
from core.model_validation import audit_dataset_split
from core.release_validation import assess_end_to_end_validation, end_to_end_metadata
from core.db import (
    HandleTaken, accept_follow, count_follows, follow_status, get_account_by_profile, get_profile_by_handle,
    list_follows, remove_follow, request_follow, set_public_profile,
    add_assignment, analysis_allowance, apply_subscription_change, award_points, link_camp_mission, list_camp_missions,
    list_points, list_training_sessions, record_training_session, redeem_points_for_analysis, apply_checkout_event, consume_email_verification_token,
    consume_password_reset_token,
    assign_fighter_to_fight, create_fighter, set_account_type, create_moderation_report, create_oauth_account, delete_account, delete_fight,
    delete_legal_acceptances_for_resource, get_account, get_account_by_email,
    get_account_for_oauth_identity, get_annotations, get_fight, get_fight_review, is_down_check, is_strike_check, get_fighter, get_profile,
    get_report_share, init_db, list_accounts, list_all_fight_storage, list_annotations, list_assignments,
    list_expired_fight_videos, list_fighters, list_fights, list_legal_acceptances, list_moderation_reports,
    link_oauth_identity, list_oauth_identities,
    list_outbound_messages, list_security_events, list_subscription_actions, mark_fight_video_deleted,
    mark_email_verified, mark_outbound_message_sent,
    queue_outbound_message, record_account_signup_acceptance, record_legal_acceptance,
    record_security_event, record_subscription_action, release_analysis, reserve_analysis,
    list_active_report_shares, resolve_moderation_report, revoke_account_sessions,
    revoke_report_shares, save_annotation,
    get_story_share, list_profile_posts, list_profile_story_shares, list_story_shares, revoke_story_shares,
    set_story_on_profile, story_share,
    save_email_verification_token, save_fight, save_password_reset_token, save_report_share,
    set_guardian_approval_status,
    set_account_status, set_annotation_sequence,
    page_view_summary, plan_interest_counts, plans_wanted_by, policies_outdated,
    record_page_view, record_plan_interest, record_policy_reacceptance,
    set_fight_review_status, toggle_assignment, update_cookie_preferences,
    update_marketing_consent, update_password_hash, update_profile,
)
from core.evidence_trust import accepted_model_event, report_evidence_trust
from core.fight_stats import normalize_outcome, summarize_fight_events
from core.coaching import build_coaching, build_training_plan
from core.payments import comparison_rows as plan_comparison, roster_capacity, PLANS, cancel_subscription_at_period_end, create_checkout, effective_plan_key, plan_for_key, subscription_change, verify_webhook
from core.legal import LEGAL_DOCUMENTS, launch_readiness, resolve_document
from core import feed, social, worker_alerts
from core.notifications import EmailNotSent, deliver_email, email_settings_problem, send_transactional_email
from core.progress_insights import build_progress
from core.quality_guardian import inspect_video_quality
from core.metric_catalog import BY_KEY as METRIC_CATALOG, readings as metric_readings
from core.chunked_upload import (
    ChunkedUploadError, StoredUpload, append as append_chunk, begin as begin_chunked,
    discard as discard_chunked, extend_lease, finalise as finalise_chunked,
    load as load_chunked, release_idle_sessions as release_idle_chunked_sessions,
)
from core.css_minify import minify as minify_css
from core.preflight_client import client_thresholds
from core.report_visuals import build as report_visuals
from core.upload_security import (
    FIGHT_VIDEO_ACCEPT, FIGHT_VIDEO_EXTENSIONS, FIGHT_VIDEO_LABEL,
    UploadBodyLimitMiddleware, UploadCapacityError, is_fight_upload, limit_upload_route, looks_like_video,
    scan_upload,
    reserve_upload_storage, release_upload_storage,
)
from core import sport_check
from core.fight_stats import _deduplicate as _deduplicate_strikes
from core.ground import STRIKING_SPORTS, looks_like_grappling
from core.camp import (
    DAILY_COUNTED_SESSIONS, IMPROVED_POINTS, MISSION_DONE_POINTS, REDEEM_COST, REDEEM_PER_MONTH,
    SESSION_POINTS, camp_standing,
    fight_camp_missions, mission_board, mission_result, missions_from_report, next_step, paid_sessions_on,
    points_history,
)
from core.training_check import check_training_video
from core.share_image import preview_png as story_preview_png
from core.count_plausibility import counts_implausible
from core.report import (
    build_preliminary_scorecard, identity_failure, identity_verdict, kick_minimum_check, observed_summary,
    ESTIMATE_NOTE, STRIKE_COUNTS_PRECISION_VALIDATED, STRIKE_COUNTS_PUBLISHED, published_families,
    coaching_moments, refresh_identity_integrity, share_card, unattributed_kick_total,
)
from core.retention import (
    GUEST_RETENTION_HOURS, cleanup_abandoned_processing_files, cleanup_expired_guest_jobs,
    guest_job_valid,
)
from core.build_info import ANALYSIS_VERSION, result_check
from core.orientation import needed_turn, tag_rotation
from core.person_detect import detect_people as detect_people_in_frame, find_clear_moment, pair_score
from core.scoring import RULESETS, SPORTS, deduplicate_scoring_events, event_legality, is_verified_scoring_event, normalize_ruleset, score_fight, sport_counted_families, sport_of, sport_unobserved, coverage_note
from core.sport_policy import counting_policy
from core.sport_profiles import SPORT_IDENTITIES, sport_identity
from core.coaching import count_of
from core.squad import build_squad_view, compare_movement, compare_with_previous, fight_choice_label, fight_choice_stamp
from core.squad import movement_value as squad_movement_value
from core.social_auth import SOCIAL_AUTH
from core.types import AnalysisRequest, StrikeEvent
from core.video import (
    detect_shot_changes, ffmpeg_frame, get_video_info, normalize_container, opencv_decodes,
    playback_file, probe_upload, read_frame, remove_derivative, selection_frame,
)
from core.video import _ffmpeg_exe

# No public API documentation. /openapi.json, /docs and /redoc were reachable
# signed out and listed every route - /admin/*, /api/worker/claim, the dataset
# and worker-job endpoints - which is a map for anyone probing the site and
# documents nothing a visitor can use. An administrator can still read the
# schema at /admin/openapi.json (see admin_openapi below).
app = FastAPI(title="WarriorIQ", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
app.add_middleware(UploadBodyLimitMiddleware)
_oauth_cookie_secure = SETTINGS.public_base_url.lower().startswith("https://")
def _session_secret() -> str:
    """A signing key that survives a restart and is shared across processes.

    This used to fall back to secrets.token_urlsafe(48) when the setting was
    missing, which reads as a safe default and is not one: the key is what
    signs the OAuth state cookie, so a fresh key per process means a sign-in
    begun on one worker cannot be finished on another, and every restart
    invalidates every login in flight. The symptom is an intermittent "the
    identity provider could not verify this sign in" that no amount of
    checking the provider's console explains.

    Set WARRIORIQ_OAUTH_STATE_SECRET in production. Without it, a key is
    generated once and kept beside the database so that at least it is stable
    for this host, and the warning says what to do.
    """
    configured = (SETTINGS.oauth_state_secret or "").strip()
    if configured:
        return configured
    path = DATA_ROOT / "session-secret.txt"
    try:
        if path.exists():
            stored = path.read_text(encoding="utf-8").strip()
            if stored:
                return stored
        generated = secrets.token_urlsafe(48)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated, encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        logging.getLogger("warrioriq").warning(
            "WARRIORIQ_OAUTH_STATE_SECRET is not set; generated one at %s. "
            "Set it in the environment instead - a key on disk is fine for a "
            "single host and wrong the moment there is a second one.", path)
        return generated
    except OSError:
        logging.getLogger("warrioriq").error(
            "WARRIORIQ_OAUTH_STATE_SECRET is not set and %s is not writable. "
            "Falling back to a per-process key: sign-in will fail whenever a "
            "callback lands on a different process than the one that started "
            "it, and after every restart.", path)
        return secrets.token_urlsafe(48)


app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret(),
    session_cookie="warrioriq_oauth",
    max_age=600,
    same_site="none" if _oauth_cookie_secure else "lax",
    https_only=_oauth_cookie_secure,
)
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))


def _fight_moment(value: str | None, with_time: bool = True) -> str:
    """A stored timestamp as a person reads it.

    Pages were printing the ISO string straight out of the database -
    "2026-09-14T15:52:56.914959+00:00" on /profile - beside a raw ruleset
    enum. Rendered in UTC here and relabelled to the reader's own zone by
    the script in base.html, which is why the markup is a <time> element.

    The audit expected a `timezone` cookie to read; there is none, and there
    is no timezone handling anywhere in the app. The browser already knows,
    so nothing has to be stored to ask it.
    """
    if not value:
        return "—"
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return str(value)[:10]
    day = f"{moment.day} {moment:%b %Y}"
    return f"{day}, {moment:%H:%M}" if with_time else day


def _ruleset_label(value: str | None) -> str:
    """"KICK_LIGHT" is a database value, not a thing to show somebody."""
    key = str(value or "").strip()
    if not key:
        return "—"
    return RULESET_LABELS.get(key, key.replace("_", " ").title())


templates.env.filters["fight_moment"] = _fight_moment
# "1 attempt", never "1 attempts": {{ n|count_of('attempt') }}.
templates.env.filters["count_of"] = count_of
templates.env.filters["ruleset_label"] = _ruleset_label


def _asset_version() -> str:
    """A cache key that changes whenever a stylesheet or script changes.

    base.html used to carry a hand-typed "?v=20260906-simple1". It never
    changed, so every CSS and JS deploy was invisible to anyone who had already
    visited: their browser kept serving the old stylesheet against the new
    markup, which looks like the site broke rather than like a stale cache.

    Hashing the files themselves means the token moves on its own with every
    real change and stays put when nothing changed, so caches still do their
    job between deploys.

    The contents, not the modification time. This read st_mtime_ns, and the
    deploy copies the tree with `cp -R` (see .cpanel.yml), which writes a fresh
    mtime on every file whether or not it changed. So the token moved on every
    single deploy and threw away the stylesheet cache of everybody who had ever
    visited - the exact opposite of the sentence above it, and the reason this
    was worth fixing on a host where the server already answers in 8 to 27
    seconds against an application that takes 25 to 46 milliseconds.

    Reading 231 KB of CSS and JS at import costs about a millisecond and
    happens once per process.
    """
    digest = hashlib.sha256()
    static_dir = ROOT / "app" / "static"
    for path in sorted(static_dir.glob("*.css")) + sorted(static_dir.glob("*.js")):
        try:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:12]


ASSET_VERSION = _asset_version()
templates.env.globals["asset_version"] = ASSET_VERSION
# The picker filter and the sentence under it both come from the one tuple in
# core.upload_security, so neither can drift from what the server accepts.
templates.env.globals["fight_video_accept"] = FIGHT_VIDEO_ACCEPT
templates.env.globals["fight_video_label"] = FIGHT_VIDEO_LABEL
templates.env.globals["fight_video_extensions"] = sorted(FIGHT_VIDEO_EXTENSIONS)
# The pre-upload check compares footage against the same numbers the worker's
# probe uses, rather than a second set copied into JavaScript.
templates.env.globals["preflight_limits"] = json.dumps(client_thresholds())
# One name, one definition and one reference per measurement, so the report
# cannot call the same number three things in three sections.
templates.env.globals["metric_catalog"] = METRIC_CATALOG
templates.env.filters["metric_readings"] = metric_readings
# Pressure 0-100, centre as a percentage, footwork in body lengths a second:
# the units the report and Progress already use. See core.squad.movement_value.
templates.env.filters["movement_value"] = squad_movement_value

# Every page pulled eight separate stylesheets, so a phone opening WarriorIQ
# made eight blocking round trips to a shared host before it could paint
# anything. They are concatenated here instead, at import, with no build step.
#
# Two bundles rather than one, because base.html puts {% block head %} between
# these groups: a page's own stylesheet has to land after the first group and
# before the second, or the cascade changes and pages restyle themselves. Two
# requests instead of eight keeps the order identical.
CSS_BUNDLES: dict[str, tuple[str, ...]] = {
    "base": ("style.css", "navigation.css", "fixes.css", "product.css", "system.css"),
    # a11y.css is last, and has to stay last: it sets floor values for tap
    # targets and control font size that must win over whatever a component
    # asked for, and load order is how they do that without !important.
    "shell": ("premium.css", "motion.css", "redesign.css", "a11y.css"),
}


def _build_css_bundle(names: tuple[str, ...]) -> str:
    parts = []
    for name in names:
        path = ROOT / "app" / "static" / name
        try:
            parts.append(f"/* {name} */\n{path.read_text(encoding='utf-8')}")
        except OSError:
            LOGGER.warning("css_bundle_missing file=%s", name)
    # Concatenated, then squeezed. Both happen once, at import: 216 KB of
    # stylesheet reached every page on a site whose heaviest page carries
    # one image. The files stay readable on disk; only what goes over the
    # wire is shrunk, and core/css_minify.py is deliberately timid about
    # what it will touch.
    joined = "\n".join(parts)
    return minify_css(joined) if SETTINGS.css_minify_enabled else joined


CSS_BUNDLE_TEXT = {key: _build_css_bundle(names) for key, names in CSS_BUNDLES.items()}
executor = ThreadPoolExecutor(max_workers=1)
init_db()

SESSION_COOKIE = "warrioriq_session"
GUEST_COOKIE = "warrioriq_guest"
ACTIVE_ANALYSIS_COOKIE = "warrioriq_active_analysis"
ACTIVE_SPORT_COOKIE = "warrioriq_sport"
LAST_COMPLETED_ANALYSIS_COOKIE = "warrioriq_last_completed_analysis"
COOKIE_PREFERENCES_COOKIE = "warrioriq_cookie_preferences"
CSRF_COOKIE = "warrioriq_csrf"
# Routes that a browser never posts to, and that therefore cannot carry a
# token: the worker authenticates with a bearer token compared in constant
# time, and Stripe signs its webhook body. Both are checked before anything
# happens, so a forged request from a browser fails there instead. Exempting
# them is not a hole; requiring a cookie-derived token from a machine client
# that has no cookies would simply break them.
CSRF_EXEMPT_PREFIXES = ("/api/worker/", "/stripe/")
# The OAuth callback is a cross-site POST *by design*: providers using the
# form_post response mode post the result straight from their own origin, so it
# can never carry a token of ours. It has the defence the OAuth spec prescribes
# instead - a `state` value signed into the session cookie, matched against the
# provider and refused after 600 seconds, all before anything happens. This is
# an exact path, not a prefix, because /auth/{provider}/start posts from our
# own form and must stay protected.
CSRF_EXEMPT_PATHS = ("/auth/{provider}/callback",)
_last_guest_cleanup = 0.0
_last_saved_video_cleanup = 0.0
_rate_windows: dict[str, list[float]] = {}
MAX_FIGHT_BYTES = SETTINGS.max_fight_bytes
MAX_PROFILE_PHOTO_BYTES = 15 * 1024 * 1024
MAX_PROFILE_VIDEO_BYTES = 500 * 1024 * 1024
limit_upload_route("/profile", MAX_PROFILE_PHOTO_BYTES + MAX_PROFILE_VIDEO_BYTES)
_progress_report_cache: dict[str, tuple[int, dict]] = {}
LOGGER = logging.getLogger("warrioriq")


def _configure_logging() -> None:
    """Give the application's warnings somewhere to land.

    Every logger here was created and none was ever given a handler, so Python
    fell back to writing warnings at stderr - which under Passenger goes to a
    log this host has not written to since August. The result was that the one
    line naming why a sign-in failed, or why an email was not sent, existed in
    the code and was visible nowhere, and every failure had to be reverse
    engineered from outside the server.

    Writes beside the database, which is by definition a directory this account
    can write to. A failure to open the file must never take the site down with
    it: no log is bad, no site is worse.
    """
    logger = logging.getLogger("warrioriq")
    logger.setLevel(logging.INFO)
    if any(getattr(handler, "_warrioriq_file", False) for handler in logger.handlers):
        return
    target = os.getenv("WARRIORIQ_LOG_FILE", "").strip()
    path = Path(target) if target else DATA_ROOT / "warrioriq.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    except OSError as exc:
        logging.getLogger().warning("warrioriq_log_unavailable path=%s error=%s", path, type(exc).__name__)
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler._warrioriq_file = True
    logger.addHandler(handler)
    logger.propagate = False


_configure_logging()

# Every page that may be indexed, and - because sitemap_xml is built from this
# same tuple - every page offered to a crawler. They were two hand-maintained
# lists with the same contents, which is how /analyze came to be missing from
# both: the sport setup pages carry a unique title, description and body per
# sport and are the natural landing page for "kickboxing fight analysis", but
# they were served noindex,nofollow with no canonical URL and were absent from
# the sitemap. One list, so a page cannot be offered to a crawler that the
# page itself then refuses.
PUBLIC_INDEX_ROUTES = (
    "/", "/pricing", "/privacy", "/legal", "/terms", "/cookies",
    "/acceptable-use", "/refunds", "/video-upload-policy", "/sports-medical-disclaimer",
    "/eula", "/dmca", "/accessibility", "/ai-transparency", "/security",
    "/subprocessors", "/contact", "/kickboxing-fight-analysis", "/k1-fight-analysis",
    "/fight-video-analysis-for-coaches", "/how-to-record-a-fight-for-analysis",
    "/analyze", "/analyze/kickboxing", "/analyze/boxing", "/analyze/muay_thai",
    "/analyze/taekwondo", "/analyze/mma",
)
# Pages where the sport chip is noise rather than navigation. Kept next to the
# other route groupings so it is obvious there are three of them. Plans,
# Progress and Coach used to be listed too, which made the header change shape
# between the main workspace tabs - the chip appeared on Analyze and the fight
# library and vanished on the three tabs beside them.
SPORT_IRRELEVANT_PREFIXES = ("/profile", "/settings", "/legal", "/privacy", "/terms")
PRIVATE_ROUTE_PREFIXES = (
    "/api/", "/frame/", "/select/", "/pending/", "/progress/", "/result/", "/replay/", "/review/",
    "/media/", "/fighter-portrait/", "/selection-image/", "/live-frame/", "/dashboard", "/history",
    "/compare", "/coach", "/camp", "/profile", "/validation", "/s/", "/share/", "/shares/",
    "/account/", "/settings/", "/admin", "/checkout/", "/stripe/", "/purchase/",
    "/auth/", "/guardian",
)
# Private areas robots.txt does not name. Listing them there advertised the
# admin console, validation tooling and payment webhook to anyone who reads the
# file (QA, 2026-09); they are kept out of search by the X-Robots-Tag header
# every private response carries instead (_apply_response_headers).
UNADVERTISED_PRIVATE_PREFIXES = ("/admin", "/validation", "/stripe/")

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
            {"question": "Can WarriorIQ analyse sparring as well as competition footage?", "answer": "Yes. Sparring and competition footage are analysed the same way."},
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
    # Not sent for a solo session, which follows one person (core/solo.py).
    fighter_b_box: list[float] | None = None
    solo: bool = False
    focus_fighter: str | None = None
    analysis_target: str | None = None
    # Which corner Fighter A is in, as the person drawing the boxes says it.
    # Optional so older clients keep working; None means "not said", and the
    # report then names no corner rather than guessing red for A.
    fighter_a_corner: str | None = None


class PairCheckPayload(BaseModel):
    fighter_a_box: list[float]
    fighter_b_box: list[float]


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


class SportCheckPayload(BaseModel):
    sport: str
    frames: list[str]


class StrikeCheckPayload(BaseModel):
    seconds: float
    fighter: str
    family: str
    verdict: str


# What a fighter can say about one counted strike, in one tap. "punch", "kick"
# and "knee" mean it was a strike of that type rather than the one counted.
STRIKE_CHECK_VERDICTS = ("right", "not_a_strike", "wrong_fighter", "punch", "kick", "knee")


class DownCheckPayload(BaseModel):
    seconds: float
    verdict: str


# Who went down at a moment core.ground found: the analysis cannot tell, so
# the owner says. "nobody" means nobody went down there.
DOWN_CHECK_VERDICTS = ("A", "B", "nobody")
# Annotations are one row per fight and time. Strike answers are stored at
# the strike's time rounded to the millisecond, so a down answer is stored
# half a millisecond later and can never overwrite one at the same instant.
DOWN_CHECK_TIME_OFFSET = 0.0005


class WorkerIdentityPayload(BaseModel):
    worker_id: str
    # core/build_info.ANALYSIS_VERSION of the worker's code. Absent from
    # workers built before it existed, which are older by definition.
    analysis_version: int | None = None


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


def _trusted_request_host(value: str) -> bool:
    if not value or any(char in value for char in "/\\@?#, \t\r\n"):
        return False
    try:
        parsed = urlsplit(f"//{value}")
        _ = parsed.port
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    public_host = urlsplit(SETTINGS.public_base_url).hostname
    allowed = {"localhost", "127.0.0.1", "::1", *SETTINGS.allowed_hosts}
    if public_host:
        allowed.update({public_host.lower(), f"www.{public_host.lower()}"})
    if host in allowed:
        return True
    # Infrastructure reaches the application by address, not by name: a shared
    # host's health check, a local probe, a container gateway. Rejecting those
    # takes the whole site down for a header nobody chose, and an address
    # cannot do the damage this check exists to prevent - every URL WarriorIQ
    # emits is pinned by _public_base, so a Host header can never decide where
    # a password-reset or OAuth token is sent.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _public_base(request: Request) -> str:
    """Never let an incoming Host header choose where account tokens are sent."""
    configured = SETTINGS.public_base_url
    if configured:
        parsed = urlsplit(configured)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise HTTPException(503, "The site's public address needs configuration.")
        return configured
    # Local development has no public origin. Host validation runs before routes.
    if request.url.hostname in {"localhost", "127.0.0.1", "::1"}:
        return str(request.base_url).rstrip("/")
    return "https://warrioriq.eu"


def _request_is_secure(request: Request) -> bool:
    return _external_scheme(request) == "https"


def _public_analysis_error(exc: Exception) -> str:
    """Return a useful status without leaking model names or server paths."""
    if isinstance(exc, (FileNotFoundError, ImportError, ModuleNotFoundError)):
        return "The analysis engine is unavailable on this server. Your upload and fighter selections are preserved."
    if type(exc).__name__ == "UnreadableVideo":
        return ("The analysis machine could not decode this video's format, so nothing was analysed. "
                "Your upload and fighter selections are preserved; exporting the video as MP4 "
                "(H.264) and uploading that copy will work.")
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return "This analysis exceeded the server's available memory. Your upload and fighter selections are preserved."
    # The upload finished and the video was decoded here well enough to cut a
    # fighter-selection frame, so a file the analysis machine cannot open means
    # the copy to that machine stopped early rather than that the footage is
    # bad. Saying "could not analyse your video" sends people off to re-export
    # a file that was never the problem; starting again is what actually works.
    detail = str(exc).lower()
    if "could not open" in detail or "stopped early" in detail or "could not read" in detail:
        return (
            "The video did not reach the analysis machine in one piece, so nothing was analysed. "
            "Your upload and fighter selections are preserved - start the analysis again."
        )
    return "WarriorIQ could not finish this analysis. Your upload and fighter selections are preserved so you can try again."


def _worker_failure_message(error_code: str | None) -> str:
    """What a worker's failure means to the person waiting, by its cause."""
    if str(error_code or "") == "UnreadableVideo":
        return _public_analysis_error(type("UnreadableVideo", (RuntimeError,), {})())
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


# A fight that has not moved in this long is not being analysed any more. The
# worker reports progress continuously and its lease is three minutes, so
# silence for hours means the run died - the machine slept, the worker was
# killed, the process crashed. Without this the job sat in the top bar for ever:
# one real user carried "Analysis paused" at 39.9% from days earlier and had no
# way to clear it.
_STALE_PROCESSING_SECONDS = 2 * 60 * 60


def _is_live_processing_job(job: dict) -> bool:
    if job.get("status") not in {"queued", "running", "interrupted"}:
        return False
    updated = float(job.get("updated_at_epoch", 0) or 0)
    if updated <= 0:
        return False
    return (time.time() - updated) <= _STALE_PROCESSING_SECONDS


# How old the job listing behind the top-bar chip may be. Every request used
# to call list_jobs(), which opens, file-locks and parses every session file
# under outputs/ - hundreds on the live host, phantom jobs included - and it ran
# for static files too. Measured with 300 stored jobs: 34 ms of pure scanning
# per request against 5 ms without, before the shared host's 8-12x slowdown,
# so one person opening pages quickly was enough to saturate the account.
#
# A write in this process invalidates the snapshot at once (state_generation),
# so an upload or a start shows up on the very next page. Only changes made by
# another process - an external worker moving a job on - can be up to this
# many seconds late, and the chip's own poll reads the job directly.
_NAVIGATION_SNAPSHOT_SECONDS = 5.0
_navigation_snapshot: tuple[int, float, list[tuple[str, dict]]] | None = None
_navigation_snapshot_lock = threading.Lock()


def _navigation_jobs() -> list[tuple[str, dict]]:
    """list_jobs(), reused for a few seconds unless this process changed a job."""
    global _navigation_snapshot
    with _navigation_snapshot_lock:
        cached = _navigation_snapshot
        now = time.monotonic()
        if (cached is not None and cached[0] == state_generation()
                and now - cached[1] < _NAVIGATION_SNAPSHOT_SECONDS):
            return cached[2]
        generation = state_generation()
        jobs = list_jobs()
        # Stamped with the generation from before the scan: a write that lands
        # during it leaves the snapshot already stale, which is the safe side.
        _navigation_snapshot = (generation, time.monotonic(), jobs)
        return jobs


def _active_job_for_owner(owner_key: str) -> dict | None:
    jobs = [
        {"job_id": job_id, **job}
        for job_id, job in _navigation_jobs()
        if job.get("owner_key") == owner_key and _is_live_processing_job(job)
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
    if active is not None and not _is_live_processing_job(active):
        active = None
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
    r"""Only ever redirect to a path on this site.

    "starts with / and not //" is the usual rule and it is not enough. A
    browser normalises the backslash in "/\evil.example" to a forward slash
    before resolving it, so that value passes the test and then leaves the
    site - the classic bypass. Control characters can be used the same way,
    to smuggle something past a check that reads only the first two.
    """
    value = (value or "").strip()
    if not value.startswith("/") or value.startswith("//"):
        return fallback
    if chr(92) in value:                      # any backslash at all
        return fallback
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return fallback
    return value


def _is_admin(request: Request) -> bool:
    account = _account(request)
    return bool(account and account.get("email", "").lower() in SETTINGS.admin_emails)


async def require_csrf(request: Request) -> None:
    """Refuse a state-changing request that cannot echo the visitor's token.

    Applied per route rather than in middleware, deliberately. Checking this in
    middleware means reading the request body before the endpoint does, and
    getting that wrong silently empties uploads; a dependency runs inside the
    normal request lifecycle, and FastAPI hands the endpoint the same parsed
    form it hands this. `test_every_browser_mutation_requires_a_csrf_token`
    is what makes "per route" safe - it walks the route table and fails if any
    unsafe route is missing this, so a new endpoint cannot quietly skip it.

    The header is checked first so that fetch callers never pay for form
    parsing. For a JSON body `request.form()` returns empty without touching
    the stream, because Starlette dispatches on the content type - so asking
    for the field costs nothing on those routes and cannot consume the body.
    """
    submitted = request.headers.get("x-csrf-token")
    if not submitted:
        form = await request.form()
        submitted = form.get("csrf_token")
    expected = usable_csrf_token(request.cookies.get(CSRF_COOKIE))
    if csrf_tokens_match(expected, submitted if isinstance(submitted, str) else None):
        return
    record_security_event(
        "csrf_token_rejected", severity="warning",
        resource_type="route", resource_id=request.url.path[:200],
    )
    # 403 rather than 400: the request was understood and refused. The wording
    # has to be usable by somebody who simply left a tab open overnight, which
    # is the common cause by a wide margin - not an attack.
    raise HTTPException(403, "This form expired or came from another site. Reload the page and try again.")


def _peer_is_routable(host: str) -> bool:
    """Is this an address a visitor could actually have reached us from?

    `is_global` is exactly the question: it is false for loopback, private and
    link-local ranges - the addresses a reverse proxy on this machine connects
    from - and also for the documentation ranges, which nobody browses from
    either. Anything true here reached us directly, so the socket is the truth.
    """
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def _client_ip(request: Request) -> str:
    """The visitor's address, as far as it can be trusted.

    This matters more than it looks. The rate limiter used `request.client.host`
    directly, and warrioriq.eu runs behind Apache and Passenger - so every
    visitor arrives from 127.0.0.1 and shares **one** bucket. One person
    failing to sign in thirty times would lock every other visitor out of the
    login page for five minutes, and no per-visitor limit was ever really in
    force. That is worse than having no limiter.

    `X-Forwarded-For` cannot simply be believed either: anyone can send it, and
    trusting it turns every limit into an opt-out. The rule here is to believe
    the socket whenever the connection came from a public address - an attacker
    on the internet cannot make their own connection appear to come from
    loopback - and to read the header only when the peer is a local proxy.

    The **rightmost** entry is used, not the leftmost. A proxy appends the
    address it actually saw to the end of the chain, so the right-hand end is
    the nearest hop's own observation; everything to the left of it is whatever
    the client chose to send. `_forwarded_header` takes the leftmost for host
    and scheme, which is the convention for those and the wrong one here.
    """
    peer = request.client.host if request.client else ""
    if peer and _peer_is_routable(peer):
        return peer
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        nearest = forwarded.rsplit(",", 1)[-1].strip()
        if nearest:
            return nearest[:64]
    return peer or "unknown"


# How many rate-limit buckets to keep. Each is a scope plus a client address,
# and the dict never used to drop any of them - a process serving a month of
# traffic accumulated an entry per address per endpoint and never freed one.
# Passenger keeps a worker alive for a long time, so this was a slow leak.
MAX_RATE_WINDOWS = 20000


def _prune_rate_windows(now: float) -> None:
    """Drop buckets whose newest timestamp is older than any window in use."""
    stale = [key for key, stamps in _rate_windows.items() if not stamps or now - stamps[-1] > 3600]
    for key in stale:
        _rate_windows.pop(key, None)
    if len(_rate_windows) > MAX_RATE_WINDOWS:
        # Still too many: drop the least recently touched. Losing a bucket
        # forgives requests already counted, which is the safe direction to
        # fail - it can let somebody through, never lock somebody out.
        for key in sorted(_rate_windows, key=lambda k: _rate_windows[k][-1])[:len(_rate_windows) - MAX_RATE_WINDOWS]:
            _rate_windows.pop(key, None)


# When each bucket last wrote a rate_limit_exceeded event. One event per
# window is the evidence; one per refused request turned a flood into a stream
# of database writes on exactly the occasion the server could least afford them.
_rate_limit_reported: dict[str, float] = {}


class RateLimited(Exception):
    """A request refused by a rate limit, with how long until it would pass."""

    def __init__(self, retry_after: int, detail: str):
        super().__init__(detail)
        self.retry_after = max(1, int(retry_after))
        self.detail = detail


def _take_rate_slot(scope: str, client: str, limit: int, window_seconds: int) -> int | None:
    """Count one request. Returns None if it may proceed, else seconds to wait."""
    now = time.monotonic()
    key = f"{scope}:{client}"
    recent = [stamp for stamp in _rate_windows.get(key, []) if now - stamp < window_seconds]
    if len(recent) >= limit:
        _rate_windows[key] = recent
        return math.ceil(window_seconds - (now - recent[0])) if recent else window_seconds
    recent.append(now)
    _rate_windows[key] = recent
    if len(_rate_windows) > MAX_RATE_WINDOWS:
        _prune_rate_windows(now)
    return None


def _report_rate_limit_once(scope: str, client: str, window_seconds: int) -> None:
    key = f"{scope}:{client}"
    now = time.monotonic()
    if now - _rate_limit_reported.get(key, -1e9) < window_seconds:
        return
    _rate_limit_reported[key] = now
    if len(_rate_limit_reported) > MAX_RATE_WINDOWS:
        _rate_limit_reported.clear()
    record_security_event(
        "rate_limit_exceeded", severity="warning", resource_type="route", resource_id=scope,
        metadata={"client": client},
    )


def _enforce_rate_limit(request: Request, scope: str, limit: int, window_seconds: int) -> None:
    """Small single-process safety limit; production should add an edge/shared limiter too.

    Always answered as a 429 with Retry-After, never by dropping the request.
    """
    client = _client_ip(request)
    wait = _take_rate_slot(scope, client, limit, window_seconds)
    if wait is None:
        return
    _report_rate_limit_once(scope, client, window_seconds)
    raise HTTPException(
        429, _rate_limit_detail(wait), headers={"Retry-After": str(max(1, wait))},
    )


def _rate_limit_detail(wait_seconds: int) -> str:
    wait_seconds = max(1, int(wait_seconds))
    if wait_seconds < 60:
        when = f"{wait_seconds} second{'s' if wait_seconds != 1 else ''}"
    else:
        minutes = math.ceil(wait_seconds / 60)
        when = f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"That was a lot of requests in a short time. Wait about {when} and try again - nothing you saved is lost."


# Segments that are an identifier rather than a page: all digits, or hex long
# enough to be a job id. Written to match those and nothing else, because a
# guide slug like /how-to-record-a-fight-for-analysis is long too, and folding
# it into ":id" would throw away the one number worth having - which page.
_IDENTIFIER_SEGMENT = re.compile(r"^(?:[0-9]+|[0-9a-f]{8,}|[0-9a-f-]{32,})$", re.IGNORECASE)

# Read from the user agent, never stored. Keeping the string would make this a
# record about a visitor instead of a count of pages, which is the whole
# distinction that lets it run without a consent banner.
_ROBOT_MARKERS = (
    "bot", "crawler", "spider", "slurp", "facebookexternalhit", "embedly",
    "headlesschrome", "python-requests", "curl/", "wget", "httpx", "axios",
    "lighthouse", "pagespeed", "pingdom", "uptimerobot", "monitoring", "preview",
)


def _counted_path(path: str) -> str:
    """The page a request is for, with identifiers folded away.

    Without this, one report per visitor is one row per visitor, and a table
    of a thousand paths seen once each answers nothing.
    """
    parts = [segment for segment in path.split("/") if segment]
    if not parts:
        return "/"
    return "/" + "/".join(
        ":id" if _IDENTIFIER_SEGMENT.match(segment) else segment[:40]
        for segment in parts[:4])


def _is_counted_page_view(request: Request, response) -> bool:
    """Whether this request was a person opening a public page.

    HTML only, which is what excludes assets, JSON polling and redirects
    without having to list them. Private routes are left out on purpose: the
    question this answers is whether anyone is finding the site, and counting
    the workspace would mostly count the owner using their own product.
    """
    if request.method != "GET" or response.status_code != 200:
        return False
    if not response.headers.get("content-type", "").lower().startswith("text/html"):
        return False
    if request.url.path.startswith(PRIVATE_ROUTE_PREFIXES):
        return False
    agent = request.headers.get("user-agent", "").lower()
    if not agent or any(marker in agent for marker in _ROBOT_MARKERS):
        return False
    return True


def _cookie_preferences(request: Request) -> dict:
    """Read the stored cookie choice, and whether it still applies.

    The choice is stored as "<choice>:<policy version>". A choice made
    against a superseded policy version is treated as undecided, which is
    what re-opens the banner - previously the cookie recorded only the
    choice, so bumping WARRIORIQ_POLICY_VERSION re-prompted nobody.

    A cookie with no version was written before this and is honoured as it
    stands. Those visitors did answer the question; re-asking all of them
    once to backfill a version would be a worse reading of "re-prompt when
    the policy changes" than simply recording it from here on.
    """
    raw = request.cookies.get(COOKIE_PREFERENCES_COOKIE, "")
    raw, _, version = raw.partition(":")
    if version and version != SETTINGS.policy_version:
        return {"decided": False, "analytics": False}
    if raw == "all":
        return {"decided": True, "analytics": True}
    if raw == "custom-analytics":
        return {"decided": True, "analytics": True}
    # "custom-marketing" is a value this no longer writes: marketing storage was
    # offered as a choice while nothing acted on it, so the control is gone.
    # Cookies already carrying it are still read rather than treated as
    # undecided, because those visitors did answer - they declined analytics,
    # and re-asking them would be re-asking a question they settled.
    if raw == "custom-marketing":
        return {"decided": True, "analytics": False}
    if raw == "essential":
        return {"decided": True, "analytics": False}
    return {"decided": False, "analytics": False}


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


def _full_progress_report(path: Path) -> dict | None:
    """The fields Progress needs from a full report, identity gate re-applied.

    Cached by modification time, so each report is parsed once per process.
    """
    try:
        modified = path.stat().st_mtime_ns
        cached = _progress_report_cache.get(str(path))
        if cached and cached[0] == modified:
            return cached[1]
        full_report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    report = refresh_identity_integrity({
        key: full_report.get(key, {})
        for key in ("video", "setup", "integrity", "metrics", "coaching",
                    "training_plan", "tracking")
    })
    _progress_report_cache[str(path)] = (modified, report)
    return report


def _reports_for_profile(profile_id: int) -> list[dict]:
    records = []
    fights = list_fights(profile_id)
    newest_legacy_ids = {fight["job_id"] for fight in fights[:2]}
    for fight in reversed(fights):
        base = {"job_id": fight["job_id"], "created_at": fight["created_at"],
                "fighter_id": fight.get("fighter_id"), "fighter_name": fight.get("fighter_name")}
        compact = (fight.get("summary") or {}).get("progress_report")
        if isinstance(compact, dict):
            # The snapshot's integrity was stored at analysis time, before the
            # "fighters look too alike" check counted - so a fight whose report
            # page says "not safe to use" was still charted here. The gate is
            # re-applied: from the snapshot's own tracking fields when it has
            # them, from the full report (read once, cached) when it predates
            # them.
            if compact.get("tracking"):
                report = refresh_identity_integrity(deepcopy(compact))
            else:
                full = _full_progress_report(Path(fight["report_path"]))
                report = compact if full is None else {
                    **compact, "integrity": full.get("integrity", compact.get("integrity", {})),
                    "coaching": full.get("coaching", compact.get("coaching", {})),
                    "training_plan": full.get("training_plan", compact.get("training_plan", {})),
                }
            records.append({**base, "report": report})
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
            records.append({**base, "report": minimal})
            continue
        # The identity gate is re-applied, as the report page does, so a fight
        # the report now withholds is not charted here.
        report = _full_progress_report(Path(fight["report_path"]))
        if report is None:
            continue
        records.append({**base, "report": report})
    return records


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _analysed_span_summary(report: dict) -> dict | None:
    """Which part of the video the report describes, in words.

    Silent truncation was the bug: a frame picked at 0:30 of a 1:56 video
    produced a report of 1:26 that called itself the fight. Every report now
    says what it covers. Reports made before the analysed span was recorded
    still started at their selection frame, and say so from setup.start_seconds.
    """
    span = (report.get("video") or {}).get("analysed_span")
    if span:
        start = float(span.get("start_seconds") or 0.0)
        end = float(span.get("end_seconds") or 0.0)
        duration = float(span.get("video_duration_seconds") or end)
        reason = span.get("excluded_reason_text")
    else:
        setup = report.get("setup") or {}
        start = float(setup.get("start_seconds") or 0.0)
        rounds = report.get("rounds") or []
        end = max([float(r.get("end_seconds") or 0.0) for r in rounds] or [start])
        duration = float((report.get("performance") or {}).get("segment_duration_seconds") or 0.0) + start
        duration = max(duration, end)
        reason = "older reports started at the frame chosen for fighter selection"
    if duration <= 0 or end <= start:
        return None
    whole = start < 1.0 and duration - end < 1.0
    label = (f"Whole video · {_clock(duration)}" if whole
             else f"{_clock(start)}–{_clock(end)} of {_clock(duration)}")
    note = None
    if start >= 1.0:
        note = (f"The first {_clock(start)} of the video is not in this report"
                + (f": {reason}." if reason else ".")
                + " Every number below describes " + f"{_clock(start)}–{_clock(end)} only.")
    elif duration - end >= 1.0:
        note = f"The last {_clock(duration - end)} of the video is not in this report."
    return {"label": label, "note": note, "whole": whole,
            "start_seconds": start, "end_seconds": end, "duration_seconds": duration}


def _numbers_state(report: dict) -> dict:
    """Whether the numbers on a report may be shown, and as what.

    "ok": shown as the selected fighter's own. "unverified": the identity check
    failed, so the numbers may include other people - shown greyed out, never
    as "you", and nothing is shared or posted from them. "no_fight": too little
    of the video was usable fight footage to measure anything, so no numbers
    are shown at all, and the report says why.
    """
    footage = (report.get("video") or {}).get("fight_footage") or {}
    integrity = report.get("integrity") or {}
    if footage and (integrity.get("fight_footage_sufficient") is False or footage.get("sufficient") is False):
        fight = float(footage.get("fight_seconds") or 0.0)
        excluded = float(footage.get("excluded_seconds") or 0.0)
        why = footage.get("main_exclusion_text")
        if excluded >= 1.0 and why:
            message = (f"Only {_clock(fight)} of this video could be used as fight footage; the other "
                       f"{_clock(excluded)} was left out because {why}. That is too little to measure, "
                       "so this report shows no numbers.")
        else:
            message = (f"Only {_clock(fight)} of footage was analysed, which is too little to measure, "
                       "so this report shows no numbers.")
        return {"state": "no_fight", "message": message,
                "share_reason": "There are no numbers in this report, so there is nothing to put on a story card."}
    if not integrity.get("identity_evidence_trusted", True):
        return {"state": "unverified", "message": None,
                "share_reason": ("WarriorIQ could not confirm who was who in this fight, so the numbers on "
                                 "this page are unverified and may include other people. They cannot go on "
                                 "a story card; a coach link shows your coach the report without them.")}
    return {"state": "ok", "message": None, "share_reason": None}


def _fight_footage_summary(report: dict) -> dict | None:
    """"Fight footage analysed: X of Y", and what was left out and why."""
    footage = (report.get("video") or {}).get("fight_footage")
    if not footage:
        return None
    span = (report.get("video") or {}).get("analysed_span") or {}
    duration = float(span.get("video_duration_seconds") or footage.get("analysed_seconds") or 0.0)
    fight = float(footage.get("fight_seconds") or 0.0)
    excluded = float(footage.get("excluded_seconds") or 0.0)
    note = None
    if excluded >= 1.0 and footage.get("main_exclusion_text"):
        note = (f"{_clock(excluded)} of the analysed footage was left out of every number because "
                f"{footage['main_exclusion_text']}.")
    return {"label": f"{_clock(fight)} of {_clock(duration)}", "note": note}


def _moment_fighters(report: dict) -> list[str]:
    """Whose coaching moments the replay shows: the focus fighter, or both."""
    video = report.get("video") or {}
    focus = video.get("focus_fighter") or video.get("analysis_target") or "BOTH"
    return [focus] if focus in {"A", "B"} else ["A", "B"]


def _visual_focus(report: dict) -> str:
    """Whose round the visual sections are about.

    The same choice the vitals strip at the top of the page makes, so the two
    cannot disagree about which fighter "you" means.
    """
    video = report.get("video") or {}
    focus = str(video.get("focus_fighter") or video.get("analysis_target") or "A")
    return focus if focus in ("A", "B") else "A"


def _analysis_quality_summary(report: dict) -> dict:
    """Summarize evidence quality without pretending coverage is accuracy."""
    video = report.get("video", {})
    focus = video.get("focus_fighter") or video.get("analysis_target", "BOTH")
    if focus not in {"A", "B"}:
        focus = "A"
    metrics = report.get("metrics", {})
    integrity = report.get("integrity", {})
    # The same verdict the score, the numbers and the coaching read
    # (core.report.identity_verdict), so this box cannot call the evidence
    # good or the identity stable beside a section that says otherwise.
    verdict = identity_verdict(report)
    coverage = verdict["coverage"]
    pose = max(0.0, min(1.0, float(metrics.get(focus, {}).get("pose_coverage", 0.0) or 0.0)))
    stable = verdict["trusted"]
    if not stable:
        label, tone = "Needs another fighter selection", "bad"
    elif not verdict["followed_enough_to_score"]:
        # The score section says "we lost sight of a fighter too often".
        label, tone = "Partial observation evidence", "review"
    elif pose >= .80:
        label, tone = "Strong observation evidence", "strong"
    elif pose >= .55:
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


def _save_upload_limited(upload: UploadFile, destination: Path, limit: int) -> str:
    free_before = shutil.disk_usage(destination.parent).free
    reserve = int(SETTINGS.minimum_free_storage_gb * 1024**3)
    if free_before <= reserve:
        raise HTTPException(507, "Fight uploads are temporarily paused while storage capacity is restored.")
    total = 0
    digest = hashlib.sha256()
    try:
        with destination.open("wb") as handle:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, "The uploaded file exceeds the maximum allowed size.")
                if free_before - total <= reserve or shutil.disk_usage(destination.parent).free - len(chunk) <= reserve:
                    raise HTTPException(507, "This upload would exceed WarriorIQ's private storage safety reserve.")
                handle.write(chunk)
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        destination.unlink(missing_ok=True)
        raise


async def _admit_fight_upload(request: Request, call_next):
    account = _account(request)
    if not account:
        return JSONResponse({"detail": "Create a free account or sign in to analyse a fight."}, status_code=401)
    job_id = uuid.uuid4().hex[:12]
    account_id = int(account["id"])
    accepted = False
    reserved = False
    try:
        _enforce_rate_limit(request, "fight-upload", 12, 600)
        await run_in_threadpool(reserve_upload_storage, account_id, job_id, min(MAX_FIGHT_BYTES, SETTINGS.max_upload_bytes))
        reserved = await run_in_threadpool(reserve_analysis, account_id, job_id)
        if not reserved:
            return JSONResponse({"detail": "Your analysis allowance is used for this period. It will reset automatically."}, status_code=429)
        request.state.upload_job_id = job_id
        response = await call_next(request)
        accepted = response.status_code in {201, 303}
        return response
    except UploadCapacityError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    finally:
        # Nothing in here is awaited, deliberately. This is the finally of an
        # async function, so every await is a point where a cancelled request -
        # a phone that hung up mid-transfer, a shutdown - can skip the rest of
        # the cleanup. A storage lease that outlives its request is not a leaked
        # row: with max_pending_uploads at 2, two abandoned uploads lock the
        # account out of uploading until the lease expires half an hour later.
        # Both releases are single indexed deletes, so they cost less than the
        # await they replace. Orphaned files are the one thing left best-effort,
        # and cleanup_abandoned_processing_files already sweeps those.
        if reserved and not accepted:
            release_analysis(account_id, job_id)
            # This ID was generated for this request; never touch a prior fight.
            for path in UPLOADS.glob(f"{job_id}*"):
                if path.is_file() and path.resolve().parent == UPLOADS.resolve():
                    path.unlink(missing_ok=True)
            job_dir = OUTPUTS / job_id
            if job_dir.exists() and job_dir.resolve().parent == OUTPUTS.resolve():
                shutil.rmtree(job_dir)
            delete_job(job_id)
            delete_legal_acceptances_for_resource(job_id)
        release_upload_storage(job_id)


# When this process began serving, for the uptime reported by /health.
PROCESS_STARTED_AT = time.time()


def _resident_megabytes() -> float | None:
    """This process's resident memory, or None where it cannot be read.

    Never raises and never becomes a dependency: psutil is not required by this
    project, and a health probe that fails because a diagnostic is unavailable
    would be worse than one that simply says less.
    """
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:                                               # noqa: BLE001
        pass
    try:
        # Linux, where the host actually runs. /proc is read directly so the
        # answer does not depend on a package being installed there.
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
    except Exception:                                               # noqa: BLE001
        return None


# Requests that never render a page, read an account or need the job list:
# assets, probes and the two crawler files. They skip the per-visitor context
# entirely - a stylesheet has no use for the visitor's session or analysis
# chip, and paying for both on every asset multiplied the cost of a page view
# by the number of files on it.
_HEALTH_PATHS = frozenset({"/health", "/healthz"})
_LIGHTWEIGHT_PATHS = _HEALTH_PATHS | {"/favicon.ico", "/robots.txt", "/sitemap.xml"}
_LIGHTWEIGHT_PREFIXES = ("/static/", "/assets/")


def _is_lightweight_request(path: str) -> bool:
    return path in _LIGHTWEIGHT_PATHS or path.startswith(_LIGHTWEIGHT_PREFIXES)


def _wants_json(request: Request) -> bool:
    return request.url.path.startswith(("/api/", "/stripe/")) or "application/json" in request.headers.get("accept", "")


def _rate_limited_response(request: Request, wait_seconds: int):
    """The 429 for the site-wide limit, built without touching the database.

    A standalone page rather than error.html: the limiter answers before the
    session is read, precisely so a flood costs as little as possible, and the
    full layout needs that session. It is still a page a person can read, with
    the wait spelled out, and the connection is always answered.
    """
    detail = _rate_limit_detail(wait_seconds)
    headers = {"Retry-After": str(max(1, int(wait_seconds))), "Cache-Control": "no-store"}
    if _wants_json(request):
        return JSONResponse({"detail": detail}, status_code=429, headers=headers)
    return templates.TemplateResponse(
        request=request, name="rate_limited.html",
        context={"request": request, "detail": detail, "retry_after": max(1, int(wait_seconds))},
        status_code=429, headers=headers,
    )


def _load_viewer_state(request: Request) -> None:
    """Who is asking and what they have running: the per-page context.

    Reads the session from the database and the job listing from disk, so it
    runs on a worker thread rather than on the event loop. Called inline from
    the async middleware, it used to stall every other request this process
    was serving for as long as one visitor's lookups took.
    """
    request.state.account = resolve_session(request.cookies.get(SESSION_COOKIE))
    request.state.analysis_navigation = _analysis_navigation_state(
        _owner_key(request),
        request.cookies.get(ACTIVE_ANALYSIS_COOKIE),
        request.cookies.get(LAST_COMPLETED_ANALYSIS_COOKIE),
    )
    request.state.active_analysis = request.state.analysis_navigation["display"]
    chosen_sport = (request.cookies.get(ACTIVE_SPORT_COOKIE) or "").strip().lower()
    request.state.active_sport = sport_identity(chosen_sport) if chosen_sport in SPORTS else None
    # Where naming a sport means nothing. The chip is a switcher for the
    # analysis flow; on the price list, the workspace overview and a coach's
    # squad it is a control that changes nothing on the page it sits on, which
    # invites a click that navigates away from what the reader came for.
    request.state.sport_switch_hidden = request.url.path.startswith(
        SPORT_IRRELEVANT_PREFIXES)
    # The switcher is a real menu now, so it needs something to list. It
    # looked like a dropdown - bordered pill, chevron - and was a plain link
    # to /analyze, so pressing it left whatever the reader was in the middle
    # of to show them a page of five sports.
    request.state.sports = [sport_identity(key) for key in SPORTS]
    request.state.launch = launch_readiness()
    request.state.legal_is_draft = SETTINGS.legal_is_draft
    request.state.minimum_account_age = SETTINGS.minimum_account_age
    request.state.oauth_providers = SOCIAL_AUTH.provider_buttons
    # Sign-in-only providers. /signup never offers these; /login shows a quiet
    # recovery line so an account created through one is not stranded.
    request.state.legacy_oauth_providers = SOCIAL_AUTH.legacy_provider_buttons
    request.state.cookie_preferences = _cookie_preferences(request)
    request.state.analytics_measurement_id = SETTINGS.analytics_measurement_id
    request.state.site_verification_token = SETTINGS.site_verification_token
    request.state.gtm_container_id = SETTINGS.gtm_container_id
    request.state.external_ai_available = bool(os.getenv("OPENAI_API_KEY", "").strip())
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


_maintenance_lock = threading.Lock()


def _run_periodic_maintenance(now: float | None = None) -> None:
    """Expire guest jobs, retained videos and abandoned processing files.

    Used to run inline in the request middleware, on the event loop, in
    whichever request happened to arrive after the timer ran out - deleting
    directories while that visitor, and everyone queued behind them, waited.
    It now runs on a background thread (see _schedule_maintenance); calling it
    directly is still the way to run it synchronously, as the tests do.
    """
    global _last_guest_cleanup, _last_saved_video_cleanup
    now = time.monotonic() if now is None else now
    if now - _last_guest_cleanup > 600:
        jobs_before_cleanup = dict(list_jobs())
        cutoff = time.time() - SETTINGS.failed_upload_retention_hours * 3600
        protected = {
            job_id for job_id, job in jobs_before_cleanup.items()
            if job.get("status") in {"queued", "running"}
            or (job.get("status") == "complete" and job.get("history_saved") is False and job.get("persist_result"))
            or (job.get("status") == "selecting" and float(job.get("updated_at_epoch", 0)) > cutoff)
        }
        for job_id in cleanup_expired_guest_jobs(protected):
            delete_legal_acceptances_for_resource(job_id)
            delete_job(job_id)
        # Chunked uploads whose page went away: give back their lease, their
        # reserved analysis and their partial file without waiting for the
        # same person to try again.
        release_idle_chunked_sessions()
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
        jobs_before_cleanup = dict(list_jobs())
        cutoff = time.time() - SETTINGS.failed_upload_retention_hours * 3600
        protected = {
            job_id for job_id, job in jobs_before_cleanup.items()
            if job.get("status") in {"queued", "running"}
            or (job.get("status") == "complete" and job.get("history_saved") is False and job.get("persist_result"))
            or (job.get("status") == "selecting" and float(job.get("updated_at_epoch", 0)) > cutoff)
        }
        saved = {item["job_id"] for item in list_all_fight_storage()}
        for abandoned_job_id in cleanup_abandoned_processing_files(
            protected, saved, older_than_hours=SETTINGS.failed_upload_retention_hours,
        ):
            abandoned = jobs_before_cleanup.get(abandoned_job_id, {})
            if abandoned.get("usage_reserved") and abandoned.get("account_id"):
                release_analysis(int(abandoned["account_id"]), abandoned_job_id)
            delete_legal_acceptances_for_resource(abandoned_job_id)
            delete_job(abandoned_job_id)
            record_security_event(
                "abandoned_processing_files_deleted", resource_type="fight", resource_id=abandoned_job_id,
            )
        _last_saved_video_cleanup = now


def _schedule_maintenance() -> None:
    """Start the periodic sweep in the background when one is due.

    Single-flight: if a sweep is already running, a request never starts a
    second one and never waits for the first.
    """
    now = time.monotonic()
    if now - _last_guest_cleanup <= 600 and now - _last_saved_video_cleanup <= 3600:
        return
    if not _maintenance_lock.acquire(blocking=False):
        return

    def sweep() -> None:
        try:
            _run_periodic_maintenance()
        except Exception:                                           # noqa: BLE001
            LOGGER.exception("periodic_maintenance_failed")
        finally:
            _maintenance_lock.release()

    try:
        threading.Thread(target=sweep, name="warrioriq-maintenance", daemon=True).start()
    except Exception:                                               # noqa: BLE001
        _maintenance_lock.release()
        raise


def _apply_response_headers(request: Request, response) -> None:
    """Security and caching headers every response carries, page or asset."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
    if request.url.path.startswith(PRIVATE_ROUTE_PREFIXES):
        response.headers.setdefault("X-Robots-Tag", "noindex, nofollow")
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
    # A per-response nonce rather than 'unsafe-inline': an injected <script>
    # cannot know the value, so it does not run. Inline on*= attributes cannot
    # carry a nonce, which is why the templates have none (see base.html).
    script_src = f"'self' 'nonce-{request.state.csp_nonce}'"
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
    # Sign-in forms redirect to the provider, and form-action is enforced across
    # that redirect: with 'self' alone every social button is blocked by the
    # browser before it leaves the page. Only the pages carrying those forms
    # need the provider origins, and only /login carries the legacy GitHub
    # recovery line - so github.com is no longer allowed on every page of a
    # site that offers no GitHub sign-up.
    form_action = " ".join(["'self'", *SOCIAL_AUTH.form_action_origins_for(request.url.path)])
    # blob: in media-src is the fight the visitor just chose, played from their
    # own device while it uploads. A blob: URL names something this page itself
    # created from a file the person picked - it cannot address anything remote
    # - so it widens nothing an attacker can reach, and without it the browser
    # refuses to play the file its owner is sitting in front of.
    response.headers.setdefault(
        "Content-Security-Policy",
        f"default-src 'self' data:; script-src {script_src}; style-src 'self' 'unsafe-inline'; "
        f"img-src {img_src}; media-src 'self' blob:; connect-src {connect_src}; frame-src {frame_src}; "
        f"frame-ancestors 'none'; form-action {form_action}",
    )
    if request.url.path.startswith(("/result/", "/replay/", "/media/", "/api/", "/profile", "/history", "/dashboard", "/coach", "/camp", "/s/", "/f/")):
        # setdefault, so a route that has already chosen keeps its value.
        # /media/ does exactly that: a fight video is private but not
        # volatile, and re-sending 101 MB on every seek is not privacy.
        response.headers.setdefault("Cache-Control", "no-store")
    elif request.url.path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "public, max-age=604800")
    if _request_is_secure(request):
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")


@app.middleware("http")
async def viewer_context(request: Request, call_next):
    rejected_host = next(
        (
            candidate for candidate in (
                request.headers.get("host", ""),
                _forwarded_header(request, "x-forwarded-host", "") if request.headers.get("x-forwarded-host") else None,
            )
            if candidate is not None and not _trusted_request_host(candidate)
        ),
        None,
    )
    if rejected_host is not None:
        # Name the host that was refused. Without this a deployment reached
        # under a hostname nobody listed in WARRIORIQ_ALLOWED_HOSTS is a blank
        # 400 with nothing to act on. Logged rather than recorded as a security
        # event, so an unauthenticated request cannot drive a database write.
        LOGGER.warning("untrusted_request_host host=%r path=%s", rejected_host[:120], request.url.path)
        return JSONResponse({"detail": "Unrecognized website address."}, status_code=400)
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
    if request.url.path not in _HEALTH_PATHS and SETTINGS.public_base_url.startswith("https://") and forwarded_scheme != "https":
        target = f"{SETTINGS.public_base_url}{request.url.path}"
        if request.url.query:
            target += f"?{request.url.query}"
        return RedirectResponse(target, status_code=308)
    # One unpredictable value per response, carried by every <script> the
    # templates emit, so the policy can name it instead of 'unsafe-inline'.
    request.state.csp_nonce = secrets.token_urlsafe(18)
    if _is_lightweight_request(request.url.path):
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request_id)
        response.headers.setdefault(
            "Server-Timing", f"app;dur={(time.perf_counter() - request_started) * 1000.0:.1f}")
        _apply_response_headers(request, response)
        return response
    if SETTINGS.request_rate_limit_per_minute:
        client = _client_ip(request)
        wait = _take_rate_slot("site", client, SETTINGS.request_rate_limit_per_minute, 60)
        if wait is not None:
            LOGGER.warning("site_rate_limited path=%s retry_after=%s", request.url.path, wait)
            response = _rate_limited_response(request, wait)
            _apply_response_headers(request, response)
            return response
    guest_id = request.cookies.get(GUEST_COOKIE)
    new_guest = not guest_id or len(guest_id) < 24 or len(guest_id) > 96
    request.state.guest_id = session_token() if new_guest else guest_id
    # The CSRF token is per visitor, not per form, and is issued to signed-out
    # visitors too - the login and signup posts are exactly the ones that need
    # it, and neither has an account yet. A cookie of the wrong shape counts as
    # absent and is replaced, so a stale value cannot lock somebody out.
    existing_csrf = usable_csrf_token(request.cookies.get(CSRF_COOKIE))
    new_csrf = existing_csrf is None
    request.state.csrf_token = issue_csrf_token() if new_csrf else existing_csrf
    await run_in_threadpool(_load_viewer_state, request)
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
    _schedule_maintenance()
    if is_fight_upload(request.scope):
        response = await _admit_fight_upload(request, call_next)
    else:
        response = await call_next(request)
    duration_ms = (time.perf_counter() - request_started) * 1000.0
    response.headers.setdefault("X-Request-ID", request_id)
    response.headers.setdefault("Server-Timing", f"app;dur={duration_ms:.1f}")
    LOGGER.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    if _is_counted_page_view(request, response):
        try:
            record_page_view(_counted_path(request.url.path))
        except Exception:  # pragma: no cover - counting must never cost a page
            # A visitor came for the page, not for the statistic. If the write
            # fails the page still has to be served, so this swallows rather
            # than raises, and says so in the log instead.
            LOGGER.warning("page_view_not_counted path=%s", request.url.path)
    _apply_response_headers(request, response)
    if new_guest:
        response.set_cookie(
            GUEST_COOKIE, request.state.guest_id, max_age=60 * 60 * 24,
            httponly=True, samesite="lax", secure=_request_is_secure(request),
        )
    if new_csrf:
        # httponly on purpose. The usual double-submit recipe makes this
        # readable so scripts can copy it, which hands the token to any
        # injected script as well; the page carries the value instead, from
        # server state. See core/csrf.py.
        response.set_cookie(
            CSRF_COOKIE, request.state.csrf_token, max_age=60 * 60 * 24 * 30,
            httponly=True, samesite="lax", secure=_request_is_secure(request),
        )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_error_page(request: Request, exc: StarletteHTTPException):
    if _wants_json(request):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
    if not hasattr(request.state, "account"):
        # An asset or probe request: the page layout needs the visitor context
        # those requests deliberately skip, so the refusal is plain text.
        return PlainTextResponse(str(exc.detail or ""), status_code=exc.status_code, headers=exc.headers)
    title = {
        400: "That request needs attention",
        403: "This area is private",
        404: "That page left the ring",
        410: "This link has expired",
        413: "That file is too large",
        # Every 429 that reaches this page is a request-rate limit; the
        # analysis allowance answers as JSON on the upload routes instead.
        429: "Slow down for a moment",
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
    if _wants_json(request):
        return JSONResponse({"detail": detail}, status_code=500)
    if not hasattr(request.state, "account"):
        return PlainTextResponse(detail, status_code=500)
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
        # Rebuilding an event without its evidence quietly changes what it
        # means. Scoring reads foot_lift_torsos from here to tell a kick from a
        # step, so dropping it made every kick in a saved report unscoreable -
        # a different answer from the same fight depending on whether it came
        # from the analysis or from disk.
        evidence=dict(event.get("evidence") or {}),
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


def _counted_strikes(report: dict, families: tuple[str, ...]) -> list[dict]:
    """Every strike behind the counts on the page, so a fighter can check them.

    Built from the same list and the same duplicate rule as the statistics
    block (fight_stats._deduplicate over event_feed), and filtered to the same
    families, so the list and the number above it cannot disagree.
    """
    feed = report.get("event_feed") or []
    rows = []
    for item in _deduplicate_strikes(feed):
        family = item.get("family")
        if family not in families or item.get("fighter") not in {"A", "B"}:
            continue
        try:
            seconds = float(item.get("time_seconds", item.get("peak_time")))
        except (TypeError, ValueError):
            continue
        rows.append({"fighter": item["fighter"], "family": family, "seconds": seconds,
                     "round": item.get("round_number"),
                     "clock": "%d:%04.1f" % (int(seconds // 60), seconds % 60)})
    return sorted(rows, key=lambda row: row["seconds"])


def _with_checks(job_id: str, rows: list[dict]) -> list[dict]:
    """The owner's earlier one-tap answers, so the page shows what they said."""
    checks = _strike_checks(job_id)
    for row in rows:
        row["check"] = checks.get((row["fighter"], round(row["seconds"], 3)))
    return rows


def _wrong_sport(report: dict, job: dict) -> dict | None:
    """An MMA video analysed as a striking sport, told plainly (core.ground.looks_like_grappling).

    Worked out from the fight itself, so it costs nothing and needs no AI
    service. It can tell ground fighting from stand-up, and nothing finer:
    kickboxing against taekwondo against boxing is not separable this way.
    """
    sport = (report.get("scorecard") or {}).get("sport") or _job_sport(job)
    if sport not in STRIKING_SPORTS:
        return None
    grappling = looks_like_grappling(report.get("went_down"))
    if grappling is None:
        return None
    return {**grappling, "chosen": RULESET_SPORTS.get(sport, sport)}


def _went_down(job_id: str, report: dict) -> list[dict]:
    """The moments someone went down (core.ground), with the owner's answers.

    Not attributed to a fighter, so shown whether or not identity held: it
    only says that someone was down at that second.
    """
    answers = _down_checks(job_id)
    rows = []
    for moment in (report.get("went_down") or {}).get("moments") or []:
        try:
            seconds = float(moment["seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"seconds": seconds, "round": moment.get("round"),
                     "down_seconds": int(moment.get("down_seconds") or 0),
                     "clock": "%d:%02d" % (int(seconds // 60), int(seconds % 60)),
                     "check": answers.get(round(seconds, 3))})
    return rows


def _estimate_score_withheld_for_punches(report: dict) -> None:
    """Give a stored report the estimated score it would get today.

    Reports analysed while punch counts were withheld stored the status
    "punch_counting_unavailable" and no score. That status is only reached
    after the identity and coverage checks passed, so rebuilding the
    scorecard from the stored events changes nothing but that one decision.
    Rebuilt on read, never written back.
    """
    if not STRIKE_COUNTS_PUBLISHED or STRIKE_COUNTS_PRECISION_VALIDATED:
        return
    scorecard = report.get("scorecard") or {}
    if scorecard.get("status") != "punch_counting_unavailable":
        return
    report["scorecard"] = build_preliminary_scorecard(
        [_strike_from_dict(item) for item in report.get("events", [])],
        scorecard.get("ruleset") or report.get("setup", {}).get("ruleset", "K1"),
        [int(item["number"]) for item in report.get("rounds", []) if item.get("selected", True)],
        report.get("tracking", {}),
        report.get("video", {}).get("analysis_target", "BOTH"),
    )


def _keep_plausibility_verdict(statistics: dict, previous: dict) -> dict:
    """A rebuilt statistics block keeps an earlier "impossible counts" verdict.

    Rebuilds here have no analysed duration to judge a rate against, so they
    cannot repeat the check (core/count_plausibility.py) - and must not undo it.
    """
    earlier = (previous or {}).get("plausibility") or {}
    if earlier.get("implausible"):
        statistics["plausibility"] = earlier
        statistics["attempt_counts_available"] = False
        statistics["action_labels_available"] = False
    return statistics


def _withhold_score_while_counts_are_off(report: dict) -> None:
    """No estimated score while strike counts are switched off.

    The estimated score is built from the same automatic strike counts the
    flag withholds (core.report.STRIKE_COUNTS_PUBLISHED), so a report stored
    while counts were on loses its score on read too. Never written back.
    """
    implausible = counts_implausible(report)
    if (STRIKE_COUNTS_PUBLISHED or STRIKE_COUNTS_PRECISION_VALIDATED) and not implausible:
        return
    scorecard = report.get("scorecard") or {}
    if scorecard.get("available") or scorecard.get("status") == "punch_counting_unavailable":
        # A score built from counts no real fight could produce is withheld
        # too (core/count_plausibility.py).
        report["scorecard"] = {
            **scorecard, "available": False, "totals": {"A": None, "B": None}, "rounds": [],
            "winner_estimate": None,
            "status": "strike_counts_implausible" if implausible else "strike_counts_off",
        }


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
    report["sport_coaching"] = {"A": None, "B": None}
    report["training_plan"] = {"A": [], "B": []}
    report["training_progression"] = {"A": [], "B": []}
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
    automated_trusted = bool(trust["automated_evidence_trusted"]) and all(
        accepted_model_event(event) for event in report.get("events", [])
    )
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
                if not accepted_model_event(event):
                    continue
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
    if trust["automated_evidence_trusted"]:
        reason = "Some action candidates lacked an accepted model decision; action facts were withheld."
    integrity["action_metrics_trusted"] = False
    _withhold_unverified_action_report(report, reason)
    feed = report.get("event_feed") or []
    previous_stats = report.get("statistics") or {}
    if feed or previous_stats:
        if previous_stats.get("action_labels_available") or any(
            item.get("verification") != "observed" for item in feed
        ):
            statistics = summarize_fight_events(feed, None, 0, False)
            for fighter in ("A", "B"):
                previous = (previous_stats.get("fighters") or {}).get(fighter) or {}
                statistics["fighters"][fighter]["observation_coverage"] = previous.get("observation_coverage", 0.0)
            report["statistics"] = _keep_plausibility_verdict(statistics, previous_stats)
        for item in feed:
            item.update({"verification": "observed", "technique": None, "limb": None,
                         "target": None, "outcome": "unclassified"})
    confirmed_events = [_strike_from_dict(item) for item in displayed.values() if item.get("human_verified")]
    if scorecard_review_complete:
        _apply_human_scorecard(report, confirmed_events)
    if not full_review_complete:
        return

    confirmed_feed = [{
        "fighter": event.fighter, "family": event.family, "limb": event.limb,
        "technique": event.technique, "target": event.target,
        "outcome": normalize_outcome(event.outcome), "round_number": event.round_number,
        "time_seconds": event.peak_time, "confidence": 1.0, "verification": "verified",
    } for event in confirmed_events]
    report["event_feed"] = confirmed_feed
    previous_stats = report.get("statistics") or {}
    statistics = summarize_fight_events(confirmed_feed, None, 0, True)
    for fighter in ("A", "B"):
        previous = (previous_stats.get("fighters") or {}).get(fighter) or {}
        statistics["fighters"][fighter]["observation_coverage"] = previous.get("observation_coverage", 0.0)
    report["statistics"] = _keep_plausibility_verdict(statistics, previous_stats)

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
        fighter_b_box=list(fighter_b_box or []),
        original_name=job.get("original_name"),
        analysis_target="A" if job.get("solo") else "BOTH",
        solo=bool(job.get("solo")),
        focus_fighter=focus_fighter,
        fight_type=job["fight_type"],
        ruleset=job["ruleset"],
        # The whole video unless a start was explicitly requested; the boxes
        # belong to the selection frame. A job stored before the two were
        # separated has only start_seconds, which was the selection frame.
        start_seconds=float(job.get("requested_start_seconds", 0.0) or 0.0),
        selection_seconds=float(job.get("selection_seconds", job.get("start_seconds", 0.0)) or 0.0),
        round_count=job["round_count"],
        round_duration_seconds=job["round_duration_seconds"],
        break_duration_seconds=job["break_duration_seconds"],
        selected_rounds=job.get("selected_rounds"),
        end_seconds=job.get("end_seconds"),
        job_id=job_id,
        profile_id=job.get("profile_id", 1),
        persist_result=bool(job.get("persist_result", False)),
        openai_identity_recovery=bool(job.get("openai_identity_recovery") and job.get("external_ai_opted_in")),
        fighter_id=job.get("fighter_id"),
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
        output_dir = analysis_run_directory(job_id, analysis_run_id)
        report = _analyze(replace(req, output_dir=str(output_dir), persist_result=False), cb)
        artifacts = {name: output_dir / name for name in ("tracking.jsonl", "events.json", "report.html") if (output_dir / name).is_file()}
        if not finalize_job_from_worker(job_id, worker_id, analysis_run_id, report, artifacts):
            LOGGER.warning("analysis_completion_discarded job_id=%s run_id=%s", job_id, analysis_run_id)
            return
        _save_remote_fight(job_id, get_job(job_id), report)
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


def _wake_expectation() -> str:
    """How long this fight will actually wait, from what the machine has done before.

    The analysis GPU sleeps between fights. A magic packet may or may not cross
    the visitor's router, so the honest promise is the scheduled drain - the one
    mechanism that cannot fail - narrowed to the real median once the machine
    has answered enough wakes to have a track record.
    """
    observed = wake_status()
    seconds = observed["median_seconds"] if observed["observations"] >= 3 else None
    if seconds is None:
        seconds = SETTINGS.wake_drain_interval_seconds
        basis = "at the latest"
    else:
        basis = "usually"
    if seconds < 90:
        window = f"{max(10, int(round(seconds / 10) * 10))} seconds"
    else:
        window = f"{max(1, int(round(seconds / 60)))} minutes"
    return f"starts {basis} within {window}"


def _deferred_analysis_message() -> str:
    return (
        "The analysis machine is asleep and is being woken now. Your fight is "
        f"saved and {_wake_expectation()} - you can close this page."
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

    update_job(job_id, {"wake_requested_at_epoch": time.time()})

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


def _looks_alike_message(similarity: float, kit: dict | None = None) -> str:
    """The selection-page warning, naming the actual cause.

    It used to say "look N% alike, where a bout we can read is usually nearer
    60%" for every cause - a number from a histogram that could not tell black
    from white - and on black-and-white footage that is simply what the film
    is, not something a better frame changes.
    """
    reason = kit_alike_reason(kit)
    if reason == "black_and_white":
        return ("This footage has no colour, so WarriorIQ cannot use kit to tell these two apart. "
                "It will still analyse the fight and follow each fighter by position and movement; "
                "the report will say if it ever could not tell them apart.")
    if reason == "small":
        return ("The fighters are small in this picture, so their kit is only a few pixels and "
                "cannot be compared. WarriorIQ will follow them by position and movement; filming "
                "closer gives it more to work with.")
    return (f"Their kit matches closely ({similarity:.0%} alike, comparing head, top and shorts). "
            "WarriorIQ will still analyse the fight and follow each fighter by position and movement, "
            "and the report will say if it ever could not tell them apart. If their kit differs "
            "anywhere - headgear, gloves, shorts - pick a frame where that difference is visible.")


def _analysis_started_response(request: Request, job_id: str, deferred: bool = False,
                              looks_alike: float | None = None,
                              on_official: dict | None = None,
                              kit: dict | None = None) -> JSONResponse:
    response = JSONResponse({
        "ok": True,
        "progress_url": f"/progress/{job_id}",
        "deferred": deferred,
        **({"notice": _deferred_analysis_message()} if deferred else {}),
        **({"fighters_look_alike": {
            "similarity": round(float(looks_alike), 3),
            "message": _looks_alike_message(float(looks_alike), kit),
        }} if looks_alike is not None else {}),
        **({"seed_looks_like_official": on_official} if on_official else {}),
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
    base = _public_base(request)
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


@app.post("/auth/{provider}/start", dependencies=[Depends(require_csrf)])
async def social_auth_start(
    request: Request,
    provider: str,
    mode: str = Form("login"),
    next_path: str = Form("/dashboard"),
    accept_terms: bool = Form(False),
    age_confirmed: bool = Form(False),
    accept_policies: bool = Form(False),
    marketing_consent: bool = Form(False),
    age_group: str = Form(""),
):
    _enforce_rate_limit(request, "social-auth-start", 30, 300)
    group = age_group.strip().lower() or ("adult" if age_confirmed else "")
    if mode == "signup" and group == "minor":
        # The guardian's details are asked on the email form, so that is the
        # way in for someone under the minimum age.
        return _auth_page(
            request, "signup",
            f"Under {SETTINGS.minimum_account_age}? Create your account with your email below, so WarriorIQ can "
            "ask your parent or guardian to approve it.",
            next_path,
        )
    age_confirmed = group == "adult"
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
    # Signing in is not the moment to collect consent: see login(). An account
    # whose accepted version is behind is asked once, after it is known.
    authorize_options = {"response_mode": "form_post"} if provider == "apple" else {}
    try:
        response = await client.authorize_redirect(
            request, _oauth_redirect_uri(request, provider), **authorize_options,
        )
    except Exception as exc:                            # noqa: BLE001
        # Starting an OIDC sign-in makes this server fetch the provider's
        # discovery document first. When that call cannot complete - a shared
        # host with outbound HTTPS blocked, a provider having a bad day - the
        # button used to spin for ever because nothing ever answered the POST.
        # Say so instead. The client timeout keeps this to a few seconds.
        LOGGER.error(
            "social_auth_start_failed provider=%s error=%s", provider, type(exc).__name__,
        )
        return _auth_page(
            request, mode,
            f"WarriorIQ could not reach {provider.title()} to start sign-in. "
            "Please try again, or use your email and password.",
            next_path,
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


def _link_verified_identity(identity) -> dict | None:
    """Let an existing account sign in with a provider for the first time.

    Someone who signed up with a password and later presses Continue with
    Google has no linked identity, so the callback refused them and pointed at
    Create account - which then refused too, because the email is already
    taken. Settings has no connect button either, so there was no way through
    at all: the advice to "sign in with its password first" led nowhere.

    Linking is allowed only on an address the provider has verified. Without
    that check, registering a victim's address at any provider would be enough
    to walk into their account, so an unverified email is treated as no email.
    """
    email = (identity.email or "").strip().lower()
    if not identity.email_verified or not email or not valid_email(email):
        return None
    account = get_account_by_email(email)
    if not account or account.get("account_status") != "active":
        return None
    link_oauth_identity(identity.provider, identity.subject, int(account["id"]), email)
    record_security_event(
        "oauth_identity_linked", account_id=int(account["id"]),
        metadata={"provider": identity.provider},
    )
    return get_account_for_oauth_identity(identity.provider, identity.subject)


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
        # "OAuthError" alone cannot be acted on: a wrong client secret, a reused
        # code and a mismatched redirect URI are one class and three different
        # fixes. authlib carries the provider's own error code and description,
        # neither of which contains a token or a secret, so both are recorded.
        LOGGER.warning(
            "social_auth_rejected provider=%s error=%s code=%s detail=%s",
            provider, type(exc).__name__,
            getattr(exc, "error", "") or "unknown",
            (getattr(exc, "description", "") or str(exc))[:200],
        )
        return _social_auth_error(request, intent, "The identity provider could not verify this sign-in.")
    except Exception as exc:
        LOGGER.warning("social_auth_unavailable provider=%s error=%s", provider, type(exc).__name__)
        return _social_auth_error(request, intent, "Secure sign-in is temporarily unavailable. Please try again.")

    account = get_account_for_oauth_identity(identity.provider, identity.subject)
    created = False
    if not account:
        account = _link_verified_identity(identity)
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
    base = _public_base(request)
    verify_url = f"{base}/verify-email/{token}"
    try:
        return send_transactional_email(
            account["email"], "Verify your WarriorIQ email",
            f"Verify your private WarriorIQ workspace within 24 hours:\n\n{verify_url}\n\nIf you did not create this account, ignore this message.",
        )
    except Exception:
        return False


# Accounts for people under the minimum age (SETTINGS.minimum_account_age).
#
# Sign-up used to be adults only. A fighter under 18 can now have an account,
# with a parent or guardian's approval: sign-up asks for their name and email,
# WarriorIQ emails them a link, and until they approve the account can look
# around but not upload a fight - footage is the personal data that matters
# here. The approval is a signed link (no new table): it names the account and
# the guardian address it was sent to, and expires.
GUARDIAN_PENDING = "pending_guardian"
GUARDIAN_APPROVED = "guardian_approved"
GUARDIAN_DECLINED = "guardian_declined"
GUARDIAN_LINK_DAYS = 14


def _guardian_signer():
    from itsdangerous import URLSafeTimedSerializer

    return URLSafeTimedSerializer(_session_secret(), salt="warrioriq-guardian-approval")


def _guardian_hold(account: dict | None) -> str | None:
    """Why this account may not upload yet, or None."""
    status = str((account or {}).get("guardian_approval_status") or "not_applicable")
    if status == GUARDIAN_PENDING:
        return ("Your parent or guardian has not approved your account yet. Fight videos can be uploaded "
                "once they open the link WarriorIQ emailed them. You can send it again from /guardian.")
    if status == GUARDIAN_DECLINED:
        return "Your parent or guardian did not approve this account, so it cannot upload fight videos."
    return None


def _guardian_request(account: dict) -> dict | None:
    """The guardian named at sign-up, from the consent record."""
    for record in list_legal_acceptances(profile_id=int(account["profile_id"])):
        if record.get("kind") == "guardian_consent" and record["metadata"].get("guardian_email"):
            return record["metadata"]
    return None


def _send_guardian_email(request: Request, account: dict, guardian_name: str, guardian_email: str) -> bool:
    token = _guardian_signer().dumps({"a": int(account["id"]), "g": token_digest(guardian_email.lower())})
    approve_url = f"{_public_base(request)}/guardian/approve/{token}"
    body = (
        f"Hello {guardian_name},\n\n"
        f"Someone signing up to WarriorIQ as {account['email']} said they are under "
        f"{SETTINGS.minimum_account_age} and named you as their parent or guardian.\n\n"
        "WarriorIQ analyses fight videos for combat-sports training. Until you approve, the account cannot "
        "upload any video. To read what it does with footage and approve or decline, open this link within "
        f"{GUARDIAN_LINK_DAYS} days:\n\n{approve_url}\n\n"
        "If you do not know this person, ignore this message and nothing will be uploaded."
    )
    try:
        return send_transactional_email(guardian_email, "Approve a WarriorIQ account", body)
    except Exception:                                                   # noqa: BLE001
        return False


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, next: str = "/dashboard"):
    if _account(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return _auth_page(request, "signup", next_path=next)


@app.post("/signup", dependencies=[Depends(require_csrf)])
def signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next_path: str = Form("/dashboard"),
    accept_terms: bool = Form(False),
    age_confirmed: bool = Form(False),
    marketing_consent: bool = Form(False),
    # "adult" or "minor". age_confirmed=true (the old checkbox) still means adult.
    age_group: str = Form(""),
    guardian_name: str = Form(""),
    guardian_email: str = Form(""),
):
    _enforce_rate_limit(request, "signup", 20, 300)
    group = age_group.strip().lower() or ("adult" if age_confirmed else "")
    if not accept_terms or group not in {"adult", "minor"}:
        return _auth_page(
            request, "signup",
            f"Say whether you are {SETTINGS.minimum_account_age} or older, and accept the Terms of Service and "
            "Privacy Policy.",
            next_path,
        )
    minor = group == "minor"
    guardian_name = " ".join(guardian_name.split())[:80]
    guardian_email = guardian_email.strip()
    if minor and (not guardian_name or not valid_email(guardian_email)
                  or normalize_email(guardian_email) == normalize_email(email)):
        return _auth_page(
            request, "signup",
            f"Under {SETTINGS.minimum_account_age}, a parent or guardian has to approve your account. Enter their "
            "name and their own email address - not yours.",
            next_path,
        )
    try:
        account = register(email, password)
    except ValueError as exc:
        return _auth_page(request, "signup", str(exc), next_path)
    record_account_signup_acceptance(
        int(account["id"]), terms_version=SETTINGS.policy_version,
        privacy_version=SETTINGS.policy_version, marketing_consent=bool(marketing_consent),
        guardian_approval_status=GUARDIAN_PENDING if minor else "not_applicable",
    )
    age_record = ("age_under_minimum_declared" if minor else "age_18_plus_confirmation", "accepted")
    for kind, status in (
        ("terms_acceptance", "accepted"),
        ("privacy_acknowledgement", "accepted"),
        age_record,
        ("marketing_consent", "accepted" if marketing_consent else "declined"),
    ):
        record_legal_acceptance(
            kind, SETTINGS.policy_version, profile_id=int(account["profile_id"]),
            metadata={"source": "signup", "enabled": bool(marketing_consent) if kind == "marketing_consent" else True},
            current_status=status,
        )
    record_security_event("account_created", account_id=int(account["id"]), metadata={"policy_version": SETTINGS.policy_version})
    if minor:
        record_legal_acceptance(
            "guardian_consent", SETTINGS.policy_version, profile_id=int(account["profile_id"]),
            metadata={"guardian_name": guardian_name, "guardian_email": guardian_email},
            current_status="requested",
        )
        delivered = _send_guardian_email(request, account, guardian_name, guardian_email)
        record_security_event(
            "guardian_approval_requested", account_id=int(account["id"]),
            metadata={"email_delivery": "sent" if delivered else "unavailable"},
        )
        next_path = "/guardian"
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


def _masked_email(address: str) -> str:
    local, _, domain = address.partition("@")
    return f"{local[:1]}{'*' * max(2, len(local) - 1)}@{domain}" if domain else address


@app.get("/guardian", response_class=HTMLResponse)
def guardian_status_page(request: Request, sent: str = ""):
    """Where an account under the minimum age waits for its guardian."""
    account = _account(request)
    if account is None:
        return RedirectResponse("/login?next=/guardian", status_code=303)
    status = str(account.get("guardian_approval_status") or "not_applicable")
    if status in {"not_applicable", GUARDIAN_APPROVED}:
        return RedirectResponse("/dashboard", status_code=303)
    named = _guardian_request(account) or {}
    return templates.TemplateResponse(
        request=request, name="guardian.html",
        context={"request": request, "mode": "pending", "status": status, "sent": sent == "1",
                 "guardian_email": _masked_email(str(named.get("guardian_email") or "")),
                 "minimum_age": SETTINGS.minimum_account_age},
    )


@app.post("/guardian/resend", dependencies=[Depends(require_csrf)])
def guardian_resend(request: Request):
    _enforce_rate_limit(request, "guardian-resend", 5, 3600)
    account = _account(request)
    if account is None:
        return RedirectResponse("/login?next=/guardian", status_code=303)
    named = _guardian_request(account)
    if str(account.get("guardian_approval_status")) == GUARDIAN_PENDING and named:
        delivered = _send_guardian_email(request, account, str(named.get("guardian_name") or ""),
                                         str(named["guardian_email"]))
        record_security_event("guardian_approval_requested", account_id=int(account["id"]),
                              metadata={"email_delivery": "sent" if delivered else "unavailable", "resend": True})
    return RedirectResponse("/guardian?sent=1", status_code=303)


def _guardian_link(token: str) -> tuple[dict | None, dict | None]:
    """(account, guardian request) a still-valid approval link names."""
    from itsdangerous import BadSignature

    try:
        payload = _guardian_signer().loads(token, max_age=GUARDIAN_LINK_DAYS * 86400)
        account = get_account(int(payload["a"]))
    except (BadSignature, KeyError, TypeError, ValueError):
        return None, None
    named = _guardian_request(account) if account else None
    # The link only works for the address it was sent to: a guardian named
    # later replaces the earlier one.
    if not named or token_digest(str(named["guardian_email"]).lower()) != payload.get("g"):
        return None, None
    return account, named


@app.get("/guardian/approve/{token}", response_class=HTMLResponse)
def guardian_approve_page(request: Request, token: str):
    account, named = _guardian_link(token)
    if account is None:
        raise HTTPException(404, "This approval link has expired or is not valid. Ask for a new one to be sent.")
    return templates.TemplateResponse(
        request=request, name="guardian.html",
        context={"request": request, "mode": "approve", "token": token, "child_email": account["email"],
                 "guardian_name": named.get("guardian_name") or "", "minimum_age": SETTINGS.minimum_account_age,
                 "status": str(account.get("guardian_approval_status") or ""),
                 "retention_days": SETTINGS.saved_video_retention_days},
    )


@app.post("/guardian/approve/{token}", response_class=HTMLResponse, dependencies=[Depends(require_csrf)])
def guardian_approve(request: Request, token: str, decision: str = Form(""),
                     guardian_confirmed: bool = Form(False)):
    _enforce_rate_limit(request, "guardian-approve", 20, 3600)
    account, named = _guardian_link(token)
    if account is None:
        raise HTTPException(404, "This approval link has expired or is not valid. Ask for a new one to be sent.")
    approve = decision == "approve"
    if approve and not guardian_confirmed:
        return templates.TemplateResponse(
            request=request, name="guardian.html", status_code=400,
            context={"request": request, "mode": "approve", "token": token, "child_email": account["email"],
                     "guardian_name": named.get("guardian_name") or "", "minimum_age": SETTINGS.minimum_account_age,
                     "status": str(account.get("guardian_approval_status") or ""),
                     "retention_days": SETTINGS.saved_video_retention_days,
                     "error": "Tick the box to confirm you are their parent or guardian."},
        )
    status = GUARDIAN_APPROVED if approve else GUARDIAN_DECLINED
    set_guardian_approval_status(int(account["id"]), status)
    record_legal_acceptance(
        "guardian_consent", SETTINGS.policy_version, profile_id=int(account["profile_id"]),
        metadata={"guardian_name": named.get("guardian_name"), "guardian_email": named.get("guardian_email"),
                  "decision": "approved" if approve else "declined"},
        current_status="accepted" if approve else "declined",
    )
    record_security_event("guardian_approval_" + ("granted" if approve else "declined"),
                          account_id=int(account["id"]))
    return templates.TemplateResponse(
        request=request, name="guardian.html",
        context={"request": request, "mode": "done", "approved": approve, "child_email": account["email"],
                 "minimum_age": SETTINGS.minimum_account_age},
    )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/dashboard"):
    if _account(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return _auth_page(request, "login", next_path=next)


@app.post("/login", dependencies=[Depends(require_csrf)])
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next_path: str = Form("/dashboard"),
):
    """Sign in.

    This asked for the Terms, Privacy Policy and Acceptable Use Policy to be
    accepted on every sign-in. Acceptance belongs at signup; re-accepting on
    each login records nothing new and is asked of everybody because at the
    form nobody has been identified yet. The account row carries the version
    it accepted, so the question is now asked after authentication, only of
    the accounts whose version is behind - see /policies.
    """
    _enforce_rate_limit(request, "login", 30, 300)
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
    destination = _safe_next(next_path)
    if policies_outdated(account):
        destination = f"/policies?next={quote(destination, safe='')}"
    response = RedirectResponse(destination, status_code=303)
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


@app.post("/verify-email", response_class=HTMLResponse, dependencies=[Depends(require_csrf)])
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


@app.post("/logout", dependencies=[Depends(require_csrf)])
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


@app.post("/forgot-password", response_class=HTMLResponse, dependencies=[Depends(require_csrf)])
def request_password_reset(request: Request, email: str = Form(...)):
    _enforce_rate_limit(request, "password-reset", 8, 900)
    account = get_account_by_email(email.strip().lower()) if valid_email(email) else None
    if account and account.get("account_status", "active") == "active":
        token = session_token()
        expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        save_password_reset_token(int(account["id"]), token_digest(token), expires)
        reset_url = f"{_public_base(request)}/reset-password/{token}"
        failure = None
        try:
            deliver_email(
                account["email"], "Reset your WarriorIQ password",
                f"Use this one-time link within 30 minutes:\n\n{reset_url}\n\nIf you did not request this, ignore this message.",
            )
        except EmailNotSent as reason:
            failure = str(reason)
            LOGGER.error("password_reset_email_not_sent account_id=%s reason=%s", account["id"], failure)
        record_security_event(
            "password_reset_requested", account_id=int(account["id"]),
            severity="warning" if failure else "info",
            metadata={"email_delivery": "sent" if failure is None else "failed", "reason": failure},
        )
    return templates.TemplateResponse(
        request=request, name="password_reset.html",
        context={
            "request": request, "token": "",
            "message": "If an eligible account exists, a time-limited reset link has been queued for its email provider.",
        },
    )


@app.get("/reset-password/{token}", response_class=HTMLResponse)
def reset_password_page(request: Request, token: str, error: str = ""):
    return templates.TemplateResponse(
        request=request, name="password_reset.html",
        context={"request": request, "token": token, "message": "", "error": error[:200]},
    )


@app.post("/reset-password/{token}", dependencies=[Depends(require_csrf)])
def reset_password(request: Request, token: str, password: str = Form(...)):
    # /forgot-password is limited, but consuming the token was not - so the
    # token itself could be guessed at without limit. It is 32 random bytes
    # and not realistically guessable, which is a reason to keep the limit
    # loose rather than a reason to have none.
    _enforce_rate_limit(request, "password-reset-complete", 15, 900)
    if not valid_password(password):
        # Back to the form with the reason, rather than a 400 page over it.
        # The token stays in the path, so the link is still usable - which a
        # 400 page made look untrue at the moment it mattered most.
        return RedirectResponse(
            "/reset-password/%s?%s" % (
                quote(token, safe=""),
                urlencode({"error": "Password must contain between 10 and 1,024 characters."})),
            status_code=303)
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


def _reported_strike_families(sport: str) -> dict:
    """The strike families a report for this sport will actually count.

    `sport_counted_families` is what the sport scores. The report publishes
    less: while STRIKE_COUNTS_PRECISION_VALIDATED is False only kicks are
    shown, because the punch count was measured overstated and the knee bucket
    was measured to contain punches (core/report.py observed_summary).
    """
    # Every sentence about counting comes from core.sport_policy, so the upload
    # page, live view, report, replay and share card cannot drift apart again.
    policy = counting_policy(sport, published=STRIKE_COUNTS_PUBLISHED,
                             validated=STRIKE_COUNTS_PRECISION_VALIDATED)
    return {
        "reported_families": _prose_list(policy.counted),
        "withheld_families": _prose_list(policy.withheld),
        "no_strike_counts": not policy.counted,
        "counts_are_estimates": policy.estimates,
        "estimate_note": policy.estimate_note,
        "counting_policy": policy.as_dict(),
    }


def _sport_coverage_badge(sport: str) -> dict:
    """The one-line coverage badge on a sport's chooser card.

    Boxing's card said "Full scoring coverage" while every boxing report
    withheld punches, the only thing boxing scores. The badge now describes
    what a report will count, not what the detector proposes.
    """
    reported = _reported_strike_families(sport)
    policy = reported["counting_policy"]
    # MMA is decided on the ground as much as on the feet, and none of
    # takedowns, control or submissions is read yet.
    covered = "no" if (reported["no_strike_counts"] or reported["withheld_families"]
                       or sport == "mma") else "yes"
    return {"covered": covered, "label": policy["badge"]}


def _prose_list(items) -> str:
    """"a, b and c" - built here rather than with a chain of template filters."""
    items = [str(item) for item in items if item]
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} and {items[-1]}"


# The format each sport is usually fought at, as (rounds, seconds per round).
#
# A default, not an assertion: the upload form preselects it and the athlete
# changes it, and anyone who does not know picks "use the whole video". These
# are the amateur/most-common formats rather than the championship ones - a
# five-round title fight is rarer than a three-round club bout, and the person
# with the title fight is the one who will notice the field and change it.
SPORT_ROUND_DEFAULTS: dict[str, tuple[int, int]] = {
    "kickboxing": (3, 180),     # K-1 3x3
    "boxing": (3, 180),
    "muay_thai": (5, 180),      # the full-rules bout is five threes
    "taekwondo": (3, 120),
    "mma": (3, 300),
}


def _sport_context(request: Request, sport: str) -> dict:
    """Everything a single sport's setup page needs to describe itself."""
    account = _account(request)
    keys = SPORTS[sport]
    default_rounds, default_seconds = SPORT_ROUND_DEFAULTS.get(sport, (3, 180))
    return {
        "request": request,
        "sport": sport,
        "sport_label": RULESET_SPORTS[sport],
        # "a MMA bout" read wrong: the article follows the sound, and MMA
        # starts with "em".
        "sport_article": "an" if RULESET_SPORTS[sport].startswith(("MMA", "A", "E", "I", "O", "U")) else "a",
        # Preselected on the round pickers. See SPORT_ROUND_DEFAULTS.
        "default_round_count": default_rounds,
        "default_round_seconds": default_seconds,
        # The smaller of what WarriorIQ allows and what the host will carry, so
        # a file that cannot possibly arrive is refused here instead of after a
        # minute of uploading.
        "upload_limit_bytes": min(MAX_FIGHT_BYTES, SETTINGS.max_upload_bytes),
        "identity": sport_identity(sport),
        "sports": RULESET_SPORTS,
        # Boxing and MMA each have exactly one ruleset, so asking which one is a
        # question with a single answer. The page drops the field and posts the
        # key instead of showing a select the reader cannot get wrong.
        "rulesets": [(key, RULESET_LABELS[key]) for key in keys],
        "only_ruleset": keys[0] if len(keys) == 1 else None,
        # The workspace roster, so a fight can be filed against the person it
        # was actually about. Empty for a guest, who has no workspace.
        "roster": (
            list_fighters(int(_account(request)["profile_id"]))
            if _account(request) else []
        ),
        # A workspace with one seat has one fighter, so it is told rather than
        # asked which one this fight is about.
        "single_fighter": (_request_plan(request) or {}).get("roster_limit") == 1,
        "unobserved": sport_unobserved(sport),
        # Only offered when a provider is configured; see core/sport_check.py.
        "sport_check_enabled": sport_check.provider() is not None,
        "sport_labels": {key: RULESET_SPORTS[key] for key in SPORTS},
        # What this sport can actually score, rather than a fixed sentence.
        "counted_families": _prose_list(sport_counted_families(sport)),
        # What the *report* will show, which is narrower. Punch and knee counts
        # are withheld until their precision is validated, so "We count
        # punches, kicks and knees" was a promise every report then broke - and
        # a boxing report has no strike numbers at all. Said before upload.
        **_reported_strike_families(sport),
        # True when every ruleset in the sport shares the same blind spot. When
        # only one does - jumping-kick bonuses exist in Point Fighting and
        # nowhere else in kickboxing - the page has to say "depending on the
        # rules" instead of blaming the whole sport.
        "unobserved_is_sport_wide": all(
            set(RULESETS[key].unobserved) == set(sport_unobserved(sport))
            for key in SPORTS[sport]
        ),
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

    An analysis needs an account: /upload answers 401 to a signed-out visitor,
    and that is the only place a job is ever created. So a signed-out visitor
    is told here, before choosing a sport and picking a file, rather than being
    bounced to /login or finding out after the upload.

    This page used to say the analysis would be "deleted after two hours"
    instead. GUEST_RETENTION_HOURS and cleanup_expired_guest_jobs are both
    still real, but nothing can reach them - a guest job cannot be created at
    all - so the sentence described a path that does not exist and implied a
    guest mode this product does not have.

    This route used to redirect while /dashboard, /history, /coach, /profile
    and /compare all answered 200 with the same signed-out shell, so someone
    who bookmarked one of the six got a sales page and someone who bookmarked
    this one got a login form. The shell is the better of the two: the page
    keeps its address, and its call to action carries the destination.
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
            "signed_in": account is not None,
            "sports": RULESET_SPORTS,
            # Boxing's single ruleset is named after the sport, so listing it
            # tells the reader nothing; say what is actually true instead.
            "sport_rulesets": {s: _ruleset_summary(s) for s in SPORTS},
            "identities": SPORT_IDENTITIES,
            "sport_unobserved": {sport: sport_unobserved(sport) for sport in SPORTS},
            "sport_coverage": {sport: _sport_coverage_badge(sport) for sport in SPORTS},
            "version": SETTINGS.version,
        },
    )


@app.get("/analyze/{sport}", response_class=HTMLResponse)
def sport_setup(request: Request, sport: str):
    key = (sport or "").strip().lower()
    if key not in SPORTS:
        raise HTTPException(status_code=404, detail="Unknown sport")
    # The nav chip is rendered from request.state, which the middleware filled
    # in from the *incoming* cookie - the sport of the previous visit. The
    # cookie for this request is only set on the response below, so without
    # this line the page reads "Set up your Boxing fight." under a nav chip
    # that still says Taekwondo. The URL is what the visitor asked for, so it
    # wins over the cookie for this render.
    request.state.active_sport = sport_identity(key)
    response = templates.TemplateResponse(
        request=request, name="analyze.html", context=_sport_context(request, key),
    )
    # Remembered so the shell can keep showing which sport this session is in,
    # and so returning to the product lands where the visitor left off.
    response.set_cookie(
        ACTIVE_SPORT_COOKIE, key, max_age=60 * 60 * 24 * 180,
        httponly=False, samesite="lax", secure=_request_is_secure(request),
    )
    return response


def _record_shot_profile(job_id: str, video_path: Path) -> None:
    """Note whether an upload looks like an edited reel. Runs after the response.

    A sixty-second file in this project's own library is an event reel -
    announcer, crowd, a table of medals, a fight-card poster - and it went
    through the pipeline without complaint, producing a confident report of a
    tracker that spent part of the run following a photograph of a man on a
    poster. Worth knowing about, which is why this exists.

    Not worth making somebody wait for, which is why it is here. It never
    influenced the selection frame or anything else the next page reads, and
    scanning the whole file for cuts is the single most expensive thing in the
    upload handler.

    Everything is swallowed on purpose. This is telemetry about a file that has
    already been accepted, and there is no caller left to return an error to.
    """
    try:
        shots = detect_shot_changes(video_path)
        if not shots.get("looks_edited"):
            return
        LOGGER.info(
            "upload_looks_edited job_id=%s cuts=%s longest_shot=%.1fs of %.1fs",
            job_id, shots["cut_count"], shots["longest_shot_seconds"],
            shots["duration_seconds"])
        record_security_event(
            "upload_looks_edited", severity="info",
            resource_type="fight", resource_id=job_id,
            metadata={"cuts": shots["cut_count"],
                      "longest_shot_seconds": shots["longest_shot_seconds"]})
    except Exception:                                            # noqa: BLE001
        LOGGER.warning("shot_profile_failed job_id=%s", job_id, exc_info=True)


@app.post("/upload", dependencies=[Depends(require_csrf)])
async def upload(
    request: Request,
    background: BackgroundTasks,
    video: UploadFile = File(...),
    fight_type: str = Form("competition"),
    analysis_target: str = Form("BOTH"),
    ruleset: str = Form("K1"),
    start_seconds: float = Form(0.0),
    end_seconds: str = Form(""),
    round_count: int = Form(3),
    # 0 means "the whole video, however long it is". A number here was a
    # standard round length, and there is no such thing: rounds run two
    # minutes, three, five, and a fight can be one continuous span. The real
    # structure is read from the footage by RoundDetector afterwards.
    round_duration_seconds: float = Form(0.0),
    break_duration_seconds: float = Form(60.0),
    selected_rounds: str = Form("ALL"),
    openai_identity_recovery: bool = Form(False),
    external_ai_guardian_permission: bool = Form(False),
    # Which fighter this bout is about. Either an existing roster id, or a new
    # name typed on the setup page - a coach adding a fighter should not have
    # to go somewhere else first.
    fighter_id: str = Form(""),
    fighter_name: str = Form(""),
    rights_confirmed: bool = Form(False),
    people_permissions_confirmed: bool = Form(False),
    minor_permission_status: str = Form(""),
):
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
    if hold := _guardian_hold(account):
        raise HTTPException(403, hold)
    if openai_identity_recovery and not request.state.external_ai_available:
        raise HTTPException(400, "Optional external identity recovery is unavailable.")
    if openai_identity_recovery and minor_permission_status == "guardian_authorized" and not external_ai_guardian_permission:
        raise HTTPException(400, "Parent or guardian permission is also needed before selected frames of minors can be sent to OpenAI.")
    if not video.filename:
        raise HTTPException(400, "Choose a fight video.")
    job_id = getattr(request.state, "upload_job_id", None)
    if not job_id:
        # Admission control assigns this id after reserving the account's
        # allowance and its storage lease. Reaching the handler without one
        # means the two never ran, so accepting the video here would take
        # footage nobody has capacity for. Refuse rather than proceed.
        LOGGER.error("fight_upload_admission_skipped path=%s root_path=%r",
                     request.url.path, request.scope.get("root_path", ""))
        raise HTTPException(500, "WarriorIQ could not start this upload. Please try again.")
    suffix = Path(video.filename).suffix.lower() or ".mp4"
    if suffix not in FIGHT_VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported video format. WarriorIQ reads {FIGHT_VIDEO_LABEL}.")

    if isinstance(video, StoredUpload):
        # A chunked upload assembled this file already. Everything below -
        # the container check, the scanner, the normaliser, the decoder, the
        # limits, the job row - is identical for both paths and runs here
        # rather than being duplicated into a second route where the two could
        # drift. Only the copy is skipped, because copying would mean writing
        # a second half-gigabyte file on a shared host to move it a few inches.
        video_path = video.path
        video_digest = await run_in_threadpool(video.digest)
    else:
        video_path = UPLOADS / f"{job_id}{suffix}"
        # UploadFile uses a spooled file. Keep the blocking disk copy outside the
        # event loop so one large phone upload cannot freeze every other request.
        video_digest = await run_in_threadpool(_save_upload_limited, video, video_path, min(MAX_FIGHT_BYTES, SETTINGS.max_upload_bytes))

    # Before the scanner and before the decoder: the suffix is chosen by
    # whoever names the file, so it says nothing about what is inside it.
    # An eleven-byte text file called fight.mp4 was accepted this far.
    if not await run_in_threadpool(looks_like_video, video_path):
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            400,
            "That file is not a video. The name ends in a video extension but the "
            f"contents are not {FIGHT_VIDEO_LABEL}. Pick the clip straight "
            "from your camera roll.",
        )

    scan = await run_in_threadpool(scan_upload, video_path)
    if not scan["clean"]:
        video_path.unlink(missing_ok=True)
        if scan["status"] == "infected":
            record_security_event("malware_upload_blocked", severity="warning")
            raise HTTPException(400, "This file did not pass the upload safety scan.")
        raise HTTPException(503, "Fight uploads are paused because the safety scanner is unavailable.")

    # Copy the streams into a standard MP4 when the container is not one a
    # browser expects. Measured at 0.20 s on a 101 MB iPhone upload, because
    # nothing is re-encoded: the streams inside are already H.264 and AAC.
    # Footage that genuinely needs re-encoding is left alone here rather than
    # spending minutes of shared-host CPU on a request somebody is waiting on.
    await run_in_threadpool(normalize_container, video_path)

    try:
        info = await run_in_threadpool(get_video_info, video_path)
    except HTTPException:
        video_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        LOGGER.info("upload_not_decodable job_id=%s reason=%s", job_id, type(exc).__name__)
        raise HTTPException(
            400,
            "WarriorIQ could not read this file as a video. It may be corrupted, "
            "still uploading, or saved in a format this build cannot decode. "
            "Try MP4 or MOV exported straight from your phone or camera.",
        ) from exc

    try:
        if info.duration > SETTINGS.max_video_duration_seconds:
            raise HTTPException(413, "This video is longer than the configured analysis limit.")
        if info.width * info.height > SETTINGS.max_video_pixels:
            raise HTTPException(413, "This video's resolution exceeds the configured processing limit.")
        if info.duration <= 0 or info.fps <= 0:
            raise HTTPException(400, "This video has no readable playing time. Please re-export it and try again.")
        if info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
            # A container this build cannot decode still opens: OpenCV reports
            # 0x0 pixels and -1 frames rather than refusing. Nothing checked
            # for that, so the upload passed every guard above - a zero pixel
            # count is not "too large" - and died much later on the selection
            # frame, telling the uploader WarriorIQ could not prepare their
            # video. Reading frames from such a file can also block, so this
            # has to come before inspect_video_quality touches it.
            LOGGER.info(
                "upload_undecodable_stream job_id=%s size=%sx%s frames=%s",
                job_id, info.width, info.height, info.frame_count,
            )
            raise HTTPException(
                400,
                "WarriorIQ could not decode the video track in this file. "
                "Please upload the original MP4 or MOV from your phone or camera "
                "rather than a converted or re-wrapped copy.",
            )
        # One read of the opening answers both questions this asks - is the
        # footage usable, and where should the fighter picker open - because
        # both come from the same frames. Asking them separately meant seeking
        # through the file twice, which is cheap on an MP4 and ruinous on a
        # WebM: measured 78 s of an upload on a sixty-second 1080p WebM, all
        # of it after the last byte had arrived.
        probed_frame, quality_samples = await run_in_threadpool(probe_upload, video_path, info)
        quality = await run_in_threadpool(
            inspect_video_quality, video_path, info, quality_samples)
        # An analysis assumes one continuous view of one bout, and until now
        # nothing checked. A sixty-second file in this project's own library is
        # an edited event reel - announcer, crowd, a table of medals, a
        # fight-card poster - and it went through the pipeline without
        # complaint, producing a confident report of a tracker that spent part
        # of the run following a photograph of a man on a poster.
        #
        # Reported, not refused. The thresholds are fitted against five files
        # of which exactly one is edited, which is not a basis for rejecting
        # somebody's fight; see core/video.py.
    except Exception:
        video_path.unlink(missing_ok=True)
        raise
    # Where the analysed span starts: the beginning of the video unless the
    # caller explicitly asked for later. Not where the selection frame is -
    # that frame only says who is who (core/backtrack.py).
    requested_start = max(0.0, min(float(start_seconds), max(0.0, info.duration - 0.001)))
    start = requested_start
    end = None if not end_seconds.strip() else max(start, min(float(end_seconds), info.duration))
    count = max(1, min(20, int(round_count)))
    # Frame 0 is where a round starts, which is where the referee stands
    # between the fighters with both arms out - the biggest, most central,
    # highest-confidence person on the mat. Seeding there picks the referee,
    # which is exactly the "it analysed the referee" failure people report. So
    # when no explicit start was asked for, open the picker on a moment where
    # the two are actually working. The uploader can still scrub anywhere.
    # Shot detection runs AFTER the response, not before it.
    #
    # It scans the whole file, and measured on a 5 minute 640x480 upload it is
    # 2.44s of the 2.80s a user spends staring at a finished progress bar -
    # every other post-upload step together is 0.36s. What it buys is a log
    # line and a security event. It does not choose the selection frame; that
    # comes from probe_upload, which costs 0.32s and stays on the critical
    # path. So the uploader was waiting two and a half seconds for telemetry
    # about their own file.
    #
    # Deferring it changes nothing a user sees and nothing a later step reads.
    # It does change one thing worth stating: a failure in here no longer
    # deletes the upload. It never should have - the comment above says this is
    # "reported, not refused", and a crash while counting cuts is not a reason
    # to throw away somebody's fight.
    background.add_task(_record_shot_profile, job_id, video_path)
    selection_frame_index = int(round(start * info.fps))
    if start <= 0.0:
        selection_frame_index = probed_frame
    job_dir = OUTPUTS / job_id
    selection_path = job_dir / "selection.jpg"
    # A real fight is never refused for the frame it opens on. The picker frame
    # is tried at several moments, with OpenCV and then ffmpeg, skipping black
    # frames (core.video.selection_frame). If nothing decodes on this host -
    # an AV1 WebM, which this build of OpenCV cannot read at all - the upload
    # is still kept, and the frame picker lets the browser, which can play it,
    # choose the frame. The analysis machine converts the file before reading.
    server_decodes = await run_in_threadpool(opencv_decodes, video_path)
    try:
        chosen_index, frame = await run_in_threadpool(selection_frame, video_path, info, selection_frame_index)
    except Exception as exc:
        LOGGER.warning("upload_selection_frame_unreadable job_id=%s error=%s", job_id, type(exc).__name__)
        chosen_index, frame = None, None
    # Filmed sideways with no rotation tag: turn it upright before anything
    # reads it, so the analysis, the replay and the "left / right" words on the
    # selection page all see the people where they really are (core/orientation.py).
    orientation = {"turned_clockwise": 0, "warning": None}
    if frame is not None and server_decodes:
        turn = await run_in_threadpool(needed_turn, frame, detect_people_in_frame)
        if turn:
            if await run_in_threadpool(tag_rotation, video_path, turn, _ffmpeg_exe()):
                orientation["turned_clockwise"] = turn
                info = await run_in_threadpool(get_video_info, video_path)
                try:
                    chosen_index, frame = await run_in_threadpool(
                        selection_frame, video_path, info, selection_frame_index)
                except Exception as exc:                                # noqa: BLE001
                    LOGGER.warning("upload_rotated_frame_unreadable job_id=%s error=%s", job_id, type(exc).__name__)
                    chosen_index, frame = None, None
            else:
                orientation["warning"] = (
                    "This video looks like it was filmed sideways, and WarriorIQ could not turn it. "
                    "Rotate it on your phone and upload it again for the best result.")
    if frame is not None:
        # Failing to *save* a frame that decoded is this server's disk, not the
        # video, and the browser's frame would fail to save the same way.
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            if not await run_in_threadpool(cv2.imwrite, str(selection_path), frame):
                raise OSError("OpenCV could not save the fighter-selection frame")
        except Exception as exc:
            video_path.unlink(missing_ok=True)
            shutil.rmtree(job_dir, ignore_errors=True)
            LOGGER.warning("upload_selection_frame_failed job_id=%s error=%s", job_id, type(exc).__name__)
            raise HTTPException(
                422,
                "WarriorIQ could not save this video's fighter-selection frame on the server, so the upload "
                "was not kept. Nothing is wrong with your video - try the upload again in a minute.",
            ) from exc
    selection_pending = frame is None
    if selection_pending:
        LOGGER.info("upload_selection_frame_deferred_to_browser job_id=%s", job_id)
        selection_frame_index = 0
    else:
        selection_frame_index = int(chosen_index)
    selection_seconds = selection_frame_index / info.fps if info.fps > 0 else 0.0
    if not server_decodes:
        # Nothing this host could sample, so no brightness or sharpness verdict
        # is honest. The analysis measures the footage after converting it.
        quality = {"status": "unmeasured", "score": None, "notes": [
            "This file's format is converted before analysis, so its picture quality is checked then."]}

    profile_id = int(account["profile_id"]) if account else 0
    # A guest has no workspace to hold a roster, so their fight simply has no
    # fighter attached rather than a half-made one.
    chosen_fighter = None
    if account:
        if fighter_name.strip():
            # The seat count is what a coach's plan sells, so it is checked
            # here rather than described on the pricing page. An existing name
            # is not a new seat: create_fighter resolves it to the same person.
            roster = list_fighters(profile_id)
            already = next((f for f in roster if f["name"].lower() == " ".join(fighter_name.split()).lower()), None)
            if already is None:
                capacity = roster_capacity(_request_plan(request), len(roster))
                if not capacity["can_add"]:
                    video_path.unlink(missing_ok=True)
                    shutil.rmtree(OUTPUTS / job_id, ignore_errors=True)
                    raise HTTPException(
                        402,
                        f"This plan holds {capacity['limit']} "
                        f"fighter{'s' if capacity['limit'] != 1 else ''} and they are all in use. "
                        "Archive one you no longer coach, or move to a plan with more room.",
                    )
            chosen_fighter = create_fighter(profile_id, fighter_name)
        elif fighter_id.strip().isdigit():
            chosen_fighter = get_fighter(profile_id, int(fighter_id))
    create_job(
        job_id,
        {
            "video_path": str(video_path),
            "source_video_sha256": video_digest,
            "original_name": video.filename,
            "fight_type": fight_type.lower(),
            "analysis_target": analysis_target.upper(),
            "ruleset": normalize_ruleset(ruleset),
            # The selection frame's time. Kept under its old name because
            # the frame picker and older workers read it; the analysis start
            # is requested_start_seconds.
            "start_seconds": selection_seconds,
            "selection_seconds": selection_seconds,
            "requested_start_seconds": requested_start,
            # True when no frame could be decoded here; the frame picker then
            # takes it from the browser (selection_frame_image).
            "selection_pending": selection_pending,
            # Who chose the selection frame: "auto" (picked here from motion,
            # nobody has confirmed two fighters are in it), "requested" (the
            # uploader asked for a start time), later "auto_pair", "chosen" or
            # "recheck" (_seek_selection_frame). The selection page says which,
            # so the frame-choice step is never skipped without a word.
            "selection_source": "requested" if requested_start > 0 else "auto",
            "server_decodes": bool(server_decodes),
            "end_seconds": end,
            "round_count": count,
            "round_duration_seconds": (
                float(round_duration_seconds) if round_duration_seconds > 0 else float(info.duration)
            ),
            "break_duration_seconds": float(break_duration_seconds),
            "selected_rounds": _parse_rounds(selected_rounds, count),
            "video_width": info.width,
            "video_height": info.height,
            # Quarter turns applied to a sideways upload, and the warning when
            # one was needed but could not be applied.
            "orientation": orientation,
            "video_duration": info.duration,
            "selection_frame": selection_frame_index,
            "profile_id": profile_id,
            "account_id": int(account["id"]) if account else None,
            "usage_reserved": True,
            "persist_result": bool(account),
            "owner_key": _owner_key(request),
            "quality": quality,
            "upload_scan_status": scan["status"],
            "openai_identity_recovery": bool(openai_identity_recovery),
            "external_ai_opted_in": bool(openai_identity_recovery),
            "fighter_id": chosen_fighter["id"] if chosen_fighter else None,
        },
    )
    acceptance_owner = {"profile_id": int(account["profile_id"])} if account else {"guest_id": request.state.guest_id}
    record_legal_acceptance(
        "fight_video_upload_permission", SETTINGS.policy_version,
        resource_id=job_id,
        metadata={
            # What was actually submitted, not a literal. The guard above means
            # these are both true by the time we get here, but a consent record
            # that hardcodes the answer stops being evidence of consent.
            "rights_confirmed": bool(rights_confirmed),
            "people_permissions_confirmed": bool(people_permissions_confirmed),
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
            metadata={"provider": "OpenAI", "purpose": "fighter_identity_recovery",
                      "explicit_opt_in": True, "guardian_permission": bool(external_ai_guardian_permission),
                      "minor_permission_status": minor_permission_status},
            **acceptance_owner,
        )
    # Straight to the fighters: the clear moment is found on that page
    # (core/person_detect.py), not scrubbed for by hand - unless there is no
    # frame yet, in which case the browser picks one first.
    next_url = f"/frame/{job_id}" if selection_pending else f"/select/{job_id}"
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
    _pin_sport_to_fight(request, _job_sport(job))
    return templates.TemplateResponse(request=request, name="frame.html", context={"request": request, "job_id": job_id, "job": job})


@app.get("/select/{job_id}", response_class=HTMLResponse)
def select_page(request: Request, job_id: str, seconds: float | None = None):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    if job.get("selection_pending") and not (OUTPUTS / job_id / "selection.jpg").exists():
        # No frame could be decoded on the server yet; the browser picks one.
        return RedirectResponse(f"/frame/{job_id}", status_code=303)
    # Arriving with a timestamp means the analysis asked to be told who is who,
    # and named the moment it lost track. Land on that frame rather than making
    # somebody scrub for it - they are here because we already failed once.
    if seconds is not None:
        try:
            _seek_selection_frame(job_id, job, float(seconds), source="recheck")
        except Exception:                                           # noqa: BLE001
            LOGGER.warning("recheck_seek_failed job=%s seconds=%s", job_id, seconds)
    # Which fighter gets the detailed report is a per-fight choice, made on this
    # page, and it already is one - the radio posts focus_fighter with the boxes
    # and the job stores it. What it was not doing is starting from the answer
    # the account already gave: "My identity in reports" on /profile drove the
    # progress dashboard while this page hardcoded Fighter A, so an athlete who
    # had said they were Fighter B had to say it again on every single upload
    # and silently got the wrong report whenever they forgot.
    #
    # The profile value is the DEFAULT only. Which corner somebody is in changes
    # from fight to fight, so the per-fight radio still wins and nothing here
    # writes back to the profile.
    profile_id = _profile_id(request)
    profile = get_profile(profile_id) if profile_id is not None else None
    default_focus = str((profile or {}).get("default_fighter") or "A").upper()
    if default_focus not in {"A", "B"}:
        default_focus = "A"
    # The fighter this upload was filed against on the setup page. It used to
    # vanish here, so "Who are you training?" asked about "Fighter A / B" when
    # the athlete had already said "Theodoulos" one step earlier.
    _pin_sport_to_fight(request, _job_sport(job))
    fighter = None
    if profile_id is not None and str(job.get("fighter_id") or "").isdigit():
        fighter = get_fighter(profile_id, int(job["fighter_id"]))
    return templates.TemplateResponse(
        request=request, name="select.html",
        context={"request": request, "job_id": job_id, "job": job,
                 "default_focus": default_focus,
                 "frame_source": job.get("selection_source"),
                 "frame_clock": _clock(float(job.get("selection_seconds", job.get("start_seconds")) or 0.0)),
                 "fighter_name": (fighter or {}).get("name"),
                 "default_corner": job.get("fighter_a_corner") or ""})


@app.get("/selection-image/{job_id}")
def selection_image(request: Request, job_id: str):
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    path = OUTPUTS / job_id / "selection.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/live-frame/{job_id}")
def live_frame(request: Request, job_id: str, t: float = 0.0):
    """One frame of the fight as a JPEG, for the live page on a browser that
    cannot play the upload (HEVC phone footage in Chrome). The page asks for
    the frame of the newest observation, so the skeletons drawn on it were
    measured on this exact picture."""
    # The page asks about once a second and a half while it is open.
    _enforce_rate_limit(request, "live-frame", 300, 300)
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    path = Path(job["video_path"])
    if not path.exists():
        raise HTTPException(404)
    info = get_video_info(str(path))
    seconds = max(0.0, min(float(t), max(0.0, info.duration - 0.001)))
    frame = read_frame(str(path), int(round(seconds * info.fps)))
    if frame is None:
        raise HTTPException(404)
    height, width = frame.shape[:2]
    if width > 960:
        frame = cv2.resize(frame, (960, max(1, int(round(height * 960 / width)))))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "Could not prepare this frame.")
    return Response(encoded.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


def _seek_selection_frame(job_id: str, job: dict, seconds: float, source: str = "chosen") -> tuple[float, int]:
    """Point this job's selection frame at a moment in the video.

    ``source`` records who chose it: "chosen" (the person, on the frame
    picker), "recheck" (the analysis asked to be shown who is who there) or
    "auto_pair" (the automatic pick found two whole people in it).
    """
    seconds = max(0.0, min(seconds, max(0.0, float(job["video_duration"]) - 0.001)))
    info = get_video_info(job["video_path"])
    frame_number = int(round(seconds * info.fps))
    try:
        frame = read_frame(job["video_path"], frame_number)
    except Exception:                                               # noqa: BLE001
        frame = None
    if frame is None:
        # OpenCV cannot decode some formats at all (AV1 WebM); ffmpeg can.
        frame = ffmpeg_frame(job["video_path"], seconds)
    if frame is None:
        # Nothing on this host decodes it. The frame picker answers this by
        # capturing the frame in the browser (selection_frame_image).
        raise HTTPException(422, "This moment cannot be decoded on the server; your browser will capture it instead.")
    (OUTPUTS / job_id).mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUTS / job_id / "selection.jpg"), frame):
        raise HTTPException(500, "Could not save the selected fighter frame.")
    # A moment somebody chose (or the analysis asked for) is never replaced by
    # the automatic pick.
    # Only who-is-who moves with it: the analysis still covers the whole video.
    update_job(job_id, {"selection_frame": frame_number, "start_seconds": seconds,
                        "selection_seconds": seconds, "auto_frame_done": True,
                        "selection_pending": False, "selection_source": source})
    return seconds, frame_number


# A browser-captured frame is one picture: a 4K PNG is under 25 MB, a JPEG far
# less. Anything bigger is not what the frame picker sends.
MAX_SELECTION_IMAGE_BYTES = 25 * 1024 * 1024


@app.post("/api/selection-frame/{job_id}/image", dependencies=[Depends(require_csrf)])
async def selection_frame_image(request: Request, job_id: str, seconds: float = 0.0):
    """Take the fighter-selection frame from the browser.

    For a file nothing on the web host can decode - an AV1 WebM, which this
    build of OpenCV cannot read and the host may have no ffmpeg for. The
    browser that plays the video captures the paused frame and sends it here,
    so the upload is kept instead of refused. The analysis machine converts
    the video itself before reading it (core.analyzer).
    """
    _enforce_rate_limit(request, "selection-frame", 90, 300)
    job = _authorized_job(request, job_id)
    if not job or "video_path" not in job:
        raise HTTPException(404)
    declared = int(request.headers.get("content-length") or 0)
    if declared > MAX_SELECTION_IMAGE_BYTES:
        raise HTTPException(413, "That frame is larger than a single picture should be.")
    body = await request.body()
    if not body or len(body) > MAX_SELECTION_IMAGE_BYTES:
        raise HTTPException(400, "No frame arrived. Pause the video on a clear moment and try again.")
    image = await run_in_threadpool(cv2.imdecode, np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "That is not a picture. Pause the video on a clear moment and try again.")
    width, height = int(job.get("video_width") or 0), int(job.get("video_height") or 0)
    got_height, got_width = image.shape[:2]
    if width > 0 and height > 0 and (got_width, got_height) != (width, height):
        # The browser reports the picture size it decoded; it must be this
        # video's shape, or the fighter boxes would land on the wrong pixels.
        if abs(got_width / max(1, got_height) - width / max(1, height)) > 0.02:
            raise HTTPException(400, "That frame is not the shape of this video.")
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    duration = float(job.get("video_duration") or 0.0)
    seconds = max(0.0, min(float(seconds), max(0.0, duration - 0.001)))
    fps = float(get_video_info(job["video_path"]).fps or 0.0) or 30.0
    (OUTPUTS / job_id).mkdir(parents=True, exist_ok=True)
    if not await run_in_threadpool(cv2.imwrite, str(OUTPUTS / job_id / "selection.jpg"), image):
        raise HTTPException(500, "Could not save the selected fighter frame.")
    update_job(job_id, {"selection_frame": int(round(seconds * fps)), "start_seconds": seconds,
                        "selection_seconds": seconds, "auto_frame_done": True,
                        "selection_pending": False, "selection_from_browser": True,
                        "selection_source": "chosen"})
    return {"ok": True, "seconds": seconds}


_auto_frame_locks: dict[str, threading.Lock] = {}
_auto_frame_locks_guard = threading.Lock()


def _auto_pick_selection_frame(job_id: str) -> list[dict] | None:
    """Move a fresh upload's selection frame to its clearest early moment.

    Runs once per job, on the first candidate request from the selection page,
    so the upload itself never waits on it. Returns the people found in the
    chosen frame, or None when nothing was moved (no detector, no clear
    moment, or already decided).
    """
    with _auto_frame_locks_guard:
        lock = _auto_frame_locks.setdefault(job_id, threading.Lock())
    with lock:
        job = get_job(job_id)
        if not job or job.get("auto_frame_done"):
            return None
        try:
            moment = find_clear_moment(job["video_path"])
        except Exception as exc:                                    # noqa: BLE001
            LOGGER.warning("auto_frame_failed job=%s error=%s", job_id, type(exc).__name__)
            moment = None
        if moment is None:
            update_job(job_id, {"auto_frame_done": True})
            return None
        try:
            _seek_selection_frame(job_id, job, float(moment["seconds"]), source="auto_pair")
        except Exception as exc:                                    # noqa: BLE001
            LOGGER.warning("auto_frame_seek_failed job=%s error=%s", job_id, type(exc).__name__)
            update_job(job_id, {"auto_frame_done": True})
            return None
        return moment["people"]


@app.post("/api/selection-frame/{job_id}", dependencies=[Depends(require_csrf)])
def set_selection_frame(request: Request, job_id: str, payload: SelectionFramePayload):
    # Each call seeks and decodes a frame from the uploaded video. Cheap
    # once, not cheap in a loop, and it is reachable before any analysis
    # allowance is spent. Generous enough to scrub through a round.
    _enforce_rate_limit(request, "selection-frame", 90, 300)
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    seconds, frame_number = _seek_selection_frame(job_id, job, float(payload.seconds))
    return {"ok": True, "seconds": seconds, "frame": frame_number}


@app.get("/api/detect/{job_id}")
def detect_people(request: Request, job_id: str):
    """Return optional person candidates without blocking manual selection."""
    job = _authorized_job(request, job_id)
    path = OUTPUTS / job_id / "selection.jpg"
    if not job or not path.exists():
        raise HTTPException(404)
    # The light detector (core/person_detect.py: 3.8 MB, OpenCV only) is the
    # one built for the web host, so it always runs. WARRIORIQ_SELECTION_DETECTION
    # governs only the pose-model fallback further down. It used to switch both
    # off, and it is off on Render: no candidate box was ever drawn and no
    # clear moment ever found there, while the page said WarriorIQ boxes the
    # people for you (QA, 2026-09).
    moved = _auto_pick_selection_frame(job_id)
    job = get_job(job_id) or job
    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(500, "Could not read selection image")
    # How this frame was chosen, for the page to say so: an automatic pick is
    # only presented as a good one when two whole people were found in it.
    about_frame = {
        "frame_moved": moved is not None,
        "seconds": float(job.get("selection_seconds", job.get("start_seconds")) or 0.0),
        "frame_source": job.get("selection_source"),
    }
    people = moved if moved is not None else detect_people_in_frame(frame)
    if people is not None:
        # Small figures are crowd, not fighters, and too small to tap.
        tall = [p for p in people if p["box"][3] - p["box"][1] >= 0.10 * frame.shape[0]]
        return {
            "people": tall, "width": job["video_width"], "height": job["video_height"],
            "availability": "candidates_ready",
            "pair_found": pair_score(tall, frame.shape[0])[1] is not None, **about_frame,
        }
    if not SETTINGS.selection_detection_enabled:
        return {
            "people": [], "width": job["video_width"], "height": job["video_height"],
            "availability": "manual_only", "pair_found": False, **about_frame,
        }
    try:
        tracker = _get_pose_tracker()
        results = tracker.predict_selection(frame)
    except Exception as exc:
        LOGGER.warning("Selection candidate detection unavailable: %s", type(exc).__name__)
        return {
            "people": [], "width": job["video_width"], "height": job["video_height"],
            "availability": "manual_only", "pair_found": False, **about_frame,
        }
    boxes = []
    result = results[0]
    if result.boxes is not None:
        # Same tensor's attributes, so equal length by construction - strict
        # says so, and would complain loudly if a future ultralytics ever
        # returned them mismatched instead of quietly dropping detections.
        for box, conf in zip(
            result.boxes.xyxy.detach().cpu().numpy(),
            result.boxes.conf.detach().cpu().numpy(),
            strict=True,
        ):
            boxes.append({"box": [float(x) for x in box], "confidence": float(conf)})
    return {
        "people": boxes, "width": job["video_width"], "height": job["video_height"],
        "availability": "candidates_ready",
        "pair_found": pair_score(boxes, frame.shape[0])[1] is not None, **about_frame,
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


def _person_inside(person: list[float], drawn: list[float]) -> bool:
    """Whether a detected person is what a drawn box is around.

    Most of the person has to be inside the box, and they have to fill a fair
    part of it: a box drawn around a punch bag beside its holder, or around a
    blank title card with a small figure in a corner, is not a box around a
    fighter.
    """
    inter = (max(0.0, min(person[2], drawn[2]) - max(person[0], drawn[0]))
             * max(0.0, min(person[3], drawn[3]) - max(person[1], drawn[1])))
    person_area = max(1.0, (person[2] - person[0]) * (person[3] - person[1]))
    drawn_area = max(1.0, (drawn[2] - drawn[0]) * (drawn[3] - drawn[1]))
    return inter / person_area >= 0.5 and inter / drawn_area >= 0.2


def _boxes_without_a_person(frame, boxes: dict[str, list[float]], width: float, height: float) -> list[str]:
    """The labels of drawn boxes that hold no detected person.

    QA, 2026-10-04: a punch bag and a blank title card were both accepted as
    fighters. Checked on the selection frame with the same light detector the
    page uses to offer candidates (core/person_detect.py). When that detector
    cannot run, nothing can be checked here and nothing is refused; the
    analysis's own seed check still stops a box with nobody in it from being
    followed as a fighter.
    """
    if frame is None:
        return []
    people = detect_people_in_frame(frame)
    if people is None:
        LOGGER.info("selection_person_check_unavailable")
        return []
    scale_x = frame.shape[1] / max(1.0, float(width))
    scale_y = frame.shape[0] / max(1.0, float(height))
    missing = []
    for label, box in boxes.items():
        scaled = [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]
        if not any(_person_inside(person["box"], scaled) for person in people):
            missing.append(label)
    return missing


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
        "fighter_b_box": list(job.get("fighter_b_box") or []),
        "solo": bool(job.get("solo")),
        "analysis_target": "A" if job.get("solo") else "BOTH",
        "focus_fighter": job.get("focus_fighter") or "A",
        "fight_type": job["fight_type"],
        "ruleset": job["ruleset"],
        # A worker that predates requested_start_seconds reads start_seconds as
        # where to start; such a worker is now refused at claim time (see
        # remote_worker_claim), so this only keeps the payload readable. A
        # current one analyses from requested_start and seeds identity at
        # selection_seconds.
        "start_seconds": float(job.get("selection_seconds", job.get("start_seconds", 0.0)) or 0.0),
        "selection_seconds": float(job.get("selection_seconds", job.get("start_seconds", 0.0)) or 0.0),
        "requested_start_seconds": float(job.get("requested_start_seconds", 0.0) or 0.0),
        "end_seconds": job.get("end_seconds"),
        "round_count": int(job.get("round_count", 1)),
        "round_duration_seconds": float(job.get("round_duration_seconds", 120.0)),
        "break_duration_seconds": float(job.get("break_duration_seconds", 60.0)),
        "selected_rounds": job.get("selected_rounds"),
        "openai_identity_recovery": bool(job.get("openai_identity_recovery") and job.get("external_ai_opted_in")),
        "external_ai_opted_in": bool(job.get("external_ai_opted_in")),
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
    persist_completed_job(job_id, str(job.get("analysis_run_id") or ""))


@app.post("/api/worker/heartbeat", dependencies=[Depends(_require_remote_worker)])
def remote_worker_heartbeat(request: Request, payload: WorkerIdentityPayload):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(payload.worker_id)
    record_worker_heartbeat(worker_id)
    return {"ok": True}


@app.post("/api/worker/claim", dependencies=[Depends(_require_remote_worker)])
def remote_worker_claim(request: Request, payload: WorkerIdentityPayload):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(payload.worker_id)
    record_worker_heartbeat(worker_id)
    worker_alerts.note_worker_connected(outputs=OUTPUTS, recipients=SETTINGS.admin_emails,
                                        send=send_transactional_email)
    if payload.analysis_version is None or payload.analysis_version < ANALYSIS_VERSION:
        # QA, 2026-10-04: a GPU worker deployed from an older checkout kept
        # analysing from the fighter-selection frame for days after the web
        # app had the fix, and its reports said "Analysis complete". A worker
        # older than this server gets no work; the fight stays queued for one
        # that is current, and the log says exactly what to redeploy.
        LOGGER.error(
            "worker_outdated worker_id=%s analysis_version=%s required=%s - redeploy the worker "
            "(modal deploy deploy/modal_worker.py, or restart a local worker on the current code)",
            worker_id, payload.analysis_version, ANALYSIS_VERSION)
        return {"job": None, "refused": "worker_outdated", "required_analysis_version": ANALYSIS_VERSION}
    claimed = claim_next_job(worker_id)
    if not claimed:
        return {"job": None}
    job_id, job = claimed
    record_worker_heartbeat(worker_id, job_id)
    return {"job": _remote_job_payload(job_id, job)}


@app.get("/api/worker/jobs/{job_id}/video", dependencies=[Depends(_require_remote_worker)])
def remote_worker_video(request: Request, job_id: str, worker_id: str, analysis_run_id: str):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(worker_id)
    job = _owned_worker_job(job_id, worker_id, analysis_run_id)
    video_path = Path(str(job.get("video_path") or ""))
    if not video_path.is_file() or video_path.parent.resolve() != UPLOADS.resolve():
        raise HTTPException(404, "Fight video not found.")
    record_worker_heartbeat(worker_id, job_id)
    return FileResponse(video_path, filename=f"{job_id}{video_path.suffix.lower()}", media_type="video/mp4")


@app.post("/api/worker/dataset/backfill", dependencies=[Depends(_require_remote_worker)])
def remote_worker_dataset_backfill(request: Request):
    """Write the pose windows for labels that were saved without one.

    The label endpoint only exports a training sequence when the owning profile
    has allow_model_training switched on, which is off by default and correct -
    nobody's fights should train a model unasked. The cost is that turning it on
    afterwards leaves earlier labels as verdicts with no data attached, and a
    labelling session is thirty minutes of someone's evening.

    This regenerates them from the tracking file the worker uploaded, for
    annotations whose owner has since consented. It can only fill gaps: a label
    with a sequence already is left alone, and a fight whose tracking file has
    aged out simply cannot be recovered and is reported as skipped.
    """
    _require_remote_worker(request)

    filled = skipped_no_consent = skipped_no_tracking = already = 0
    for annotation in list_annotations():
        if is_strike_check(annotation) or is_down_check(annotation):
            continue      # one-tap answers, not a technique: never a training label
        if annotation.get("sequence_path"):
            already += 1
            continue
        job_id = annotation["job_id"]
        fight = get_fight(job_id)
        profile = get_profile(int(fight["profile_id"])) if fight else None
        if not profile or not profile.get("allow_model_training"):
            skipped_no_consent += 1
            continue
        corrected = annotation.get("corrected") or {}
        directory = completed_artifact_directory(job_id)
        if directory is None:
            skipped_no_tracking += 1
            continue
        path = export_sequence(
            job_id, int(annotation["id"]), corrected, float(annotation["event_time"]),
            tracking_path=directory / "tracking.jsonl",
            source_fight_id=(get_job(job_id) or {}).get("source_video_sha256"),
        )
        if path:
            set_annotation_sequence(int(annotation["id"]), path)
            filled += 1
        else:
            skipped_no_tracking += 1

    return {
        "filled": filled,
        "already_had_sequences": already,
        "skipped_without_consent": skipped_no_consent,
        "skipped_tracking_missing": skipped_no_tracking,
    }


@app.get("/api/worker/dataset", dependencies=[Depends(_require_remote_worker)])
def remote_worker_dataset(request: Request):
    """Hand the labelled training set to the machine that does the training.

    Labels are made in the browser and land on the web server; the model is
    trained on the GPU box. Nothing connected the two, so a labelling session
    was stranded server-side with no way to reach the trainer short of a file
    manager this host does not provide.

    Worker token, not a login: the puller runs on the same machine as the
    worker and already holds it, and it keeps a training corpus off a
    cookie-authenticated path.
    """
    _require_remote_worker(request)
    folder = DATASET / "sequences"
    files = sorted(folder.glob("*.npz")) if folder.exists() else []
    if not files:
        raise HTTPException(404, "No labelled sequences have been recorded yet.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            bundle.write(path, arcname=path.name)
    payload = buffer.getvalue()
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="warrioriq-dataset.zip"',
            "X-WarriorIQ-Sequence-Count": str(len(files)),
        },
    )


@app.post("/api/worker/jobs/{job_id}/progress", dependencies=[Depends(_require_remote_worker)])
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


@app.post("/api/worker/jobs/{job_id}/complete", status_code=201, dependencies=[Depends(_require_remote_worker)])
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
        and completed_artifact_directory(job_id, existing) is not None
        and (completed_artifact_directory(job_id, existing) / "report.json").is_file()
    ):
        # A worker may retry after the web server committed the result but the
        # success response was lost. Treat that exact generation as complete;
        # never re-run it or turn a successful analysis into an error.
        persist_completed_job(job_id, analysis_run_id)
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
        build = result_check(report)
        if build["outdated"]:
            # Claimed before this server was deployed, by a worker that has not
            # been. Its report may cover less of the video than this code
            # promises, so it is not stored; the run's lease lapses and a
            # current worker analyses the fight again.
            LOGGER.error("worker_result_outdated job_id=%s worker_id=%s analysis_version=%s required=%s",
                         job_id, worker_id, build["analysis_version"], ANALYSIS_VERSION)
            raise HTTPException(409, "This result came from an outdated analysis worker and was not stored.")
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


@app.post("/api/worker/jobs/{job_id}/failed", dependencies=[Depends(_require_remote_worker)])
def remote_worker_failed(request: Request, job_id: str, payload: WorkerFailurePayload):
    _require_remote_worker(request)
    worker_id = _validated_worker_id(payload.worker_id)
    job = _owned_worker_job(job_id, worker_id, payload.analysis_run_id)
    if job.get("usage_reserved") and job.get("account_id"):
        release_analysis(int(job["account_id"]), job_id)
        update_job(job_id, {"usage_reserved": False})
    if not update_job_for_worker(job_id, worker_id, payload.analysis_run_id, {
        "status": "error",
        "message": _worker_failure_message(payload.error_code),
        "worker_lease_expires_epoch": None,
        "worker_error_code": str(payload.error_code or "analysis_failed")[:80],
    }, renew_lease=False):
        raise HTTPException(409, "This worker no longer owns the analysis generation.")
    record_worker_heartbeat(worker_id)
    return {"ok": True}


@app.post("/api/pair-check/{job_id}", dependencies=[Depends(require_csrf)])
def pair_check(request: Request, job_id: str, payload: PairCheckPayload):
    """Do the two boxed fighters look alike? Asked as soon as both are drawn.

    The same comparison /api/start makes, moved to the moment it can still
    change anything. It used to be answered only after Start, as a toast on
    the way to the progress page - so two fighters in the same kit were found
    out two or three minutes later, when the report withheld everything. Here
    the person who can pick a better frame is still looking at this one.
    """
    _enforce_rate_limit(request, "pair-check", 60, 600)
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    chosen_frame = cv2.imread(str(OUTPUTS / job_id / "selection.jpg"))
    if chosen_frame is None:
        return {"checked": False}
    height, width = chosen_frame.shape[:2]
    fighter_a_box = _validated_fighter_box(payload.fighter_a_box, width, height, "Fighter A")
    fighter_b_box = _validated_fighter_box(payload.fighter_b_box, width, height, "Fighter B")
    kit = kit_similarity(chosen_frame, fighter_a_box, fighter_b_box)
    if kit is None:
        return {"checked": False}
    alike = kit["similarity"]
    looks_alike = float(alike) >= SETTINGS.max_kit_similarity
    return {
        "checked": True,
        "similarity": round(float(alike), 3),
        "looks_alike": looks_alike,
        "cause": kit_alike_reason(kit) if looks_alike else None,
        **({"message": _looks_alike_message(float(alike), kit)} if looks_alike else {}),
    }


@app.post("/api/start/{job_id}", dependencies=[Depends(require_csrf)])
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
    if payload.solo:
        return _start_solo(request, job_id, job, fighter_a_box, video_width, video_height, capacity)
    if payload.fighter_b_box is None:
        raise HTTPException(400, "Draw a box around Fighter B too, or choose a solo session for a video of one person.")
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

    # Asked here, on one frame, in a few milliseconds - because it decides
    # whether anything the analysis says afterwards about *which* fighter did
    # what can be believed, and because this is the last moment the person who
    # can actually answer it is still looking at the screen.
    chosen_frame = cv2.imread(str(OUTPUTS / job_id / "selection.jpg"))
    empty = _boxes_without_a_person(
        chosen_frame, {"Fighter A": fighter_a_box, "Fighter B": fighter_b_box}, video_width, video_height)
    if empty:
        raise HTTPException(400, (
            f"No person was found inside the {' or the '.join(empty)} box. Draw each box around a "
            "fighter's whole body - WarriorIQ cannot follow a punch bag, a title card or an empty "
            "part of the frame. If the fighters are hard to see here, pick a clearer moment."))
    kit = kit_similarity(chosen_frame, fighter_a_box, fighter_b_box)
    alike = None if kit is None else kit["similarity"]
    # Warned, never refused. Most footage is not shot for us, and a user with a
    # phone video of two fighters in the same club kit still deserves an
    # analysis - they just deserve to be told which parts of it to trust. The
    # results carry the same finding through to the report.
    looks_alike = (
        alike is not None and alike >= SETTINGS.max_kit_similarity
    )
    if looks_alike:
        LOGGER.info("fighters_look_alike job=%s similarity=%.3f", job_id, float(alike))
    # Asked in the same breath and for the same reason: the seed decides who
    # the report is about, and this is the last moment the person who knows is
    # still looking at the frame.
    on_official = _seed_official_warning(chosen_frame, fighter_a_box, fighter_b_box)
    if on_official:
        LOGGER.info("seed_looks_like_official job=%s fighters=%s probability=%.3f",
                    job_id, ",".join(on_official["fighters"]), on_official["probability"])
    # A/B is the report focus, not a tracking shortcut. WarriorIQ always
    # analyzes both selected fighters so identity context and the scorecard do
    # not disappear when the user asks for a detailed report on one athlete.
    focus_fighter = (payload.focus_fighter or payload.analysis_target or "A").upper()
    if focus_fighter not in {"A", "B"}:
        raise HTTPException(400, "Choose Fighter A or Fighter B for the detailed report.")
    fighter_a_corner = (payload.fighter_a_corner or "").strip().lower() or None
    if fighter_a_corner not in {None, "red", "blue"}:
        raise HTTPException(400, "Choose the red corner, the blue corner, or no corners for Fighter A.")

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
            "fighter_a_corner": fighter_a_corner,
            **({"message": _deferred_analysis_message()} if capacity["deferred"] else {}),
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
    return _analysis_started_response(request, job_id, capacity["deferred"],
                                      looks_alike=float(alike) if looks_alike else None,
                                      on_official=on_official, kit=kit)


def _start_solo(request: Request, job_id: str, job: dict, fighter_a_box: list[float],
                video_width: float, video_height: float, capacity: dict):
    """Queue a solo session: one person, movement and guard only, no scoring."""
    chosen_frame = cv2.imread(str(OUTPUTS / job_id / "selection.jpg"))
    if _boxes_without_a_person(chosen_frame, {"person": fighter_a_box}, video_width, video_height):
        raise HTTPException(400, (
            "No person was found inside the box. Draw it around the whole body of the person "
            "training - WarriorIQ cannot follow a punch bag, a title card or an empty part of "
            "the frame."))
    _save_fighter_portrait(job_id, "A", fighter_a_box)
    req = _analysis_request(job_id, {**job, "solo": True}, fighter_a_box, [], "A")
    if job.get("account_id") and not job.get("usage_reserved"):
        if not reserve_analysis(int(job["account_id"]), job_id):
            plan = _request_plan(request)
            raise HTTPException(429, f"Your {plan['label']} plan includes {plan['limit_label'].lower()}. Your allowance will reset automatically.")
        update_job(job_id, {"usage_reserved": True})
    try:
        analysis_run_id = prepare_job_run(job_id, {
            "analysis_target": "A", "focus_fighter": "A", "solo": True,
            "fighter_a_box": fighter_a_box, "fighter_b_box": None, "fighter_a_corner": None,
            **({"message": _deferred_analysis_message()} if capacity["deferred"] else {}),
        })
    except AnalysisStateNotPersisted as exc:
        if job.get("account_id") and get_job(job_id).get("usage_reserved"):
            release_analysis(int(job["account_id"]), job_id)
            update_job(job_id, {"usage_reserved": False})
        LOGGER.error("analysis_queue_not_persisted job_id=%s", job_id, exc_info=exc)
        raise HTTPException(
            503, "WarriorIQ could not queue this analysis. Your video and selection are preserved.",
        ) from exc
    if SETTINGS.analysis_worker_mode == "inprocess":
        executor.submit(_run_job, job_id, req, analysis_run_id)
    else:
        _wake_analysis_worker(job_id)
    return _analysis_started_response(request, job_id, capacity["deferred"])


@app.post("/api/restart/{job_id}", dependencies=[Depends(require_csrf)])
def restart_interrupted_analysis(request: Request, job_id: str):
    # Restarting queues GPU work. Without a cap one account can fill the
    # analysis queue by holding a key down, and the machine doing the work
    # is a single RTX 5060 - there is no second one to absorb it.
    _enforce_rate_limit(request, "analysis-restart", 12, 600)
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
    b_box_ok = bool(job.get("solo")) or (isinstance(fighter_b_box, list) and len(fighter_b_box) == 4)
    if not isinstance(fighter_a_box, list) or len(fighter_a_box) != 4 or not b_box_ok:
        raise HTTPException(409, "The saved session predates resumable analysis. Return to fighter selection once; your video is still available.")
    req = _analysis_request(job_id, job, fighter_a_box, fighter_b_box, focus_fighter)
    try:
        analysis_run_id = prepare_job_run(job_id, {
            "message": _deferred_analysis_message() if capacity["deferred"] else "Restarting the preserved analysis session",
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
    _pin_sport_to_fight(request, _job_sport(job))
    return templates.TemplateResponse(request=request, name="progress.html", context={
        "request": request, "job_id": job_id, "initial_status": _public_job_status(job_id, job),
        # The same counting policy as the upload page and the report, so the
        # live view cannot claim "leg strikes only" above a feed of punches.
        "live_counting_note": counting_policy(_job_sport(job)).live_note,
        "strike_counts_published": STRIKE_COUNTS_PUBLISHED,
        # The families this sport's report counts, so the boxing live page
        # does not list kick statistics.
        "live_families": published_families(_job_sport(job)),
        # A solo session follows one person; Fighter B's panel is hidden.
        "solo": bool(job.get("solo")),
    })


def _public_job_status(job_id: str, job: dict) -> dict:
    public_fields = {
        "job_id", "status", "percent", "message", "stage", "elapsed_seconds", "eta_seconds",
        "processed_video_seconds", "video_duration_seconds", "fighter_a_confidence",
        "fighter_b_confidence", "current_round", "quality_mode", "live_event_mode",
        "live_events", "provisional_stats", "latest_observation", "focus_fighter", "analysis_run_id",
    }
    payload = {key: value for key, value in job.items() if key in public_fields}
    if not STRIKE_COUNTS_PUBLISHED:
        # No strike counts on the live page while they are switched off: no
        # event feed, and of the provisional statistics only how much of each
        # fighter was observed. Done here, not in the page, so the numbers are
        # not sent at all.
        payload["live_events"] = []
        stats = payload.get("provisional_stats") or {}
        payload["provisional_stats"] = {
            "attempt_counts_available": False, "action_labels_available": False,
            "fighters": {
                fighter: {"observation_coverage": (item or {}).get("observation_coverage")}
                for fighter, item in (stats.get("fighters") or {}).items()
            },
        }
    payload.setdefault("job_id", job_id)
    payload.setdefault("video_duration_seconds", job.get("video_duration", 0.0))
    # Where the analysed span actually starts: reported by the run once it
    # knows (it can begin later than requested if the fighters could not be
    # followed back that far), else the requested start.
    start_seconds = float(job.get("analysed_from_seconds", job.get("requested_start_seconds", 0.0)) or 0.0)
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


def _worker_connected() -> bool:
    """Is an analysis worker heartbeating? Cheap: one small file read."""
    if SETTINGS.analysis_worker_mode == "inprocess":
        return True
    age = worker_status_heartbeat_age()
    return age is not None and age <= SETTINGS.worker_stale_seconds


@app.get("/api/status/{job_id}")
def status(request: Request, response: Response, job_id: str):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    if job.get("status") == "queued" and SETTINGS.worker_alert_seconds:
        # Someone is waiting on this fight right now: if no worker is there to
        # take it, tell the owner (core/worker_alerts.py).
        worker_alerts.check_waiting_fight(
            job, worker_online=_worker_connected(), outputs=OUTPUTS,
            recipients=SETTINGS.admin_emails, threshold_seconds=SETTINGS.worker_alert_seconds,
            send=send_transactional_email)
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


def _seed_official_warning(image, fighter_a_box, fighter_b_box) -> dict | None:
    """Is either selection on an official rather than on a competitor?

    The seed decides everything downstream, and seeding the referee does not
    produce a slightly worse report - it produces a confident report about the
    wrong person. It is also easy to do by accident: the referee stands between
    the fighters, so on a small frame their box is a few pixels from a
    fighter's, and on a paused video the fighters move between the moment a
    frame is chosen and the moment the boxes are drawn.

    Measured 2026-09-15 across three bouts, scoring boxes whose subject was
    known because the frames had been rendered and looked at:

        known referee        median 0.993   p10 0.657
        119 fighter samples  median 0.007-0.044   max 0.084   none above 0.5

    So `min_referee_probability` (0.5) separates them with a wide margin, and
    the classifier never called a fighter an official on any footage here.

    **Warned, never refused**, the same as the look-alike check beside it. The
    margin is wide but it is three bouts, and the lesson of the last two days
    is that a number measured on narrow footage does not transfer - see
    core/config.py on the release threshold. A false refusal would block a
    legitimate upload with no way round it; a false warning costs a sentence.
    """
    import numpy as np

    from core.referee import referee_probabilities

    if image is None:
        return None
    try:
        scores = referee_probabilities(
            image, np.asarray([fighter_a_box, fighter_b_box], dtype=np.float32))
    except Exception:                                               # noqa: BLE001
        # A seed check must never be the reason a fight cannot be analysed.
        return None
    flagged = [
        (name, float(score)) for name, score in zip(("A", "B"), list(scores) + [None, None])
        if score is not None and score >= SETTINGS.min_referee_probability
    ]
    # **Exactly one, never both.** The score is not an absolute reading of
    # "official"; it is a reading of uniform brightness against the room, and
    # on footage it was not calibrated for it can say "official" about
    # everybody. Tried on 1947 black-and-white boxing, where the print is
    # desaturated and every person is pale against a dark ring: the two boxers
    # scored 0.997 and the white-clad referee 0.989, so the absolute test
    # flagged the correct selection more confidently than the wrong one.
    #
    # The signal that survives is relative. One of two boxes looking like an
    # official while the other does not is informative; both looking like one
    # means the classifier has no discrimination on this footage and the honest
    # thing is to say nothing. A warning that fires on every upload is worse
    # than no warning, because it teaches the user to dismiss it - and the one
    # time it is right is the time it is dismissed.
    if len(flagged) != 1:
        return None
    which = " and ".join(f"Fighter {name}" for name, _ in flagged)
    return {
        "fighters": [name for name, _ in flagged],
        "probability": round(max(score for _, score in flagged), 3),
        "message": (
            f"The box drawn for {which} looks like an official rather than a "
            "competitor - the kit reads as a referee's. We will still analyse it, "
            "but if that is the referee the whole report will describe the wrong "
            "person, and it will look confident while doing it. Worth picking again."
        ),
    }


def _appearance_observation(image, box):
    """Wrap a selection box so it can be compared the way the analysis will.

    Deliberately the same descriptor the identity manager uses. A check that
    measured similarity differently from the thing it is predicting would be
    worse than no check at all.
    """
    from core.identity import appearance_hist
    from core.types import PersonObservation

    import numpy as np

    if image is None or box is None:
        return None
    return PersonObservation(
        track_id=None, box=np.asarray(box, dtype=np.float32), confidence=1.0,
        appearance=appearance_hist(image, box),
    )


def _identity_lost_to_camera(report: dict) -> bool:
    """The identity check failed for a reason re-picking the fighters cannot fix.

    The name is historical: it began as "the camera kept losing them". It now
    covers every such cause (core.report.identity_failure), so every "pick
    them again" on the page gives way to the one recommendation that applies.
    """
    cause = _identity_failure(report)
    if cause is not None:
        return not cause["repick"]
    from core.report import identity_churned

    return any(identity_churned((report or {}).get("tracking") or {}).values())


def _identity_failure(report: dict) -> dict | None:
    """The one cause and recommendation for a failed identity check."""
    tracking = (report or {}).get("tracking") or {}
    target = ((report or {}).get("video") or {}).get("analysis_target", "BOTH")
    required = ("A", "B") if target == "BOTH" else (target,)
    return identity_failure(tracking, required)


def _identity_withheld(report: dict, cause: dict, job_id: str | None) -> dict:
    """The score box for a failed identity check, from the shared cause."""
    tracking = report.get("tracking") or {}
    withheld = {
        "reason": f"{cause['headline']} Rather than guess, no strike is credited to either name.",
        "fix": cause["advice"],
    }
    if cause["repick"] and job_id:
        # Land them on the moment we lost track, when we know one; otherwise
        # on the frame they picked.
        recheck = f"/select/{job_id}"
        moment = tracking.get("last_identity_confusion_frame")
        fps = float((report.get("video") or {}).get("fps") or 0) or 30.0
        if moment:
            recheck += f"?seconds={max(0.0, float(moment) / fps):.2f}"
        withheld["action"] = {"label": "Show me who is who", "url": recheck}
    return withheld


def _score_withheld(report: dict, job_id: str | None = None) -> dict | None:
    """Why this fight has no score, in words a fighter can act on.

    The page used to say "Strike scoring needs the action model", which names
    an internal component rather than a reason and is usually not even the
    right one - a score is withheld far more often because a fighter was lost
    on camera. The accurate explanation existed all along in the scorecard's
    disclaimer, buried inside the deep-dive section at the bottom of the
    report, where someone looking at "Not scored" never saw it.
    """
    scorecard = report.get("scorecard") or {}
    if scorecard.get("available"):
        return None
    tracking = report.get("tracking") or {}
    status = scorecard.get("status")

    def _pct(key: str) -> str:
        try:
            return f"{float(tracking.get(key, 0)) * 100:.0f}%"
        except (TypeError, ValueError):
            return "an unknown share"

    if status in {"fighters_not_separable", "identity_integrity_failed"}:
        if status == "fighters_not_separable" and "fighters_separable" not in tracking:
            # The status is the record of the verdict; an older report may
            # carry it without the field the resolver reads.
            report = {**report, "tracking": {**tracking, "fighters_separable": False}}
        cause = _identity_failure(report)
        if cause is not None:
            return _identity_withheld(report, cause, job_id)
    if status == "both_fighters_required":
        return {
            "reason": "You analysed one fighter, so there is no opponent to score against.",
            "fix": "Run it again and choose Analyze both fighters.",
        }
    # Withheld because strike counting is not validated, not because of the
    # footage. A boxing report said "Tracking was not steady enough" at 89%
    # coverage, and a taekwondo report said "None were clear enough" above a
    # panel counting 28 kicks - both sending the athlete to redo a selection
    # or refilm a bout that was fine.
    if status == "strike_counts_off":
        return {
            "reason": ("A score is built from counted strikes, and WarriorIQ does not count strikes "
                       "until its counting is accurate enough."),
            "fix": "Nothing to redo - the movement, guard and balance numbers below are measured and real.",
            "disclaimer": ("No score is shown. A score is built from counted strikes, and automatic "
                           "strike counting is switched off until it is accurate enough. Movement, "
                           "guard, balance, centre and pressure below are unaffected."),
        }
    if status == "strike_counts_implausible":
        return {
            "reason": ("The strike counts for this fight came out impossible for a real fight, "
                       "so neither they nor a score built from them are shown."),
            "fix": ("Usually the fighters were mixed up or lost for part of the video. A steadier, "
                    "wider shot with both fighters in frame helps."),
            "disclaimer": ("No score is shown: the strike counts failed a check against real fight "
                           "statistics. Movement, guard, balance, centre and pressure below are unaffected."),
        }
    if status == "punch_counting_unavailable":
        return {
            "reason": (
                "Boxing is scored on punches, and WarriorIQ cannot count punches accurately yet, "
                "so there is nothing fair to score."
                if scorecard.get("sport") == "boxing" else
                "Scoring a round needs punches counted as well as kicks, and WarriorIQ cannot "
                "count punches accurately yet."
            ),
            "fix": "Nothing to redo - the movement, guard and balance numbers below are measured and real.",
            "disclaimer": (
                "No score is shown. Boxing is scored on punches, and WarriorIQ's punch counting "
                "is not accurate enough yet - checked against video, the punch count was "
                "overstated. Movement, guard, balance and coverage below are unaffected."
                if scorecard.get("sport") == "boxing" else
                "No score is shown. Scoring a round needs punches counted as well as kicks, and "
                "WarriorIQ's punch counting is not accurate enough yet - checked against video, "
                "the kick count came out right and the punch count was overstated. Movement, "
                "guard, balance and the kick count below are unaffected."
            ),
        }
    if (status == "insufficient_scoring_actions"
            and not (report.get("integrity") or {}).get("action_metrics_trusted", False)):
        return {
            "reason": (
                "Scoring needs to know which strikes landed, and WarriorIQ cannot tell that "
                "reliably yet, so no strike is counted towards a score."
            ),
            "fix": "Nothing to redo - the movement, guard and balance numbers below are measured and real.",
            "disclaimer": (
                "No score is shown. A score needs to know which strikes landed, and WarriorIQ "
                "cannot tell that reliably yet, so no strike is counted towards one. Movement, "
                "guard, balance and coverage below are unaffected."
            ),
        }
    if status == "insufficient_observation_coverage":
        required = f"{SETTINGS.min_tracking_coverage_for_score * 100:.0f}%"
        return {
            "reason": (
                f"We lost sight of a fighter too often. We followed Fighter A for "
                f"{_pct('fighter_A_coverage')} of the fight and Fighter B for "
                f"{_pct('fighter_B_coverage')}, and a fair score needs {required} of each."
            ),
            # When the recording itself is the cause - a 220-pixel copy sent
            # through a messaging app - re-picking the fighters changes nothing.
            # The preflight check already says what would; it was only shown
            # in the downloadable report, and here the page told a fighter who
            # had picked two people standing well apart to pick them apart.
            "fix": (((report.get("tracking") or {}).get("recording") or {}).get("advice") or [None])[0]
                   or "Pick both fighters again on a frame where they are clearly apart, then re-run.",
        }
    if status == "insufficient_scoring_actions":
        counted = (scorecard.get("evidence") or {}).get("scoring_action_candidates")
        seen = f"Only {counted} were clear enough." if counted else "None were clear enough."
        return {
            "reason": f"Too few strikes could be counted with confidence. {seen}",
            "fix": "Footage shot closer, steadier or from the side usually reads far better.",
        }
    return {
        "reason": "Tracking was not steady enough for a fair score.",
        "fix": "Pick both fighters again on a clearer frame, then re-run.",
    }


def _pin_sport_to_fight(request: Request, sport: str | None) -> None:
    """Name the fight's own sport in the header on a page about one fight.

    The chip came from a cookie that /analyze/<sport> writes, so opening the
    Muay Thai setup in another tab relabelled a kickboxing result "Muay Thai".
    The cookie is a remembered preference for the analysis flow; a page about
    one fight is about that fight's sport, whatever the last tab chose.
    """
    if sport in SPORTS:
        request.state.active_sport = sport_identity(sport)


def _job_sport(job: dict) -> str | None:
    try:
        return sport_of(str(job.get("ruleset") or ""))
    except (KeyError, ValueError):
        return None


def _fighter_names(request: Request, job: dict, report: dict) -> dict:
    """What the report calls each side.

    The fighter the upload was filed under (the roster entry chosen on the
    setup page) is the report's focus: the selection page asks "which one is
    <name>?" and that answer is the focus fighter. So that side carries the
    name, and the other side is the opponent. The report said "Fighter A /
    Fighter B" while the library already showed the name (QA, 2026-10-04).
    Without a filed fighter, the letters stay.
    """
    names = {"A": "Fighter A", "B": "Fighter B"}
    profile_id = _profile_id(request)
    if profile_id is None or not str(job.get("fighter_id") or "").isdigit():
        return names
    fighter = get_fighter(profile_id, int(job["fighter_id"]))
    name = " ".join(str((fighter or {}).get("name") or "").split())
    if not name:
        return names
    video = report.get("video") or {}
    focus = str(job.get("focus_fighter") or video.get("focus_fighter") or "A")
    if focus not in names:
        return names
    names[focus] = name
    names["B" if focus == "A" else "A"] = "Opponent"
    return names


def _name_the_fighters(report: dict, names: dict) -> None:
    """Put the names into the coaching text, which was written as "Fighter A".

    core/coaching.py writes its sentences at analysis time with the letters;
    the page swaps them for what _fighter_names calls each side. Only the
    coaching and training text - nothing that is matched or stored - and never
    written back.
    """
    swaps = {f"Fighter {side}": name for side, name in names.items() if name != f"Fighter {side}"}
    if not swaps:
        return

    def rename(value):
        if isinstance(value, str):
            for letter, name in swaps.items():
                value = value.replace(letter, name)
            return value
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    for key in ("coaching", "training_plan", "training_progression"):
        if key in report:
            report[key] = rename(report[key])


def _corner_labels(job: dict) -> dict:
    """The corner each fighter was in, as the person who drew the boxes said.

    This was hard-coded - Fighter A was always "Red corner" - and A is simply
    whoever was boxed first, so on a WAKO clip where the blue fighter was
    drawn first the report put the wrong colour over both of them. A corner
    nobody stated is not named at all.
    """
    corner = str(job.get("fighter_a_corner") or "").lower()
    if corner not in {"red", "blue"}:
        return {"A": None, "B": None}
    other = "blue" if corner == "red" else "red"
    return {"A": f"{corner.title()} corner", "B": f"{other.title()} corner"}


def _sharing_state(request: Request, job_id: str, profile_id: int | None) -> dict:
    """What the athlete can see about their coach links.

    Creating a link is the only moment its address exists in readable form, so
    that one arrives as a query parameter and is shown once. Everything after
    that is a count and an expiry date, which is what makes revoking visible:
    without it the button redirected to an identical page and looked dead.
    """
    empty = {"links": [], "new_link": None, "new_link_expires": None, "revoked": None}
    if profile_id is None:
        return empty
    links = [dict(link) for link in list_active_report_shares(job_id, profile_id)]
    for link in links:
        link["expires_label"] = _friendly_date(link.get("expires_at"))
    token = (request.query_params.get("share") or "").strip()
    new_link = f"{_public_base(request)}/s/{token}" if token else None
    revoked_raw = request.query_params.get("revoked")
    try:
        revoked = int(revoked_raw) if revoked_raw is not None else None
    except ValueError:
        revoked = None
    return {
        "links": links,
        "new_link": new_link,
        "new_link_expires": links[-1]["expires_label"] if (new_link and links) else None,
        "revoked": revoked,
    }


def _friendly_date(value: str | None) -> str:
    """An ISO timestamp as a date a person would say out loud."""
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y")
    except ValueError:
        return ""


def _require_completed_artifact(job_id: str, name: str) -> Path:
    directory = completed_artifact_directory(job_id)
    if directory is None or not (directory / name).is_file():
        raise HTTPException(404, "This analysis has no completed result yet.")
    return directory / name


def _score_and_identity_as_shown(report: dict) -> None:
    """Decide, on read, whether the score may be shown and whose numbers are whose.

    The result page's rules, kept in one place because the public fight link
    (story_page) must show exactly what the owner's own page shows.
    """
    coverage_ok = min(float(report.get("tracking", {}).get("fighter_A_coverage", 0)), float(report.get("tracking", {}).get("fighter_B_coverage", 0))) >= SETTINGS.min_tracking_coverage_for_score
    report.setdefault("scorecard", {})["available"] = bool(report.get("scorecard", {}).get("available", coverage_ok) and coverage_ok)
    if report.get("video", {}).get("analysis_target", "BOTH") != "BOTH":
        report["scorecard"]["available"] = False
        report["scorecard"]["disclaimer"] = "To receive an estimated scorecard, choose Analyze both fighters. A one-fighter analysis does not count the opponent's points."
    # Customer reports are fully automatic. Human annotations remain isolated
    # in the model-validation lab and never become required report work.
    _apply_report_annotations(report, [])
    refresh_identity_integrity(report)


@app.get("/result/{job_id}", response_class=HTMLResponse)
def result_page(request: Request, job_id: str):
    job = _authorized_job(request, job_id)
    if not job:
        raise HTTPException(404)
    if job.get("history_saved") is False:
        persist_completed_job(job_id, str(job.get("analysis_run_id") or ""))
    path = _require_completed_artifact(job_id, "report.json")
    if not path.exists():
        raise HTTPException(404)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") == "solo":
        # One person, no opponent and no strikes (core/solo.py): its own page,
        # since every section of the fight report is about two fighters.
        solo_name = _fighter_names(request, job, report)["A"]
        return templates.TemplateResponse(request=request, name="solo_result.html", context={
            "request": request, "job_id": job_id, "report": report,
            "analysed_span": _analysed_span_summary(report),
            "analysis_build": result_check(report),
            "subject_name": None if solo_name == "Fighter A" else solo_name,
        })
    if "key_moments" not in report:
        report["key_moments"] = [e for e in report.get("events", []) if e.get("outcome") in {"clean", "likely_landed"} and float(e.get("confidence", 0)) >= .72 and float(e.get("contact_confidence", 0)) >= .62][:18]
    _score_and_identity_as_shown(report)
    report_access = _request_plan(request)
    # Opening the exact completed report acknowledges its one-time notification.
    # Without this reset, the green "Results ready" chip was written back on
    # every report view and could appear indefinitely before a new analysis.
    displayed = request.state.active_analysis
    if displayed and displayed.get("job_id") == job_id and displayed.get("status") == "complete":
        request.state.analysis_navigation["last_completed"] = None
        request.state.analysis_navigation["display"] = request.state.analysis_navigation.get("active")
        request.state.active_analysis = request.state.analysis_navigation.get("active")
    # The athlete's half of the split: one comparison against their last fight,
    # so the page answers "am I better than last time" without becoming the
    # coach's squad view. See core/squad.py.
    _profile = _profile_id(request)
    progress_since_last = (
        compare_with_previous(report, list_fights(_profile), job_id)
        if _profile is not None else {"available": False}
    )
    can_share = bool(_account(request) and report_access.get("can_share"))
    # Every per-fighter strike block is an attribution. When identity failed
    # the page has already said it cannot tell whose strikes were whose, so
    # the per-fighter kick cards, the "You hit / You got hit" visuals and the
    # kick-minimum table are not built at all, and one unattributed total is
    # given instead. See core.report.unattributed_kick_total.
    identity_trusted = bool((report.get("integrity") or {}).get("identity_evidence_trusted", True))
    numbers = _numbers_state(report)
    # A coach link is allowed whatever the numbers' state: the coach's page
    # (shared.html) withholds unverified or absent numbers and says why, so the
    # link carries the explanation with it. Only the public story card, which
    # posts numbers as the fighter's own, stays off until they are verified.
    _pin_sport_to_fight(request, report.get("scorecard", {}).get("sport") or _job_sport(job))
    _estimate_score_withheld_for_punches(report)
    _withhold_score_while_counts_are_off(report)
    names = _fighter_names(request, job, report)
    _name_the_fighters(report, names)
    score_withheld = _score_withheld(report, job_id)
    # The scorecard box prints the disclaimer stored when the fight was
    # analysed. For a score withheld because strike counting is not validated,
    # that stored text could say "this analysis verified 4" and point at an
    # "action timeline below" the page no longer shows, beside a reason at the
    # top saying no strike is counted. The current wording replaces it, so
    # reports analysed before this change say the same thing top and bottom.
    if score_withheld and score_withheld.get("disclaimer"):
        report["scorecard"]["disclaimer"] = score_withheld["disclaimer"]
    # The sport panel's counting sentence was stored with the report, so
    # reports written under an older policy kept saying "punches are not
    # shown" beside punch counts. It is re-read from the one policy instead.
    for coaching in (report.get("sport_coaching") or {}).values():
        if isinstance(coaching, dict) and coaching.get("sport"):
            coaching["report_frame"] = counting_policy(coaching["sport"]).report_frame
    # Written at analysis time too, and the stored copy blamed the whole sport
    # for a ruleset-only gap; see core.scoring.coverage_note.
    try:
        report["scorecard"]["coverage_note"] = coverage_note(report["scorecard"]["ruleset"])
    except (KeyError, ValueError):
        pass
    response = templates.TemplateResponse(request=request, name="result.html", context={
        "request": request, "job_id": job_id, "report": report,
        "corners": _corner_labels(job),
        "names": names,
        "analysed_span": _analysed_span_summary(report),
        "analysis_build": result_check(report),
        "fight_footage": _fight_footage_summary(report),
        "numbers": numbers,
        "camera_lost": _identity_lost_to_camera(report),
        "identity_failure": _identity_failure(report),
        # "Punches and knees are not counted" on taekwondo, which awards no
        # knees - the note names only what this sport actually scores.
        "withheld_families": _reported_strike_families(
            (report.get("scorecard") or {}).get("sport") or _job_sport(job) or "kickboxing"
        )["withheld_families"],
        "progress_since_last": progress_since_last,
        "moment_count": len(coaching_moments(
            report, _moment_fighters(report),
            str(report_access.get("report_tier") or ""), report_access.get("coaching_items"))),
        "identity": sport_identity(report.get("scorecard", {}).get("sport", "kickboxing")),
        "report_access": report_access,
        "analysis_quality": _analysis_quality_summary(report),
        # Built at render time rather than stored in the report, so every
        # analysis already on disk gains these sections without being re-run.
        "visuals": (report_visuals(
            report, _visual_focus(report),
            outcomes_counted=(bool((report.get("statistics") or {}).get("action_labels_available"))
                              and STRIKE_COUNTS_PUBLISHED),
        ) if identity_trusted else None),
        "can_share": can_share,
        "sharing": _sharing_state(request, job_id, _profile) if can_share else None,
        "score_withheld": score_withheld,
        "observed": (observed_summary(report)
                     if identity_trusted and not (report.get("scorecard") or {}).get("available")
                     else None),
        "kick_total": None if identity_trusted else unattributed_kick_total(report),
        "strike_counts_published": STRIKE_COUNTS_PUBLISHED,
        # Attributions, like the per-fighter cards, so only when identity held.
        "counted_strikes": (_with_checks(job_id, _counted_strikes(report, published_families(
            (report.get("scorecard") or {}).get("sport") or _job_sport(job), report)))
            if identity_trusted and STRIKE_COUNTS_PUBLISHED else []),
        "can_check_strikes": bool(_account(request)),
        "went_down": _went_down(job_id, report),
        "wrong_sport": _wrong_sport(report, job),
        # Stats-only story card for Instagram, TikTok, WhatsApp and the rest:
        # an image made in the browser, no link and no video, so every plan.
        "share_card": share_card(report) if _account(request) else None,
        # Why there is no card, so the button can say so instead of vanishing.
        "share_card_missing": (
            "account" if not _account(request)
            else "identity" if not identity_trusted
            else "no_fight" if numbers["state"] == "no_fight"
            else "stats"),
        # The fight's live public links, one per fighter (story_page).
        "story_links": [
            {"side": link["side"], "name": link["name"], "url": f"{_public_base(request)}/f/{link['token']}",
             "on_profile": bool(link.get("on_profile_at"))}
            for link in (list_story_shares(job_id, _profile) if _profile is not None and _account(request) else [])
        ],
        "went_down_note": (report.get("went_down") or {}).get("note"),
        "estimate_note": counting_policy(
            (report.get("scorecard") or {}).get("sport") or _job_sport(job)).estimate_note,
        "families_shown": published_families(
            (report.get("scorecard") or {}).get("sport") or _job_sport(job), report),
        # Only Full Contact has an obligatory kick count, so this is None for
        # every other discipline and the block simply does not render.
        "kick_minimum": kick_minimum_check(report) if identity_trusted else None,
    })
    response.delete_cookie(LAST_COMPLETED_ANALYSIS_COOKIE, httponly=True, samesite="lax")
    if request.cookies.get(ACTIVE_ANALYSIS_COOKIE) == job_id:
        response.delete_cookie(ACTIVE_ANALYSIS_COOKIE, httponly=True, samesite="lax")
    return response


@app.get("/result/{job_id}/report.json", include_in_schema=False)
def download_report_json(request: Request, job_id: str):
    """Hand the owner their own analysis, as the file the analyser wrote.

    Everything measured about a fight already exists as one JSON document on
    disk - it is what the report page renders from - and there was no way to
    get a copy of it. That matters in three ways, in increasing order of how
    often it comes up:

      * it is the reader's own data, and an account export exists for the
        account but not for the analyses inside it;
      * a coach who wants the numbers in a spreadsheet has to retype them off
        a page that deliberately shows fewer of them than it holds;
      * and when an analysis comes out wrong, the report is the evidence. Not
        being able to send it means describing it instead, which is how
        "tracking was bad on the red corner" ends up standing in for four
        hundred frames of measurement.

    Owner only, through the same check the report page uses, and no-store:
    this is somebody's footage measured, not a public document.
    """
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    path = _require_completed_artifact(job_id, "report.json")
    if not path.exists():
        raise HTTPException(404, "Fight report not found")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(404, "Fight report not found") from None
    return Response(
        json.dumps(_without_hardware(report), indent=1), media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="warrioriq-{job_id}.json"',
            "Cache-Control": "no-store",
        },
    )


def _without_hardware(report: dict) -> dict:
    """The owner's report without the machine it ran on.

    Which graphics card analysed a fight, and where its model files sit on
    that machine, is operations detail: it stays in the worker's logs and the
    stored report, and is not handed out. Only whether a graphics card was used
    is kept, because that is what explains the speed.
    """
    performance = report.get("performance")
    if isinstance(performance, dict):
        gpu = performance.pop("gpu", None)
        if gpu is not None:
            performance["compute"] = "processor" if str(gpu).upper() == "CPU" else "graphics card"
        model = performance.get("pose_model")
        if isinstance(model, str):
            performance["pose_model"] = Path(model).name
        performance.pop("vram_free_at_start", None)
    return report


@app.post("/api/sport-check", dependencies=[Depends(require_csrf)])
def check_sport(request: Request, payload: SportCheckPayload):
    """Name the sport in three frames from the chosen video, before upload.

    Advisory and off unless a provider is configured (core/sport_check.py).
    The frames are forwarded to that provider and never stored here.
    """
    if sport_check.provider() is None:
        return {"available": False}
    if not _account(request):
        raise HTTPException(401, "Sign in first")
    _enforce_rate_limit(request, "sport_check", 20, 600)
    if payload.sport not in SPORTS:
        raise HTTPException(400, "Unknown sport")
    try:
        images = sport_check.decode_frames(payload.frames)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, "Invalid frames") from exc
    result = sport_check.verdict(payload.sport, sport_check.detect_sport(images))
    if result.get("detected"):
        result["detected_label"] = RULESET_SPORTS[result["detected"]]
        result["chosen_label"] = RULESET_SPORTS[payload.sport]
    return {"available": True, **result}


@app.post("/api/strike-check/{job_id}", dependencies=[Depends(require_csrf)])
def check_counted_strike(request: Request, job_id: str, payload: StrikeCheckPayload):
    """One tap on a counted strike: right, not a strike, or another type.

    Stored in the annotations table beside full corrections - one row per
    fight and moment, no new schema - marked source "strike_check" so it can
    be told apart. Open to any signed-in owner of the fight rather than only
    correcting plans: these answers are what grows the labelled benchmark
    (tools/export_strike_checks.py), and a free user's answer is as true as
    a paying one's. Nothing is exported for training here; that still needs
    the full correction and the owner's training consent.
    """
    _enforce_rate_limit(request, "strike_checks", 300, 300)
    if not _account(request) or not _authorized_job(request, job_id):
        raise HTTPException(403, "Sign in to check your own fight.")
    fighter = payload.fighter.upper()
    family = payload.family.lower()
    verdict = payload.verdict.lower()
    if (not math.isfinite(payload.seconds) or payload.seconds < 0
            or fighter not in {"A", "B"} or family not in {"punch", "kick", "knee"}
            or verdict not in STRIKE_CHECK_VERDICTS):
        raise HTTPException(400, "Invalid answer")
    report_path = _require_completed_artifact(job_id, "report.json")
    if not report_path.exists():
        raise HTTPException(404, "Fight report not found")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counted = [row for row in _counted_strikes(report, (family,))
               if row["fighter"] == fighter and abs(row["seconds"] - payload.seconds) <= 0.05]
    if not counted:
        raise HTTPException(404, "No counted strike at that moment")
    seconds = round(counted[0]["seconds"], 3)
    if verdict == "right":
        truth = family
    elif verdict in {"punch", "kick", "knee"}:
        truth = verdict
    else:
        truth = "none"
    predicted = {"fighter": fighter, "family": family, "technique": family}
    corrected = {"fighter": fighter, "family": truth, "technique": truth,
                 "verdict": verdict, "source": "strike_check"}
    save_annotation(job_id, seconds, report.get("setup", {}).get("ruleset", "K1"), predicted, corrected)
    return {"saved": True, "seconds": seconds, "verdict": verdict}


def _strike_checks(job_id: str) -> dict[tuple[str, float], str]:
    """This fight's one-tap answers, keyed by (fighter, seconds)."""
    out = {}
    for item in get_annotations(job_id):
        corrected = item.get("corrected") or {}
        if corrected.get("source") == "strike_check":
            out[(corrected.get("fighter"), round(float(item["event_time"]), 3))] = corrected.get("verdict")
    return out


@app.post("/api/down-check/{job_id}", dependencies=[Depends(require_csrf)])
def check_went_down(request: Request, job_id: str, payload: DownCheckPayload):
    """One tap on a moment someone went down: fighter A, fighter B, or nobody.

    The analysis finds these moments but cannot yet tell who went down (see
    core/ground.py), so the owner's answer is the only attribution there is.
    Stored like a strike check, marked source "down_check".
    """
    _enforce_rate_limit(request, "down_checks", 120, 300)
    if not _account(request) or not _authorized_job(request, job_id):
        raise HTTPException(403, "Sign in to check your own fight.")
    verdict = payload.verdict if payload.verdict in {"A", "B"} else payload.verdict.lower()
    if not math.isfinite(payload.seconds) or payload.seconds < 0 or verdict not in DOWN_CHECK_VERDICTS:
        raise HTTPException(400, "Invalid answer")
    report_path = _require_completed_artifact(job_id, "report.json")
    if not report_path.exists():
        raise HTTPException(404, "Fight report not found")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    found = [row for row in _went_down(job_id, report) if abs(row["seconds"] - payload.seconds) <= 0.05]
    if not found:
        raise HTTPException(404, "No moment at that time")
    seconds = round(found[0]["seconds"], 3)
    save_annotation(job_id, seconds + DOWN_CHECK_TIME_OFFSET, report.get("setup", {}).get("ruleset", "K1"),
                    {"event": "went_down"}, {"went_down": verdict, "source": "down_check"})
    return {"saved": True, "seconds": seconds, "verdict": verdict}


def _down_checks(job_id: str) -> dict[float, str]:
    """This fight's answers on who went down, keyed by the moment's seconds."""
    out = {}
    for item in get_annotations(job_id):
        if is_down_check(item):
            out[round(float(item["event_time"]) - DOWN_CHECK_TIME_OFFSET, 3)] = item["corrected"].get("went_down")
    return out


@app.post("/api/annotations/{job_id}", dependencies=[Depends(require_csrf)])
def annotate_event(request: Request, job_id: str, payload: AnnotationPayload):
    # Every correction writes an .npz sequence to disk, so this is the one
    # authenticated endpoint that grows storage without an upload. The cap
    # is deliberately high: labelling a round legitimately means dozens of
    # saves in a few minutes, and blocking that would cost real data.
    _enforce_rate_limit(request, "annotations", 300, 300)
    if not _account(request) or not _authorized_job(request, job_id) or not _request_plan(request).get("can_correct"):
        raise HTTPException(403, "Evidence corrections are available with a complete-report plan.")
    report_path = _require_completed_artifact(job_id, "report.json")
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
    sequence_path = export_sequence(
        job_id, annotation_id, corrected, contact_time,
        tracking_path=report_path.parent / "tracking.jsonl",
        source_fight_id=(get_job(job_id) or {}).get("source_video_sha256"),
    ) if training_consent else None
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
    # Highest confidence first. The page shows 40 at a time, and chronological
    # order made that a random-quality sample - on a real session 27 of 30 were
    # rejected. Confidence only became worth sorting on once it stopped
    # saturating at 0.94 for nearly every candidate; now it spreads across
    # 0.44-0.91, so the strongest hypotheses get the labelling time.
    def _rank(item: dict) -> float:
        # A null confidence is not a zero-confidence event, it is an event
        # nobody scored - a manually added label carries no confidence at all.
        # float(None) raised here and took the whole labelling page with it.
        total = 0.0
        for key in ("confidence", "contact_confidence"):
            try:
                value = item.get(key)
                total += float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                continue
        return total

    selected.sort(key=_rank, reverse=True)
    return selected


@app.get("/review/{job_id}", response_class=HTMLResponse)
def review_evidence_page(request: Request, job_id: str, page: int = 1, mode: str = "scorecard"):
    if not _account(request) or not _authorized_job(request, job_id) or not _request_plan(request).get("can_correct"):
        raise HTTPException(403, "Evidence review is available with a complete-report plan.")
    report_path = _require_completed_artifact(job_id, "report.json")
    if not report_path.exists():
        raise HTTPException(404)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    refresh_identity_integrity(report)
    if not report.get("integrity", {}).get("identity_evidence_trusted", True):
        return RedirectResponse(f"/select/{job_id}", status_code=303)
    mode = "dataset" if mode == "dataset" else "scorecard"
    all_candidates = _review_candidates(report, mode)
    annotations = [item for item in get_annotations(job_id) if not is_down_check(item)]
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


@app.post("/review/{job_id}/complete", dependencies=[Depends(require_csrf)])
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
        report_path = _require_completed_artifact(job_id, "report.json")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        refresh_identity_integrity(report)
        if not report.get("integrity", {}).get("identity_evidence_trusted", True):
            return RedirectResponse(f"/select/{job_id}", status_code=303)
        candidates = _review_candidates(report, mode)
        reviewed_times = [float(item["event_time"]) for item in get_annotations(job_id)
                          if not is_down_check(item)]
        remaining = [item for item in candidates if not any(abs(float(item["peak_time"]) - value) <= .02 for value in reviewed_times)]
        if remaining:
            raise HTTPException(409, f"Review the remaining {count_of(len(remaining), 'candidate')} before completing the fight.")
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
    """Model-validation counts, for whoever is building this.

    Not linked from anywhere in the site. It reports label counts, dataset
    splits and whether the action model is release ready - true things, and
    meaningless to a fighter, who reads "0 labels, not release ready" as a
    verdict on the product they are being asked to pay for. Signing in is the
    gate rather than an admin role, because no such role exists here and
    inventing one to hide a page would be the wrong way round.
    """
    if not _account(request):
        raise HTTPException(404)
    profile_id = _profile_id(request)
    owned_jobs = {fight["job_id"] for fight in list_fights(profile_id)} if profile_id is not None else set()
    annotations = [item for item in list_annotations()
                   if item["job_id"] in owned_jobs and not is_strike_check(item) and not is_down_check(item)]
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
    # strict, because these two must stay the same length: adding a fifth
    # label and forgetting a fifth fraction would silently drop a chapter
    # rather than say anything.
    for label, fraction in zip(labels, fractions, strict=True):
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
    path = _require_completed_artifact(job_id, "report.json")
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
    sport = (report.get("scorecard") or {}).get("sport") or _job_sport(_authorized_job(request, job_id) or {})
    policy = counting_policy(sport)
    if replay_mode == "movement_chapters" and identity_safe and STRIKE_COUNTS_PUBLISHED:
        # The report lists every counted strike ("Watch every counted strike")
        # while this page said no action had passed. The same list is offered
        # here, labelled as what it is: automatic counts, estimates.
        counted = _counted_strikes(report, published_families(sport, report))
        if fighter and fighter.upper() in {"A", "B"}:
            counted = [row for row in counted if row["fighter"] == fighter.upper()]
        if family and family.lower() in {"punch", "kick", "knee"}:
            counted = [row for row in counted if row["family"] == family.lower()]
        if counted:
            replay_chapters = [{
                "time": round(row["seconds"], 3), "lead_seconds": 1.0,
                "label": f"{row['clock']} · Fighter {row['fighter']} · {row['family'].title()}",
                "detail": "Counted automatically (estimate)", "kind": "counted_strike",
            } for row in counted[:200]]
            replay_mode = "counted_strikes"
    _pin_sport_to_fight(request, (report.get("scorecard") or {}).get("sport"))
    plan = _request_plan(request)
    return templates.TemplateResponse(
        request=request,
        name="replay.html",
        context={
            "request": request, "job_id": job_id, "report": report,
            "identity_safe": identity_safe,
            "replay_chapters": replay_chapters,
            "replay_mode": replay_mode,
            # The report's coaching points that carry moments, as cards beside
            # the skeleton video. Same identity and plan rules as the report.
            "moments": coaching_moments(report, _moment_fighters(report),
                                        str(plan.get("report_tier") or ""), plan.get("coaching_items")),
            "names": _fighter_names(request, _authorized_job(request, job_id) or {}, report),
            "counting_policy": policy.as_dict(),
            "evidence_filter": " · ".join(
                value.replace("_", " ").title()
                for value in (fighter, family, outcome) if value
            ),
        },
    )


def _tracking_response_chunks(path: Path):
    """Keep server memory bounded instead of rebuilding an entire replay array."""
    yield b'{"frames":['
    chunks = []
    size = 0
    first = True
    with path.open("rb") as handle:
        for line in handle:
            try:
                json.loads(line)
            except (ValueError, UnicodeError):
                continue
            chunk = (b"" if first else b",") + line.strip()
            first = False
            chunks.append(chunk)
            size += len(chunk)
            if size >= 64 * 1024:
                yield b"".join(chunks)
                chunks.clear()
                size = 0
    if chunks:
        yield b"".join(chunks)
    yield b"]}"


@app.get("/api/tracking/{job_id}")
def tracking_data(request: Request, job_id: str, run: str | None = None):
    if not _authorized_job(request, job_id):
        raise HTTPException(404)
    path = _require_completed_artifact(job_id, "tracking.jsonl")
    if run and path.parent.name != run:
        raise HTTPException(409, "This fight has a newer analysis. Open its current report to replay it.")
    return StreamingResponse(_tracking_response_chunks(path), media_type="application/json")


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
    # A QuickTime upload has a browser-friendly copy beside it; the original
    # stays on disk because reprocessing reads it, and is what the replay
    # page offers for download when playback fails.
    path = playback_file(path)
    # no-store meant every seek re-downloaded the whole file. Measured at about
    # 460 KB/s, a 101 MB replay is three and a half minutes, paid again on every
    # scrub and every revisit - on footage the viewer has already been sent once.
    #
    # `private` keeps it out of shared caches; the URL is account-scoped by
    # _authorized_job above and the prefix is robots-disallowed, so the copy
    # lives in the one browser that was already allowed to see it.
    #
    # Set here rather than in the middleware, which uses setdefault: this wins,
    # while a 404 on the same prefix still falls through to no-store.
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@app.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, error: str = ""):
    profile_id = _profile_id(request)
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "request": request, "error": error[:200],
            "profile": get_profile(profile_id) if profile_id is not None else None,
            "fights": list_fights(profile_id) if profile_id is not None else [],
            "minor": social.is_minor_account(_account(request)),
            "follow_requests": list_follows(profile_id, direction="followers", status="pending") if profile_id is not None else [],
            "followers": list_follows(profile_id, direction="followers") if profile_id is not None else [],
            "following": list_follows(profile_id, direction="following") if profile_id is not None else [],
        },
    )


# ---------------------------------------------------------------------------
# Public athlete profiles and follows. The rules live in core/social.py.
# ---------------------------------------------------------------------------

def _social_target(handle: str) -> tuple[dict, bool]:
    profile = get_profile_by_handle(handle.strip().lstrip("@").lower()) if handle else None
    if not profile:
        raise HTTPException(404)
    return profile, social.is_minor_account(get_account_by_profile(int(profile["id"])))


@app.post("/profile/public", dependencies=[Depends(require_csrf)])
def save_public_profile(request: Request, handle: str = Form(""), visibility: str = Form("private"),
                        bio: str = Form(""), gym: str = Form("")):
    profile_id = _profile_id(request)
    account = _account(request)
    if profile_id is None or not account:
        return RedirectResponse("/login?next=/profile", status_code=303)
    try:
        clean = social.normalize_handle(handle)
    except ValueError as reason:
        return RedirectResponse("/profile?" + urlencode({"error": str(reason)}) + "#public-profile", status_code=303)
    minor = social.is_minor_account(get_account(int(account["id"])) or account)
    wanted = visibility if visibility in social.VISIBILITIES else "private"
    if wanted == "public" and (minor or not clean):
        wanted = "private"
    try:
        set_public_profile(profile_id, clean, wanted, " ".join(bio.split())[:280], " ".join(gym.split())[:80])
    except HandleTaken:
        return RedirectResponse("/profile?" + urlencode({"error": "That username is taken. Choose another."})
                                + "#public-profile", status_code=303)
    record_security_event("profile_visibility_changed", account_id=int(account["id"]),
                          resource_type="profile", resource_id=str(profile_id), metadata={"visibility": wanted})
    return RedirectResponse("/profile#public-profile", status_code=303)


@app.get("/athlete/{handle}", response_class=HTMLResponse)
def athlete_page(request: Request, handle: str):
    profile, minor = _social_target(handle)
    viewer = _profile_id(request)
    if not social.is_discoverable(viewer_profile_id=viewer, profile=profile, minor=minor):
        raise HTTPException(404)
    follow = follow_status(viewer, int(profile["id"])) if viewer is not None else None
    visible = social.can_view(viewer_profile_id=viewer, profile=profile, minor=minor, follow=follow)
    standing = None
    fights_analysed = 0
    posts = []
    if visible:
        # Fights the athlete posted: their public stats-only links (story_page),
        # so nothing here is more than anyone holding the link already sees.
        for share in list_profile_posts(int(profile["id"]), limit=9):
            card = _story_card(share["job_id"])
            if card is None:
                continue
            me = card["fighters"][share["side"]]
            posts.append({"url": f"/f/{share['token']}", "image": f"/f/{share['token']}/card.png",
                          "sport": card["sport"], "total": me.get("total"), "posted_at": share["on_profile_at"],
                          "job_id": share["job_id"], "side": share["side"]})
        standing = camp_standing(list_points(int(profile["id"])), list_training_sessions(int(profile["id"])),
                                 datetime.now(timezone.utc).date())
        fights_analysed = len(list_fights(int(profile["id"])))
    return templates.TemplateResponse(
        request=request, name="athlete.html",
        context={
            "request": request, "athlete": profile, "visible": visible, "minor": minor,
            "is_owner": viewer is not None and int(viewer) == int(profile["id"]),
            "follow": follow, "can_follow": social.can_follow(viewer_profile_id=viewer, profile=profile, minor=minor),
            "signed_in": viewer is not None, "standing": standing, "fights_analysed": fights_analysed,
            "posts": posts,
            "counts": count_follows(int(profile["id"])),
            "public": social.effective_visibility(profile, minor) == "public",
            "reported": request.query_params.get("reported") == "1",
        },
    )


FEED_ITEMS = 40
FEED_POSTS_PER_ATHLETE = 5


@app.get("/feed", response_class=HTMLResponse)
def feed_page(request: Request):
    """What the athletes this person follows have done, newest first.

    Only accepted follows, and each athlete's page rules are applied again
    here (core/social.py), so the feed never shows more than their profile
    would: a profile made private keeps its approved followers, and a minor's
    is shown to nobody.
    """
    viewer = _profile_id(request)
    if viewer is None:
        return RedirectResponse("/login?next=/feed", status_code=303)
    items = []
    following = list_follows(viewer, direction="following")
    for person in following:
        profile = get_profile(int(person["id"])) or {}
        if not profile.get("handle"):
            continue
        minor = social.is_minor_account(get_account_by_profile(int(profile["id"])))
        if not social.can_view(viewer_profile_id=viewer, profile=profile, minor=minor, follow="accepted"):
            continue
        who = {"name": profile.get("display_name") or profile["handle"], "handle": profile["handle"],
               "photo": profile.get("photo_path")}
        for share in list_profile_posts(int(profile["id"]), limit=FEED_POSTS_PER_ATHLETE):
            card = _story_card(share["job_id"])
            if card is not None:
                items.append({"kind": "post", "at": share["on_profile_at"], "who": who, "sport": card["sport"],
                              "url": f"/f/{share['token']}", "image": f"/f/{share['token']}/card.png"})
        for item in feed.level_ups(list_points(int(profile["id"]))):
            items.append({**item, "who": who})
        for item in feed.streak_milestones(list_training_sessions(int(profile["id"]))):
            items.append({**item, "who": who})
    return templates.TemplateResponse(
        request=request, name="feed.html",
        context={"request": request, "items": feed.newest_first(items, FEED_ITEMS),
                 "following_count": len(following),
                 "own_handle": (get_profile(viewer) or {}).get("handle")},
    )


@app.post("/athlete/{handle}/follow", dependencies=[Depends(require_csrf)])
def follow_athlete(request: Request, handle: str):
    viewer = _profile_id(request)
    if viewer is None:
        return RedirectResponse(f"/login?next=/athlete/{quote(handle)}", status_code=303)
    _enforce_rate_limit(request, "follow", 60, 3600)
    profile, minor = _social_target(handle)
    if not social.can_follow(viewer_profile_id=viewer, profile=profile, minor=minor):
        raise HTTPException(404)
    request_follow(viewer, int(profile["id"]), accepted=social.effective_visibility(profile, minor) == "public")
    return RedirectResponse(f"/athlete/{quote(profile['handle'])}", status_code=303)


@app.post("/athlete/{handle}/unfollow", dependencies=[Depends(require_csrf)])
def unfollow_athlete(request: Request, handle: str):
    viewer = _profile_id(request)
    if viewer is None:
        return RedirectResponse("/login", status_code=303)
    profile, _ = _social_target(handle)
    remove_follow(viewer, int(profile["id"]))
    return RedirectResponse(f"/athlete/{quote(profile['handle'])}", status_code=303)


@app.post("/profile/followers/{follower_id}/{action}", dependencies=[Depends(require_csrf)])
def answer_follower(request: Request, follower_id: int, action: str):
    """Approve or decline a follow request, or remove an existing follower."""
    owner = _profile_id(request)
    if owner is None:
        return RedirectResponse("/login?next=/profile", status_code=303)
    if action == "approve":
        accept_follow(follower_id, owner)
    elif action in {"decline", "remove"}:
        remove_follow(follower_id, owner)
    else:
        raise HTTPException(404)
    return RedirectResponse("/profile#followers", status_code=303)


@app.post("/athlete/{handle}/report", dependencies=[Depends(require_csrf)])
def report_athlete(request: Request, handle: str, reason: str = Form("")):
    """Anyone can report a profile; it lands in the admin moderation queue."""
    _enforce_rate_limit(request, "profile-report", 10, 3600)
    profile, _ = _social_target(handle)
    account = _account(request)
    report_id = create_moderation_report(
        "profile", (account or {}).get("email", "anonymous"), " ".join(reason.split())[:2000] or "No reason given",
        f"profile:{profile['id']}:@{profile['handle']}")
    record_security_event("profile_reported", resource_type="moderation_report", resource_id=str(report_id))
    return RedirectResponse(f"/athlete/{quote(profile['handle'])}?reported=1", status_code=303)


@app.get("/settings")
def settings_root(request: Request):
    return RedirectResponse("/settings/privacy", status_code=303)


@app.get("/settings/{section}", response_class=HTMLResponse)
def settings_page(request: Request, section: str, notice: str = "", error: str = ""):
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
            "error": error[:200],
            "account": account,
            "profile": get_profile(int(account["profile_id"])) or {},
            "fights": list_fights(int(account["profile_id"])),
            "video_retention_days": SETTINGS.saved_video_retention_days,
            "plan": plan_for_key(account.get("plan_override") or account.get("plan")),
            "subscription_actions": list_subscription_actions(int(account["id"])),
        },
    )


@app.post("/settings/marketing", dependencies=[Depends(require_csrf)])
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


@app.post("/settings/sessions/revoke", dependencies=[Depends(require_csrf)])
def revoke_other_sessions(request: Request):
    account = _account(request)
    if not account:
        raise HTTPException(403)
    current = request.cookies.get(SESSION_COOKIE)
    revoke_account_sessions(int(account["id"]), token_digest(current) if current else None)
    record_security_event("other_sessions_revoked", account_id=int(account["id"]))
    return RedirectResponse("/settings/privacy?notice=Other+sessions+signed+out", status_code=303)


@app.post("/settings/videos/{job_id}/delete", dependencies=[Depends(require_csrf)])
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


@app.post("/settings/billing/cancel", dependencies=[Depends(require_csrf)])
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


@app.post("/settings/billing/withdraw", dependencies=[Depends(require_csrf)])
def request_contract_withdrawal(request: Request, confirm: bool = Form(False)):
    account = _account(request)
    if not account or not confirm:
        return _form_refusal("Confirm the withdrawal request.",
                             "/settings/billing", "/settings/billing")
    full_account = get_account(int(account["id"])) or account
    if not full_account.get("stripe_subscription_id"):
        return _form_refusal("No connected purchase is available for withdrawal review.",
                             "/settings/billing", "/settings/billing")
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


@app.post("/profile", dependencies=[Depends(require_csrf)])
async def save_profile(
    request: Request,
    display_name: str = Form(...),
    notes: str = Form(""),
    default_fighter: str = Form("A"),
    allow_model_training: bool = Form(False),
    account_type: str = Form("athlete"),
    photo: UploadFile | None = File(None),
    profile_video: UploadFile | None = File(None),
):
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/profile", status_code=303)
    current_profile = get_profile(profile_id) or {}
    default_fighter = default_fighter.upper()
    if default_fighter not in {"A", "B"}:
        return _form_refusal("Default fighter must be A or B.", "/profile", "/profile")
    # Anything unexpected settles on athlete, which is the narrower of the two:
    # it never grants roster room nobody paid for.
    set_account_type(profile_id, account_type)
    photo_path = None
    profile_video_path = None
    if photo and photo.filename:
        folder = ROOT / "app" / "static" / "profile"
        folder.mkdir(parents=True, exist_ok=True)
        suffix = Path(photo.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            return _form_refusal("Profile photo must be JPG, PNG or WEBP.", "/profile", "/profile")
        file_path = folder / f"profile_{profile_id}{suffix}"
        await run_in_threadpool(_save_upload_limited, photo, file_path, MAX_PROFILE_PHOTO_BYTES)
        if await run_in_threadpool(cv2.imread, str(file_path)) is None:
            file_path.unlink(missing_ok=True)
            return _form_refusal("The profile photo could not be decoded as a safe image.",
                                 "/profile", "/profile")
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


def _athlete_name(profile: dict, fights: list[dict]) -> str:
    """Whose progress this is, by the name the fights were filed under.

    The page printed the workspace display name, which starts as "My Athlete"
    for everyone, while every fight in the library was filed under the actual
    fighter. The fighter most fights were filed against wins; the display name
    is the fallback for a workspace that never named one.
    """
    names = Counter(str(fight.get("fighter_name")).strip()
                    for fight in fights if str(fight.get("fighter_name") or "").strip())
    if names:
        return names.most_common(1)[0][0]
    return str(profile.get("display_name") or SETTINGS.default_profile_name)


def _athlete_fighter_id(fights: list[dict]) -> int | None:
    """The roster fighter this page follows: the one most fights are filed under."""
    ids = Counter(fight["fighter_id"] for fight in fights if fight.get("fighter_id") is not None)
    return ids.most_common(1)[0][0] if ids else None


def _report_available(job_id: str) -> bool:
    """Whether /result/{job_id} would find a report to show.

    The library listed every saved fight and linked each one, so a fight
    whose report files were gone - a moved data folder, a lost disk - was a
    link to "page not found". The card says so instead, and can still be
    deleted. Never raises: a card is not worth an error page.
    """
    try:
        directory = completed_artifact_directory(job_id)
        return directory is not None and (directory / "report.json").is_file()
    except Exception:  # noqa: BLE001
        return False


# What the fight library's "Pending" section lists: everything somebody has
# uploaded that is not a finished report yet. Analyses in progress never
# appeared in /history at all, and an upload waiting for its fighters could
# only be found again through the top-bar chip - which shows one job - so the
# "finish your pending fights" refusal pointed at things nobody could see.
PENDING_LIBRARY_STATUSES = ("selecting", "queued", "running", "interrupted", "error")
_PENDING_LABELS = {
    "selecting": "Waiting for you to pick the fighters",
    "queued": "Waiting for the analysis machine",
    "running": "Being analysed",
    "interrupted": "Paused - open it to restart",
    "error": "Did not finish - open it for the reason",
}


def _pending_jobs_for_owner(owner_key: str) -> list[dict]:
    """This person's unfinished fights, newest first. Saved fights are not
    pending - they are already in the library below."""
    pending = []
    for job_id, job in list_jobs():
        status = job.get("status")
        if job.get("owner_key") != owner_key or status not in PENDING_LIBRARY_STATUSES:
            continue
        if get_fight(job_id):
            continue
        percent = max(0.0, min(99.0, float(job.get("percent") or 0.0)))
        label = _PENDING_LABELS[status]
        if status == "running" and percent > 0:
            label = f"Being analysed - {int(percent)}%"
        pending.append({
            "job_id": job_id,
            "status": status,
            "label": label,
            "continue_url": f"/select/{job_id}" if status == "selecting" else f"/progress/{job_id}",
            "continue_label": "Pick the fighters" if status == "selecting" else "Open",
            "original_name": str(job.get("original_name") or "Fight video")[:80],
            "ruleset": job.get("ruleset"),
            "created_at": datetime.fromtimestamp(
                float(job.get("created_at_epoch") or job.get("updated_at_epoch") or time.time()),
                tz=timezone.utc).isoformat(),
            "sort_key": float(job.get("created_at_epoch") or 0.0),
        })
    pending.sort(key=lambda item: item["sort_key"], reverse=True)
    return pending


def _discard_pending_job(job_id: str, job: dict) -> None:
    """Remove an unfinished fight and give back everything it held.

    Deleting the session first is what stops an analysis in flight: a worker
    claims and reports progress only against a session it can read, so its
    next progress report is refused and it abandons the run.
    """
    delete_job(job_id)
    if job.get("usage_reserved") and job.get("account_id"):
        release_analysis(int(job["account_id"]), job_id)
    video = Path(job.get("video_path") or "missing").resolve()
    if video.parent == UPLOADS.resolve():
        remove_derivative(video)
        video.unlink(missing_ok=True)
    job_dir = (OUTPUTS / job_id).resolve()
    if job_dir.parent == OUTPUTS.resolve() and job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    delete_legal_acceptances_for_resource(job_id)


@app.post("/pending/{job_id}/cancel", dependencies=[Depends(require_csrf)])
def cancel_pending_job(request: Request, job_id: str):
    """Cancel an upload waiting for its fighters, or an analysis not finished."""
    owner = _owner_key(request)
    job = get_job(job_id)
    if (not job or job.get("owner_key") != owner
            or job.get("status") not in PENDING_LIBRARY_STATUSES or get_fight(job_id)):
        raise HTTPException(404)
    _discard_pending_job(job_id, job)
    account = _account(request)
    record_security_event(
        "pending_fight_cancelled", account_id=int(account["id"]) if account else None,
        resource_type="fight", resource_id=job_id, metadata={"status": job.get("status")},
    )
    if "application/json" in request.headers.get("accept", ""):
        return {"cancelled": True}
    return RedirectResponse("/history#pending", status_code=303)


def _is_solo_fight(fight: dict) -> bool:
    """Whether a saved analysis was a solo session (one person, core/solo.py)."""
    summary = fight.get("summary") or {}
    setup = (summary.get("progress_report") or {}).get("setup") or {}
    return setup.get("mode") == "solo"


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    profile_id = _profile_id(request)
    fights = list_fights(profile_id) if profile_id is not None else []
    for fight in fights:
        fight["report_available"] = _report_available(fight["job_id"])
    pending = _pending_jobs_for_owner(_owner_key(request)) if profile_id is not None else []
    return templates.TemplateResponse(
        request=request, name="history.html",
        context={"request": request, "fights": fights, "pending": pending,
                 "signed_in": profile_id is not None},
    )


def _remove_fight_files(fight: dict) -> None:
    job_dir = (OUTPUTS / fight["job_id"]).resolve()
    if job_dir.parent == OUTPUTS.resolve() and job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    video = Path(fight["video_path"]).resolve()
    if video.parent == UPLOADS.resolve():
        remove_derivative(video)
        video.unlink(missing_ok=True)


def _remove_profile_file(value: str | None) -> None:
    if not value or not value.startswith("/static/profile/"):
        return
    path = (ROOT / "app" / value.lstrip("/")).resolve()
    if path.parent == (ROOT / "app" / "static" / "profile").resolve():
        path.unlink(missing_ok=True)


@app.post("/delete/{job_id}", dependencies=[Depends(require_csrf)])
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
    # Only offer fights that can actually be compared. The row outlives its
    # report - retention removes the outputs, and a failed or cleaned-up job
    # leaves the row behind - and both halves of this page need a readable
    # report for each side. Offering the rest meant a reader could pick two
    # fights, submit, and get a page with no comparison and no movement
    # table, still captioned "Choose two different saved fights" as though
    # they had not chosen. It also makes the "two fights required" gate above
    # count what it is actually gating on.
    fights = [f for f in fights if completed_artifact_directory(f["job_id"]) is not None
              and (completed_artifact_directory(f["job_id"]) / "report.json").is_file()
              # A solo session (core/solo.py) has no opponent and no Fighter
              # B, so set against a fight it read as "tracked at 0% coverage".
              and not _is_solo_fight(f)]
    for fight in fights:
        # Every option read "Fight analysis · <date>", so a reader with six
        # fights on one day was choosing between six identical lines.
        fight["choice_label"] = fight_choice_label(
            fight.get("ruleset"), fight.get("created_at"), fight.get("fight_type"),
            fight.get("fighter_name"))
        fight["choice_stamp"] = fight_choice_stamp(fight.get("created_at"))
    allowed = {fight["job_id"] for fight in fights}
    reports = []
    for job_id in (a, b):
        report = None
        if job_id in allowed:
            path = _require_completed_artifact(job_id, "report.json")
            report = json.loads(path.read_text(encoding="utf-8"))
        if report is not None:
            _apply_report_annotations(report, [])
            refresh_identity_integrity(report)
        reports.append(report)
    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "request": request, "fights": fights, "a": a, "b": b, "reports": reports,
            # Each card is headed with its fight, not "Fight analysis 1" - and
            # the page can say when the two are different fighters.
            "picked": [next((f for f in fights if f["job_id"] == job_id), None) for job_id in (a, b)],
            "signed_in": profile_id is not None,
            # The page promised a movement comparison "below" and rendered
            # nothing. These are the numbers that survive the strike gate.
            "movement": compare_movement(reports),
        },
    )


@app.get("/camp", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/coach", response_class=HTMLResponse)
def fight_camp_page(request: Request, error: str = "", name: str = "", session: str = "", redeem: str = ""):
    """Fight Camp: missions from the latest fight, progress, and the squad.

    Progress (/dashboard) and the coach portal (/coach) were two pages about
    the same thing - what this athlete should work on and whether it is
    working. They are one page now. Both old addresses still answer, with the
    same page, because sign-in lands on /dashboard and the coach forms return
    to /coach#squad and /coach#assignments.
    """
    profile_id = _profile_id(request)
    if profile_id is None:
        return templates.TemplateResponse(
            request=request, name="camp.html",
            context={"request": request, "profile": None, "signed_in": False, "progress": None,
                     "assignments": [], "camp": None},
        )
    profile = get_profile(profile_id) or {}
    fights = list_fights(profile_id)
    for fight in fights:
        # The saved-evidence list printed the raw ruleset enum beside every
        # entry: "Fight analysis · KICK_LIGHT".
        fight["choice_label"] = fight_choice_label(
            fight.get("ruleset"), fight.get("created_at"), fight.get("fight_type"),
            fight.get("fighter_name"))
    # One athlete's progress. The page is headed with one fighter's name, and
    # it charted every fight in the workspace: a teammate's 40% guard became
    # the athlete's "last fight" and the headline read -18 points. Fights with
    # no roster fighter (analysed before the roster existed) stay in.
    athlete_id = _athlete_fighter_id(fights)
    records = [record for record in _reports_for_profile(profile_id)
               if athlete_id is None or record.get("fighter_id") in (None, athlete_id)]
    default_fighter = profile.get("default_fighter", "A")
    progress = build_progress(records, default_fighter)
    other_fights = sum(1 for fight in fights
                       if athlete_id is not None and fight.get("fighter_id") not in (None, athlete_id))
    assignments = list_assignments(profile_id)

    newest_first = [{"job_id": r["job_id"], "created_at": r.get("created_at"), "report": r["report"],
                     "fighter": _camp_fighter(r["report"], default_fighter)} for r in reversed(records)]
    camp = fight_camp_missions(newest_first, assignments)
    # The big award: a fight after the mission was taken shows its number
    # moved. Checked here, where the fights are already loaded; award_points
    # gives it once however often the page is opened.
    results = {}
    linked = {}
    for mission in list_camp_missions(profile_id):
        linked[mission["assignment_id"]] = mission
        later = [r for r in reversed(newest_first)
                 if r["job_id"] != mission["job_id"] and str(r.get("created_at") or "") > mission["created_at"]]
        result = mission_result(mission, later)
        if result is not None:
            results[mission["assignment_id"]] = result
            if result["improved"]:
                award_points(profile_id, IMPROVED_POINTS, "improved", str(mission["assignment_id"]))
    sessions = list_training_sessions(profile_id)
    counted = {}
    for item in sessions:
        if item["verdict"] == "counted" and item["assignment_id"] is not None:
            counted[item["assignment_id"]] = counted.get(item["assignment_id"], 0) + 1
    today = datetime.now(timezone.utc).date()
    board = mission_board(camp, assignments, linked, results, counted)
    points = list_points(profile_id)
    return templates.TemplateResponse(
        request=request, name="camp.html",
        context={
            "request": request, "profile": profile, "signed_in": True, "progress": progress,
            "athlete_name": _athlete_name(profile, fights),
            "other_fighter_fights": other_fights,
            "assignments": assignments,
            "camp": camp,
            "standing": camp_standing(points, sessions, today),
            "points_history": points_history(points),
            # The stats-only Fight Camp story card (static/camp_share.js).
            "camp_card": {
                "improved": sum(1 for result in results.values() if result["improved"]),
                "sessions": sum(counted.values()),
            },
            # One card per mission at whatever stage it is, and the one thing
            # to do next at the top. See core/camp.py.
            "board": board,
            "today": next_step(camp, board, paid_sessions_on(sessions, today), bool(fights)),
            "session_message": TRAINING_VERDICT_MESSAGES.get(session, "").format(
                points=SESSION_POINTS, cap=DAILY_COUNTED_SESSIONS),
            # For the pop-up: only a session that just earned points.
            "session_points": SESSION_POINTS if session == "counted" else 0,
            "points_rules": {"session": SESSION_POINTS, "daily": DAILY_COUNTED_SESSIONS,
                             "done": MISSION_DONE_POINTS, "improved": IMPROVED_POINTS,
                             "redeem_cost": REDEEM_COST, "redeem_per_month": REDEEM_PER_MONTH},
            "redeem_message": REDEEM_MESSAGES.get(redeem, "").format(cost=REDEEM_COST, cap=REDEEM_PER_MONTH),
            "extra_analyses": (analysis_allowance(int(account["id"]))["bonus_remaining"]
                               if (account := _account(request)) else 0),
            "fights": fights,
            # A coach triages a squad; an athlete fixes one thing. This is the
            # coach half - every fight in order and which way the numbers are
            # moving. See core/squad.py.
            "squad": build_squad_view(fights),
            # The roster is what the per-row assign control offers.
            "roster": list_fighters(profile_id),
            # A rejected roster addition comes back here rather than as a 400
            # page, so the coach keeps the squad they were looking at. The
            # typed name comes back with it - being told the name was wrong
            # and then having to retype it is two punishments for one slip.
            "roster_error": error[:200],
            "roster_name": name[:80],
        },
    )


def _form_refusal(message: str, next_path: str, fallback: str) -> RedirectResponse:
    """Send a refused submission back to the page it came from, with the reason.

    Raising HTTPException renders the error page over whatever the visitor was
    doing. For a genuinely broken request that is right; for a mistyped
    password or an unticked box it throws away the page, every other field on
    it, and any idea of where they were. These refusals belong beside the
    field.

    The message rides in ?error= rather than in a session, because there is no
    session store to put it in and the page reading it is the page they were
    already on. _safe_next keeps the destination on this site; anything else
    falls back.
    """
    split = urlsplit(_safe_next(next_path, fallback))
    query = parse_qs(split.query)
    query["error"] = [message]
    return RedirectResponse(
        urlunsplit(split._replace(query=urlencode(query, doseq=True))), status_code=303)


def _roster_refusal(message: str, typed: str) -> RedirectResponse:
    """Send a rejected roster addition back to /coach, message and name intact."""
    query = urlencode({"error": message, "name": " ".join(typed.split())[:80]})
    return RedirectResponse(f"/coach?{query}#squad", status_code=303)


@app.post("/coach/fighters", dependencies=[Depends(require_csrf)])
def add_coach_fighter(request: Request, name: str = Form(...), next_path: str = Form("/coach#squad")):
    """Add someone to the roster without having to analyse a fight first.

    Filing an old fight needs a fighter to file it against, and until now the
    only way to create one was to upload a new fight and type a name on the
    setup page - so a workspace with fights but no roster had no way out of it.
    """
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/coach", status_code=303)
    roster = list_fighters(profile_id)
    cleaned = " ".join(name.split())
    # Both refusals return to /coach with the message instead of raising, which
    # rendered a full-page 400 and threw away the squad view. required= on the
    # input catches an empty box but not a box holding only spaces, so this is
    # reachable by typing one space, and it was the whole page for that.
    if not cleaned:
        return _roster_refusal("A fighter needs a name.", name)
    if not any(f["name"].lower() == cleaned.lower() for f in roster):
        capacity = roster_capacity(_request_plan(request), len(roster))
        if not capacity["can_add"]:
            return _roster_refusal(
                f"This plan holds {capacity['limit']} "
                f"fighter{'s' if capacity['limit'] != 1 else ''} and they are all in use. "
                "Archive one you no longer coach, or move to a plan with more room.",
                name,
            )
    create_fighter(profile_id, cleaned)
    return RedirectResponse(_safe_next(next_path, "/coach#squad"), status_code=303)


@app.post("/coach/fights/{job_id}/fighter", dependencies=[Depends(require_csrf)])
def assign_fight_to_fighter(
    request: Request,
    job_id: str,
    fighter_id: str = Form(""),
    next_path: str = Form("/coach#squad"),
):
    """File a past fight against a fighter.

    Fights analysed before the roster existed have no owner, so they produce no
    comparison and no trend. Rather than guess one for them - which is the bug
    the roster replaced - they are listed as unfiled and assigned here.
    """
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/coach", status_code=303)
    chosen = int(fighter_id) if fighter_id.strip().isdigit() else None
    if not assign_fighter_to_fight(profile_id, job_id, chosen):
        raise HTTPException(404, "That fight or fighter is not in this workspace.")
    return RedirectResponse(_safe_next(next_path, "/coach#squad"), status_code=303)


@app.post("/coach/assignments", dependencies=[Depends(require_csrf)])
def create_coach_assignment(
    request: Request,
    title: str = Form(...),
    detail: str = Form(""),
    next_path: str = Form("/camp#assignments"),
):
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/coach", status_code=303)
    title, detail = title.strip()[:100], detail.strip()[:600]
    if not title:
        raise HTTPException(400, "Assignment title is required.")
    add_assignment(profile_id, title, detail)
    return RedirectResponse(_safe_next(next_path, "/camp#assignments"), status_code=303)


@app.post("/coach/assignments/{assignment_id}/toggle", dependencies=[Depends(require_csrf)])
def update_coach_assignment(
    request: Request,
    assignment_id: int,
    next_path: str = Form("/camp#assignments"),
):
    profile_id = _profile_id(request)
    if profile_id is None or not toggle_assignment(assignment_id, profile_id):
        raise HTTPException(404)
    # Finishing something you trained for: once per item, and only with at
    # least one counted session behind it, so ticking a box alone earns nothing.
    done = any(item["id"] == assignment_id and item["status"] == "complete"
               for item in list_assignments(profile_id))
    trained = any(session["assignment_id"] == assignment_id and session["verdict"] == "counted"
                  for session in list_training_sessions(profile_id))
    if done and trained:
        award_points(profile_id, MISSION_DONE_POINTS, "mission_done", str(assignment_id))
    return RedirectResponse(_safe_next(next_path, "/camp#assignments"), status_code=303)


@app.post("/camp/missions", dependencies=[Depends(require_csrf)])
def take_camp_mission(request: Request, job_id: str = Form(...), index: int = Form(...)):
    """Take on a mission: it joins the list, remembering the number it is for.

    The mission is rebuilt from the fight's own report rather than read from
    the form, so the number a later fight is compared with is the measured
    one and not whatever a request says it was.
    """
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/camp", status_code=303)
    fight = get_fight(job_id)
    if not fight or int(fight["profile_id"]) != profile_id:
        raise HTTPException(404)
    report = json.loads(Path(fight["report_path"]).read_text(encoding="utf-8"))
    refresh_identity_integrity(report)
    fighter = _camp_fighter(report, (get_profile(profile_id) or {}).get("default_fighter", "A"))
    missions = missions_from_report(report, fighter)["missions"]
    if not 0 <= index < len(missions):
        raise HTTPException(404)
    mission = missions[index]
    detail = (mission["exercise"] or "") + (f" Target: {mission['target']}" if mission["target"] else "")
    assignment_id = add_assignment(profile_id, mission["title"][:100], detail.strip()[:600])
    if mission.get("metric") and mission.get("measured") is not None:
        link_camp_mission(assignment_id, profile_id, job_id, fighter, mission["metric"], mission["measured"])
    return RedirectResponse("/camp#missions", status_code=303)


# Training clips are checked and deleted inside the request, so they have to
# fit one: under the host's ~134 MiB request ceiling (core/chunked_upload.py).
TRAINING_UPLOAD_BYTES = 120 * 1024 * 1024
limit_upload_route("/camp/sessions/", TRAINING_UPLOAD_BYTES)
TRAINING_VERDICT_MESSAGES = {
    "counted": "Session counted: +{points} points.",
    "daily_cap": "Session counted, but you have had today's {cap} sessions' points. It still counts for your streak.",
    "duplicate": "That clip was already uploaded, so it was not counted again.",
    "too_short": "That clip is under a minute. Film at least one minute of the drill.",
    "too_long": "That clip is over 20 minutes. Upload one drill at a time.",
    "no_movement": "Nothing was moving in most of that clip. Film yourself doing the drill.",
    "unreadable": "That file could not be read as a video.",
    "too_large": "That video's resolution is too high to check. Film at 1080p or lower.",
}


REDEEM_MESSAGES = {
    "bought": "Done: you have an extra analysis. It is used when your plan's own analyses run out.",
    "not_enough_points": "Not enough points yet: a free analysis costs {cost}.",
    "monthly_cap": "You have had this month's {cap} free analyses. More next month.",
}


@app.post("/camp/redeem", dependencies=[Depends(require_csrf)])
def redeem_camp_points(request: Request):
    """Spend Fight Camp points on one extra analysis (core.db.redeem_points_for_analysis)."""
    _enforce_rate_limit(request, "camp_redeem", 10, 3600)
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/camp", status_code=303)
    outcome = redeem_points_for_analysis(profile_id, REDEEM_COST, REDEEM_PER_MONTH,
                                         datetime.now(timezone.utc).strftime("%Y-%m"))
    if outcome == "bought":
        account = _account(request)
        record_security_event("points_redeemed_for_analysis",
                              account_id=int(account["id"]) if account else None,
                              resource_type="points", resource_id=str(REDEEM_COST))
    return RedirectResponse(f"/camp?redeem={outcome}", status_code=303)


@app.post("/camp/sessions/{assignment_id}", dependencies=[Depends(require_csrf)])
def upload_training_session(request: Request, assignment_id: int, video: UploadFile = File(...)):
    """One training clip for one item on the list: checked, scored, deleted.

    Only the clip's fingerprint and what the check found are kept
    (training_sessions); the video is removed before the response, whatever
    the verdict. See core/training_check.py for what the check can and cannot
    tell.
    """
    _enforce_rate_limit(request, "training_sessions", 12, 3600)
    profile_id = _profile_id(request)
    if profile_id is None:
        return RedirectResponse("/login?next=/camp", status_code=303)
    if not any(item["id"] == assignment_id for item in list_assignments(profile_id)):
        raise HTTPException(404)
    folder = UPLOADS / "training"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.video"
    try:
        digest = _save_upload_limited(video, path, TRAINING_UPLOAD_BYTES)
        # The same two gates as a fight, before any decoder sees the file: the
        # bytes have to be a video container, and the malware scan has to pass.
        if not looks_like_video(path):
            found = {"verdict": "unreadable", "duration_seconds": None, "moving_share": None}
            earlier = []
        else:
            scan = scan_upload(path)
            if not scan["clean"]:
                if scan["status"] == "infected":
                    record_security_event("malware_upload_blocked", severity="warning")
                    raise HTTPException(400, "This file did not pass the upload safety scan.")
                raise HTTPException(503, "Uploads are paused because the safety scanner is unavailable.")
            earlier = list_training_sessions(profile_id)
            if any(session["video_sha256"] == digest for session in earlier):
                found = {"verdict": "duplicate", "duration_seconds": None, "moving_share": None}
            else:
                found = check_training_video(str(path))
    finally:
        path.unlink(missing_ok=True)
    verdict, points = found["verdict"], 0
    if verdict == "counted":
        if paid_sessions_on(earlier, datetime.now(timezone.utc).date()) < DAILY_COUNTED_SESSIONS:
            points = SESSION_POINTS
    session_id = record_training_session(profile_id, assignment_id, digest, found["duration_seconds"],
                                         found["moving_share"], verdict, points)
    if points:
        award_points(profile_id, points, "session", str(session_id))
    shown = "daily_cap" if verdict == "counted" and not points else verdict
    return RedirectResponse(f"/camp?session={shown}#today", status_code=303)


def _camp_fighter(report: dict, default_fighter: str) -> str:
    """The fighter a report follows for this athlete."""
    video = report.get("video") or {}
    fighter = video.get("focus_fighter") or video.get("analysis_target")
    return fighter if fighter in {"A", "B"} else default_fighter


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


@app.post("/cookie-preferences", dependencies=[Depends(require_csrf)])
def save_cookie_preferences(
    request: Request,
    choice: str = Form(...),
    next_path: str = Form("/"),
    analytics: bool = Form(False),
):
    if choice == "all":
        analytics = True
        value = "all"
    elif choice == "essential":
        analytics = False
        value = "essential"
    elif choice == "custom":
        value = "custom-analytics" if analytics else "essential"
    else:
        raise HTTPException(400, "Choose a valid cookie preference.")
    account = _account(request)
    owner = {"profile_id": int(account["profile_id"])} if account else {"guest_id": request.state.guest_id}
    if account:
        update_cookie_preferences(int(account["id"]), analytics=analytics, marketing=False)
    record_legal_acceptance(
        "cookie_preferences", SETTINGS.policy_version, metadata={"analytics": analytics},
        current_status="accepted" if analytics else "declined", **owner,
    )
    response = RedirectResponse(_safe_next(next_path, "/"), status_code=303)
    response.set_cookie(
        COOKIE_PREFERENCES_COOKIE, f"{value}:{SETTINGS.policy_version}",
        max_age=60 * 60 * 24 * 365,
        httponly=True, samesite="lax", secure=_request_is_secure(request),
    )
    return response


@app.get("/policies", response_class=HTMLResponse)
def policies_updated(request: Request, next: str = "/dashboard"):
    """Ask an account whose accepted policy version is behind the current one.

    Reached only after authentication, so there is an account to compare
    against - which is what /login could never do, and why it asked everybody
    on every sign-in instead.
    """
    account = _account(request)
    if account is None:
        return RedirectResponse("/login", status_code=303)
    if not policies_outdated(account):
        return RedirectResponse(_safe_next(next), status_code=303)
    return templates.TemplateResponse(
        request=request, name="policies.html",
        context={
            "request": request, "next_path": _safe_next(next),
            "accepted_version": account["terms_version"],
            "policy_version": SETTINGS.policy_version,
        },
    )


@app.post("/policies", dependencies=[Depends(require_csrf)])
def accept_updated_policies(
    request: Request,
    next_path: str = Form("/dashboard"),
    accept_policies: bool = Form(False),
):
    account = _account(request)
    if account is None:
        return RedirectResponse("/login", status_code=303)
    if not accept_policies:
        return templates.TemplateResponse(
            request=request, name="policies.html",
            context={
                "request": request, "next_path": _safe_next(next_path),
                "accepted_version": account["terms_version"],
                "policy_version": SETTINGS.policy_version,
                "error": "Accept the updated policies to continue.",
            },
            status_code=400,
        )
    record_policy_reacceptance(int(account["id"]))
    record_legal_acceptance(
        "account_policy_reacceptance", SETTINGS.policy_version,
        profile_id=int(account["profile_id"]),
        metadata={"previous_version": account["terms_version"]},
    )
    return RedirectResponse(_safe_next(next_path), status_code=303)


@app.get("/legal", response_class=HTMLResponse)
def legal_center(request: Request):
    return templates.TemplateResponse(
        request=request, name="legal.html",
        context={
            "request": request, "documents": LEGAL_DOCUMENTS,
            "launch": launch_readiness(), "policy_version": SETTINGS.policy_version,
            "launch_checklist_visible": SETTINGS.show_launch_checklist,
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
    # resolve_document, not a raw LEGAL_DOCUMENTS lookup: /contact names the
    # address for each purpose inline, and the placeholders are filled per
    # request so a changed setting needs only a restart.
    document = resolve_document(slug)
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
        context={"request": request, "submitted": False, "reference": None,
                 "error": "", "form": {}},
    )


@app.post("/copyright-report", response_class=HTMLResponse, dependencies=[Depends(require_csrf)])
def submit_copyright_report(
    request: Request,
    email: str = Form(...),
    details: str = Form(...),
    resource_id: str = Form(""),
    good_faith: bool = Form(False),
):
    _enforce_rate_limit(request, "copyright-report", 6, 3600)
    if not valid_email(email) or len(details.strip()) < 40 or not good_faith:
        # Raising here replaced the page with a 400 and threw the report away.
        # The browser's own minlength counts raw characters while this counts
        # stripped ones, so a padded report passes the form and fails here -
        # reachable by ordinary typing, and it cost the visitor everything
        # they had written.
        return templates.TemplateResponse(
            request=request, name="copyright_report.html",
            context={
                "request": request, "submitted": False, "reference": None,
                "error": "Provide a valid email, a detailed good-faith report "
                         "and the required confirmation.",
                "form": {"email": email, "details": details, "resource_id": resource_id,
                         "good_faith": good_faith},
            },
            status_code=400,
        )
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
            "traffic": page_view_summary(30),
            "email_problem": email_settings_problem(),
            "email_test": request.query_params.get("email_test", "")[:600],
        },
    )


@app.post("/admin/email/test", dependencies=[Depends(require_csrf)])
def admin_send_test_email(request: Request):
    """Send a test message to the signed-in admin and show exactly what happened.

    Password resets, verification and guardian emails all go through the same
    sender, so this is the one place an administrator can see why they fail.
    """
    if not _is_admin(request):
        raise HTTPException(404)
    actor = _account(request)
    try:
        deliver_email(actor["email"], "WarriorIQ test email",
                      "This is a test from the WarriorIQ admin page. Email is working.")
        outcome = f"Sent to {actor['email']}. If it has not arrived within a few minutes, check the spam folder."
    except EmailNotSent as reason:
        outcome = f"Not sent: {reason}"
    record_security_event("admin_email_test", account_id=int(actor["id"]), resource_type="email",
                          metadata={"outcome": outcome[:300]})
    return RedirectResponse("/admin?" + urlencode({"email_test": outcome}), status_code=303)


@app.post("/admin/accounts/{account_id}/status", dependencies=[Depends(require_csrf)])
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


@app.post("/admin/reports/{report_id}/resolve", dependencies=[Depends(require_csrf)])
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


@app.post("/admin/videos/delete", dependencies=[Depends(require_csrf)])
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


@app.get("/admin/openapi.json", include_in_schema=False)
def admin_openapi(request: Request):
    """The API schema, for a signed-in administrator only."""
    if not _is_admin(request):
        raise HTTPException(404)
    return JSONResponse(app.openapi())


@app.get("/assets/{bundle}.css", include_in_schema=False)
def css_bundle(bundle: str):
    """Serve one concatenated stylesheet instead of several.

    Served from memory rather than written into app/static, because the web
    host's application directory is not somewhere to be generating files at
    import time. The version token in the URL is the same asset hash the
    individual stylesheets used, so a deploy still busts the cache.
    """
    text = CSS_BUNDLE_TEXT.get(bundle)
    if text is None:
        raise HTTPException(404)
    return Response(
        content=text,
        media_type="text/css",
        headers={"Cache-Control": "public, max-age=604800"},
    )


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
    disallowed = "\n".join(f"Disallow: {prefix}" for prefix in PRIVATE_ROUTE_PREFIXES
                            if prefix not in UNADVERTISED_PRIVATE_PREFIXES)
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


# What this process started with, captured once at import.
#
# /health used to read the file on every request, so it reported whatever the
# last deploy had written even when the worker serving that request was still
# running the previous code. A deploy that copied every file but never recycled
# the process therefore looked perfectly healthy. This constant can only move
# when the process itself is replaced, which is the whole point of it.
RUNNING_COMMIT = _deployed_commit()


# --------------------------------------------------------------------------
# Chunked upload
#
# A phone films 1080p at 8-17 Mbps, so two minutes of fight is 140-260 MB,
# and a single POST of that has nothing to resume from. The same path to the
# live host measured 183 KiB/s and 3.5 MiB/s within an hour - twenty-three
# minutes and seventy seconds for the same file - and neither sample came
# from a phone on mobile data at a venue. That spread is the argument: a
# dropped connection costs one chunk here rather than the whole fight.
#
# /upload is untouched and stays the fallback. See core/chunked_upload.py.
# --------------------------------------------------------------------------


def _chunked_enabled() -> None:
    """404 when the path is switched off, before anything else runs.

    As an inline check at the top of each handler this ran *after*
    require_csrf, so a caller with no token got 403 - which says the route
    exists. Live, that is what /api/upload/probe answered while disabled. A
    switched-off endpoint should be indistinguishable from one that was never
    built, and dependencies run in order, so the gate goes first.
    """
    if not SETTINGS.chunked_upload_enabled:
        raise HTTPException(404)


def _probe_enabled() -> None:
    """The same, for the diagnostic endpoint - which matters more, because it
    exists to absorb large bodies."""
    if not SETTINGS.upload_probe_enabled:
        raise HTTPException(404)


def _chunked_failure(error: ChunkedUploadError) -> JSONResponse:
    payload = {"detail": error.detail}
    if error.offset is not None:
        payload["offset"] = error.offset
    return JSONResponse(payload, status_code=error.status)


def _release_chunked(account_id: int, job_id: str) -> None:
    """Give back everything begin took."""
    discard_chunked(job_id)
    release_analysis(account_id, job_id)
    release_upload_storage(job_id)


@app.post("/api/upload/begin", dependencies=[Depends(_chunked_enabled), Depends(require_csrf)])
async def chunked_upload_begin(request: Request):
    """Reserve capacity and open a session, before any bytes are sent.

    Admission is the part that changes shape. _admit_fight_upload reserves and
    then releases in a `finally` unless the response was a 201 - which is right
    for one request that carries the whole fight, and wrong here, where the
    reservation has to outlive this response and cover every chunk that
    follows. So this reserves, and the release moves to finish, abort, and the
    lease sweep as the backstop.

    The consent answers arrive here too, which is stricter than the single
    request path rather than looser: not one byte of somebody's footage is
    accepted before they have said they have the right to upload it.
    """
    account = _account(request)
    if not account:
        return JSONResponse(
            {"detail": "Create a free account or sign in to analyse a fight."},
            status_code=401)
    if hold := _guardian_hold(account):
        return JSONResponse({"detail": hold}, status_code=403)
    account_id = int(account["id"])
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - a malformed body is a 400, not a 500
        raise HTTPException(400, "Malformed request.") from None
    if not isinstance(payload, dict):
        raise HTTPException(400, "Malformed request.")

    form = {key: value for key, value in payload.items()
            if key not in {"filename", "size"}}
    if not (form.get("rights_confirmed") and form.get("people_permissions_confirmed")):
        raise HTTPException(400, "Confirm you have the right to upload this video.")
    if str(form.get("minor_permission_status") or "") not in {"no_minors", "guardian_authorized"}:
        raise HTTPException(400, "Answer whether anyone in the video is under 18.")

    job_id = uuid.uuid4().hex[:12]
    reserved = False
    ceiling = min(MAX_FIGHT_BYTES, SETTINGS.max_chunked_upload_bytes)
    try:
        declared = int(payload.get("size") or 0)
    except (TypeError, ValueError):
        declared = 0
    try:
        _enforce_rate_limit(request, "fight-upload", 12, 600)
        # Reserve what this file says it is, not the largest file allowed. The
        # declared size is enforced chunk by chunk (append refuses anything
        # past it), so it is a real bound; reserving the 512 MB ceiling three
        # times over for a 40 MB clip made two unfinished uploads look like a
        # full account.
        await run_in_threadpool(
            reserve_upload_storage, account_id, job_id,
            min(declared, ceiling) if declared > 0 else ceiling)
        reserved = await run_in_threadpool(reserve_analysis, account_id, job_id)
        if not reserved:
            release_upload_storage(job_id)
            return JSONResponse(
                {"detail": "Your analysis allowance is used for this period. "
                           "It will reset automatically."}, status_code=429)
        session = await run_in_threadpool(
            begin_chunked, job_id, account_id,
            filename=str(payload.get("filename") or "fight.mp4"),
            declared_bytes=int(payload.get("size") or 0), form=form)
    except UploadCapacityError as exc:
        _unwind_chunked(account_id, job_id, reserved)
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)
    except ChunkedUploadError as exc:
        _unwind_chunked(account_id, job_id, reserved)
        return _chunked_failure(exc)
    except Exception:
        _unwind_chunked(account_id, job_id, reserved)
        raise
    return JSONResponse({
        "job_id": session.job_id,
        "chunk_size": SETTINGS.upload_chunk_bytes,
        "offset": 0,
        "expires_in": SETTINGS.upload_timeout_seconds,
    }, status_code=201)


def _unwind_chunked(account_id: int, job_id: str, reserved: bool) -> None:
    """Undo exactly as much of begin as actually happened."""
    if reserved:
        _release_chunked(account_id, job_id)
    else:
        discard_chunked(job_id)
        release_upload_storage(job_id)


@app.put("/api/upload/{job_id}/chunk", dependencies=[Depends(_chunked_enabled), Depends(require_csrf)])
async def chunked_upload_chunk(request: Request, job_id: str, offset: int = 0):
    """Append one piece, if it continues where the file currently ends."""
    account = _account(request)
    if not account:
        raise HTTPException(401, "Sign in to continue this upload.")
    try:
        session = await run_in_threadpool(load_chunked, job_id, int(account["id"]))
        body = await request.body()
        received = await run_in_threadpool(append_chunk, session, offset, body)
    except ChunkedUploadError as exc:
        return _chunked_failure(exc)
    # Every piece pushes the expiry out. Without this the sweep reclaims an
    # upload that is still arriving, which is exactly the slow transfer this
    # path exists to carry.
    await run_in_threadpool(extend_lease, job_id)
    return JSONResponse({"offset": received, "declared": session.declared_bytes,
                         "complete": received >= session.declared_bytes})


@app.get("/api/upload/{job_id}/status", dependencies=[Depends(_chunked_enabled)])
async def chunked_upload_status(request: Request, job_id: str):
    """Where to resume. Lets a reloaded page pick up an upload in progress."""
    account = _account(request)
    if not account:
        raise HTTPException(401, "Sign in to continue this upload.")
    try:
        session = await run_in_threadpool(load_chunked, job_id, int(account["id"]))
    except ChunkedUploadError as exc:
        return _chunked_failure(exc)
    return JSONResponse({"offset": session.received_bytes,
                         "declared": session.declared_bytes,
                         "complete": session.complete})


@app.post("/api/upload/{job_id}/abort", dependencies=[Depends(_chunked_enabled), Depends(require_csrf)])
async def chunked_upload_abort(request: Request, job_id: str):
    """Give the capacity back rather than waiting for the sweep.

    max_pending_uploads is 2, so two abandoned sessions lock an account out of
    uploading until their leases expire. Somebody who changes their mind
    should not have to wait fifteen minutes.
    """
    account = _account(request)
    if not account:
        raise HTTPException(401, "Sign in to continue this upload.")
    account_id = int(account["id"])
    try:
        await run_in_threadpool(load_chunked, job_id, account_id)
    except ChunkedUploadError as exc:
        return _chunked_failure(exc)
    await run_in_threadpool(_release_chunked, account_id, job_id)
    return JSONResponse({"released": True})


@app.post("/api/upload/{job_id}/finish", dependencies=[Depends(_chunked_enabled), Depends(require_csrf)])
async def chunked_upload_finish(request: Request, job_id: str):
    """Hand the assembled file to the pipeline the single-request path uses.

    Everything after the bytes land is identical for both paths - the
    container check, the malware scan, the container normalisation, the
    decode, the duration and pixel limits, the selection frame, the job row -
    so it is called rather than copied. A second implementation of two hundred
    lines of upload handling is a second implementation to keep in step, and
    the one thing worse than an upload path with a bug is two of them with
    different bugs.

    The form answers were captured at `begin`, before any footage was
    accepted, and are replayed here.
    """
    account = _account(request)
    if not account:
        raise HTTPException(401, "Sign in to continue this upload.")
    account_id = int(account["id"])
    try:
        session = await run_in_threadpool(load_chunked, job_id, account_id)
        video_path = await run_in_threadpool(finalise_chunked, session)
    except ChunkedUploadError as exc:
        return _chunked_failure(exc)

    form = session.form

    def text(name: str, fallback: str = "") -> str:
        value = form.get(name)
        return fallback if value is None else str(value)

    def number(name: str, fallback: float) -> float:
        try:
            return float(form.get(name))
        except (TypeError, ValueError):
            return fallback

    def flag(name: str) -> bool:
        value = form.get(name)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    # The handler reads the job id off request.state, because for /upload the
    # admission middleware puts it there. Here admission happened at `begin`,
    # so the same slot is filled from the session rather than re-reserved.
    request.state.upload_job_id = session.job_id
    # The handler renders the frame page unless the caller asked for JSON, and
    # the caller here is always a script. Forcing it makes finish's contract
    # its own rather than a consequence of what the client happened to send -
    # and the first run of this returned a 200 HTML page, which the success
    # check below then read as a failure and handed the reservation back
    # underneath a fight that had actually uploaded.
    request.scope["headers"] = [
        (name, value) for name, value in request.scope["headers"]
        if name != b"accept"
    ] + [(b"accept", b"application/json")]
    # Starlette memoises request.headers on first read, and require_csrf has
    # already read it by the time this runs - so rewriting the scope alone
    # changed nothing and the handler went on rendering HTML.
    request.__dict__.pop("_headers", None)
    background = BackgroundTasks()
    try:
        response = await upload(
            request=request,
            background=background,
            video=StoredUpload(video_path, session.original_name),
            fight_type=text("fight_type", "competition"),
            analysis_target=text("analysis_target", "BOTH"),
            ruleset=text("ruleset", "K1"),
            start_seconds=number("start_seconds", 0.0),
            end_seconds=text("end_seconds"),
            round_count=int(number("round_count", 3)),
            round_duration_seconds=number("round_duration_seconds", 0.0),
            break_duration_seconds=number("break_duration_seconds", 60.0),
            selected_rounds=text("selected_rounds", "ALL"),
            openai_identity_recovery=flag("openai_identity_recovery"),
            external_ai_guardian_permission=flag("external_ai_guardian_permission"),
            fighter_id=text("fighter_id"),
            fighter_name=text("fighter_name"),
            rights_confirmed=flag("rights_confirmed"),
            people_permissions_confirmed=flag("people_permissions_confirmed"),
            minor_permission_status=text("minor_permission_status", "no_minors"),
        )
    except HTTPException as exc:
        # The pipeline refused the file - not a video, would not decode, too
        # long. The reservation has to go back, exactly as the single-request
        # path's `finally` would have returned it.
        await run_in_threadpool(_release_chunked, account_id, job_id)
        raise exc
    except Exception:
        await run_in_threadpool(_release_chunked, account_id, job_id)
        raise

    # Any refusal, however it is spelled. Listing the success codes instead
    # meant a response nobody anticipated was treated as a failure.
    if response.status_code >= 400:
        await run_in_threadpool(_release_chunked, account_id, job_id)
        return response
    # The job owns the reservation now; only the storage lease is handed back,
    # because the fight row is what accounts for the bytes from here on.
    await run_in_threadpool(release_upload_storage, job_id)
    # upload() collects deferred work - the shot profile scan - on the
    # BackgroundTasks it was handed. Calling it directly means attaching them
    # to the response ourselves, or they are silently dropped.
    response.background = background
    return response


@app.put("/api/upload/probe", include_in_schema=False,
         dependencies=[Depends(_probe_enabled), Depends(require_csrf)])
async def upload_probe(request: Request):
    """Measure what this host accepts in one request body, and how.

    The 130 MiB ceiling WarriorIQ enforces is attributed to the web host, and
    that attribution is doubtful. The recorded symptom - refused at exactly
    130 MiB with a 500 rather than a 413 - is exactly what this application's
    own UploadBodyLimitMiddleware does when a body arrives with no
    Content-Length to pre-check, and 130 MiB is exactly the default of
    SETTINGS.max_upload_bytes. Asked directly, the live host returns
    100 Continue for a declared 200 MiB, so Apache is not refusing on size
    before the body.

    Designing a chunked upload around a ceiling nobody has measured would be
    building on the same guess. So this measures three things a local test
    cannot:

      * the largest body this host will carry into the application;
      * whether it arrives streamed or buffered, which decides chunk size;
      * how long it is allowed to take, which is a separate wall from size.

    It writes nothing. The body is counted and discarded, so a probe cannot
    fill the disk or leave an orphan. Off by default, admin only, and it
    deliberately bypasses no limit - the point is to find out where they are.
    """
    if not _is_admin(request):
        raise HTTPException(404)
    _enforce_rate_limit(request, "upload-probe", 20, 600)
    declared = request.headers.get("content-length")
    started = time.perf_counter()
    received = 0
    pieces = 0
    largest = 0
    # Streamed, never read(): the question is partly whether this host hands
    # the body over in pieces at all, and request.body() would hide that by
    # buffering it first.
    async for chunk in request.stream():
        received += len(chunk)
        if chunk:
            pieces += 1
            largest = max(largest, len(chunk))
    elapsed = time.perf_counter() - started
    LOGGER.info("upload_probe received=%d declared=%s pieces=%d seconds=%.2f",
                received, declared, pieces, elapsed)
    return JSONResponse({
        "received_bytes": received,
        "received_mib": round(received / 1048576, 2),
        "declared_content_length": declared,
        # More than one piece means the body streamed. One piece the size of
        # the whole body means something upstream buffered it, and a chunk
        # size has to be chosen against that rather than against the wire.
        "pieces": pieces,
        "largest_piece_bytes": largest,
        "streamed": pieces > 1,
        "seconds": round(elapsed, 2),
        "app_limit_mib": round(min(SETTINGS.max_fight_bytes, SETTINGS.max_upload_bytes) / 1048576, 1),
    })


@app.get("/health", include_in_schema=False)
def health_check(request: Request):
    """Minimal deployment probe with no account, model or filesystem details.

    "commit" is the code actually running. "deployed" only appears when the
    files on disk are newer than the process serving them, which means a deploy
    copied its files without restarting the application.
    """
    payload = {"status": "ok", "service": "WarriorIQ", "commit": RUNNING_COMMIT}
    # The probe skips the per-visitor context (see _is_lightweight_request), so
    # the session is read here, and only when there is one to read.
    if not hasattr(request.state, "account") and request.cookies.get(SESSION_COOKIE):
        request.state.account = resolve_session(request.cookies.get(SESSION_COOKIE))
    # How long THIS process has been alive, and how much memory it is holding.
    #
    # Added to answer a question nothing else could. Measured from outside, the
    # site was taking 34 and 82 seconds to send a first byte while reporting 50
    # to 90 ms of its own work on the very same requests - and fast requests
    # were interleaved with the slow ones under continuous load. That shape is
    # not a slow application; it is requests waiting on a process that is
    # starting up, and `import app.main` alone costs 3.3s on a development
    # machine against a host measured 8 to 12 times slower.
    #
    # If successive calls to this endpoint report an uptime that keeps resetting
    # then the application is being recycled and every recycle is a stall for
    # whoever is browsing. If it climbs steadily, the wait is somewhere in front
    # of the application and no amount of work in here will shorten it. Two
    # numbers, one answer, and no way to get it from outside without them.
    # Shown only to a signed-in administrator. The probe is deliberately
    # minimal for everybody else - no account, model, path or worker detail -
    # and that decision has a test guarding it, so this adds nothing to what
    # the public sees.
    if _is_admin(request):
        payload["uptime_seconds"] = round(time.time() - PROCESS_STARTED_AT, 1)
        payload["pid"] = os.getpid()
        resident = _resident_megabytes()
        if resident is not None:
            payload["rss_mb"] = resident
    on_disk = _deployed_commit()
    if on_disk != RUNNING_COMMIT:
        payload["deployed"] = on_disk
        payload["restart_required"] = True
    return payload


@app.api_route("/healthz", methods=["GET", "HEAD"], include_in_schema=False)
def healthz():
    """The cheapest possible liveness answer, for an uptime monitor.

    No database, no disk, no session, no template: if this does not answer,
    the process is not serving at all, or something in front of it - the host,
    its firewall, a bot-challenge layer - is not letting the request through.
    /health adds the running commit; /ready checks the database, storage and
    worker and is the one to alert on for "analyses will not run".
    """
    return Response('{"status":"ok"}', media_type="application/json",
                    headers={"Cache-Control": "no-store"})


@app.get("/ready", include_in_schema=False)
def readiness_check():
    """Operational readiness: database, private storage and analysis worker."""
    from core.readiness import operational_readiness

    readiness = operational_readiness(worker_status())
    readiness["wake"] = {
        **wake_status(),
        "drain_interval_seconds": SETTINGS.wake_drain_interval_seconds,
        "magic_packet_configured": bool(SETTINGS.wol_mac and SETTINGS.wol_host),
    }
    return JSONResponse(readiness, status_code=200 if readiness["ready"] else 503)


@app.get("/sitemap.xml")
def sitemap_xml():
    base = SETTINGS.public_base_url
    # The same tuple the noindex rule reads, rather than a second copy of it.
    urls = "" if not base else "".join(
        f"<url><loc>{html.escape(base + path)}</loc></url>" for path in PUBLIC_INDEX_ROUTES
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
    """Every plan on one page.

    This was split into an athlete tab and a coach tab, on the theory that each
    side should only see what applies to them. In practice a coach had to find
    a tab before finding a plan, and anyone landing on the wrong one concluded
    the other plans did not exist. The seat count on each card already says who
    it is for.
    """
    account = _account(request)
    return templates.TemplateResponse(
        request=request,
        name="pricing.html",
        context={
            "request": request, "plans": PLANS,
            # Seven cards cannot be compared on a phone; this can.
            "plan_comparison": plan_comparison(),
            # So a plan too small for this roster says so on the card, rather
            # than failing after somebody has committed to buying it.
            "roster_held": len(list_fighters(int(account["profile_id"]))) if account else 0,
            "payments_enabled": SETTINGS.payments_enabled, "account": account,
            "allowance": analysis_allowance(int(account["id"])) if account else None,
            # The strip read analysis_allowance (which resolves through
            # effective_plan_key) while the card badge compared
            # plan_override or plan by hand - a second copy of the same
            # lookup, missing its complimentary-grant branch. So an account
            # holding a granted plan was told "Your current plan: Gym" above
            # a Starter card badged "Current plan". One resolver, both places.
            "current_plan_key": effective_plan_key(
                account.get("plan"), account.get("plan_override"), account.get("email"),
            ) if account else None,
            "plans_wanted": plans_wanted_by(int(account["id"])) if account else set(),
        },
    )


@app.post("/plan-interest/{plan_key}", dependencies=[Depends(require_csrf)])
def register_plan_interest(request: Request, plan_key: str, next_path: str = Form("/pricing")):
    """Record that somebody wants a paid plan that is not open yet.

    Every button on a paid card opened the workspace the visitor already had,
    so during early access there was no way to say which plan you actually
    wanted and no way to find out afterwards that anybody had.
    """
    account = _account(request)
    if not account:
        return RedirectResponse("/login?next=/pricing", status_code=303)
    if plan_key not in PLANS or plan_key == "free":
        raise HTTPException(404, "Unknown plan.")
    _enforce_rate_limit(request, "plan-interest", 30, 3600)
    record_plan_interest(int(account["id"]), plan_key)
    return RedirectResponse(f"{_safe_next(next_path, '/pricing')}#plan-{plan_key}", status_code=303)


@app.post("/share/{job_id}", dependencies=[Depends(require_csrf)])
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
    # The link is only ever readable at this moment: the database keeps a digest,
    # not the token. Sending the coach's own view back here would strand the
    # athlete on a page with no address to copy, so return to the report with the
    # one-time link in hand.
    return RedirectResponse(f"/result/{job_id}?share={token}", status_code=303)


@app.get("/s/{token}", response_class=HTMLResponse)
def shared_report(request: Request, token: str):
    # The person holding the link is a coach, not the athlete: a bare "Not
    # Found" with "Start an analysis" told them nothing about what happened.
    gone = "This coach link has expired or was turned off. Ask the athlete to send a new one."
    share = get_report_share(token_digest(token))
    if not share:
        raise HTTPException(404, gone)
    path = _require_completed_artifact(share["job_id"], "report.json")
    if not path.exists():
        raise HTTPException(404, gone)
    report = json.loads(path.read_text(encoding="utf-8"))
    _apply_report_annotations(report, [])
    refresh_identity_integrity(report)
    return templates.TemplateResponse(
        request=request, name="shared.html",
        context={"request": request, "report": report, "expires_at": share["expires_at"],
                 "expires_label": _friendly_date(share["expires_at"]),
                 # The same verdicts the athlete's page shows, so a coach sees
                 # no number the athlete was told not to trust.
                 "numbers": _numbers_state(report),
                 "identity_cause": identity_failure(report.get("tracking") or {})},
    )


@app.post("/shares/{job_id}/revoke", dependencies=[Depends(require_csrf)])
def revoke_shares(request: Request, job_id: str):
    profile_id = _profile_id(request)
    fight = get_fight(job_id)
    if profile_id is None or not fight or int(fight["profile_id"]) != profile_id:
        raise HTTPException(404)
    revoked = revoke_report_shares(job_id, profile_id)
    return RedirectResponse(f"/result/{job_id}?revoked={revoked}", status_code=303)


# ---- Fight links ------------------------------------------------------------
#
# A public page for one fight's stats-only card (warrioriq.eu/f/<token>), made
# by the fight's owner to post. Its link preview carries the card as a picture
# (core/share_image.py), so on WhatsApp, Facebook and Messenger the post is the
# stats with warrioriq.eu under them, and tapping it opens the page. Only what
# core.report.share_card allows is on it: no video, and no opponent's name.

STORY_NAME_CHARS = 40
STORY_GONE = "This fight link was turned off by the fighter who shared it."


def _story_card(job_id: str) -> dict | None:
    """The fight's share card, built by the same steps as its result page."""
    try:
        report = json.loads(_require_completed_artifact(job_id, "report.json").read_text(encoding="utf-8"))
    except (HTTPException, OSError, json.JSONDecodeError):
        return None
    _score_and_identity_as_shown(report)
    _estimate_score_withheld_for_punches(report)
    return share_card(report)


def _story_name(value: str) -> str | None:
    """The owner's name for the page: printable characters, one space apart, kept short."""
    cleaned = " ".join("".join(ch for ch in value if ch.isprintable()).split())
    return cleaned[:STORY_NAME_CHARS] or None


@app.post("/story/{job_id}", dependencies=[Depends(require_csrf)])
def create_story_link(request: Request, job_id: str, side: str = Form(...), name: str = Form("")):
    """Make (or reuse) the public link for one fight and one fighter."""
    _enforce_rate_limit(request, "story-link", 20, 3600)
    profile_id = _profile_id(request)
    fight = get_fight(job_id)
    if profile_id is None or not fight or int(fight["profile_id"]) != profile_id:
        raise HTTPException(404)
    if side not in {"A", "B"}:
        raise HTTPException(400, "Choose which fighter you were.")
    if _story_card(job_id) is None:
        raise HTTPException(409, "This fight has no stats that can be shared.")
    job = _authorized_job(request, job_id) or {}
    corner = str(job.get("fighter_a_corner") or "").lower()
    if corner in {"red", "blue"} and side == "B":
        corner = "blue" if corner == "red" else "red"
    token = story_share(job_id, profile_id, side, corner if corner in {"red", "blue"} else None,
                        _story_name(name))
    return {"url": f"{_public_base(request)}/f/{token}"}


@app.post("/story/{job_id}/profile", dependencies=[Depends(require_csrf)])
def post_story_to_profile(request: Request, job_id: str, side: str = Form(...), name: str = Form(""),
                          posted: str = Form("1"), next_path: str = Form("")):
    """Put this fight's stats-only link on the owner's athlete page, or take it off.

    The page shows the same card as the link itself, to whoever may see the
    profile (core/social.py): private profiles to approved followers only.
    """
    _enforce_rate_limit(request, "story-link", 20, 3600)
    profile_id = _profile_id(request)
    fight = get_fight(job_id)
    if profile_id is None or not fight or int(fight["profile_id"]) != profile_id:
        raise HTTPException(404)
    if side not in {"A", "B"}:
        raise HTTPException(400, "Choose which fighter you were.")
    wanted = posted == "1"
    existing = next((link for link in list_story_shares(job_id, profile_id) if link["side"] == side), None)
    if wanted:
        if _story_card(job_id) is None:
            raise HTTPException(409, "This fight has no stats that can be shared.")
        job = _authorized_job(request, job_id) or {}
        corner = str(job.get("fighter_a_corner") or "").lower()
        if corner in {"red", "blue"} and side == "B":
            corner = "blue" if corner == "red" else "red"
        token = story_share(job_id, profile_id, side, corner if corner in {"red", "blue"} else None,
                            _story_name(name))
    else:
        token = existing["token"] if existing else None
    if token:
        set_story_on_profile(token, profile_id, wanted)
    handle = (get_profile(profile_id) or {}).get("handle")
    if next_path == "athlete" and handle:
        return RedirectResponse(f"/athlete/{quote(handle)}#posts", status_code=303)
    return {"url": f"{_public_base(request)}/f/{token}" if token else None, "posted": wanted and bool(token),
            "profile_url": f"/athlete/{quote(handle)}" if handle else None}


@app.post("/story/{job_id}/revoke", dependencies=[Depends(require_csrf)])
def revoke_story_links(request: Request, job_id: str):
    profile_id = _profile_id(request)
    fight = get_fight(job_id)
    if profile_id is None or not fight or int(fight["profile_id"]) != profile_id:
        raise HTTPException(404)
    return {"revoked": revoke_story_shares(job_id, profile_id)}


def _live_story(token: str) -> tuple[dict, dict]:
    share = get_story_share(token)
    card = _story_card(share["job_id"]) if share else None
    if not share or card is None:
        raise HTTPException(404, STORY_GONE)
    return share, card


@app.get("/f/{token}", response_class=HTMLResponse)
def story_page(request: Request, token: str):
    share, card = _live_story(token)
    side = share["side"]
    fight = get_fight(share["job_id"]) or {}
    base = _public_base(request)
    # Not cached (the middleware's no-store for /f/): turning a link off has to
    # take effect for the next person who opens it.
    return templates.TemplateResponse(request=request, name="story.html", context={
        "request": request, "card": card, "side": side, "other": "B" if side == "A" else "A",
        "corner": share.get("corner"), "name": share.get("name"),
        "fought_at": fight.get("created_at"),
        "page_url": f"{base}/f/{token}", "image_url": f"{base}/f/{token}/card.png",
    })


@app.get("/f/{token}/card.png")
def story_preview(token: str):
    share, card = _live_story(token)
    return Response(story_preview_png(card, share["side"], share.get("corner")), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=600"})


@app.post("/account/export", dependencies=[Depends(require_csrf)])
def export_account_data(request: Request, password: str = Form(...),
                        next_path: str = Form("/profile")):
    _enforce_rate_limit(request, "account-export", 5, 3600)
    account = _account(request)
    if not account or not authenticate(account["email"], password):
        # The password is not carried back, for the obvious reason.
        return _form_refusal(
            "Enter the current account password to export your data.",
            next_path, "/profile")
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
        "follows": {
            "followers": list_follows(profile_id, direction="followers"),
            "following": list_follows(profile_id, direction="following"),
            "pending_requests": list_follows(profile_id, direction="followers", status="pending"),
        },
        "coach_assignments": list_assignments(profile_id),
        # Fight Camp: missions taken, training sessions (the check's findings
        # and a fingerprint - the clips themselves are never kept) and points.
        "fight_camp": {
            "missions": list_camp_missions(profile_id),
            "training_sessions": list_training_sessions(profile_id),
            "points": list_points(profile_id),
        },
        "fight_links": list_profile_story_shares(profile_id),
        "connected_sign_in_identities": list_oauth_identities(int(account["id"])),
        "legal_acceptances": list_legal_acceptances(profile_id=profile_id),
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": 'attachment; filename="warrioriq-account-data.json"', "Cache-Control": "no-store"},
    )


@app.post("/account/delete", dependencies=[Depends(require_csrf)])
def delete_account_route(request: Request, password: str = Form(...), confirmation: str = Form(...),
                         next_path: str = Form("/profile")):
    # Takes a password, so it is a place to guess one. Nobody deletes their
    # account five times an hour.
    _enforce_rate_limit(request, "account-delete", 5, 3600)
    account = _account(request)
    if not account:
        raise HTTPException(403)
    if confirmation.strip().upper() != "DELETE" or not authenticate(account["email"], password):
        return _form_refusal(
            "Enter DELETE and your current password to remove the account.",
            next_path, "/profile")
    if any(
        job.get("owner_key") == f"account:{account['id']}" and job.get("status") in {"queued", "running"}
        for _, job in list_jobs()
    ):
        return _form_refusal(
            "Wait for the running analysis to finish before deleting the account.",
            next_path, "/profile")
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


@app.post("/checkout/{plan_key}", dependencies=[Depends(require_csrf)])
def checkout(request: Request, plan_key: str, billing_acceptance: bool = Form(False)):
    account = _account(request)
    if not account:
        return RedirectResponse("/login?next=/pricing", status_code=303)
    if not billing_acceptance:
        raise HTTPException(400, "Confirm the recurring price, renewal and cancellation terms before checkout.")
    from core.readiness import release_readiness

    readiness = release_readiness(worker_status())
    if not readiness["release_ready"]:
        raise HTTPException(503, "Paid checkout is blocked until WarriorIQ's real operator identity, email verification, storage, and analysis worker are ready.")
    if plan_key not in PLANS or plan_key == "free":
        raise HTTPException(400, "Choose a valid paid plan.")
    # A plan with fewer seats than the roster already holds is refused here
    # rather than sold and reconciled afterwards. The alternative is archiving
    # fighters on somebody's behalf to make the numbers fit, and quietly
    # removing people from a coach's squad because of a billing change is not
    # something this should ever do.
    target_seats = PLANS[plan_key].get("roster_limit")
    if target_seats is not None:
        held = len(list_fighters(int(account["profile_id"])))
        if held > int(target_seats):
            raise HTTPException(
                409,
                f"{PLANS[plan_key]['label']} holds {target_seats} "
                f"fighter{'s' if int(target_seats) != 1 else ''}, and this workspace has {held}. "
                f"Archive {held - int(target_seats)} you no longer coach first - "
                "WarriorIQ will not remove anyone for you.",
            )
    record_legal_acceptance(
        "recurring_billing_terms", SETTINGS.policy_version,
        profile_id=int(account["profile_id"]), resource_id=plan_key,
        metadata={"price": PLANS[plan_key]["price"], "period": PLANS[plan_key]["period"]},
    )
    base = _public_base(request)
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
    # Everything after checkout: renewals, failed cards, cancellations taking
    # effect, plan changes, full refunds and chargebacks. See
    # core.payments.subscription_change for what each one does.
    change = subscription_change(event)
    if change is not None:
        changed = apply_subscription_change(str(event.get("id", "")), str(event.get("type", "")), change)
        if changed is not None and change.get("plan") == "free":
            record_security_event(f"plan_ended_{change['status']}", account_id=changed,
                                  severity="warning" if change["status"] == "disputed" else "info",
                                  resource_type="plan", resource_id="free")
    return {"received": True}
