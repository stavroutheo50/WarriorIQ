from __future__ import annotations

import atexit
import collections
import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.config import DATASET, DB_PATH, SETTINGS
from core.payments import PLANS, effective_plan_key, plan_for_key

LOGGER = logging.getLogger("warrioriq.db")

# Whether WAL was actually accepted, so it is logged once rather than per
# connection. None until the first connection has tried.
_wal_state: bool | None = None


def _configure(con: sqlite3.Connection) -> None:
    """Make concurrent access survivable.

    WarriorIQ runs a web process and a worker against one file. The rollback
    journal takes an exclusive lock for the whole of a write, so a reader is
    blocked for its duration. WAL lets readers see the last committed state
    while a write is in progress, and only writers queue.

    Measured here, six writers making 25 writes each: 2.68s on the rollback
    journal against 1.84s on WAL. Neither produced a single failure, which is
    worth stating plainly - sqlite3.connect() already applies a five-second
    busy timeout when none is passed, so contention was being waited out
    rather than raised even before this. Setting busy_timeout explicitly
    changes no behaviour at the default; it makes the value configurable and
    stops it depending on a driver default that could change.

    WAL is attempted, not assumed. It needs a shared-memory file beside the
    database and is unreliable over a network filesystem, which a cheap shared
    host may be using. A refusal is logged and the old journal keeps working,
    because a slower database beats a broken one.
    """
    global _wal_state
    # Set first and unconditionally: it is what makes contention wait rather
    # than raise, and it is safe in every journal mode.
    con.execute("PRAGMA busy_timeout=%d" % int(SETTINGS.sqlite_busy_timeout_ms))
    if not SETTINGS.sqlite_wal:
        return
    try:
        mode = con.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    except sqlite3.Error as exc:
        if _wal_state is not False:
            LOGGER.warning("sqlite_wal_refused error=%s - staying on the rollback journal",
                           type(exc).__name__)
            _wal_state = False
        return
    accepted = str(mode).lower() == "wal"
    if _wal_state is None or _wal_state != accepted:
        LOGGER.info("sqlite_journal_mode=%s", mode)
        _wal_state = accepted
    if accepted:
        # Safe to relax only under WAL: a commit still survives a process
        # crash, and only an OS-level crash can lose the most recent commits.
        con.execute("PRAGMA synchronous=NORMAL")


@contextmanager
def connection():
    """The one place WarriorIQ opens a database connection.

    Every query in this module goes through here, which is what makes the
    engine a single decision rather than 155 of them. Moving to another engine
    means changing this function and the SQL dialect it serves - see
    docs/postgres-migration.md for what that involves and why it has not been
    done blind.
    """
    con = sqlite3.connect(DB_PATH, timeout=SETTINGS.sqlite_busy_timeout_ms / 1000.0)
    con.row_factory = sqlite3.Row
    _configure(con)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _remove_annotation_sequences(con: sqlite3.Connection, job_id: str) -> None:
    allowed = (DATASET / "sequences").resolve()
    rows = con.execute("SELECT sequence_path FROM annotations WHERE job_id=?", (job_id,)).fetchall()
    for row in rows:
        if not row[0]:
            continue
        path = Path(row[0]).resolve()
        if path.parent == allowed:
            path.unlink(missing_ok=True)


def init_db() -> None:
    with connection() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL,
                photo_path TEXT,
                video_path TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                profile_id INTEGER NOT NULL,
                original_name TEXT NOT NULL,
                video_path TEXT NOT NULL,
                report_path TEXT NOT NULL,
                fight_type TEXT NOT NULL,
                ruleset TEXT NOT NULL,
                analysis_target TEXT NOT NULL,
                created_at TEXT NOT NULL,
                summary_json TEXT,
                FOREIGN KEY(profile_id) REFERENCES profiles(id)
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                event_time REAL NOT NULL,
                ruleset TEXT NOT NULL,
                predicted_json TEXT NOT NULL,
                corrected_json TEXT NOT NULL,
                sequence_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(job_id, event_time)
            );

            CREATE TABLE IF NOT EXISTS fight_reviews (
                job_id TEXT PRIMARY KEY,
                profile_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_progress',
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY(job_id) REFERENCES fights(job_id),
                FOREIGN KEY(profile_id) REFERENCES profiles(id)
            );

            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                profile_id INTEGER UNIQUE NOT NULL,
                plan TEXT NOT NULL DEFAULT 'free',
                credits INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY(profile_id) REFERENCES profiles(id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE TABLE IF NOT EXISTS oauth_identities (
                provider TEXT NOT NULL,
                subject TEXT NOT NULL,
                account_id INTEGER NOT NULL,
                email_at_link TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(provider, subject),
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE TABLE IF NOT EXISTS coach_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                detail TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY(profile_id) REFERENCES profiles(id)
            );

            CREATE TABLE IF NOT EXISTS report_shares (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                profile_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT,
                FOREIGN KEY(job_id) REFERENCES fights(job_id),
                FOREIGN KEY(profile_id) REFERENCES profiles(id)
            );

            CREATE TABLE IF NOT EXISTS payment_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                received_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analysis_usage (
                job_id TEXT PRIMARY KEY,
                account_id INTEGER NOT NULL,
                period_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE TABLE IF NOT EXISTS legal_acceptances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER,
                guest_id TEXT,
                kind TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                resource_id TEXT,
                accepted_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(profile_id) REFERENCES profiles(id)
            );

            CREATE TABLE IF NOT EXISTS subscription_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                effective_at TEXT,
                provider_reference TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'info',
                resource_type TEXT,
                resource_id TEXT,
                occurred_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE TABLE IF NOT EXISTS athlete_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fighter_id INTEGER NOT NULL,
                job_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                accuracy REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                landed INTEGER NOT NULL DEFAULT 0,
                output_per_round REAL,
                defensive_lapses INTEGER,
                pose_coverage REAL,
                evidence_trusted INTEGER NOT NULL DEFAULT 0,
                rated INTEGER NOT NULL DEFAULT 0,
                rated_on TEXT NOT NULL DEFAULT '',
                rating_before REAL,
                rating_after REAL,
                rating_delta REAL,
                UNIQUE(fighter_id, job_id)
            );

            CREATE TABLE IF NOT EXISTS athlete_ratings (
                fighter_id INTEGER PRIMARY KEY,
                rating REAL NOT NULL,
                rated_sessions INTEGER NOT NULL DEFAULT 0,
                total_sessions INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analysis_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                account_id INTEGER,
                stage TEXT NOT NULL DEFAULT 'analysis',
                reason TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 1,
                occurred_at TEXT NOT NULL,
                reviewed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_analysis_failures_job
                ON analysis_failures(job_id);

            CREATE TABLE IF NOT EXISTS moderation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT NOT NULL,
                reporter_email TEXT NOT NULL,
                resource_id TEXT,
                details TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS outbound_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER,
                message_type TEXT NOT NULL,
                recipient TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                sent_at TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE INDEX IF NOT EXISTS idx_analysis_usage_account_period
            ON analysis_usage(account_id, period_key);

            -- Admission leases exist before multipart parsing starts. Existing
            -- accounts/fights are unchanged; expired transfer leases are reaped.
            CREATE TABLE IF NOT EXISTS upload_leases (
                job_id TEXT PRIMARY KEY,
                account_id INTEGER NOT NULL,
                reserved_bytes INTEGER NOT NULL,
                expires_epoch REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_legal_acceptances_profile
            ON legal_acceptances(profile_id, accepted_at);

            CREATE INDEX IF NOT EXISTS idx_security_events_account_time
            ON security_events(account_id, occurred_at);

            CREATE INDEX IF NOT EXISTS idx_subscription_actions_account_time
            ON subscription_actions(account_id, requested_at);

            CREATE INDEX IF NOT EXISTS idx_oauth_identities_account
            ON oauth_identities(account_id);

            -- Who a fight was about.
            --
            -- A workspace used to be assumed to hold one person, so "since your
            -- last fight" compared whatever two analyses came last. That is
            -- true for an athlete and false for a coach, whose workspace holds
            -- a squad, and comparing two different fighters and calling the
            -- difference progress is not a fact about either of them.
            CREATE TABLE IF NOT EXISTS fighters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(profile_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_fighters_profile ON fighters(profile_id);

            -- Who asked to be told when a paid plan opens.
            --
            -- Paid checkout is blocked during early access, so every button on
            -- a paid card led to the workspace the visitor already had. There
            -- was no way to say "this is the one I want" and no way to find out
            -- afterwards which plans anyone had wanted.
            --
            -- One row per account per plan: asking twice is the same request,
            -- not two, so the pair is unique and a repeat updates the time
            -- rather than inflating a count that would then read as demand.
            CREATE TABLE IF NOT EXISTS plan_interest (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                plan_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(account_id, plan_key),
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );
            CREATE INDEX IF NOT EXISTS idx_plan_interest_plan ON plan_interest(plan_key);

            -- Traffic the analytics tag cannot see.
            --
            -- Consent Mode leaves analytics_storage denied until a visitor
            -- accepts cookies, and Google only turns denied-consent pings into
            -- reportable numbers once a property clears its modelling
            -- threshold - on the order of a thousand events a day. Below that,
            -- everyone who ignores the banner is simply absent from the
            -- reports, which on this site is nearly everyone.
            --
            -- This counts them: one row per day per path, a number that only
            -- goes up. There is deliberately no visitor column, no IP and no
            -- user agent, so a row cannot be tied to a person and needs no
            -- consent to keep. Adding any of those three would change that,
            -- and would make this the thing the banner exists to ask about.
            CREATE TABLE IF NOT EXISTS page_views (
                day TEXT NOT NULL,
                path TEXT NOT NULL,
                views INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(day, path)
            );
            CREATE INDEX IF NOT EXISTS idx_page_views_day ON page_views(day);
            """
        )
        columns = {row[1] for row in con.execute("PRAGMA table_info(profiles)").fetchall()}
        if "video_path" not in columns:
            con.execute("ALTER TABLE profiles ADD COLUMN video_path TEXT")
        if "default_fighter" not in columns:
            con.execute("ALTER TABLE profiles ADD COLUMN default_fighter TEXT NOT NULL DEFAULT 'A'")
        if "allow_model_training" not in columns:
            con.execute("ALTER TABLE profiles ADD COLUMN allow_model_training INTEGER NOT NULL DEFAULT 0")
        if "account_type" not in columns:
            # Athlete unless told otherwise: an existing workspace holds one
            # person's fights, and upgrading somebody to a coach account they
            # did not ask for would offer them seats they are not paying for.
            con.execute("ALTER TABLE profiles ADD COLUMN account_type TEXT NOT NULL DEFAULT 'athlete'")
        fight_columns = {row[1] for row in con.execute("PRAGMA table_info(fights)").fetchall()}
        if "fighter_id" not in fight_columns:
            # Nullable on purpose: every fight analysed before the roster
            # existed has no owner, and guessing one would be inventing data.
            con.execute("ALTER TABLE fights ADD COLUMN fighter_id INTEGER")
        account_columns = {row[1] for row in con.execute("PRAGMA table_info(accounts)").fetchall()}
        if "plan_override" not in account_columns:
            con.execute("ALTER TABLE accounts ADD COLUMN plan_override TEXT")
        account_migrations = {
            "terms_version": "TEXT",
            "privacy_version": "TEXT",
            "policies_accepted_at": "TEXT",
            "age_confirmed_at": "TEXT",
            "guardian_approval_status": "TEXT NOT NULL DEFAULT 'not_applicable'",
            "marketing_consent": "INTEGER NOT NULL DEFAULT 0",
            "marketing_consent_at": "TEXT",
            "cookie_analytics": "INTEGER NOT NULL DEFAULT 0",
            "cookie_marketing": "INTEGER NOT NULL DEFAULT 0",
            "account_status": "TEXT NOT NULL DEFAULT 'active'",
            "stripe_customer_id": "TEXT",
            "stripe_subscription_id": "TEXT",
            "subscription_status": "TEXT",
            "subscription_period_end": "TEXT",
            "subscription_cancelled_at": "TEXT",
            "email_verified_at": "TEXT",
            "password_login_enabled": "INTEGER NOT NULL DEFAULT 1",
        }
        for column, definition in account_migrations.items():
            if column not in account_columns:
                con.execute(f"ALTER TABLE accounts ADD COLUMN {column} {definition}")
        fight_columns = {row[1] for row in con.execute("PRAGMA table_info(fights)").fetchall()}
        if "video_delete_after" not in fight_columns:
            con.execute("ALTER TABLE fights ADD COLUMN video_delete_after TEXT")
        if "video_deleted_at" not in fight_columns:
            con.execute("ALTER TABLE fights ADD COLUMN video_deleted_at TEXT")
        acceptance_columns = {row[1] for row in con.execute("PRAGMA table_info(legal_acceptances)").fetchall()}
        if "current_status" not in acceptance_columns:
            con.execute("ALTER TABLE legal_acceptances ADD COLUMN current_status TEXT NOT NULL DEFAULT 'accepted'")
        row = con.execute("SELECT id FROM profiles ORDER BY id LIMIT 1").fetchone()
        if row is None:
            now = datetime.now(timezone.utc).isoformat()
            con.execute(
                "INSERT INTO profiles(display_name, photo_path, video_path, notes, created_at, updated_at) VALUES(?,?,?,?,?,?)",
                (SETTINGS.default_profile_name, None, None, "", now, now),
            )


def get_profile(profile_id: int = 1) -> dict | None:
    init_db()
    with connection() as con:
        row = con.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        return dict(row) if row else None


def update_profile(
    profile_id: int,
    display_name: str,
    photo_path: str | None = None,
    video_path: str | None = None,
    notes: str = "",
    default_fighter: str = "A",
    allow_model_training: bool | None = None,
) -> dict:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        current = con.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        if current is None:
            con.execute(
                "INSERT INTO profiles(id, display_name, photo_path, video_path, notes, default_fighter, allow_model_training, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (profile_id, display_name, photo_path, video_path, notes, default_fighter, int(bool(allow_model_training)), now, now),
            )
        else:
            con.execute(
                """UPDATE profiles SET display_name=?, photo_path=COALESCE(?,photo_path),
                   video_path=COALESCE(?,video_path), notes=?, default_fighter=?,
                   allow_model_training=COALESCE(?,allow_model_training), updated_at=? WHERE id=?""",
                (
                    display_name, photo_path, video_path, notes, default_fighter,
                    None if allow_model_training is None else int(bool(allow_model_training)), now, profile_id,
                ),
            )
    return get_profile(profile_id) or {}


def save_fight(
    job_id: str,
    profile_id: int,
    original_name: str,
    video_path: str,
    report_path: str,
    fight_type: str,
    ruleset: str,
    analysis_target: str,
    summary: dict,
    video_delete_after: str | None = None,
    fighter_id: int | None = None,
) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            """
            INSERT INTO fights(job_id, profile_id, original_name, video_path, report_path,
                               fight_type, ruleset, analysis_target, created_at, summary_json, video_delete_after,
                               fighter_id)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET
                report_path=excluded.report_path,
                summary_json=excluded.summary_json
            """,
            (
                job_id,
                profile_id,
                original_name,
                video_path,
                report_path,
                fight_type,
                ruleset,
                analysis_target,
                now,
                json.dumps(summary),
                video_delete_after,
                fighter_id,
            ),
        )


def list_fights(profile_id: int = 1) -> list[dict]:
    init_db()
    with connection() as con:
        rows = con.execute(
            # The fighter's name travels with the fight so callers never have
            # to look it up per row.
            "SELECT fights.*, fighters.name AS fighter_name "
            "FROM fights LEFT JOIN fighters ON fighters.id = fights.fighter_id "
            "WHERE fights.profile_id=? ORDER BY fights.id DESC",
            (profile_id,),
        ).fetchall()
    fights = []
    for row in rows:
        item = dict(row)
        try:
            item["summary"] = json.loads(item.pop("summary_json") or "{}")
        except Exception:
            item["summary"] = {}
        fights.append(item)
    return fights


def list_all_fight_storage() -> list[dict]:
    """Internal cleanup inventory; never expose this across account boundaries."""
    init_db()
    with connection() as con:
        rows = con.execute("SELECT job_id,profile_id,video_path,report_path FROM fights").fetchall()
    return [dict(row) for row in rows]


def get_fight(job_id: str) -> dict | None:
    init_db()
    with connection() as con:
        row = con.execute("SELECT * FROM fights WHERE job_id=?", (job_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    try:
        item["summary"] = json.loads(item.pop("summary_json") or "{}")
    except Exception:
        item["summary"] = {}
    return item


def delete_fight(job_id: str) -> dict | None:
    fight = get_fight(job_id)
    if fight is None:
        return None
    with connection() as con:
        _remove_annotation_sequences(con, job_id)
        con.execute("DELETE FROM annotations WHERE job_id=?", (job_id,))
        con.execute("DELETE FROM fight_reviews WHERE job_id=?", (job_id,))
        con.execute("DELETE FROM report_shares WHERE job_id=?", (job_id,))
        con.execute("DELETE FROM legal_acceptances WHERE resource_id=?", (job_id,))
        con.execute("DELETE FROM fights WHERE job_id=?", (job_id,))
    return fight


def mark_fight_video_deleted(job_id: str, profile_id: int) -> dict | None:
    """Detach the original video while preserving its generated report."""
    fight = get_fight(job_id)
    if fight is None or int(fight["profile_id"]) != int(profile_id):
        return None
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            "UPDATE fights SET video_deleted_at=?,video_path='' WHERE job_id=? AND profile_id=?",
            (now, job_id, int(profile_id)),
        )
    fight["video_deleted_at"] = now
    return fight


def list_expired_fight_videos(now: datetime | None = None) -> list[dict]:
    moment = (now or datetime.now(timezone.utc)).isoformat()
    init_db()
    with connection() as con:
        rows = con.execute(
            """SELECT * FROM fights WHERE video_path<>'' AND video_deleted_at IS NULL
               AND video_delete_after IS NOT NULL AND video_delete_after<=? ORDER BY id""",
            (moment,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["summary"] = json.loads(item.pop("summary_json") or "{}")
        result.append(item)
    return result


def save_annotation(job_id: str, event_time: float, ruleset: str, predicted: dict, corrected: dict) -> int:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            """INSERT INTO annotations(job_id,event_time,ruleset,predicted_json,corrected_json,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?) ON CONFLICT(job_id,event_time) DO UPDATE SET
               ruleset=excluded.ruleset,corrected_json=excluded.corrected_json,
               updated_at=excluded.updated_at""",
            (job_id, float(event_time), ruleset, json.dumps(predicted), json.dumps(corrected), now, now),
        )
        row = con.execute("SELECT id FROM annotations WHERE job_id=? AND event_time=?", (job_id, float(event_time))).fetchone()
        return int(row[0])


def set_annotation_sequence(annotation_id: int, sequence_path: str | None) -> None:
    with connection() as con:
        con.execute("UPDATE annotations SET sequence_path=? WHERE id=?", (sequence_path, annotation_id))


def list_annotations() -> list[dict]:
    init_db()
    with connection() as con:
        rows = con.execute("SELECT * FROM annotations ORDER BY id DESC").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["predicted"] = json.loads(item.pop("predicted_json"))
        item["corrected"] = json.loads(item.pop("corrected_json"))
        result.append(item)
    return result


def get_annotations(job_id: str) -> list[dict]:
    return [item for item in list_annotations() if item["job_id"] == job_id]


def record_legal_acceptance(
    kind: str,
    policy_version: str,
    *,
    profile_id: int | None = None,
    guest_id: str | None = None,
    resource_id: str | None = None,
    metadata: dict | None = None,
    current_status: str = "accepted",
) -> int:
    if not profile_id and not guest_id:
        raise ValueError("A profile or guest identifier is required")
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        cursor = con.execute(
            """INSERT INTO legal_acceptances(
                   profile_id,guest_id,kind,policy_version,resource_id,accepted_at,metadata_json,current_status
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                profile_id, guest_id, kind[:80], policy_version[:40], resource_id, now,
                json.dumps(metadata or {}), current_status[:24],
            ),
        )
        return int(cursor.lastrowid)


def list_legal_acceptances(*, profile_id: int | None = None, guest_id: str | None = None) -> list[dict]:
    init_db()
    if profile_id is None and guest_id is None:
        return []
    field, value = ("profile_id", profile_id) if profile_id is not None else ("guest_id", guest_id)
    with connection() as con:
        rows = con.execute(
            f"SELECT * FROM legal_acceptances WHERE {field}=? ORDER BY id DESC", (value,)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        result.append(item)
    return result


def delete_legal_acceptances_for_resource(resource_id: str) -> None:
    init_db()
    with connection() as con:
        con.execute("DELETE FROM legal_acceptances WHERE resource_id=?", (resource_id,))


def get_fight_review(job_id: str) -> dict:
    init_db()
    with connection() as con:
        row = con.execute("SELECT * FROM fight_reviews WHERE job_id=?", (job_id,)).fetchone()
    return dict(row) if row else {"job_id": job_id, "status": "in_progress", "completed_at": None}


def set_fight_review(job_id: str, profile_id: int, complete: bool) -> dict:
    """Record the owner's explicit full-video review declaration."""
    return set_fight_review_status(job_id, profile_id, "complete" if complete else "in_progress")


def set_fight_review_status(job_id: str, profile_id: int, status: str) -> dict:
    """Record scorecard-only or full-dataset human review completion."""
    if status not in {"in_progress", "scorecard_complete", "complete"}:
        raise ValueError("Invalid fight review status")
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    completed_at = now if status in {"scorecard_complete", "complete"} else None
    with connection() as con:
        con.execute(
            """INSERT INTO fight_reviews(job_id,profile_id,status,updated_at,completed_at)
               VALUES(?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET
               status=excluded.status,updated_at=excluded.updated_at,
               completed_at=excluded.completed_at""",
            (job_id, int(profile_id), status, now, completed_at),
        )
    return get_fight_review(job_id)


def create_account(email: str, password_hash: str) -> dict:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        if con.execute("SELECT id FROM accounts WHERE email=?", (email,)).fetchone():
            raise ValueError("An account with that email already exists.")
        first_account = con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 0
        if first_account:
            profile_id = int(con.execute("SELECT id FROM profiles ORDER BY id LIMIT 1").fetchone()[0])
        else:
            cursor = con.execute(
                "INSERT INTO profiles(display_name,photo_path,video_path,notes,default_fighter,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                ("My Athlete", None, None, "", "A", now, now),
            )
            profile_id = int(cursor.lastrowid)
        cursor = con.execute(
            "INSERT INTO accounts(email,password_hash,profile_id,plan,credits,created_at) VALUES(?,?,?,?,?,?)",
            (email, password_hash, profile_id, "free", 1, now),
        )
        account_id = int(cursor.lastrowid)
    return get_account(account_id) or {}


def create_oauth_account(
    provider: str,
    subject: str,
    email: str,
    password_hash: str,
    display_name: str | None = None,
) -> dict:
    """Atomically create a social-only account and bind its stable provider ID."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        linked = con.execute(
            """SELECT oauth_identities.account_id,accounts.account_status
               FROM oauth_identities JOIN accounts ON accounts.id=oauth_identities.account_id
               WHERE oauth_identities.provider=? AND oauth_identities.subject=?""",
            (provider, subject),
        ).fetchone()
        if linked:
            if linked["account_status"] != "active":
                raise ValueError("This WarriorIQ account is not currently available.")
            account_id = int(linked["account_id"])
        else:
            if con.execute("SELECT id FROM accounts WHERE email=?", (email,)).fetchone():
                raise ValueError(
                    "A WarriorIQ account already uses this email. Sign in with its password first to protect that account."
                )
            first_account = con.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 0
            if first_account:
                profile_id = int(con.execute("SELECT id FROM profiles ORDER BY id LIMIT 1").fetchone()[0])
                if display_name:
                    con.execute(
                        "UPDATE profiles SET display_name=?,updated_at=? WHERE id=?",
                        (display_name, now, profile_id),
                    )
            else:
                cursor = con.execute(
                    "INSERT INTO profiles(display_name,photo_path,video_path,notes,default_fighter,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    (display_name or "My Athlete", None, None, "", "A", now, now),
                )
                profile_id = int(cursor.lastrowid)
            cursor = con.execute(
                """INSERT INTO accounts(
                       email,password_hash,password_login_enabled,profile_id,plan,credits,created_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (email, password_hash, 0, profile_id, "free", 1, now),
            )
            account_id = int(cursor.lastrowid)
            con.execute(
                """INSERT INTO oauth_identities(
                       provider,subject,account_id,email_at_link,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?)""",
                (provider, subject, account_id, email, now, now),
            )
    return get_account(account_id) or {}


def get_account(account_id: int) -> dict | None:
    init_db()
    with connection() as con:
        row = con.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
    return dict(row) if row else None


def get_account_by_email(email: str) -> dict | None:
    init_db()
    with connection() as con:
        row = con.execute("SELECT * FROM accounts WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def get_account_for_oauth_identity(provider: str, subject: str) -> dict | None:
    init_db()
    with connection() as con:
        row = con.execute(
            """SELECT accounts.* FROM oauth_identities
               JOIN accounts ON accounts.id=oauth_identities.account_id
               WHERE oauth_identities.provider=? AND oauth_identities.subject=?
               AND accounts.account_status='active'""",
            (provider, subject),
        ).fetchone()
    return dict(row) if row else None


def link_oauth_identity(provider: str, subject: str, account_id: int, email: str | None) -> None:
    """Bind a provider identity to an account that already exists.

    Only ever called for an address the provider itself has verified. The
    caller owns that check: without it, anyone able to register a victim's
    address at any provider could walk into their WarriorIQ account.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            """INSERT INTO oauth_identities(provider,subject,account_id,email_at_link,created_at,updated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(provider,subject) DO UPDATE SET
                 account_id=excluded.account_id, email_at_link=excluded.email_at_link,
                 updated_at=excluded.updated_at""",
            (provider, subject, int(account_id), email, now, now),
        )


def list_oauth_identities(account_id: int) -> list[dict]:
    init_db()
    with connection() as con:
        rows = con.execute(
            """SELECT provider,subject,email_at_link,created_at,updated_at
               FROM oauth_identities WHERE account_id=? ORDER BY created_at""",
            (int(account_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def policies_outdated(account) -> bool:
    """Whether this account accepted an older policy version than the current one.

    Signing in is not the moment to collect consent: at the login form nobody
    has been identified yet, so there is nothing to compare against and the
    only option is to ask everybody every time. The account row already carries
    the version it accepted, so the question can be asked once, of the people
    it actually applies to, after they are known.
    """
    if not account:
        return False
    try:
        accepted = account["terms_version"]
    except (KeyError, IndexError, TypeError):
        accepted = None
    # Never accepted anything recorded - an account predating the field - is not
    # treated as outdated: it is not evidence that the policy moved.
    return bool(accepted) and str(accepted) != str(SETTINGS.policy_version)


def record_policy_reacceptance(account_id: int) -> dict:
    """Bring an account up to the current policy version."""
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            """UPDATE accounts SET terms_version=?,privacy_version=?,policies_accepted_at=?
               WHERE id=?""",
            (SETTINGS.policy_version, SETTINGS.policy_version, now, int(account_id)),
        )
    return get_account(account_id) or {}


def record_account_signup_acceptance(
    account_id: int,
    *,
    terms_version: str,
    privacy_version: str,
    marketing_consent: bool,
) -> dict:
    """Persist the account contract and optional marketing choice separately."""
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            """UPDATE accounts SET terms_version=?,privacy_version=?,policies_accepted_at=?,
               age_confirmed_at=?,guardian_approval_status='not_applicable',
               marketing_consent=?,marketing_consent_at=? WHERE id=?""",
            (
                terms_version, privacy_version, now, now, int(marketing_consent),
                now if marketing_consent else None, int(account_id),
            ),
        )
    return get_account(account_id) or {}


def update_marketing_consent(account_id: int, enabled: bool) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            "UPDATE accounts SET marketing_consent=?,marketing_consent_at=? WHERE id=?",
            (int(enabled), now, int(account_id)),
        )
    return get_account(account_id) or {}


def update_cookie_preferences(account_id: int, *, analytics: bool, marketing: bool) -> dict:
    with connection() as con:
        con.execute(
            "UPDATE accounts SET cookie_analytics=?,cookie_marketing=? WHERE id=?",
            (int(analytics), int(marketing), int(account_id)),
        )
    return get_account(account_id) or {}


def revoke_account_sessions(account_id: int, keep_token_hash: str | None = None) -> int:
    with connection() as con:
        if keep_token_hash:
            cursor = con.execute(
                "DELETE FROM sessions WHERE account_id=? AND token_hash<>?",
                (int(account_id), keep_token_hash),
            )
        else:
            cursor = con.execute("DELETE FROM sessions WHERE account_id=?", (int(account_id),))
        return int(cursor.rowcount)


def update_password_hash(account_id: int, password_hash: str) -> bool:
    with connection() as con:
        cursor = con.execute(
            """UPDATE accounts SET password_hash=?,password_login_enabled=1
               WHERE id=? AND account_status='active'""",
            (password_hash, int(account_id)),
        )
        if cursor.rowcount:
            con.execute("DELETE FROM sessions WHERE account_id=?", (int(account_id),))
        return cursor.rowcount == 1


def save_password_reset_token(account_id: int, token_hash: str, expires_at: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute("DELETE FROM password_reset_tokens WHERE expires_at<=? OR used_at IS NOT NULL", (now,))
        con.execute(
            "INSERT INTO password_reset_tokens(account_id,token_hash,created_at,expires_at) VALUES(?,?,?,?)",
            (int(account_id), token_hash, now, expires_at),
        )


def consume_password_reset_token(token_hash: str) -> int | None:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            """SELECT id,account_id FROM password_reset_tokens
               WHERE token_hash=? AND used_at IS NULL AND expires_at>?""",
            (token_hash, now),
        ).fetchone()
        if row is None:
            return None
        con.execute("UPDATE password_reset_tokens SET used_at=? WHERE id=?", (now, int(row["id"])))
        return int(row["account_id"])


def save_email_verification_token(account_id: int, token_hash: str, expires_at: str) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            "DELETE FROM email_verification_tokens WHERE account_id=? OR expires_at<=? OR used_at IS NOT NULL",
            (int(account_id), now),
        )
        con.execute(
            "INSERT INTO email_verification_tokens(account_id,token_hash,created_at,expires_at) VALUES(?,?,?,?)",
            (int(account_id), token_hash, now, expires_at),
        )


def consume_email_verification_token(token_hash: str) -> int | None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        row = con.execute(
            """SELECT id,account_id FROM email_verification_tokens
               WHERE token_hash=? AND used_at IS NULL AND expires_at>?""",
            (token_hash, now),
        ).fetchone()
        if not row:
            return None
        con.execute("UPDATE email_verification_tokens SET used_at=? WHERE id=?", (now, int(row["id"])))
        con.execute("UPDATE accounts SET email_verified_at=? WHERE id=?", (now, int(row["account_id"])))
        return int(row["account_id"])


def mark_email_verified(account_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            "UPDATE accounts SET email_verified_at=COALESCE(email_verified_at,?) WHERE id=?",
            (now, int(account_id)),
        )


def set_account_status(account_id: int, status: str) -> bool:
    if status not in {"active", "suspended", "disabled"}:
        raise ValueError("Invalid account status")
    with connection() as con:
        cursor = con.execute("UPDATE accounts SET account_status=? WHERE id=?", (status, int(account_id)))
        if status != "active":
            con.execute("DELETE FROM sessions WHERE account_id=?", (int(account_id),))
        return cursor.rowcount == 1


def list_accounts(search: str = "", limit: int = 100) -> list[dict]:
    init_db()
    query = f"%{search.strip().lower()}%"
    with connection() as con:
        rows = con.execute(
            """SELECT id,email,profile_id,plan,plan_override,account_status,created_at,
               terms_version,privacy_version,policies_accepted_at,marketing_consent
               FROM accounts WHERE lower(email) LIKE ? ORDER BY id DESC LIMIT ?""",
            (query, max(1, min(int(limit), 250))),
        ).fetchall()
    return [dict(row) for row in rows]


def set_plan_override(account_id: int, plan: str | None) -> bool:
    """Set a permanent local entitlement without altering billing state."""
    init_db()
    normalized = None if plan is None else str(plan).strip().lower()
    if normalized is not None and normalized not in PLANS:
        raise ValueError(f"Unknown plan override: {plan}")
    with connection() as con:
        cursor = con.execute(
            "UPDATE accounts SET plan_override=? WHERE id=?",
            (normalized, int(account_id)),
        )
        return cursor.rowcount == 1


def _effective_plan_key(account) -> str:
    if not account:
        return "free"
    email = account["email"] if "email" in account.keys() else None
    return effective_plan_key(account["plan"], account["plan_override"], email)


def save_session(account_id: int, token_hash: str, expires_at: str) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute("DELETE FROM sessions WHERE expires_at<=?", (now,))
        con.execute(
            "INSERT INTO sessions(account_id,token_hash,created_at,expires_at) VALUES(?,?,?,?)",
            (account_id, token_hash, now, expires_at),
        )


def account_for_session(token_hash: str) -> dict | None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        row = con.execute(
            """SELECT accounts.id,accounts.email,accounts.profile_id,accounts.plan,accounts.plan_override,
               accounts.credits,accounts.created_at,accounts.account_status,accounts.marketing_consent,
               accounts.marketing_consent_at,accounts.cookie_analytics,accounts.cookie_marketing,
               accounts.terms_version,accounts.privacy_version,accounts.policies_accepted_at,
               accounts.stripe_customer_id,accounts.stripe_subscription_id,accounts.subscription_status,
               accounts.subscription_period_end,accounts.subscription_cancelled_at,
               accounts.email_verified_at,accounts.password_login_enabled
               FROM sessions JOIN accounts ON accounts.id=sessions.account_id
               WHERE sessions.token_hash=? AND sessions.expires_at>? AND accounts.account_status='active'""",
            (token_hash, now),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token_hash: str) -> None:
    with connection() as con:
        con.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))


def apply_checkout_event(
    event_id: str,
    event_type: str,
    account_id: int,
    plan: str,
    credits: int,
    *,
    customer_id: str | None = None,
    subscription_id: str | None = None,
    subscription_status: str | None = None,
    period_end: str | None = None,
) -> bool:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        try:
            con.execute(
                "INSERT INTO payment_events(event_id,event_type,received_at) VALUES(?,?,?)",
                (event_id, event_type, now),
            )
        except sqlite3.IntegrityError:
            return False
        con.execute(
            """UPDATE accounts SET plan=?,credits=MAX(credits,?),
               stripe_customer_id=COALESCE(?,stripe_customer_id),
               stripe_subscription_id=COALESCE(?,stripe_subscription_id),
               subscription_status=COALESCE(?,subscription_status),
               subscription_period_end=COALESCE(?,subscription_period_end) WHERE id=?""",
            (
                plan, max(0, int(credits)), customer_id, subscription_id,
                subscription_status, period_end, account_id,
            ),
        )
    return True


def record_subscription_action(
    account_id: int,
    action_type: str,
    status: str,
    *,
    effective_at: str | None = None,
    provider_reference: str | None = None,
    metadata: dict | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        cursor = con.execute(
            """INSERT INTO subscription_actions(
               account_id,action_type,status,requested_at,effective_at,provider_reference,metadata_json
               ) VALUES(?,?,?,?,?,?,?)""",
            (
                int(account_id), action_type[:40], status[:32], now, effective_at,
                provider_reference, json.dumps(metadata or {}),
            ),
        )
        if action_type == "cancel" and status in {"scheduled", "complete"}:
            con.execute(
                "UPDATE accounts SET subscription_status=?,subscription_cancelled_at=? WHERE id=?",
                ("cancel_at_period_end" if status == "scheduled" else "cancelled", now, int(account_id)),
            )
        row = con.execute("SELECT * FROM subscription_actions WHERE id=?", (cursor.lastrowid,)).fetchone()
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json") or "{}")
    return result


def list_subscription_actions(account_id: int) -> list[dict]:
    init_db()
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM subscription_actions WHERE account_id=? ORDER BY id DESC",
            (int(account_id),),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        result.append(item)
    return result


def record_athlete_session(fighter_id: int, signals, *, occurred_at: str | None = None) -> dict:
    """File one analysed fight against an athlete, and move the rating if it may.

    Recording and rating are separate on purpose. Every session is kept,
    because the history is worth having whatever the evidence gate said; only
    a session carrying signals the scorecard already trusts moves the number.
    A session that could not be rated is stored with rated=0 and an empty
    rated_on, so "why did the rating not change" has an answer in the row
    rather than in somebody's memory.

    Re-filing the same fight replaces its row and leaves the rating alone. An
    analysis re-run is the same fight, and letting it score twice would let
    anyone inflate a rating by pressing the button again.
    """
    from core.rating import (
        BASELINE_RATING, next_rating, performance_score,
    )

    now = occurred_at or datetime.now(timezone.utc).isoformat()
    score = performance_score(signals)
    with connection() as con:
        existing = con.execute(
            "SELECT id FROM athlete_sessions WHERE fighter_id=? AND job_id=?",
            (int(fighter_id), signals.job_id),
        ).fetchone()
        row = con.execute(
            "SELECT rating, rated_sessions, total_sessions FROM athlete_ratings WHERE fighter_id=?",
            (int(fighter_id),),
        ).fetchone()
        rating = float(row["rating"]) if row else BASELINE_RATING
        rated_sessions = int(row["rated_sessions"]) if row else 0
        total_sessions = int(row["total_sessions"]) if row else 0

        if existing is not None:
            con.execute(
                """UPDATE athlete_sessions SET occurred_at=?,accuracy=?,attempts=?,landed=?,
                   output_per_round=?,defensive_lapses=?,pose_coverage=?,evidence_trusted=?
                   WHERE id=?""",
                (now, signals.accuracy, int(signals.attempts), int(signals.landed),
                 signals.output_per_round, signals.defensive_lapses, signals.pose_coverage,
                 1 if signals.evidence_trusted else 0, int(existing["id"])),
            )
            return {"rating": rating, "rated_sessions": rated_sessions,
                    "total_sessions": total_sessions, "rated": False, "delta": 0.0,
                    "reason": "this fight is already filed against this athlete"}

        after, delta = (rating, 0.0)
        if score is not None:
            after, delta = next_rating(rating, rated_sessions, score)

        con.execute(
            """INSERT INTO athlete_sessions(
               fighter_id,job_id,occurred_at,accuracy,attempts,landed,output_per_round,
               defensive_lapses,pose_coverage,evidence_trusted,rated,rated_on,
               rating_before,rating_after,rating_delta
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (int(fighter_id), signals.job_id, now, signals.accuracy, int(signals.attempts),
             int(signals.landed), signals.output_per_round, signals.defensive_lapses,
             signals.pose_coverage, 1 if signals.evidence_trusted else 0,
             1 if score is not None else 0, ",".join(signals.used),
             rating, after, delta),
        )
        total_sessions += 1
        if score is not None:
            rated_sessions += 1
        con.execute(
            """INSERT INTO athlete_ratings(fighter_id,rating,rated_sessions,total_sessions,updated_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(fighter_id) DO UPDATE SET
                 rating=excluded.rating, rated_sessions=excluded.rated_sessions,
                 total_sessions=excluded.total_sessions, updated_at=excluded.updated_at""",
            (int(fighter_id), after, rated_sessions, total_sessions, now),
        )
    return {"rating": after, "rated_sessions": rated_sessions, "total_sessions": total_sessions,
            "rated": score is not None, "delta": delta,
            "reason": "" if score is not None else "no signal the scorecard trusts"}


def athlete_rating(fighter_id: int) -> dict:
    """The athlete's current standing, with the label to show for it."""
    from core.rating import BASELINE_RATING, describe, is_provisional

    with connection() as con:
        row = con.execute(
            "SELECT * FROM athlete_ratings WHERE fighter_id=?", (int(fighter_id),)).fetchone()
    if row is None:
        return {"fighter_id": int(fighter_id), "rating": BASELINE_RATING, "rated_sessions": 0,
                "total_sessions": 0, "provisional": True, "label": describe(BASELINE_RATING, 0)}
    rated = int(row["rated_sessions"])
    return {
        "fighter_id": int(fighter_id), "rating": float(row["rating"]),
        "rated_sessions": rated, "total_sessions": int(row["total_sessions"]),
        "provisional": is_provisional(rated), "label": describe(float(row["rating"]), rated),
        "updated_at": row["updated_at"],
    }


def athlete_history(fighter_id: int, limit: int = 50) -> list[dict]:
    """Every session filed against an athlete, newest first."""
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM athlete_sessions WHERE fighter_id=? ORDER BY id DESC LIMIT ?",
            (int(fighter_id), int(limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def record_analysis_failure(
    job_id: str,
    reason: str,
    *,
    detail: str = "",
    account_id: int | None = None,
    stage: str = "analysis",
) -> int:
    """Keep a failed analysis where it can be looked at later.

    The worker already catches, releases the visitor's allowance and marks the
    job `error`, so one bad upload never stalled the queue - that part was
    never broken. What it did not do was keep the reason anywhere but the log,
    so "why did these four fail" could only be answered by someone with shell
    access reading files, and only until the logs rotated.

    A repeat of the same job increments `attempts` rather than adding a row,
    so a video that fails every retry reads as one stubborn problem instead of
    five separate ones.
    """
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        existing = con.execute(
            "SELECT id, attempts FROM analysis_failures WHERE job_id=? AND reviewed_at IS NULL"
            " ORDER BY id DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        if existing is not None:
            con.execute(
                "UPDATE analysis_failures SET attempts=?, reason=?, detail=?, occurred_at=?"
                " WHERE id=?",
                (int(existing["attempts"]) + 1, reason[:120], detail[:500], now, int(existing["id"])),
            )
            return int(existing["id"])
        cursor = con.execute(
            """INSERT INTO analysis_failures(
               job_id,account_id,stage,reason,detail,attempts,occurred_at
               ) VALUES(?,?,?,?,?,?,?)""",
            (job_id, account_id, stage[:40], reason[:120], detail[:500], 1, now),
        )
        return int(cursor.lastrowid)


def list_analysis_failures(limit: int = 50, include_reviewed: bool = False) -> list[dict]:
    """The dead-letter queue, newest first."""
    clause = "" if include_reviewed else " WHERE reviewed_at IS NULL"
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM analysis_failures%s ORDER BY id DESC LIMIT ?" % clause,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_analysis_failure_reviewed(failure_id: int) -> bool:
    """Take one off the queue once somebody has dealt with it."""
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        cursor = con.execute(
            "UPDATE analysis_failures SET reviewed_at=? WHERE id=? AND reviewed_at IS NULL",
            (now, int(failure_id)),
        )
        return cursor.rowcount > 0


def record_security_event(
    event_type: str,
    *,
    account_id: int | None = None,
    severity: str = "info",
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict | None = None,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        cursor = con.execute(
            """INSERT INTO security_events(
               account_id,event_type,severity,resource_type,resource_id,occurred_at,metadata_json
               ) VALUES(?,?,?,?,?,?,?)""",
            (
                account_id, event_type[:80], severity[:16], resource_type, resource_id,
                now, json.dumps(metadata or {}),
            ),
        )
        return int(cursor.lastrowid)


def list_security_events(limit: int = 200) -> list[dict]:
    init_db()
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM security_events ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        result.append(item)
    return result


def create_moderation_report(report_type: str, reporter_email: str, details: str, resource_id: str = "") -> int:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        cursor = con.execute(
            """INSERT INTO moderation_reports(report_type,reporter_email,resource_id,details,status,created_at)
               VALUES(?,?,?,?,?,?)""",
            (report_type[:40], reporter_email[:320], resource_id[:120] or None, details[:6000], "open", now),
        )
        return int(cursor.lastrowid)


def list_moderation_reports(limit: int = 200) -> list[dict]:
    init_db()
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM moderation_reports ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
    return [dict(row) for row in rows]


def resolve_moderation_report(report_id: int) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        cursor = con.execute(
            "UPDATE moderation_reports SET status='resolved',resolved_at=? WHERE id=?",
            (now, int(report_id)),
        )
        return cursor.rowcount == 1


def queue_outbound_message(account_id: int | None, message_type: str, recipient: str, payload: dict) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        cursor = con.execute(
            """INSERT INTO outbound_messages(account_id,message_type,recipient,status,created_at,payload_json)
               VALUES(?,?,?,?,?,?)""",
            (account_id, message_type[:60], recipient[:320], "queued", now, json.dumps(payload)),
        )
        return int(cursor.lastrowid)


def mark_outbound_message_sent(message_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            "UPDATE outbound_messages SET status='sent',sent_at=? WHERE id=?",
            (now, int(message_id)),
        )


def list_outbound_messages(account_id: int | None = None) -> list[dict]:
    init_db()
    with connection() as con:
        if account_id is None:
            rows = con.execute("SELECT * FROM outbound_messages ORDER BY id DESC").fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM outbound_messages WHERE account_id=? ORDER BY id DESC", (int(account_id),)
            ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        result.append(item)
    return result


def save_completed_analysis(job_id: str, job: dict, report: dict, report_path: str) -> None:
    """Persist the report only after its analysis generation has been committed."""
    if not job.get("persist_result"):
        return
    # Imported here: core.db is loaded by everything, and core.report pulls in
    # the scoring and coaching modules that nothing else in db needs.
    from core.report import identity_tracking
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
            **{key: report.get(key, {}) for key in (
                "video", "setup", "integrity", "metrics", "statistics", "coaching", "training_plan",
            )},
            # What the identity gate needs, so Progress re-applies it.
            "tracking": identity_tracking(report.get("tracking", {})),
        },
    }
    save_fight(
        job_id=job_id, profile_id=int(job.get("profile_id", 0)),
        original_name=str(job.get("original_name") or "Fight video"),
        video_path=str(job["video_path"]), report_path=report_path,
        fight_type=str(job["fight_type"]), ruleset=str(job["ruleset"]),
        analysis_target=str(job.get("focus_fighter") or "A"), summary=summary,
        fighter_id=job.get("fighter_id"),
        video_delete_after=(datetime.now(timezone.utc) + timedelta(days=SETTINGS.saved_video_retention_days)).isoformat(),
    )


def consume_credit(account_id: int) -> bool:
    init_db()
    with connection() as con:
        cursor = con.execute(
            "UPDATE accounts SET credits=credits-1 WHERE id=? AND credits>0",
            (account_id,),
        )
        return cursor.rowcount == 1


def refund_credit(account_id: int) -> None:
    init_db()
    with connection() as con:
        con.execute("UPDATE accounts SET credits=credits+1 WHERE id=?", (account_id,))


def _usage_period(plan: dict, moment: datetime) -> tuple[str, int | None]:
    if plan.get("unlimited"):
        return "unlimited", None
    if plan.get("daily_limit") is not None:
        return f"day:{moment.date().isoformat()}", int(plan["daily_limit"])
    if plan.get("monthly_limit") is not None:
        return f"month:{moment.strftime('%Y-%m')}", int(plan["monthly_limit"])
    return "unavailable", 0


def reserve_analysis(account_id: int, job_id: str, now: datetime | None = None) -> bool:
    """Atomically reserve one analysis against the account's current plan."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    with connection() as con:
        con.execute("BEGIN IMMEDIATE")
        existing = con.execute(
            "SELECT account_id FROM analysis_usage WHERE job_id=?", (job_id,),
        ).fetchone()
        if existing:
            return int(existing["account_id"]) == int(account_id)
        account = con.execute("SELECT plan,plan_override,email FROM accounts WHERE id=?", (account_id,)).fetchone()
        if not account:
            return False
        plan = plan_for_key(_effective_plan_key(account))
        period_key, limit = _usage_period(plan, moment)
        if limit is not None:
            used = int(con.execute(
                "SELECT COUNT(*) FROM analysis_usage WHERE account_id=? AND period_key=?",
                (account_id, period_key),
            ).fetchone()[0])
            if used >= limit:
                return False
        con.execute(
            "INSERT INTO analysis_usage(job_id,account_id,period_key,created_at) VALUES(?,?,?,?)",
            (job_id, account_id, period_key, moment.astimezone(timezone.utc).isoformat()),
        )
        return True


def release_analysis(account_id: int, job_id: str) -> bool:
    """Release a failed analysis so it does not consume the user's allowance."""
    with connection() as con:
        cursor = con.execute(
            "DELETE FROM analysis_usage WHERE account_id=? AND job_id=?", (account_id, job_id),
        )
        return cursor.rowcount == 1


def analysis_allowance(account_id: int, now: datetime | None = None) -> dict:
    """Return the current period allowance for display and API decisions."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    with connection() as con:
        account = con.execute("SELECT plan,plan_override,email FROM accounts WHERE id=?", (account_id,)).fetchone()
        plan = plan_for_key(_effective_plan_key(account))
        period_key, limit = _usage_period(plan, moment)
        used = int(con.execute(
            "SELECT COUNT(*) FROM analysis_usage WHERE account_id=? AND period_key=?",
            (account_id, period_key),
        ).fetchone()[0])
    return {
        "plan": plan,
        "period_key": period_key,
        "used": used,
        "limit": limit,
        "remaining": None if limit is None else max(0, limit - used),
    }


def add_assignment(profile_id: int, title: str, detail: str) -> int:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        existing = con.execute(
            """SELECT id FROM coach_assignments
               WHERE profile_id=? AND title=? AND detail=? AND status='active'
               ORDER BY id DESC LIMIT 1""",
            (profile_id, title, detail),
        ).fetchone()
        if existing:
            return int(existing["id"])
        cursor = con.execute(
            "INSERT INTO coach_assignments(profile_id,title,detail,status,created_at) VALUES(?,?,?,?,?)",
            (profile_id, title, detail, "active", now),
        )
        return int(cursor.lastrowid)


def list_assignments(profile_id: int) -> list[dict]:
    init_db()
    with connection() as con:
        rows = con.execute(
            "SELECT * FROM coach_assignments WHERE profile_id=? ORDER BY status ASC,id DESC",
            (profile_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def toggle_assignment(assignment_id: int, profile_id: int) -> bool:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        row = con.execute(
            "SELECT status FROM coach_assignments WHERE id=? AND profile_id=?",
            (assignment_id, profile_id),
        ).fetchone()
        if not row:
            return False
        status = "complete" if row["status"] == "active" else "active"
        completed_at = now if status == "complete" else None
        con.execute(
            "UPDATE coach_assignments SET status=?,completed_at=? WHERE id=? AND profile_id=?",
            (status, completed_at, assignment_id, profile_id),
        )
    return True


def save_report_share(job_id: str, profile_id: int, token_hash: str, expires_at: str) -> None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            "INSERT INTO report_shares(job_id,profile_id,token_hash,created_at,expires_at) VALUES(?,?,?,?,?)",
            (job_id, profile_id, token_hash, now, expires_at),
        )


def get_report_share(token_hash: str) -> dict | None:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        row = con.execute(
            "SELECT * FROM report_shares WHERE token_hash=? AND revoked_at IS NULL AND expires_at>?",
            (token_hash, now),
        ).fetchone()
    return dict(row) if row else None


def list_active_report_shares(job_id: str, profile_id: int) -> list[dict]:
    """Coach links that are still live for this fight, soonest to expire first.

    The token itself is only ever stored as a digest, so a link can be counted
    and dated here but never shown again after the moment it was created.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        rows = con.execute(
            "SELECT created_at,expires_at FROM report_shares"
            " WHERE job_id=? AND profile_id=? AND revoked_at IS NULL AND expires_at>?"
            " ORDER BY expires_at",
            (job_id, profile_id, now),
        ).fetchall()
    return [dict(row) for row in rows]


def revoke_report_shares(job_id: str, profile_id: int) -> int:
    """Kill every live coach link for this fight and report how many died.

    Expired links are excluded deliberately. They already fail to open, so
    sweeping them in would inflate the count the report page shows and tell
    the athlete three links were killed when only two could still be used.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        cursor = con.execute(
            "UPDATE report_shares SET revoked_at=?"
            " WHERE job_id=? AND profile_id=? AND revoked_at IS NULL AND expires_at>?",
            (now, job_id, profile_id, now),
        )
        return int(cursor.rowcount or 0)


def delete_account(account_id: int) -> dict | None:
    account = get_account(account_id)
    if not account:
        return None
    profile_id = int(account["profile_id"])
    profile = get_profile(profile_id)
    fights = list_fights(profile_id)
    with connection() as con:
        for fight in fights:
            _remove_annotation_sequences(con, fight["job_id"])
            con.execute("DELETE FROM annotations WHERE job_id=?", (fight["job_id"],))
            con.execute("DELETE FROM fight_reviews WHERE job_id=?", (fight["job_id"],))
            con.execute("DELETE FROM report_shares WHERE job_id=?", (fight["job_id"],))
        con.execute("DELETE FROM fights WHERE profile_id=?", (profile_id,))
        # The roster holds the names of real athletes, entered by the coach
        # who is deleting this workspace - and some of them are minors. It
        # outlived the account it belonged to. Deleted after the fights that
        # reference it, so nothing is left pointing at a row that is gone.
        # Before the roster goes: athlete history and ratings hang off
        # fighter_id, and once the fighters are gone there is nothing left to
        # match them against.
        con.execute(
            "DELETE FROM athlete_sessions WHERE fighter_id IN"
            " (SELECT id FROM fighters WHERE profile_id=?)", (profile_id,))
        con.execute(
            "DELETE FROM athlete_ratings WHERE fighter_id IN"
            " (SELECT id FROM fighters WHERE profile_id=?)", (profile_id,))
        con.execute("DELETE FROM fighters WHERE profile_id=?", (profile_id,))
        con.execute("DELETE FROM coach_assignments WHERE profile_id=?", (profile_id,))
        con.execute("DELETE FROM legal_acceptances WHERE profile_id=?", (profile_id,))
        con.execute("DELETE FROM analysis_usage WHERE account_id=?", (account_id,))
        con.execute("DELETE FROM upload_leases WHERE account_id=?", (account_id,))
        # The dead-letter queue carries account_id, so it holds account data
        # and has to go with everything else. Caught by
        # test_every_table_holding_account_data_is_cleared, which is the whole
        # reason that test enumerates tables rather than naming them.
        con.execute("DELETE FROM analysis_failures WHERE account_id=?", (account_id,))
        con.execute("DELETE FROM plan_interest WHERE account_id=?", (account_id,))
        con.execute("DELETE FROM subscription_actions WHERE account_id=?", (account_id,))
        con.execute("DELETE FROM outbound_messages WHERE account_id=?", (account_id,))
        con.execute("DELETE FROM password_reset_tokens WHERE account_id=?", (account_id,))
        con.execute("DELETE FROM email_verification_tokens WHERE account_id=?", (account_id,))
        con.execute("DELETE FROM oauth_identities WHERE account_id=?", (account_id,))
        con.execute(
            "UPDATE security_events SET account_id=NULL,metadata_json='{}' WHERE account_id=?",
            (account_id,),
        )
        con.execute("DELETE FROM sessions WHERE account_id=?", (account_id,))
        con.execute("DELETE FROM accounts WHERE id=?", (account_id,))
        con.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
    return {"account": account, "profile": profile, "fights": fights}


def list_fighters(profile_id: int, include_archived: bool = False) -> list[dict]:
    """The roster for a workspace, newest name last so the list reads stably."""
    init_db()
    clause = "" if include_archived else " AND archived=0"
    with connection() as con:
        rows = con.execute(
            f"SELECT * FROM fighters WHERE profile_id=?{clause} ORDER BY name COLLATE NOCASE",
            (profile_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_fighter(profile_id: int, name: str) -> dict | None:
    """Add a fighter, or return the existing one of that name.

    Names are the coach's own labels, so the same name twice is the same
    person rather than an error worth showing anybody.
    """
    cleaned = " ".join(str(name or "").split())[:80]
    if not cleaned:
        return None
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        existing = con.execute(
            "SELECT * FROM fighters WHERE profile_id=? AND name=? COLLATE NOCASE",
            (profile_id, cleaned),
        ).fetchone()
        if existing:
            # Same person, however it was typed. Un-archive rather than insert:
            # a second row would take a second seat and split their fights
            # across two trends.
            con.execute("UPDATE fighters SET archived=0 WHERE id=?", (existing["id"],))
            return dict(existing) | {"archived": 0}
        con.execute(
            "INSERT INTO fighters(profile_id, name, created_at) VALUES(?,?,?)",
            (profile_id, cleaned, now),
        )
        row = con.execute(
            "SELECT * FROM fighters WHERE profile_id=? AND name=? COLLATE NOCASE",
            (profile_id, cleaned),
        ).fetchone()
    return dict(row) if row else None


def get_fighter(profile_id: int, fighter_id: int) -> dict | None:
    """Always scoped to the workspace, so an id from elsewhere resolves to nothing."""
    init_db()
    with connection() as con:
        row = con.execute(
            "SELECT * FROM fighters WHERE id=? AND profile_id=?", (int(fighter_id), profile_id),
        ).fetchone()
    return dict(row) if row else None


def archive_fighter(profile_id: int, fighter_id: int) -> bool:
    """Archived, never deleted: their fights still refer to them."""
    init_db()
    with connection() as con:
        changed = con.execute(
            "UPDATE fighters SET archived=1 WHERE id=? AND profile_id=?",
            (int(fighter_id), profile_id),
        ).rowcount
    return bool(changed)


def assign_fighter_to_fight(profile_id: int, job_id: str, fighter_id: int | None) -> bool:
    """File an existing fight against a fighter, or unfile it.

    Both sides are scoped to the workspace, so neither a fight nor a fighter
    from somebody else's account can be reached by guessing an id. Passing None
    clears the assignment, which is the honest way back from a mistake: a fight
    filed against the wrong person is worse than one filed against nobody,
    because the wrong one silently feeds that fighter's trend.
    """
    init_db()
    with connection() as con:
        if fighter_id is not None:
            owned = con.execute(
                "SELECT 1 FROM fighters WHERE id=? AND profile_id=?",
                (int(fighter_id), profile_id),
            ).fetchone()
            if not owned:
                return False
        changed = con.execute(
            "UPDATE fights SET fighter_id=? WHERE job_id=? AND profile_id=?",
            (int(fighter_id) if fighter_id is not None else None, job_id, profile_id),
        ).rowcount
    return bool(changed)


def set_account_type(profile_id: int, account_type: str) -> str:
    """Athlete or coach. Anything else is treated as athlete."""
    chosen = "coach" if str(account_type).strip().lower() == "coach" else "athlete"
    init_db()
    with connection() as con:
        con.execute("UPDATE profiles SET account_type=? WHERE id=?", (chosen, profile_id))
    return chosen


def record_plan_interest(account_id: int, plan_key: str) -> bool:
    """Note that this account wants to hear when a plan opens.

    Returns True when this is a new request. Asking again is the same request,
    so the timestamp moves and nothing is added: a count of rows here has to
    mean "people who want this", or it is not worth keeping.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        existing = con.execute(
            "SELECT id FROM plan_interest WHERE account_id=? AND plan_key=?",
            (int(account_id), str(plan_key)),
        ).fetchone()
        if existing:
            con.execute(
                "UPDATE plan_interest SET created_at=? WHERE id=?", (now, int(existing["id"])))
            return False
        con.execute(
            "INSERT INTO plan_interest(account_id,plan_key,created_at) VALUES(?,?,?)",
            (int(account_id), str(plan_key), now),
        )
    return True


def plans_wanted_by(account_id: int) -> set[str]:
    """Which plans this account has already asked about, for the page to mark."""
    init_db()
    with connection() as con:
        rows = con.execute(
            "SELECT plan_key FROM plan_interest WHERE account_id=?", (int(account_id),)
        ).fetchall()
    return {str(row["plan_key"]) for row in rows}


def plan_interest_counts() -> dict[str, int]:
    """How many accounts asked for each plan. For the operator, not the visitor."""
    init_db()
    with connection() as con:
        rows = con.execute(
            "SELECT plan_key, COUNT(*) AS total FROM plan_interest GROUP BY plan_key"
        ).fetchall()
    return {str(row["plan_key"]): int(row["total"]) for row in rows}


# Counting a page view must not cost more than serving one.
#
# Every write in this module opens a connection and closes it again, and
# closing the last open connection checkpoints the WAL. Measured on the
# development machine: 19 ms for one counted view - 0.2 ms of which is the
# statement - against the 8-9 ms the app takes to build the page being
# counted. Everywhere else in this module that cost is paid once per action a
# person took. Here it would be paid on every page view, so views are added up
# in memory and written in one batch when the buffer is old enough or wide
# enough.
#
# A crash loses at most FLUSH_SECONDS of counts. For "is anyone visiting the
# site" that is a rounding error; it is also the reason nothing that has to be
# exact should ever be counted this way.
_PAGE_VIEW_FLUSH_SECONDS = 30.0
_PAGE_VIEW_FLUSH_ROWS = 100
_pending_page_views: collections.Counter = collections.Counter()
_page_view_lock = threading.Lock()
_page_views_flushed_at = 0.0


def record_page_view(path: str, *, day: str | None = None) -> None:
    """Note one view of this path. Written to the database in a batch.

    Nothing about who asked is recorded - see the table comment. Days are UTC
    so a count cannot move when the host's timezone does.
    """
    global _page_views_flushed_at
    stamp = str(day or datetime.now(timezone.utc).strftime("%Y-%m-%d"))[:10]
    now = time.monotonic()
    with _page_view_lock:
        _pending_page_views[(stamp, str(path)[:200])] += 1
        if not _page_views_flushed_at:
            _page_views_flushed_at = now
        due = (now - _page_views_flushed_at >= _PAGE_VIEW_FLUSH_SECONDS
               or len(_pending_page_views) >= _PAGE_VIEW_FLUSH_ROWS)
    if due:
        flush_page_views()


def flush_page_views() -> int:
    """Write everything counted since the last write. Returns views written.

    A failed write puts the counts back rather than dropping them, so a locked
    database delays the numbers instead of losing them.
    """
    global _page_views_flushed_at
    with _page_view_lock:
        pending = list(_pending_page_views.items())
        _pending_page_views.clear()
        _page_views_flushed_at = time.monotonic()
    if not pending:
        return 0
    try:
        # Inside the try, not before it. The buffer has already been emptied by
        # this point, so anything that raises between here and the write loses
        # the counts it was holding - which is what init_db() raising did.
        init_db()
        with connection() as con:
            con.executemany(
                "INSERT INTO page_views(day,path,views) VALUES(?,?,?) "
                "ON CONFLICT(day,path) DO UPDATE SET views=views+excluded.views",
                [(day, path, count) for (day, path), count in pending],
            )
    except Exception:
        with _page_view_lock:
            for key, count in pending:
                _pending_page_views[key] += count
        raise
    return sum(count for _key, count in pending)


def _flush_page_views_at_exit() -> None:
    """One last write on the way out, and never a traceback on the way out.

    By interpreter shutdown the database can be gone - at the end of a test run
    the temporary directory holding it has already been removed - and an atexit
    handler that raises prints its traceback after everything else has
    finished, where it reads as a crash in whatever ran last. Losing the last
    few counts is the lesser of the two.
    """
    try:
        flush_page_views()
    except Exception:
        LOGGER.debug("page_views_not_flushed_at_exit", exc_info=True)


atexit.register(_flush_page_views_at_exit)


def page_view_summary(days: int = 30) -> dict:
    """Recent traffic, for the operator: per day, per page, and the total.

    Days with no visitors are returned as zero rather than left out. A gap in
    a list of dates reads as a missing measurement; a zero is the measurement.

    Flushes first, so the operator is never shown a number that is behind what
    has already been counted.
    """
    flush_page_views()
    init_db()
    span = max(1, int(days))
    first = (datetime.now(timezone.utc) - timedelta(days=span - 1)).strftime("%Y-%m-%d")
    with connection() as con:
        per_day = {
            str(row["day"]): int(row["views"]) for row in con.execute(
                "SELECT day, SUM(views) AS views FROM page_views "
                "WHERE day >= ? GROUP BY day", (first,)).fetchall()
        }
        pages = [
            {"path": str(row["path"]), "views": int(row["views"])}
            for row in con.execute(
                "SELECT path, SUM(views) AS views FROM page_views "
                "WHERE day >= ? GROUP BY path ORDER BY views DESC, path", (first,)).fetchall()
        ]
    today = datetime.now(timezone.utc)
    timeline = []
    for offset in range(span - 1, -1, -1):
        stamp = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        timeline.append({"day": stamp, "views": per_day.get(stamp, 0)})
    return {"days": timeline, "pages": pages, "total": sum(per_day.values())}
