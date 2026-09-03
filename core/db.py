from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from core.config import DATASET, DB_PATH, SETTINGS
from core.payments import PLANS, effective_plan_key, plan_for_key


@contextmanager
def connection():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
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


def list_oauth_identities(account_id: int) -> list[dict]:
    init_db()
    with connection() as con:
        rows = con.execute(
            """SELECT provider,subject,email_at_link,created_at,updated_at
               FROM oauth_identities WHERE account_id=? ORDER BY created_at""",
            (int(account_id),),
        ).fetchall()
    return [dict(row) for row in rows]


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


def revoke_report_shares(job_id: str, profile_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connection() as con:
        con.execute(
            "UPDATE report_shares SET revoked_at=? WHERE job_id=? AND profile_id=? AND revoked_at IS NULL",
            (now, job_id, profile_id),
        )


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
        con.execute("DELETE FROM coach_assignments WHERE profile_id=?", (profile_id,))
        con.execute("DELETE FROM legal_acceptances WHERE profile_id=?", (profile_id,))
        con.execute("DELETE FROM analysis_usage WHERE account_id=?", (account_id,))
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
        con.execute(
            "INSERT INTO fighters(profile_id, name, created_at) VALUES(?,?,?) "
            "ON CONFLICT(profile_id, name) DO UPDATE SET archived=0",
            (profile_id, cleaned, now),
        )
        row = con.execute(
            "SELECT * FROM fighters WHERE profile_id=? AND name=?", (profile_id, cleaned),
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
