from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone

from core.db import account_for_session, create_account, delete_session, get_account_by_email, save_session


PASSWORD_ITERATIONS = 600_000
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def valid_email(value: str) -> bool:
    email = normalize_email(value)
    return len(email) <= 320 and bool(EMAIL_PATTERN.fullmatch(email))


def valid_password(value: str) -> bool:
    return 10 <= len(value or "") <= 1024


def hash_password(password: str) -> str:
    if not valid_password(password):
        raise ValueError("Password must contain between 10 and 1,024 characters.")
    salt = secrets.token_bytes(18)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def session_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def register(email: str, password: str) -> dict:
    email = normalize_email(email)
    if not valid_email(email):
        raise ValueError("Enter a valid email address.")
    return create_account(email, hash_password(password))


def authenticate(email: str, password: str) -> dict | None:
    if not valid_email(email) or not valid_password(password):
        return None
    account = get_account_by_email(normalize_email(email))
    if not account or not verify_password(password, account.get("password_hash", "")):
        return None
    return account


def issue_session(account_id: int, days: int = 30) -> str:
    token = session_token()
    expires = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
    save_session(account_id, token_digest(token), expires)
    return token


def resolve_session(token: str | None) -> dict | None:
    if not token:
        return None
    return account_for_session(token_digest(token))


def end_session(token: str | None) -> None:
    if token:
        delete_session(token_digest(token))
