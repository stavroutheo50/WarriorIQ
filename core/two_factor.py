"""Optional two-step sign-in with an authenticator app (owner, 2026-11-12).

Standard time-based one-time codes (RFC 6238: HMAC-SHA1, 30 seconds, 6 digits)
that work with Google Authenticator, Microsoft Authenticator, 1Password,
Authy and the like, written with the standard library only.

Three rules this module exists to keep:

  * a code is accepted once: `verify` takes the last time step an account
    already used and refuses that step and every earlier one, so a code seen
    over someone's shoulder or in a log cannot be replayed;
  * recovery codes are the way back in when the phone is lost. They are
    random, shown once, stored only as hashes, and each works once;
  * the secret has to be readable by the server (a hash cannot make a code),
    so it is encrypted at rest when WARRIORIQ_TWO_FACTOR_KEY is set and the
    `cryptography` package is installed, and stored as plain text otherwise.
    The prefix on the stored value says which. A value that cannot be opened
    (key lost, package missing) fails closed: authenticator codes stop
    working and the recovery codes still do.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import struct
import time
from urllib.parse import quote

STEP_SECONDS = 30
DIGITS = 6
# One step either side of now: a phone clock a few seconds off still works.
WINDOW_STEPS = 1
RECOVERY_CODE_COUNT = 10
# No 0/O/1/I/L: read aloud or off paper, they would be mistaken for each other.
_RECOVERY_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def new_secret() -> str:
    """A fresh 160-bit secret as base32, the form every authenticator app takes."""
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _key(secret: str) -> bytes:
    cleaned = "".join(secret.split()).upper()
    return base64.b32decode(cleaned + "=" * (-len(cleaned) % 8))


def code_at(secret: str, step: int) -> str:
    digest = hmac.new(_key(secret), struct.pack(">Q", int(step)), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(value % (10 ** DIGITS)).zfill(DIGITS)


def current_step(now: float | None = None) -> int:
    return int((time.time() if now is None else now) // STEP_SECONDS)


def verify(secret: str, code: str, last_used_step: int | None = None, now: float | None = None) -> int | None:
    """The time step the code belongs to, or None if it is wrong or already used.

    Every candidate step is compared (no early exit) and in constant time, so
    timing says nothing about how close a wrong guess was.
    """
    submitted = "".join(ch for ch in (code or "") if ch.isdigit())
    if len(submitted) != DIGITS:
        return None
    now_step = current_step(now)
    accepted = None
    for step in range(now_step - WINDOW_STEPS, now_step + WINDOW_STEPS + 1):
        try:
            expected = code_at(secret, step)
        except (ValueError, TypeError):
            return None
        if hmac.compare_digest(expected, submitted) and (last_used_step is None or step > int(last_used_step)):
            accepted = step
    return accepted


def provisioning_uri(secret: str, account_label: str, issuer: str = "WarriorIQ") -> str:
    """The otpauth:// address an authenticator app reads (typed in, or as a QR)."""
    label = quote(f"{issuer}:{account_label}", safe="")
    return f"otpauth://totp/{label}?secret={secret}&issuer={quote(issuer, safe='')}&digits={DIGITS}&period={STEP_SECONDS}"


def grouped(secret: str) -> str:
    """The secret in fours, for typing into an app by hand."""
    return " ".join(secret[i:i + 4] for i in range(0, len(secret), 4))


# --- Recovery codes ---------------------------------------------------------

def new_recovery_codes(count: int = RECOVERY_CODE_COUNT) -> list[str]:
    return ["".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(10)) for _ in range(count)]


def display_code(code: str) -> str:
    return f"{code[:5]}-{code[5:]}"


def normalize_recovery(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def recovery_hash(code: str) -> str:
    return hashlib.sha256(("wiq-recovery:" + normalize_recovery(code)).encode("utf-8")).hexdigest()


def looks_like_recovery(value: str) -> bool:
    """Ten letters and digits is a recovery code; an authenticator code is six digits."""
    return len(normalize_recovery(value)) == 10


# --- The secret at rest ----------------------------------------------------

def _fernet():
    key = os.getenv("WARRIORIQ_TWO_FACTOR_KEY", "").strip()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet

        # Any passphrase works: it is stretched into a Fernet key.
        derived = base64.urlsafe_b64encode(hashlib.pbkdf2_hmac("sha256", key.encode("utf-8"), b"warrioriq-2fa", 200_000))
        return Fernet(derived)
    except Exception:  # noqa: BLE001 - a missing package or a bad key both mean "do not encrypt"
        # Said out loud: a key that is set but ignored would leave secrets in
        # plain text while whoever set it believes they are encrypted.
        logging.getLogger("warrioriq").warning(
            "WARRIORIQ_TWO_FACTOR_KEY is set but the cryptography package is not installed: "
            "authenticator secrets are being stored WITHOUT encryption.")
        return None


def seal(secret: str) -> str:
    box = _fernet()
    if box is None:
        return "plain:" + secret
    return "fernet:" + box.encrypt(secret.encode("ascii")).decode("ascii")


def unseal(stored: str | None) -> str | None:
    if not stored:
        return None
    if stored.startswith("plain:"):
        return stored[6:]
    if stored.startswith("fernet:"):
        box = _fernet()
        if box is None:
            return None
        try:
            return box.decrypt(stored[7:].encode("ascii")).decode("ascii")
        except Exception:  # noqa: BLE001 - wrong key or damaged value: fail closed
            return None
    return None
