from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

from core.config import SETTINGS


LOGGER = logging.getLogger("warrioriq.notifications")


def _from_header(configured: str, username: str) -> str:
    """A From header that is actually an address.

    A display name on its own is not a valid From header and the mail server
    rejects it, and a name is the natural thing to type into a box labelled
    "email from". Rather than send it as-is and fail, it is paired with the
    authenticated mailbox. A value that already contains an address is passed
    through untouched, so a correctly configured server behaves exactly as
    before.
    """
    if "@" in configured:
        return configured
    if not username:
        return ""
    return f"{configured} <{username}>" if configured else username


def send_transactional_email(recipient: str, subject: str, body: str) -> bool:
    """Send through configured SMTP without persisting secret one-time links."""
    if SETTINGS.email_provider.lower() != "smtp":
        return False
    host = os.getenv("WARRIORIQ_SMTP_HOST", "").strip()
    username = os.getenv("WARRIORIQ_SMTP_USERNAME", "").strip()
    password = os.getenv("WARRIORIQ_SMTP_PASSWORD", "")
    sender = _from_header(os.getenv("WARRIORIQ_EMAIL_FROM", "").strip(), username)
    if not host or not sender:
        return False
    # A username with no password cannot authenticate, and every attempt raises
    # deep inside smtplib where the caller can only record "something failed".
    # Refusing here says which setting is missing, in the log, once.
    if username and not password:
        LOGGER.error(
            "email_not_sent reason=missing_smtp_password host=%s username=%s "
            "detail=WARRIORIQ_SMTP_PASSWORD is empty, so password resets cannot be delivered",
            host, username,
        )
        return False
    port = int(os.getenv("WARRIORIQ_SMTP_PORT", "587"))
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=15) as client:
        client.starttls(context=context)
        if username:
            client.login(username, password)
        client.send_message(message)
    return True
