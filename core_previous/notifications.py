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


class EmailNotSent(Exception):
    """Why a message was not delivered, in words an administrator can act on."""


def email_settings_problem() -> str | None:
    """The first missing email setting, named; None when sending can be tried."""
    if SETTINGS.email_provider.lower() != "smtp":
        return "WARRIORIQ_EMAIL_PROVIDER is not set to smtp, so no email is sent at all"
    host = os.getenv("WARRIORIQ_SMTP_HOST", "").strip()
    username = os.getenv("WARRIORIQ_SMTP_USERNAME", "").strip()
    if not host:
        return "WARRIORIQ_SMTP_HOST is empty"
    if not _from_header(os.getenv("WARRIORIQ_EMAIL_FROM", "").strip(), username):
        return "neither WARRIORIQ_EMAIL_FROM nor WARRIORIQ_SMTP_USERNAME gives a sender address"
    # A username with no password cannot authenticate, and every attempt raises
    # deep inside smtplib where the caller can only record "something failed".
    if username and not os.getenv("WARRIORIQ_SMTP_PASSWORD", ""):
        return "WARRIORIQ_SMTP_PASSWORD is empty, so the mail server login cannot succeed"
    return None


def _explain(exc: Exception, host: str, port: int) -> str:
    """A failure reason with the likely fix. Never includes the password."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        hint = (" For Gmail this needs a 16-character App Password (Google account -> Security -> "
                "2-Step Verification -> App passwords), not the normal password; an app password "
                "stops working when the Google password changes.") if "gmail" in host.lower() else ""
        return f"the mail server rejected the username or password (code {exc.smtp_code}).{hint}"
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return "the mail server refused the recipient address"
    if isinstance(exc, smtplib.SMTPSenderRefused):
        return "the mail server refused the sender address (WARRIORIQ_EMAIL_FROM)"
    if isinstance(exc, (TimeoutError, ConnectionRefusedError, OSError)) and not isinstance(exc, smtplib.SMTPException):
        return (f"could not connect to {host} on port {port} ({type(exc).__name__}). The host this "
                "website runs on may block outgoing email ports, or the host/port is wrong.")
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def deliver_email(recipient: str, subject: str, body: str) -> None:
    """Send one message, or raise EmailNotSent saying exactly why not."""
    problem = email_settings_problem()
    if problem:
        raise EmailNotSent(problem)
    host = os.getenv("WARRIORIQ_SMTP_HOST", "").strip()
    username = os.getenv("WARRIORIQ_SMTP_USERNAME", "").strip()
    password = os.getenv("WARRIORIQ_SMTP_PASSWORD", "")
    sender = _from_header(os.getenv("WARRIORIQ_EMAIL_FROM", "").strip(), username)
    port = int(os.getenv("WARRIORIQ_SMTP_PORT", "587"))
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    context = ssl.create_default_context()
    try:
        # Port 465 speaks TLS from the first byte; 587 (and 25) start plain and
        # upgrade with STARTTLS. Using STARTTLS on 465 hangs and then fails,
        # which is what every provider that hands out 465 would have hit.
        if port == 465:
            client = smtplib.SMTP_SSL(host, port, timeout=15, context=context)
        else:
            client = smtplib.SMTP(host, port, timeout=15)
        with client:
            if port != 465:
                client.starttls(context=context)
            if username:
                client.login(username, password)
            client.send_message(message)
    except Exception as exc:                                        # noqa: BLE001
        raise EmailNotSent(_explain(exc, host, port)) from exc


def send_transactional_email(recipient: str, subject: str, body: str) -> bool:
    """Send through configured SMTP without persisting secret one-time links.

    False when the message was not delivered, and the reason is logged: a
    password reset that silently never arrives is what this used to do.
    """
    try:
        deliver_email(recipient, subject, body)
    except EmailNotSent as reason:
        LOGGER.error("email_not_sent subject=%r reason=%s", subject[:80], reason)
        return False
    return True
