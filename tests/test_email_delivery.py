"""Account emails say why they did not arrive, and an admin can test them."""

import smtplib
from types import SimpleNamespace
from unittest import mock

import pytest

from core import notifications

GMAIL = {"WARRIORIQ_SMTP_HOST": "smtp.gmail.com", "WARRIORIQ_SMTP_USERNAME": "box@gmail.com",
         "WARRIORIQ_SMTP_PASSWORD": "app-password", "WARRIORIQ_EMAIL_FROM": "WarriorIQ"}


def _env(**overrides):
    env = {**GMAIL, **overrides}
    return mock.patch.dict(notifications.os.environ, env, clear=False)


def _smtp(provider="smtp"):
    return mock.patch.object(notifications, "SETTINGS", SimpleNamespace(email_provider=provider))


def test_email_switched_off_is_named():
    with _env(), _smtp(""):
        assert "WARRIORIQ_EMAIL_PROVIDER" in notifications.email_settings_problem()
        with pytest.raises(notifications.EmailNotSent, match="WARRIORIQ_EMAIL_PROVIDER"):
            notifications.deliver_email("a@b.com", "s", "b")


def test_port_465_uses_ssl_from_the_start_and_587_uses_starttls():
    with _env(WARRIORIQ_SMTP_PORT="465"), _smtp(), mock.patch.object(notifications.smtplib, "SMTP_SSL") as ssl_smtp, \
            mock.patch.object(notifications.smtplib, "SMTP") as plain:
        notifications.deliver_email("a@b.com", "s", "b")
    ssl_smtp.assert_called_once()
    plain.assert_not_called()
    ssl_smtp.return_value.starttls.assert_not_called()
    with _env(WARRIORIQ_SMTP_PORT="587"), _smtp(), mock.patch.object(notifications.smtplib, "SMTP") as plain:
        notifications.deliver_email("a@b.com", "s", "b")
    plain.return_value.starttls.assert_called_once()


def test_a_rejected_gmail_password_explains_app_passwords_and_never_shows_it():
    client = mock.MagicMock()
    client.login.side_effect = smtplib.SMTPAuthenticationError(535, b"BadCredentials")
    with _env(), _smtp(), mock.patch.object(notifications.smtplib, "SMTP", return_value=client):
        with pytest.raises(notifications.EmailNotSent) as raised:
            notifications.deliver_email("a@b.com", "s", "b")
    assert "App Password" in str(raised.value)
    assert "app-password" not in str(raised.value)


def test_a_blocked_port_is_explained():
    with _env(), _smtp(), mock.patch.object(notifications.smtplib, "SMTP", side_effect=TimeoutError()):
        with pytest.raises(notifications.EmailNotSent, match="block outgoing email ports"):
            notifications.deliver_email("a@b.com", "s", "b")


def test_send_transactional_email_logs_the_reason_and_returns_false(caplog):
    with _env(), _smtp(), mock.patch.object(notifications.smtplib, "SMTP", side_effect=ConnectionRefusedError()):
        assert notifications.send_transactional_email("a@b.com", "Subject", "b") is False
    assert "could not connect to smtp.gmail.com" in caplog.text


def test_admin_test_button_shows_the_result(monkeypatch):
    import sys
    import uuid
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from browser_client import BrowserClient

    from app import main
    from core.auth import register
    from core.config import SETTINGS

    email = f"mail-admin-{uuid.uuid4().hex[:8]}@example.com"
    register(email, "Strong-Local-Password")
    previous = SETTINGS.admin_emails
    object.__setattr__(SETTINGS, "admin_emails", (email,))
    try:
        with BrowserClient(main.app) as client:
            client.post("/login", data={"email": email, "password": "Strong-Local-Password"})

            def fail(*args):
                raise notifications.EmailNotSent("the mail server rejected the username or password (code 535).")

            monkeypatch.setattr(main, "deliver_email", fail)
            result = client.post("/admin/email/test")
            assert "Not sent: the mail server rejected" in result.text
            monkeypatch.setattr(main, "deliver_email", lambda *args: None)
            result = client.post("/admin/email/test")
            assert f"Sent to {email}" in result.text
            # Not an admin: the button does not exist.
            object.__setattr__(SETTINGS, "admin_emails", ())
            assert client.post("/admin/email/test").status_code == 404
    finally:
        object.__setattr__(SETTINGS, "admin_emails", previous)
