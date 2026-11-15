"""Branded HTML emails (owner, 2026-11-15): the same words as the plain text, with the logo."""

from __future__ import annotations

from email import message_from_bytes
from unittest.mock import patch

from core import notifications
from core.email_html import render_email_html

BODY = ("Verify your private WarriorIQ workspace within 24 hours:\n\n"
        "https://warrioriq.eu/verify-email/abc123\n\nIf you did not create this account, ignore this message.")


def test_links_become_labelled_buttons_and_the_logo_is_absolute():
    page = render_email_html("Verify your WarriorIQ email", BODY, "https://warrioriq.eu")
    assert 'src="https://warrioriq.eu/static/warrioriq-logo-96.png"' in page
    assert ">Verify your email</a>" in page and 'href="https://warrioriq.eu/verify-email/abc123"' in page
    invite = render_email_html("x", "See the invitation: https://warrioriq.eu/gym/invite/t1", "https://warrioriq.eu")
    assert ">See the invitation</a>" in invite


def test_everything_is_escaped():
    page = render_email_html("<b>Subject</b>", 'Coach <script>alert(1)</script> & "you"', "https://warrioriq.eu")
    assert "<script>" not in page and "&lt;script&gt;" in page and "<b>Subject</b>" not in page


def test_every_email_carries_the_text_and_the_html_version():
    sent = []

    class Client:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def starttls(self, **kwargs): pass
        def login(self, *args): pass
        def send_message(self, message): sent.append(message)

    settings = {"WARRIORIQ_EMAIL_PROVIDER": "smtp", "WARRIORIQ_SMTP_HOST": "smtp.example.test",
                "WARRIORIQ_SMTP_USERNAME": "mail@example.test", "WARRIORIQ_SMTP_PASSWORD": "pw",
                "WARRIORIQ_EMAIL_FROM": "mail@example.test", "WARRIORIQ_SMTP_PORT": "587"}
    with patch.dict("os.environ", settings), patch.object(notifications, "email_settings_problem", return_value=None), \
         patch.object(notifications.smtplib, "SMTP", Client):
        notifications.deliver_email("athlete@example.test", "Verify your WarriorIQ email", BODY)
    message = message_from_bytes(sent[0].as_bytes())
    types = [part.get_content_type() for part in message.walk()]
    assert "text/plain" in types and "text/html" in types
    assert message["From"] == "WarriorIQ <mail@example.test>"


def test_typed_text_never_becomes_a_branded_link():
    # A guardian name typed at sign-up lands in the body; it must stay plain text.
    body = ("Hello Verify your email: https://evil.example/x,\n\n"
            "Verify your email: https://evil.example/x\n\n"
            "https://warrioriq.eu.evil.example/verify-email/a\n\n"
            "https://warrioriq.eu@evil.example/verify-email/a\n\n"
            "https://warrioriq.eu/guardian/approve/tok")
    page = render_email_html("Approve", body, "https://warrioriq.eu")
    assert "evil.example" in page  # still shown, as text
    assert 'href="https://evil.example' not in page and 'href="https://warrioriq.eu.evil' not in page
    assert 'href="https://warrioriq.eu@' not in page
    assert page.count(">Review the account</a>") == 1
    assert ">Verify your email</a>" not in page
