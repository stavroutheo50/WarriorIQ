"""Legal pages print real provider names, and a build fails on a placeholder.

QA, 2026-10-07: /subprocessors listed the email provider literally as "smtp" -
WARRIORIQ_EMAIL_PROVIDER switches sending on and was also printed as the
company's name - and no host or analysis provider was named.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from core import legal
from core.config import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def _sections(**settings):
    with patch.object(legal, "SETTINGS", dataclasses.replace(SETTINGS, **settings)):
        return dict(legal.resolve_document("subprocessors")["sections"])


def test_the_transport_is_never_printed_as_the_email_company():
    sections = _sections(email_provider="smtp", email_provider_name="")
    text = " ".join(list(sections) + list(sections.values()))
    assert "smtp" not in text.lower()
    assert "Email delivery" in sections


def test_the_configured_company_names_are_printed():
    sections = _sections(email_provider="smtp", email_provider_name="Postmark", hosting_provider="Render",
                         analysis_worker_mode="remote", worker_wake_url="", analysis_provider_name="Hetzner")
    assert "Postmark (email)" in sections
    assert "Render (website hosting and storage)" in sections
    assert "Hetzner (fight analysis)" in sections


def test_placeholders_are_problems():
    good = dataclasses.replace(SETTINGS, email_provider="smtp", email_provider_name="Postmark",
                               hosting_provider="Render", analysis_provider_name="",
                               support_email="support@warrioriq.eu", privacy_email="privacy@warrioriq.eu")
    assert legal.public_config_problems(good) == []
    for change, words in (({"email_provider_name": "smtp"}, "not a company name"),
                          ({"email_provider_name": ""}, "WARRIORIQ_EMAIL_PROVIDER_NAME"),
                          ({"hosting_provider": "server"}, "not a company name"),
                          ({"hosting_provider": ""}, "cannot name the host"),
                          ({"support_email": "warrioriq@gmail.com"}, "gmail.com address"),
                          ({"privacy_email": ""}, "placeholder")):
        problems = legal.public_config_problems(dataclasses.replace(good, **change))
        assert any(words in problem for problem in problems), (change, problems)


def test_the_build_check_fails_on_smtp_and_passes_when_named():
    env_bad = {"WARRIORIQ_EMAIL_PROVIDER": "smtp", "WARRIORIQ_EMAIL_PROVIDER_NAME": "smtp",
               "WARRIORIQ_HOSTING_PROVIDER": "Render", "WARRIORIQ_SUPPORT_EMAIL": "support@warrioriq.eu",
               "WARRIORIQ_PRIVACY_EMAIL": "privacy@warrioriq.eu", "PATH": "/usr/bin:/bin"}
    run = lambda env: subprocess.run([sys.executable, str(ROOT / "tools" / "check_public_config.py")],
                                     cwd=ROOT, env=env, capture_output=True, text=True)
    bad = run(env_bad)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "not a company name" in bad.stdout
    good = run({**env_bad, "WARRIORIQ_EMAIL_PROVIDER_NAME": "Postmark"})
    assert good.returncode == 0, good.stdout + good.stderr


def test_one_flag_carries_the_draft_banner_on_every_legal_page():
    """WARRIORIQ_LEGAL_DRAFT (default on) is the single flag; it is not removed here."""
    assert SETTINGS.legal_is_draft is True
    templates = ROOT / "app" / "templates"
    for name in ("legal.html", "privacy.html", "legal_document.html"):
        assert "request.state.legal_is_draft" in (templates / name).read_text(encoding="utf-8"), name
