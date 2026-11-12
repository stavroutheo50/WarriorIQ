"""The public contact addresses are on the domain (owner, 2026-11-12)."""

from __future__ import annotations

from pathlib import Path

from core import config

ROOT = Path(__file__).resolve().parents[1]


def test_the_retired_gmail_address_is_replaced_by_the_domain_one(monkeypatch):
    monkeypatch.setattr(config, "LEGAL_IDENTITY", {})
    monkeypatch.setenv("WARRIORIQ_SUPPORT_EMAIL", "WarriorIQAI@gmail.com")
    assert config.contact_address("support_email", "WARRIORIQ_SUPPORT_EMAIL", "support@warrioriq.eu") == "support@warrioriq.eu"
    # ...wherever it is configured, including the legal identity JSON.
    monkeypatch.setattr(config, "LEGAL_IDENTITY", {"privacy_email": "warrioriqai@gmail.com"})
    assert config.contact_address("privacy_email", "WARRIORIQ_PRIVACY_EMAIL", "privacy@warrioriq.eu") == "privacy@warrioriq.eu"


def test_any_other_configured_address_is_shown_as_configured(monkeypatch):
    monkeypatch.setattr(config, "LEGAL_IDENTITY", {})
    monkeypatch.setenv("WARRIORIQ_DMCA_EMAIL", "legal@example.org")
    assert config.contact_address("dmca_email", "WARRIORIQ_DMCA_EMAIL", "copyright@warrioriq.eu") == "legal@example.org"
    monkeypatch.delenv("WARRIORIQ_DMCA_EMAIL")
    assert config.contact_address("dmca_email", "WARRIORIQ_DMCA_EMAIL", "copyright@warrioriq.eu") == ""


def test_no_page_or_example_setting_names_the_gmail_address():
    for path in list((ROOT / "app" / "templates").glob("*.html")) + [ROOT / ".env.example"]:
        assert "warrioriqai@gmail.com" not in path.read_text(encoding="utf-8").lower(), path.name
