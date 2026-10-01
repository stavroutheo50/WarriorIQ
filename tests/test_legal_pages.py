"""Legal pages describe this deployment, not a template (QA 2026-09)."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

from core import legal
from core.config import SETTINGS


def _sections(**settings):
    with patch.object(legal, "SETTINGS", dataclasses.replace(SETTINGS, **settings)):
        return dict(legal.resolve_document("subprocessors")["sections"])


def test_subprocessors_name_the_real_host_and_analysis_provider():
    sections = _sections(hosting_provider="Render", worker_wake_url="https://x--warrioriq-worker-wake.modal.run",
                         analysis_worker_mode="remote", analytics_measurement_id="G-TEST")
    headings = " ".join(sections)
    assert "Render (website hosting and storage)" in sections
    assert "Modal (fight analysis)" in sections
    assert "Google" in headings
    assert "no WarriorIQ cloud host" not in " ".join(sections.values())


def test_switched_off_providers_are_not_listed():
    sections = _sections(hosting_provider="", worker_wake_url="", analysis_worker_mode="inprocess",
                         analytics_measurement_id="", gtm_container_id="", email_provider="",
                         payments_enabled=False)
    headings = " ".join(sections)
    for absent in ("Modal", "Google", "Stripe", "OpenAI", "Render"):
        assert absent not in headings


def test_no_page_carries_notes_meant_for_the_operator():
    for slug in legal.LEGAL_DOCUMENTS:
        document = legal.resolve_document(slug)
        text = " ".join([document["intro"]] + [body for _, body in document["sections"]])
        for note in ("operator must", "operator should", "final operator", "Launch readiness does not claim"):
            assert note not in text, (slug, note)
