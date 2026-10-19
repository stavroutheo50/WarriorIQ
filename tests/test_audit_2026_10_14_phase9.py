"""Audit of 14 Oct 2026, phase 9: legal (code side; the rest is in OPS_TODO.md).

#9  the operator's legal identity is one config value read by every legal page
#10 /subprocessors names Render, Modal, Google and Stripe with purpose,
    location and data processed; the policy version is bumped
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from core import legal
from core.config import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def _settings_with(env: dict) -> dict:
    """Read Settings in a fresh interpreter with this environment."""
    code = ("import json, core.config as c; s=c.SETTINGS; print(json.dumps({k: getattr(s, k) for k in "
            "('operator_name','operator_address','governing_country','support_email','privacy_email','dmca_email')}))")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("WARRIORIQ_OPERATOR", "WARRIORIQ_LEGAL",
                                                                      "WARRIORIQ_SUPPORT", "WARRIORIQ_PRIVACY"))}
    out = subprocess.run([sys.executable, "-c", code], env={**clean, **env}, cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


# ---------------------------------------------------------------------- #9
def test_one_value_sets_the_whole_legal_identity():
    identity = {"name": "WarriorIQ Ltd", "address": "1 Example Street, Nicosia", "country": "Cyprus",
                "support_email": "support@warrioriq.eu", "privacy_email": "privacy@warrioriq.eu",
                "dmca_email": "copyright@warrioriq.eu"}
    values = _settings_with({"WARRIORIQ_LEGAL_IDENTITY": json.dumps(identity),
                             "WARRIORIQ_OPERATOR_NAME": "Old name"})
    assert values == {"operator_name": "WarriorIQ Ltd", "operator_address": "1 Example Street, Nicosia",
                      "governing_country": "Cyprus", "support_email": "support@warrioriq.eu",
                      "privacy_email": "privacy@warrioriq.eu", "dmca_email": "copyright@warrioriq.eu"}


def test_the_older_single_settings_still_work_and_bad_json_is_ignored():
    values = _settings_with({"WARRIORIQ_LEGAL_IDENTITY": "{not json", "WARRIORIQ_OPERATOR_NAME": "Solo Name"})
    assert values["operator_name"] == "Solo Name"


def test_every_legal_page_reads_the_same_fields():
    operator = legal.launch_readiness()["operator"]
    assert operator["name"] == SETTINGS.operator_name and operator["support_email"] == SETTINGS.support_email
    for name in ("legal.html", "legal_document.html", "privacy.html"):
        page = (ROOT / "app" / "templates" / name).read_text(encoding="utf-8")
        assert "@gmail.com" not in page


def test_the_example_config_uses_the_own_domain():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in ("SUPPORT", "PRIVACY", "DMCA"):
        line = next(line for line in example.splitlines() if line.startswith(f"WARRIORIQ_{key}_EMAIL="))
        assert line.endswith("@warrioriq.eu")


# --------------------------------------------------------------------- #10
def _sections(**settings):
    with patch.object(legal, "SETTINGS", dataclasses.replace(SETTINGS, **settings)), \
            patch.dict(os.environ, {"STRIPE_SECRET_KEY": "sk_test_x"}):
        return dict(legal.resolve_document("subprocessors")["sections"])


def test_the_list_names_render_modal_google_and_stripe():
    sections = _sections(hosting_provider="Render", hosting_location="Frankfurt, Germany (EU)",
                         analysis_worker_mode="remote", worker_wake_url="", analysis_provider_name="Modal",
                         google_client_id="id", google_client_secret="secret", oauth_state_secret="s" * 32,
                         payments_enabled=True)
    for heading in ("Render (website hosting and storage)", "Modal (fight analysis)", "Google (sign-in)",
                    "Stripe (payments)"):
        assert heading in sections, heading
        assert "Data processed:" in sections[heading]
        assert "Location:" in sections[heading]
    assert "Frankfurt, Germany (EU)" in sections["Render (website hosting and storage)"]
    # No location invented for Modal until one is configured.
    assert "Location: not yet published." in sections["Modal (fight analysis)"]


def test_stripe_is_listed_only_when_billing_is_live():
    sections = _sections(payments_enabled=False)
    assert "Stripe (payments)" not in sections


def test_the_policy_version_is_bumped():
    assert SETTINGS.policy_version >= "2026-10-19" or os.getenv("WARRIORIQ_POLICY_VERSION")
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "key: WARRIORIQ_HOSTING_PROVIDER\n        value: Render" in render
    assert "key: WARRIORIQ_ANALYSIS_PROVIDER_NAME\n        value: Modal" in render
