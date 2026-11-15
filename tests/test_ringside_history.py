"""Ringside Scorer: past scorecards and CSV export, kept on the phone (owner's brief, 2026-11-14)."""

from __future__ import annotations

from pathlib import Path

import app.main as web
from browser_client import BrowserClient

SCRIPT = (Path(__file__).resolve().parents[1] / "app" / "static" / "ringside.js").read_text(encoding="utf-8")


def test_the_page_offers_export_and_past_scorecards():
    page = BrowserClient(web.app).get("/scorer").text
    assert 'id="scorerCsv"' in page and 'id="scorerHistory"' in page
    assert "not a judges' decision" in page


def test_history_stays_in_the_browser_and_csv_cells_cannot_run_as_formulas():
    # Saved to localStorage only; the only request the scorer makes is the scoring call.
    assert "wiq-ringside-history-v1" in SCRIPT and SCRIPT.count("fetch(") == 1
    assert "/^[=+\\-@\\t\\r]/" in SCRIPT
