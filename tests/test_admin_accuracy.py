"""Accuracy from people's answers on the admin page (owner, 2026-11-16).

Counts only: each figure is what the answers said, never an estimate.
"""

from __future__ import annotations

import uuid

import app.main as web
from app import accuracy
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session
from core.config import SETTINGS


def _check(family, verdict):
    return {"predicted": {"fighter": "A", "family": family, "technique": family},
            "corrected": {"fighter": "A", "family": family, "verdict": verdict, "source": "strike_check"}}


def test_strike_answers_are_counted_per_family():
    rows = [_check("kick", "right"), _check("kick", "kick"), _check("kick", "punch"),
            _check("kick", "not_a_strike"), _check("punch", "wrong_fighter"),
            {"predicted": {"event": "went_down"}, "corrected": {"went_down": "nobody", "source": "down_check"}}]
    counts = accuracy.strike_answers(rows)
    assert counts["kick"] == {"answered": 4, "right": 2, "not_a_strike": 1, "wrong_fighter": 0, "other_type": 1}
    assert counts["punch"]["answered"] == 1 and counts["punch"]["wrong_fighter"] == 1
    assert counts["knee"]["answered"] == 0


def test_down_answers_count_false_alarms():
    rows = [{"corrected": {"went_down": verdict, "source": "down_check"}} for verdict in ("A", "nobody", "nobody")]
    rows.append(_check("kick", "right"))
    assert accuracy.down_answers(rows) == {"answered": 3, "A": 1, "B": 0, "nobody": 2}


def test_corrections_use_the_validation_measure_and_skip_one_tap_answers():
    rows = [{"job_id": "j", "event_time": 1.0, "ruleset": "K1",
             "predicted": {"fighter": "A", "technique": "jab"}, "corrected": {"fighter": "A", "technique": "jab"}},
            {"job_id": "j", "event_time": 2.0, "ruleset": "K1",
             "predicted": {"fighter": "A", "technique": "jab"}, "corrected": {"fighter": "B", "technique": "cross"}},
            _check("kick", "right")]
    counts = accuracy.corrections(rows)
    assert counts["total"] == 2
    assert (counts["technique"]["correct"], counts["technique"]["total"]) == (1, 2)
    assert (counts["fighter_identity"]["correct"], counts["fighter_identity"]["total"]) == (1, 2)


def test_crossing_answers_count_disagreements_once_per_moment():
    jobs = [
        # Re-run already: anchors stored, one disagreed.
        ("j1", {"identity_anchors": [{"frame": 10, "swapped": True}, {"frame": 20, "swapped": False}]}),
        # Answered, not re-run: anchors worked out from the answers.
        ("j2", {"identity_answers": {"30": "A"}}),
        # A new answer about a stored moment replaces it.
        ("j3", {"identity_anchors": [{"frame": 40, "swapped": True}], "identity_answers": {"40": "A"}}),
        ("j4", {}),
    ]
    pending = {"j2": [{"frame": 30, "swapped": False}], "j3": [{"frame": 40, "swapped": False}]}
    counts = accuracy.identity_answers(jobs, lambda job_id, job: pending.get(job_id, []))
    assert counts == {"fights": 3, "answers": 4, "disagreed": 1, "fights_with_a_swap": 1, "fights_rerun": 2}


def test_the_admin_page_shows_counts_and_no_rate_under_20_answers(monkeypatch):
    monkeypatch.setattr(accuracy, "_cache", {})
    monkeypatch.setattr(accuracy, "list_annotations",
                        lambda: [_check("kick", "right")] * 3 + [_check("kick", "not_a_strike")])
    email = f"accuracy-admin-{uuid.uuid4().hex[:10]}@example.test"
    created = db.create_account(email, hash_password("Strong-Local-Password"))
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    previous = SETTINGS.admin_emails
    object.__setattr__(SETTINGS, "admin_emails", (email.lower(),))
    try:
        page = client.get("/admin").text
    finally:
        object.__setattr__(SETTINGS, "admin_emails", previous)
    assert 'id="adminAccuracy"' in page and 'href="#adminAccuracy"' in page
    assert "right 3 of 4;" in page and "(75%)" not in page
