"""The report's quick check: owners answer a few counted strikes, one tap each (owner, 2026-11-24).

"I don't want to label ... can we skip labelling?" The answers come instead
from the people whose fights they are, a tap at a time. Who may answer, what
is asked, and that it never changes the report.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_client import BrowserClient  # noqa: E402

from app import main  # noqa: E402
from core import db  # noqa: E402
from core.auth import issue_session  # noqa: E402
from test_social_profiles import _shareable_fight  # noqa: E402

FEED = [{"fighter": "A", "family": "punch", "time_seconds": 10.0 + i * 3} for i in range(20)] + \
       [{"fighter": "B", "family": "kick", "time_seconds": 11.5}]


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr(main, "_enforce_rate_limit", lambda *args, **kwargs: None)


def _person(minor=False):
    account = db.create_account(f"qc-{uuid.uuid4().hex[:10]}@example.test", "x")
    if minor:
        with db.connection() as con:
            con.execute("UPDATE accounts SET guardian_approval_status='guardian_approved' WHERE id=?", (account["id"],))
    client = BrowserClient(main.app)
    client.cookies.set(main.SESSION_COOKIE, issue_session(account["id"]))
    return client, account


def _fight_with_feed(profile_id: int) -> str:
    job_id = _shareable_fight(profile_id)
    path = main.OUTPUTS / job_id / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["event_feed"] = FEED
    path.write_text(json.dumps(report), encoding="utf-8")
    return job_id


def _answer(client, job_id, seconds=10.0, verdict="right"):
    return client.post(f"/api/strike-check/{job_id}", json={"seconds": seconds, "fighter": "A",
                                                            "family": "punch", "verdict": verdict})


def test_the_owner_may_answer_and_nobody_else_may():
    owner, account = _person()
    job_id = _fight_with_feed(int(account["profile_id"]))
    assert _answer(owner, job_id).status_code == 200
    stranger, _ = _person()
    assert _answer(stranger, job_id, 13.0).status_code == 403
    assert _answer(BrowserClient(main.app), job_id, 13.0).status_code in {401, 403}


def test_an_under_18_account_is_never_asked():
    young, account = _person(minor=True)
    job_id = _fight_with_feed(int(account["profile_id"]))
    assert _answer(young, job_id).status_code == 403


def _quick(monkeypatch, job_id, allowed=True, **report_extra):
    monkeypatch.setattr(main, "_may_check_strikes", lambda request, job: allowed)
    report = {"event_feed": FEED, **report_extra}
    return main._quick_check(None, job_id, report, ("punch", "kick", "knee"), True, {"A": "Theo", "B": "Opponent"})


def test_the_moments_are_spread_over_the_fight_and_answered_ones_are_not_asked_again(monkeypatch):
    owner, account = _person()
    job_id = _fight_with_feed(int(account["profile_id"]))
    first = _quick(monkeypatch, job_id)
    assert len(first["items"]) == main.QUICK_CHECK_POOL and first["answered"] == 0
    times = [item["seconds"] for item in first["items"]]
    assert times == sorted(times) and times[0] == 10.0 and times[-1] > 50          # across the fight, not the start
    assert first["names"] == {"A": "Theo", "B": "Opponent"}
    monkeypatch.undo()
    assert _answer(owner, job_id, 10.0, "not_a_strike").status_code == 200
    second = _quick(monkeypatch, job_id)
    assert 10.0 not in [item["seconds"] for item in second["items"]] and second["answered"] == 1


def test_no_card_without_a_question_worth_asking(monkeypatch):
    _, account = _person()
    job_id = _fight_with_feed(int(account["profile_id"]))
    assert _quick(monkeypatch, job_id, allowed=False) is None
    assert _quick(monkeypatch, job_id, video_removed=True) is None
    monkeypatch.setattr(main, "_may_check_strikes", lambda request, job: True)
    assert main._quick_check(None, job_id, {"event_feed": FEED}, ("punch",), False, {}) is None   # who-is-who failed
    assert main._quick_check(None, job_id, {"event_feed": []}, ("punch",), True, {}) is None       # nothing counted


def test_an_answer_never_changes_the_report():
    owner, account = _person()
    job_id = _fight_with_feed(int(account["profile_id"]))
    before = (main.OUTPUTS / job_id / "report.json").read_bytes()
    assert _answer(owner, job_id, 13.0, "not_a_strike").status_code == 200
    assert (main.OUTPUTS / job_id / "report.json").read_bytes() == before


def test_the_card_renders_with_its_answers_and_says_it_is_optional():
    from test_web import NoPunchClaimLeaksTests

    data = {"items": [{"seconds": 10.0, "fighter": "A", "family": "punch", "clock": "0:10.0"}],
            "answered": 0, "names": {"A": "Theo", "B": "Opponent"}}
    page = NoPunchClaimLeaksTests._render(NoPunchClaimLeaksTests._report(), quick_check=data)
    assert 'id="quickCheck"' in page and "optional" in page and "They don't change this report." in page
    for verdict in ("right", "not_a_strike", "wrong_fighter", "skip"):
        assert f'data-verdict="{verdict}"' in page
    assert 'id="quickCheck"' not in NoPunchClaimLeaksTests._render(NoPunchClaimLeaksTests._report())


def test_the_privacy_policy_says_what_the_answers_are_for():
    page = BrowserClient(main.app).get("/privacy").text
    assert "Optional quick checks" in page and "not used as training data" in page


def test_the_stat_sheet_points_to_the_quick_check_only_when_it_is_offered():
    sheet = (Path(main.__file__).resolve().parent / "templates" / "_stat_sheet.html").read_text(encoding="utf-8")
    link = sheet.split('href="#quickCheck"')[0].rsplit("{%", 1)[-1]
    assert "quick_check" in link


def test_the_admin_counts_fights_shared_for_training(monkeypatch):
    from app import accuracy
    from core import training_consent

    monkeypatch.setattr(training_consent, "kept", lambda: [{"job_id": "a"}, {"job_id": "b"}])
    jobs = [("1", {"training_consent": True}), ("2", {}), ("3", {"training_consent": True})]
    assert accuracy.shared_fights(jobs) == {"uploads": 3, "ticked": 2, "kept": 2}
