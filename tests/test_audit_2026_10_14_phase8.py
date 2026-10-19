"""Audit of 14 Oct 2026, phase 8: data cleanup.

#32 a one-off script deletes the QA test fighters and their fights from one
    account - dry run by default, --confirm to act, nothing else touched -
    and QA gets an account of its own
#52 the "plan too small for your roster" notice is shown once, not per card
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from browser_client import BrowserClient

import app.main as web
from core import db as database
from tools import delete_qa_test_data as cleanup
from tools import seed_qa_account

ROOT = Path(__file__).resolve().parents[1]


def _account():
    email = f"p8-{uuid.uuid4().hex[:8]}@example.com"
    from core.auth import hash_password

    account = database.create_account(email, hash_password("Strong-Local-Password"))
    return email, int(account["profile_id"])


def _fight(profile_id, fighter_id=None):
    job_id = "p8" + uuid.uuid4().hex[:10]
    directory = web.OUTPUTS / job_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps({}), encoding="utf-8")
    video = web.UPLOADS / f"{job_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"x")
    database.save_fight(job_id, profile_id, "qa.mp4", str(video), str(directory / "report.json"),
                        "sparring", "K1", "A", {}, "2999-01-01T00:00:00+00:00", fighter_id=fighter_id)
    return job_id, directory, video


@pytest.fixture
def workspace():
    email, profile_id = _account()
    qa = [database.create_fighter(profile_id, name) for name in cleanup.QA_FIGHTER_NAMES]
    real = database.create_fighter(profile_id, "Qasim Real Athlete")
    near = database.create_fighter(profile_id, "QA Test Boxer 2")
    qa_fight = _fight(profile_id, qa[0]["id"])
    xss_fight = _fight(profile_id, qa[4]["id"])
    real_fight = _fight(profile_id, real["id"])
    unfiled = _fight(profile_id)
    return {"email": email, "profile_id": profile_id, "qa": qa, "real": real, "near": near,
            "qa_fights": [qa_fight, xss_fight], "kept_fights": [real_fight, unfiled]}


def test_the_dry_run_lists_exactly_the_qa_rows_and_changes_nothing(workspace, capsys):
    assert cleanup.main(["--email", workspace["email"]]) == 0
    out = capsys.readouterr().out
    assert "Dry run: nothing was changed" in out
    for fighter in workspace["qa"]:
        assert repr(fighter["name"]) in out
    assert "Qasim" not in out and "QA Test Boxer 2" not in out
    assert len(database.list_fighters(workspace["profile_id"], include_archived=True)) == 7
    assert all(database.get_fight(job) for job, _, _ in workspace["qa_fights"])


def test_confirm_deletes_only_the_qa_fighters_and_their_fights(workspace):
    other_email, other_profile = _account()
    other_qa = database.create_fighter(other_profile, "QA Test Boxer")
    assert cleanup.main(["--email", workspace["email"], "--confirm"]) == 0

    left = {f["name"] for f in database.list_fighters(workspace["profile_id"], include_archived=True)}
    assert left == {"Qasim Real Athlete", "QA Test Boxer 2"}
    for job_id, directory, video in workspace["qa_fights"]:
        assert database.get_fight(job_id) is None
        assert not directory.exists() and not video.exists()
    for job_id, directory, video in workspace["kept_fights"]:
        assert database.get_fight(job_id) is not None
        assert directory.exists() and video.exists()
    # Another account's fighter of the same name is untouched.
    assert database.get_fighter(other_profile, other_qa["id"]) is not None
    events = [e for e in database.list_security_events(500) if e["event_type"] == "qa_test_data_deleted"]
    assert {e["resource_id"] for e in events} >= {job for job, _, _ in workspace["qa_fights"]}


def test_an_unknown_account_changes_nothing():
    with pytest.raises(SystemExit):
        cleanup.main(["--email", "nobody-p8@example.com", "--confirm"])


def test_qa_gets_an_account_of_its_own(monkeypatch):
    email = f"qa-{uuid.uuid4().hex[:8]}@example.com"
    first = seed_qa_account.seed(email, "Strong-Local-Password")
    again = seed_qa_account.seed(email, None)                # re-running reuses, never duplicates
    assert {f["name"] for f in first["fighters"]} == set(cleanup.QA_FIGHTER_NAMES)
    assert [f["id"] for f in again["fighters"]] == [f["id"] for f in first["fighters"]]
    # Never pointed at a real workspace.
    real_email, profile_id = _account()
    database.create_fighter(profile_id, "Real Athlete")
    with pytest.raises(SystemExit):
        seed_qa_account.seed(real_email, None)


def test_no_password_is_hard_coded():
    source = (ROOT / "tools" / "seed_qa_account.py").read_text(encoding="utf-8")
    assert 'os.getenv("WARRIORIQ_QA_PASSWORD")' in source


# --------------------------------------------------------------------- #52
def test_the_roster_notice_is_shown_once():
    client = BrowserClient(web.app)
    email = f"p8c-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/signup", data={"email": email, "password": "Strong-Local-Password", "accept_terms": "true",
                                 "age_confirmed": "true"}, follow_redirects=False)
    profile_id = int(database.get_account_by_email(email)["profile_id"])
    for n in range(8):
        database.create_fighter(profile_id, f"Athlete {n}")
    page = client.get("/pricing?audience=coach").text
    assert page.count('id="rosterTooBig"') == 1
    assert page.count('class="plan-blocked') == 1
    assert "Too small" not in page
