"""Quick-check answers as strike-training data, from policy version 2026-11-25 (app/quick_check_training.py)."""

from __future__ import annotations

import json
import uuid

import app.main as web
from app import quick_check_training as qct
from browser_client import BrowserClient
from core import db
from core.auth import issue_session

SINCE = qct.TRAINING_SINCE


def _owner(version=SINCE, accepted_at=f"{SINCE}T09:00:00+00:00"):
    account = db.create_account(f"qct-{uuid.uuid4().hex[:10]}@example.test", "x")
    with db.connection() as con:
        con.execute("UPDATE accounts SET terms_version=?, policies_accepted_at=? WHERE id=?",
                    (version, accepted_at, int(account["id"])))
    return db.get_account(int(account["id"]))


def _answer(at=f"{SINCE}T10:00:00+00:00", source="strike_check"):
    return {"job_id": "j", "event_time": 12.0, "created_at": at, "predicted": {"family": "punch"},
            "corrected": {"source": source, "verdict": "not_a_strike", "fighter": "A"}}


def test_an_answer_under_the_new_policy_is_usable():
    owner = _owner()
    assert qct.usable(_answer(), {"profile_id": owner["profile_id"]}, owner)


def test_nothing_given_under_the_old_promise_is_used():
    owner = _owner()
    fight = {"profile_id": owner["profile_id"]}
    assert not qct.usable(_answer(at="2026-11-24T23:59:00+00:00"), fight, owner)          # before the version
    old = _owner(version="2026-10-19", accepted_at="2026-10-20T00:00:00+00:00")
    assert not qct.usable(_answer(), {"profile_id": old["profile_id"]}, old)               # never accepted it
    late = _owner(accepted_at=f"{SINCE}T11:00:00+00:00")
    assert not qct.usable(_answer(), {"profile_id": late["profile_id"]}, late)             # answered before accepting


def test_never_from_an_under_18_account_and_only_strike_checks(monkeypatch):
    owner = _owner()
    fight = {"profile_id": owner["profile_id"]}
    assert not qct.usable(_answer(source="correction"), fight, owner)
    monkeypatch.setattr("core.social.is_minor_account", lambda account: True)
    assert not qct.usable(_answer(), fight, owner)


def test_turning_it_off_takes_out_answers_already_given_and_on_brings_them_back():
    owner = _owner()
    fight = {"profile_id": owner["profile_id"]}
    qct.set_choice(owner["profile_id"], False, SINCE)
    assert qct.objected(owner["profile_id"]) and not qct.usable(_answer(), fight, owner)
    qct.set_choice(owner["profile_id"], True, SINCE)
    assert not qct.objected(owner["profile_id"]) and qct.usable(_answer(), fight, owner)


def test_the_window_is_measurements_around_the_moment(tmp_path):
    tracking = tmp_path / "tracking.jsonl"
    lines = [{"time_seconds": t, "source_frame": int(t * 10), "frame_path": "secret.jpg",
              "fighter_A": {"observation": {"box": [1, 2, 3, 4], "keypoints": [[1, 1]], "keypoint_conf": [0.9]}},
              "fighter_B": {"observation": None}} for t in (10.5, 11.2, 12.0, 12.9, 13.5)]
    tracking.write_text("\n".join(json.dumps(line) for line in lines))
    window = qct._window(tracking, 12.0)
    assert [w["time_seconds"] for w in window] == [11.2, 12.0, 12.9]
    assert set(window[0]) == {"time_seconds", "fighter_A", "fighter_B"}
    assert window[0]["fighter_A"]["box"] == [1, 2, 3, 4] and window[0]["fighter_B"]["box"] is None


def test_the_worker_endpoint_needs_the_worker_token():
    refused = BrowserClient(web.app).get("/api/worker/quick-check-training")
    assert refused.status_code in {401, 403, 404, 503} and "answers" not in refused.text


def test_settings_switch_records_the_choice_and_is_hidden_from_minors(monkeypatch):
    owner = _owner()
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(owner["id"]))
    page = client.get("/settings/privacy").text
    assert "Let my quick-check answers help train strike counting" in page
    client.post("/settings/quick-check-training", data={}, follow_redirects=False)
    assert qct.objected(owner["profile_id"])
    client.post("/settings/quick-check-training", data={"enabled": "true"}, follow_redirects=False)
    assert not qct.objected(owner["profile_id"])
    monkeypatch.setattr(web.social, "is_minor_account", lambda account: True)
    assert "Let my quick-check answers help train strike counting" not in client.get("/settings/privacy").text


def test_the_training_machine_gets_only_usable_answers_with_their_window(tmp_path, monkeypatch):
    from core.config import SETTINGS

    owner, old = _owner(), _owner(version="2026-10-19", accepted_at="2026-10-20T00:00:00+00:00")
    jobs = {}
    for who in (owner, old):
        job_id = f"qct{uuid.uuid4().hex[:9]}"
        db.save_fight(job_id, int(who["profile_id"]), "f.mp4", "", "", "sparring", "K1", "A", {})
        (tmp_path / job_id).mkdir()
        (tmp_path / job_id / "tracking.jsonl").write_text(json.dumps(
            {"time_seconds": 12.0, "fighter_A": {"observation": {"box": [1, 2, 3, 4]}}, "fighter_B": {}}))
        db.save_annotation(job_id, 12.0, "K1", {"fighter": "A", "family": "punch", "technique": "punch"},
                           {"fighter": "A", "family": "none", "technique": "none", "verdict": "not_a_strike",
                            "source": "strike_check"})
        jobs[who["id"]] = job_id
    with db.connection() as con:                                     # both answered on the new policy's day
        con.execute("UPDATE annotations SET created_at=? WHERE job_id IN (?, ?)",
                    (f"{SINCE}T10:00:00+00:00", *jobs.values()))
    monkeypatch.setattr(web, "OUTPUTS", tmp_path)
    previous = (SETTINGS.analysis_worker_mode, SETTINGS.worker_token)
    object.__setattr__(SETTINGS, "analysis_worker_mode", "remote")
    object.__setattr__(SETTINGS, "worker_token", "test-worker-token")
    try:
        answers = BrowserClient(web.app).get("/api/worker/quick-check-training",
                                             headers={"Authorization": "Bearer test-worker-token"}).json()["answers"]
    finally:
        object.__setattr__(SETTINGS, "analysis_worker_mode", previous[0])
        object.__setattr__(SETTINGS, "worker_token", previous[1])
    mine = [a for a in answers if a["job_id"] in jobs.values()]
    assert [a["job_id"] for a in mine] == [jobs[owner["id"]]]          # the old-policy owner's answer is left out
    assert mine[0]["verdict"] == "not_a_strike" and mine[0]["window"][0]["fighter_A"]["box"] == [1, 2, 3, 4]
