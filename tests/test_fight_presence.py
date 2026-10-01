"""Only fight footage is measured, and too little of it means no numbers.

QA 2026-09: a mostly-interview news clip got "150 strike attempts", and a
street scene with a bystander boxed as Fighter B got a full report.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from core import fight_presence as fp
from core.types import PersonObservation

FPS = 10.0


def _person(x, y=100.0, height=200.0, hips=True, track_id=1):
    keypoints = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros((17,), dtype=np.float32)
    for index in range(17):
        keypoints[index] = (x + 10 + (index % 3) * 5, y + height * index / 17.0)
        conf[index] = 0.9
    if not hips:
        conf[11:] = 0.0
    return PersonObservation(track_id=track_id, box=np.asarray([x, y, x + height * 0.4, y + height], dtype=np.float32),
                             confidence=0.9, keypoints=keypoints, keypoint_conf=conf)


def _feed(presence, seconds_from, seconds_to, make):
    t = seconds_from
    while t < seconds_to - 1e-9:
        a, b, people = make(t)
        presence.observe(t, a, b, people)
        t += 1.0 / FPS


def _fighting(t):
    # Close, full-length, moving: footwork plus limbs.
    a = _person(300 + 40 * np.sin(t * 3), track_id=1)
    b = _person(420 + 40 * np.cos(t * 3), track_id=2)
    return a, b, [a, b]


def _apart(t):
    a = _person(100 + 30 * np.sin(t * 3), track_id=1)
    b = _person(1500, track_id=2)
    return a, b, [a, b]


def _interview(t):
    a = _person(300 + 2 * np.sin(t), height=600, hips=False, track_id=1)
    b = _person(700, height=600, hips=False, track_id=2)
    return a, b, [a, b]


def _graphics(t):
    return None, None, []


def _tracking_lost(t):
    a, b, people = _fighting(t)
    return a, None, people


def test_a_real_exchange_is_fight_footage():
    presence = fp.FightPresence(0.0, 20.0)
    _feed(presence, 0.0, 20.0, _fighting)
    summary = presence.summary()
    assert summary["fight_seconds"] == 20.0
    assert summary["sufficient"]


@pytest.mark.parametrize("make,reason", [
    (_apart, "apart"),
    (_interview, "not_full_body"),
    (_graphics, "nobody"),
])
def test_non_fight_footage_is_left_out_with_its_reason(make, reason):
    presence = fp.FightPresence(0.0, 20.0)
    _feed(presence, 0.0, 20.0, make)
    summary = presence.summary()
    assert summary["fight_seconds"] == 0.0
    assert summary["main_exclusion_reason"] == reason
    assert not summary["sufficient"]


def test_losing_a_fighter_is_not_evidence_of_no_fight():
    """A tracking gap is reported as coverage; calling it "not a fight" would be false."""
    presence = fp.FightPresence(0.0, 20.0)
    _feed(presence, 0.0, 20.0, _tracking_lost)
    assert presence.summary()["fight_seconds"] == 20.0


def test_a_news_clip_keeps_only_its_fight():
    presence = fp.FightPresence(0.0, 60.0)
    _feed(presence, 0.0, 20.0, _interview)
    _feed(presence, 20.0, 40.0, _fighting)
    _feed(presence, 40.0, 60.0, _graphics)
    summary = presence.summary()
    assert summary["segments"] == [(20.0, 40.0)]
    assert summary["fight_seconds"] == 20.0
    assert summary["excluded_seconds"] == 40.0
    assert presence.is_fight(25.0) and not presence.is_fight(5.0) and not presence.is_fight(50.0)


def test_a_short_pause_inside_a_fight_is_kept():
    presence = fp.FightPresence(0.0, 10.0)
    _feed(presence, 0.0, 4.0, _fighting)
    _feed(presence, 4.0, 6.0, _apart)
    _feed(presence, 6.0, 10.0, _fighting)
    assert presence.summary()["segments"] == [(0.0, 10.0)]


def test_too_little_fight_footage_is_not_sufficient():
    presence = fp.FightPresence(0.0, 30.0)
    _feed(presence, 0.0, 4.0, _fighting)
    _feed(presence, 4.0, 30.0, _graphics)
    summary = presence.summary()
    assert summary["fight_seconds"] == 4.0 and not summary["sufficient"]


# --- the result page --------------------------------------------------------

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "report_sample.json"


@pytest.fixture
def result_client(tmp_path, monkeypatch):
    import app.main as web
    from browser_client import BrowserClient
    from core import db
    from core.auth import issue_session

    account = db.create_account(f"numbers-{uuid.uuid4().hex}@example.test", "not-a-login-hash")
    db.set_plan_override(int(account["id"]), "athlete_pro")
    reports = {}

    def install(report):
        path = tmp_path / f"{uuid.uuid4().hex}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        reports["path"] = path

    monkeypatch.setattr(web, "_authorized_job", lambda request, job_id: {"status": "complete", "ruleset": "K1"})
    monkeypatch.setattr(web, "_require_completed_artifact", lambda job_id, name: reports["path"])
    monkeypatch.setattr(web, "get_fight", lambda job_id: {"profile_id": account["profile_id"], "job_id": job_id})
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(account["id"]))
    yield client, install
    client.close()


def _report(**integrity):
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report.setdefault("integrity", {}).update(integrity)
    if integrity.get("identity_evidence_trusted") is False:
        # The page re-applies the identity gate from the tracking data, so the
        # failure has to be in the tracking: two fighters it cannot tell apart.
        report.setdefault("tracking", {})["fighters_separable"] = False
        report["tracking"]["fighter_pair_similarity"] = 0.97
    return report


def test_too_little_fight_footage_shows_no_numbers(result_client):
    client, install = result_client
    report = _report(fight_footage_sufficient=False)
    report["video"]["fight_footage"] = {
        "fight_seconds": 4.0, "analysed_seconds": 60.0, "excluded_seconds": 56.0, "sufficient": False,
        "main_exclusion_text": fp.REASON_TEXT["not_full_body"], "segments": [[20.0, 24.0]],
    }
    install(report)
    page = client.get("/result/abcdef123456").text
    assert "Too little fight footage to measure." in page
    assert "Only 0:04 of this video could be used as fight footage" in page
    assert 'class="fight-vitals"' not in page
    assert "Your fight in numbers" not in page
    assert 'action="/share/abcdef123456"' not in page


def test_an_unverified_report_greys_out_its_numbers_and_shares_nothing(result_client):
    client, install = result_client
    install(_report(identity_evidence_trusted=False))
    page = client.get("/result/abcdef123456").text
    assert "Unverified &mdash; may include other people" in page or "Unverified — may include other people" in page
    assert 'data-unverified="true"' in page
    assert "how far you travel each second" not in page
    assert 'action="/share/abcdef123456"' not in page
    assert 'aria-disabled="true"' in page
    assert "Nothing is shared or posted from an unverified report." in page
    refused = client.post("/share/abcdef123456", follow_redirects=False)
    assert refused.status_code == 409


def test_a_verified_report_reads_as_the_fighters_own(result_client):
    client, install = result_client
    report = _report(identity_evidence_trusted=True)
    report["video"]["fight_footage"] = {
        "fight_seconds": 100.0, "analysed_seconds": 116.0, "excluded_seconds": 16.0, "sufficient": True,
        "main_exclusion_text": fp.REASON_TEXT["nobody"], "segments": [[0.0, 100.0]],
    }
    report["video"]["analysed_span"] = {"start_seconds": 0.0, "end_seconds": 116.0, "video_duration_seconds": 116.0}
    install(report)
    page = client.get("/result/abcdef123456").text
    assert 'data-unverified="true"' not in page
    assert "Fight footage analysed" in page and "1:40 of 1:56" in page
    assert "0:16 of the analysed footage was left out of every number" in page
