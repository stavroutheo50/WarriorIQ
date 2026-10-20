"""Strikes marked by a person while watching, against the automatic count (2026-10-20).

The owner asked for a way to make the strike count accurate. The first step is
measuring it on whole fights: a person taps each strike as it is thrown, and
the page says how many counted strikes were real and how many real ones were
missed (core/strike_marks.py, app.main.strike_marks_page).
"""

from __future__ import annotations

import json
import uuid

import pytest
from browser_client import BrowserClient

import app.main as web
from core import db as database
from core import strike_marks


def _counted(*rows):
    return [{"fighter": fighter, "family": family, "seconds": seconds} for fighter, family, seconds in rows]


def _mark(t, fighter, family=None, landed=None):
    return {"t": t, "fighter": fighter, "family": family, "landed": landed}


# ------------------------------------------------------------------ matching
def test_a_tap_just_after_a_counted_strike_is_the_same_strike():
    result = strike_marks.compare([_mark(10.6, "A")], _counted(("A", "punch", 10.0)), 30)
    assert (result["marked"], result["counted"], result["matched"]) == (1, 1, 1)
    assert result["counted_real_share"] == 1.0 and result["marked_found_share"] == 1.0


def test_a_tap_too_late_or_for_the_other_fighter_is_not():
    late = strike_marks.compare([_mark(11.2, "A")], _counted(("A", "punch", 10.0)), 30)
    other = strike_marks.compare([_mark(10.3, "B")], _counted(("A", "punch", 10.0)), 30)
    assert late["matched"] == 0 and other["matched"] == 0


def test_each_strike_and_each_tap_is_used_once_nearest_first():
    # Two quick punches, two taps: one pair each, not both taps on the first.
    result = strike_marks.compare([_mark(10.3, "A"), _mark(10.9, "A")],
                                  _counted(("A", "punch", 10.0), ("A", "punch", 10.6)), 30)
    assert result["matched"] == 2


def test_only_the_watched_part_is_compared():
    # A strike counted after the person stopped watching is neither found nor extra.
    result = strike_marks.compare([_mark(5.2, "A")], _counted(("A", "kick", 5.0), ("A", "kick", 50.0)), 20)
    assert (result["counted"], result["matched"], result["covered_seconds"]) == (1, 1, 20.0)


def test_the_type_is_checked_only_where_the_person_gave_it():
    result = strike_marks.compare([_mark(5.2, "A", "kick"), _mark(8.2, "B")],
                                  _counted(("A", "punch", 5.0), ("B", "kick", 8.0)), 20)
    assert (result["type_checked"], result["type_right"]) == (1, 0)


def test_nothing_marked_and_nothing_counted_has_no_shares():
    result = strike_marks.compare([], [], 10)
    assert result["counted_real_share"] is None and result["marked_found_share"] is None


@pytest.mark.parametrize("bad", [
    "not a list", [{"t": -1, "fighter": "A"}], [{"t": 5, "fighter": "C"}], [{"t": "x", "fighter": "A"}],
    [{"t": 5, "fighter": "A", "family": "elbow"}], [{"t": 5, "fighter": "A", "landed": "yes"}],
    [{"t": 500, "fighter": "A"}],
])
def test_bad_marks_are_refused(bad):
    with pytest.raises(ValueError):
        strike_marks.clean_marks(bad, duration=60)


def test_marks_are_kept_in_time_order():
    cleaned = strike_marks.clean_marks([{"t": 9, "fighter": "b"}, {"t": 2.5, "fighter": "A", "family": "kick"}], 60)
    assert [m["t"] for m in cleaned] == [2.5, 9] and cleaned[1]["fighter"] == "B"


# --------------------------------------------------------------------- pages
def _owner_with_fight():
    client = BrowserClient(web.app)
    email = f"mark-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/signup", data={"email": email, "password": "Strong-Local-Password", "accept_terms": "true",
                                 "age_confirmed": "true"}, follow_redirects=False)
    profile_id = int(database.get_account_by_email(email)["profile_id"])
    job_id = "mk" + uuid.uuid4().hex[:10]
    directory = web.OUTPUTS / job_id
    directory.mkdir(parents=True, exist_ok=True)
    feed = [{"kind": "strike", "fighter": "A", "family": "punch", "time_seconds": 3.0, "round_number": 1},
            {"kind": "strike", "fighter": "B", "family": "kick", "time_seconds": 7.5, "round_number": 1}]
    report = {"setup": {"ruleset": "K1"}, "scorecard": {"sport": "kickboxing"}, "event_feed": feed,
              "video": {"analysed_span": {"start_seconds": 0, "end_seconds": 60, "video_duration_seconds": 60}}}
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    video = web.UPLOADS / f"{job_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video-placeholder")
    database.save_fight(job_id, profile_id, "f.mp4", str(video), str(directory / "report.json"),
                        "sparring", "K1", "BOTH", {}, "2999-01-01T00:00:00+00:00")
    return client, job_id


def test_the_owner_marks_strikes_and_sees_the_comparison():
    client, job_id = _owner_with_fight()
    page = client.get(f"/mark/{job_id}")
    assert page.status_code == 200 and "Mark the strikes" in page.text
    saved = client.post(f"/api/strike-marks/{job_id}", json={
        "marks": [{"t": 3.4, "fighter": "A", "family": "punch"}, {"t": 20.0, "fighter": "A"}],
        "covered_seconds": 30, "playback_rate": 0.5})
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["saved"] == 2
    comparison = body["comparison"]
    assert (comparison["marked"], comparison["counted"], comparison["matched"]) == (2, 2, 1)
    stored = json.loads((web.OUTPUTS / job_id / strike_marks_file()).read_text(encoding="utf-8"))
    assert stored["covered_seconds"] == 30 and len(stored["marks"]) == 2
    # Going back to re-watch does not shrink what was seen.
    client.post(f"/api/strike-marks/{job_id}", json={"marks": [], "covered_seconds": 5})
    assert json.loads((web.OUTPUTS / job_id / strike_marks_file()).read_text())["covered_seconds"] == 30
    assert f'href="/mark/{job_id}"' in client.get(f"/replay/{job_id}").text


def strike_marks_file():
    return web.STRIKE_MARKS_FILE


def test_someone_else_cannot_open_or_save_the_marks():
    _, job_id = _owner_with_fight()
    stranger = BrowserClient(web.app)
    stranger.post("/signup", data={"email": f"other-{uuid.uuid4().hex[:8]}@example.com",
                                   "password": "Strong-Local-Password", "accept_terms": "true",
                                   "age_confirmed": "true"}, follow_redirects=False)
    assert stranger.get(f"/mark/{job_id}").status_code == 404
    refused = stranger.post(f"/api/strike-marks/{job_id}", json={"marks": [], "covered_seconds": 1})
    assert refused.status_code == 403
    assert not (web.OUTPUTS / job_id / web.STRIKE_MARKS_FILE).exists()
    assert BrowserClient(web.app).get(f"/mark/{job_id}", follow_redirects=False).status_code == 303


def test_a_bad_mark_is_refused_and_nothing_is_written():
    client, job_id = _owner_with_fight()
    refused = client.post(f"/api/strike-marks/{job_id}", json={"marks": [{"t": 5, "fighter": "Z"}]})
    assert refused.status_code == 400
    assert not (web.OUTPUTS / job_id / web.STRIKE_MARKS_FILE).exists()
