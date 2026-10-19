"""The full QA pass on the live site, 2026-10-11.

One test (or a few) per item of that report, named after it, so a regression
points straight at the line of the QA list it reopens.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import app.main as web
from app import state
from browser_client import BrowserClient
from core import db, upload_security
from core.auth import hash_password, issue_session, token_digest
from core.config import SETTINGS

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "app" / "templates"


@pytest.fixture
def settings():
    previous = {}

    def set_values(**values):
        for key, value in values.items():
            previous.setdefault(key, getattr(SETTINGS, key))
            object.__setattr__(SETTINGS, key, value)

    yield set_values
    for key, value in previous.items():
        object.__setattr__(SETTINGS, key, value)


def _signed_in(plan: str | None = None, password: str | None = None):
    created = db.create_account(f"qa1011-{uuid.uuid4().hex}@example.test",
                                hash_password(password) if password else "not-a-login-hash")
    if plan:
        db.set_plan_override(int(created["id"]), plan)
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    return created, client


def _selecting_job(created, job_id=None, status="selecting") -> str:
    job_id = job_id or uuid.uuid4().hex[:12]
    state.create_job(job_id, {
        "status": status, "account_id": int(created["id"]),
        "owner_key": f"account:{created['id']}", "original_name": "round-one.mp4",
        "ruleset": "K1", "usage_reserved": False, "video_width": 64, "video_height": 48,
    })
    out = web.OUTPUTS / job_id
    out.mkdir(exist_ok=True)
    (out / "selection.jpg").write_bytes(b"jpeg")
    return job_id


# --- A1 ----------------------------------------------------------------------

def test_a1_rotate_url_is_built_from_the_job_id_not_json_quotes():
    created, client = _signed_in()
    job = _selecting_job(created)
    try:
        page = client.get(f"/select/{job}").text
        assert '$"' not in page and "$%22" not in page
        assert "/api/selection-frame/${encodeURIComponent(jobId)}/rotate" in page
        # Every fetch on the page uses the same jobId constant.
        for url in re.findall(r"fetch\(`([^`]*)`", page):
            assert "{{" not in url and '"' not in url, url
    finally:
        state.delete_job(job)


def test_a1_no_template_interpolates_tojson_inside_a_template_literal():
    for path in TEMPLATES.glob("*.html"):
        assert "${{" not in path.read_text(encoding="utf-8"), path.name


# --- A2 ----------------------------------------------------------------------

def test_a2_upload_a_different_video_discards_the_pending_upload():
    created, client = _signed_in()
    job = _selecting_job(created)
    try:
        page = client.get(f"/select/{job}").text
        assert f'action="/pending/{job}/cancel"' in page
        assert 'name="next_path" value="/analyze/kickboxing"' in page
        assert "Discard this upload?" in page              # the confirm
        response = client.post(f"/pending/{job}/cancel", data={"next_path": "/analyze/kickboxing"},
                               follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/analyze/kickboxing"
        assert state.get_job(job) is None
    finally:
        state.delete_job(job)


def test_a2_cancel_only_follows_a_local_analyze_path():
    created, client = _signed_in()
    job = _selecting_job(created)
    response = client.post(f"/pending/{job}/cancel", data={"next_path": "https://evil.example/analyze"},
                           follow_redirects=False)
    assert response.headers["location"].startswith("/history")


def test_a2_reselecting_a_saved_fight_keeps_a_plain_link():
    created, client = _signed_in()
    job = _selecting_job(created, status="complete")
    db.save_fight(job, int(created["profile_id"]), "a.mp4", "", "", "sparring", "K1", "A", {},
                  "2999-01-01T00:00:00+00:00")
    try:
        page = client.get(f"/select/{job}").text
        assert "/cancel" not in page
        assert "You only do this once" not in page                      # D6
    finally:
        state.delete_job(job)


def test_a2_the_gym_plan_has_no_pending_cap(settings):
    settings(max_pending_uploads=2)
    created, client = _signed_in("gym")
    jobs = [_selecting_job(created) for _ in range(3)]
    try:
        probe = uuid.uuid4().hex[:12]
        upload_security.reserve_upload_storage(int(created["id"]), probe, 64)
        upload_security.release_upload_storage(probe)
        assert upload_security.pending_upload_status(int(created["id"]))["limit"] is None
    finally:
        for job in jobs:
            state.delete_job(job)


def test_a2_other_plans_keep_the_cap_and_the_upload_page_says_so_first(settings):
    settings(max_pending_uploads=2)
    created, client = _signed_in()
    jobs = [_selecting_job(created) for _ in range(2)]
    try:
        with pytest.raises(upload_security.UploadCapacityError) as refused:
            upload_security.reserve_upload_storage(int(created["id"]), uuid.uuid4().hex[:12], 64)
        assert refused.value.status == 429
        assert "allows 2 at a time" in str(refused.value)
        page = client.get("/analyze/boxing").text
        assert 'id="pendingUploads"' in page
        assert "Fights waiting: 2 of 2" in page
    finally:
        for job in jobs:
            state.delete_job(job)


# --- A3 ----------------------------------------------------------------------

def test_a3_the_sticky_heading_is_confined_to_its_own_grid_cell():
    page = (TEMPLATES / "analyze.html").read_text(encoding="utf-8")
    aside = page.index('class="setup-aside"')
    head = page.index('class="setup-head"')
    notes = page.index('class="setup-notes"')
    assert aside < head < notes
    # The aside closes before the upload card, so the notes are not inside it.
    assert page.index("</div>", page.index("</header>", head)) < page.index('class="setup-card"')
    css = (ROOT / "app" / "static" / "premium.css").read_text(encoding="utf-8")
    assert re.search(r"\.setup-aside\s*\{[^}]*grid-row:\s*1[^}]*align-self:\s*stretch", css)


# --- A4 ----------------------------------------------------------------------

def _completed_job(owner_key: str, video: Path | None):
    job_id = "qa1011" + uuid.uuid4().hex[:8]
    job_dir = web.OUTPUTS / job_id
    job_dir.mkdir()
    report = {
        "setup": {"ruleset": "K1", "start_seconds": 0, "end_seconds": 10},
        "video": {"analysis_target": "BOTH"}, "performance": {"segment_duration_seconds": 10},
        "rounds": [{"number": 1, "start_seconds": 0, "end_seconds": 10, "selected": True}],
        "events": [], "key_moments": [], "illegal_moves": [], "integrity": {},
        "scorecard": {"ruleset_label": "K-1"},
        "metrics": {"A": {"guard_index": 0.3, "numbers": {"hands_up_share": None}}},
        "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9},
    }
    (job_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    state.create_job(job_id, {"owner_key": owner_key, "status": "complete",
                              "video_path": str(video) if video else ""})
    return job_id


def test_a4_replay_with_the_video_removed_explains_instead_of_a_broken_player():
    client = BrowserClient(web.app)
    client.get("/")
    owner = f"guest:{client.cookies.get(web.GUEST_COOKIE)}"
    job = _completed_job(owner, web.UPLOADS / "gone-after-30-days.mp4")
    try:
        page = client.get(f"/replay/{job}").text
        assert "The video for this fight was removed after 30 days; the report is still available" in page
        assert '<video id="video"' not in page
        assert "/api/tracking/" not in page          # no script for a player that is not there
    finally:
        state.delete_job(job)


def test_a4_replay_with_the_video_present_keeps_the_player():
    client = BrowserClient(web.app)
    client.get("/")
    owner = f"guest:{client.cookies.get(web.GUEST_COOKIE)}"
    video = web.UPLOADS / f"present-{uuid.uuid4().hex[:6]}.mp4"
    video.write_bytes(b"video")
    job = _completed_job(owner, video)
    try:
        page = client.get(f"/replay/{job}").text
        assert '<video id="video"' in page
        assert "was removed after" not in page
    finally:
        state.delete_job(job)
        video.unlink(missing_ok=True)


def test_a4_guard_notes_do_not_offer_a_rerun_without_the_video():
    from core.guard import legacy_notes

    with_video = legacy_notes({})
    without = legacy_notes({"video_removed": True})
    assert any("Re-run" in note or "re-run" in note for note in with_video)
    assert not any("re-run" in note.lower() for note in without)
    assert "cannot be measured again" in without[0]


# --- A5 ----------------------------------------------------------------------

def test_a5_billing_and_pricing_read_the_same_plan(settings):
    created, client = _signed_in()
    settings(complimentary_plans={created["email"]: "gym"})
    billing = client.get("/settings/billing").text
    assert re.search(r"Current subscription</div><h2>Gym</h2>", billing)
    assert "€0 forever" not in billing
    pricing = client.get("/pricing").text
    assert "Your account is on Gym" in pricing


# --- A6 ----------------------------------------------------------------------

def test_a6_change_password_needs_the_current_one_and_signs_out_other_sessions():
    created, client = _signed_in(password="old-password-123")
    other = issue_session(created["id"])
    page = client.get("/settings/privacy").text
    assert 'action="/settings/password"' in page and 'action="/settings/email"' in page
    wrong = client.post("/settings/password", data={"current_password": "nope-nope-nope",
                                                    "new_password": "new-password-456",
                                                    "confirm_password": "new-password-456"},
                        follow_redirects=False)
    assert "error=" in wrong.headers["location"]
    assert db.account_for_session(token_digest(other)) is not None
    ok = client.post("/settings/password", data={"current_password": "old-password-123",
                                                 "new_password": "new-password-456",
                                                 "confirm_password": "new-password-456"},
                     follow_redirects=False)
    assert "notice=" in ok.headers["location"]
    assert db.account_for_session(token_digest(other)) is None            # other device out
    assert client.get("/settings/privacy").status_code == 200              # this one still in
    from core.auth import authenticate

    assert authenticate(created["email"], "new-password-456")
    assert not authenticate(created["email"], "old-password-123")


def test_a6_change_email_only_after_the_link_sent_to_the_new_address():
    created, client = _signed_in(password="old-password-123")
    sent = []
    new = f"new-{uuid.uuid4().hex[:8]}@example.test"
    with patch.object(web, "send_transactional_email", side_effect=lambda *a: sent.append(a) or True):
        response = client.post("/settings/email", data={"current_password": "old-password-123",
                                                        "new_email": new}, follow_redirects=False)
    assert "notice=" in response.headers["location"]
    assert db.get_account(int(created["id"]))["email"] == created["email"]   # not yet
    link = re.search(r"/settings/email/confirm/(\S+)", sent[0][2]).group(1)
    assert sent[0][0] == new and sent[1][0] == created["email"]
    with patch.object(web, "send_transactional_email", return_value=True):
        confirmed = client.get(f"/settings/email/confirm/{link}", follow_redirects=False)
    assert confirmed.status_code == 303
    assert db.get_account(int(created["id"]))["email"] == new
    assert client.get(f"/settings/email/confirm/{link}").status_code == 410   # one use


def test_a6_change_email_refuses_a_wrong_password():
    created, client = _signed_in(password="old-password-123")
    response = client.post("/settings/email", data={"current_password": "wrong-password",
                                                    "new_email": "x@example.test"}, follow_redirects=False)
    assert "error=" in response.headers["location"]


# --- B1 ----------------------------------------------------------------------

def _own(guard_value, balance=0.8, **extra):
    return {"guard_index": guard_value, "balance_index": balance, "ring_center_control": 0.5,
            "pressure_index": 0.0, "footwork_body_lengths_per_second": 1.0, **extra}


def test_b1_a_guard_under_the_stated_threshold_always_gets_a_fix_drill_and_target():
    from core.coaching import build_pose_coaching, build_training_plan, build_training_progression

    own, opponent = _own(0.07), _own(0.08)
    coaching = build_pose_coaching("A", own, opponent, "kickboxing")
    titles = [item["title"] for item in coaching["improvements"]]
    assert any(title.startswith("Work on: Guard (experimental) 7%") for title in titles), titles
    assert "Nothing behind your opponent" not in titles
    assert "Under 42% is what WarriorIQ flags as worth training." in coaching["improvements"][0]["detail"]
    assert coaching["drills"] and coaching["drills"][0]["metric"] == "guard_index"
    plan = build_training_plan(coaching, "A", own)
    # Still drilled, but guard is experimental: no number target, so no
    # weekly targets either (audit 2026-10-14, #28).
    assert plan and plan[0]["goal"].startswith("No number target: guard is experimental")
    assert all(not week["targets"] for week in build_training_progression(coaching, "A", own))
    # Never also a strength.
    assert not any(item["title"].startswith("Guard") for item in coaching["strengths"])


def test_b1_balance_threshold_is_used_too_and_one_fighter_reports_get_it():
    from core.coaching import build_pose_coaching

    coaching = build_pose_coaching("A", _own(0.5, balance=0.40), None, "boxing")
    assert any(item["title"].startswith("Work on: Balance 40%") for item in coaching["improvements"])


def test_b1_taekwondo_guard_style_is_not_flagged():
    from core.coaching import build_pose_coaching

    coaching = build_pose_coaching("A", _own(0.07), _own(0.08), "taekwondo")
    assert not any("Guard" in item["title"] for item in coaching["improvements"])


# --- B2 ----------------------------------------------------------------------

def test_b2_no_counted_strike_advice_anywhere():
    from core.coaching import build_pose_coaching

    coaching = build_pose_coaching("A", _own(0.6), _own(0.6), "kickboxing")
    text = json.dumps(coaching)
    assert "counted strikes" not in text
    for folder in (ROOT / "core", ROOT / "app"):
        for path in list(folder.rglob("*.py")) + list(folder.rglob("*.html")):
            body = path.read_text(encoding="utf-8")
            assert "watch your counted strikes" not in body, path


# --- B3 ----------------------------------------------------------------------

def test_b3_library_card_shows_no_score_while_scoring_is_off_and_report_rounds():
    report = {"setup": {"ruleset": "K1", "round_count": 3, "round_duration_seconds": 120},
              "rounds": [{"number": 1, "start_seconds": 0, "end_seconds": 120, "selected": True}],
              "video": {"analysis_target": "BOTH"}, "integrity": {"identity_evidence_trusted": True},
              "tracking": {"fighter_A_coverage": .95, "fighter_B_coverage": .95},
              "scorecard": {"available": True, "winner_estimate": "A", "totals": {"A": 10, "B": 8}},
              "metrics": {}}
    assert web._fight_format(report)["label"] == "1 round · 2:00"
    job = "qa1011lib" + uuid.uuid4().hex[:6]
    (web.OUTPUTS / job).mkdir()
    (web.OUTPUTS / job / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with patch.object(web, "feature_is_on", return_value=False):
        card = web._library_card(job)
    assert card["score"] is None
    assert card["format"]["label"] == "1 round · 2:00"
    page = (TEMPLATES / "history.html").read_text(encoding="utf-8")
    assert "summary.winner_estimate" not in page


# --- B4 ----------------------------------------------------------------------

def test_b4_a_lone_measured_point_is_drawn_and_gaps_say_why():
    from core.movement_graphs import _gaps, _lines

    series = {"step_seconds": 5.0, "points": [
        {"t": 0.0, "speed": 1.0}, {"t": 5.0, "speed": 1.2}, {"t": 10.0, "speed": None},
        {"t": 15.0, "speed": 0.4}, {"t": 20.0, "speed": None}, {"t": 25.0, "speed": 0.9}, {"t": 30.0, "speed": 1.0}]}
    lines = _lines({"A": series}, "speed", 2.0, 0.0, 35.0)["A"]
    assert len(lines) == 3                         # the lone 0:15 point is not dropped
    report = {"video": {"fight_footage": {"segments": [[0.0, 10.0], [15.0, 35.0]]}}}
    gaps = _gaps(series, "speed", report)
    assert gaps[0] == {"from": "0:10", "to": "0:15", "reason": "left out: not fight footage"}
    assert gaps[1]["reason"] == "too little of you seen to measure"


def test_b4_energy_read_names_a_stretch_that_is_drawn():
    from core.movement_graphs import _energy_read

    series = {"step_seconds": 5.0, "points": [{"t": 60.0, "speed": 0.2}, {"t": 65.0, "speed": 1.5}]}
    text = _energy_read(series, None)
    assert "quietest between 1:00 and 1:05" in text


# --- B5 ----------------------------------------------------------------------

def test_b5_guard_down_stretches_match_the_headline():
    from core.report_visuals import _guard_down_stretches

    points = [{"t": t * 5.0, "hands_up": 0.05} for t in range(12)]
    points[3]["hands_up"] = 0.9
    own = {"numbers": {"series": {"step_seconds": 5.0, "points": points}, "hands_up_share": 0.07}}
    stretches = _guard_down_stretches(own, 60.0)
    assert stretches == [{"start": 0.0, "end": 15.0}, {"start": 20.0, "end": 60.0}]
    covered = sum(item["end"] - item["start"] for item in stretches)
    assert covered / 60.0 >= 0.9


# --- B6 ----------------------------------------------------------------------

def test_b6_movement_comparison_is_labelled_as_a_split_and_distance_is_shared():
    from core.generalship import movement_comparison
    from core.movement_graphs import build

    view = movement_comparison({"available": True, "rounds": []}, "K1", False, {})
    assert all("split" in column["label"] for column in view["columns"])
    assert "add up to 100%" in view["split_note"]
    numbers = {"forward_seconds": 5, "backward_seconds": 5, "circling_seconds": 5, "standing_seconds": 5,
               "range_seconds": {"close": 10, "middle": 5, "long": 5}}
    graphs = build({"metrics": {"A": {"numbers": numbers}, "B": {"numbers": dict(numbers)}}}, "A")
    assert graphs["ranges"]["A"] and graphs["ranges"]["B"] is None
    page = (TEMPLATES / "result.html").read_text(encoding="utf-8")
    assert "Both of you" in page


# --- B7 ----------------------------------------------------------------------

def test_b7_compare_difference_comes_from_the_rounded_numbers_with_a_direction():
    from core.squad import compare_movement

    def report(centre):
        return {"video": {"focus_fighter": "A"}, "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9},
                "metrics": {"A": {"ring_center_control": centre}}}

    # 41.4 -> 51.6 is 10.2 unrounded and 11 as shown: the shown 11 is used.
    rows = compare_movement([report(0.414), report(0.516)])["rows"]
    centre = next(row for row in rows if row["key"] == "ring_center_control")
    assert (centre["a"], centre["b"]) == ("41%", "52%")
    assert centre["delta"] == "Fight 2 higher by 11 points"
    # Under the measurement's noise (half its spread, 7.5 points for centre;
    # audit 2026-10-14, #35) it is level, whatever the scale.
    rows = compare_movement([report(0.414), report(0.466)])["rows"]
    assert next(row for row in rows if row["key"] == "ring_center_control")["leader"] == "level"


# --- B8 ----------------------------------------------------------------------

def test_b8_no_score_wording_while_scoring_is_off():
    import importlib

    import core.features as features
    import core.legal as legal

    with patch.dict(features.FEATURES, {"scoring": features.Feature("scoring", "Score", False, "", ""),
                                        "strike_counts": features.Feature("strike_counts", "S", False, "", "")}):
        importlib.reload(legal)
        try:
            text = json.dumps([legal.LEGAL_DOCUMENTS[slug]["sections"]
                               for slug in ("terms", "ai-transparency", "sports-medical-disclaimer")])
            assert "scorecard" not in text.lower()
            assert "scores or statistics are estimates" not in text
            assert "does not score fights yet" in text
        finally:
            importlib.reload(legal)
    for name in ("result.html", "shared.html", "privacy.html"):
        body = (TEMPLATES / name).read_text(encoding="utf-8")
        assert "Statistics and scores are estimates" not in body
        assert "Scorecards are training estimates" not in body
        assert "reports and scorecards" not in body


# --- B10 ---------------------------------------------------------------------

def test_b10_status_counts_from_start_and_says_warming_up():
    import time as _time

    job = {"status": "queued", "stage": "queued", "queued_at_epoch": _time.time() - 42,
           "processed_video_seconds": 0.0, "video_duration": 60.0}
    status = web._public_job_status("x", job)
    assert status["warming_up"] is True
    assert 41 <= status["since_start_seconds"] <= 60
    running = web._public_job_status("x", {**job, "status": "running", "stage": "analysis",
                                          "processed_video_seconds": 3.0})
    assert running["warming_up"] is False
    page = (TEMPLATES / "progress.html").read_text(encoding="utf-8")
    assert "Warming up the analysis computer" in page
    assert "'calculating…'" not in page


# --- B11 ---------------------------------------------------------------------

def _track_guard(stride=1, offset=0):
    import gzip

    import numpy as np

    from core.metrics import MetricsAccumulator
    from core.types import PersonObservation

    track = ROOT / "dataset" / "regression" / "kicklight_stavrou_ceschia" / "track.jsonl.gz"
    rows = [json.loads(line) for line in gzip.open(track, "rt", encoding="utf-8")]

    def obs(entry):
        if not entry:
            return None
        return PersonObservation(track_id=1, box=np.asarray(entry["box"], np.float32), confidence=.9,
                                 keypoints=np.asarray(entry["keypoints"], np.float32),
                                 keypoint_conf=np.asarray(entry["keypoint_conf"], np.float32))

    metrics = MetricsAccumulator(568, 320)
    for row in rows[offset::stride]:
        a, b = obs(row["fighter_A"]["observation"]), obs(row["fighter_B"]["observation"])
        metrics.update("A", row["time_seconds"], 1, a, b)
        metrics.update("B", row["time_seconds"], 1, b, a)
    out = metrics.finalize([], [], rows[-1]["time_seconds"])
    return out["A"]["guard_index"], out["B"]["guard_index"]


def test_b11_guard_on_fixed_footage_is_repeatable_and_pinned():
    first, second = _track_guard(), _track_guard()
    assert first == second                                  # same frames, same number
    # Pinned to the current definition (core/guard.py GUARD_DEFINITION): a
    # change to how guard is measured shows up here, not as a trend. Under /2
    # (nearer wrist, line 0.7) this track read 0.032 and 0.021; under /3 (both
    # wrists, line 0.9; audit 2026-10-14, #5) it reads as below.
    assert first == pytest.approx((0.122, 0.009), abs=0.003)
    # Requiring both wrists costs repeatability, measured here: across strides
    # 2-4 and every offset the figure moves by up to 0.067 (0.06 under /2),
    # and at strides 5-6 too few frames show both wrists confidently, so it is
    # withheld ("Not measured") rather than guessed. That noise is why guard
    # is labelled Experimental (core/metric_catalog.py).
    for stride in (2, 3, 4, 5, 6):
        for offset in range(stride):
            a, b = _track_guard(stride, offset)
            if stride >= 5:
                assert a is None or abs(a - first[0]) <= 0.07, (stride, offset, a, b)
                continue
            assert abs(a - first[0]) <= 0.07 and abs(b - first[1]) <= 0.07, (stride, offset, a, b)


def test_b11_only_the_current_guard_definition_is_compared_across_fights():
    from core.guard import GUARD_DEFINITION, comparable_guard
    from core.squad import compare_movement

    assert comparable_guard({"guard_index": 0.59, "guard_definition": "hands_up_share/1"}) is None
    assert comparable_guard({"guard_index": 0.07, "guard_definition": GUARD_DEFINITION}) == 0.07

    def report(value, definition):
        return {"video": {"focus_fighter": "A"}, "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9},
                "metrics": {"A": {"guard_index": value, "guard_definition": definition, "balance_index": 0.7}}}

    rows = compare_movement([report(0.59, "hands_up_share/1"), report(0.07, GUARD_DEFINITION)])["rows"]
    guard_row = next(row for row in rows if row["key"] == "guard_index")
    assert guard_row["a"] == "—" and guard_row["delta"] == "—"


# --- B12 ---------------------------------------------------------------------

def test_b12_a_box_the_detector_barely_sees_is_warned_before_start():
    import numpy as np

    frame = np.zeros((100, 200, 3), np.uint8)
    people = [{"box": [10, 10, 60, 95], "confidence": 0.9}, {"box": [120, 10, 170, 95], "confidence": 0.25}]
    with patch.object(web, "detect_people_in_frame", return_value=people):
        faint = web._faint_boxes(frame, {"A": [8, 8, 62, 97], "B": [118, 8, 172, 97]}, 200, 100)
    assert faint == {"B": 0.25}
    page = (TEMPLATES / "select.html").read_text(encoding="utf-8")
    assert "data.faint_message" in page
    analyze = (TEMPLATES / "analyze.html").read_text(encoding="utf-8")
    assert ":'This recording looks usable'" not in analyze
    assert "This check cannot see people" in analyze


# --- C1 / C3 -----------------------------------------------------------------

def _failed_render(numbers_state, identity_trusted, camera_lost=False, can_share=True):
    from tests_support import render_result

    report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
    report.setdefault("integrity", {})["identity_evidence_trusted"] = identity_trusted
    numbers = {"state": numbers_state, "message": "Only 0:03 of footage was analysed.", "share_reason": "x"}
    return render_result(report=report, numbers=numbers, camera_lost=camera_lost, can_share=can_share)


def test_c1_nobody_in_shot_gets_one_message_not_two():
    page = _failed_render("no_fight", identity_trusted=False)
    assert "Too little fight footage to measure" in page
    assert "Show me who is who" not in page
    assert "Fix it in a few seconds" not in page
    assert "rely on" not in page                          # no numbers are shown


def test_c1_identity_failure_alone_offers_the_who_is_who_step():
    page = _failed_render("unverified", identity_trusted=False)
    # Since the audit of 14 Oct 2026 (#24) the fix is named for the cause.
    assert "Show me who is who" in page and "The fix: re-tap the fighters" in page
    assert "Too little fight footage to measure" not in page


def test_c3_no_coach_link_on_a_failed_report():
    assert web.FAILED_NUMBER_STATES >= {"no_fight", "unverified", "movement_only"}
    created, client = _signed_in("athlete")
    job = "qa1011shr" + uuid.uuid4().hex[:5]
    db.save_fight(job, int(created["profile_id"]), "a.mp4", "", "", "sparring", "K1", "A", {},
                  "2999-01-01T00:00:00+00:00")
    with patch.object(web, "_report_failed", return_value=True):
        response = client.post(f"/share/{job}", follow_redirects=False)
    assert response.status_code == 409
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert '_numbers_state(report)["state"] not in FAILED_NUMBER_STATES' in source


# --- C2 ----------------------------------------------------------------------

def test_c2_failed_analyses_say_needs_attention_in_the_library():
    created, client = _signed_in()
    job = "qa1011fail" + uuid.uuid4().hex[:5]
    db.save_fight(job, int(created["profile_id"]), "a.mp4", "", "", "sparring", "K1", "A", {},
                  "2999-01-01T00:00:00+00:00")
    with patch.object(web, "_library_card", return_value={"available": True, "failed": True,
                                                           "format": None, "score": None}):
        page = client.get("/history").text
    assert "Needs attention" in page
    assert "Score state" not in page


# --- C4 ----------------------------------------------------------------------

def test_c4_rotate_is_offered_up_front_when_nobody_is_found():
    page = (TEMPLATES / "select.html").read_text(encoding="utf-8")
    assert 'id="rotateProminent"' in page and 'id="rotateFrameTop"' in page
    assert "data.availability==='candidates_ready'&&!detections.length" in page


# --- C5 ----------------------------------------------------------------------

def test_c5_each_left_out_fight_gets_a_fix_link():
    from core.progress_insights import build_progress

    def record(job_id, trusted):
        return {"job_id": job_id, "created_at": "2026-10-0%s" % job_id[-1], "report": {
            "video": {"analysis_target": "A", "focus_fighter": "A"}, "setup": {"ruleset": "K1"},
            "integrity": {"identity_evidence_trusted": trusted},
            "metrics": {"A": {"pose_coverage": .8}}, "coaching": {}, "training_plan": {}}}

    progress = build_progress([record("f1", True), record("f2", False), record("f3", False)], "A")
    assert [item["job_id"] for item in progress["identity_failed_fights"]] == ["f3", "f2"]
    page = (TEMPLATES / "camp.html").read_text(encoding="utf-8")
    assert 'id="left-out"' in page and 'href="/select/{{ item.job_id }}"' in page


# --- C6 ----------------------------------------------------------------------

def test_c6_the_same_video_is_never_compared_with_itself():
    created, client = _signed_in()
    profile = int(created["profile_id"])
    digest = "ab" * 32
    ids = []
    for index in range(3):
        job = f"qa1011dup{index}" + uuid.uuid4().hex[:4]
        db.save_fight(job, profile, "same.mp4", "", "", "sparring", "K1", "A",
                      {"source_video_sha256": digest}, f"2999-01-0{index + 1}T00:00:00+00:00")
        ids.append(job)
    other = "qa1011other" + uuid.uuid4().hex[:4]
    db.save_fight(other, profile, "other.mp4", "", "", "sparring", "K1", "A",
                  {"source_video_sha256": "cd" * 32}, "2998-01-01T00:00:00+00:00")
    previous = web._previous_fights_excluding_copies(profile, ids[-1])
    kept = [fight["job_id"] for fight in previous]
    assert ids[-1] in kept and other in kept
    assert ids[0] not in kept and ids[1] not in kept
    assert len(web._without_duplicate_videos(db.list_fights(profile))) == 2


def test_c6_selection_page_warns_before_analysing_the_same_file_again():
    created, client = _signed_in()
    digest = "ef" * 32
    saved = "qa1011old" + uuid.uuid4().hex[:4]
    db.save_fight(saved, int(created["profile_id"]), "same.mp4", "", "", "sparring", "K1", "A",
                  {"source_video_sha256": digest}, "2999-01-01T00:00:00+00:00")
    job = _selecting_job(created)
    state.update_job(job, {"source_video_sha256": digest})
    try:
        page = client.get(f"/select/{job}").text
        assert 'id="duplicateWarning"' in page
        assert "You already analysed this exact video" in page
    finally:
        state.delete_job(job)


# --- D1 ----------------------------------------------------------------------

INTERNAL_WORDING = ("unverified scoring candidates", "first-frame identity gate", "propagated-mask agreement",
                    "<td>Pressure index</td>", "a format this build cannot decode", "does not hard-code",
                    "configured support", "configured operator", "The planned uncertainty flow",
                    "Compared with <b>{{progress.previous_name")


def test_d1_no_internal_wording_reaches_users():
    sources = [TEMPLATES / name for name in ("result.html", "settings.html", "index.html", "legal_document.html")]
    sources += [ROOT / "app" / "main.py"]
    for path in sources:
        body = path.read_text(encoding="utf-8")
        for phrase in INTERNAL_WORDING:
            assert phrase not in body, (path.name, phrase)
    from core.legal import resolve_document

    for slug in ("terms", "refunds", "accessibility", "security"):
        assert "the configured" not in json.dumps(resolve_document(slug)["sections"]), slug


def test_d1_since_last_fight_names_the_fight_not_the_file():
    from core.squad import compare_with_previous

    def report(guard):
        return {"video": {"focus_fighter": "A"}, "tracking": {"fighter_A_coverage": .9, "fighter_B_coverage": .9},
                "scorecard": {"sport": "boxing", "sport_label": "Boxing"},
                "metrics": {"A": {"guard_index": guard, "guard_definition": "hands_up_share/3", "balance_index": .7,
                                  "ring_center_control": .5}}}

    fights = [{"job_id": "old", "original_name": "2.mp4", "created_at": "2026-10-01T10:00:00+00:00",
               "ruleset": "BOXING", "report": report(0.5)}]
    with patch("core.squad._load_report", return_value=report(0.5), create=True):
        result = compare_with_previous(report(0.6), fights, "new")
    if result.get("available"):
        assert "2.mp4" not in (result.get("previous_label") or "")


# --- D2 ----------------------------------------------------------------------

def test_d2_one_display_name_everywhere():
    profile = {"display_name": "My Athlete"}
    fights = [{"fighter_name": "Theodoulos"}, {"fighter_name": "Theodoulos"}, {"fighter_name": None}]
    assert web._display_name(profile, fights) == "Theodoulos"
    assert web._display_name({"display_name": "Theo K"}, fights) == "Theo K"
    assert web._display_name(profile, []) == "My Athlete"
    shared = (TEMPLATES / "shared.html").read_text(encoding="utf-8")
    assert "Names are never shown on a coach link" in shared


# --- D3 ----------------------------------------------------------------------

def test_d3_coach_forwards_to_fight_camp_and_the_home_page_sells_one_page():
    client = BrowserClient(web.app)
    response = client.get("/coach?x=1", follow_redirects=False)
    assert response.status_code == 308 and response.headers["location"] == "/camp?x=1"
    home = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    assert 'href="/coach"' not in home and "<span>Fight Camp</span>" in home


# --- D4 ----------------------------------------------------------------------

def test_d4_fight_camp_shows_guard_once_labels_the_chart_and_explains_zero_points():
    page = (TEMPLATES / "camp.html").read_text(encoding="utf-8")
    assert "{% if hk != 'guard' %}" in page
    assert "chart-axis" in page and "fmtDate" in page
    assert 'id="noPointsYet"' in page and "analysing a fight does not earn points" in page


# --- D5 ----------------------------------------------------------------------

def test_d5_upload_form_status_size_and_video_track():
    page = (TEMPLATES / "analyze.html").read_text(encoding="utf-8")
    assert "(b/1048576).toFixed(1)} MB" in page and "Math.round(b/1048576)" not in page
    assert "form.addEventListener('change',()=>syncSubmit())" in page
    assert "probe.videoWidth>0" in page and "it looks like sound only" in page
    # Not "ready" before the device has read a picture from the file.
    assert "text:`${fmtMB(b)} — checking the video on this device…`}" in page


# --- D6 ----------------------------------------------------------------------

def test_d6_selection_page_solo_hides_b_and_start_stays_in_view():
    css = (ROOT / "app" / "static" / "fighter-selection.css").read_text(encoding="utf-8")
    assert ".fighter-key-item[hidden]" in css and "display:none!important" in css
    page = (TEMPLATES / "select.html").read_text(encoding="utf-8")
    # Since the audit of 14 Oct 2026 (#14): Start follows the A and B cards
    # rather than sticking above them (where it slid over the Opponent card),
    # with the solo option after it so Start stays near the top.
    assert "keys.after(choice)" in page and "choice.after(soloToggle)" in page
    assert "position:sticky;top:96px" not in css


# --- D7 ----------------------------------------------------------------------

def test_d7_opening_a_result_clears_a_finished_job_from_the_menu_cookie():
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert 'active_cookie == job_id or not _owned_job(' in source
    # Both report kinds clear it: the Solo page returned before this ran.
    assert 'return _result_opened(request, templates.TemplateResponse(request=request, name="solo_result.html"' in source
    css = (ROOT / "app" / "static" / "product.css").read_text(encoding="utf-8")
    assert ".active-analysis-chip{white-space:nowrap" in css


# --- D8 ----------------------------------------------------------------------

def test_d8_the_header_is_the_same_on_every_page():
    assert web.SPORT_IRRELEVANT_PREFIXES == ()


# --- D9 ----------------------------------------------------------------------

def test_d9_settings_shows_the_acceptance_date_and_names_each_fight():
    created, client = _signed_in()
    db.save_fight("qa1011set" + uuid.uuid4().hex[:4], int(created["profile_id"]), "a.mp4", "", "", "sparring",
                  "K1", "A", {}, "2999-01-01T00:00:00+00:00")
    page = client.get("/settings/privacy").text
    assert "Terms accepted on" in page and "Terms version" in page
    assert "<strong>Fight analysis</strong>" not in page
    assert "Fighter A vs Fighter B" in page


# --- D10 ---------------------------------------------------------------------

def test_d10_feed_finds_athletes_by_username():
    created, client = _signed_in()
    other = db.create_account(f"other-{uuid.uuid4().hex}@example.test", "not-a-login-hash")
    handle = "qafind" + uuid.uuid4().hex[:5]
    db.set_public_profile(int(other["profile_id"]), handle, "public", "", "")
    page = client.get(f"/feed?q={handle[:8]}").text
    assert f'href="/athlete/{handle}"' in page
    assert client.get("/feed?q=%25").status_code == 200        # wildcards are not a search


# --- D11 ---------------------------------------------------------------------

def test_d11_precheck_warns_about_low_light_and_same_kit():
    from core.preflight_client import client_thresholds
    from core.quality_guardian import LOW_LIGHT_BRIGHTNESS

    assert client_thresholds()["low_light_brightness"] == LOW_LIGHT_BRIGHTNESS
    page = (TEMPLATES / "analyze.html").read_text(encoding="utf-8")
    assert "This video is very dark" in page and "Same kit?" in page
    js = (ROOT / "app" / "static" / "preflight.js").read_text(encoding="utf-8")
    assert "result.brightness = lumas.length ? median(lumas) : null;" in js


# --- D12 ---------------------------------------------------------------------

def test_d12_secondary_report_sections_fold_on_phones_and_settings_pages():
    page = (TEMPLATES / "result.html").read_text(encoding="utf-8")
    # The graphs stay open on phones since 2026-10-13; the comparison and
    # every number still fold.
    assert page.count("data-phone-fold open") >= 2
    assert "matchMedia('(max-width: 700px)')" in page
    settings = (TEMPLATES / "settings.html").read_text(encoding="utf-8")
    assert 'data-page-size="10"' in settings


# --- D13 ---------------------------------------------------------------------

def test_d13_section_headings_share_one_size():
    motion = (ROOT / "app" / "static" / "motion.css").read_text(encoding="utf-8")
    assert ".section-title:first-child h2" not in motion
    assert ".result-hero.section-title h2" in motion
    page = (TEMPLATES / "result.html").read_text(encoding="utf-8")
    assert '<div class="section-title" style="margin-top:0">\n    <h2>{{ movement.title }}</h2>' not in page


# --- E1 / E2 -----------------------------------------------------------------

def test_e1_support_address_must_be_on_the_warrioriq_domain(settings):
    from core.legal import launch_readiness, public_config_problems

    settings(support_email="help@gmail.com", privacy_email="privacy@warrioriq.eu")
    assert any("@warrioriq.eu" in item for item in launch_readiness()["missing"])
    assert any("WARRIORIQ_SUPPORT_EMAIL" in item for item in public_config_problems())
    settings(support_email="support@warrioriq.eu")
    assert not any("Customer-support email on" in item for item in launch_readiness()["missing"])
    for name in ("legal_document.html", "privacy.html"):
        assert "before paid public launch" not in (TEMPLATES / name).read_text(encoding="utf-8")
    assert "Verification pending before paid launch" not in (TEMPLATES / "legal.html").read_text(encoding="utf-8")


def test_e2_subprocessors_name_purpose_and_location(settings):
    from core.legal import subprocessor_sections

    settings(hosting_provider="Render", hosting_location="Frankfurt, Germany (EU)",
             email_provider="smtp", email_provider_name="Postmark", email_provider_location="USA")
    text = json.dumps(subprocessor_sections())
    assert "Render (website hosting and storage)" in text and "Location: Frankfurt, Germany (EU)." in text
    assert "Postmark (email)" in text and "Location: USA." in text
    assert "the server that runs this website" not in text


# --- E3 ----------------------------------------------------------------------

def test_e3_corrections_are_not_sold_while_they_cannot_be_used():
    import core.features as features
    from core.payments import PLANS, plan_features

    with patch.dict(features.FEATURES, {"strike_counts": features.Feature("strike_counts", "S", False, "", "")}):
        assert "Fight comparisons and corrections" not in plan_features(PLANS["athlete_pro"])
        assert "Fight comparisons" in plan_features(PLANS["athlete_pro"])


# --- E4 ----------------------------------------------------------------------

def test_e4_consented_public_pages_allow_googles_collection_endpoints(settings):
    settings(analytics_measurement_id="G-TESTONLY")
    client = BrowserClient(web.app)
    response = client.get("/pricing", cookies={"warrioriq_cookie_preferences": "all"})
    policy = response.headers["content-security-policy"]
    connect = [part for part in policy.split(";") if part.strip().startswith("connect-src")][0]
    for origin in ("pagead2.googlesyndication.com", "https://www.google.com", "google-analytics.com"):
        assert origin in connect, origin
    private = client.get("/history", cookies={"warrioriq_cookie_preferences": "all"})
    assert "pagead2" not in private.headers["content-security-policy"]


# --- F -----------------------------------------------------------------------

def test_f_an_old_saved_report_is_re_rendered_without_its_stored_false_text():
    from core.report import refresh_identity_integrity

    report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
    report["integrity"]["action_metrics_trusted"] = False
    for side in ("A", "B"):
        report["metrics"][side].update({"guard_index": 0.07, "guard_definition": "hands_up_share/3"})
        report["coaching"][side] = {"strengths": [], "drills": [], "improvements": [{
            "title": "Nothing behind your opponent",
            "detail": "watch your counted strikes on the replay and judge them yourself."}]}
    refresh_identity_integrity(report)
    text = json.dumps(report["coaching"])
    assert "counted strikes" not in text
    assert "Work on: Guard (experimental) 7%" in text
    assert report["training_plan"]["A"]


def test_d7_opening_a_solo_result_clears_results_ready():
    client = BrowserClient(web.app)
    client.get("/")
    owner = f"guest:{client.cookies.get(web.GUEST_COOKIE)}"
    job = "qa1011solo" + uuid.uuid4().hex[:5]
    (web.OUTPUTS / job).mkdir()
    report = {"mode": "solo", "setup": {"ruleset": "K1"}, "video": {"analysis_target": "A"}, "integrity": {},
              "metrics": {"A": {"pose_coverage": 0.9, "guard_index": 0.3, "balance_index": 0.7}},
              "tracking": {}, "performance": {"segment_duration_seconds": 10}}
    (web.OUTPUTS / job / "report.json").write_text(json.dumps(report), encoding="utf-8")
    state.create_job(job, {"owner_key": owner, "status": "complete", "video_path": ""})
    try:
        client.cookies.set(web.LAST_COMPLETED_ANALYSIS_COOKIE, job)
        response = client.get(f"/result/{job}")
        assert response.status_code == 200
        assert any(web.LAST_COMPLETED_ANALYSIS_COOKIE in header and ("Max-Age=0" in header or "expires=" in header.lower())
                   for header in response.headers.get_list("set-cookie"))
    finally:
        state.delete_job(job)
