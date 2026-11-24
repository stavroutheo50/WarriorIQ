"""People are emailed when their report is ready, or when the analysis failed (owner, 2026-11-22).

Analyses run on the owner's own computer and a fight once waited 46 hours, so
the page can promise "we email you" only if the site actually does.
"""

from __future__ import annotations

import uuid
from unittest import mock

import pytest

import app.main as web
from core import db
from core.auth import hash_password


@pytest.fixture(autouse=True)
def _synchronous(monkeypatch):
    monkeypatch.setattr(web, "_EMAIL_IN_BACKGROUND", False)


def _account():
    email = f"outcome-{uuid.uuid4().hex[:10]}@example.test"
    return db.create_account(email, hash_password("Strong-Local-Password")), email


def test_a_ready_report_is_emailed_once_with_its_link():
    created, email = _account()
    job_id = f"ready{uuid.uuid4().hex[:8]}"
    job = {"owner_key": f"account:{created['id']}"}
    with mock.patch.object(web, "send_transactional_email", return_value=True) as send:
        web._email_analysis_outcome(job_id, job, ready=True)
        web._email_analysis_outcome(job_id, job, ready=True)          # a worker retry
    assert send.call_count == 1
    to, subject, body = send.call_args.args
    assert to == email and "report is ready" in subject and f"/result/{job_id}" in body


def test_a_failed_analysis_is_emailed_and_guests_never_are():
    created, email = _account()
    job_id = f"fail{uuid.uuid4().hex[:8]}"
    with mock.patch.object(web, "send_transactional_email", return_value=True) as send:
        web._journey_failed({"owner_key": f"account:{created['id']}"}, "decode_failed", job_id)
        web._journey_failed({"owner_key": "guest:abc"}, "decode_failed", f"g{job_id}")
    assert send.call_count == 1
    assert send.call_args.args[0] == email and "/history" in send.call_args.args[2]


def test_saving_a_finished_fight_sends_the_ready_email():
    created, _ = _account()
    job_id = f"save{uuid.uuid4().hex[:8]}"
    with mock.patch.object(web, "persist_completed_job"), \
            mock.patch.object(web, "send_transactional_email", return_value=True) as send:
        web._save_remote_fight(job_id, {"owner_key": f"account:{created['id']}", "analysis_run_id": "r1"}, {})
    assert send.call_count == 1


def test_a_mail_failure_never_breaks_saving_the_result():
    created, _ = _account()
    with mock.patch.object(web, "persist_completed_job"), \
            mock.patch.object(web, "send_transactional_email", side_effect=RuntimeError("smtp down")):
        web._save_remote_fight(f"x{uuid.uuid4().hex[:8]}", {"owner_key": f"account:{created['id']}"}, {})


# --- The first-report screen (shown at /camp until the first saved fight) -----

def _signed_in():
    from browser_client import BrowserClient
    from core.auth import issue_session

    created, _ = _account()
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    return created, client


def test_a_new_account_sees_the_four_steps_and_one_button():
    _, client = _signed_in()
    page = client.get("/camp").text
    assert "Let's get your first fight report." in page or "Let&#39;s get your first fight report." in page
    assert "Step 1 of 4" in page and 'href="/analyze"' in page
    assert "Rookie" not in page and "Share my camp" not in page


def test_an_uploaded_video_moves_it_to_picking_the_fighters_then_waiting():
    created, client = _signed_in()
    owner = f"account:{created['id']}"
    with mock.patch.object(web, "list_jobs", return_value=[("job1", {"owner_key": owner, "status": "selecting",
                                                                        "original_name": "spar.mp4"})]):
        page = client.get("/camp").text
    assert "Step 2 of 4" in page and 'href="/select/job1"' in page
    with mock.patch.object(web, "list_jobs", return_value=[("job1", {"owner_key": owner, "status": "queued"})]):
        page = client.get("/camp").text
    assert "Step 3 of 4" in page and 'href="/progress/job1"' in page


def test_anyone_past_the_first_report_or_coaching_keeps_fight_camp():
    created, client = _signed_in()
    db.save_fight(f"f{uuid.uuid4().hex[:8]}", int(created["profile_id"]), "f.mp4", "", "", "sparring", "K1", "A", {})
    assert "Step 1 of 4" not in client.get("/camp").text
    coach, coach_client = _signed_in()
    db.set_account_type(int(coach["profile_id"]), "coach")
    assert "Step 1 of 4" not in coach_client.get("/camp").text
    _, other = _signed_in()
    assert "Step 1 of 4" not in other.get("/camp?full=1").text


# --- One four-step bar from upload to report -------------------------------------

def test_the_four_step_bar_marks_the_current_step():
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(web.ROOT / "app" / "templates")), autoescape=True)
    template = env.get_template("_workflow_steps.html")
    for step in (1, 2, 3):
        html = template.render(step=step, step_note="")
        assert html.count('aria-current="step"') == 1 and html.count('class="complete"') == step - 1
        assert f"step {step} of 4" in html
    for name in ("analyze.html", "select.html", "frame.html", "progress.html"):
        assert '{% include "_workflow_steps.html" %}' in (web.ROOT / "app" / "templates" / name).read_text()


def test_a_queued_fight_says_it_will_be_emailed():
    assert "we email you when your report is ready" in web._deferred_analysis_message()


def test_the_ready_email_waits_for_a_confirmed_address():
    from core.config import SETTINGS

    previous = SETTINGS.require_email_verification
    object.__setattr__(SETTINGS, "require_email_verification", True)
    try:
        created, _ = _account()                                      # never confirmed
        with mock.patch.object(web, "send_transactional_email", return_value=True) as send:
            web._email_analysis_outcome(f"u{uuid.uuid4().hex[:8]}", {"owner_key": f"account:{created['id']}"}, ready=True)
        assert not send.called
        db.mark_email_verified(int(created["id"]))
        with mock.patch.object(web, "send_transactional_email", return_value=True) as send:
            web._email_analysis_outcome(f"v{uuid.uuid4().hex[:8]}", {"owner_key": f"account:{created['id']}"}, ready=True)
        assert send.called
    finally:
        object.__setattr__(SETTINGS, "require_email_verification", previous)


def test_the_ready_email_mentions_the_quick_check_but_not_to_a_minor():
    created, _ = _account()
    job = {"owner_key": f"account:{created['id']}"}
    with mock.patch.object(web, "send_transactional_email", return_value=True) as send:
        web._email_analysis_outcome(f"qc{uuid.uuid4().hex[:8]}", job, ready=True)
    assert "one tap each" in send.call_args.args[2]
    with mock.patch.object(web.social, "is_minor_account", return_value=True), \
            mock.patch.object(web, "send_transactional_email", return_value=True) as send:
        web._email_analysis_outcome(f"qc{uuid.uuid4().hex[:8]}", job, ready=True)
    assert "one tap each" not in send.call_args.args[2]
