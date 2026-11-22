"""The owner hears about fights waiting on an absent analysis PC, once."""

from unittest import mock

from core import worker_alerts


def _run(job, tmp_path, online=False, now=2000.0, recipients=("owner@example.com",)):
    sent = []
    with mock.patch.object(worker_alerts, "_send_in_background",
                           side_effect=lambda send, to, subject, body: sent.append((tuple(to), subject))):
        result = worker_alerts.check_waiting_fight(
            job, worker_online=online, outputs=tmp_path, recipients=recipients,
            threshold_seconds=600, send=None, now=now)
    return result, sent


QUEUED = {"status": "queued", "wake_requested_at_epoch": 1000.0}


def test_a_long_wait_with_no_worker_emails_the_owner(tmp_path):
    sent_flag, sent = _run(QUEUED, tmp_path)
    assert sent_flag and sent[0][0] == ("owner@example.com",)
    assert "not connected" in sent[0][1]


def test_no_alert_while_a_worker_is_connected_or_before_the_threshold(tmp_path):
    assert _run(QUEUED, tmp_path, online=True)[0] is False
    assert _run(QUEUED, tmp_path, now=1300.0)[0] is False                  # waited 5 minutes
    assert _run({"status": "running", "wake_requested_at_epoch": 1000.0}, tmp_path)[0] is False
    assert _run(QUEUED, tmp_path, recipients=())[0] is False


def test_one_email_per_outage_not_one_per_poll(tmp_path):
    assert _run(QUEUED, tmp_path, now=2000.0)[0] is True
    assert _run(QUEUED, tmp_path, now=2010.0)[0] is False                  # next poll
    assert _run(QUEUED, tmp_path, now=2000.0 + worker_alerts.REPEAT_SECONDS + 1)[0] is True


def test_the_owner_hears_when_it_is_back_and_only_after_an_alert(tmp_path):
    sent = []
    capture = mock.patch.object(worker_alerts, "_send_in_background",
                                side_effect=lambda send, to, subject, body: sent.append(subject))
    with capture:
        assert worker_alerts.note_worker_connected(outputs=tmp_path, recipients=("o@e.com",), send=None) is False
    _run(QUEUED, tmp_path)
    with capture:
        assert worker_alerts.note_worker_connected(outputs=tmp_path, recipients=("o@e.com",), send=None) is True
        assert worker_alerts.note_worker_connected(outputs=tmp_path, recipients=("o@e.com",), send=None) is False
    assert sent == ["WarriorIQ: the analysis PC is back"]


def test_the_background_sweep_alerts_even_when_nobody_keeps_the_page_open(tmp_path, monkeypatch):
    """2026-11-22: the page now says it can be closed, so the alert cannot rely on it."""
    import time as _time
    from unittest import mock

    import app.main as web
    from core.config import SETTINGS

    waiting = {"status": "queued", "wake_requested_at_epoch": _time.time() - 3600}
    previous = SETTINGS.admin_emails
    object.__setattr__(SETTINGS, "admin_emails", ("owner@example.test",))
    sent = []
    try:
        monkeypatch.setattr(web, "OUTPUTS", tmp_path)
        monkeypatch.setattr(web, "list_jobs", lambda: [("waiting1", waiting), ("waiting2", dict(waiting))])
        monkeypatch.setattr(web, "_worker_connected", lambda: False)
        monkeypatch.setattr(web.worker_alerts, "_send_in_background",
                            lambda send, recipients, subject, body: sent.append(subject))
        with mock.patch.object(web, "_warm_plausibility_verdicts", return_value=0):
            web._run_periodic_maintenance(now=0.0)
        assert len(sent) == 1 and "analysis PC is not connected" in sent[0]
    finally:
        object.__setattr__(SETTINGS, "admin_emails", previous)
