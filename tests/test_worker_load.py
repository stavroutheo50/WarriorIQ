"""The analysis worker must be light on the shared web host.

OuiHeberg, 2026-10: the worker asked for work every second (a heartbeat and a
claim, so two requests a second, all day) and reported progress every one to
three seconds. From the analysis PC's address, together with the owner's own
browsing, that tripped the host's firewall. They asked for 5-10 s, or a backoff
while the queue is empty.
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path
from unittest.mock import patch

import worker as worker_module
from core.config import SETTINGS


def test_an_idle_worker_backs_off_and_sends_one_request_per_poll():
    calls = []
    sleeps = []

    class Client:
        def heartbeat(self):
            calls.append("heartbeat")

        def claim(self):
            calls.append("claim")
            if len(calls) >= 8:
                raise SystemExit(0)
            return None

    settings = dataclasses.replace(SETTINGS, worker_poll_seconds=5.0, worker_idle_poll_max_seconds=30.0)
    with patch.object(worker_module, "SETTINGS", settings), \
         patch.object(worker_module, "RemoteWorkerClient", lambda *args: Client()), \
         patch.object(worker_module, "_code_changed_since", lambda code: False), \
         patch.object(worker_module, "refresh_worker_lock", lambda: None), \
         patch.object(worker_module.time, "sleep", sleeps.append):
        try:
            worker_module.run_remote_worker("test-worker")
        except SystemExit:
            pass
    assert "heartbeat" not in calls          # the claim records the heartbeat
    assert sleeps[0] == 5.0
    assert sleeps == sorted(sleeps) and max(sleeps) == 30.0
    # About two requests a minute once settled, against 120 before.
    assert 60.0 / max(sleeps) <= 2.0


def test_progress_is_sent_at_most_every_few_seconds_but_stages_at_once():
    sent = []
    clock = {"now": 100.0}

    class Client:
        def progress(self, job_id, run_id, patch):
            sent.append(patch.get("stage"))

        def download_video(self, job, destination):
            Path(destination).write_bytes(b"x")

        def complete(self, *args):
            pass

    def fake_analyze(request, progress):
        for stage, step in [("tracking", 0.5)] * 3 + [("analysis", 1.0)] * 12 + [("report", 0.1)]:
            clock["now"] += step
            progress({"stage": stage})

    settings = dataclasses.replace(SETTINGS, worker_progress_seconds=5.0, worker_lease_seconds=3600)
    with patch.object(worker_module, "SETTINGS", settings), \
         patch("core.analyzer.analyze", fake_analyze), \
         patch.object(worker_module, "_worker_result_archive", lambda *args: None), \
         patch.object(worker_module, "_remote_request", lambda job, path: None), \
         patch.object(worker_module.time, "monotonic", lambda: clock["now"]):
        worker_module.run_remote_claimed_job(Client(), {"job_id": "loadtest01", "analysis_run_id": uuid.uuid4().hex})
    # Every stage change goes out; within "analysis" (12 s) only every 5 s.
    assert sent[0] == "tracking" and sent[-1] == "report"
    assert sent.count("tracking") == 1
    assert sent.count("analysis") == 3


def test_the_live_page_does_not_poll_faster_than_the_worker_reports():
    page = (Path(__file__).resolve().parents[1] / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
    assert "retryDelay=900" not in page
    assert "retryDelay=2500" in page
