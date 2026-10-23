"""The remote worker sends progress beside the analysis, not in its way (2026-10-23).

Measured on the owner's RTX 5060: 0.026 s of every analysed frame went to no
named step - a quarter of the frame pass - and every 40 frames the analysis
waited for warrioriq.eu to answer a progress update (worker._ProgressSender).
"""

from __future__ import annotations

import threading
import time

import pytest

import worker
from app.state import AnalysisRunLost
from core.worker_client import RemoteWorkerError


class _SlowClient:
    def __init__(self, delay=0.3, fail=None):
        self.delay, self.fail, self.sent = delay, fail, []

    def progress(self, job_id, run_id, patch):
        time.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        self.sent.append(dict(patch))


def test_an_update_does_not_wait_for_the_web_app():
    client = _SlowClient(delay=0.5)
    sender = worker._ProgressSender(client, "job", "run", threading.Event())
    started = time.perf_counter()
    for percent in range(10):
        sender({"percent": percent})
    assert time.perf_counter() - started < 0.1
    sender.close(timeout=5)
    # Merged while one was in flight; the last value always arrives.
    assert client.sent[-1]["percent"] == 9 and len(client.sent) < 10


def test_fields_from_merged_updates_are_all_kept():
    client = _SlowClient(delay=0.3)
    sender = worker._ProgressSender(client, "job", "run", threading.Event())
    sender({"percent": 1})
    sender({"message": "Analysing fight", "percent": 2})
    sender({"live_events": [1]})
    sender.close(timeout=5)
    merged = {}
    for patch in client.sent:
        merged.update(patch)
    assert merged == {"percent": 2, "message": "Analysing fight", "live_events": [1]}


def test_a_lost_run_stops_the_analysis_on_the_next_update():
    lost = threading.Event()
    sender = worker._ProgressSender(_SlowClient(delay=0.0, fail=AnalysisRunLost("gone")), "job", "run", lost)
    sender({"percent": 1})
    assert lost.wait(2)
    with pytest.raises(AnalysisRunLost):
        sender({"percent": 2})
    sender.close(timeout=1)


def test_a_failed_send_does_not_stop_the_analysis():
    client = _SlowClient(delay=0.0, fail=RemoteWorkerError("timeout"))
    sender = worker._ProgressSender(client, "job", "run", threading.Event())
    sender({"percent": 1})
    sender.close(timeout=2)
    sender({"percent": 2})          # still accepted
