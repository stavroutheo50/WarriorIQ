"""A worker older than the web app gets no work, and its results are refused.

QA, 2026-10-04: the GPU worker on Modal had been deployed from a checkout that
predated the whole-video fix, so new reports still started at the selection
frame while the web app's code said they could not. Every result now carries
core/build_info.stamp(), and the server compares it with its own.
"""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import patch

import pytest

from core import build_info


def test_stamp_carries_version_and_commit(monkeypatch):
    build_info.build_commit.cache_clear()
    monkeypatch.setenv("WARRIORIQ_BUILD_COMMIT", "abc123def4567890")
    try:
        assert build_info.stamp() == {"analysis_version": build_info.ANALYSIS_VERSION,
                                      "commit": "abc123def456"}
    finally:
        build_info.build_commit.cache_clear()


def test_result_check_flags_missing_and_older_stamps():
    assert build_info.result_check({})["outdated"] is True
    old = {"analysis_build": {"analysis_version": build_info.ANALYSIS_VERSION - 1, "commit": "x"}}
    assert build_info.result_check(old)["outdated"] is True
    current = {"analysis_build": build_info.stamp()}
    assert build_info.result_check(current)["outdated"] is False


def _client():
    from fastapi.testclient import TestClient

    import app.main as web

    return web, TestClient(web.app)


def test_claim_from_an_outdated_worker_is_refused(monkeypatch):
    web, client = _client()
    web.app.dependency_overrides[web._require_remote_worker] = lambda: None
    monkeypatch.setattr(web, "_require_remote_worker", lambda request: None)
    claimed = []
    monkeypatch.setattr(web, "claim_next_job", lambda worker_id: claimed.append(worker_id) or None)
    try:
        for body in ({"worker_id": "gpu-1"},
                     {"worker_id": "gpu-1", "analysis_version": build_info.ANALYSIS_VERSION - 1}):
            response = client.post("/api/worker/claim", json=body)
            assert response.status_code == 200
            assert response.json()["job"] is None
            assert response.json()["refused"] == "worker_outdated"
        assert claimed == []
        response = client.post("/api/worker/claim",
                               json={"worker_id": "gpu-1", "analysis_version": build_info.ANALYSIS_VERSION})
        assert response.json() == {"job": None}
        assert claimed == ["gpu-1"]
    finally:
        web.app.dependency_overrides.clear()


def test_worker_client_sends_its_version_and_reports_a_refusal():
    from core.worker_client import RemoteWorkerClient, RemoteWorkerError

    client = RemoteWorkerClient("https://example.invalid", "token", "gpu-1")
    sent = []

    def fake(method, path, payload=None, **kwargs):
        sent.append(payload)
        return {"job": None, "refused": "worker_outdated", "required_analysis_version": 99}

    with patch.object(client, "_request", side_effect=fake):
        with pytest.raises(RemoteWorkerError, match="Redeploy the worker"):
            client.claim()
    assert sent[0]["analysis_version"] == build_info.ANALYSIS_VERSION


def test_report_page_prints_the_build(tmp_path):
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("app/templates"))
    source = env.loader.get_source(env, "result.html")[0]
    assert "data-analysis-build" in source
    assert "analysis_build.commit" in source and "analysis_build.outdated" in source
