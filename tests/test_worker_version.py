"""A worker older than the web app gets no work, and its results are refused.

QA, 2026-10-04: the GPU worker had been deployed from a checkout that
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
                                      "commit": "abc123def456",
                                      "fingerprint": build_info.analysis_fingerprint()}
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
                               json={"worker_id": "gpu-1", "analysis_version": build_info.ANALYSIS_VERSION,
                                     "analysis_fingerprint": build_info.analysis_fingerprint()})
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
    assert sent[0]["analysis_fingerprint"] == build_info.analysis_fingerprint()


def test_report_page_prints_the_build(tmp_path):
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("app/templates"))
    source = env.loader.get_source(env, "result.html")[0]
    assert "data-analysis-build" in source
    assert "analysis_build.commit" in source and "analysis_build.outdated" in source


def test_a_worker_running_other_analysis_code_is_refused(monkeypatch, tmp_path):
    """2026-10-13: a worker run from a stale copy passed the version check for days."""
    web, client = _client()
    web.app.dependency_overrides[web._require_remote_worker] = lambda: None
    monkeypatch.setattr(web, "_require_remote_worker", lambda request: None)
    claimed = []
    monkeypatch.setattr(web, "claim_next_job", lambda worker_id: claimed.append(worker_id) or None)
    import app.state as state

    monkeypatch.setattr(state, "OUTPUTS", tmp_path)
    try:
        for fingerprint in (None, "0000000000000000"):
            body = {"worker_id": "gpu-1", "analysis_version": build_info.ANALYSIS_VERSION}
            if fingerprint:
                body["analysis_fingerprint"] = fingerprint
            response = client.post("/api/worker/claim", json=body).json()
            assert response["job"] is None and response["refused"] == "worker_code_differs"
            assert response["required_fingerprint"] == build_info.analysis_fingerprint()
        assert claimed == []
        assert state.recent_worker_refusal()["reason"] == "worker_code_differs"
        client.post("/api/worker/claim", json={"worker_id": "gpu-1", "analysis_version": build_info.ANALYSIS_VERSION,
                                               "analysis_fingerprint": build_info.analysis_fingerprint()})
        assert claimed == ["gpu-1"]
        assert state.recent_worker_refusal() is None
    finally:
        web.app.dependency_overrides.clear()


def test_the_fingerprint_ignores_line_endings(tmp_path, monkeypatch):
    core = tmp_path / "core"
    core.mkdir()
    (core / "a.py").write_bytes(b"x = 1\ny = 2\n")
    monkeypatch.setattr(build_info, "ROOT", tmp_path)
    build_info.analysis_fingerprint.cache_clear()
    try:
        unix = build_info.analysis_fingerprint()
        build_info.analysis_fingerprint.cache_clear()
        (core / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
        assert build_info.analysis_fingerprint() == unix
        build_info.analysis_fingerprint.cache_clear()
        (core / "a.py").write_bytes(b"x = 2\ny = 2\n")
        assert build_info.analysis_fingerprint() != unix
    finally:
        build_info.analysis_fingerprint.cache_clear()


def test_the_worker_says_how_to_fix_a_code_mismatch():
    from core.worker_client import RemoteWorkerClient, RemoteWorkerError

    client = RemoteWorkerClient("https://example.invalid", "token", "gpu-1")
    with patch.object(client, "_request", return_value={"job": None, "refused": "worker_code_differs",
                                                       "required_fingerprint": "abc"}):
        with pytest.raises(RemoteWorkerError, match="same version from GitHub"):
            client.claim()


def test_a_refusal_names_the_files_and_strays_are_no_reason_to_refuse(monkeypatch, tmp_path):
    """2026-10-14: a refused worker sat refused for twenty minutes and nothing said which file differed."""
    web, client = _client()
    web.app.dependency_overrides[web._require_remote_worker] = lambda: None
    monkeypatch.setattr(web, "_require_remote_worker", lambda request: None)
    claimed = []
    monkeypatch.setattr(web, "claim_next_job", lambda worker_id: claimed.append(worker_id) or None)
    import app.state as state

    monkeypatch.setattr(state, "OUTPUTS", tmp_path)
    own = build_info.analysis_file_hashes()
    first = sorted(own)[0]
    body = {"worker_id": "gpu-1", "analysis_version": build_info.ANALYSIS_VERSION,
            "analysis_fingerprint": "0000000000000000"}
    try:
        changed = dict(own, **{first: "ffffffffffffffff"})
        response = client.post("/api/worker/claim", json=dict(body, analysis_files=changed)).json()
        assert response["refused"] == "worker_code_differs"
        assert response["differences"] == {"changed": [first], "missing_on_worker": [], "extra_on_worker": []}

        missing = {name: digest for name, digest in own.items() if name != first}
        response = client.post("/api/worker/claim", json=dict(body, analysis_files=missing)).json()
        assert response["differences"]["missing_on_worker"] == [first]
        assert claimed == []

        strays = dict(own, **{"analyzer_backup.py": "1234"})
        response = client.post("/api/worker/claim", json=dict(body, analysis_files=strays)).json()
        assert "refused" not in response and claimed == ["gpu-1"]
    finally:
        web.app.dependency_overrides.clear()


def test_the_worker_window_names_the_files_that_differ():
    from core.worker_client import RemoteWorkerClient, RemoteWorkerError

    client = RemoteWorkerClient("https://example.invalid", "token", "gpu-1")
    sent = []
    refusal = {"job": None, "refused": "worker_code_differs", "required_fingerprint": "abc",
               "differences": {"changed": ["analyzer.py"], "missing_on_worker": ["slow_watch.py"],
                               "extra_on_worker": []}}
    with patch.object(client, "_request", side_effect=lambda method, path, body: sent.append(body) or refusal):
        with pytest.raises(RemoteWorkerError, match="different here: analyzer.py; missing here: slow_watch.py"):
            client.claim()
    assert sent[0]["analysis_files"] == build_info.analysis_file_hashes()


def test_health_shows_the_analysis_fingerprint_and_deploy_replaces_core_whole():
    """2026-10-14: refused workers could not be told apart from a website keeping a stale core file."""
    from pathlib import Path

    web, client = _client()
    assert client.get("/health").json()["fingerprint"] == build_info.analysis_fingerprint()
    deploy = (Path(__file__).resolve().parents[1] / ".cpanel.yml").read_text(encoding="utf-8")
    assert "/bin/rm -rf $DEPLOYPATH/core\n" in deploy
    assert deploy.index("cp -R core $DEPLOYPATH/core.new") < deploy.index("mv $DEPLOYPATH/core.new $DEPLOYPATH/core")
