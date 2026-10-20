"""A refused worker installs the website's analysis code itself (2026-10-19).

Every website deploy that touched core/ paused every analysis until
update-worker was run on the GPU PC. The web app now serves the code it
compares against, and a refused worker installs it and restarts on it.
"""

from __future__ import annotations

import hashlib

import pytest
from browser_client import BrowserClient

import app.main as web
import worker
from core import build_info


def _fingerprint(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + files[name].encode().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


class _Client:
    def __init__(self, served: dict):
        self.served = served

    def analysis_code(self) -> dict:
        return self.served


@pytest.fixture
def worker_folder(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    (core / "scoring.py").write_text("OLD = 1\n")
    (core / "stray.py").write_text("STRAY = 1\n")
    (core / "notes.txt").write_text("kept")
    return tmp_path


def test_the_web_app_serves_exactly_the_code_it_compares_against(monkeypatch):
    monkeypatch.setitem(web.app.dependency_overrides, web._require_remote_worker, lambda: None)
    monkeypatch.setattr(web, "_require_remote_worker", lambda request: None)
    response = BrowserClient(web.app).get("/api/worker/analysis-code")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fingerprint"] == build_info.analysis_fingerprint()
    assert set(body["files"]) == {path.name for path in (build_info.ROOT / "core").glob("*.py")}
    assert _fingerprint(body["files"]) == body["fingerprint"]


def test_the_code_needs_the_worker_token():
    response = BrowserClient(web.app).get("/api/worker/analysis-code")
    assert response.status_code in {401, 503}
    assert "files" not in response.text


def test_a_refused_worker_installs_the_websites_code(worker_folder):
    files = {"scoring.py": "NEW = 2\n", "contact.py": "ADDED = 3\n"}
    assert worker._sync_analysis_code(_Client({"files": files, "fingerprint": _fingerprint(files)}),
                                      worker_folder)
    core = worker_folder / "core"
    assert {path.name: path.read_text() for path in core.glob("*.py")} == files
    assert (core / "notes.txt").exists()
    assert (worker_folder / "core_previous" / "scoring.py").read_text() == "OLD = 1\n"
    assert (worker_folder / "core_previous" / "stray.py").exists()


@pytest.mark.parametrize("served", [
    {"files": {"scoring.py": "NEW = 2\n"}, "fingerprint": "0" * 16},          # corrupted download
    {"files": {"../app.py": "x"}, "fingerprint": _fingerprint({"../app.py": "x"})},  # unsafe name
    {"files": {}, "fingerprint": _fingerprint({})},
])
def test_a_bad_download_changes_nothing(worker_folder, served):
    assert not worker._sync_analysis_code(_Client(served), worker_folder)
    assert (worker_folder / "core" / "scoring.py").read_text() == "OLD = 1\n"
    assert not (worker_folder / "core_previous").exists()


def test_it_can_be_turned_off(worker_folder, monkeypatch):
    monkeypatch.setenv("WARRIORIQ_WORKER_AUTO_UPDATE", "0")
    files = {"scoring.py": "NEW = 2\n"}
    assert not worker._sync_analysis_code(_Client({"files": files, "fingerprint": _fingerprint(files)}),
                                          worker_folder)
    assert (worker_folder / "core" / "scoring.py").read_text() == "OLD = 1\n"


def test_the_loop_syncs_on_a_refusal_and_then_restarts(monkeypatch):
    synced = []

    class Refusing:
        def __init__(self, *args):
            pass

        def claim(self):
            raise worker.WorkerRefused("refused", "worker_code_differs")

    monkeypatch.setattr(worker, "RemoteWorkerClient", Refusing)
    monkeypatch.setattr(worker, "retry_heartbeat", lambda client: None)
    monkeypatch.setattr(worker, "refresh_worker_lock", lambda: None)
    monkeypatch.setattr(worker, "_sync_analysis_code", lambda client: synced.append(client) or True)
    assert worker.run_remote_worker("gpu-1", once=True) == 1
    assert len(synced) == 1


def test_a_refusal_is_its_own_error_type():
    # Imported here: other tests reload core.worker_client.
    from core import worker_client

    client = worker_client.RemoteWorkerClient("http://x", "t", "gpu-1")
    client._request = lambda *a, **k: {"refused": "worker_code_differs", "required_fingerprint": "f"}
    with pytest.raises(worker_client.WorkerRefused) as refused:
        client.claim()
    assert refused.value.reason == "worker_code_differs"
