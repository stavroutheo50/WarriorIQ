"""Audit of 14 Oct 2026, phase 1: analysis speed, live progress, time left, worker versions.

#1  the Modal worker keeps one warm container on an A10/L4 and batches pose frames
#21 the live percentage, stage label and "analysed through" line describe one stage
#22 time left is estimated once a pass has measured about 50 frames
#23 a refused worker pauses the analysis honestly and alerts the owner
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- #21
def test_identity_pass_percentage_cursor_and_label_agree():
    from core.progress_view import stage_view

    # The audited state: 24.2% of the whole job, inside the identity pass.
    job = {"status": "running", "stage": "tracking", "percent": 24.2, "processed_video_seconds": 0.0,
           "stage_fraction": 0.67, "tracked_video_seconds": 16.1}
    view = stage_view(job, 24.0)
    assert view["label"] == "Following both fighters"
    assert view["percent"] == pytest.approx(67.0)
    assert view["cursor"] == "Fighters followed through 0:16 of 0:24"
    assert (view["step"], view["steps"]) == (2, 4)
    # Never the movement-pass cursor while that pass has not started.
    assert "analysed through" not in view["cursor"]


def test_an_older_worker_without_stage_fraction_still_reads_consistently():
    from core.progress_view import stage_view

    view = stage_view({"status": "running", "stage": "tracking", "percent": 24.2}, 24.0)
    # (24.2 - 2) / 33 of the identity pass, and the cursor from the same fraction.
    assert view["percent"] == pytest.approx(67.3, abs=0.1)
    assert view["cursor"] == "Fighters followed through 0:16 of 0:24"


def test_movement_pass_percentage_is_the_cursor_over_the_duration():
    from core.progress_view import stage_view

    view = stage_view({"status": "running", "stage": "analysis", "percent": 60.0,
                       "processed_video_seconds": 12.0}, 24.0)
    assert view["percent"] == pytest.approx(50.0)
    assert view["cursor"] == "Video analysed through 0:12 of 0:24"
    assert view["label"] == "Reading movement and strikes"


def test_paused_view_has_no_percentage():
    from core.progress_view import stage_view

    view = stage_view({"status": "queued"}, 24.0, paused=True)
    assert view["label"] == "Paused for an update" and view["percent"] is None


def test_live_page_renders_the_stage_view_not_the_job_percentage():
    template = (ROOT / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
    assert "Of the whole job" not in template
    assert "$('analysisCursor').textContent=view.cursor" in template
    assert "$('stageCaption').textContent=`Step ${view.step} of ${view.steps} · ${view.label}`" in template
    assert "Locking Fighter A and Fighter B" not in template


# --------------------------------------------------------------------- #22
def test_time_left_appears_after_fifty_frames_and_counts_down():
    from core.progress_eta import StageEta

    eta = StageEta(fps=25.0)
    assert eta.update("tracking", 0.0, 24.0, now=0.0) is None
    # 1 s of video = 25 frames: not yet measured.
    assert eta.update("tracking", 1.0, 24.0, now=2.0) is None
    # 2.4 s = 60 frames covered in 4.8 s: 0.5 video-seconds per second.
    first = eta.update("tracking", 2.4, 24.0, now=4.8)
    assert first == pytest.approx((24.0 - 2.4) / 0.5)
    later = eta.update("tracking", 12.0, 24.0, now=24.0)
    assert later is not None and later < first


def test_time_left_restarts_measuring_when_the_pass_changes():
    from core.progress_eta import StageEta

    eta = StageEta(fps=30.0)
    eta.update("tracking", 0.0, 10.0, now=0.0)
    assert eta.update("tracking", 5.0, 10.0, now=5.0) is not None
    # A new pass runs at its own speed; nothing is borrowed from the last one.
    assert eta.update("analysis", 0.0, 10.0, now=6.0) is None


def test_the_identity_pass_reports_time_left_for_this_step(monkeypatch):
    """The analyzer's progress for the SAM2 pass carries an ETA scoped to the step."""
    source = (ROOT / "core" / "analyzer.py").read_text(encoding="utf-8")
    assert 'stage_eta.update("tracking", tracked_seconds, segment_duration, elapsed)' in source
    assert 'eta_scope = "stage"' in source
    template = (ROOT / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
    assert "working it out from the first frames" not in template
    assert "left in this step" in template


# --------------------------------------------------------------------- #23
def test_release_version_is_the_analysis_version_and_fingerprint():
    from core.build_info import ANALYSIS_VERSION, analysis_fingerprint, release_version

    assert release_version() == f"{ANALYSIS_VERSION}.{analysis_fingerprint()[:12]}"


def test_health_reports_the_release(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as web
    from core.build_info import release_version

    with TestClient(web.app) as client:
        assert client.get("/health").json()["release"] == release_version()


def test_a_refused_worker_pauses_the_job_and_stops_the_timer(monkeypatch):
    import app.main as web

    monkeypatch.setattr(web, "recent_worker_refusal", lambda: {"reason": "worker_code_differs"})
    payload = web._public_job_status("job1", {"status": "queued", "video_duration": 24.0, "queued_at_epoch": 1.0})
    assert payload["paused_for_update"] is True
    assert payload["warming_up"] is False
    assert payload["worker_refused"].startswith("Analysis is paused for an update")
    assert payload["stage_view"]["label"] == "Paused for an update"
    template = (ROOT / "app" / "templates" / "progress.html").read_text(encoding="utf-8")
    assert "if(d.paused_for_update){$('timing').textContent='Paused for an update · no timer is running" in template


def test_version_mismatch_alerts_once_and_flags_the_admin_page(tmp_path):
    from core import worker_alerts

    sent = []
    send = lambda to, subject, body: sent.append((to, subject)) or True   # noqa: E731
    first = worker_alerts.note_version_mismatch(outputs=tmp_path, recipients=["owner@example.com"], send=send,
                                                worker_id="gpu-1", reason="worker_code_differs", now=1000.0)
    again = worker_alerts.note_version_mismatch(outputs=tmp_path, recipients=["owner@example.com"], send=send,
                                                worker_id="gpu-1", reason="worker_code_differs", now=1060.0)
    assert first is True and again is False
    flag = worker_alerts.version_mismatch_flag(tmp_path)
    assert flag["worker_id"] == "gpu-1" and flag["reason"] == "worker_code_differs"
    worker_alerts.clear_version_mismatch(tmp_path)
    assert worker_alerts.version_mismatch_flag(tmp_path) is None


def test_claim_from_a_mismatched_worker_raises_the_alert(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import app.main as web
    from app import state

    alerts = []
    original = web._require_remote_worker
    web.app.dependency_overrides[original] = lambda: None
    monkeypatch.setattr(web, "_require_remote_worker", lambda request: None)
    monkeypatch.setattr(web, "OUTPUTS", tmp_path)
    monkeypatch.setattr(state, "OUTPUTS", tmp_path)
    monkeypatch.setattr(web.worker_alerts, "note_version_mismatch", lambda **kw: alerts.append(kw["reason"]))
    try:
        response = TestClient(web.app).post("/api/worker/claim", json={"worker_id": "gpu-1", "analysis_version": 1})
    finally:
        web.app.dependency_overrides.pop(original, None)
    assert response.json().get("refused") == "worker_outdated", response.text
    assert alerts == ["worker_outdated"]


# ---------------------------------------------------------------------- #1
def test_modal_worker_keeps_one_warm_container_on_a_supported_gpu():
    source = (ROOT / "deploy" / "modal_worker.py").read_text(encoding="utf-8")
    assert 'ALLOWED_GPUS = ("A10", "L4")' in source
    assert 'MIN_CONTAINERS = int(os.environ.get("WARRIORIQ_MODAL_MIN_CONTAINERS", "1"))' in source
    assert "min_containers=MIN_CONTAINERS" in source
    assert "@modal.enter()" in source and "_warm_both_sizes()" in source
    assert 'gpu="T4"' not in source


def test_engine_name_carries_the_batch_it_accepts(monkeypatch):
    from core import trt_engine
    from core.config import SETTINGS

    previous = SETTINGS.pose_batch_size
    object.__setattr__(SETTINGS, "pose_batch_size", 8)
    try:
        assert trt_engine.engine_path_for("/m", "NVIDIA A10").name.endswith("_b8.engine")
        object.__setattr__(SETTINGS, "pose_batch_size", 1)
        assert not trt_engine.engine_path_for("/m", "NVIDIA A10").name.endswith("_b1.engine")
    finally:
        object.__setattr__(SETTINGS, "pose_batch_size", previous)


class _FakeBoxes:
    def __init__(self, value):
        self.data = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(
            numpy=lambda: np.full((1, 6), value, dtype=np.float32))))


class _FakeModel:
    def __init__(self, fail_batches=False):
        self.calls = []
        self.callbacks = {}
        self.fail_batches = fail_batches

    def predict(self, source, **kwargs):
        frames = source if isinstance(source, list) else [source]
        self.calls.append(len(frames))
        if self.fail_batches and len(frames) > 1:
            raise RuntimeError("engine built for batch 1")
        return [SimpleNamespace(boxes=_FakeBoxes(float(frame[0, 0, 0])), keypoints=None, path="", names={0: "person"})
                for frame in frames]


def _tracker(model):
    from core.pose_tracker import PoseTracker

    tracker = PoseTracker.__new__(PoseTracker)
    tracker.model = model
    tracker.model_path = "yolo26m-pose.pt"
    tracker.device = "cpu"
    tracker._prepared = {}
    return tracker


def test_batched_preparation_keeps_what_one_by_one_would_keep():
    frames = [(index, np.full((4, 4, 3), index, dtype=np.uint8)) for index in (10, 12, 14)]
    batched, single = _tracker(_FakeModel()), _tracker(_FakeModel())
    batched.prepare_detections_batch(frames, 640)
    for index, frame in frames:
        single.prepare_detections(frame, index, 640)
    assert batched.model.calls == [3] and single.model.calls == [1, 1, 1]
    assert batched._prepared.keys() == single._prepared.keys()
    for key in single._prepared:
        assert np.array_equal(batched._prepared[key][0], single._prepared[key][0])


def test_an_engine_that_refuses_a_batch_falls_back_frame_by_frame():
    frames = [(index, np.full((4, 4, 3), index, dtype=np.uint8)) for index in (1, 2)]
    tracker = _tracker(_FakeModel(fail_batches=True))
    tracker.prepare_detections_batch(frames, 640)
    assert tracker.model.calls == [2, 1, 1]
    assert set(tracker._prepared) == {(1, 640), (2, 640)}


def test_benchmark_prints_the_realtime_ratio():
    source = (ROOT / "tools" / "benchmark_realtime.py").read_text(encoding="utf-8")
    assert '"ratio": round(wall / max(1e-6, analysed), 3)' in source
    assert "[24.0, 60.0, 180.0]" in source


def test_every_analysis_logs_its_stage_seconds():
    source = (ROOT / "core" / "analyzer.py").read_text(encoding="utf-8")
    assert 'LOGGER.info("analysis_stage_seconds job=%s' in source
