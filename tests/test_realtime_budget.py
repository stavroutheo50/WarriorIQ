"""An analysis must not take longer than the video it analyses.

QA, 2026-10-04: 0:22 for a 0:07 clip, 0:30 for 0:29, 0:10 for 0:01. Two fixed
costs were found and cut:

* the preflight probe ran eight frames at imgsz 1600 before the first frame of
  the fight (9.2 s of a 22.6 s CPU run on a 4-second clip); it now looks at
  640 first and climbs only when it has to;
* the GPU worker image had no TensorRT, so the pose engine was never built and
  every cold start paid an ONNX export that then failed (deploy/modal_worker.py).

The end-to-end check against a reference clip needs a CUDA device - the budget
is a GPU promise - and is skipped without one. The probe checks run anywhere.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class _Boxes:
    def __init__(self, boxes):
        self.xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.asarray(boxes, dtype=np.float32)))
        self._n = len(boxes)

    def __len__(self):
        return self._n


class _FakeModel:
    """Finds people whose size depends on the inference size, like the real one."""

    def __init__(self, people_at):
        self.sizes = []
        self.people_at = people_at

    def predict(self, frame, imgsz, **kwargs):
        self.sizes.append(imgsz)
        return [SimpleNamespace(boxes=_Boxes(self.people_at(imgsz)), keypoints=None)]


def _video(path, seconds=6, fps=10, size=(1280, 720)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(seconds * fps):
        frame = np.full((size[1], size[0], 3), 60 + index % 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_big_fighters_are_measured_at_640_only(tmp_path):
    from core import preflight

    two_big = [[100, 50, 400, 700], [700, 60, 1000, 700]]
    model = _FakeModel(lambda size: two_big)
    report = preflight.probe(str(_video(tmp_path / "v.mp4")), model)
    assert report.measured
    assert set(model.sizes) == {640}
    assert report.recommended_inference_size == 640


def test_small_fighters_still_get_the_larger_probe(tmp_path):
    from core import preflight

    # Tiny people at 640 imply a larger size: the probe must climb to 1600.
    small = [[100, 300, 120, 360], [700, 300, 720, 360]]
    model = _FakeModel(lambda size: small)
    report = preflight.probe(str(_video(tmp_path / "v.mp4")), model)
    assert 1600 in model.sizes
    assert report.recommended_inference_size > 640


def test_one_person_found_at_640_is_not_enough(tmp_path):
    from core import preflight

    # A tall spectator found at 640 with the fighters missed must not settle it.
    model = _FakeModel(lambda size: [[0, 0, 300, 700]] if size == 640 else
                       [[0, 0, 300, 700], [600, 300, 630, 380], [800, 300, 830, 380]])
    preflight.probe(str(_video(tmp_path / "v.mp4")), model)
    assert 1600 in model.sizes


def test_a_size_the_engine_cannot_serve_keeps_the_smaller_measurement(tmp_path):
    from core import preflight

    def people(size):
        if size > 1920:
            raise RuntimeError("input larger than the engine profile")
        return [[100, 300, 200, 420]]

    model = _FakeModel(people)
    report = preflight.probe(str(_video(tmp_path / "v.mp4")), model)
    assert report.measured


def test_a_failed_engine_build_is_not_retried_on_every_cold_start(tmp_path):
    from core import trt_engine

    with mock.patch("torch.cuda.is_available", return_value=True), \
            mock.patch.object(trt_engine, "gpu_slug", return_value="nvidia-a10"), \
            mock.patch.object(trt_engine, "build_pose_engine", side_effect=ImportError("tensorrt")) as build:
        assert trt_engine.ensure_pose_engine(tmp_path, retry_failed_after_seconds=3600) is None
        assert trt_engine.ensure_pose_engine(tmp_path, retry_failed_after_seconds=3600) is None
        assert build.call_count == 1
        # A local worker (no backoff) keeps trying, as it always has.
        trt_engine.ensure_pose_engine(tmp_path)
        assert build.call_count == 2


def test_the_gpu_image_carries_tensorrt_and_never_installs_at_run_time():
    source = (ROOT / "deploy" / "modal_worker.py").read_text(encoding="utf-8")
    assert 'pip_install_from_requirements("requirements-trt-cuda12.txt")' in source
    assert '"YOLO_AUTOINSTALL": "false"' in source
    assert "retry_failed_after_seconds=24 * 3600" in source
    requirements = (ROOT / "requirements-trt-cuda12.txt").read_text(encoding="utf-8")
    assert "tensorrt-cu12==10." in requirements and "onnxslim==" in requirements


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:                                               # noqa: BLE001
        return False


def reference_clip(path: Path, seconds: int = 8, fps: int = 25) -> Path:
    """Two real people (Ultralytics' zidane.jpg), gently panning, as video."""
    import ultralytics

    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(seconds * fps):
        shift = np.float32([[1, 0, int(10 * np.sin(index / 6))], [0, 1, 0]])
        writer.write(cv2.warpAffine(image, shift, (width, height), borderMode=cv2.BORDER_REFLECT))
    writer.release()
    return path


@pytest.mark.skipif(not _cuda(), reason="the within-video-length budget is a GPU promise; needs CUDA")
def test_the_reference_clip_is_analysed_within_its_own_length(tmp_path):
    from core import analyzer
    from core.types import AnalysisRequest

    video = reference_clip(tmp_path / "reference.mp4")
    seconds = 8.0
    # Boxes around the two people in zidane.jpg, drawn on the selection frame
    # half-way in, so the backward pass runs too.
    req = AnalysisRequest(video_path=str(video), fighter_a_box=[120, 200, 1050, 715],
                          fighter_b_box=[740, 45, 1140, 715], start_seconds=0.0,
                          selection_seconds=4.0, round_count=1, round_duration_seconds=120,
                          output_dir=str(tmp_path / "run"), persist_result=False)
    analyzer.get_pose_tracker()          # model loading is a per-container cost, not per-fight
    started = time.perf_counter()
    report = analyzer.analyze(req)
    wall = time.perf_counter() - started
    assert report["video"]["analysed_span"]["whole_video"] is True
    assert wall <= seconds, f"took {wall:.1f}s for {seconds:.0f}s of video"


def test_nobody_anywhere_costs_three_frames_per_larger_size(tmp_path):
    from core import preflight

    # Measured: 50 s on CPU for a 0:06 clip with nobody in it, eight frames at
    # each of 640, 1600 and 2048. Still climbs - distant fighters are found
    # only at the larger sizes - but looks before paying for every frame.
    model = _FakeModel(lambda size: [])
    report = preflight.probe(str(_video(tmp_path / "v.mp4")), model)
    assert not report.measured
    assert model.sizes == [640] * 8 + [1600] * 3 + [2048] * 3


def test_distant_fighters_found_at_a_larger_size_are_measured_on_every_frame(tmp_path):
    from core import preflight

    small = [[100, 300, 130, 380], [700, 300, 730, 380]]
    model = _FakeModel(lambda size: small if size >= 1600 else [])
    report = preflight.probe(str(_video(tmp_path / "v.mp4")), model)
    assert report.measured
    assert model.sizes.count(1600) == 8
    assert report.people_in_frame == 2
