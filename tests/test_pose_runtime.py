from types import SimpleNamespace

import numpy as np
import pytest

from core import preflight


@pytest.mark.parametrize("device", [0, "cpu", None])
def test_preflight_keeps_the_loaded_models_device(monkeypatch, device):
    properties = {
        preflight.cv2.CAP_PROP_FRAME_WIDTH: 640,
        preflight.cv2.CAP_PROP_FRAME_HEIGHT: 480,
        preflight.cv2.CAP_PROP_FPS: 30,
        preflight.cv2.CAP_PROP_FRAME_COUNT: 300,
    }
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = SimpleNamespace(
        isOpened=lambda: True, get=lambda key: properties[key],
        set=lambda *args: None, read=lambda: (True, frame), release=lambda: None,
    )
    calls = []

    def predict(frame, **kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(boxes=None)]

    monkeypatch.setattr(preflight.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(preflight, "_global_shift", lambda a, b: 0)
    kwargs = {"device": device} if device is not None else {}
    preflight.probe("fight.mp4", SimpleNamespace(predict=predict), samples=2, **kwargs)

    assert calls
    for call in calls:
        if device is None:
            assert "device" not in call
        else:
            assert call["device"] == device


def test_analysis_passes_tracker_device_into_preflight(monkeypatch, tmp_path):
    from core import analyzer
    from core.types import AnalysisRequest, VideoInfo

    class ProbeReached(BaseException):
        pass

    calls = []

    def probe(*args, **kwargs):
        calls.append(kwargs)
        raise ProbeReached

    monkeypatch.setattr(analyzer, "get_video_info", lambda path: VideoInfo(path, 30, 300, 640, 480, 10))
    monkeypatch.setattr(analyzer, "get_pose_tracker", lambda: SimpleNamespace(model=object(), device="cpu"))
    monkeypatch.setattr(analyzer, "probe_video", probe)
    monkeypatch.setattr(analyzer, "build_recovery", lambda: None)
    monkeypatch.setattr(analyzer, "ActionEngine", lambda: None)
    monkeypatch.setattr(analyzer, "_log_gpu_state", lambda: None)
    monkeypatch.setattr(analyzer, "_log_host_memory", lambda *args: None)
    monkeypatch.setattr(analyzer.torch.cuda, "is_available", lambda: False)
    request = AnalysisRequest("unused.mp4", [0, 0, 50, 100], [100, 0, 150, 100],
                              output_dir=str(tmp_path), persist_result=False)
    with pytest.raises(ProbeReached):
        analyzer.analyze(request)
    assert calls[0].get("device") == "cpu"


@pytest.mark.parametrize("inference_fails", [False, True])
def test_preflight_ignores_tracking_callbacks_and_restores_them(monkeypatch, inference_fails):
    from functools import partial

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    properties = {preflight.cv2.CAP_PROP_FRAME_WIDTH: 640, preflight.cv2.CAP_PROP_FRAME_HEIGHT: 480,
                  preflight.cv2.CAP_PROP_FPS: 30, preflight.cv2.CAP_PROP_FRAME_COUNT: 300}
    capture = SimpleNamespace(isOpened=lambda: True, get=lambda key: properties[key],
                              set=lambda *args: None, read=lambda: (True, frame), release=lambda: None)
    tracking_calls, custom_calls = [], []

    def on_predict_postprocess_end(persist=False):
        tracking_calls.append(persist)

    on_predict_postprocess_end.__module__ = "ultralytics.trackers.track"
    callbacks = {"on_predict_postprocess_end": [partial(on_predict_postprocess_end, persist=True),
                                                lambda: custom_calls.append(True)]}
    original = callbacks["on_predict_postprocess_end"]

    def predict(frame, **kwargs):
        for callback in callbacks["on_predict_postprocess_end"]:
            callback()
        if inference_fails:
            raise RuntimeError("test inference failure")
        return [SimpleNamespace(boxes=None)]

    monkeypatch.setattr(preflight.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(preflight, "_global_shift", lambda a, b: 0)
    model = SimpleNamespace(predict=predict, callbacks=callbacks)
    if inference_fails:
        with pytest.raises(RuntimeError, match="test inference failure"):
            preflight.probe("unused.mp4", model, samples=2)
    else:
        preflight.probe("unused.mp4", model, samples=2)
    assert tracking_calls == []
    assert custom_calls
    assert callbacks["on_predict_postprocess_end"] is original
