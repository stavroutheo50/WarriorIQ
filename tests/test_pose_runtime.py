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


def test_analysis_passes_tracker_device_into_preflight(monkeypatch):
    from core import analyzer

    calls = []
    monkeypatch.setattr(analyzer, "probe_video", lambda *args, **kwargs: calls.append(kwargs) or "measured")
    tracker = SimpleNamespace(model=object(), device="cpu")
    assert analyzer._measure_footage("unused.mp4", tracker, 0.0, None) == "measured"
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
    # A size the backend cannot serve is recorded and the probe carries on
    # (core/preflight.py), so a failing inference does not escape - and the
    # callbacks must be restored either way.
    preflight.probe("unused.mp4", model, samples=2)
    assert tracking_calls == []
    assert custom_calls
    assert callbacks["on_predict_postprocess_end"] is original
