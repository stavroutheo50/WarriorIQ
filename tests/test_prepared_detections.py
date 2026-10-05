"""The forward pass reuses the pose model output the backward pass computed.

Measured on a 0:06 clip picked at 0:03: the backward identity pass ran the pose
model on 31 frames before the chosen one, and the forward pass then ran it on
the same 31 frames again - 8.1 s of a 29.1 s analysis spent twice. What the
model sees in a frame does not depend on which way the tracker walks, so the
backward pass now keeps it (PoseTracker.prepare_detections) and the forward
pass only re-runs the tracker's association. These tests pin that the reuse
changes nothing the tracker or the report sees.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results
from ultralytics.trackers import track as ultralytics_track

from core import analyzer, backtrack
from core.pose_tracker import PoseTracker

ROOT = Path(__file__).resolve().parents[1]
IMAGE = np.zeros((360, 640, 3), dtype=np.uint8)
NAMES = {0: "person"}


# --- which frames the backward pass samples ------------------------------------

def test_backward_samples_sit_on_the_forward_pass_frames():
    frames = backtrack.sample_frames(0, 31, 30.0, step=2)
    assert frames[0] == 31                       # the chosen frame, where identity is seeded
    assert frames[1:] == list(range(30, -1, -2))  # then exactly the frames the forward pass analyses


def test_without_a_step_sampling_is_unchanged():
    assert backtrack.sample_frames(0, 900, 30.0) == list(range(900, -1, -3))


# --- the backward pass hands each original frame to the model once ------------

class _FakeTracker:
    def __init__(self):
        self.prepared, self.tracked, self.resets = [], [], 0

    def prepare_detections(self, frame, frame_index, imgsz=None):
        self.prepared.append((frame_index, int(frame[0, 0, 0])))

    def track(self, frame, imgsz=None, frame_index=None):
        self.tracked.append(frame_index)
        return []

    def reset_tracking(self):
        self.resets += 1


def _video(path: Path, frames: int) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 8, dtype=np.uint8))
    writer.release()
    return str(path)


def test_the_backward_pass_prepares_every_sample_from_the_original_frame(tmp_path):
    video = _video(tmp_path / "clip.avi", 12)
    tracker = _FakeTracker()
    backtrack.backtrack(video, tracker, 10.0, 0, 9, [0, 0, 10, 10], [20, 0, 30, 10], 640,
                        find_initial=lambda a, b, people, image: (None, None, 0.0, 0.0),
                        workdir=tmp_path, step=2)
    expected = backtrack.sample_frames(0, 9, 10.0, step=2)
    assert sorted(index for index, _ in tracker.prepared) == sorted(expected)
    # The pixels the model saw are the decoded frame's own (MJPG keeps a flat
    # grey within a level or two), not something re-read later.
    for index, grey in tracker.prepared:
        assert abs(grey - index * 8) <= 2
    assert tracker.tracked[0] == 9               # tracked from the prepared output, by frame


# --- the tracker is fed exactly what Ultralytics' callback would feed it ------

class _RecordingTracker:
    """Returns tracks for the given detection rows, like BYTETracker.update."""

    def __init__(self, keep):
        self.keep, self.calls = keep, []
        self.args = SimpleNamespace(with_reid=False)

    def update(self, det, image, feats=None):
        self.calls.append((det.xyxy.copy(), image))
        rows = [[*det.xyxy[i] + 1.0, 7 + i, det.conf[i], 0, i] for i in self.keep]
        return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def _prepared(boxes):
    data = np.asarray([[*box, 0.9, 0.0] for box in boxes], dtype=np.float32).reshape(-1, 6)
    keypoints = np.zeros((len(boxes), 17, 3), dtype=np.float32)
    return (data, keypoints, "image0.jpg", NAMES, 0.25)


def test_matched_detections_take_the_tracker_boxes_and_ids():
    tracker = _RecordingTracker(keep=[1])
    result = PoseTracker._track_prepared(tracker, _prepared([[10, 10, 50, 90], [200, 10, 260, 90]]), IMAGE)
    assert tracker.calls[0][1] is IMAGE           # motion compensation sees this frame
    assert result.boxes.id.tolist() == [8]
    assert result.boxes.xyxy.tolist() == [[201, 11, 261, 91]]
    assert len(result.keypoints.data) == 1


def test_no_tracks_leaves_the_untracked_detections_as_ultralytics_does():
    tracker = _RecordingTracker(keep=[])
    result = PoseTracker._track_prepared(tracker, _prepared([[10, 10, 50, 90]]), IMAGE)
    assert len(tracker.calls) == 1               # the tracker still steps, even with nothing to match
    assert result.boxes.id is None and len(result.boxes) == 1


def _bare_tracker(model):
    pose = PoseTracker.__new__(PoseTracker)
    pose.model, pose.device, pose.model_path = model, "cpu", "x.pt"
    pose.timing = {"pose_model": 0.0, "appearance": 0.0}
    pose._prepared, pose.prepared_seconds_used = {}, 0.0
    return pose


def test_preparing_never_steps_the_live_tracker():
    stepped = []
    tracking = partial(ultralytics_track.on_predict_postprocess_end, persist=True)
    other = lambda predictor: None                                        # noqa: E731
    callbacks = {"on_predict_start": [partial(ultralytics_track.on_predict_start, persist=True)],
                 "on_predict_postprocess_end": [other, tracking]}

    def predict(frame, **kwargs):
        stepped.extend(cb for cb in callbacks["on_predict_postprocess_end"] if cb is tracking)
        assert not callbacks["on_predict_start"]
        return [Results(frame, path="image0.jpg", names=NAMES,
                        boxes=torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9, 0.0]]),
                        keypoints=torch.zeros((1, 17, 3)))]

    pose = _bare_tracker(SimpleNamespace(callbacks=callbacks, predict=predict))
    pose.prepare_detections(IMAGE, 5, 640)
    assert stepped == []
    assert callbacks["on_predict_postprocess_end"] == [other, tracking]   # restored
    assert len(callbacks["on_predict_start"]) == 1
    assert (5, 640) in pose._prepared


def test_track_uses_prepared_output_and_counts_its_model_time():
    live = _RecordingTracker(keep=[0])
    model = SimpleNamespace(predictor=SimpleNamespace(trackers=[live]),
                            track=lambda *a, **k: pytest.fail("the model ran again"))
    pose = _bare_tracker(model)
    pose._prepared[(4, 640)] = _prepared([[10, 10, 50, 90]])
    with patch("core.pose_tracker.embed", side_effect=lambda f, b: [None] * len(b)), \
         patch("core.pose_tracker.referee_probabilities", side_effect=lambda f, b: [None] * len(b)):
        people = pose.track(IMAGE, 640, frame_index=4)
    assert [p.track_id for p in people] == [7]
    assert pose.prepared_seconds_used == pytest.approx(0.25)


def test_a_tracker_that_needs_the_model_run_is_never_fed_prepared_output():
    live = _RecordingTracker(keep=[0])
    live.args.with_reid = True
    ran = []
    model = SimpleNamespace(predictor=SimpleNamespace(trackers=[live]),
                            track=lambda frame, **k: ran.append(1) or [Results(frame, "p", NAMES)])
    pose = _bare_tracker(model)
    pose._prepared[(4, 640)] = _prepared([[10, 10, 50, 90]])
    pose.track(IMAGE, 640, frame_index=4)
    assert ran == [1] and live.calls == []


# --- the analyser -------------------------------------------------------------

def test_the_analyser_backtracks_on_its_own_stride_and_forgets_afterwards():
    info = SimpleNamespace(fps=30.0, duration=116.0, width=1280, height=720, frame_count=3480)
    forgotten = []
    pose = SimpleNamespace(forget_prepared=lambda: forgotten.append(1))
    seen = {}

    def fake_backtrack(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    from core.types import AnalysisRequest
    req = AnalysisRequest(video_path="fight.mp4", fighter_a_box=[0, 0, 10, 10], fighter_b_box=[20, 0, 30, 10],
                          start_seconds=0.0, selection_seconds=30.0)
    with patch.object(analyzer, "get_video_info", return_value=info), \
         patch.object(analyzer, "get_pose_tracker", return_value=pose), \
         patch.object(analyzer._backtrack, "backtrack", side_effect=fake_backtrack), \
         patch.object(analyzer, "_analyze", side_effect=RuntimeError("forward failed")):
        with pytest.raises(RuntimeError, match="forward failed"):
            analyzer._analyze_from_seed(req, None)
    assert seen["step"] == analyzer.QualityController(30.0, 1280, 720).stride
    assert len(forgotten) == 2                    # before the backward pass, and after - even on failure


def test_frames_reused_are_counted_in_the_frame_pass_not_the_overhead():
    elapsed, overhead = analyzer._frame_pass_clock(pass_seconds=4.0, overhead_seconds=10.0, reused_seconds=3.0)
    assert (elapsed, overhead) == (7.0, 7.0)


# --- with the real model ------------------------------------------------------

WEIGHTS = ROOT / "yolo26m-pose.pt"


def _people_frames():
    from ultralytics.utils import ASSETS

    photo = cv2.imread(str(Path(ASSETS) / "zidane.jpg"))
    if photo is None:
        return None
    # The same two people drifting a few pixels a frame: enough motion for
    # the tracker's association and motion compensation to do real work.
    return [np.ascontiguousarray(photo[20:660, 8 * i:8 * i + 1080]) for i in range(6)]


@pytest.mark.skipif(not WEIGHTS.exists(), reason="needs the pose model weights")
def test_prepared_tracking_is_identical_to_tracking_with_the_real_model():
    frames = _people_frames()
    if frames is None:
        pytest.skip("needs Ultralytics' sample photo")
    pose = PoseTracker()

    def run(prepared):
        pose.reset_tracking()
        pose.forget_prepared()
        if prepared:
            for index, frame in enumerate(frames):
                pose.prepare_detections(frame, index, 320)
        out = []
        for index, frame in enumerate(frames):
            people = pose.track(frame, 320, frame_index=index if prepared else None)
            out.append([(p.track_id, p.box.tolist(), p.confidence, p.keypoints.tolist()) for p in people])
        return out

    pose.track(frames[0], 320)                   # the tracker exists, as after the backward pass
    plain, reused = run(prepared=False), run(prepared=True)
    assert all(len(people) >= 2 for people in plain)    # real people, really tracked
    assert all(person[0] is not None for people in plain for person in people)
    assert reused == plain
    assert pose.prepared_seconds_used > 0
