"""The identity sweep's frames, decoded faster but exactly as before (2026-10-20).

Measured on the owner's RTX 5060 the SAM2 sweep was 43 s of an 84 s analysis.
It read() every source frame to keep one in `stride`. core/sam_recovery.py
_SampledFrames now grab()s the frames it skips and decodes on a helper thread.
These tests hold it to the old loop: the same frames, in the same order, the
same pixels, stopping where the old loop stopped.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from core.sam_recovery import _DecodedClip, _SampledFrames

SIZE = 64


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 96))
    rng = np.random.default_rng(7)
    for i in range(47):
        frame = rng.integers(0, 255, (96, 160, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    return str(path)


def _old_loop(path, start, end, stride):
    """The sweep's loop before 2026-10-20, kept here as the reference."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out, source = [], start
    while source < end:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (source - start) % stride == 0:
            out.append((source, _DecodedClip.prepare(frame, SIZE)))
        source += 1
    cap.release()
    return out


@pytest.mark.parametrize("start,end,stride", [(0, 47, 5), (3, 40, 4), (0, 200, 7), (10, 11, 1)])
def test_the_same_frames_in_the_same_order(video, start, end, stride):
    expected = _old_loop(video, start, end, stride)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = _SampledFrames(cap, start, end, stride, SIZE, depth=2)
    try:
        got = [(source, prepared) for source, prepared, _ in frames]
    finally:
        frames.close()
        cap.release()
    assert [s for s, _ in got] == [s for s, _ in expected]
    for (_, a), (_, b) in zip(got, expected):
        assert np.array_equal(a, b)


def test_a_prepared_frame_fills_the_clip_exactly_as_add_did():
    frame = np.random.default_rng(3).integers(0, 255, (96, 160, 3), dtype=np.uint8)
    one, two = _DecodedClip(SIZE, 1, 96, 160), _DecodedClip(SIZE, 1, 96, 160)
    one.add(frame)
    two.add_prepared(_DecodedClip.prepare(frame, SIZE))
    assert torch.equal(one.taken(), two.taken())


def test_stopping_early_does_not_hang_or_leak_the_thread(video):
    cap = cv2.VideoCapture(video)
    frames = _SampledFrames(cap, 0, 47, 1, SIZE, depth=1)
    first = next(iter(frames))
    assert first[0] == 0
    frames.close()
    assert not frames._thread.is_alive()
    cap.release()


def test_a_decoding_error_reaches_the_sweep():
    class Broken:
        def grab(self):
            raise RuntimeError("disk gone")

    frames = _SampledFrames(Broken(), 0, 10, 1, SIZE)
    with pytest.raises(RuntimeError, match="disk gone"):
        list(frames)
    frames.close()


# The masks' boxes are found on the card now (2026-10-20: 7.4 s of a 37 s sweep).
def _cases():
    rng = np.random.default_rng(11)
    empty = torch.full((1, 40, 60), -1.0)
    three = empty.clone()
    three[0, 5, 7] = three[0, 6, 9] = three[0, 30, 50] = 2.0
    four = three.clone()
    four[0, 39, 0] = 0.5
    blob = torch.from_numpy(rng.normal(size=(1, 40, 60)).astype(np.float32))
    edge = torch.full((40, 60), -3.0)
    edge[0, :] = 1.0
    edge[:, 59] = 1.0
    return [empty, three, four, blob, edge, torch.zeros(2, 4, 4) + 1.0, torch.full((1, 40, 60), 0.0)]


@pytest.mark.parametrize("logits", _cases())
def test_the_box_from_the_card_is_the_box_numpy_found(logits):
    from core.sam_recovery import SamRecovery

    expected = SamRecovery._mask_to_box((logits > 0.0).detach().cpu().numpy())
    got = SamRecovery._logits_to_box(logits)
    if expected is None:
        assert got is None
    else:
        assert got is not None and got.dtype == np.float32 and np.array_equal(got, expected)


# The sweep beside the frame pass (2026-10-21): the same guidance as running it first.
class _FakeRecovery:
    """Finishes sampled frames in order, slowly, as SamRecovery.track_segment does."""

    def __init__(self, stride=5, end=300, fail_at=None, delay=0.0005):
        self.frame_done, self.available, self.sweep_seconds = None, True, {"propagation": 0.0}
        self.continuous_frames, self.continuous_failure_reason = 0, None
        self.stride, self.end, self.fail_at, self.delay = stride, end, fail_at, delay
        self.released_before_done = None
        self.events = []

    def track_segment(self, video_path, start, end, fps, a, b):
        import time as _time

        rng = np.random.default_rng(5)
        tracks = {}
        for source in range(start, end, self.stride):
            _time.sleep(self.delay * float(rng.integers(0, 4)))
            if self.fail_at is not None and source >= self.fail_at:
                raise RuntimeError("CUDA out of memory")
            guided = {"A": np.asarray([source, 0, source + 10, 10], dtype=np.float32)} if source % 3 else {}
            if guided:
                tracks[source] = guided
            if self.frame_done is not None:
                self.frame_done(source, guided or None)
        self.continuous_frames = len(tracks)
        return tracks

    def release(self):
        self.events.append("release")


def test_the_background_sweep_gives_the_guidance_running_it_first_gave():
    from core.sam_recovery import SweepInBackground, nearest_guidance

    first = _FakeRecovery()
    first.frame_done = None
    finished = first.track_segment("v", 10, 300, 30.0, None, None)
    sweep = SweepInBackground(_FakeRecovery(), "v", 10, 300, 30.0, None, None)
    for source in range(10, 300, 2):              # the frame pass asks as it goes
        got = sweep.guidance(source, 5)
        want = nearest_guidance(finished, source, 5)
        assert (got is None) == (want is None), source
        if got is not None:
            assert np.array_equal(got["A"], want["A"]), source
    assert sweep.join().keys() == finished.keys()
    assert sweep.waited >= 0 and not sweep.finished_empty()


def test_a_sweep_that_fails_ends_with_no_tracks_and_releases_first():
    from core.sam_recovery import SweepInBackground

    recovery = _FakeRecovery(fail_at=120)
    sweep = SweepInBackground(recovery, "v", 10, 300, 30.0, None, None)
    sweep.guidance(290, 5)                       # waits for the end, which is the failure
    assert sweep.finished_empty() and sweep.join() == {}
    assert recovery.events == ["release"] and recovery.frame_done is None
    assert "CUDA out of memory" in recovery.continuous_failure_reason
    assert sweep.guidance(50, 5) is None         # after the failure: no tracks, as before
