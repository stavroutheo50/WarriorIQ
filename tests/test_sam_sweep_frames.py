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
