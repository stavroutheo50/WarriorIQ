"""Decode the next frames while the current one is being analysed.

The frame pass decodes every source frame and analyses one in `stride`. On
1080p60 phone footage that is about nine decodes per analysed frame, and the
analyser used to wait for each of them in turn while the GPU sat idle. This
moves the decoding onto a helper thread, so it overlaps with the pose model,
identity and the rest of the work on the frame before.

The results must not change, so the helper decodes exactly what the loop would
have decoded, in the same order, and converts (retrieves) exactly the frames
the loop would have converted:

  * every frame, when the fallback recovery buffer is on;
  * one frame a second for the optional identity referee's history;
  * the next frame to analyse - which the loop decides *before* it analyses
    the current one, so the helper is told it with `allow_through()` and may
    not read past it until told the next one.

OpenCV and torch both release the GIL while they work, so the two threads do
run at the same time. `ahead=False` runs the same rules inline, without a
thread, which is what the comparison tests use.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import cv2


@dataclass
class FedFrame:
    source_frame: int
    pts_ms: float
    frame: object | None   # BGR image, or None when no one asked for it


_END = object()


class FrameFeed:
    def __init__(self, cap, *, first_source_frame: int, next_inference_frame: int,
                 retrieve_all: bool, history_step: int | None, next_history_frame: int,
                 ahead: bool = True, depth: int = 64):
        self._cap = cap
        self._next_frame = first_source_frame
        self._gate = next_inference_frame
        self._retrieve_all = retrieve_all
        self._history_step = history_step
        self._next_history = next_history_frame
        self._ahead = ahead
        self._condition = threading.Condition()
        self._stopped = False
        self._error: BaseException | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._thread = None
        if ahead:
            self._thread = threading.Thread(target=self._run, name="warrioriq-frame-feed", daemon=True)
            self._thread.start()

    # -- the decoding rules, shared by both modes --------------------------
    def _decode_one(self) -> FedFrame | None:
        """Grab the next frame and convert it if anything will look at it."""
        if not self._cap.grab():
            return None
        index = self._next_frame
        self._next_frame += 1
        pts_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        wanted = self._retrieve_all or index >= self._gate
        if self._history_step and index >= self._next_history:
            wanted = True
            self._next_history = index + self._history_step
        frame = None
        if wanted:
            ok, image = self._cap.retrieve()
            frame = image if ok else None
        return FedFrame(index, pts_ms, frame)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    # Never read past the next frame to analyse until the loop
                    # has decided which one comes after it.
                    while not self._stopped and self._next_frame > self._gate:
                        self._condition.wait()
                    if self._stopped:
                        return
                item = self._decode_one()
                if item is None:
                    self._put(_END)
                    return
                if not self._put(item):
                    return
        except BaseException as exc:  # handed to the loop, never swallowed
            self._error = exc
            self._put(_END)

    def _put(self, item) -> bool:
        while True:
            if self._stopped:
                return False
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue

    # -- what the analysis loop calls --------------------------------------
    def next(self) -> FedFrame | None:
        """The next decoded frame, or None at the end of the video."""
        if not self._ahead:
            return self._decode_one()
        item = self._queue.get()
        if item is _END:
            if self._error is not None:
                raise self._error
            return None
        return item

    def allow_through(self, next_inference_frame: int) -> None:
        """The loop has chosen the next frame to analyse."""
        with self._condition:
            self._gate = next_inference_frame
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)
