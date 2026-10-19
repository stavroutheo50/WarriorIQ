"""Time-left estimates measured from the pass that is actually running.

QA, 2026-10-14: the live page said "time left: working it out from the first
frames" for the whole of a 3.5 minute run. The only estimate came from the
per-frame movement pass, and the identity pass in front of it - most of the
wait on that clip - never produced one.

Each pass that walks the video is timed on its own. Once it has covered about
MIN_FRAMES frames its measured rate (video seconds per wall second) and the
length of video it still has to cover give the time left *in that pass*. A
rate from a different pass is never borrowed: the identity pass and the frame
pass run at unrelated speeds, and extrapolating one from the other is the
confident wrong number core/analyzer.py already removed once.
"""

from __future__ import annotations

MIN_FRAMES = 50
# A light EWMA: absorbs a slow exchange without jumping by minutes.
SMOOTHING = 0.2


class StageEta:
    """Seconds left in the current pass, or None until it has measured itself."""

    def __init__(self, fps: float, min_frames: int = MIN_FRAMES):
        self.fps = max(1.0, float(fps or 0.0))
        self.min_frames = int(min_frames)
        self.stage: str | None = None
        self._started_at = 0.0
        self._started_done = 0.0
        self._estimate: float | None = None

    def update(self, stage: str, done_seconds: float, total_seconds: float, now: float) -> float | None:
        """Feed the pass's progress (video seconds covered of total) at wall time ``now``."""
        done = max(0.0, float(done_seconds))
        total = max(0.0, float(total_seconds))
        if stage != self.stage:
            self.stage, self._started_at, self._started_done, self._estimate = stage, float(now), done, None
            return None
        covered = done - self._started_done
        spent = float(now) - self._started_at
        if covered * self.fps < self.min_frames or spent <= 0.0:
            return self._estimate
        rate = covered / spent
        if rate <= 0.0:
            return self._estimate
        remaining = max(0.0, total - done) / rate
        if self._estimate is None:
            self._estimate = remaining
        else:
            # The previous estimate has itself counted down since it was made.
            self._estimate = (1.0 - SMOOTHING) * self._estimate + SMOOTHING * remaining
            self._estimate = min(self._estimate, remaining * 1.5)
        return max(0.0, self._estimate)
