"""Why an analysis ran slow, written down while it happens.

2026-10-13: a 3:17 fight took 8.3 minutes on an RTX 5060. The run measured
0.048 s a frame when it started and then went five times slower, and the
report could only say "other" for the difference. Whatever slowed the machine
- another program on the graphics card, the card throttling, the computer
running out of memory - was gone by the time anybody looked.

So the frame pass is watched against its own start: when the recent frames
take SLOW_FACTOR times longer than the first ones, the state of the machine is
read once, there and then, and the report carries it with a plain reason.
Reading costs a few hundred milliseconds, once, and only on a run that is
already slow. Nothing here changes how a frame is analysed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections import deque
from statistics import median

LOGGER = logging.getLogger("warrioriq.slow_watch")

# Frames skipped before the baseline is taken: model warm-up and the first
# tracker frames are not representative.
WARMUP_FRAMES = 5
BASELINE_FRAMES = 20
RECENT_FRAMES = 20
# How many times slower than its own start a run must get to be called slow.
SLOW_FACTOR = 3.0
HOT_GPU_CELSIUS = 83
LOW_MEMORY_BYTES = 1.5 * 1024 ** 3
BUSY_CPU_PERCENT = 90.0


def _nvidia_smi(*query: str) -> list[list[str]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run([executable, *query, "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


def machine_snapshot() -> dict:
    """The graphics card, memory and processor right now; whatever can be read."""
    snapshot: dict = {}
    gpu = _nvidia_smi("--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,"
                      "clocks_throttle_reasons.active")
    if gpu:
        row = gpu[0]
        try:
            snapshot["gpu"] = {"temperature_c": float(row[0]), "utilization_percent": float(row[1]),
                               "memory_used_mb": float(row[2]), "memory_total_mb": float(row[3]),
                               "throttle_reasons": row[4] if len(row) > 4 else None}
        except (ValueError, IndexError):
            pass
    apps = _nvidia_smi("--query-compute-apps=pid,process_name,used_memory")
    own = os.getpid()
    others = []
    for row in apps:
        try:
            if int(row[0]) == own:
                continue
            others.append({"name": os.path.basename(row[1]), "memory_mb": float(row[2])})
        except (ValueError, IndexError):
            continue
    if apps:
        snapshot["other_gpu_programs"] = others
    try:
        import psutil

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        snapshot["memory"] = {"available_bytes": int(memory.available), "total_bytes": int(memory.total),
                              "swap_used_percent": float(swap.percent)}
        snapshot["cpu_percent"] = float(psutil.cpu_percent(interval=0.2))
    except Exception:  # noqa: BLE001 - optional, and never worth failing a run over
        pass
    return snapshot


def slow_reasons(snapshot: dict) -> list[str]:
    """What in a snapshot explains a slow run, in words. Empty when nothing does."""
    reasons = []
    gpu = snapshot.get("gpu") or {}
    if gpu.get("temperature_c") is not None and gpu["temperature_c"] >= HOT_GPU_CELSIUS:
        reasons.append(f"the graphics card was hot ({gpu['temperature_c']:.0f} °C) and slowing itself down")
    elif gpu.get("throttle_reasons") not in (None, "", "0x0000000000000000", "0x0000000000000001"):
        reasons.append(f"the graphics card was holding its speed down (reason code {gpu['throttle_reasons']})")
    others = snapshot.get("other_gpu_programs") or []
    if others:
        names = ", ".join(sorted({item["name"] for item in others})[:4])
        reasons.append(f"other programs were using the graphics card ({names})")
    memory = snapshot.get("memory") or {}
    if memory.get("available_bytes") is not None and memory["available_bytes"] < LOW_MEMORY_BYTES:
        reasons.append(f"the computer was low on memory ({memory['available_bytes'] / 1024 ** 3:.1f} GB free)")
    if snapshot.get("cpu_percent") is not None and snapshot["cpu_percent"] >= BUSY_CPU_PERCENT:
        reasons.append(f"the processor was busy ({snapshot['cpu_percent']:.0f}% in use)")
    return reasons


class SlowRunWatch:
    """Fed the time of every analysed frame; reads the machine once if the run slows down."""

    def __init__(self, snapshot=machine_snapshot) -> None:
        self._snapshot = snapshot
        self._last: float | None = None
        self._seen = 0
        self._baseline_samples: list[float] = []
        self._recent: deque[float] = deque(maxlen=RECENT_FRAMES)
        self.baseline: float | None = None
        self.event: dict | None = None

    def frame(self, now: float, video_seconds: float) -> None:
        if self._last is None:
            self._last = now
            return
        took, self._last = max(0.0, now - self._last), now
        self._seen += 1
        if self._seen <= WARMUP_FRAMES:
            return
        if self.baseline is None:
            self._baseline_samples.append(took)
            if len(self._baseline_samples) >= BASELINE_FRAMES:
                self.baseline = median(self._baseline_samples)
            return
        self._recent.append(took)
        if self.event is not None or len(self._recent) < RECENT_FRAMES or self.baseline <= 0:
            return
        recent = median(self._recent)
        if recent >= SLOW_FACTOR * self.baseline:
            snapshot = self._snapshot()
            self.event = {"at_video_seconds": round(float(video_seconds), 1),
                          "baseline_seconds_per_frame": round(self.baseline, 4),
                          "recent_seconds_per_frame": round(recent, 4),
                          "times_slower": round(recent / self.baseline, 1),
                          "snapshot": snapshot,
                          "reasons": slow_reasons(snapshot)}
            LOGGER.warning("analysis_slowed at=%.1fs times_slower=%.1f reasons=%s", video_seconds,
                           recent / self.baseline, "; ".join(self.event["reasons"]) or "not identified")

    def summary(self) -> dict | None:
        """For the report: None when the run never slowed down."""
        return self.event
