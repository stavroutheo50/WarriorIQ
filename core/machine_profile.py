"""What a frame costs on this machine, measured once and then reused.

The sampling stride is planned from how long a frame takes here. Measuring that
inside the run means reading a clock, and a clock moves with machine load - so
the same video, on the same machine, from the same commit, planned stride 3 on
some runs and stride 4 on others. Measured on three fights, that flip moved
fighter B's coverage by 0.468 on one, 0.138 the other way on another, and not
at all on the third. There is no safe stride to prefer; what there is, is a
requirement that the same input gives the same answer.

So the cost is measured on the first analysis that needs it, rounded hard, and
written down. Every later analysis on that machine reads the written value and
plans identically. The rounding is what makes it stick: a value kept to the
microsecond would be a different number every time it was re-measured, and a
stride derived from it would go on flipping at the boundaries.

Keyed by device and inference size because those are what actually move the
number - the same card is twice as slow at twice the pixels, and a machine
without a GPU is not comparable to one with it.

And keyed by PIPELINE_VERSION, because write-once has a failure mode that took
a day to find. On 2026-09-18 a fix stopped the recovery predictor loading a
second TensorRT execution context, which had been taking the card from 3582 to
7164 MiB of 8151 and starving SAM2. Frames got far cheaper. The stored cost did
not move, because it never moves - and the planner went on believing a frame
cost 0.083081 s when it now cost about a quarter of that.

That direction is the quiet one. A machine slower than its profile misses its
deadline and plan_for_budget now says so. A machine FASTER than its profile
looks fine and silently *reduces sampling*, because the stale number says it
cannot afford the frames: measured here, stride 2 cut to 3 while sitting on
three times the headroom it needed. Frames are identity and strike evidence, so
that is a real cost with nothing anywhere reporting it.

So: bump PIPELINE_VERSION whenever a change alters what a frame costs. Old
entries stop matching, the next run measures once and writes a fresh one, and
determinism is preserved within a version - which is all it was ever protecting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from core.config import DATA_ROOT

# Cost buckets, as a multiplier per step. A measurement is snapped to the
# nearest bucket before it is stored, so ordinary run-to-run jitter lands on
# the same bucket and plans the same stride. 12% steps: comfortably wider than
# the jitter seen between runs here, and narrow enough that the estimate stays
# within about a stride of the truth across the range that matters.
BUCKET_RATIO = 1.12
MIN_COST_SECONDS = 0.001
MAX_COST_SECONDS = 10.0


def profile_path() -> Path:
    return Path(os.getenv(
        "WARRIORIQ_MACHINE_PROFILE",
        str(DATA_ROOT / "machine_profile.json"))).expanduser()


# Bump this when a change alters what a frame costs, so stored numbers from the
# old pipeline stop being read. See the module docstring for why write-once
# needs an escape hatch.
#
#   1  original
#   2  2026-09-18: the recovery predictor stopped loading a second TensorRT
#      execution context, which had been starving SAM2 and had every cost on
#      this machine measured against a card with under 1 GB free.
PIPELINE_VERSION = 2


def _key(device: str, imgsz: int) -> str:
    # The version leads, so an entry from an older pipeline simply does not
    # match and is re-measured once rather than needing a migration. Old entries
    # are left in the file: they are a few bytes, and they are a record of what
    # this machine used to cost.
    return "v%d|%s@%d" % (PIPELINE_VERSION,
                          (device or "unknown").strip() or "unknown",
                          max(1, int(imgsz)))


def snap(seconds: float) -> float:
    """Round a measured cost to its bucket, so re-measuring gives the same value.

    Geometric rather than linear because the quantity spans two orders of
    magnitude - 0.02s on a small model, 0.2s at imgsz 1632 on this footage -
    and a fixed step would be far too coarse at one end and pointless at the
    other.
    """
    import math

    value = float(seconds)
    if not value > 0 or value != value:                  # zero, negative, NaN
        return 0.0
    value = min(MAX_COST_SECONDS, max(MIN_COST_SECONDS, value))
    steps = round(math.log(value / MIN_COST_SECONDS, BUCKET_RATIO))
    return round(MIN_COST_SECONDS * (BUCKET_RATIO ** steps), 6)


def _load() -> dict:
    try:
        with open(profile_path(), "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        return stored if isinstance(stored, dict) else {}
    except Exception:                       # noqa: BLE001 - absent or unreadable
        return {}


def frame_cost(device: str, imgsz: int) -> float | None:
    """The stored cost for this machine and size, or None if never measured."""
    value = _load().get(_key(device, imgsz))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def record_frame_cost(device: str, imgsz: int, seconds: float) -> float | None:
    """Store a measured cost, once. Later measurements do not overwrite it.

    Deliberately write-once. Blending each run's measurement into the stored
    value would move it a little every time, and a stride derived from a
    moving number is exactly the flip this module exists to stop. Delete the
    file, or point WARRIORIQ_MACHINE_PROFILE elsewhere, to re-measure after a
    hardware or driver change.
    """
    snapped = snap(seconds)
    if not snapped:
        return None
    key = _key(device, imgsz)
    stored = _load()
    if key in stored:
        try:
            existing = float(stored[key])
            if existing > 0:
                return existing
        except (TypeError, ValueError):
            pass
    stored[key] = snapped
    try:
        path = profile_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written whole and moved into place, so a run killed mid-write cannot
        # leave a half-file that every later run then fails to parse.
        temporary = path.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(stored, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except Exception:                       # noqa: BLE001 - never fail a run
        return snapped
    return snapped
