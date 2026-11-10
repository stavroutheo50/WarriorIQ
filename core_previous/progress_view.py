"""What the live analysis page says about where a run is.

QA, 2026-10-14: the page showed "24.2%" beside "Video analysed through 0:00"
and "Locking Fighter A and Fighter B". All three were true of different
things - the percentage was the whole job, the cursor was the movement pass
that had not started, the label was the identity pass that was a quarter done
- and together they read as a contradiction.

So the page now shows one stage at a time: its name, how far through *that
stage* the run is, and a cursor line about the same stage. The three are
computed here, together, from the job's progress fields, so they cannot
disagree.
"""

from __future__ import annotations

STEPS = 4

_STARTING = {None, "", "queued", "preparing", "loading", "models"}


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds or 0.0))))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def stage_view(job: dict, analysis_duration: float, *, paused: bool = False) -> dict:
    """The step, its label, its own percentage and a cursor line, all for one stage."""
    status = job.get("status")
    stage = job.get("stage")
    duration = max(0.0, float(analysis_duration or 0.0))
    fraction = job.get("stage_fraction")
    fraction = None if fraction is None else _clamp(fraction)
    if status == "complete" or stage == "complete":
        return {"step": STEPS, "steps": STEPS, "label": "Report ready", "percent": 100.0,
                "cursor": f"Full selected footage analysed · {_clock(duration)}"}
    if paused:
        return {"step": 1, "steps": STEPS, "label": "Paused for an update", "percent": None,
                "cursor": "Your fight is saved and keeps its place in the queue"}
    if stage == "tracking":
        tracked = job.get("tracked_video_seconds")
        if fraction is None:
            # A worker older than stage_fraction: its identity pass runs 2-35%
            # of the whole job (core/analyzer.py).
            percent = float(job.get("percent") or 0.0)
            fraction = _clamp((percent - 2.0) / 33.0) if percent >= 2.0 else None
            if tracked is None and fraction is not None:
                tracked = fraction * duration
        # A stage fraction with no tracked seconds is the backward pass, which
        # walks from the chosen frame to the start: no forward cursor to show.
        cursor = (f"Fighters followed through {_clock(tracked)} of {_clock(duration)}"
                  if tracked is not None else "Finding both fighters in the video")
        return {"step": 2, "steps": STEPS, "label": "Following both fighters",
                "percent": None if fraction is None else round(fraction * 100.0, 1), "cursor": cursor}
    if stage == "analysis":
        processed = float(job.get("processed_video_seconds") or 0.0)
        if fraction is None:
            fraction = _clamp(processed / duration) if duration > 0 else 0.0
        return {"step": 3, "steps": STEPS, "label": "Reading movement and strikes",
                "percent": round(fraction * 100.0, 1),
                "cursor": f"Video analysed through {_clock(processed)} of {_clock(duration)}"}
    if stage == "report":
        return {"step": 4, "steps": STEPS, "label": "Writing the report",
                "percent": None if fraction is None else round(fraction * 100.0, 1),
                "cursor": f"Full selected footage analysed · {_clock(duration)}"}
    # Queued, preparing, loading models - or an error before any pass began.
    label = "Waiting for the analysis computer" if status == "queued" else "Starting the analysis"
    if stage not in _STARTING and status not in {"queued", "running"}:
        label = "Stopped"
    return {"step": 1, "steps": STEPS, "label": label, "percent": None,
            "cursor": "Waiting for the first analysed frame"}
