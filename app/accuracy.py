"""How often the analysis was right, from the answers people gave (owner, 2026-11-16).

The owner will not label fights by hand, so the only measure of accuracy on
real use is the answers already given on reports and on the live page:

- one tap on a counted strike: right, not a strike, wrong fighter, or another
  type (source "strike_check");
- one tap on a moment someone went down: fighter A, B or nobody
  ("down_check");
- full corrections, which name the right technique;
- "tap your fighter" or a drawn box at a crossing, which says whether the
  analysis had A and B the right way round there.

Nothing here is estimated. Every figure is a count of answers, shown with how
many answers it comes from, because a handful of answers is not a rate. The
people who answer are not a random sample either: they answer what caught
their eye, and today strike and down answers come from administrators only.

In app/ rather than core/: presentation for the admin page, and every core/*.py
is part of the analysis-code fingerprint the worker must match.
"""

from __future__ import annotations

import time
from typing import Callable

from core.annotations import accuracy_summary
from core.db import is_down_check, is_strike_check, list_annotations

FAMILIES = ("punch", "kick", "knee")
CACHE_SECONDS = 300
_cache: dict = {}


def strike_answers(annotations: list[dict]) -> dict:
    """One-tap answers per strike family: how many, and what they said."""
    families = {family: {"answered": 0, "right": 0, "not_a_strike": 0, "wrong_fighter": 0, "other_type": 0}
                for family in FAMILIES}
    for item in annotations:
        if not is_strike_check(item):
            continue
        family = (item.get("predicted") or {}).get("family")
        verdict = (item.get("corrected") or {}).get("verdict")
        if family not in families or not verdict:
            continue
        row = families[family]
        row["answered"] += 1
        if verdict in {"right", "not_a_strike", "wrong_fighter"}:
            row[verdict] += 1
        elif verdict in FAMILIES:
            row["right" if verdict == family else "other_type"] += 1
    return families


def down_answers(annotations: list[dict]) -> dict:
    """Answers on "who went down": the analysis finds the moment, people name the fighter."""
    counts = {"answered": 0, "A": 0, "B": 0, "nobody": 0}
    for item in annotations:
        if not is_down_check(item):
            continue
        verdict = (item.get("corrected") or {}).get("went_down")
        if verdict in {"A", "B", "nobody"}:
            counts["answered"] += 1
            counts[verdict] += 1
    return counts


def corrections(annotations: list[dict]) -> dict:
    """Full corrections from every account, measured the way /validation measures one person's.

    core.annotations.accuracy_summary is the tested measure behind the
    validation page; this only runs it over everyone's corrections.
    """
    full = [item for item in annotations if not is_strike_check(item) and not is_down_check(item)]
    metrics = accuracy_summary(full)["metrics"] if full else {}
    return {"total": len(full),
            **{name: metrics.get(name, {"correct": 0, "total": 0})
               for name in ("technique", "fighter_identity", "target")}}


def identity_answers(jobs: list[tuple[str, dict]],
                     anchors_for: Callable[[str, dict], list[dict]]) -> dict:
    """Crossing answers: at how many did the analysis have A and B the wrong way round?

    `anchors_for(job_id, job)` turns a fight's answers into anchors, each with
    "swapped": True when the answer disagreed with the analysis. Fights that
    were already re-run keep their anchors in the job.
    """
    counts = {"fights": 0, "answers": 0, "disagreed": 0, "fights_with_a_swap": 0, "fights_rerun": 0}
    for job_id, job in jobs:
        stored = [anchor for anchor in (job.get("identity_anchors") or []) if isinstance(anchor, dict)]
        pending = anchors_for(job_id, job) if job.get("identity_answers") else []
        # A newer answer about the same moment replaces the stored one.
        answered_again = {anchor.get("frame") for anchor in pending}
        anchors = [anchor for anchor in stored if anchor.get("frame") not in answered_again] + pending
        if not anchors:
            continue
        counts["fights"] += 1
        counts["answers"] += len(anchors)
        swapped = sum(1 for anchor in anchors if anchor.get("swapped"))
        counts["disagreed"] += swapped
        counts["fights_with_a_swap"] += 1 if swapped else 0
        counts["fights_rerun"] += 1 if any(anchor.get("swapped") for anchor in stored) else 0
    return counts


def summary(jobs: list[tuple[str, dict]], anchors_for: Callable[[str, dict], list[dict]],
            now: float | None = None) -> dict:
    """Everything above, counted at most once every CACHE_SECONDS."""
    now = time.monotonic() if now is None else now
    if _cache and now - _cache["at"] < CACHE_SECONDS:
        return _cache["value"]
    annotations = list_annotations()
    value = {"strikes": strike_answers(annotations), "down": down_answers(annotations),
             "corrections": corrections(annotations), "identity": identity_answers(jobs, anchors_for)}
    _cache.update(at=now, value=value)
    return value
