"""What every analysis must satisfy before a deploy (QA, 2026-10-04, item 22).

Pure functions over a finished report, its events and the wall time it took,
so the same checks run in the unit suite on hand-made reports and in
tools/predeploy_regression.py on real clips. Each returns a list of failures in
words; an empty list is a pass.

The four product rules they hold:

  * the whole video is analysed (from 0:00 to the end);
  * no kick or knee is counted where the legs were not visible;
  * the analysis takes no longer than the video;
  * the report gives one verdict - no section contradicts another.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

LEG_FAMILIES = ("kick", "knee")


def whole_video(report: dict, tolerance_seconds: float = 1.0) -> list[str]:
    span = (report.get("video") or {}).get("analysed_span")
    if not span:
        return ["the report does not say which part of the video it covers"]
    start = float(span.get("start_seconds") or 0.0)
    end = float(span.get("end_seconds") or 0.0)
    duration = float(span.get("video_duration_seconds") or 0.0)
    failures = []
    if start > tolerance_seconds:
        failures.append(f"analysis started at {start:.1f}s, not 0:00")
    if duration - end > tolerance_seconds:
        failures.append(f"analysis stopped at {end:.1f}s of {duration:.1f}s")
    return failures


def _legs_seen(sample: dict, side: str | None) -> bool:
    from core.action import legs_visible

    keypoints, conf = sample.get("attacker_keypoints"), sample.get("attacker_conf")
    if keypoints is None:
        return False
    item = SimpleNamespace(keypoints=np.asarray(keypoints, dtype=np.float32),
                           conf=None if conf is None else np.asarray(conf, dtype=np.float32))
    return legs_visible([item], side)  # type: ignore[list-item]


def no_kicks_without_legs(events: list[dict], *, legs_in_shot: bool = True) -> list[str]:
    """Every kick and knee had a visible leg; none at all on a waist-up clip."""
    failures = []
    for event in events:
        if str(event.get("family")) not in LEG_FAMILIES:
            continue
        if not legs_in_shot:
            failures.append(f"{event.get('family')} at {float(event.get('peak_time') or 0):.1f}s on a clip "
                            "with no legs in shot")
            continue
        samples = (event.get("evidence") or {}).get("contact_samples") or []
        side = "left" if str(event.get("limb", "")).startswith("left") else "right"
        if samples and not any(_legs_seen(sample, side) for sample in samples):
            failures.append(f"{event.get('family')} at {float(event.get('peak_time') or 0):.1f}s with the "
                            "leg never confidently seen")
    return failures


def within_video_length(wall_seconds: float, report: dict) -> list[str]:
    span = (report.get("video") or {}).get("analysed_span") or {}
    duration = float(span.get("video_duration_seconds") or 0.0)
    if duration <= 0:
        return ["the report gives no video duration to time against"]
    if wall_seconds > duration:
        return [f"took {wall_seconds:.1f}s for {duration:.1f}s of video"]
    return []


def one_verdict(report: dict) -> list[str]:
    """No section may contradict the identity verdict (core.report.identity_verdict)."""
    if report.get("mode") == "solo":
        failures = []
        if (report.get("scorecard") or {}).get("available"):
            failures.append("a solo session was scored")
        if (report.get("statistics") or {}).get("fighters"):
            failures.append("a solo session carries strike statistics")
        return failures
    from core.report import identity_verdict, refresh_identity_integrity

    report = refresh_identity_integrity(dict(report))
    verdict = identity_verdict(report)
    integrity = report.get("integrity") or {}
    failures = []
    if bool(integrity.get("identity_evidence_trusted")) != verdict["trusted"]:
        failures.append("integrity disagrees with the identity verdict")
    if not verdict["trusted"]:
        if (report.get("scorecard") or {}).get("available"):
            failures.append("scored although identity is not trusted")
        for fighter, plan in (report.get("training_plan") or {}).items():
            if plan:
                failures.append(f"training plan for Fighter {fighter} although identity is not trusted")
        for fighter, coaching in (report.get("coaching") or {}).items():
            if (coaching or {}).get("strengths") or (coaching or {}).get("improvements"):
                failures.append(f"coaching claims for Fighter {fighter} although identity is not trusted")
    if not verdict["followed_enough_to_score"] and (report.get("scorecard") or {}).get("available"):
        failures.append("scored although a fighter was not followed enough to score")
    return failures
