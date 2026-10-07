"""Solo sessions: one person - shadowboxing, bag work, pad work.

QA, 2026-10-04: a video of one person was a dead end. Selection needed two
different people, and the fight analysis would have excluded the footage
anyway, because it only counts time when two people are fighting.

A solo session follows one person through the whole video and measures what
one person's movement can show: how much they moved, how often the guard was
up, and how balanced they stood. It does not count strikes, measure anything
relative to an opponent, or score. There is no opponent here, and a pad holder
is not one.

Following the person is deliberately simple. The pose tracker's track id
carries them from frame to frame. If the track is lost, it is picked up again
only next to where they were last seen and within a second. After a longer gap
it is picked up only by someone who looks like them, and the same holds going
backwards from the frame the person picked.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from core.build_info import stamp as build_stamp
from core.config import OUTPUTS
from core.identity import appearance_similarity, box_iou
from core.metrics import MetricsAccumulator
from core.types import AnalysisProgress, AnalysisRequest, PersonObservation

LOGGER = logging.getLogger("warrioriq.solo")

# Longest the person may go unseen and still be picked up by position alone.
MAX_POSITION_GAP_SECONDS = 1.0
# After a longer gap the candidate has to look like the person did on the
# frame that was picked.
REACQUIRE_APPEARANCE = 0.75
# A drawn box and the detected person must agree this much on the picked frame.
MIN_SEED_IOU = 0.3
# The same coverage the two-fighter identity gate asks of each fighter
# (core/report.py identity_ready_by_fighter).
MIN_TRUSTED_COVERAGE = 0.45
# The movement measurements a solo session reports. Everything else in a
# fighter's metrics is about strikes or an opponent.
SOLO_METRICS = ("pose_coverage", "measurement", "footwork_body_lengths_per_second",
                "guard_index", "guard_definition", "balance_index", "moments", "spread", "numbers")


def _centre(box) -> np.ndarray:
    return np.array([(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0])


def _height(box) -> float:
    return max(1.0, float(box[3]) - float(box[1]))


def seed_person(people: list[PersonObservation], drawn) -> tuple[PersonObservation | None, float]:
    """The detected person the drawn box is around, and how well they agree."""
    if not people:
        return None, 0.0
    best = max(people, key=lambda person: box_iou(person.box, drawn))
    overlap = box_iou(best.box, drawn)
    return (best, overlap) if overlap >= MIN_SEED_IOU else (None, overlap)


def follow(samples: list[tuple[int, float, list[PersonObservation]]], seed_index: int,
           seed: PersonObservation) -> list[PersonObservation | None]:
    """The person in every sample, walking out from the seed both ways.

    Pure: ``samples`` are (frame, seconds, people) in time order. Kept free of
    the model so it can be tested on its own.
    """
    found: list[PersonObservation | None] = [None] * len(samples)
    found[seed_index] = seed
    anchor = seed.appearance
    for direction in (1, -1):
        last, last_seconds = seed, samples[seed_index][1]
        index = seed_index + direction
        while 0 <= index < len(samples):
            _, seconds, people = samples[index]
            choice = None
            if last.track_id is not None:
                choice = next((p for p in people if p.track_id == last.track_id), None)
            gap = abs(seconds - last_seconds)
            if choice is None and people and gap <= MAX_POSITION_GAP_SECONDS:
                near = min(people, key=lambda p: float(np.linalg.norm(_centre(p.box) - _centre(last.box))))
                if float(np.linalg.norm(_centre(near.box) - _centre(last.box))) <= _height(last.box):
                    choice = near
            if choice is None and people and gap > MAX_POSITION_GAP_SECONDS and anchor is not None:
                scored = [(appearance_similarity(anchor, p.appearance), p) for p in people]
                similarity, best = max(scored, key=lambda item: item[0])
                if similarity >= REACQUIRE_APPEARANCE:
                    choice = best
            if choice is not None:
                found[index] = choice
                last, last_seconds = choice, seconds
            index += direction
    return found


def _sample_frames(cap, start_frame: int, end_frame: int, stride: int, seed_frame: int):
    """Decode once, forward, yielding every stride-th frame and the seed frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    index = start_frame
    while index < end_frame:
        if not cap.grab():
            break
        if (index - start_frame) % stride == 0 or index == seed_frame:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                break
            yield index, frame
        index += 1


def _observation_record(seconds: float, obs: PersonObservation | None) -> dict:
    if obs is None:
        return {"t": round(seconds, 3), "A": None}
    return {"t": round(seconds, 3), "A": {
        "box": [round(float(v), 1) for v in obs.box],
        "keypoints": None if obs.keypoints is None else np.round(obs.keypoints, 1).tolist(),
        "keypoint_conf": None if obs.keypoint_conf is None else np.round(obs.keypoint_conf, 3).tolist(),
    }}


def analyze_solo(req: AnalysisRequest, progress_callback=None) -> dict:
    """Follow one person through the whole video and measure their movement."""
    from core.analyzer import get_pose_tracker
    from core.pose_tracker import QualityController
    from core.video import get_video_info

    wall_start = time.perf_counter()
    info = get_video_info(req.video_path)
    fps = float(info.fps) if info.fps > 0 else 30.0
    start_seconds = max(0.0, min(float(req.start_seconds), max(0.0, info.duration - 0.001)))
    end_seconds = info.duration if req.end_seconds is None else max(
        start_seconds, min(float(req.end_seconds), info.duration))
    seed_seconds = max(start_seconds, min(float(
        req.selection_seconds if req.selection_seconds is not None else start_seconds), end_seconds))
    start_frame = int(round(start_seconds * fps))
    end_frame = max(start_frame + 1, min(info.frame_count, int(np.ceil(end_seconds * fps))))
    seed_frame = min(end_frame - 1, int(round(seed_seconds * fps)))
    duration = max(0.001, end_seconds - start_seconds)

    def progress(percent: float, message: str, processed: float, stage: str) -> None:
        if progress_callback is None:
            return
        elapsed = time.perf_counter() - wall_start
        speed = processed / elapsed if elapsed > 0 else 0.0
        progress_callback(AnalysisProgress(
            percent=percent, message=message, elapsed_seconds=elapsed,
            processed_video_seconds=processed, speed=speed,
            eta_seconds=(duration - processed) / speed if speed > 0 else None,
            stage=stage, video_duration_seconds=float(info.duration),
            analysed_from_seconds=start_seconds,
        ).to_dict())

    progress(1.0, "Following the person through the video", 0.0, "tracking")
    pose_tracker = get_pose_tracker()
    quality = QualityController(info.fps, info.width, info.height)
    cap = cv2.VideoCapture(req.video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video")
    samples: list[tuple[int, float, list[PersonObservation]]] = []
    pose_tracker.reset_tracking()
    next_report = time.perf_counter() + 1.0
    try:
        warmed = False
        for frame_index, frame in _sample_frames(cap, start_frame, end_frame, quality.stride, seed_frame):
            if not warmed:
                pose_tracker.warmup(frame)
                warmed = True
            seconds = frame_index / fps
            samples.append((frame_index, seconds, pose_tracker.track(frame, quality.imgsz)))
            if time.perf_counter() >= next_report:
                processed = seconds - start_seconds
                progress(2.0 + 90.0 * processed / duration, "Following the person through the video",
                         processed, "analysis")
                next_report = time.perf_counter() + 1.0
    finally:
        cap.release()
        pose_tracker.reset_tracking()
    if not samples:
        raise RuntimeError("No frames could be read from the video")

    seed_index = min(range(len(samples)), key=lambda i: abs(samples[i][0] - seed_frame))
    seed, seed_iou = seed_person(samples[seed_index][2], req.fighter_a_box)
    followed = follow(samples, seed_index, seed) if seed is not None else [None] * len(samples)

    metrics = MetricsAccumulator(info.width, info.height)
    job_dir = Path(req.output_dir) if req.output_dir else OUTPUTS / req.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "tracking.jsonl").open("w", encoding="utf-8") as tracking_file:
        for (_, seconds, _), obs in zip(samples, followed):
            metrics.update("A", seconds, 1, obs, None)
            tracking_file.write(json.dumps(_observation_record(seconds, obs)) + "\n")
    (job_dir / "events.json").write_text("[]", encoding="utf-8")
    measured = metrics.finalize([], [], duration)["A"]
    coverage = sum(1 for obs in followed if obs is not None) / len(samples)
    trusted = seed is not None and coverage >= MIN_TRUSTED_COVERAGE
    analysis_seconds = time.perf_counter() - wall_start

    if seed is None:
        note = ("WarriorIQ could not find the person you boxed on the frame you picked, so nothing "
                "below is measured. Pick a moment where they are clearly visible and box them again.")
    elif not trusted:
        note = (f"The person you boxed was followed for {coverage * 100:.0f}% of the video, which is "
                "too little to measure reliably.")
    else:
        note = None
    report = {
        "mode": "solo",
        "video": {
            "source_name": req.original_name or Path(req.video_path).name,
            "analysis_target": "A", "focus_fighter": "A",
            "fps": fps, "width": info.width, "height": info.height,
            "analysed_span": {
                "start_seconds": round(start_seconds, 3), "end_seconds": round(end_seconds, 3),
                "video_duration_seconds": round(float(info.duration), 3),
                "selection_seconds": round(seed_seconds, 3),
                "whole_video": start_seconds < 1.0 and float(info.duration) - end_seconds < 1.0,
                "excluded_start_seconds": round(start_seconds, 3),
                "excluded_reason": None, "excluded_reason_text": None, "backtrack": None,
            },
        },
        "setup": {"mode": "solo", "ruleset": req.ruleset, "fight_type": req.fight_type,
                  "start_seconds": start_seconds, "round_count": 1},
        "rounds": [{"number": 1, "start_seconds": start_seconds, "end_seconds": end_seconds, "selected": True}],
        "performance": {
            "analysis_seconds": analysis_seconds,
            "segment_duration_seconds": duration,
            "realtime_speed": duration / analysis_seconds if analysis_seconds > 0 else 0.0,
            "within_video_length_budget": analysis_seconds <= duration,
            "analysed_frames": len(samples), "stride": quality.stride, "imgsz": quality.imgsz,
            "pose_model": str(getattr(pose_tracker, "model_path", "")),
        },
        "tracking": {
            "solo": True,
            "fighter_A_coverage": coverage, "fighter_B_coverage": 0.0,
            "fighter_A_seed_source": "pose_detector" if seed is not None else "manual_anchor",
            "initial_iou_A": seed_iou, "analyzed_frames": len(samples),
            "selection_source_frame": samples[seed_index][0],
        },
        "classifier": {"action_classifier": "none - solo sessions count no strikes"},
        "metrics": {"A": {key: measured.get(key) for key in SOLO_METRICS}},
        "scorecard": {"available": False, "status": "solo_session", "totals": {"A": None, "B": None},
                      "rounds": [], "winner_estimate": None,
                      "disclaimer": "A solo session has no opponent, so nothing is scored."},
        "coaching": {"A": {"strengths": [], "improvements": [], "drills": [], "note": note}},
        "training_plan": {"A": []},
        "integrity": {
            "identity_evidence_trusted": trusted, "fighter_identity_trusted": {"A": trusted},
            "action_metrics_trusted": False, "fight_footage_sufficient": True,
            "solo_note": note,
        },
        "statistics": {"fighters": {}, "attempt_counts_available": False, "action_labels_available": False},
        "events": [],
        "analysis_build": build_stamp(),
    }
    # Only the JSON: core.report.write_report renders a two-fighter page, and
    # the web app renders solo reports itself (solo_result.html).
    (job_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("solo_analysis frames=%s coverage=%.3f seed_iou=%.3f seconds=%.2f video=%.2f",
                len(samples), coverage, seed_iou, analysis_seconds, duration)
    progress(99.9, "Finalizing report", duration, "report")
    return report
