from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

from core.types import AnalysisRequest, RoundSpec, VideoInfo


def get_video_info(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fight video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(str(path), fps, frame_count, width, height, duration)


def read_frame(path: str | Path, frame_index: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return frame


def build_round_schedule(req: AnalysisRequest, info: VideoInfo) -> list[RoundSpec]:
    """Build active round windows from user-entered fight format.

    The user can choose all rounds or a subset. Breaks are intentionally not
    scored, but the tracker may still sample them at a cheaper rate so A/B
    identity survives into the next round.
    """
    start = max(0.0, float(req.start_seconds))
    selected = set(req.selected_rounds or range(1, int(req.round_count) + 1))
    rounds: list[RoundSpec] = []

    cursor = start
    for number in range(1, max(1, int(req.round_count)) + 1):
        round_start = cursor
        round_end = min(info.duration, round_start + max(1.0, float(req.round_duration_seconds)))
        if req.end_seconds is not None:
            round_end = min(round_end, float(req.end_seconds))
        rounds.append(RoundSpec(number, round_start, round_end, number in selected))
        cursor = round_end + max(0.0, float(req.break_duration_seconds))
        if cursor >= info.duration or (req.end_seconds is not None and cursor >= req.end_seconds):
            break

    # If the declared rounds do not cover the video and the user asked for a
    # single generic round, cover the requested segment instead of silently
    # dropping footage.
    if len(rounds) == 1 and req.round_count == 1:
        end = info.duration if req.end_seconds is None else min(info.duration, req.end_seconds)
        rounds[0].end_seconds = max(rounds[0].start_seconds, end)

    return rounds


def round_at_time(rounds: Iterable[RoundSpec], seconds: float) -> RoundSpec | None:
    for spec in rounds:
        if spec.start_seconds <= seconds < spec.end_seconds:
            return spec
    return None


def requested_segment_end(req: AnalysisRequest, info: VideoInfo, rounds: list[RoundSpec]) -> float:
    if req.end_seconds is not None:
        return min(info.duration, max(req.start_seconds, float(req.end_seconds)))
    if rounds:
        return min(info.duration, max(r.end_seconds for r in rounds))
    return info.duration
