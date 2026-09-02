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


# How far into a fight the automatic frame pick may reach. Everything before
# the chosen frame goes unanalysed, so this is a budget, not a preference.
_MAX_SELECTION_SECONDS = 20.0


def pick_selection_frame(path: str | Path, info: VideoInfo, search_share: float = 0.25,
                         samples: int = 28) -> int:
    """Pick an opening frame where the two fighters are working, not posed.

    Frame 0 was the default, and it is the worst frame in a fight video. A
    round starts with the referee standing between the fighters with both arms
    out: the largest, most central, highest-confidence person on the mat. A
    selection box drawn there lands on the referee. Measured on 3.mp4 frame 0,
    the referee detects at 0.87 confidence and fighter A at 0.37, and seeding
    from that frame tracked the referee for the whole bout - 81.6% coverage on
    a person who never threw a strike, while the real fighters track at 96%.

    Motion is the proxy for "the referee has stepped away and these two are
    fighting". No model is involved: this runs on the web host, which has no
    GPU. Only the middle of the mat is measured, so a crowd shifting in their
    seats and a scoreboard ticking over do not outvote the fight.

    The frame chosen here is also where the analysis begins, so a later pick
    costs real footage. The search is therefore capped at both a share of the
    video and a hard twenty seconds - on a four-minute bout the quarter-share
    alone chose 56s, which would have thrown away most of a round. Twenty
    seconds of a fight's opening is walk-on, glove touch and instructions;
    past that the cost outweighs a marginally cleaner frame.

    Returns a frame index; frame 0 on anything it cannot read, which is no
    worse than the behaviour this replaces.
    """
    if info.fps <= 0 or info.frame_count <= 2:
        return 0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    try:
        first = max(1, int(info.frame_count * 0.02))
        last = max(first + 2, min(int(info.frame_count * search_share),
                                  int(_MAX_SELECTION_SECONDS * info.fps)))
        step = max(1, (last - first) // max(1, samples))
        scored: list[tuple[int, float]] = []
        for index in range(first, last, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok_a, frame_a = cap.read()
            ok_b, frame_b = cap.read()
            if not ok_a or not ok_b or frame_a is None or frame_b is None:
                continue
            height, width = frame_a.shape[:2]
            top, bottom = int(height * 0.30), int(height * 0.75)
            left, right = int(width * 0.25), int(width * 0.75)
            if bottom <= top or right <= left:
                continue
            crop_a = cv2.cvtColor(frame_a[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            crop_b = cv2.cvtColor(frame_b[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            # Blur first: block noise in a compressed 480p stream is the same
            # order of magnitude as a hand moving, and it is everywhere.
            crop_a = cv2.GaussianBlur(crop_a, (5, 5), 0)
            crop_b = cv2.GaussianBlur(crop_b, (5, 5), 0)
            scored.append((index, float(cv2.absdiff(crop_a, crop_b).mean())))
        if not scored:
            return 0
        # The earliest moment that is clearly action, not the busiest one.
        # Everything before the pick goes unanalysed, so a frame at 17s that
        # is marginally livelier than one at 6s is a bad trade: it costs ten
        # seconds of a fight to gain nothing the seeding can use.
        best = max(score for _, score in scored)
        return next(index for index, score in scored if score >= best * 0.70)
    except cv2.error:
        return 0
    finally:
        cap.release()


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

    # Never analyse less of the video than the person uploaded. The round
    # numbers are a guess about the fight's shape, and when they fall short the
    # tail was simply dropped without saying so: a nine-minute bout entered as
    # 3 x 2 min had three of its nine minutes thrown away, and nothing in the
    # report mentioned it. Rounds decide where the round lines fall; they do
    # not decide how much footage is worth looking at.
    #
    # Only when every round is selected. Someone who deliberately asked for
    # round 2 of 5 means it, and their choice is left exactly as entered.
    if rounds and all(spec.selected for spec in rounds):
        end = info.duration if req.end_seconds is None else min(info.duration, req.end_seconds)
        if rounds[-1].end_seconds < end:
            rounds[-1].end_seconds = max(rounds[-1].start_seconds, end)

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
