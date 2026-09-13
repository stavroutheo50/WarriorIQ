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

# How much of the opening to skip before looking for action. This was a flat
# 2% of the frame count, which is a couple of seconds on a fight and four
# minutes on a tournament recording - and past about sixteen minutes of
# footage it overran the twenty-second cap above, collapsing the search to a
# two-frame window and handing back whatever sat 2% in. Somebody uploading a
# whole afternoon's recording got no motion search at all. The intent was
# always "skip the walk-on", which is a number of seconds, not a share.
_SKIP_OPENING_SECONDS = 3.0


def _selection_window(info: VideoInfo, search_share: float) -> tuple[int, int]:
    first = max(1, min(int(info.frame_count * 0.02),
                       int(_SKIP_OPENING_SECONDS * info.fps)))
    last = max(first + 2, min(int(info.frame_count * search_share),
                              int(_MAX_SELECTION_SECONDS * info.fps)))
    return first, min(last, max(first + 2, info.frame_count))


def _motion_crop(region):
    """Grey, shrunk and blurred - the three things the motion score needs.

    Shrinking first is not only speed. Block noise in a compressed stream is
    the same order of magnitude as a hand moving and it is everywhere, so the
    blur matters; doing both on a 1080p centre crop costs several times what
    it costs on a small one, and the score is a mean over the crop either way.
    """
    grey = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 320.0 / max(1, grey.shape[1]))
    if scale < 1.0:
        grey = cv2.resize(grey, (max(1, int(grey.shape[1] * scale)),
                                 max(1, int(grey.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(grey, (5, 5), 0)


def _capture_quality(frame) -> tuple[float, float]:
    """Mean brightness and Laplacian variance, at the scale the thresholds assume."""
    height, width = frame.shape[:2]
    scale = min(1.0, 640.0 / max(width, height))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                           interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(grey.mean()), float(cv2.Laplacian(grey, cv2.CV_64F).var())


def _scan_selection_window(path: str | Path, info: VideoInfo, search_share: float,
                           samples: int, want_quality: bool):
    """Walk the opening once, forwards, measuring motion and capture quality.

    This used to be three passes that all seeked: twenty-eight seeks to score
    motion, three more to sample brightness and sharpness, one more to re-read
    whichever frame won. On an MP4 a seek is cheap and nobody noticed. On WebM
    - which is what a browser's own recorder writes, and what several of the
    sample bouts are - OpenCV has no index to seek with, so it decodes from
    frame zero every single time, and the cost grows with how far in the seek
    lands. Measured on a sixty-second 1080p WebM, the three quality samples
    cost 10.0 s and the twenty-eight motion samples 66.4 s: seventy-eight
    seconds of an upload spent re-decoding the same opening thirty-one times
    over, after the last byte had already arrived.

    Reading forward once removes the repetition. Frames between samples go
    through grab(), which advances the decoder without paying for the colour
    conversion and the copy, and that is most of them.

    The quality samples now come from this window rather than from 12%, 50%
    and 88% of the whole file. That is a real reduction in what is measured -
    a bout whose last round is lit differently will not be caught here - and
    it is taken deliberately: reaching 88% of a long WebM means decoding the
    entire file on the web host while somebody waits, and this check exists to
    answer "is this footage usable at all", which the opening answers. Every
    frame is read later anyway, on the GPU, and coverage is reported from
    there.
    """
    first, last = _selection_window(info, search_share)
    step = max(1, (last - first) // max(1, samples))
    # Brightness and sharpness want the window's span, not its busiest part,
    # so they are taken at a few points spread across whatever we walk.
    quality_at: set[int] = set()
    if want_quality:
        span = max(1, last - first - 2)
        quality_at = {first + int(span * ratio) for ratio in (0.0, 0.5, 1.0)}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], []
    scored: list[tuple[int, float]] = []
    quality: list[tuple[float, float]] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        index = first
        pending_index: int | None = None
        pending_crop = None
        while index < last:
            starts_pair = (index - first) % step == 0
            if not (starts_pair or index == pending_index or index in quality_at):
                if not cap.grab():
                    break
                index += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index in quality_at:
                quality.append(_capture_quality(frame))
            height, width = frame.shape[:2]
            top, bottom = int(height * 0.30), int(height * 0.75)
            left, right = int(width * 0.25), int(width * 0.75)
            if bottom > top and right > left:
                crop = _motion_crop(frame[top:bottom, left:right])
                if pending_crop is not None and index == pending_index:
                    scored.append((index - 1, float(cv2.absdiff(pending_crop, crop).mean())))
                    pending_crop, pending_index = None, None
                elif starts_pair:
                    # A pair is scored against the frame straight after it, so
                    # the next read is spoken for.
                    pending_crop, pending_index = crop, index + 1
            index += 1
    except cv2.error:
        return scored, quality
    finally:
        cap.release()
    return scored, quality


def _pick_from_scores(scored: list[tuple[int, float]]) -> int:
    if not scored:
        return 0
    # The earliest moment that is clearly action, not the busiest one.
    # Everything before the pick goes unanalysed, so a frame at 17s that is
    # marginally livelier than one at 6s is a bad trade: it costs ten seconds
    # of a fight to gain nothing the seeding can use.
    best = max(score for _, score in scored)
    return next(index for index, score in scored if score >= best * 0.70)


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
    scored, _ = _scan_selection_window(path, info, search_share, samples, want_quality=False)
    return _pick_from_scores(scored)


def probe_upload(path: str | Path, info: VideoInfo, search_share: float = 0.25,
                 samples: int = 28) -> tuple[int, list[tuple[float, float]]]:
    """Everything the upload needs to know about a file, in one read of it.

    Returns the frame to open the fighter picker on, and the capture-quality
    samples behind the usability verdict. They are gathered together because
    they come from the same frames, and reading the file twice to collect them
    separately is what made an upload wait.
    """
    if info.fps <= 0 or info.frame_count <= 2:
        return 0, []
    scored, quality = _scan_selection_window(path, info, search_share, samples, want_quality=True)
    return _pick_from_scores(scored), quality


def read_frame(path: str | Path, frame_index: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    target = max(0, int(frame_index))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame

        # Seeking by frame index is unreliable on exactly the files people
        # upload from a phone: WebM written by the browser's MediaRecorder, and
        # variable-frame-rate camera footage. The seek reports success, lands
        # nowhere, and the read fails. Walking the file always works, so fall
        # back to that rather than losing the upload. Only reached when a seek
        # actually failed, so the cost is paid by broken files alone.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = None
        for _ in range(target + 1):
            ok, candidate = cap.read()
            if not ok or candidate is None:
                break
            frame = candidate
        if frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {path}")
        return frame
    finally:
        cap.release()


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
