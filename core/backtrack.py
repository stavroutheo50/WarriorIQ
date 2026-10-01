"""Follow both fighters backwards from the frame the person picked.

The frame on the fighter-selection page is where the person says who is who.
It used to also be where the analysis *started*: the selection time was stored
as the analysis start, so picking a frame at 0:30 of a 1:56 video analysed only
1:26, and the automatically chosen frame silently cut the opening off too. The
report then said "1 round - 1:25" as though that were the fight.

The selection frame is now an identity seed only. Before the forward pass this
module runs the same detector and tracker over the footage *before* the seed,
in reverse order - BoT-SORT's motion model does not care which way time runs -
and follows each fighter's track back towards the start of the video. The
forward analysis is then seeded at the earliest frame both fighters were
followed to, with the boxes the tracker had for them there.

It stops early, and says why, rather than guess:

  * ``camera_cut``      - the picture changes completely between two samples,
                          so a track cannot honestly be carried across it;
  * ``fighter_lost``    - one of the two was not seen for longer than a
                          tracker can bridge;
  * ``unsure_who_is_who`` - the learned appearance says the two may have
                          swapped places;
  * ``not_detected_at_seed`` - the detector did not find one of the boxes the
                          person drew, so there is no track to follow.

The analyser then checks the result where it matters: when the forward pass
reaches the seed frame, Fighter A must be on the box the person drew for A and
B on B's. If not, the hand-off is discarded and the analysis runs from the seed
frame instead, with the excluded span stated in the report. A wrong identity
before the seed would be worse than no analysis of it.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np

from core.identity import box_iou
from core.reid import pool as reid_pool
from core.reid import similarity as reid_similarity

LOGGER = logging.getLogger("warrioriq.backtrack")

# How often the backward pass looks. A tracker needs neighbouring samples close
# enough that a fighter moves less than a body width between them; ten a second
# is the forward pass's own floor on fast footage.
BACKTRACK_FPS = 10.0
# Longest a fighter may go unseen before the track is no longer trusted. A
# clinch or a referee stepping between them hides someone for a few tenths.
MAX_GAP_SECONDS = 1.0
# Two samples whose colour histograms correlate less than this are different
# shots: an edit, a replay, a cut to the crowd.
CUT_CORRELATION = 0.45
# Cross-matching has to beat straight matching by this much before the two are
# called swapped. Same-person cosine runs 0.93-0.98 and different-person
# 0.56-0.82 on pooled embeddings (core/reid.py), so a real swap clears it by a
# wide margin and two identical kits - where both sides score alike - never do.
SWAP_MARGIN = 0.08
# Boxes overlapping more than this are in a clinch. A hand-off is never made on
# such a frame, because seeding there is exactly where a swap starts.
CLINCH_IOU = 0.30
# How far before the seed is worth following at all. Less than this and the
# analysis simply starts at the seed, as it did before.
MIN_BACKTRACK_SECONDS = 0.5

REASONS = {
    "camera_cut": "the picture cuts to a different shot",
    "fighter_lost": "one of the fighters could not be followed back past it",
    "unsure_who_is_who": "the two fighters could not be told apart reliably before it",
    "not_detected_at_seed": "the fighters were not detected clearly on the frame you picked",
    "unverified": "who was who before it could not be confirmed against the frame you picked",
    "unavailable": "the footage before it could not be processed",
}


@dataclass
class Handoff:
    """Where the forward analysis should start, and with which boxes."""

    seed_frame: int
    requested_start_frame: int
    frame: int
    a_box: list[float] | None = None
    b_box: list[float] | None = None
    reason: str | None = None
    samples: int = 0
    seconds: float = 0.0
    # The seed observations, so the pair can be compared on the frame the
    # person chose rather than wherever the hand-off lands.
    seed_a: object | None = field(default=None, repr=False)
    seed_b: object | None = field(default=None, repr=False)

    @property
    def moved(self) -> bool:
        return self.frame < self.seed_frame and self.a_box is not None and self.b_box is not None

    def as_dict(self, fps: float) -> dict:
        fps = fps if fps > 0 else 30.0
        return {
            "seed_seconds": round(self.seed_frame / fps, 3),
            "requested_start_seconds": round(self.requested_start_frame / fps, 3),
            "reached_seconds": round(self.frame / fps, 3),
            "reason": self.reason,
            "samples": self.samples,
            "seconds_spent": round(self.seconds, 2),
        }


def sample_frames(start_frame: int, seed_frame: int, fps: float) -> list[int]:
    """Frames to look at, seed first, descending to the requested start."""
    if seed_frame <= start_frame:
        return [seed_frame]
    step = max(1, int(round((fps if fps > 0 else 30.0) / BACKTRACK_FPS)))
    frames = list(range(seed_frame, start_frame - 1, -step))
    if frames[-1] != start_frame:
        frames.append(start_frame)
    return frames


def _track(people, track_id):
    for person in people:
        if person.track_id is not None and person.track_id == track_id:
            return person
    return None


def follow_back(
    sequence: Iterable[tuple[int, list]],
    seed_a_box,
    seed_b_box,
    fps: float,
    requested_start_frame: int,
    *,
    find_initial: Callable,
    cuts: set[int] | None = None,
) -> Handoff:
    """Carry the two identities from the seed back through ``sequence``.

    ``sequence`` yields ``(frame_index, people)`` in descending frame order,
    seed first, where ``people`` came from a tracker run in that same order so
    track ids persist across it. ``cuts`` holds frame indices that start a new
    shot (the sample at that index looks nothing like the one before it in
    time), so the track may not be carried from it to anything earlier.

    Pure apart from what ``sequence`` does, which is what makes it testable
    without a model.
    """
    fps = fps if fps > 0 else 30.0
    cuts = cuts or set()
    iterator = iter(sequence)
    try:
        seed_frame, seed_people = next(iterator)
    except StopIteration:
        raise ValueError("follow_back needs at least the seed frame") from None
    seed_a, seed_b, _, _ = find_initial(
        np.asarray(seed_a_box, dtype=np.float32), np.asarray(seed_b_box, dtype=np.float32), seed_people)
    handoff = Handoff(seed_frame=seed_frame, requested_start_frame=requested_start_frame,
                      frame=seed_frame, samples=1, seed_a=seed_a, seed_b=seed_b)
    if seed_a is None or seed_b is None or seed_a.track_id is None or seed_b.track_id is None:
        handoff.reason = "not_detected_at_seed"
        return handoff

    track_a, track_b = seed_a.track_id, seed_b.track_id
    anchors_a = [seed_a.reid] if getattr(seed_a, "reid", None) is not None else []
    anchors_b = [seed_b.reid] if getattr(seed_b, "reid", None) is not None else []
    last_seen = {"A": seed_frame, "B": seed_frame}
    previous_frame = seed_frame
    best = (seed_frame, [float(v) for v in seed_a.box], [float(v) for v in seed_b.box])
    reason = None
    for frame_index, people in iterator:
        handoff.samples += 1
        if previous_frame in cuts:
            # The later sample began a new shot; nothing before it belongs to
            # the same continuous view.
            reason = "camera_cut"
            break
        previous_frame = frame_index
        obs_a, obs_b = _track(people, track_a), _track(people, track_b)
        if obs_a is not None:
            last_seen["A"] = frame_index
        if obs_b is not None:
            last_seen["B"] = frame_index
        if (last_seen["A"] - frame_index) / fps > MAX_GAP_SECONDS or (
                last_seen["B"] - frame_index) / fps > MAX_GAP_SECONDS:
            reason = "fighter_lost"
            break
        if obs_a is None or obs_b is None:
            continue
        pooled_a, pooled_b = reid_pool(anchors_a), reid_pool(anchors_b)
        straight = [reid_similarity(obs_a.reid, pooled_a), reid_similarity(obs_b.reid, pooled_b)]
        cross = [reid_similarity(obs_a.reid, pooled_b), reid_similarity(obs_b.reid, pooled_a)]
        if None not in straight and None not in cross and sum(cross) > sum(straight) + 2 * SWAP_MARGIN:
            reason = "unsure_who_is_who"
            break
        # Pool the first few looks at each fighter, as the identity manager
        # does for its anchor: one crop of a small person describes little.
        if len(anchors_a) < 5 and obs_a.reid is not None:
            anchors_a.append(obs_a.reid)
        if len(anchors_b) < 5 and obs_b.reid is not None:
            anchors_b.append(obs_b.reid)
        if box_iou(obs_a.box, obs_b.box) < CLINCH_IOU:
            best = (frame_index, [float(v) for v in obs_a.box], [float(v) for v in obs_b.box])
    handoff.frame, handoff.a_box, handoff.b_box = best
    if reason is None and handoff.frame > requested_start_frame:
        # Ran out of samples without reaching the start: the last stretch was a
        # clinch or one fighter was briefly missing at the very beginning.
        reason = "fighter_lost" if handoff.frame - requested_start_frame > fps * MAX_GAP_SECONDS else None
    handoff.reason = reason
    return handoff


def _shot_signature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, max(1, int(round(frame.shape[0] * 160 / max(1, frame.shape[1]))))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _buffer_samples(video_path: str, frames_wanted: list[int], workdir: Path,
                    should_continue: Callable[[], bool]) -> tuple[dict[int, Path], set[int]]:
    """Decode the span once, forward, keeping only the sampled frames.

    Kept as JPEG on disk rather than in memory: a minute before the seed at ten
    samples a second is 600 frames, which is 3.7 GB of raw 1080p and about
    150 MB as files. Also finds shot cuts between neighbouring samples.
    """
    wanted = sorted(set(frames_wanted))
    first, last = wanted[0], wanted[-1]
    wanted_set = set(wanted)
    stored: dict[int, Path] = {}
    cuts: set[int] = set()
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            return stored, cuts
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        index = first
        previous_signature = None
        while index <= last:
            if not capture.grab():
                break
            if index in wanted_set:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    break
                path = workdir / f"{index:08d}.jpg"
                if cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    stored[index] = path
                signature = _shot_signature(frame)
                if previous_signature is not None and cv2.compareHist(
                        previous_signature, signature, cv2.HISTCMP_CORREL) < CUT_CORRELATION:
                    # This sample starts a new shot relative to the one before.
                    cuts.add(index)
                previous_signature = signature
                if not should_continue():
                    break
            index += 1
    finally:
        capture.release()
    return stored, cuts


def backtrack(
    video_path: str,
    pose_tracker,
    fps: float,
    requested_start_frame: int,
    seed_frame: int,
    seed_a_box,
    seed_b_box,
    imgsz: int,
    *,
    find_initial: Callable,
    workdir: Path | None = None,
    progress: Callable[[float], None] | None = None,
) -> Handoff:
    """Run the backward pass on a real video with the analysis's own tracker.

    The tracker's state is reset before and after, so nothing learned running
    backwards leaks into the forward pass.
    """
    started = time.perf_counter()
    frames = sample_frames(requested_start_frame, seed_frame, fps)
    if len(frames) < 2 or (seed_frame - requested_start_frame) / max(fps, 1e-6) < MIN_BACKTRACK_SECONDS:
        return Handoff(seed_frame=seed_frame, requested_start_frame=requested_start_frame, frame=seed_frame)
    if workdir is not None:
        # The run's output directory is created by the forward pass, which has
        # not started yet.
        Path(workdir).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".backtrack-", dir=str(workdir) if workdir else None))
    try:
        stored, cuts = _buffer_samples(video_path, frames, scratch, lambda: True)
        if seed_frame not in stored:
            return Handoff(seed_frame=seed_frame, requested_start_frame=requested_start_frame,
                           frame=seed_frame, reason="not_detected_at_seed")
        pose_tracker.reset_tracking()
        total = len(frames)

        def sequence() -> Iterator[tuple[int, list]]:
            for done, index in enumerate(frames):
                path = stored.get(index)
                if path is None:
                    return
                image = cv2.imread(str(path))
                if image is None:
                    return
                if progress is not None:
                    progress(done / max(1, total))
                yield index, pose_tracker.track(image, imgsz)

        def initial(a_box, b_box, people):
            seed_image = cv2.imread(str(stored[seed_frame]))
            return find_initial(a_box, b_box, people, seed_image)

        handoff = follow_back(sequence(), seed_a_box, seed_b_box, fps, requested_start_frame,
                              find_initial=initial, cuts=cuts)
    finally:
        pose_tracker.reset_tracking()
        shutil.rmtree(scratch, ignore_errors=True)
    handoff.seconds = time.perf_counter() - started
    LOGGER.info(
        "backtrack seed_frame=%s reached_frame=%s requested_start=%s reason=%s samples=%s seconds=%.2f",
        seed_frame, handoff.frame, requested_start_frame, handoff.reason, handoff.samples, handoff.seconds)
    return handoff


def seed_verdict(fighter_a, fighter_b, seed_a_box, seed_b_box) -> str:
    """Does the forward pass, arriving at the seed, agree with the person?

    "match" when every fighter present sits on their own drawn box, "swapped"
    when every fighter present sits on the other one's, "unknown" otherwise.
    """
    votes = []
    for observed, own, other in ((fighter_a, seed_a_box, seed_b_box), (fighter_b, seed_b_box, seed_a_box)):
        if observed is None:
            continue
        own_iou, other_iou = box_iou(observed.box, own), box_iou(observed.box, other)
        if own_iou >= 0.2 and own_iou > other_iou:
            votes.append("match")
        elif other_iou >= 0.2 and other_iou > own_iou:
            votes.append("swapped")
        else:
            votes.append("unknown")
    if not votes or "unknown" in votes:
        return "unknown"
    if all(vote == "match" for vote in votes):
        return "match" if len(votes) == 2 else "partial"
    if all(vote == "swapped" for vote in votes):
        return "swapped"
    return "unknown"
