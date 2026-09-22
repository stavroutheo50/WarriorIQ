"""The numbers the browser's pre-upload check compares footage against.

The landing page promises "a quick quality check catches hard-to-see footage
before the analysis starts". Until now the browser checked the file's size and
its duration and nothing else - `videoWidth` was never read - so the check
could not catch the one thing that actually decides whether an analysis is
possible: how big the fighters are in the picture.

The real measurement lives in `core.preflight`, runs on the worker, and needs
the pose model. It therefore happens *after* the upload, which on a phone
connection is several minutes after the filmer could still have done anything
about it. This module carries the subset of its thresholds that a browser can
check on its own, so the same numbers decide the warning before the upload and
the report after it.

Imported rather than restated: a threshold that drifts between the warning and
the analysis is worse than no warning, because it teaches the filmer that the
check is noise.
"""

from __future__ import annotations

from core.preflight import (
    MAX_INFERENCE_SIZE,
    MIN_INFERENCE_SIZE,
    MIN_USABLE_LONG_EDGE,
    MIN_USABLE_SECONDS,
    REFERENCE_SUBJECT_PX,
    TARGET_SUBJECT_PX,
    WELL_FRAMED_SHARE,
)

# Two rows of the detection curve in core/preflight.py, named so the browser
# can quote them. From that table, measured on fight 3 by sweeping only the
# inference size:
#
#     fighter px in net    people found    max conf
#             86               0.7           0.29     <- finds almost nobody
#            120               3.2           0.49
#            171               9.3           0.81
#            214              15.0           0.85     <- finds everybody
#
# Below the floor there is no amount of framing that rescues the clip; above
# the reliable line the detector is doing its job. Between them is the band
# where an analysis runs but drops people, which is what produced the 64-88%
# tracking coverage the audit measured.
DETECTOR_FLOOR_PX = 120
DETECTOR_RELIABLE_PX = 190

# A phone filming a round at 24 fps or less loses fast hands between frames.
# core/preflight.py warns at the same number after the upload.
MIN_USABLE_FPS = 24

# What the motion estimate is allowed to claim. When the moving region covers
# more than this share of the frame, the camera is panning or the crowd is
# moving with the fighters, and the height of that region is not a fighter's
# height. The browser says it could not measure rather than reporting a number
# it would have to invent.
MAX_TRUSTWORTHY_MOTION_SHARE = 0.55
# Under this share there is too little movement in the sample to call it a
# fighter at all - a still frame of an empty mat, or a clip sampled during a
# break between rounds.
#
# Deliberately tiny, and it took a measurement to find out how tiny. This was
# 0.004 on the reasoning that a fight ought to move a good part of the frame.
# It does not: two fighters 8% of the height of a 1080p frame - the far-away
# case this whole check exists to catch - moved between 0.28% and 0.52% of the
# pixels across seven sampled pairs, median 0.39%. So the floor meant to
# exclude an empty mat was instead excluding the exact footage that most needs
# a warning, and doing it silently, by reporting that nothing in the video
# moved.
#
# A truly static clip differs only by compression noise, which the per-pixel
# luma threshold in preflight.js already absorbs, so the real separation is
# between "a few hundred pixels changed" and "almost none did". This sits just
# above the identical-frames floor used there and leaves the discrimination to
# the density test, which is what actually distinguishes a subject from a
# moving background.
MIN_TRUSTWORTHY_MOTION_SHARE = 0.0008


def client_thresholds() -> dict:
    """What the page's pre-upload check needs, as JSON-safe values.

    Deliberately a flat dict of numbers and not a rendered set of sentences:
    the wording belongs next to the markup that shows it, and the numbers
    belong next to the measurement that produced them.
    """
    return {
        "target_subject_px": TARGET_SUBJECT_PX,
        "detector_floor_px": DETECTOR_FLOOR_PX,
        "detector_reliable_px": DETECTOR_RELIABLE_PX,
        "min_inference_size": MIN_INFERENCE_SIZE,
        "max_inference_size": MAX_INFERENCE_SIZE,
        "min_usable_long_edge": MIN_USABLE_LONG_EDGE,
        "min_usable_seconds": MIN_USABLE_SECONDS,
        "min_usable_fps": MIN_USABLE_FPS,
        "well_framed_share": WELL_FRAMED_SHARE,
        "reference_subject_px": list(REFERENCE_SUBJECT_PX),
        "max_motion_share": MAX_TRUSTWORTHY_MOTION_SHARE,
        "min_motion_share": MIN_TRUSTWORTHY_MOTION_SHARE,
    }


def subject_px_in_network(long_edge: int, subject_px: float) -> float:
    """How tall this subject ends up inside the detector's input.

    The same arithmetic as `core.preflight.inference_size_for_subject`, read
    the other way round: that one asks what size to run, this one asks what
    the filmer gets once the size has been clamped. The clamp is the point -
    a subject small enough in the source cannot be brought up to target at any
    size we are willing to run, and the shortfall is what the filmer needs to
    be told about while they can still move closer.
    """
    if long_edge <= 0 or subject_px <= 0:
        return 0.0
    wanted = TARGET_SUBJECT_PX * float(long_edge) / float(subject_px)
    stepped = int(round(wanted / 32.0) * 32)
    size = max(MIN_INFERENCE_SIZE, min(MAX_INFERENCE_SIZE, stepped))
    return float(subject_px) * size / float(long_edge)
