"""Guard, decided once per frame and summarised once per fighter.

QA, 2026-10-07: one report said "Guard 27%" in the top card, the comparison
bars, "Work on" and the plan, while its own Defence table said "Hands up by the
face 0%" and "Longest with hands down 20 s" - and the opponent's 45% sat beside
"0%" and "0 s". All three figures came from the same per-frame reading
(core/metrics.py) and were each summarised differently:

  * the card was the *average* of a 0..1 closeness-of-hand-to-head reading,
    printed as if it were a share of the round;
  * "Hands up" was the share of frames with that reading at 0.5 or above;
  * "Longest with hands down" was the longest run below a *different* line,
    0.3, so a fighter whose hands sat between the two was neither up nor down;
  * the guard-drop markers were simply the four lowest readings, whether or
    not the hands were ever down.

So every frame is now classified once - hands up, or not - against one line,
and every guard figure is derived from that classification: the share (which
is ``guard_index``), the seconds up and down, the longest stretch down, the
moments the guard dropped, and the block-level spread coaching compares on.
Nothing else computes a guard figure.

**The reading itself** (QA, 2026-10-07: the same clip gave 27% / 45% at
1280x720 and 18% / 18% at 160x90, with no warning) is the distance from the
farther of the two wrists to the chin (both hands must be up; audit
2026-10-14, #5), over shoulder width - so it is in the fighter's own
proportions, not the picture's - and only on frames where the face, both
shoulders, the hips and a wrist were found confidently and the body is big
enough in the picture for a wrist to be placed. Each reading is then smoothed
over a fraction of a second before it is classified, so one misplaced wrist
does not count as a dropped guard. Below those floors guard is "Not measured",
with the reason, rather than a number made of pose-model noise.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

# Stored on each fighter's metrics so a report says which guard it carries.
# Reports without it were made by the old, inconsistent summaries; see
# reconcile_report_guard.
GUARD_DEFINITION = "hands_up_share/3"
# Consistent summaries, all of them: /1 classified the old body-length
# reading, /2 the nearer wrist to the chin, /3 both wrists to the chin (audit
# 2026-10-14, #5). Only reports made before any of them are reconciled; the
# older two stay on their own reports and out of trends (comparable_guard).
CONSISTENT_DEFINITIONS = frozenset({"hands_up_share/1", "hands_up_share/2", GUARD_DEFINITION})
LEGACY_DEFINITION = "legacy_hands_up_share"

# ---- The per-frame reading -------------------------------------------------
# COCO keypoints have no chin. It is placed this far from the head towards the
# middle of the shoulders: a nose sits roughly a third of the way from the top
# of the head to the shoulder line in front view, and the chin about the same
# again below it.
CHIN_FRACTION = 0.35
# Keypoint confidence each part must reach before it is used.
FACE_CONFIDENCE = 0.4
SHOULDER_CONFIDENCE = 0.5
HIP_CONFIDENCE = 0.3
WRIST_CONFIDENCE = 0.5
# Side-on, the two shoulders overlap in the picture and their distance falls
# towards zero, which would make every hand look far from the face. The scale
# is therefore the larger of the measured shoulder width and the width an
# upright body of this torso length has square-on (about 0.8 of the
# shoulder-to-hip length). On the repository's real kick-light track the
# fighters stand side-on: measured shoulder width was a median 0.39 of torso.
UPRIGHT_SHOULDER_WIDTH = 0.8
# The scale, in pixels, below which a wrist cannot be placed well enough.
# Measured by shrinking the real track and rounding keypoints to whole pixels
# with 0.6 px of pose jitter: the hands-up share held within about 3 points
# down to an 8 px scale, drifted by up to 8 points at 4 px, and broke at
# 2.5 px - which is where a 160x90 copy of a 1280x720 fight sits.
MIN_SCALE_PX = 8.0
# Both wrists within this many shoulder widths of the chin: hands up.
# Audit 2026-10-14 (#5): the nearer-wrist reading counted a guard with one
# hand up and the other at the hip; a guard is both hands. Judged on the
# farther wrist, the line is looser than the old nearer-wrist 0.7 (about
# 28 cm): 0.9 is about 36 cm on an adult, which allows the lead hand held out
# in front of the face. A definition, not a validated norm - no labelled
# guard data exists - which is why guard is shown as Experimental
# (core/metric_catalog.py).
HANDS_UP = 0.9
# Readings are the median over +-this many seconds before they are
# classified, so one misplaced wrist is not a dropped guard.
SMOOTH_SECONDS = 0.25

_FACE = (0, 1, 2, 3, 4)

# Why a frame gave no reading.
TOO_SMALL = "too_small"
UNCLEAR = "unclear"
NOT_MEASURED_TEXT = {
    TOO_SMALL: ("Not measured: the fighter is too small in this video for a wrist to be placed "
                "reliably. Film from closer, or upload the original file rather than a smaller copy."),
    UNCLEAR: ("Not measured: the face, shoulders or wrists were not seen clearly often enough "
              "to tell where the hands were."),
}
NOT_MEASURED_SHORT = {TOO_SMALL: "too small in the picture", UNCLEAR: "hands or face not seen clearly"}


def reading(keypoints, confidence) -> tuple[float | None, str | None]:
    """Farther wrist to chin, over shoulder width, for one frame.

    Returns (reading, None), or (None, why) where ``why`` is TOO_SMALL or
    UNCLEAR. Lower readings are hands nearer the face.
    """
    if keypoints is None or confidence is None or len(confidence) < 17 or len(keypoints) < 17:
        return None, UNCLEAR
    points = np.asarray(keypoints, dtype=np.float32)[:, :2]
    conf = np.asarray(confidence, dtype=np.float32)

    def seen(index: int, floor: float) -> bool:
        return float(conf[index]) >= floor and float(points[index][0]) > 0 and float(points[index][1]) > 0

    if not (seen(5, SHOULDER_CONFIDENCE) and seen(6, SHOULDER_CONFIDENCE)
            and seen(11, HIP_CONFIDENCE) and seen(12, HIP_CONFIDENCE)):
        return None, UNCLEAR
    face = [points[i] for i in _FACE if seen(i, FACE_CONFIDENCE)]
    wrists = [points[i] for i in (9, 10) if seen(i, WRIST_CONFIDENCE)]
    # Both hands, or no reading: a frame with one wrist unseen cannot say
    # whether the guard was up, and guessing from one hand is what made the
    # old figure meaningless.
    if not face or len(wrists) < 2:
        return None, UNCLEAR
    neck = (points[5] + points[6]) / 2.0
    torso = float(np.linalg.norm(neck - (points[11] + points[12]) / 2.0))
    scale = max(float(np.linalg.norm(points[5] - points[6])), UPRIGHT_SHOULDER_WIDTH * torso)
    if scale < MIN_SCALE_PX:
        return None, TOO_SMALL
    head = np.mean(np.asarray(face, dtype=np.float32), axis=0)
    chin = head + CHIN_FRACTION * (neck - head)
    # The farther wrist: hands up only when both are by the chin.
    return float(max(np.linalg.norm(wrist - chin) for wrist in wrists)) / scale, None


def not_measured_reason(rejected: Counter | dict) -> str:
    """The reason guard was not measured, from per-frame rejection counts."""
    counts = Counter(rejected or {})
    return TOO_SMALL if counts.get(TOO_SMALL, 0) >= counts.get(UNCLEAR, 0) and counts.get(TOO_SMALL) else UNCLEAR

# A sample stands for the time until the next one, never for more than this:
# across a longer gap nothing was measured. The same rule as
# core/fight_numbers.py, so guard seconds and movement seconds agree.
MAX_SAMPLE_SECONDS = 0.5

# Moments chosen as evidence are at least this far apart, so one exchange is
# not cited four times.
MOMENT_SPACING_SECONDS = 3.0
MOMENT_LIMIT = 4

# Coaching compares fighters over two-second blocks (core/metrics.py _spread).
BLOCK_SECONDS = 2.0

# Shown where an older report's guard figure cannot be recovered. They name
# the engine that made the report: "older report" said nothing a fighter could
# check, and was said of a report three days old (QA, 2026-10-07).
# Analysis engine version that first measured the guard this way
# (core/build_info.py ANALYSIS_VERSION 4).
GUARD_ENGINE_VERSION = 5
LEGACY_NOTE = ("Not measured in this report: it was made by {engine}, before the guard was measured "
               "consistently (engine v{current}). Re-run the analysis to measure it.")
LEGACY_PARTIAL_NOTE = ("This report was made by {engine}, before the guard was measured consistently "
                       "(engine v{current}), so only the share of time with hands up is shown. Re-run "
                       "the analysis for the longest stretch with hands down and the moments the guard "
                       "dropped.")
LEGACY_SHORT_NOTE = "made by {engine} — re-run to measure"


# The same notes for a fight whose video is gone (removed after the retention
# period, or deleted by its owner): "Re-run the analysis" is impossible without
# it, so the notes do not offer it (QA, 2026-10-11).
NO_VIDEO_NOTE = ("Not measured in this report: it was made by {engine}, before the guard was measured "
                 "consistently (engine v{current}). The video for this fight has been removed, so it "
                 "cannot be measured again.")
NO_VIDEO_PARTIAL_NOTE = ("This report was made by {engine}, before the guard was measured consistently "
                         "(engine v{current}), so only the share of time with hands up is shown. The video "
                         "for this fight has been removed, so the rest cannot be measured again.")
NO_VIDEO_SHORT_NOTE = "made by {engine} — not measured"


def comparable_guard(own: dict | None) -> float | None:
    """This fighter's guard figure, only when it was measured the current way.

    For comparing fights. The guard has been measured three different ways
    (legacy averages, /1 a body-length reading, /2 wrist-to-chin), and a trend
    across them printed 59%, 13.6%, 7%, 2% for similar footage (QA,
    2026-10-11): the change was the ruler, not the fighter. A figure from an
    older definition stays on its own report but is left out of trends,
    comparisons and "since your last fight".
    """
    own = own or {}
    value = own.get("guard_index")
    if value is None or own.get("guard_definition") != GUARD_DEFINITION:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def legacy_notes(report: dict) -> tuple[str, str, str]:
    """(full note, partial note, short note) naming the engine that made ``report``.

    ``report["video_removed"]`` (set where the report is rendered) drops the
    advice to re-run, which needs the video.
    """
    from core.build_info import engine_name

    engine = engine_name(report)
    fill = {"engine": engine, "current": GUARD_ENGINE_VERSION}
    short_engine = engine.replace("analysis engine", "engine")
    if report.get("video_removed"):
        return (NO_VIDEO_NOTE.format(**fill), NO_VIDEO_PARTIAL_NOTE.format(**fill),
                NO_VIDEO_SHORT_NOTE.format(engine=short_engine))
    return (LEGACY_NOTE.format(**fill), LEGACY_PARTIAL_NOTE.format(**fill),
            LEGACY_SHORT_NOTE.format(engine=short_engine))


def is_up(reading) -> bool | None:
    """Hands up by the face on this frame, or None when it was not measured."""
    if reading is None:
        return None
    return float(reading) <= HANDS_UP


def _smoothed(times: list[float], values: list[float]) -> list[float]:
    """Each reading replaced by the median of those within SMOOTH_SECONDS."""
    stamps = np.asarray(times, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    low = np.searchsorted(stamps, stamps - SMOOTH_SECONDS, side="left")
    high = np.searchsorted(stamps, stamps + SMOOTH_SECONDS, side="right")
    return [float(np.median(data[a:b])) for a, b in zip(low, high)]


def _weights(times: list[float]) -> list[float]:
    out = []
    for index, moment in enumerate(times):
        if index + 1 < len(times):
            out.append(min(MAX_SAMPLE_SECONDS, max(0.0, times[index + 1] - moment)))
        else:
            out.append(0.0)
    if len(out) >= 2:
        out[-1] = float(np.median(out[:-1]))
    return out


def _runs(times: list[float], weights: list[float], states: list[bool], want: bool) -> list[tuple[float, float]]:
    """(start, seconds) of each unbroken stretch in state ``want``."""
    runs: list[tuple[float, float]] = []
    start = None
    total = 0.0
    previous = None
    for moment, weight, state in zip(times, weights, states):
        broken = previous is not None and moment - previous > MAX_SAMPLE_SECONDS
        if start is not None and (state != want or broken):
            runs.append((start, total))
            start = None
        if state == want:
            if start is None:
                start, total = moment, 0.0
            total += weight
        previous = moment
    if start is not None:
        runs.append((start, total))
    return runs


def _spaced(runs: list[tuple[float, float]]) -> list[float]:
    """Starts of the longest stretches, spread out, in time order."""
    picked: list[float] = []
    for start, _seconds in sorted(runs, key=lambda run: (-run[1], run[0])):
        if any(abs(start - chosen) < MOMENT_SPACING_SECONDS for chosen in picked):
            continue
        picked.append(round(float(start), 2))
        if len(picked) >= MOMENT_LIMIT:
            break
    return sorted(picked)


def summarise(readings) -> dict | None:
    """Every guard figure for one fighter, from (seconds, reading) pairs.

    Readings of None are frames where the wrists or head were not measured and
    are left out. None when nothing was measured.
    """
    pairs = sorted((float(t), float(v)) for t, v in readings if v is not None)
    if not pairs:
        return None
    times = [t for t, _ in pairs]
    states = [bool(is_up(v)) for v in _smoothed(times, [v for _, v in pairs])]
    weights = _weights(times)
    measured = float(sum(weights))
    up_seconds = float(sum(w for w, up in zip(weights, states) if up))
    if measured > 0:
        share = up_seconds / measured
    else:
        # A single frame: the share of frames is all there is.
        share = float(sum(states)) / len(states)
    down_runs = _runs(times, weights, states, False)
    up_runs = _runs(times, weights, states, True)

    blocks: dict[int, list[float]] = {}
    for moment, up in zip(times, states):
        blocks.setdefault(int(moment // BLOCK_SECONDS), []).append(1.0 if up else 0.0)
    means = [float(np.mean(values)) for values in blocks.values()]
    spread = ({"blocks": len(means), "standard_error": float(np.std(means, ddof=1) / np.sqrt(len(means)))}
              if len(means) >= 2 else None)

    return {
        "definition": GUARD_DEFINITION,
        "share": round(share, 3),
        "samples": len(pairs),
        "measured_seconds": round(measured, 1),
        "up_seconds": round(up_seconds, 1),
        "down_seconds": round(measured - up_seconds, 1),
        "longest_down_seconds": round(max((seconds for _, seconds in down_runs), default=0.0), 1),
        # When the hands came down and when they were up, longest first.
        "drops": _spaced(down_runs),
        "held": _spaced(up_runs),
        "spread": spread,
    }


def consistency_errors(guard_share, longest_down, measured_seconds, tolerance: float = 0.15) -> list[str]:
    """Ways the guard figures for one fighter contradict each other.

    Empty when they agree. Used by the tests as the invariant, and cheap
    enough to check anywhere a report is assembled.
    """
    if guard_share is None or longest_down is None or measured_seconds is None:
        return []
    errors = []
    down = (1.0 - float(guard_share)) * float(measured_seconds)
    if float(longest_down) > down + tolerance:
        errors.append(f"longest hands-down {longest_down}s exceeds the {down:.1f}s the hands were down")
    if float(guard_share) >= 1.0 and float(longest_down) > 0:
        errors.append("hands up the whole time, yet a hands-down stretch is reported")
    if float(guard_share) <= 0.0 and float(measured_seconds) > tolerance and float(longest_down) <= 0:
        errors.append("hands never up, yet no hands-down stretch is reported")
    return errors


def reconcile_report_guard(report: dict) -> dict:
    """Bring a stored report's guard figures onto one definition, in place.

    Reports made with GUARD_DEFINITION are left alone. Older reports stored the
    average reading as ``guard_index`` beside a hands-up share and a hands-down
    run measured against different lines. Their per-frame readings were not
    kept, so the one figure that can be recovered is the hands-up share: it is
    used for every guard figure, and the rest - the longest stretch down, the
    drop moments, the spread - are withheld with a reason rather than shown
    disagreeing with it. Idempotent.
    """
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return report
    full_note, partial_note, short_note = legacy_notes(report)
    for fighter in ("A", "B"):
        own = metrics.get(fighter)
        if not isinstance(own, dict):
            continue
        if own.get("guard_definition") in CONSISTENT_DEFINITIONS | {LEGACY_DEFINITION}:
            continue
        if "guard_index" not in own and not own.get("numbers"):
            continue
        numbers = own.get("numbers") if isinstance(own.get("numbers"), dict) else None
        share = numbers.get("hands_up_share") if numbers else None
        own["guard_index"] = None if share is None else float(share)
        own["guard_definition"] = LEGACY_DEFINITION
        own["guard_note"] = full_note if share is None else partial_note
        own["guard_note_short"] = short_note
        if numbers is not None:
            numbers["longest_hands_down_seconds"] = None
        moments = own.get("moments")
        if isinstance(moments, dict):
            moments.pop("guard_index", None)
        spread = own.get("spread")
        if isinstance(spread, dict):
            spread.pop("guard_index", None)
        availability = own.get("availability")
        if isinstance(availability, dict) and share is None:
            availability["guard"] = {**(availability.get("guard") or {}), "available": False,
                                     "reason": full_note}
    return report
