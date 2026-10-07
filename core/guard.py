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
"""

from __future__ import annotations

import numpy as np

# Stored on each fighter's metrics so a report says which guard it carries.
# Reports without it were made by the old, inconsistent summaries; see
# reconcile_report_guard.
GUARD_DEFINITION = "hands_up_share/1"
LEGACY_DEFINITION = "legacy_hands_up_share"

# The per-frame reading (core/metrics.py) at or above which the hands count as
# up by the face. The line "Hands up by the face" already used, so the figure
# a fighter has seen in the Defence table keeps its meaning.
HANDS_UP = 0.5

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

# Shown where an older report's guard figure cannot be recovered.
LEGACY_NOTE = ("Not measured in this report: it was made before the guard was measured "
               "consistently. Re-run the analysis to measure it.")
LEGACY_PARTIAL_NOTE = ("This report was made before the guard was measured consistently, so only "
                       "the share of time with hands up is shown. Re-run the analysis for the "
                       "longest stretch with hands down and the moments the guard dropped.")


def is_up(reading) -> bool | None:
    """Hands up by the face on this frame, or None when it was not measured."""
    if reading is None:
        return None
    return float(reading) >= HANDS_UP


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
    states = [bool(is_up(v)) for _, v in pairs]
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
    for fighter in ("A", "B"):
        own = metrics.get(fighter)
        if not isinstance(own, dict):
            continue
        if own.get("guard_definition") in (GUARD_DEFINITION, LEGACY_DEFINITION):
            continue
        if "guard_index" not in own and not own.get("numbers"):
            continue
        numbers = own.get("numbers") if isinstance(own.get("numbers"), dict) else None
        share = numbers.get("hands_up_share") if numbers else None
        own["guard_index"] = None if share is None else float(share)
        own["guard_definition"] = LEGACY_DEFINITION
        own["guard_note"] = LEGACY_NOTE if share is None else LEGACY_PARTIAL_NOTE
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
                                     "reason": LEGACY_NOTE}
    return report
