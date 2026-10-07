"""Plain numbers about how a fight went, from the tracked bodies and the counted strikes.

Everything here is a count of seconds, a distance or a count of strikes,
worked out from what the analysis already measured frame by frame. Nothing is
graded and nothing is inferred beyond the geometry: "in the middle" means the
inner half of the area the two fighters actually used, "moving forward" means
moving towards the opponent, and so on. Seconds are seconds the fighter was
seen; the report says how many that was.
"""

from __future__ import annotations

import numpy as np

from core import guard as guard_measure

# A sample stands for the time until the next one, but never for more than
# this: across a gap where the fighter was not seen, nothing was measured.
MAX_SAMPLE_SECONDS = 0.5
# The inner half of the area used, by distance from its middle.
CENTRE_ZONE = 0.5
# Slower than this, in body lengths a second, is standing, not moving.
MOVING_SPEED = 0.35
# Direction of a step relative to the opponent (cosine): this far towards
# them is forward, this far away is backward, anything between is circling.
DIRECTIONAL = 0.5
# Distance between the two fighters, in body lengths.
CLOSE_RANGE = 0.9
LONG_RANGE = 1.8
# Balance readings are 0..1 (core/metrics.py). Below this the fighter is off
# balance. Guard is not decided here: core/guard.py classifies every frame
# once, and the hands-up share and longest stretch down below are its figures.
OFF_BALANCE = 0.45
OFF_BALANCE_SECONDS = 0.5
BUSY_WINDOW_SECONDS = 10.0
# movement_numbers' default: work the guard out from the samples.
_FROM_SAMPLES = object()


def _durations(times: list[float]) -> list[float]:
    """How long each sample stands for: until the next one, capped."""
    out = []
    for index, moment in enumerate(times):
        following = times[index + 1] if index + 1 < len(times) else None
        step = (following - moment) if following is not None else 0.0
        out.append(min(MAX_SAMPLE_SECONDS, max(0.0, step)) if following is not None else 0.0)
    if len(out) >= 2:
        out[-1] = float(np.median(out[:-1]))
    return out


def _runs(times: list[float], flags: list[bool], minimum: float) -> int:
    """Separate stretches where the flag held for at least `minimum` seconds."""
    count, start, previous = 0, None, None
    for moment, flag in zip(times + [None], flags + [False]):
        broken = moment is not None and previous is not None and moment - previous > MAX_SAMPLE_SECONDS
        if start is not None and (not flag or broken):
            if previous - start >= minimum:
                count += 1
            start = None
        if flag and moment is not None and start is None:
            start = moment
        previous = moment
    return count


def movement_numbers(samples: list[dict], middle: np.ndarray | None, radius: float | None,
                     round_of=None, guard=_FROM_SAMPLES) -> dict | None:
    """Seconds and distances for one fighter from their per-frame samples.

    Each sample: {"t", "x", "y", "body", "guard", "balance", "toward",
    "speed", "gap"} where toward is the cosine between the step and the
    direction of the opponent, speed is body lengths a second and gap the
    distance to the opponent in body lengths (None when not measured).
    `middle` and `radius` are the area the fight used (core.metrics.ring_frame).
    `round_of(seconds)` gives the round number, or None in a break.
    `guard` is this fighter's core.guard.summarise() result, worked out from
    the same samples when not passed in. None means guard was withheld (too
    few readings passed core/guard.py's gates), and every guard figure here
    is None with it - not recomputed from the few frames that did pass.
    """
    if len(samples) < 2:
        return None
    samples = sorted(samples, key=lambda item: item["t"])
    times = [float(item["t"]) for item in samples]
    weights = _durations(times)
    seen = sum(weights)
    if seen <= 0:
        return None

    def seconds(flags) -> float:
        return round(float(sum(w for w, flag in zip(weights, flags) if flag)), 1)

    centre = None
    if middle is not None and radius:
        inside = [float(np.hypot(item["x"] - middle[0], item["y"] - middle[1])) <= CENTRE_ZONE * radius
                  for item in samples]
        centre = seconds(inside)
    else:
        inside = [False] * len(samples)

    moving = [item.get("speed") is not None and item["speed"] >= MOVING_SPEED for item in samples]
    toward = [item.get("toward") for item in samples]
    forward = seconds([m and t is not None and t >= DIRECTIONAL for m, t in zip(moving, toward)])
    backward = seconds([m and t is not None and t <= -DIRECTIONAL for m, t in zip(moving, toward)])
    circling = seconds([m and t is not None and -DIRECTIONAL < t < DIRECTIONAL for m, t in zip(moving, toward)])

    gaps = [item.get("gap") for item in samples]
    ranges = {
        "close": seconds([g is not None and g < CLOSE_RANGE for g in gaps]),
        "middle": seconds([g is not None and CLOSE_RANGE <= g < LONG_RANGE for g in gaps]),
        "long": seconds([g is not None and g >= LONG_RANGE for g in gaps]),
    }

    distance = 0.0
    for previous, current in zip(samples, samples[1:]):
        if current["t"] - previous["t"] <= MAX_SAMPLE_SECONDS:
            body = max(1.0, (previous["body"] + current["body"]) / 2.0)
            distance += float(np.hypot(current["x"] - previous["x"], current["y"] - previous["y"])) / body

    if guard is _FROM_SAMPLES:
        guard = guard_measure.summarise((item["t"], item.get("guard")) for item in samples)
    balances = [item.get("balance") for item in samples]

    # Tiring: how fast they moved in the second half of the time they were
    # seen, against the first.
    half = times[0] + (times[-1] - times[0]) / 2.0
    first = [item["speed"] for item in samples if item["t"] < half and item.get("speed") is not None]
    second = [item["speed"] for item in samples if item["t"] >= half and item.get("speed") is not None]
    pace_change = None
    if len(first) >= 10 and len(second) >= 10 and np.mean(first) > 1e-6:
        pace_change = round(float((np.mean(second) - np.mean(first)) / np.mean(first) * 100.0))

    rounds: dict[int, dict] = {}
    if round_of is not None:
        for item, weight, in_middle in zip(samples, weights, inside):
            number = round_of(item["t"])
            if number is None:
                continue
            row = rounds.setdefault(int(number), {"seen_seconds": 0.0, "centre_seconds": 0.0})
            row["seen_seconds"] += weight
            if in_middle:
                row["centre_seconds"] += weight
        for row in rounds.values():
            row["seen_seconds"] = round(row["seen_seconds"], 1)
            row["centre_seconds"] = round(row["centre_seconds"], 1) if centre is not None else None

    return {
        "seen_seconds": round(seen, 1),
        "centre_seconds": centre,
        "forward_seconds": forward,
        "backward_seconds": backward,
        "circling_seconds": circling,
        "standing_seconds": seconds([not m for m in moving]),
        "range_seconds": ranges,
        "distance_body_lengths": round(distance, 1),
        # The same figures as the guard card and coaching (core/guard.py).
        "hands_up_share": guard["share"] if guard else None,
        "longest_hands_down_seconds": guard["longest_down_seconds"] if guard else None,
        "off_balance_count": (_runs(times, [b is not None and b < OFF_BALANCE for b in balances], OFF_BALANCE_SECONDS)
                              if any(b is not None for b in balances) else None),
        "pace_change_percent": pace_change,
        "rounds": {str(number): row for number, row in sorted(rounds.items())},
    }


def output_numbers(strike_times: list[float], opponent_times: list[float],
                   round_bounds: list[tuple[int, float, float]]) -> dict:
    """Numbers about when one fighter threw, from the times of their counted strikes.

    `round_bounds` are (number, start, end) seconds of the rounds analysed.
    """
    times = sorted(float(t) for t in strike_times)
    busiest = 0
    busiest_at = None
    for index, start in enumerate(times):
        inside = sum(1 for t in times[index:] if t - start < BUSY_WINDOW_SECONDS)
        if inside > busiest:
            busiest, busiest_at = inside, start
    longest_pause = None
    last_30 = {}
    for number, start, end in round_bounds:
        marks = [start] + [t for t in times if start <= t <= end] + [end]
        pause = max((b - a for a, b in zip(marks, marks[1:])), default=None)
        if pause is not None:
            longest_pause = pause if longest_pause is None else max(longest_pause, pause)
        last_30[str(number)] = sum(1 for t in times if max(start, end - 30.0) <= t <= end)
    total_seconds = sum(max(0.0, end - start) for _, start, end in round_bounds)
    first_exchange = None
    if times and opponent_times:
        first_exchange = times[0] < min(opponent_times)
    return {
        "strikes_per_minute": round(len(times) / (total_seconds / 60.0), 1) if total_seconds > 0 else None,
        "busiest_10s": busiest,
        "busiest_10s_at": None if busiest_at is None else round(busiest_at, 1),
        "longest_pause_seconds": None if longest_pause is None else round(longest_pause, 1),
        "first_strike_seconds": round(times[0], 1) if times else None,
        "threw_first": first_exchange,
        "last_30s_by_round": last_30,
    }
