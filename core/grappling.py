"""Ground control in MMA: who is on top, for how long, and takedowns.

Read from the two tracked fighters' poses in tracking.jsonl, after the
analysis has run. A frame is a ground frame when the fighters are tangled
together (their boxes overlap) and at least one of them is down by
core/ground.py's pose_down. The fighter on top is the one whose hips sit
clearly higher in the picture than the other's, and who is not the one lying
flat. A second is decided by a majority of its analysed frames; one frame
decides nothing.

Measurement only. Nothing here reaches a report or a score until it has been
measured against official statistics (tools/measure_grappling.py compares
control time and takedowns with ufcstats.com's per-round numbers). The
tracker can swap the two fighters while they are tangled on the ground, and a
swap credits one fighter's control to the other: the comparison is what
shows whether that happens.
"""

from __future__ import annotations

import math
from collections import defaultdict

from core.ground import pose_down

_HIPS = (11, 12)
_SHOULDERS = (5, 6)
_JOINT_CONFIDENCE = 0.4
# Tangled: the overlap covers this share of the smaller fighter's box.
MIN_OVERLAP_SHARE = 0.2
# On top: hips higher than the other fighter's by this share of the top
# fighter's torso. Smaller gaps are side by side, a scramble, or a camera
# looking down the length of both bodies.
TOP_MARGIN_TORSO = 0.25
# A second needs this many analysed frames, and this share of them agreeing.
MIN_FRAMES_PER_SECOND = 2
MAJORITY = 0.6
# A takedown: standing for this long, then on the ground with one fighter on
# top for at least MIN_GROUND_AFTER seconds, the switch within this window.
STANDING_BEFORE = 3
MIN_GROUND_AFTER = 3
SWITCH_WINDOW = 4


def _mid(observation: dict, pair: tuple[int, int]):
    keypoints, conf = observation.get("keypoints"), observation.get("keypoint_conf")
    if not keypoints or not conf:
        return None
    points = []
    for index in pair:
        if index >= len(keypoints) or float(conf[index]) <= _JOINT_CONFIDENCE:
            return None
        points.append(keypoints[index])
    return (points[0][0] + points[1][0]) / 2.0, (points[0][1] + points[1][1]) / 2.0


def _overlap_share(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def both_standing(record: dict) -> bool:
    """Both fighters seen, each upright with hips and shoulders visible.

    Measured on UFC 326 Holloway vs Oliveira 2, round 1 (official stats as
    the answer, nobody labelling): in the opening 30 seconds of stand-up this
    held on 136 of 148 frames where both fighters were seen; through the
    grappling that followed it almost never did - the pose model cannot pull
    two tangled bodies apart, so at least one has no readable hips or
    shoulders, or is down.
    """
    observations = [(record.get(key) or {}).get("observation") for key in ("fighter_A", "fighter_B")]
    for observation in observations:
        if not observation or _mid(observation, _HIPS) is None or _mid(observation, _SHOULDERS) is None:
            return False
        if pose_down(observation.get("keypoints"), observation.get("keypoint_conf")) is not None:
            return False
    return True


def standing_seconds(records) -> set[int]:
    """Whole seconds in which most analysed frames show both fighters standing."""
    frames: dict[int, list[bool]] = defaultdict(list)
    for record in records:
        if record.get("selected_round") is False:
            continue
        frames[int(float(record.get("time_seconds") or 0.0))].append(both_standing(record))
    return {second for second, flags in frames.items()
            if len(flags) >= MIN_FRAMES_PER_SECOND and sum(flags) > len(flags) / 2}


def strikes_while_standing(events, records) -> list:
    """The strike events thrown in a second when both fighters were standing.

    The standing-strike detector reads arm and leg motion from upright poses;
    through grappling it turns scrambles, grips and position changes into
    punches (149 strikes on a round with 41 officially, 21 of them at
    distance). This keeps only the ones thrown standing.

    Measured on UFC 326 Holloway vs Oliveira 2 (official distance strikes,
    both fighters together; designed on round 1, then checked on 3 and 4):
    round 1 149 -> 24 (official 21), round 3 157 -> 36 (12), round 4
    314 -> 20 (17). Per fighter it is worse: the tracker swaps the two at
    broadcast camera cuts (a quarter to a third of clearly-seen standing
    frames), so a right total can still credit the wrong fighter. Not good
    enough to publish; it narrows where the remaining error is.
    """
    standing = standing_seconds(records)
    kept = []
    for event in events:
        peak = event.get("peak_time") if isinstance(event, dict) else getattr(event, "peak_time", None)
        if peak is not None and int(float(peak)) in standing:
            kept.append(event)
    return kept


def frame_position(record: dict) -> dict | None:
    """{"ground": bool, "top": "A" | "B" | None}, or None when it cannot be read.

    None means one fighter is not seen or their hips and shoulders are not
    visible - that frame says nothing either way.
    """
    a = (record.get("fighter_A") or {}).get("observation")
    b = (record.get("fighter_B") or {}).get("observation")
    if not a or not b or not a.get("box") or not b.get("box"):
        return None
    hips = {"A": _mid(a, _HIPS), "B": _mid(b, _HIPS)}
    shoulders = {"A": _mid(a, _SHOULDERS), "B": _mid(b, _SHOULDERS)}
    if None in hips.values() or None in shoulders.values():
        return None
    postures = {"A": pose_down(a.get("keypoints"), a.get("keypoint_conf")),
                "B": pose_down(b.get("keypoints"), b.get("keypoint_conf"))}
    tangled = _overlap_share(a["box"], b["box"]) >= MIN_OVERLAP_SHARE
    if not tangled or (postures["A"] is None and postures["B"] is None):
        return {"ground": False, "top": None}
    top = None
    for name, other in (("A", "B"), ("B", "A")):
        torso = math.dist(hips[name], shoulders[name])
        if torso <= 0:
            continue
        higher = hips[other][1] - hips[name][1] >= TOP_MARGIN_TORSO * torso
        # Flat on the back under someone is the bottom whatever the picture
        # says; both flat is a scramble with no one on top.
        lying_alone = postures[name] == "lying" and postures[other] != "lying"
        if higher and not lying_alone:
            top = name
    return {"ground": True, "top": top}


def seconds_of(records) -> dict[int, dict]:
    """Per whole second: {"round", "ground": bool, "top": "A"|"B"|None}.

    Seconds with too few readable frames are left out rather than guessed.
    """
    frames: dict[int, list] = defaultdict(list)
    rounds: dict[int, int | None] = {}
    for record in records:
        if record.get("selected_round") is False:
            continue
        position = frame_position(record)
        if position is None:
            continue
        second = int(float(record.get("time_seconds") or 0.0))
        frames[second].append(position)
        rounds.setdefault(second, record.get("round_number"))
    out = {}
    for second, positions in frames.items():
        if len(positions) < MIN_FRAMES_PER_SECOND:
            continue
        grounded = [p for p in positions if p["ground"]]
        ground = len(grounded) >= MAJORITY * len(positions)
        top = None
        if ground:
            for name in ("A", "B"):
                if sum(1 for p in grounded if p["top"] == name) >= MAJORITY * len(grounded):
                    top = name
        out[second] = {"round": rounds.get(second), "ground": ground, "top": top}
    return out


def takedowns(seconds: dict[int, dict]) -> list[dict]:
    """Standing, then on the ground with one fighter on top: credited to them."""
    found = []
    ordered = sorted(seconds)
    last_credit = -10 ** 9
    for second in ordered:
        state = seconds[second]
        if not state["top"] or second - last_credit < STANDING_BEFORE + MIN_GROUND_AFTER:
            continue
        before = [seconds.get(s) for s in range(second - SWITCH_WINDOW - STANDING_BEFORE, second - SWITCH_WINDOW + 1)]
        if not all(b is not None and not b["ground"] for b in before[:STANDING_BEFORE]):
            continue
        after = [seconds.get(s) for s in range(second, second + MIN_GROUND_AFTER)]
        if all(a is not None and a["top"] == state["top"] for a in after):
            found.append({"seconds": second, "round": state["round"], "fighter": state["top"]})
            last_credit = second
    return found


def summary(records) -> dict[int | None, dict]:
    """Per round: ground seconds, control seconds and takedowns for A and B."""
    seconds = seconds_of(records)
    rounds: dict[int | None, dict] = defaultdict(lambda: {
        "read_seconds": 0, "ground_seconds": 0,
        "control_seconds": {"A": 0, "B": 0}, "takedowns": {"A": 0, "B": 0}})
    for state in seconds.values():
        line = rounds[state["round"]]
        line["read_seconds"] += 1
        if state["ground"]:
            line["ground_seconds"] += 1
            if state["top"]:
                line["control_seconds"][state["top"]] += 1
    for takedown in takedowns(seconds):
        rounds[takedown["round"]]["takedowns"][takedown["fighter"]] += 1
    return dict(rounds)
