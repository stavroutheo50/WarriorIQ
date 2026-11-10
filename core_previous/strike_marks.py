"""Strikes a person marked while watching a fight, and how the count compares.

The labelled benchmark so far comes from clips the detector itself proposed
(tools/benchmark_labelled_fight.py), so it measures how many counted strikes
are real but never how many real strikes were missed. A person watching the
whole fight and tapping each strike as it is thrown answers both.

A mark is the moment of a tap, so it lands after the strike: a person reacts
in a few tenths of a second, less at half speed. A counted strike and a mark
of the same fighter are the same strike when the mark falls from MARK_EARLY
before to MARK_LATE after it. Each counted strike and each mark is used once,
nearest first.

Only the part of the video the person watched is compared: counted strikes
after `covered_seconds` were never seen, so they are neither found nor extra.
"""

from __future__ import annotations

import math

FAMILIES = ("punch", "kick", "knee")
MAX_MARKS = 3000

# A tap before the counted moment is a person anticipating, or the detector
# placing the strike late; a tap after it is reaction time. Wider after.
MARK_EARLY = 0.5
MARK_LATE = 1.0


def clean_marks(raw: object, duration: float | None = None) -> list[dict]:
    """The marks as stored: valid entries only, in time order. Raises ValueError on a bad list."""
    if not isinstance(raw, list) or len(raw) > MAX_MARKS:
        raise ValueError(f"marks must be a list of at most {MAX_MARKS}")
    limit = (float(duration) + 5.0) if duration else None
    out = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each mark is an object")  # noqa: TRY004 - one error type, one 400
        try:
            seconds = float(item.get("t"))
        except (TypeError, ValueError):
            raise ValueError("a mark needs a time") from None
        fighter = str(item.get("fighter") or "").upper()
        family = item.get("family")
        landed = item.get("landed")
        if not math.isfinite(seconds) or seconds < 0 or (limit is not None and seconds > limit):
            raise ValueError("a mark is outside the video")
        if fighter not in {"A", "B"}:
            raise ValueError("a mark is for fighter A or B")
        if family is not None and family not in FAMILIES:
            raise ValueError("unknown strike type")
        if landed is not None and not isinstance(landed, bool):
            raise ValueError("landed is yes, no or not said")
        out.append({"t": round(seconds, 3), "fighter": fighter, "family": family, "landed": landed})
    return sorted(out, key=lambda mark: (mark["t"], mark["fighter"]))


def compare(marks: list[dict], counted: list[dict], covered_seconds: float) -> dict:
    """How WarriorIQ's counted strikes line up with the marks, up to `covered_seconds`.

    `counted` rows carry "fighter", "family" and "seconds" (app.main._counted_strikes).
    """
    covered = max(0.0, float(covered_seconds or 0.0))
    seen = [row for row in counted if float(row["seconds"]) <= covered]
    result = {"covered_seconds": round(covered, 1), "fighters": {}}
    totals = {"marked": 0, "counted": 0, "matched": 0, "type_checked": 0, "type_right": 0}
    for fighter in ("A", "B"):
        mine = [mark for mark in marks if mark["fighter"] == fighter]
        theirs = sorted((row for row in seen if row["fighter"] == fighter), key=lambda row: float(row["seconds"]))
        pairs = []
        for i, row in enumerate(theirs):
            for j, mark in enumerate(mine):
                gap = mark["t"] - float(row["seconds"])
                if -MARK_EARLY <= gap <= MARK_LATE:
                    pairs.append((abs(gap), i, j))
        used_counted, used_marks, matched = set(), set(), []
        for _, i, j in sorted(pairs):
            if i in used_counted or j in used_marks:
                continue
            used_counted.add(i)
            used_marks.add(j)
            matched.append((theirs[i], mine[j]))
        typed = [(row, mark) for row, mark in matched if mark.get("family")]
        right = sum(1 for row, mark in typed if row.get("family") == mark["family"])
        side = {"marked": len(mine), "counted": len(theirs), "matched": len(matched),
                "type_checked": len(typed), "type_right": right}
        result["fighters"][fighter] = side
        for key in totals:
            totals[key] += side[key]
    totals["counted_real_share"] = _share(totals["matched"], totals["counted"])
    totals["marked_found_share"] = _share(totals["matched"], totals["marked"])
    result.update(totals)
    return result


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None
