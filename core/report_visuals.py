"""Turn a finished report into things a fighter can read without reading.

The report had roughly six numbers and ten thousand characters of prose, and
the analysis underneath it was already computing far more than that. Defence
counts, combination lengths, which targets a fighter was caught on, the exact
seconds their guard dropped - all measured, all stored, none of it shown.

So this adds no measurement. It reshapes what is already in the report into
five things that carry their meaning in position, length and place on a body
rather than in sentences:

  head-to-head   the opponent as the benchmark, which beats any invented
                 target band because it came from the same fight
  body maps      where strikes arrived, on a silhouette
  timeline       when things happened, yours above the line and theirs below
  defence mix    which defences the fighter actually used
  combinations   how long their chains ran

Computed at render time rather than written into the report, so every analysis
already on disk gains it without being re-run.

**What it may not show.** Punch counting is switched off - checked against
video, the count was overstated by eleven in two bouts of three. A body map
fed from the raw event stream would quietly republish exactly that number in
a new shape. So everything here counts kicks only, from the same
`ARRIVED_OUTCOMES` set the published timeline uses, and the page says so.

Knees are excluded too, which is less obvious. They were folded in with kicks
because both are a leg arriving, so a misnamed one was harmless. Labelling the
HD bout at family level found five proposed knees and not one real knee: three
were kicks and two were punches. Keeping the bucket would publish the withheld
family under another name, so it goes.
"""

from __future__ import annotations

from collections import Counter

from core.report import ARRIVED_OUTCOMES

# The families that may be counted. Punches are excluded deliberately, and
# knees with them because the knee bucket was measured to contain punches;
# see the module docstring and core/report.py's
# STRIKE_COUNTS_PRECISION_VALIDATED.
COUNTABLE = ("kick",)

# Body zones, in the order a person reads a body.
ZONES = ("head", "body", "leg")


def _countable(event: dict) -> bool:
    """A kick that arrived, which is the only thing safe to count."""
    technique = str(event.get("technique") or "").lower()
    if not any(word in technique for word in COUNTABLE):
        return False
    return str(event.get("outcome") or "") in ARRIVED_OUTCOMES


def _zones(events: list[dict]) -> dict[str, int]:
    counted = Counter(str(e.get("target") or "") for e in events)
    return {zone: int(counted.get(zone, 0)) for zone in ZONES}


def _percent(value) -> int | None:
    return None if value is None else int(round(float(value) * 100))


def head_to_head(metrics: dict, mine: str, theirs: str, landed: dict) -> list[dict]:
    """Every published measurement as a pair, with the opponent as the scale.

    A bare "Guard 14%" tells a fighter nothing. The same 14% beside an
    opponent's 31% tells them everything, and needs no threshold anybody had
    to invent.

    `absolute` decides the scale. A percentage is read against 0-100, because
    "guard up 11% of the round" is small in a way worth seeing; a count or a
    rate is read against the pair, because there is no natural ceiling.
    """
    a, b = metrics.get(mine) or {}, metrics.get(theirs) or {}
    rows = [
        {"key": "landed", "label": "Kicks landed", "sub": "kicks only",
         "a": landed.get(mine, 0), "b": landed.get(theirs, 0), "absolute": False},
        {"key": "guard", "label": "Guard up", "sub": "% of the round",
         "a": _percent(a.get("guard_index")), "b": _percent(b.get("guard_index")),
         "absolute": True, "suffix": "%"},
        {"key": "balance", "label": "Balanced", "sub": "% of the round",
         "a": _percent(a.get("balance_index")), "b": _percent(b.get("balance_index")),
         "absolute": True, "suffix": "%"},
        {"key": "centre", "label": "Held the centre", "sub": "% of the round",
         "a": _percent(a.get("ring_center_control")), "b": _percent(b.get("ring_center_control")),
         "absolute": True, "suffix": "%"},
        {"key": "movement", "label": "Movement", "sub": "body lengths/sec",
         "a": _round(a.get("footwork_body_lengths_per_second")),
         "b": _round(b.get("footwork_body_lengths_per_second")), "absolute": False},
    ]
    out = []
    for row in rows:
        # A measurement that is missing for either fighter is not a comparison,
        # and drawing one bar against an empty space reads as a win.
        if row["a"] is None or row["b"] is None:
            continue
        ceiling = 100.0 if row["absolute"] else max(row["a"], row["b"]) * 1.15
        row["a_width"] = _width(row["a"], ceiling)
        row["b_width"] = _width(row["b"], ceiling)
        row["leader"] = "a" if row["a"] > row["b"] else "b" if row["b"] > row["a"] else ""
        out.append(row)
    return out


def _round(value) -> float | None:
    return None if value is None else round(float(value), 2)


def _width(value: float, ceiling: float) -> float:
    if ceiling <= 0:
        return 3.0
    # A floor of 3%, so a zero is still a visible mark rather than nothing at
    # all - "they landed none" is a result and should look like one.
    return max(3.0, min(100.0, float(value) / ceiling * 100.0))


def build(report: dict, focus: str = "A") -> dict | None:
    """Everything the visual sections need, or None when there is nothing."""
    metrics = report.get("metrics") or {}
    if focus not in metrics:
        return None
    other = "B" if focus == "A" else "A"
    events = [e for e in (report.get("events") or []) if _countable(e)]

    landed_by = {
        side: len([e for e in events if str(e.get("fighter")) == side])
        for side in (focus, other)
    }
    mine = metrics.get(focus) or {}

    combinations = mine.get("combinations") or {}
    # Each combination's evidence is the list of times its strikes happened,
    # so its length is the chain length. Longest first: the best chain is the
    # one worth looking at.
    chains = sorted(
        (len(times) for times in (combinations.get("evidence") or []) if times),
        reverse=True)

    rounds = report.get("rounds") or []
    span = max((float(r.get("end_seconds") or 0) for r in rounds), default=0.0)

    timeline = []
    for event in (report.get("events") or []):
        if not _countable(event) and str(event.get("outcome") or "") in ARRIVED_OUTCOMES:
            # A punch that arrived is deliberately absent rather than drawn as
            # "unknown" - an outline on the timeline is still a claim that
            # something happened then.
            continue
        if not _countable(event):
            continue
        timeline.append({
            "at": round(float(event.get("peak_time") or 0.0), 2),
            "mine": str(event.get("fighter")) == focus,
            "arrived": True,
        })

    moments = (mine.get("moments") or {}).get("guard_index") or {}
    guard_low = [round(float(t), 1) for t in (moments.get("low") or [])]

    defences = {k: int(v) for k, v in (mine.get("defenses") or {}).items() if v}

    return {
        "focus": focus,
        "opponent": other,
        "head_to_head": head_to_head(metrics, focus, other, landed_by),
        "landed": _zones([e for e in events if str(e.get("fighter")) == focus]),
        "taken": _zones([e for e in events if str(e.get("fighter")) == other]),
        "defences": dict(sorted(defences.items(), key=lambda kv: -kv[1])),
        "defence_total": sum(defences.values()),
        "defence_max": max(defences.values()) if defences else 0,
        "chains": chains,
        "chain_count": int(combinations.get("count") or len(chains)),
        "chain_longest": int(combinations.get("max_length") or (chains[0] if chains else 0)),
        "timeline": timeline,
        "guard_low": guard_low,
        "span_seconds": round(span, 1) if span > 0 else 0.0,
        # Said once, in one place, so every section below inherits it rather
        # than repeating a disclaimer five times.
        "counts": "Kicks only. Hands and knees are not counted yet.",
    }
