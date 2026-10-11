"""The report's movement graphs: what core/fight_numbers.py measured, as shapes.

A fighter asked for graphs instead of rows of numbers (2026-10-11). Nothing
here is a new measurement. It reshapes four things the analysis already
stores for each fighter (``report["metrics"][side]["numbers"]``):

  how you moved   seconds moving forward, backing up, circling, standing
  distance        seconds close, at middle range and long range
  energy          speed through the fight (``series``, analyses from
                  2026-10-11 on; older reports simply do not show it)
  guard           share of each stretch with the hands up (same ``series``)

Each with the opponent beside it where the opponent was measured too, since
the opponent from the same fight is the only fair benchmark.

The page shows this only on a report whose identity check held (app.main
builds it next to core/report_visuals.py, under the same gate): every shape
here is a claim about one named fighter.
"""

from __future__ import annotations

# The energy and guard graphs are drawn in a viewBox this size.
WIDTH, HEIGHT = 320.0, 110.0

MOVED = (("forward_seconds", "Forward"), ("backward_seconds", "Backing up"),
         ("circling_seconds", "Circling"), ("standing_seconds", "Standing still"))
RANGES = (("close", "Close"), ("middle", "Middle"), ("long", "Long"))


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return "%d:%02d" % (seconds // 60, seconds % 60)


def _split(numbers: dict | None, parts, source=None) -> list[dict] | None:
    """Seconds per part and each part's share of their total, or None when nothing was measured."""
    if not numbers:
        return None
    values = source if source is not None else numbers
    seconds = [(key, label, float(values.get(key) or 0.0)) for key, label in parts]
    total = sum(value for _, _, value in seconds)
    if total <= 0:
        return None
    return [{"key": key, "label": label, "seconds": round(value, 1), "clock": _clock(value),
             "share": round(value / total * 100.0, 1)} for key, label, value in seconds]


def _lines(series_by_side: dict[str, dict | None], value: str, top: float, start: float, span: float):
    """SVG polylines per side, broken wherever a point was not measured."""
    out = {}
    for side, series in series_by_side.items():
        if not series:
            continue
        step = float(series["step_seconds"])
        segments, current = [], []
        for point in series["points"]:
            reading = point.get(value)
            if reading is None:
                if len(current) >= 2:
                    segments.append(current)
                current = []
                continue
            x = (float(point["t"]) + step / 2.0 - start) / span * WIDTH
            y = HEIGHT - min(1.0, float(reading) / top) * HEIGHT
            current.append("%.1f,%.1f" % (x, y))
        if len(current) >= 2:
            segments.append(current)
        if segments:
            out[side] = [" ".join(segment) for segment in segments]
    return out


def build(report: dict, focus: str = "A") -> dict | None:
    """Everything the graphs section needs for ``focus``, or None when there is nothing to draw."""
    metrics = report.get("metrics") or {}
    other = "B" if focus == "A" else "A"
    mine = (metrics.get(focus) or {}).get("numbers")
    theirs = (metrics.get(other) or {}).get("numbers")
    if not mine:
        return None

    moved = {side: _split(numbers, MOVED) for side, numbers in ((focus, mine), (other, theirs))}
    ranges = {side: _split(numbers, RANGES, (numbers or {}).get("range_seconds") or {})
              for side, numbers in ((focus, mine), (other, theirs))}

    series = {focus: mine.get("series"), other: (theirs or {}).get("series")}
    drawn = [s for s in series.values() if s and s.get("points")]
    energy = guard = None
    if drawn:
        start = min(float(s["points"][0]["t"]) for s in drawn)
        end = max(float(s["points"][-1]["t"]) + float(s["step_seconds"]) for s in drawn)
        span = max(1.0, end - start)
        speeds = [float(p["speed"]) for s in drawn for p in s["points"] if p.get("speed") is not None]
        if speeds:
            top = max(0.5, max(speeds))
            lines = _lines(series, "speed", top, start, span)
            if focus in lines:
                energy = {"lines": lines, "top": round(top, 1),
                          "axis": [_clock(start), _clock(start + span / 2.0), _clock(end)]}
        own = series[focus]
        if own and any(p.get("hands_up") is not None for p in own["points"]):
            step = float(own["step_seconds"])
            bars = [{"x": round((float(p["t"]) - start) / span * WIDTH, 1),
                     "width": round(max(1.0, step / span * WIDTH - 1.5), 1),
                     "height": round(float(p["hands_up"]) * HEIGHT, 1),
                     "share": round(float(p["hands_up"]) * 100),
                     "clock": _clock(float(p["t"]))}
                    for p in own["points"] if p.get("hands_up") is not None]
            guard = {"bars": bars, "axis": [_clock(start), _clock(start + span / 2.0), _clock(end)]}

    facts = []
    if mine.get("distance_body_lengths") is not None:
        facts.append({"value": "%.0f" % mine["distance_body_lengths"], "label": "body lengths covered",
                      "sentence": "How far you moved around in the whole fight."})
    if mine.get("longest_hands_down_seconds") is not None:
        facts.append({"value": _clock(mine["longest_hands_down_seconds"]), "label": "longest with hands down",
                      "sentence": "Your longest stretch without the hands up by the face."})
    if mine.get("off_balance_count") is not None:
        count = int(mine["off_balance_count"])
        facts.append({"value": str(count), "label": "time off balance" if count == 1 else "times off balance",
                      "sentence": "Stretches of half a second or more off balance."})
    pace = mine.get("pace_change_percent")
    if pace is not None:
        facts.append({"value": "%+d%%" % pace, "label": "pace, second half",
                      "sentence": ("You moved %d%% slower in the second half than the first." % -pace if pace < 0
                                   else "You moved %d%% faster in the second half than the first." % pace if pace > 0
                                   else "You moved at the same pace in both halves.")})

    if not any([moved[focus], ranges[focus], energy, guard, facts]):
        return None
    reads = {"moved": _moved_read(moved[focus], moved[other]),
             "ranges": _ranges_read(ranges[focus]),
             "energy": _energy_read(series[focus], series[other]) if energy else None,
             "guard": _guard_read(series[focus], mine.get("hands_up_share")) if guard else None}
    return {"focus": focus, "opponent": other, "moved": moved, "ranges": ranges,
            "energy": energy, "guard": guard, "facts": facts, "reads": reads,
            "width": WIDTH, "height": HEIGHT}


# What each graph says, in a sentence with its numbers (2026-10-11: a
# fighter liked the graphs but could not tell what they showed). Each is
# read straight off the same figures the graph draws.

def _pct(parts: list[dict] | None, key: str) -> int:
    return int(round(next((p["share"] for p in parts or [] if p["key"] == key), 0.0)))


def _moved_read(mine: list[dict] | None, theirs: list[dict] | None) -> str | None:
    if not mine:
        return None
    text = (f"You went forward {_pct(mine, 'forward_seconds')}% of the time, backed up "
            f"{_pct(mine, 'backward_seconds')}%, circled {_pct(mine, 'circling_seconds')}% and stood still "
            f"{_pct(mine, 'standing_seconds')}%.")
    if theirs:
        text += f" Your opponent went forward {_pct(theirs, 'forward_seconds')}% of the time."
    return text


def _ranges_read(mine: list[dict] | None) -> str | None:
    if not mine:
        return None
    return (f"You were close to your opponent {_pct(mine, 'close')}% of the time, at middle distance "
            f"{_pct(mine, 'middle')}% and far apart {_pct(mine, 'long')}%.")


def _mean_speed(series: dict | None) -> float | None:
    speeds = [float(p["speed"]) for p in ((series or {}).get("points") or []) if p.get("speed") is not None]
    return sum(speeds) / len(speeds) if speeds else None


def _energy_read(mine: dict | None, theirs: dict | None) -> str | None:
    points = [p for p in ((mine or {}).get("points") or []) if p.get("speed") is not None]
    if not points:
        return None
    busiest = max(points, key=lambda p: p["speed"])
    quietest = min(points, key=lambda p: p["speed"])
    text = (f"You were busiest at {_clock(busiest['t'])} ({busiest['speed']:.1f} body lengths a second) "
            f"and quietest at {_clock(quietest['t'])} ({quietest['speed']:.1f}).")
    yours, other = _mean_speed(mine), _mean_speed(theirs)
    if other is not None:
        text += f" On average you moved {yours:.1f} body lengths a second, your opponent {other:.1f}."
    return text


def _guard_read(mine: dict | None, overall: float | None) -> str | None:
    points = [p for p in ((mine or {}).get("points") or []) if p.get("hands_up") is not None]
    if not points:
        return None
    lowest = min(points, key=lambda p: p["hands_up"])
    text = f"Your hands were up {round(overall * 100)}% of the time. " if overall is not None else ""
    return text + (f"They were lowest at {_clock(lowest['t'])}: up {round(lowest['hands_up'] * 100)}% "
                   "of those few seconds.")


def _energy_chart(report: dict, focus: str, top: float) -> dict | None:
    """One fighter's speed line on a given scale, so two fights can share it."""
    series = ((report.get("metrics") or {}).get(focus) or {}).get("numbers", {}) or {}
    series = series.get("series") if isinstance(series, dict) else None
    if not series or not series.get("points"):
        return None
    start = float(series["points"][0]["t"])
    end = float(series["points"][-1]["t"]) + float(series["step_seconds"])
    span = max(1.0, end - start)
    lines = _lines({focus: series}, "speed", top, start, span).get(focus)
    if not lines:
        return None
    return {"lines": lines, "axis": [_clock(start), _clock(start + span / 2.0), _clock(end)]}


def compare(reports: list[dict | None], focuses: list[str], failed: list[bool]) -> dict | None:
    """The same graphs for two fights, side by side, for the compare page.

    One fighter per fight - the one each report is about - so the bars read
    "Fight 1" against "Fight 2". The two energy graphs share one scale, so a
    taller line really is a busier fight. A fight whose fighters could not be
    followed (``failed``) gets no graphs and a reason, as on its own report.
    """
    fights = []
    for index, (report, focus, bad) in enumerate(zip(reports, focuses, failed)):
        label = f"Fight {index + 1}"
        graphs = build(report, focus) if report and not bad else None
        why = None
        if bad:
            why = f"{label}'s fighters could not be followed, so it has no graphs."
        elif graphs is None:
            why = f"{label} has nothing to draw: too little of the fighter was seen."
        fights.append({"label": label, "report": report, "focus": focus, "graphs": graphs, "why": why})
    usable = [fight for fight in fights if fight["graphs"]]
    if not usable:
        return None

    def rows(kind: str) -> list[dict]:
        return [{"label": fight["label"], "parts": fight["graphs"][kind][fight["focus"]]}
                for fight in usable if fight["graphs"][kind][fight["focus"]]]

    speeds = []
    for fight in usable:
        numbers = ((fight["report"].get("metrics") or {}).get(fight["focus"]) or {}).get("numbers") or {}
        speeds += [float(p["speed"]) for p in ((numbers.get("series") or {}).get("points") or [])
                   if p.get("speed") is not None]
    top = max(0.5, max(speeds)) if speeds else None
    energy = []
    if top is not None:
        for fight in usable:
            chart = _energy_chart(fight["report"], fight["focus"], top)
            if chart:
                energy.append({"label": fight["label"], **chart})
    guard = [{"label": fight["label"], **fight["graphs"]["guard"]} for fight in usable if fight["graphs"]["guard"]]

    by_label: dict[str, dict] = {}
    for fight in usable:
        for fact in fight["graphs"]["facts"]:
            row = by_label.setdefault(fact["label"], {"label": fact["label"], "values": {}})
            row["values"][fight["label"]] = fact["value"]
    labels = [fight["label"] for fight in fights]
    facts = [{"label": row["label"], "values": [row["values"].get(label, "—") for label in labels]}
             for row in by_label.values()]

    return {"labels": labels, "notes": [fight["why"] for fight in fights if fight["why"]],
            "moved": rows("moved"), "ranges": rows("ranges"),
            "energy": energy, "energy_top": round(top, 1) if top is not None else None,
            "guard": guard, "facts": facts, "width": WIDTH, "height": HEIGHT}
