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
    """SVG polylines per side, broken wherever a point was not measured.

    A measured point with no measured neighbour is drawn as a short flat line
    across its own step. It used to be dropped - a run needed two points - so
    the graph showed a gap exactly where the text below it named a moment
    ("quietest at 1:05", QA 2026-10-11): the graph was hiding data it had.
    """
    out = {}
    for side, series in series_by_side.items():
        if not series:
            continue
        step = float(series["step_seconds"])
        segments, current = [], []

        def close(run):
            if len(run) >= 2:
                segments.append([f"{x:.1f},{y:.1f}" for x, y, _ in run])
            elif len(run) == 1:
                _x, y, t = run[0]
                left = (t - start) / span * WIDTH
                right = (t + step - start) / span * WIDTH
                segments.append([f"{left:.1f},{y:.1f}", f"{right:.1f},{y:.1f}"])

        for point in series["points"]:
            reading = point.get(value)
            if reading is None:
                close(current)
                current = []
                continue
            x = (float(point["t"]) + step / 2.0 - start) / span * WIDTH
            y = HEIGHT - min(1.0, float(reading) / top) * HEIGHT
            current.append((x, y, float(point["t"])))
        close(current)
        if segments:
            out[side] = [" ".join(segment) for segment in segments]
    return out


def _in_fight_footage(start: float, end: float, segments) -> bool:
    """Whether any of [start, end) is fight footage (core/fight_presence.py)."""
    return any(float(a) < end and float(b) > start for a, b in segments)


def _gaps(series: dict | None, value: str, report: dict) -> list[dict]:
    """Each stretch with no reading, and why: left out as not fight footage,
    or the fighter was not seen. The hint under the graph used to blame every
    gap on not being seen, beside "Seen 93%" - which is a share of the fight
    footage only, so a stretch left out of the footage is not "not seen"."""
    if not series:
        return []
    step = float(series["step_seconds"])
    footage = ((report.get("video") or {}).get("fight_footage") or {})
    segments = footage.get("segments") if isinstance(footage.get("segments"), list) else None
    out, run = [], []
    for point in series["points"] + [None]:
        if point is not None and point.get(value) is None:
            run.append(float(point["t"]))
            continue
        if run:
            first, last = run[0], run[-1] + step
            left_out = segments is not None and not _in_fight_footage(first, last, segments)
            out.append({"from": _clock(first), "to": _clock(last),
                        "reason": ("left out: not fight footage" if left_out
                                   else "too little of you seen to measure")})
            run = []
    return out


# The moments core/metrics.py keeps for each fighter (up to four spread-out
# seconds at the top and the bottom of each measure), named for a fighter.
# Which end is the good one is part of the name, not a judgement added here:
# hands up, balanced and moving are what a coach asks for.
HIGHLIGHTS = (
    ("guard_index", "high", "Guard up tight", True),
    ("guard_index", "low", "Guard dropped", False),
    ("balance_index", "high", "Well balanced", True),
    ("balance_index", "low", "Off balance", False),
    ("pressure_index", "high", "Pressing forward", True),
    ("pressure_index", "low", "Backing off", False),
    ("footwork_body_lengths_per_second", "high", "Busiest feet", True),
    ("footwork_body_lengths_per_second", "low", "Flat-footed", False),
)


def _span(report: dict) -> float:
    rounds = report.get("rounds") or []
    return max((float(r.get("end_seconds") or 0) for r in rounds), default=0.0)


def highlights(report: dict, focus: str = "A") -> dict | None:
    """The fighter's clearest moments on one line: rounds, best and worst seconds, anyone going down.

    Every mark is a second the analysis already picked (``metrics[side].moments``)
    or found (``went_down``); each links to the replay at that second. Moments
    the report withheld (guard and balance on a sideways video) are simply
    absent from ``moments`` and so from here.
    """
    span = _span(report)
    if span <= 0:
        return None
    moments = (((report.get("metrics") or {}).get(focus) or {}).get("moments") or {})
    marks = []
    for metric, end, label, good in HIGHLIGHTS:
        for t in ((moments.get(metric) or {}).get(end) or []):
            t = float(t)
            if 0.0 <= t <= span:
                marks.append({"at": round(t, 1), "clock": _clock(t), "label": label, "good": good,
                              "left": round(t / span * 100.0, 2)})
    down = []
    for moment in ((report.get("went_down") or {}).get("moments") or []):
        t = float(moment.get("seconds") or 0.0)
        if 0.0 <= t <= span:
            down.append({"at": round(t, 1), "clock": _clock(t), "left": round(t / span * 100.0, 2),
                         "seconds": int(moment.get("down_seconds") or 0)})
    bands = []
    for index, item in enumerate(report.get("rounds") or []):
        start, end = float(item.get("start_seconds") or 0.0), float(item.get("end_seconds") or 0.0)
        if end > start and item.get("selected", True):
            bands.append({"number": item.get("number") or index + 1,
                          "left": round(start / span * 100.0, 2),
                          "width": round((min(end, span) - start) / span * 100.0, 2),
                          "clock": f"{_clock(start)}\u2013{_clock(end)}"})
    if not marks and not down:
        return None
    marks.sort(key=lambda m: m["at"])
    good = sum(1 for m in marks if m["good"])
    return {"marks": marks, "down": down, "rounds": bands if len(bands) > 1 else [],
            "span_clock": _clock(span), "mid_clock": _clock(span / 2.0), "good": good, "bad": len(marks) - good}


def by_round(report: dict, focus: str = "A") -> dict | None:
    """Each round: how much of it each fighter held the centre of the ring, and was seen.

    From ``numbers.rounds`` (centre and seen seconds per round). A round needs
    ten seen seconds for a fighter to get a share; fewer is not a share of
    anything.
    """
    metrics = report.get("metrics") or {}
    other = "B" if focus == "A" else "A"
    per_side = {side: ((metrics.get(side) or {}).get("numbers") or {}).get("rounds") or {} for side in (focus, other)}
    numbers = sorted({int(n) for rounds in per_side.values() for n in rounds if str(n).isdigit()})
    if len(numbers) < 2:
        return None
    rows = []
    for n in numbers:
        row = {"number": n}
        for side in (focus, other):
            item = per_side[side].get(str(n)) or per_side[side].get(n) or {}
            seen, centre = float(item.get("seen_seconds") or 0.0), float(item.get("centre_seconds") or 0.0)
            row[side] = round(centre / seen * 100.0) if seen >= 10.0 else None
            row[side + "_seen"] = _clock(seen)
        rows.append(row)
    shown = [r for r in rows if r[focus] is not None]
    if not shown:
        return None
    best = max(shown, key=lambda r: r[focus])
    worst = min(shown, key=lambda r: r[focus])
    read = (f"You held the centre most in round {best['number']} ({best[focus]}% of the time you were seen)"
            + (f" and least in round {worst['number']} ({worst[focus]}%)." if worst is not best else "."))
    return {"rows": rows, "read": read}


def _shifted(series: dict | None, value: str, shift: float, scale: float) -> dict | None:
    """The series with ``value`` mapped to (reading + shift) / scale, so _lines can draw it 0..1."""
    if not series:
        return None
    points = [dict(p, _v=None if p.get(value) is None else (float(p[value]) + shift) / scale)
              for p in series.get("points") or []]
    return {"step_seconds": series["step_seconds"], "points": points}


def pressure_and_distance(series_by_side: dict, focus: str, other: str, start: float, span: float, end: float):
    """Pressure for both fighters (-1 backing straight off, 1 walking straight in) and the gap between them.

    Only from analyses that stored them (2026-10-12 on); None otherwise.
    """
    pressure = distance = None
    if any(p.get("pressure") is not None for s in series_by_side.values() if s for p in s.get("points") or []):
        lines = _lines({side: _shifted(series_by_side[side], "pressure", 1.0, 2.0) for side in series_by_side},
                       "_v", 1.0, start, span)
        if focus in lines:
            mine = [float(p["pressure"]) for p in series_by_side[focus]["points"] if p.get("pressure") is not None]
            theirs = [float(p["pressure"]) for p in ((series_by_side.get(other) or {}).get("points") or [])
                      if p.get("pressure") is not None]
            share = round(sum(1 for v in mine if v > 0.2) / len(mine) * 100)
            read = f"You were walking your opponent down in {share}% of the stretches measured"
            if theirs:
                read += f"; your opponent in {round(sum(1 for v in theirs if v > 0.2) / len(theirs) * 100)}%"
            pressure = {"lines": lines, "mid": HEIGHT / 2.0,
                        "axis": [_clock(start), _clock(start + span / 2.0), _clock(end)], "read": read + "."}
    own = series_by_side.get(focus)
    gaps = [float(p["distance"]) for p in ((own or {}).get("points") or []) if p.get("distance") is not None]
    if gaps:
        top = max(2.5, max(gaps))
        lines = _lines({focus: own}, "distance", top, start, span)
        if focus in lines:
            closest = min((p for p in own["points"] if p.get("distance") is not None), key=lambda p: p["distance"])
            farthest = max((p for p in own["points"] if p.get("distance") is not None), key=lambda p: p["distance"])
            distance = {"lines": lines[focus], "top": round(top, 1),
                        "close_y": round(HEIGHT - min(1.0, 1.0 / top) * HEIGHT, 1),
                        "axis": [_clock(start), _clock(start + span / 2.0), _clock(end)],
                        "read": (f"You were closest around {_clock(closest['t'])} ({closest['distance']:.1f} body "
                                 f"lengths apart) and farthest around {_clock(farthest['t'])} "
                                 f"({farthest['distance']:.1f}).")}
    return pressure, distance


def build(report: dict, focus: str = "A") -> dict | None:
    """Everything the graphs section needs for ``focus``, or None when there is nothing to draw."""
    metrics = report.get("metrics") or {}
    other = "B" if focus == "A" else "A"
    mine = (metrics.get(focus) or {}).get("numbers")
    theirs = (metrics.get(other) or {}).get("numbers")
    if not mine:
        return None

    moved = {side: _split(numbers, MOVED) for side, numbers in ((focus, mine), (other, theirs))}
    # The distance between the two is one shared measure: drawn once. A bar
    # each showed the same distance as two slightly different numbers.
    ranges = {focus: _split(mine, RANGES, mine.get("range_seconds") or {}), other: None}

    series = {focus: mine.get("series"), other: (theirs or {}).get("series")}
    drawn = [s for s in series.values() if s and s.get("points")]
    energy = guard = pressure = distance = None
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
                          "axis": [_clock(start), _clock(start + span / 2.0), _clock(end)],
                          "gaps": _gaps(series[focus], "speed", report)}
        own = series[focus]
        if own and any(p.get("hands_up") is not None for p in own["points"]):
            step = float(own["step_seconds"])
            bars = [{"x": round((float(p["t"]) - start) / span * WIDTH, 1),
                     "width": round(max(1.0, step / span * WIDTH - 1.5), 1),
                     "height": round(float(p["hands_up"]) * HEIGHT, 1),
                     "share": round(float(p["hands_up"]) * 100),
                     "clock": _clock(float(p["t"]))}
                    for p in own["points"] if p.get("hands_up") is not None]
            guard = {"bars": bars, "axis": [_clock(start), _clock(start + span / 2.0), _clock(end)],
                     "gaps": _gaps(own, "hands_up", report),
                     # The opponent's guard as a line over your bars: the
                     # only fair benchmark is the other fighter in the same fight.
                     "theirs": _lines({other: series[other]}, "hands_up", 1.0, start, span).get(other, [])}

        pressure, distance = pressure_and_distance(series, focus, other, start, span, end)

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
            "highlights": highlights(report, focus), "by_round": by_round(report, focus),
            "pressure": pressure, "distance": distance,
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
    step = float((mine or {}).get("step_seconds") or 0.0)
    busiest = max(points, key=lambda p: p["speed"])
    quietest = min(points, key=lambda p: p["speed"])

    def when(point) -> str:
        # The stretch the point stands for, as drawn: one moment named for a
        # five-second average read as a precise time the graph did not show.
        return (f"between {_clock(point['t'])} and {_clock(float(point['t']) + step)}" if step
                else f"at {_clock(point['t'])}")
    text = (f"You were busiest {when(busiest)} ({busiest['speed']:.1f} body lengths a second) "
            f"and quietest {when(quietest)} ({quietest['speed']:.1f}).")
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
