"""The head-to-head stat sheet on the fight report (owner, 2026-10-28).

"Football apps show far more stats": the report led with the estimated score
and six numbers, and everything else sat under "See all the details". This
reads what the analysis already measured into one side-by-side sheet - a row
per stat with both fighters' values and a split bar - plus two graphs over the
fight: strikes thrown every ten seconds, and how fast each fighter moved.

It only arranges numbers the report already holds. A stat either fighter
lacks is left out rather than shown as zero, strike rows appear only for the
strike families this sport publishes, and the strike rows keep their estimate
label.
"""

from __future__ import annotations

from typing import Any

FAMILY_LABELS = {"punch": "Punches", "kick": "Kicks", "knee": "Knees"}
TIMELINE_BUCKET_SECONDS = 10.0
SIDES = ("A", "B")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None          # NaN is not a measurement


def _row(label: str, a: float | None, b: float | None, fmt: str, *, unit: str = "",
         note: str = "", count: bool = False) -> dict | None:
    if a is None or b is None:
        return None
    total = abs(a) + abs(b)
    share = 50.0 if total == 0 else abs(a) / total * 100.0
    return {"label": label, "a": a, "b": b, "a_text": fmt.format(a), "b_text": fmt.format(b),
            "unit": unit, "note": note, "share_a": round(share, 1), "count": count}


def _strike_rows(stats: dict, families: list[str]) -> list[dict]:
    fighters = stats.get("fighters") or {}
    a, b = fighters.get("A") or {}, fighters.get("B") or {}
    rows = []
    # A side with no count for any family has no total: a missing count is
    # not zero strikes.
    counted = [[_num(side.get(f"{family}_attempts")) for family in families] for side in (a, b)]
    totals = [None if all(v is None for v in values) else sum(v or 0 for v in values) for values in counted]
    rows.append(_row("Strikes thrown", totals[0], totals[1], "{:.0f}", note="estimate", count=True))
    if len(families) > 1:
        for family in families:
            rows.append(_row(FAMILY_LABELS.get(family, family.title()), _num(a.get(f"{family}_attempts")),
                             _num(b.get(f"{family}_attempts")), "{:.0f}", note="estimate", count=True))
    rows.append(_row("Strikes per minute", _num(a.get("activity_rate")), _num(b.get("activity_rate")),
                     "{:.1f}", note="estimate"))
    return [row for row in rows if row]


def _movement_rows(metrics: dict, report: dict) -> list[dict]:
    a, b = metrics.get("A") or {}, metrics.get("B") or {}
    na, nb = a.get("numbers") or {}, b.get("numbers") or {}

    def pct(value):
        value = _num(value)
        return None if value is None else value * 100.0

    def pressure(value):
        value = _num(value)
        return None if value is None else (value + 1.0) / 2.0 * 100.0   # the report's 0-100 scale

    rows = [
        _row("Distance covered", _num(na.get("distance_body_lengths")), _num(nb.get("distance_body_lengths")),
             "{:.0f}", unit="body lengths"),
        _row("Movement speed", _num(a.get("footwork_body_lengths_per_second")),
             _num(b.get("footwork_body_lengths_per_second")), "{:.1f}", unit="body lengths/sec"),
        _row("Time going forward", _num(na.get("forward_seconds")), _num(nb.get("forward_seconds")), "{:.0f}s"),
        _row("Time backing off", _num(na.get("backward_seconds")), _num(nb.get("backward_seconds")), "{:.0f}s"),
        _row("Pressure", pressure(a.get("pressure_index")), pressure(b.get("pressure_index")), "{:.0f}", unit="of 100"),
        _row("Time in the centre", pct(a.get("ring_center_control")), pct(b.get("ring_center_control")), "{:.0f}%"),
        _row("Balance", pct(a.get("balance_index")), pct(b.get("balance_index")), "{:.0f}%"),
        _row("Guard up", pct(na.get("hands_up_share")), pct(nb.get("hands_up_share")), "{:.0f}%", note="experimental"),
        _row("Off-balance moments", _num(na.get("off_balance_count")), _num(nb.get("off_balance_count")), "{:.0f}",
             count=True),
        _row("Seen by the camera", pct(a.get("pose_coverage")), pct(b.get("pose_coverage")), "{:.0f}%"),
    ]
    defences = report.get("defenses")
    if isinstance(defences, list) and defences:
        counts = [sum(1 for d in defences if isinstance(d, dict) and d.get("fighter") == side) for side in SIDES]
        rows.insert(0, _row("Defences seen", float(counts[0]), float(counts[1]), "{:.0f}",
                            note="slips, blocks and evasions", count=True))
    return [row for row in rows if row]


def _counted_strikes(report: dict) -> list[dict]:
    """The strikes the report's totals count, each with a "t" in seconds.

    The totals are built from the live strike feed after one de-duplication
    (core.fight_stats.summarize_fight_events), not from every detected event:
    plotting report["events"] drew 24 strikes beside a total of 18 on the
    sample fight. A report written before the feed existed falls back to the
    attempted events.
    """
    feed = report.get("event_feed")
    if isinstance(feed, list) and feed:
        from core.fight_stats import _deduplicate

        return [{**item, "t": item.get("time_seconds")} for item in _deduplicate(feed)]
    return [{**e, "t": e.get("peak_time")} for e in (report.get("events") or [])
            if isinstance(e, dict) and e.get("attempted")]


def _timeline(report: dict, families: list[str]) -> dict | None:
    events = [{**e, "peak_time": e["t"]} for e in _counted_strikes(report)
              if e.get("fighter") in SIDES and e.get("family") in families and _num(e.get("t")) is not None]
    if not events:
        return None
    end = max(_num((r or {}).get("end_seconds")) or 0.0 for r in (report.get("rounds") or [{}]))
    end = max(end, max(_num(e["peak_time"]) for e in events))
    count = max(1, int(end // TIMELINE_BUCKET_SECONDS) + 1)
    buckets = [{"start": i * TIMELINE_BUCKET_SECONDS, "A": 0, "B": 0} for i in range(count)]
    for event in events:
        index = min(count - 1, int(_num(event["peak_time"]) // TIMELINE_BUCKET_SECONDS))
        buckets[index][event["fighter"]] += 1
    peak = max(max(b["A"], b["B"]) for b in buckets) or 1
    for bucket in buckets:
        minute, second = divmod(int(bucket["start"]), 60)
        bucket["label"] = f"{minute}:{second:02d}"
        bucket["a_height"] = round(bucket["A"] / peak * 100, 1)
        bucket["b_height"] = round(bucket["B"] / peak * 100, 1)
    return {"buckets": buckets, "peak": peak, "bucket_seconds": int(TIMELINE_BUCKET_SECONDS)}


def _pace(metrics: dict) -> dict | None:
    """Each fighter's movement speed through the fight, as SVG polyline points."""
    series = {}
    for side in SIDES:
        points = (((metrics.get(side) or {}).get("numbers") or {}).get("series") or {}).get("points") or []
        clean = [(_num(p.get("t")), _num(p.get("speed"))) for p in points if isinstance(p, dict)]
        series[side] = [(t, v) for t, v in clean if t is not None and v is not None]
    if len(series["A"]) < 2 or len(series["B"]) < 2:
        return None
    t_max = max(t for side in SIDES for t, _ in series[side]) or 1.0
    v_max = max(v for side in SIDES for _, v in series[side]) or 1.0
    width, height = 100.0, 40.0

    def polyline(points):
        return " ".join(f"{t / t_max * width:.1f},{height - v / v_max * (height - 2):.1f}" for t, v in points)

    minutes, seconds = divmod(int(t_max), 60)
    return {"a": polyline(series["A"]), "b": polyline(series["B"]), "max_speed": round(v_max, 1),
            "end_label": f"{minutes}:{seconds:02d}"}


def build_stat_sheet(report: dict, families: list[str] | None, strike_counts_published: bool) -> dict:
    """Rows, a strike timeline and a pace graph for the report's stat sheet."""
    report = report or {}
    families = [f for f in (families or []) if f in FAMILY_LABELS] if strike_counts_published else []
    strikes = _strike_rows(report.get("statistics") or {}, families) if families else []
    movement = _movement_rows(report.get("metrics") or {}, report)
    timeline = _timeline(report, families) if families else None
    pace = _pace(report.get("metrics") or {})
    return {"strikes": strikes, "movement": movement, "timeline": timeline, "pace": pace,
            "available": bool(strikes or movement or timeline or pace)}
