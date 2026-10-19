from __future__ import annotations

import math

from core.guard import comparable_guard
from core.config import RULESET_LABELS


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def change_words(shown, unit: str) -> str:
    """A change between fights, with its unit.

    Audit 2026-10-14: "+1 pts since last fight" read like the points of the
    game (#39), and "Ring centre -13" had no unit at all (#40). A change in a
    percentage is in percentage points; pressure is on a 0-100 scale.
    """
    if unit == "percent":
        return f"{int(shown):+d} percentage point{'' if abs(int(shown)) == 1 else 's'}"
    if unit == "pressure":
        return f"{int(shown):+d} on the 0-100 scale"
    return f"{float(shown):+.1f} body lengths a second" if unit == "rate" else f"{float(shown):+.1f}"


def identity_failed(record: dict, fighter: str) -> bool:
    """Did the report behind this record say its fighters could not be trusted?

    Such a report tells its reader the result is not safe to use, and the
    Progress page then charted its guard and balance anyway - "Guard 20%,
    +11 pts" from a run whose own page withheld everything. A number the
    report disowns cannot become a trend. Missing flags (legacy rows) are not
    treated as failures; they carry pose coverage only.
    """
    integrity = (record.get("report") or {}).get("integrity") or {}
    if integrity.get("identity_evidence_trusted") is False:
        return True
    return (integrity.get("fighter_identity_trusted") or {}).get(fighter) is False


def followed_side(record: dict, fighter: str) -> str:
    """Which corner the athlete was in on this fight.

    The fight's own focus when it has one - that is the athlete chosen for its
    detailed report - and the profile's default only for a fight analysed as
    "both". It always used the default, so a fight where the athlete was boxed
    second, as Fighter B, vanished from their own progress.
    """
    video = (record.get("report") or {}).get("video", {})
    focus = video.get("focus_fighter") or video.get("analysis_target", "BOTH")
    return focus if focus in {"A", "B"} else fighter


def _share(numbers: dict, key: str, parts) -> float | None:
    """``key``'s share of the seconds in ``parts``, or None when none were measured."""
    values = [numbers.get(part) for part in parts]
    if any(not isinstance(v, (int, float)) for v in values) or sum(values) <= 0:
        return None
    return float(numbers.get(key) or 0.0) / float(sum(values))


def _point(record: dict, fighter: str) -> dict | None:
    report = record.get("report") or {}
    fighter = followed_side(record, fighter)
    if identity_failed(record, fighter):
        return None
    # Too little fight footage to measure: the report itself shows no numbers,
    # so its "0% seen" is not the athlete's form (walkthrough 2026-10-19).
    if (report.get("integrity") or {}).get("fight_footage_sufficient") is False:
        return None
    metrics = report.get("metrics", {}).get(fighter)
    if not metrics:
        return None
    attacks = metrics.get("attacks", {})
    dashboard = metrics.get("dashboard", {})
    ruleset = report.get("setup", {}).get("ruleset", "K1")
    # Missing means unknown, and unknown means withheld.
    #
    # This used to fail *open*: a report with no integrity flag - anything
    # analysed before the flag existed - was trusted if it had any attempts at
    # all, which is nearly every report. The reasoning was that those fields
    # were "already measured", but the detector that measured them is the same
    # one measured at 29% precision, so an older report is not a more reliable
    # one. It is the same numbers with less provenance.
    # The one rule every page uses for strike numbers (core.report
    # strike_numbers_shown; audit 2026-10-14, #8): Fight Camp shows a fight's
    # counts exactly when its report does, as the same estimates.
    from core.report import strike_numbers_shown

    shown_families = strike_numbers_shown(report)
    action_trusted = bool(shown_families)
    family_counts = attacks.get("families") or {}
    return {
        "job_id": record.get("job_id"),
        "created_at": record.get("created_at", ""),
        "side": fighter,
        "ruleset": RULESET_LABELS.get(ruleset, ruleset.replace("_", " ").title()),
        "action_trusted": bool(action_trusted),
        "accuracy": _number(attacks.get("accuracy")) if action_trusted else None,
        # Leg strikes, not every attempt. The report stopped publishing punch
        # counts because they were overstated by eleven in two of three bouts
        # checked against video; a progress chart summing them across fights
        # inherits that error and compounds it over time.
        "attempts": _number(sum(float(family_counts.get(family) or 0) for family in shown_families)
                            ) if action_trusted else None,
        "activity": _number(dashboard.get("activity_attempts_per_minute")) if action_trusted else None,
        "combinations": _number(dashboard.get("combinations_per_minute")) if action_trusted else None,
        "coverage": _number(metrics.get("pose_coverage")),
        # Only a guard measured the current way joins a trend (core/guard.py).
        "guard": _number(comparable_guard(metrics)),
        "balance": _number(metrics.get("balance_index")),
        "center": _number(metrics.get("ring_center_control")),
        "footwork": _number(metrics.get("footwork_body_lengths_per_second")),
        # Pose-only, like guard and balance: how directly the fighter walked
        # forward (-1..1), and shares of the fight going forward and at close
        # range (core/fight_numbers.py).
        "pressure": _number(metrics.get("pressure_index")),
        "forward": _number(_share(metrics.get("numbers") or {}, "forward_seconds",
                                  ("forward_seconds", "backward_seconds", "circling_seconds", "standing_seconds"))),
        "close": _number(_share((metrics.get("numbers") or {}).get("range_seconds") or {}, "close",
                                ("close", "middle", "long"))),
    }


def _difference(latest: dict, previous: dict, key: str):
    current, old = latest.get(key), previous.get(key)
    if current is None or old is None:
        return None
    return round(float(current) - float(old), 4)


# Each measure the "every measure" grid can chart, how its values read, and
# which way is up for the arrow. None: no direction is "better" (pressure,
# going forward and being close are style, not quality).
TREND_CARDS = (
    ("guard", "Guard up", "percent", True),
    ("balance", "Balance", "percent", True),
    ("center", "Ring centre", "percent", None),
    ("footwork", "Footwork", "speed", None),
    ("pressure", "Pressure", "pressure", None),
    ("forward", "Going forward", "percent", None),
    ("close", "At close range", "percent", None),
)
CARD_WIDTH, CARD_HEIGHT = 120.0, 36.0


def _card_value(value: float, unit: str) -> str:
    """In the units every other page uses (core/squad.movement_value): pressure
    out of 100, footwork to one decimal."""
    if unit == "percent":
        return "%.0f%%" % (value * 100)
    if unit == "speed":
        return "%.1f" % value
    return "%.0f" % ((value + 1) / 2 * 100)


def trend_cards(points: list[dict]) -> list[dict]:
    """A small line per measure across the athlete's fights, with first-to-latest change.

    Only measures with at least two fights measured; the first and the latest
    are the fights that have the measure, each named by date.
    """
    cards = []
    for key, label, unit, higher_is_better in TREND_CARDS:
        series = [(p["created_at"], float(p[key])) for p in points if p.get(key) is not None]
        if len(series) < 2:
            continue
        values = [v for _, v in series]
        low, high = min(values), max(values)
        spread = high - low or 1.0
        step = CARD_WIDTH / (len(values) - 1)
        line = " ".join(f"{i * step:.1f},{CARD_HEIGHT - 3 - (v - low) / spread * (CARD_HEIGHT - 6):.1f}"
                        for i, v in enumerate(values))
        first, latest = values[0], values[-1]
        change = latest - first
        # From the figures the card prints, so "55 ... 62" reads "+7", never "+8".
        def printed(value):
            return float(_card_value(value, unit).rstrip("%"))
        shown = printed(latest) - printed(first)
        shown = int(shown) if unit in ("percent", "pressure") else round(shown, 1)
        if shown == 0:
            direction = "flat"
        elif higher_is_better is None:
            direction = "neutral"
        else:
            direction = "up" if (change > 0) == higher_is_better else "down"
        change_text = change_words(shown, unit) if shown else "no change"
        cards.append({"key": key, "label": label, "line": line, "fights": len(values),
                      "latest": _card_value(latest, unit), "first": _card_value(first, unit),
                      "first_at": series[0][0], "latest_at": series[-1][0],
                      "change": change_text, "direction": direction,
                      "last_x": round((len(values) - 1) * step, 1),
                      "last_y": round(CARD_HEIGHT - 3 - (latest - low) / spread * (CARD_HEIGHT - 6), 1)})
    return cards


def build_progress(records: list[dict], fighter: str) -> dict:
    fighter = fighter.upper() if fighter.upper() in {"A", "B"} else "A"
    ordered = sorted(records, key=lambda item: item.get("created_at", ""))
    measured = [(record, point) for record in ordered if (point := _point(record, fighter)) is not None]
    points = [point for _, point in measured]
    latest = points[-1] if points else None
    previous = points[-2] if len(points) > 1 else None
    trends = {
        "accuracy": _difference(latest, previous, "accuracy") if latest and previous else None,
        "activity": _difference(latest, previous, "activity") if latest and previous else None,
        "combinations": _difference(latest, previous, "combinations") if latest and previous else None,
        "coverage": _difference(latest, previous, "coverage") if latest and previous else None,
        "guard": _difference(latest, previous, "guard") if latest and previous else None,
        "balance": _difference(latest, previous, "balance") if latest and previous else None,
        "center": _difference(latest, previous, "center") if latest and previous else None,
    }
    focus: list[dict] = []
    plan: list[dict] = []
    if measured:
        latest_record, latest_point = measured[-1]
        latest_report = latest_record.get("report") or {}
        side = latest_point["side"]
        focus = list(latest_report.get("coaching", {}).get(side, {}).get("improvements", []))[:3]
        plan = list(latest_report.get("training_plan", {}).get(side, []))[:3]
    measured_coverage = [point["coverage"] for point in points if point.get("coverage") is not None]
    trend_key = next(
        (key for key in ("guard", "balance", "center") if sum(point.get(key) is not None for point in points) >= 2),
        "coverage",
    )
    # The newest measured value of each movement number, with its fight. The
    # newest fight often lacks one (the athlete was barely seen), and showing
    # "—" there while the squad table showed the same number from the fight
    # before read as a contradiction. Each value says which fight it is from.
    recent = {}
    for key in ("guard", "balance", "center", "coverage"):
        found = next((point for point in reversed(points) if point.get(key) is not None), None)
        if found is not None:
            recent[key] = {"value": found[key], "created_at": found["created_at"],
                           "job_id": found["job_id"], "is_latest": found is latest}
    # Lead with a movement number the athlete can train, not with how clearly
    # the camera saw them; coverage leads only when nothing else was measured.
    headline_key = trend_key if trend_key != "coverage" else next(
        (key for key in ("guard", "balance", "center") if key in recent), "coverage")
    trend_labels = {
        "guard": "Guard position", "balance": "Post-action balance",
        "center": "Ring-centre position", "coverage": "How clearly you were seen",
    }
    return {
        "fighter": fighter,
        "fight_count": len(points),
        # Said on the page, so a library of twenty fights showing three points
        # explains itself instead of looking like lost data.
        "identity_failed_count": sum(
            identity_failed(record, followed_side(record, fighter)) for record in ordered),
        # Each fight left out, newest first, so the page can link it to the
        # who-is-who step instead of only counting it (C5, QA 2026-10-11).
        "identity_failed_fights": [
            {"job_id": record.get("job_id"), "created_at": record.get("created_at", "")}
            for record in reversed(ordered) if identity_failed(record, followed_side(record, fighter))],
        "points": points,
        "latest": latest,
        "trends": trends,
        "focus": focus,
        "training_plan": plan,
        "average_coverage": None if not measured_coverage else sum(measured_coverage) / len(measured_coverage),
        "action_fight_count": sum(bool(point.get("action_trusted")) for point in points),
        "trend_key": trend_key,
        "trend_label": trend_labels[trend_key],
        "recent": recent,
        "headline_key": headline_key,
        "headline_label": trend_labels[headline_key],
        "cards": trend_cards(points),
        "card_width": CARD_WIDTH, "card_height": CARD_HEIGHT,
    }
