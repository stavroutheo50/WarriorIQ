from __future__ import annotations

import math

from core.config import RULESET_LABELS


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _point(record: dict, fighter: str) -> dict | None:
    report = record.get("report") or {}
    video = report.get("video", {})
    target = video.get("analysis_target", "BOTH")
    focus = video.get("focus_fighter") or target
    if focus not in {"BOTH", fighter}:
        return None
    metrics = report.get("metrics", {}).get(fighter)
    if not metrics:
        return None
    attacks = metrics.get("attacks", {})
    dashboard = metrics.get("dashboard", {})
    ruleset = report.get("setup", {}).get("ruleset", "K1")
    action_trusted = report.get("integrity", {}).get("action_metrics_trusted")
    # Older imported reports did not carry the integrity flag. Preserve their
    # already-measured fields, while new reports explicitly fail closed.
    if action_trusted is None:
        action_trusted = attacks.get("accuracy") is not None or bool(attacks.get("attempts"))
    return {
        "job_id": record.get("job_id"),
        "created_at": record.get("created_at", ""),
        "ruleset": RULESET_LABELS.get(ruleset, ruleset.replace("_", " ").title()),
        "action_trusted": bool(action_trusted),
        "accuracy": _number(attacks.get("accuracy")) if action_trusted else None,
        "attempts": _number(attacks.get("attempts")) if action_trusted else None,
        "activity": _number(dashboard.get("activity_attempts_per_minute")) if action_trusted else None,
        "combinations": _number(dashboard.get("combinations_per_minute")) if action_trusted else None,
        "coverage": _number(metrics.get("pose_coverage")),
        "guard": _number(metrics.get("guard_index")),
        "balance": _number(metrics.get("balance_index")),
        "center": _number(metrics.get("ring_center_control")),
        "footwork": _number(metrics.get("footwork_body_lengths_per_second")),
    }


def _difference(latest: dict, previous: dict, key: str):
    current, old = latest.get(key), previous.get(key)
    if current is None or old is None:
        return None
    return round(float(current) - float(old), 4)


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
        latest_report = measured[-1][0].get("report") or {}
        focus = list(latest_report.get("coaching", {}).get(fighter, {}).get("improvements", []))[:3]
        plan = list(latest_report.get("training_plan", {}).get(fighter, []))[:3]
    measured_coverage = [point["coverage"] for point in points if point.get("coverage") is not None]
    trend_key = next(
        (key for key in ("guard", "balance", "center") if sum(point.get(key) is not None for point in points) >= 2),
        "coverage",
    )
    trend_labels = {
        "guard": "Guard position", "balance": "Post-action balance",
        "center": "Ring-center position", "coverage": "Pose evidence quality",
    }
    return {
        "fighter": fighter,
        "fight_count": len(points),
        "points": points,
        "latest": latest,
        "trends": trends,
        "focus": focus,
        "training_plan": plan,
        "average_coverage": None if not measured_coverage else sum(measured_coverage) / len(measured_coverage),
        "action_fight_count": sum(bool(point.get("action_trusted")) for point in points),
        "trend_key": trend_key,
        "trend_label": trend_labels[trend_key],
    }
