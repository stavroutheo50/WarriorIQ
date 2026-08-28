from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


OUTCOMES = ("landed", "missed", "blocked", "evaded", "uncertain")
COMBINATION_GAP_SECONDS = 1.20


def normalize_outcome(value: Any) -> str:
    outcome = str(value or "uncertain").strip().lower()
    if outcome in {"clean", "likely_landed", "landed"}:
        return "landed"
    if outcome in {"blocked", "checked", "parried"}:
        return "blocked"
    if outcome in {"evaded", "slipped"}:
        return "evaded"
    if outcome == "missed":
        return "missed"
    return "uncertain"


def _time(event: dict) -> float:
    try:
        return max(0.0, float(event.get("time_seconds", event.get("peak_time", 0.0)) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _round(event: dict) -> int | None:
    try:
        value = event.get("round_number")
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _deduplicate(events: Iterable[dict]) -> list[dict]:
    """Keep one public record for one physical attempt.

    The action engine already removes most frame-level duplicates. This final
    seam protects Live Analysis, saved reports, and legacy report rebuilds from
    counting the same fighter/limb action twice within 450 ms.
    """
    kept: list[dict] = []
    for source in sorted((dict(item) for item in events if item.get("fighter") in {"A", "B"}), key=_time):
        if source.get("kind", "strike") != "strike":
            continue
        source["outcome"] = normalize_outcome(source.get("outcome"))
        identity = source.get("limb") or source.get("technique") or source.get("family")
        duplicate = next((
            index for index, item in enumerate(kept)
            if item.get("fighter") == source.get("fighter")
            and (item.get("limb") or item.get("technique") or item.get("family")) == identity
            and abs(_time(item) - _time(source)) <= 0.45
        ), None)
        if duplicate is None:
            kept.append(source)
            continue
        if float(source.get("confidence", 0.0) or 0.0) > float(kept[duplicate].get("confidence", 0.0) or 0.0):
            kept[duplicate] = source
    return sorted(kept, key=_time)


def _combination_sequences(events: list[dict]) -> dict[str, list[dict]]:
    """Find uninterrupted same-fighter offensive sequences.

    A sequence requires at least two supported attempts, no opponent attempt
    between them, the same round, and no more than 1.20 seconds between
    attempts. This avoids treating two unrelated attacks as a combination.
    """
    current: dict[str, list[dict]] = {"A": [], "B": []}
    sequences: dict[str, list[dict]] = {"A": [], "B": []}

    def finish(fighter: str) -> None:
        sequence = current[fighter]
        if len(sequence) >= 2:
            sequences[fighter].append({
                "start_time": round(_time(sequence[0]), 3),
                "end_time": round(_time(sequence[-1]), 3),
                "round_number": _round(sequence[0]),
                "length": len(sequence),
                "techniques": [str(item.get("technique") or item.get("family") or "strike") for item in sequence],
                "outcomes": [normalize_outcome(item.get("outcome")) for item in sequence],
                "landed": sum(normalize_outcome(item.get("outcome")) == "landed" for item in sequence),
            })
        current[fighter] = []

    for event in events:
        fighter = str(event.get("fighter"))
        opponent = "B" if fighter == "A" else "A"
        finish(opponent)
        sequence = current[fighter]
        if sequence:
            same_round = _round(sequence[-1]) == _round(event)
            gap = _time(event) - _time(sequence[-1])
            if not same_round or gap > COMBINATION_GAP_SECONDS:
                finish(fighter)
        current[fighter].append(event)
    finish("A")
    finish("B")
    return sequences


def _family_summary(events: list[dict], trusted: bool) -> dict:
    counts = Counter(normalize_outcome(item.get("outcome")) for item in events)
    attempts = len(events)
    uncertain = counts["uncertain"]
    accuracy = None
    if trusted and attempts and uncertain == 0:
        accuracy = counts["landed"] / attempts
    evidence = {
        outcome: [
            {
                "time_seconds": round(_time(item), 3),
                "technique": item.get("technique"),
                "target": item.get("target"),
            }
            for item in events if normalize_outcome(item.get("outcome")) == outcome
        ]
        for outcome in OUTCOMES
    }
    if not trusted:
        return {
            "attempts": attempts,
            "landed": None,
            "missed": None,
            "blocked": None,
            "evaded": None,
            "uncertain": None,
            "accuracy": None,
            "evidence": {},
        }
    return {
        "attempts": attempts,
        **{outcome: int(counts[outcome]) for outcome in OUTCOMES},
        "accuracy": accuracy,
        "evidence": evidence,
    }


def summarize_fight_events(
    events: Iterable[dict],
    found: dict[str, int] | None,
    analyzed_frames: int,
    trusted: bool,
    processed_seconds: float | None = None,
) -> dict:
    strikes = _deduplicate(events)
    combinations = _combination_sequences(strikes) if trusted else {"A": [], "B": []}
    coverage = {
        fighter: (float((found or {}).get(fighter, 0)) / analyzed_frames if analyzed_frames else 0.0)
        for fighter in ("A", "B")
    }
    fighters: dict[str, dict] = {}
    for fighter in ("A", "B"):
        own = [item for item in strikes if item.get("fighter") == fighter]
        punches = [item for item in own if item.get("family") == "punch"]
        kicks = [item for item in own if item.get("family") == "kick"]
        punch = _family_summary(punches, trusted)
        kick = _family_summary(kicks, trusted)
        overall = _family_summary(own, trusted)
        sequences = combinations[fighter]
        landed_techniques = Counter(
            str(item.get("technique") or item.get("family") or "strike")
            for item in own if normalize_outcome(item.get("outcome")) == "landed"
        )
        best_weapon = None
        if trusted and landed_techniques:
            technique, count = landed_techniques.most_common(1)[0]
            best_weapon = {"technique": technique, "landed": int(count)}
        item = {
            "attempts": overall["attempts"],
            "landed": overall["landed"],
            "missed": overall["missed"],
            "blocked": overall["blocked"],
            "evaded": overall["evaded"],
            "uncertain": overall["uncertain"],
            "accuracy": overall["accuracy"],
            "punch_accuracy": punch["accuracy"],
            "kick_accuracy": kick["accuracy"],
            "punch_attempts": punch["attempts"],
            "punches_landed": punch["landed"],
            "punches_missed": punch["missed"],
            "punches_blocked": punch["blocked"],
            "punches_evaded": punch["evaded"],
            "punches_uncertain": punch["uncertain"],
            "kick_attempts": kick["attempts"],
            "kicks_landed": kick["landed"],
            "kicks_missed": kick["missed"],
            "kicks_blocked": kick["blocked"],
            "kicks_evaded": kick["evaded"],
            "kicks_uncertain": kick["uncertain"],
            "total_strikes": overall["attempts"],
            "total_landed": overall["landed"],
            "total_defended": (
                None if not trusted else int(overall["blocked"] or 0) + int(overall["evaded"] or 0)
            ),
            "activity_rate": (
                float(overall["attempts"]) / (float(processed_seconds) / 60.0)
                if processed_seconds and processed_seconds > 0 else 0.0
            ),
            "combinations": len(sequences) if trusted else None,
            "successful_combinations": (
                sum(sequence["landed"] >= 2 for sequence in sequences) if trusted else None
            ),
            "longest_combination": max((sequence["length"] for sequence in sequences), default=0) if trusted else None,
            "combination_sequences": sequences if trusted else [],
            "best_weapon": best_weapon,
            "observation_coverage": coverage[fighter],
            "families": {"punch": punch, "kick": kick},
            "evidence": overall["evidence"],
            # Compatibility names used by the earlier report surfaces.
            "clean": overall["landed"],
            "blocked_or_checked": overall["blocked"],
        }
        fighters[fighter] = item

    rounds: list[dict] = []
    round_numbers = sorted({_round(item) for item in strikes if _round(item) is not None})
    for number in round_numbers:
        round_events = [item for item in strikes if _round(item) == number]
        rounds.append({
            "round": number,
            "fighters": {
                fighter: _family_summary(
                    [item for item in round_events if item.get("fighter") == fighter], trusted
                )
                for fighter in ("A", "B")
            },
        })

    return {
        "action_labels_available": bool(trusted),
        "attempt_counts_available": True,
        "event_mode": "validated_actions" if trusted else "observed_attempts",
        "outcome_invariant": "attempts = landed + missed + blocked + evaded + uncertain",
        "accuracy_definition": "landed / attempts; withheld when attempts are zero or any outcome is uncertain",
        "combination_definition": (
            "Two or more supported attempts by the same fighter, in the same round, no more than "
            "1.20 seconds apart, with no opponent attempt between them."
        ),
        "fighters": fighters,
        "rounds": rounds,
    }
