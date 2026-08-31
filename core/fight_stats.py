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


def _technique_breakdown(events: list[dict], trusted: bool) -> dict[str, dict]:
    """Return one outcome-balanced row per supported technique label."""
    if not trusted:
        return {}
    breakdown: dict[str, dict] = {}
    techniques = sorted({str(item.get("technique") or "other") for item in events})
    for technique in techniques:
        matching = [item for item in events if str(item.get("technique") or "other") == technique]
        summary = _family_summary(matching, True)
        summary["family"] = str(matching[0].get("family") or "other")
        breakdown[technique] = summary
    return breakdown


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
        knees = [item for item in own if item.get("family") == "knee"]
        punch = _family_summary(punches, trusted)
        kick = _family_summary(kicks, trusted)
        knee = _family_summary(knees, trusted)
        overall = _family_summary(own, trusted)
        sequences = combinations[fighter]
        successful_combinations = sum(sequence["landed"] >= 2 for sequence in sequences) if trusted else None
        failed_combinations = len(sequences) - successful_combinations if trusted else None
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
            "knee_attempts": knee["attempts"],
            "knees_landed": knee["landed"],
            "knees_missed": knee["missed"],
            "knees_blocked": knee["blocked"],
            "knees_evaded": knee["evaded"],
            "knees_uncertain": knee["uncertain"],
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
            "successful_combinations": successful_combinations,
            "failed_combinations": failed_combinations,
            "combination_success_rate": (
                successful_combinations / len(sequences) if trusted and sequences else None
            ),
            "longest_combination": max((sequence["length"] for sequence in sequences), default=0) if trusted else None,
            "combination_sequences": sequences if trusted else [],
            "technique_breakdown": _technique_breakdown(own, trusted),
            "best_weapon": best_weapon,
            "observation_coverage": coverage[fighter],
            "families": {"punch": punch, "kick": kick, "knee": knee},
            "evidence": overall["evidence"],
            # Compatibility names used by the earlier report surfaces.
            "clean": overall["landed"],
            "blocked_or_checked": overall["blocked"],
        }
        fighters[fighter] = item

    combined_attempts = sum(int(fighters[fighter]["attempts"] or 0) for fighter in ("A", "B"))
    for fighter, opponent in (("A", "B"), ("B", "A")):
        own = fighters[fighter]
        against = fighters[opponent]
        own_attempts = int(own["attempts"] or 0)
        against_attempts = int(against["attempts"] or 0)
        own["initiative_share"] = own_attempts / combined_attempts if combined_attempts else None
        own["attack_mix"] = {
            family: (int(own["families"][family]["attempts"] or 0) / own_attempts if own_attempts else None)
            for family in ("punch", "kick", "knee")
        }
        outcomes_complete = trusted and against_attempts > 0 and int(against["uncertain"] or 0) == 0
        own["defensive_denial_rate"] = (
            (int(against["blocked"] or 0) + int(against["evaded"] or 0)) / against_attempts
            if outcomes_complete else None
        )
        own["clean_exposure_rate"] = (
            int(against["landed"] or 0) / against_attempts if outcomes_complete else None
        )

    rounds: list[dict] = []
    round_numbers = sorted({_round(item) for item in strikes if _round(item) is not None})
    for number in round_numbers:
        round_events = [item for item in strikes if _round(item) == number]
        round_fighters: dict[str, dict] = {}
        for fighter in ("A", "B"):
            own_round = [item for item in round_events if item.get("fighter") == fighter]
            summary = _family_summary(own_round, trusted)
            summary["families"] = {
                family: _family_summary(
                    [item for item in own_round if item.get("family") == family], trusted
                )
                for family in ("punch", "kick", "knee")
            }
            round_fighters[fighter] = summary
        rounds.append({
            "round": number,
            "fighters": round_fighters,
        })

    for fighter in ("A", "B"):
        fighter_rounds = [
            {"round": item["round"], **item["fighters"][fighter]}
            for item in rounds
        ]
        trusted_rounds = [item for item in fighter_rounds if trusted and item["landed"] is not None]
        peak = max(
            trusted_rounds,
            key=lambda item: (int(item["landed"] or 0), int(item["attempts"] or 0), -int(item["round"])),
            default=None,
        )
        first = fighter_rounds[0] if len(fighter_rounds) >= 2 else None
        last = fighter_rounds[-1] if len(fighter_rounds) >= 2 else None
        fighters[fighter]["round_profile"] = {
            "peak_landed_round": None if peak is None else int(peak["round"]),
            "peak_round_landed": None if peak is None else int(peak["landed"] or 0),
            "opening_round": None if first is None else int(first["round"]),
            "closing_round": None if last is None else int(last["round"]),
            "attempt_change": None if first is None else int(last["attempts"] or 0) - int(first["attempts"] or 0),
            "landed_change": (
                None if first is None or first["landed"] is None or last["landed"] is None
                else int(last["landed"] or 0) - int(first["landed"] or 0)
            ),
        }

    comparison = {
        "combined_attempts": combined_attempts,
        "initiative_margin": (
            None if not combined_attempts
            else fighters["A"]["initiative_share"] - fighters["B"]["initiative_share"]
        ),
        "landed_margin": (
            None if not trusted
            else int(fighters["A"]["landed"] or 0) - int(fighters["B"]["landed"] or 0)
        ),
    }

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
        "comparison": comparison,
    }
