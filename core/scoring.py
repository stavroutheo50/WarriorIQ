from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from core.config import RULESET_LABELS
from core.types import KnockdownEvent, StrikeEvent


# The detector's entire vocabulary is punches, kicks and knees. That is the
# hard boundary on what any sport here can be told about, and it is why each
# profile declares `unobserved`: a discipline whose scoring depends on actions
# the model cannot see must say so in the report rather than quietly counting
# only the part it happens to recognise. A ruleset is not a promise of
# coverage; this field is what keeps the difference honest.
OBSERVABLE_FAMILIES = frozenset({"punch", "kick", "knee"})


@dataclass(frozen=True)
class RuleProfile:
    key: str
    label: str
    ring_sport: bool
    allow_low_kick: bool
    allow_knee: bool
    allow_full_power: bool
    point_stop: bool
    allowed_backfists: frozenset[str]
    sport: str = "kickboxing"
    sport_label: str = "Kickboxing"
    allow_punch: bool = True
    allow_kick: bool = True
    # Punches to the head are forbidden in WT taekwondo and central everywhere
    # else, so it cannot be folded into allow_punch.
    allow_head_punch: bool = True
    # Per-family scoring emphasis. Taekwondo is decided by kicks; boxing has no
    # other family to weigh against. Defaults match the kickboxing baseline.
    family_value: tuple[tuple[str, float], ...] = (("punch", 1.0), ("kick", 1.15), ("knee", 1.15))
    # Actions this discipline scores that the model cannot observe at all.
    unobserved: tuple[str, ...] = ()


def _weights(profile: RuleProfile) -> dict[str, float]:
    return dict(profile.family_value)


RULESETS: dict[str, RuleProfile] = {
    # ---- Kickboxing (WAKO disciplines) -------------------------------------
    "K1": RuleProfile("K1", "K-1", True, True, True, True, False, frozenset({"spinning_backfist"})),
    "LOW_KICK": RuleProfile("LOW_KICK", "Low Kick", True, True, False, True, False, frozenset()),
    "FULL_CONTACT": RuleProfile("FULL_CONTACT", "Full Contact", True, False, False, True, False, frozenset()),
    "POINT_FIGHTING": RuleProfile("POINT_FIGHTING", "Point Fighting", False, False, False, False, True, frozenset({"backfist"})),
    "LIGHT_CONTACT": RuleProfile("LIGHT_CONTACT", "Light Contact", False, False, False, False, False, frozenset()),
    "KICK_LIGHT": RuleProfile("KICK_LIGHT", "Kick Light", False, True, False, False, False, frozenset()),

    # ---- Boxing -------------------------------------------------------------
    # The one discipline the model observes completely: its whole scoring
    # vocabulary is punches, and every punch class is already detected.
    "BOXING": RuleProfile(
        "BOXING", "Boxing", True, False, False, True, False, frozenset(),
        sport="boxing", sport_label="Boxing",
        allow_kick=False,
        family_value=(("punch", 1.0),),
    ),

    # ---- Muay Thai ----------------------------------------------------------
    # Punches, kicks and knees are observed. Elbows, clinch work and sweeps are
    # scored in the sport and are not in the detector's vocabulary, so they are
    # declared unobserved rather than silently excluded from the count.
    "MUAY_THAI": RuleProfile(
        "MUAY_THAI", "Full rules (elbows allowed)", True, True, True, True, False, frozenset({"spinning_backfist"}),
        sport="muay_thai", sport_label="Muay Thai",
        family_value=(("punch", 0.9), ("kick", 1.25), ("knee", 1.25)),
        unobserved=("elbow strikes", "clinch control", "sweeps and dumps"),
    ),
    # Amateur and many promotional cards bar elbows outright. That is not a
    # cosmetic variant: with elbows barred they stop being a scoring action the
    # analysis is blind to, so banning them measurably improves what the report
    # can honestly claim. It is the biggest coverage lever in this sport.
    "MUAY_THAI_NO_ELBOWS": RuleProfile(
        "MUAY_THAI_NO_ELBOWS", "No elbows", True, True, True, True, False, frozenset({"spinning_backfist"}),
        sport="muay_thai", sport_label="Muay Thai",
        family_value=(("punch", 0.9), ("kick", 1.25), ("knee", 1.25)),
        unobserved=("clinch control", "sweeps and dumps"),
    ),

    # ---- WT Taekwondo -------------------------------------------------------
    # Kick-decided, and punches to the head are forbidden. Body punches score
    # but rarely decide a bout, which the weighting reflects. Electronic
    # scoring hardware is not something video can reproduce.
    # ITF (International Taekwon-Do Federation) is the hand-and-foot federation:
    # punches to the head are legal and score, contact is controlled rather than
    # full, and points run 1 for a punch to any legal target, 2 for a kick to the
    # body and 3 for a kick to the head. Jumping-kick bonuses vary between ITF
    # bodies, so they are declared unobserved rather than guessed at.
    "ITF_TAEKWONDO": RuleProfile(
        "ITF_TAEKWONDO", "ITF · International Taekwon-Do Federation", False, False, False, False, False, frozenset(),
        sport="taekwondo", sport_label="Taekwondo",
        allow_head_punch=True,
        family_value=(("punch", 1.0), ("kick", 2.0)),
        unobserved=("organisation-specific jumping and spinning kick bonuses",),
    ),
    "WT_TAEKWONDO": RuleProfile(
        "WT_TAEKWONDO", "WT · World Taekwondo (Olympic)", False, False, False, True, False, frozenset(),
        sport="taekwondo", sport_label="Taekwondo",
        allow_head_punch=False,
        family_value=(("punch", 0.5), ("kick", 1.6)),
        unobserved=("electronic body and head protector scoring", "turning-kick rotation bonuses"),
    ),

    # ---- MMA ----------------------------------------------------------------
    # Standing exchanges only. Takedowns, ground position, control time,
    # ground-and-pound and submissions decide most MMA rounds and none of them
    # are in the detector's vocabulary; the two-fighter pose model is built for
    # upright athletes. This profile is deliberately explicit that a standing
    # striking read is all it offers.
    "MMA": RuleProfile(
        "MMA", "MMA (standing exchanges)", True, True, True, True, False, frozenset({"spinning_backfist"}),
        sport="mma", sport_label="MMA",
        family_value=(("punch", 1.0), ("kick", 1.15), ("knee", 1.2)),
        unobserved=(
            "takedowns and takedown defence",
            "ground position and control time",
            "ground-and-pound",
            "submission attempts",
            "elbow strikes",
        ),
    ),
}


SPORTS: dict[str, tuple[str, ...]] = {
    "kickboxing": ("K1", "LOW_KICK", "FULL_CONTACT", "POINT_FIGHTING", "LIGHT_CONTACT", "KICK_LIGHT"),
    "boxing": ("BOXING",),
    "muay_thai": ("MUAY_THAI", "MUAY_THAI_NO_ELBOWS"),
    "taekwondo": ("ITF_TAEKWONDO", "WT_TAEKWONDO"),
    "mma": ("MMA",),
}


def sport_of(ruleset: str) -> str:
    return RULESETS[normalize_ruleset(ruleset)].sport


def unobserved_actions(ruleset: str) -> tuple[str, ...]:
    """Actions this discipline scores that the analysis cannot see at all.

    Surfaced in the report so a coach reads a striking summary as exactly that,
    rather than as a complete account of the round.
    """
    return RULESETS[normalize_ruleset(ruleset)].unobserved


def sport_unobserved(sport: str) -> tuple[str, ...]:
    """Everything a sport scores that the analysis cannot see, deduplicated.

    A sport is only as covered as its least-covered discipline, so this unions
    across the rulesets rather than picking one. The upload page reads it to
    tell a visitor what the report will be silent about *before* they spend an
    hour uploading a bout that is decided on the ground.
    """
    seen: dict[str, None] = {}
    for key in SPORTS.get(sport, ()):
        for action in RULESETS[key].unobserved:
            seen[action] = None
    return tuple(seen)


def normalize_ruleset(value: str) -> str:
    key = (value or "K1").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "K_1": "K1",
        "LOWKICK": "LOW_KICK",
        "FULLCONTACT": "FULL_CONTACT",
        "POINT": "POINT_FIGHTING",
        "POINT_FIGHT": "POINT_FIGHTING",
        "LIGHT": "LIGHT_CONTACT",
        "KICKLIGHT": "KICK_LIGHT",
        "BOX": "BOXING",
        "MUAYTHAI": "MUAY_THAI",
        "THAI": "MUAY_THAI",
        "TAEKWONDO": "WT_TAEKWONDO",
        "WT": "WT_TAEKWONDO",
        "WTF": "WT_TAEKWONDO",
        "WTF_TAEKWONDO": "WT_TAEKWONDO",
        "TKD": "WT_TAEKWONDO",
        "ITF": "ITF_TAEKWONDO",
        "ITF_TAEKWON_DO": "ITF_TAEKWONDO",
        "MUAY_THAI_NO_ELBOW": "MUAY_THAI_NO_ELBOWS",
        "NO_ELBOWS": "MUAY_THAI_NO_ELBOWS",
    }
    key = aliases.get(key, key)
    if key not in RULESETS:
        raise ValueError(f"Unsupported ruleset: {value}")
    return key


def event_legality(event: StrikeEvent, ruleset: str) -> tuple[bool, str]:
    profile = RULESETS[normalize_ruleset(ruleset)]
    technique = event.technique.lower().replace("_", "")
    # Whole families first: a sport that does not permit a family at all makes
    # every technique in it illegal, whatever the target.
    if event.family == "kick" and not profile.allow_kick:
        return False, f"Kicks are not legal in {profile.label}."
    if event.family == "punch" and not profile.allow_punch:
        return False, f"Punches are not legal in {profile.label}."
    if "backfist" in technique:
        normalized = "spinning_backfist" if "spinning" in technique else "backfist"
        if normalized not in profile.allowed_backfists:
            return False, f"{event.technique.replace('_', ' ').title()} is not legal in {profile.label}."
    if event.family == "knee" and not profile.allow_knee:
        return False, f"Knee strikes are not legal in {profile.label}."
    # WT taekwondo permits punches to the body only; a punch to the head is a
    # penalty, not a score, so it must not be counted as one.
    if event.family == "punch" and event.target == "head" and not profile.allow_head_punch:
        return False, f"Punches to the head are not legal in {profile.label}."
    if event.target == "leg" and not profile.allow_low_kick:
        return False, f"Leg attacks are not legal in {profile.label}."
    if event.target == "leg" and event.family == "punch":
        return False, f"Punches to the legs are illegal in {profile.label}."
    if event.target == "leg" and event.family == "kick" and any(name in technique for name in ("frontkick", "pushkick", "sidekick", "spinningbackkick")):
        return False, f"This kick may not target the thigh in {profile.label}."
    return True, f"Legal technique and target for {profile.label}."


def is_legal_event(event: StrikeEvent, ruleset: str) -> bool:
    return event_legality(event, ruleset)[0]


def _effective_value(event: StrikeEvent, profile: RuleProfile | None = None) -> float:
    """Score one supported action, weighted for the discipline being judged.

    A kick and a punch are not worth the same everywhere: taekwondo is decided
    by kicks, boxing has nothing to weigh a punch against, and Muay Thai
    rewards kicks and knees over hands. Using one table for every sport would
    quietly score a taekwondo round as though it were kickboxing.
    """
    if event.outcome not in {"clean", "likely_landed"}:
        return 0.0
    weights = _weights(profile) if profile else {"punch": 1.0, "kick": 1.15, "knee": 1.15}
    base = weights.get(event.family, 1.0)
    target = {"head": 1.25, "body": 1.0, "leg": 0.95, None: 0.85}.get(event.target, 0.85)
    contact = max(0.45, float(event.contact_confidence))
    confidence = max(0.50, float(event.confidence))
    return base * target * (0.65 + 0.35 * contact) * (0.75 + 0.25 * confidence)


def is_verified_scoring_event(event: StrikeEvent, ruleset: str) -> bool:
    """Only evidence strong enough to support a displayed score is counted."""
    return (
        is_legal_event(event, ruleset)
        and event.outcome in {"clean", "likely_landed"}
        and float(event.confidence) >= 0.72
        and float(event.contact_confidence) >= 0.62
        and event.target in {"head", "body", "leg"}
    )


def deduplicate_scoring_events(events: Iterable[StrikeEvent], window_seconds: float = 0.48) -> tuple[list[StrikeEvent], int]:
    """Keep one candidate for each physical action without erasing combinations.

    The temporal detector can emit the same strike on several adjacent windows.
    Those repeats normally share fighter, limb and family for roughly one action
    cycle. Mutually-exclusive labels at effectively the same timestamp are also
    one candidate. Opposite-limb combinations remain separate even when fast.
    """

    def evidence_score(item: StrikeEvent) -> tuple[float, float]:
        return float(item.contact_confidence), float(item.confidence)

    def collapse(groups: list[list[StrikeEvent]]) -> tuple[list[StrikeEvent], int]:
        return [max(group, key=evidence_score) for group in groups], sum(len(group) - 1 for group in groups)

    kept: list[StrikeEvent] = []
    removed = 0
    for fighter in ("A", "B"):
        own = sorted((event for event in events if event.fighter == fighter), key=lambda event: event.peak_time)

        # Alternative labels produced for the same instant cannot represent two
        # separate techniques by the same fighter.
        instant_groups: list[list[StrikeEvent]] = []
        for event in own:
            if instant_groups and event.peak_time - instant_groups[-1][0].peak_time <= 0.02:
                instant_groups[-1].append(event)
            else:
                instant_groups.append([event])
        instant_kept, instant_removed = collapse(instant_groups)
        removed += instant_removed

        motion_buckets: dict[tuple[str, str], list[StrikeEvent]] = defaultdict(list)
        for event in instant_kept:
            limb = (event.limb or event.technique or "unknown").lower()
            motion_buckets[(event.family.lower(), limb)].append(event)

        for bucket in motion_buckets.values():
            groups: list[list[StrikeEvent]] = []
            for event in sorted(bucket, key=lambda item: item.peak_time):
                if groups and event.peak_time - groups[-1][0].peak_time <= window_seconds:
                    groups[-1].append(event)
                else:
                    groups.append([event])
            motion_kept, motion_removed = collapse(groups)
            kept.extend(motion_kept)
            removed += motion_removed
    return sorted(kept, key=lambda event: event.peak_time), removed


def _point_fighting_points(event: StrikeEvent) -> int:
    """Conservative point-fighting mapping.

    WarriorIQ currently does not classify jumping techniques reliably, so it
    never invents 2/3-point jump bonuses. Hand techniques score 1, body kicks
    1, and head kicks 2 when contact is clean/likely and the technique is legal.
    """
    if event.outcome not in {"clean", "likely_landed"}:
        return 0
    if event.family == "punch":
        return 1
    if event.family == "kick":
        return 2 if event.target == "head" else 1
    return 0


def _continuous_tatami_points(event: StrikeEvent) -> int:
    if event.outcome not in {"clean", "likely_landed"}:
        return 0
    if event.family == "kick" and event.target == "head":
        return 2
    return 1


def score_fight(events: Iterable[StrikeEvent], ruleset: str, round_numbers: Iterable[int], knockdowns: Iterable[KnockdownEvent] | None = None, *, reliable: bool = True) -> dict:
    key = normalize_ruleset(ruleset)
    profile = RULESETS[key]
    rounds = sorted(set(int(r) for r in round_numbers if r is not None))
    by_round: dict[int, dict[str, list[StrikeEvent]]] = {
        r: {"A": [], "B": []} for r in rounds
    }
    kd_counts = {r: {"A": 0, "B": 0} for r in rounds}
    for kd in (knockdowns or []):
        if kd.round_number is not None:
            kd_counts.setdefault(kd.round_number, {"A": 0, "B": 0})[kd.fighter] += 1
    illegal = []

    verified_raw = [event for event in events if is_verified_scoring_event(event, key)]
    verified_events, duplicate_count = deduplicate_scoring_events(verified_raw)
    verified_ids = {id(event) for event in verified_events}
    for event in events:
        if event.round_number is None:
            continue
        if event.round_number not in by_round:
            by_round[event.round_number] = {"A": [], "B": []}
        if id(event) in verified_ids:
            by_round[event.round_number][event.fighter].append(event)
        elif not is_legal_event(event, key) and event.outcome in {"clean", "likely_landed"}:
            item = event.to_dict()
            item["legality_reason"] = event_legality(event, key)[1]
            illegal.append(item)

    result = {
        "ruleset": key,
        "ruleset_label": profile.label,
        "sport": profile.sport,
        "sport_label": profile.sport_label,
        # What this discipline scores that the analysis cannot see at all. A
        # striking read of an MMA round is not a read of the round, and the
        # report has to say which one it is giving you.
        "unobserved_actions": list(profile.unobserved),
        "coverage_note": (
            f"Observed striking only. {profile.sport_label} also scores "
            + ", ".join(profile.unobserved)
            + ", which this analysis cannot see."
        ) if profile.unobserved else "",
        "mode": "estimated_10_point_must" if profile.ring_sport else "estimated_points",
        "rounds": [],
        "totals": {"A": 0, "B": 0},
        "illegal_or_non_scoring_events": illegal,
        "disclaimer": "WarriorIQ scoring is an evidence-based estimate, not an official judge decision.",
        "available": reliable,
        "status": "estimated" if reliable else "insufficient_tracking_evidence",
        "verified_actions_counted": len(verified_events),
        "duplicate_action_candidates_removed": duplicate_count,
    }

    if not reliable:
        result["rounds"] = []
        result["totals"] = {"A": None, "B": None}
        result["winner_estimate"] = None
        result["disclaimer"] = "No score is shown because fighter tracking was not reliable enough for a fair estimate. Re-select both fighters on a clearer frame and analyze again."
        return result

    if profile.ring_sport:
        total_a_rounds = total_b_rounds = 0
        for r in sorted(by_round):
            a_value = sum(_effective_value(e, profile) for e in by_round[r]["A"])
            b_value = sum(_effective_value(e, profile) for e in by_round[r]["B"])
            diff = a_value - b_value
            kd_a = kd_counts.get(r, {}).get("A", 0)
            kd_b = kd_counts.get(r, {}).get("B", 0)
            if kd_b > kd_a:
                a_score, b_score, winner = 10, max(7, 9 - kd_b), "A"
                total_a_rounds += 1
            elif kd_a > kd_b:
                a_score, b_score, winner = max(7, 9 - kd_a), 10, "B"
                total_b_rounds += 1
            elif abs(diff) < 0.35:
                a_score, b_score, winner = 10, 10, "EVEN"
            elif diff > 0:
                a_score, b_score = (10, 8) if diff >= 5.5 and a_value >= b_value * 2.0 + 2.0 else (10, 9)
                winner = "A"
                total_a_rounds += 1
            else:
                a_score, b_score = (8, 10) if -diff >= 5.5 and b_value >= a_value * 2.0 + 2.0 else (9, 10)
                winner = "B"
                total_b_rounds += 1
            result["rounds"].append({
                "round": r,
                "fighter_A": a_score,
                "fighter_B": b_score,
                "winner": winner,
                "effective_action_A": round(a_value, 3),
                "effective_action_B": round(b_value, 3),
                "knockdowns_A": kd_a,
                "knockdowns_B": kd_b,
            })
            result["totals"]["A"] += a_score
            result["totals"]["B"] += b_score
        result["rounds_won"] = {"A": total_a_rounds, "B": total_b_rounds}
    else:
        scorer = _point_fighting_points if key == "POINT_FIGHTING" else _continuous_tatami_points
        for r in sorted(by_round):
            a_points = sum(scorer(e) for e in by_round[r]["A"])
            b_points = sum(scorer(e) for e in by_round[r]["B"])
            result["rounds"].append({
                "round": r,
                "fighter_A": a_points,
                "fighter_B": b_points,
                "winner": "A" if a_points > b_points else "B" if b_points > a_points else "EVEN",
            })
            result["totals"]["A"] += a_points
            result["totals"]["B"] += b_points

    result["winner_estimate"] = (
        "A" if result["totals"]["A"] > result["totals"]["B"]
        else "B" if result["totals"]["B"] > result["totals"]["A"]
        else "EVEN"
    )
    return result
