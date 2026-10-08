from __future__ import annotations

import math
from collections.abc import Callable

from core.metric_catalog import BY_KEY
from core.types import StrikeEvent



def count_of(count, word: str, plural: str | None = None) -> str:
    """"1 attempt", "2 attempts" - never "1 attempts" (QA, 2026-10-04)."""
    try:
        n = int(count)
    except (TypeError, ValueError):
        n = 0
    return f"{n} {word if n == 1 else (plural or word + 's')}"

def _pct(value):
    return None if value is None else round(float(value) * 100, 1)


def _event_times(events: list[StrikeEvent], fighter: str, predicate, limit: int = 5) -> list[float]:
    return [round(e.peak_time, 2) for e in events if e.fighter == fighter and predicate(e)][:limit]


def _moment_times(own: dict, key: str, want_low: bool) -> list[float]:
    """Seconds that evidence a movement claim, from core/metrics.py.

    A strike claim has always pointed at the moments behind it; a movement
    claim shipped with an empty list, so the only claims that survive on
    pose-only footage - which is most real footage - were the ones a coach
    could not check. The report template has rendered these as clickable
    timestamps the whole time and simply never received any.

    Low moments evidence something to work on, high moments a strength.
    """
    moments = (own or {}).get("moments") or {}
    entry = moments.get(key) or {}
    times = entry.get("low" if want_low else "high") or []
    return [float(t) for t in times]


def _measured_baseline_drills(fighter: str, own: dict) -> list[dict]:
    """Choose fallback work from this fighter's measured weakest dimensions."""
    attacks = own.get("attacks", {})
    attempts = int(attacks.get("attempts", 0))
    landed = int(attacks.get("clean", 0)) + int(attacks.get("likely_landed", 0))
    accuracy = attacks.get("accuracy")
    guard = own.get("guard_index")
    balance = own.get("balance_index")
    combinations = int(own.get("combinations", {}).get("count", 0))
    defenses = sum(own.get("defenses", {}).values())

    candidates = [
        (
            1.0 if accuracy is None else 1.0 - float(accuracy),
            "Measured accuracy rounds",
            f"4 x 2 min: recreate Fighter {fighter}'s {attempts} detected attempts; count only clean contact and beat the current {landed}/{attempts or 1} result.",
            "Targets the measured conversion rate instead of prescribing a generic combination.",
        ),
        (
            1.0 if guard is None else 1.0 - float(guard),
            "Guard-retention rounds",
            f"3 x 2 min: Fighter {fighter} finishes every exchange in stance and records a successful guard reset before the partner returns.",
            (f"Targets the measured guard: hands up {_pct(guard):.0f}% of the time seen." if guard is not None
             else "Guard was not measured: too little pose evidence."),
        ),
        (
            1.0 if balance is None else 1.0 - float(balance),
            "Post-attack balance audit",
            f"3 x 90 sec: replay Fighter {fighter}'s preferred entries, freeze after the finish, then correct stance before continuing.",
            f"Targets the measured balance index ({_pct(balance) if balance is not None else 'insufficient pose evidence'}%).",
        ),
        (
            1.0 / (1.0 + combinations),
            "Combination-density rounds",
            f"5 x 1 min: Fighter {fighter} must link every first attack to a second legal technique and exceed the detected baseline of {count_of(combinations, 'combination')}.",
            "Targets the fighter's measured combination volume.",
        ),
        (
            1.0 / (1.0 + defenses),
            "Defend-and-return rounds",
            f"4 x 90 sec: Fighter {fighter} earns a repetition only after a visible defence followed by an immediate legal return; beat the detected baseline of {defenses} defences.",
            "Targets the fighter's measured defensive activity.",
        ),
    ]
    # Different stable tie order prevents identical plans when evidence is sparse,
    # while every prescription still cites that fighter's own measurements.
    tie_order = range(len(candidates)) if fighter == "A" else reversed(range(len(candidates)))
    ranked = sorted(zip(candidates, tie_order), key=lambda item: (item[0][0], item[1]), reverse=True)
    return [
        {"name": name, "prescription": prescription, "why": why}
        for (_, name, prescription, why), _ in ranked[:2]
    ]


# What each pose measurement looks like on real footage, as (typical, spread).
#
# These are the numbers coaching has always ranked against - "Guard sits near
# 0.15 on real footage and balance near 0.70" is the note that set them. They
# are lifted to module scope because the REPORT needs them too: a card that
# says "Guard 0.110" and nothing else is a number a reader cannot place, and
# the fix is not to invent a scale but to show the one already in use here.
#
# One source of truth on purpose. Two copies of a reference value drift, and
# then the coaching text and the card disagree about what normal is.

# A difference is only named - a strength, or a thing to work on - when it is
# both big enough to matter and bigger than the measurement's own noise.
#
# QA, 2026-09: "Work on: Guard 7% - You 7%, them 8% - behind your opponent
# here." One point is noise, not a weakness. So:
#   * MIN_EFFECT is the smallest gap worth a coach's time: half the spread the
#     measurement shows between fighters on real footage (POSE_DIMENSIONS) -
#     five points of guard, four of balance. A coaching judgement, not a
#     validated norm.
#   * The gap must also clear 1.96 standard errors of the difference (a 95%
#     interval excluding zero), with each side's error taken over two-second
#     blocks of the fight (core/metrics.py _spread). Older reports carry no
#     spread and are held to the effect size alone.
#   * With no opponent, a number is only a strength or a weakness a whole
#     spread away from where it usually sits (REFERENCE_SPREADS below).
Z_95 = 1.96
REFERENCE_SPREADS = 1.0

# Measurements that are style, not quality, in a given sport. Taekwondo is
# fought with the hands carried low and the legs doing the guarding, so a low
# guard there is how the sport is fought - ranking it as a fault, as for a
# boxer or a K-1 fighter, gave taekwondo players the wrong drill.
STYLE_IN_SPORT = {"taekwondo": frozenset({"guard_index"})}


def clear_difference(key: str, mine: float, theirs: float, own: dict | None = None,
                     opponent: dict | None = None) -> bool:
    """Whether one fighter is genuinely ahead of or behind the other on ``key``."""
    gap = float(mine) - float(theirs)
    if abs(gap) < MIN_EFFECT.get(key, 0.05) - 1e-9:
        return False
    errors = [(((side or {}).get("spread") or {}).get(key) or {}).get("standard_error")
              for side in (own, opponent)]
    if any(error is None for error in errors):
        return True
    combined = math.sqrt(sum(float(error) ** 2 for error in errors))
    return combined <= 0.0 or abs(gap) / combined >= Z_95

POSE_DIMENSIONS = [
    (
        "guard_index", "Guard", (0.17, 0.10),
        "Guard-return audit",
        "4 x 90 sec: after every exchange, freeze in stance and confirm both hands have returned before the partner counters.",
    ),
    (
        "balance_index", "Balance", (0.72, 0.08),
        "Balanced-finish rounds",
        "4 x 90 sec: finish each legal technique in stance, hold for one count, then move without crossing the feet.",
    ),
    (
        "ring_center_control", "Holding the middle", (0.50, 0.15),
        "Centre-line movement rounds",
        "3 x 2 min: use a marked centre lane; exit every exchange at an angle and recover the lane before restarting.",
    ),
    (
        "pressure_index", "Walking them down", (0.03, 0.10),
        "Forward-pressure rounds",
        "4 x 2 min: every time your partner steps back, take the space. Reset if you circle away instead of closing.",
    ),
    (
        "footwork_body_lengths_per_second", "Moving your feet", (1.00, 0.30),
        "Step-count rounds",
        "4 x 2 min: no more than two strikes without changing position. Feet before hands, every exchange.",
    ),
]


# Half of each measurement's spread between fighters (see clear_difference).
MIN_EFFECT = {key: reference[1] / 2 for key, _label, reference, _drill, _prescription in POSE_DIMENSIONS}


def _has_better_direction(key: str, sport: str | None = None) -> bool:
    metric = BY_KEY.get(key)
    if key in STYLE_IN_SPORT.get(str(sport or ""), ()):
        return False
    return metric is not None and metric.direction == "higher"


def build_pose_coaching(fighter: str, own: dict, opponent: dict | None = None,
                        sport: str | None = None) -> dict:
    """Build useful coaching only from identity-safe pose measurements.

    This path deliberately ignores action attempts, contacts and technique
    labels.  It keeps the report useful while the temporal action model is not
    release-validated without laundering its candidates into fight facts.
    """
    dimensions = POSE_DIMENSIONS

    # Ranked by how each number compares with the opponent's same number, not
    # against the fighter's other numbers. Guard sits near 0.15 on real footage
    # and balance near 0.70, so ranking raw values across metrics handed every
    # fighter in every fight the same verdict: balance is your strength, guard
    # and centre are your weaknesses. It said nothing about anybody.
    def _relative_gap(mine: float, theirs: float | None) -> float | None:
        if theirs is None:
            return None
        scale = abs(mine) + abs(theirs)
        if scale < 1e-6:
            return 0.0
        return (mine - theirs) / scale

    opponent = opponent or {}
    measured = []
    for key, label, reference, drill, prescription in dimensions:
        mine = own.get(key)
        if mine is None:
            continue
        gap = _relative_gap(float(mine), opponent.get(key))
        if gap is None:
            # No opponent to compare against, so compare with the band these
            # numbers sit in on real footage. Drawn from four fighters across
            # two bouts - thin, and only used to order a single fighter's own
            # numbers, never shown as a claim about anyone else.
            midpoint, spread = reference
            gap = (float(mine) - midpoint) / max(1e-6, spread)
            measured.append((None, float(mine), key, label, drill, prescription, gap))
            continue
        measured.append((gap, float(mine), key, label, drill, prescription, gap))

    if not measured:
        return {
            "strengths": [],
            "improvements": [],
            "drills": [],
            "evidence_type": "pose_only",
            "baseline_summary": "no reliable pose baseline",
            "note": "No identity-safe pose measurement was available for coaching.",
        }

    # Only a measurement with a known better direction can be a strength or a
    # thing to work on. Pressure, centre and movement have none in
    # core/metric_catalog.py: a counter-fighter gives ground on purpose, so
    # "Work on: Walking them down - behind your opponent" told them their
    # style was a fault. They stay in the baseline summary below.
    ranked_items = [item for item in measured if _has_better_direction(item[2], sport)]
    comparable = [item for item in ranked_items if item[0] is not None]
    # Which gaps are real (see MIN_EFFECT): only these are ever called ahead
    # or behind, in the titles and in the wording alike.
    clear = {
        item[2]: clear_difference(item[2], item[1], float(opponent[item[2]]), own, opponent)
        for item in comparable
    }
    # Ranked on the comparison that exists: against the opponent when there is
    # one, against the reference band when there is not.
    ranked = sorted(ranked_items, key=lambda item: item[6], reverse=True)
    if comparable:
        ordered = sorted(comparable, key=lambda item: item[0], reverse=True)
        strongest = ordered[0]
        # Only things the fighter is actually behind on. Taking the bottom two
        # regardless told a fighter who led on nearly everything to work on a
        # number they were winning, which reads as though nobody looked.
        #
        # And only by a real margin. It used to take any negative gap, so
        # 77.0% against 77.1% became "Work on: Holding the middle" plus a
        # drill, and later 7% against 8% became "Work on: Guard".
        behind = [item for item in ordered if item[0] < 0 and clear[item[2]]]
        weakest = behind[-2:][::-1]
    elif ranked:
        # Only one fighter was analysed. Rank against the reference band, and
        # only name what sits a whole spread outside it: the lowest of two
        # ordinary numbers is not a weakness.
        strongest = ranked[0]
        weakest = [item for item in ranked[::-1]
                   if item is not strongest and item[6] <= -REFERENCE_SPREADS][:2]
    else:
        strongest, weakest = None, []

    def _phrase(item) -> tuple[str, str]:
        gap, mine, key, label, _drill, _prescription, _rank = item
        if key == "pressure_index":
            shown = f"{(mine + 1) / 2 * 100:.0f}"
            unit = " of 100 (50 is neither forward nor back)"
        elif key == "footwork_body_lengths_per_second":
            shown = f"{mine:.1f}"
            unit = " body lengths a second"
        else:
            shown = f"{mine * 100:.0f}%"
            unit = ""
        if gap is None:
            return f"{label} {shown}", f"Measured at {shown}{unit}."
        theirs = opponent.get(key)
        if key == "pressure_index":
            theirs_shown = f"{(float(theirs) + 1) / 2 * 100:.0f}"
        elif key == "footwork_body_lengths_per_second":
            theirs_shown = f"{float(theirs):.1f}"
        else:
            theirs_shown = f"{float(theirs) * 100:.0f}%"
        if not clear.get(key, False):
            side = "no clear difference from"
        else:
            side = "better than" if gap > 0 else "behind"
        return (
            f"{label} {shown}",
            f"You {shown}{unit}, them {theirs_shown} - {side} your opponent here.",
        )

    strengths = []
    # A strength has to be one. The best-ranked number used to be called the
    # strength whatever it was, so a fighter level with or behind their
    # opponent on everything still got one, and the title was the bare number:
    # "Strength: Guard 22%" reads as a weak guard even where 22% was ahead of
    # the opponent's 12%. The title now says what makes it a strength; the
    # numbers stay in the detail.
    if strongest is not None:
        gap, _mine, _key, label = strongest[0], strongest[1], strongest[2], strongest[3]
        if gap is not None and gap > 0 and clear.get(_key, False):
            title = f"{label}: ahead of your opponent"
        elif gap is None and strongest[6] >= REFERENCE_SPREADS:
            title = f"{label}: your best measured area"
        else:
            title = None
        if title is not None:
            strengths.append({
                "title": title,
                "detail": _phrase(strongest)[1],
                "evidence_times": _moment_times(own, strongest[2], want_low=False),
            })
    improvements = []
    drills = []
    ranked_names = " and ".join(item[3].lower() for item in comparable or ranked_items)
    unranked = "Pressure, centre and movement depend on how you fight, so they are not ranked."
    if STYLE_IN_SPORT.get(str(sport or "")):
        unranked = ("Guard, pressure, centre and movement depend on how you fight - in taekwondo "
                    "the hands are carried low by design - so they are not ranked.")
    if comparable and not weakest:
        improvements.append({
            "title": "Nothing behind your opponent",
            "detail": (
                f"On {ranked_names} there was no clear gap in their favour. {unranked} "
                "The next gain is in the striking, which WarriorIQ "
                "can only estimate so far - watch your counted strikes on the "
                "replay and judge them yourself."
            ),
            "evidence_times": [],
        })
    elif ranked_items and not comparable and not weakest:
        improvements.append({
            "title": "Nothing clearly below the usual range",
            "detail": (
                f"Your {ranked_names} sat within the range WarriorIQ usually measures on fight "
                f"footage, so there is no movement fault to name from this video. {unranked}"
            ),
            "evidence_times": [],
        })
    for item in weakest:
        _gap, _mine, _key, label, drill, prescription, _rank = item
        title, detail = _phrase(item)
        improvements.append({
            "title": f"Work on: {title}",
            "detail": detail,
            "evidence_times": _moment_times(own, _key, want_low=True),
        })
        drills.append({
            "name": f"Fighter {fighter} · {drill}",
            "prescription": prescription,
            "why": detail,
            # Carried through so the training plan reads the number off the
            # measurement instead of matching words in the drill's name - which
            # silently gave the pressure and footwork drills a generic goal.
            "metric": _key,
            "label": label,
            "measured": _mine,
            "opponent": opponent.get(_key),
        })
    return {
        "strengths": strengths,
        "improvements": improvements,
        "drills": drills,
        "evidence_type": "pose_only",
        "baseline_summary": ", ".join(
            _phrase(item)[0].lower() for item in measured
        ),
        "note": "Pose-only coaching is shown while automatic action labels remain unvalidated.",
    }


def build_coaching(fighter: str, metrics: dict, events: list[StrikeEvent]) -> dict:
    own = metrics[fighter]
    attacks = own["attacks"]
    strengths: list[dict] = []
    improvements: list[dict] = []
    drills: list[dict] = []

    accuracy = attacks.get("accuracy")
    if accuracy is not None and attacks.get("attempts", 0) >= 4:
        evidence = _event_times(events, fighter, lambda e: e.outcome in {"clean", "likely_landed"})
        if accuracy >= 0.55:
            strengths.append({
                "title": f"Efficient shot selection · {_pct(accuracy)}%",
                "detail": f"Estimated clean/likely-landed rate is {_pct(accuracy)}% across {attacks['attempts']} detected attempts.",
                "evidence_times": evidence,
            })
        elif accuracy < 0.35:
            misses = [e for e in events if e.fighter == fighter and e.outcome == "missed"]
            missed_counts: dict[str, int] = {}
            for event in misses:
                missed_counts[event.technique] = missed_counts.get(event.technique, 0) + 1
            missed_weapon = max(missed_counts, key=missed_counts.get) if missed_counts else "scoring attack"
            missed_label = missed_weapon.replace("_", " ").title()
            improvements.append({
                "title": f"Improve {missed_label} conversion · {missed_counts.get(missed_weapon, 0)} misses",
                "detail": f"Fighter {fighter} converted {_pct(accuracy)}% overall; {missed_label} produced {missed_counts.get(missed_weapon, 0)} verified misses, the largest missed-technique group.",
                "evidence_times": _event_times(events, fighter, lambda e: e.outcome == "missed" and e.technique == missed_weapon),
            })
            drills.append({
                "name": f"{missed_label} correction · {missed_counts.get(missed_weapon, 0)}-miss baseline",
                "prescription": f"3 x 2 min: build every {missed_label.lower()} behind a visible entry, then record clean, blocked and missed outcomes separately.",
                "why": f"Reduces Fighter {fighter}'s {missed_counts.get(missed_weapon, 0)} detected {missed_label.lower()} misses.",
            })

    strongest = own.get("strongest_weapon")
    if strongest:
        strengths.append({
            "title": f"Reliable weapon · {strongest.replace('_', ' ').title()}",
            "detail": f"{strongest.replace('_', ' ').title()} produced the most detected landed actions.",
            "evidence_times": _event_times(events, fighter, lambda e: e.technique == strongest and e.outcome in {"clean", "likely_landed"}),
        })

    guard = own.get("guard_index")
    if guard is not None:
        if guard >= 0.62:
            strengths.append({
                "title": f"Consistent guard · hands up {_pct(guard):.0f}% of the time",
                "detail": f"Hands were up by the face {_pct(guard):.0f}% of the time they could be measured.",
                "evidence_times": _moment_times(own, "guard_index", want_low=False),
            })
        elif guard < 0.42:
            improvements.append({
                "title": f"Guard recovery · hands up {_pct(guard):.0f}% of the time",
                "detail": f"Hands were up by the face {_pct(guard):.0f}% of the time they could be measured, and stayed away from the head after moving or attacking.",
                "evidence_times": _moment_times(own, "guard_index", want_low=True),
            })
            drills.append({
                "name": f"Guard recovery · {_pct(guard):.0f}% hands up",
                "prescription": "4 x 90 sec technical rounds. Every strike must finish with both hands returning to defensive position before the next action.",
                "why": "Builds automatic guard recovery.",
            })

    balance = own.get("balance_index")
    if balance is not None and balance < 0.48:
        improvements.append({
            "title": f"Post-attack balance · {_pct(balance)}% index",
            "detail": f"Balance index was {_pct(balance)}% on frames with usable lower-body pose data.",
            "evidence_times": [],
        })
        drills.append({
            "name": f"Finish in stance · {_pct(balance)}% baseline",
            "prescription": "3 x 2 min on pads: freeze for one count after every combination and verify stance width, posture, and guard.",
            "why": "Reduces over-rotation and makes follow-up defence faster.",
        })

    defense_counts = own.get("defenses", {})
    total_defenses = sum(defense_counts.values())
    if total_defenses >= 3:
        best_defense = max(defense_counts, key=defense_counts.get)
        strengths.append({
            "title": f"Active defence · {count_of(total_defenses, 'action')}",
            "detail": f"Detected {total_defenses} evidence-supported defensive actions; {best_defense} was the most common.",
            "evidence_times": [],
        })

    vulnerabilities = own.get("vulnerability_targets", {})
    if vulnerabilities:
        target = max(vulnerabilities, key=vulnerabilities.get)
        count = vulnerabilities[target]
        if count >= 2:
            improvements.append({
                "title": f"Protect the {target} · {count} scoring actions conceded",
                "detail": f"Opponent had {count} detected clean/likely-landed actions to the {target}.",
                "evidence_times": [
                    round(e.peak_time, 2)
                    for e in events
                    if e.fighter != fighter and e.target == target and e.outcome in {"clean", "likely_landed"}
                ][:6],
            })
            drills.append({
                "name": f"{target.title()} defence · {count}-action baseline",
                "prescription": "Partner technical rounds with the attacker limited to two or three known entries; defender scores only by defending and returning immediately.",
                "why": f"Targets the most common detected scoring area against Fighter {fighter}.",
            })

    combos = own.get("combinations", {})
    if attacks.get("attempts", 0) >= 6 and combos.get("count", 0) == 0:
        improvements.append({
            "title": f"Combination building · {combos.get('count', 0)} detected",
            "detail": "Detected attacks were mostly isolated rather than linked into combinations.",
            "evidence_times": [],
        })
        drills.append({
            "name": "Two-to-four strike chain drill",
            "prescription": "5 x 1 min: alternate hand-hand-kick, hand-kick-hand, and defend-counter combinations without repeating the same finish twice.",
            "why": "Develops layered offense and reduces predictability.",
        })

    counters = own.get("counters", {})
    if counters.get("count", 0) >= 2:
        strengths.append({
            "title": f"Counter timing · {counters['count']} counters",
            "detail": f"Detected {counters['count']} attacks launched within one second of the opponent finishing an action.",
            "evidence_times": counters.get("times", [])[:5],
        })

    if not strengths:
        strengths.append({
            "title": f"Measured activity · {count_of(attacks.get('attempts', 0), 'attempt')}",
            "detail": f"WarriorIQ tracked {count_of(attacks.get('attempts', 0), 'attack attempt')} and {count_of(own.get('combinations', {}).get('count', 0), 'combination')} in the usable evidence window.",
            "evidence_times": _event_times(events, fighter, lambda e: True),
        })
    if not improvements:
        least_effective = min(attacks.get("techniques", {}) or {"scoring sequence": 0}, key=(attacks.get("techniques", {}) or {"scoring sequence": 0}).get)
        improvements.append({
            "title": f"Develop {least_effective.replace('_', ' ').title()} sequences",
            "detail": f"Fighter {fighter}'s evidence shows fewer reliable {least_effective.replace('_', ' ')} actions than their primary weapons. Build this specific option without weakening the current strengths.",
            "evidence_times": [],
        })
    if not drills:
        drills.extend(_measured_baseline_drills(fighter, own))
    return {
        "strengths": strengths[:3],
        "improvements": improvements[:3],
        "drills": drills[:4],
        "note": "Coaching items are generated only when the underlying analysis produced enough evidence; missing items are intentionally not fabricated.",
    }


def _metric_progress(key: str, current: float) -> tuple[float, Callable[[float], str]]:
    """Where one measurement should get to, and how to say it out loud.

    One definition, because the next-session goal and the multi-week
    progression have to agree. Two copies of this arithmetic would drift, and
    an athlete told to reach two different numbers for the same thing stops
    believing either.
    """
    if key == "pressure_index":
        # Stored as -1..1 and spoken as 0..100, so +6 spoken is +0.12 stored.
        return min(0.5, current + 0.12), lambda value: f"{(value + 1) / 2 * 100:.0f} out of 100"
    if key == "footwork_body_lengths_per_second":
        return current + 0.25, lambda value: f"{value:.1f} body lengths a second"
    return min(0.95, current + 0.08), lambda value: f"{_pct(value)}%"


def metric_goal(key: str, current: float) -> tuple[float, Callable[[float], str]]:
    """The training plan's goal for one number, and how to say it.

    For Fight Camp, which draws each mission's bar to the same goal the plan's
    sentence names (see _metric_progress for why there is only one copy).
    """
    return _metric_progress(key, current)


# Four weeks, each changing how the drill is done rather than only how much of
# it. A plan that repeats the same drill at the same intensity is a list, not
# training: the correction has to survive resistance before it survives a fight.
_PROGRESSION_WEEKS: tuple[tuple[str, str, int], ...] = (
    ("Own the shape",
     "No resistance. Slow enough that every repetition is correct, in front of a mirror or camera.", 3),
    ("Against a partner",
     "A partner feeds the situation at roughly half speed and does not try to win.", 3),
    ("Under pressure",
     "Live rounds at fight pace with one rule: the correction is the only thing being judged.", 4),
    ("Prove it",
     "Spar normally without thinking about it, then film a round and run it through WarriorIQ.", 2),
)


def build_training_progression(coaching: dict, fighter: str, own: dict) -> list[dict]:
    """A four-week block that ends where the next-session goal was pointing.

    The report gave a single next session and a target, which tells an athlete
    what to fix but not how to get there, and gives a coach nothing to plan a
    month around. The weekly targets are steps along the same line the goal
    already drew, so week four's number is the goal.
    """
    drills = [
        drill for drill in coaching.get("drills", [])
        if drill.get("metric") is not None and drill.get("measured") is not None
    ][:2]
    if not drills:
        return []
    weeks = []
    for index, (theme, method, sessions) in enumerate(_PROGRESSION_WEEKS, start=1):
        targets = []
        for drill in drills:
            current = float(drill["measured"])
            final, show = _metric_progress(drill["metric"], current)
            step = current + (final - current) * (index / len(_PROGRESSION_WEEKS))
            targets.append({
                "label": drill.get("label", "this measurement"),
                "from": show(current),
                "to": show(step),
                "final": show(final),
            })
        weeks.append({
            "week": index,
            "theme": theme,
            "method": method,
            "sessions_per_week": sessions,
            "work": [drill["prescription"] for drill in drills],
            "targets": targets,
            "check": (
                "Film a round and analyse it. These are the numbers that should have moved."
                if index == len(_PROGRESSION_WEEKS)
                else "Judge the week on whether the shape held, not on how tired you were."
            ),
        })
    return weeks


def build_training_plan(coaching: dict, fighter: str, own: dict) -> list[dict]:
    """Turn this fighter's findings into a measurable next-session schedule."""
    drills = coaching.get("drills", [])
    if not drills:
        return []
    if coaching.get("evidence_type") == "pose_only":
        baseline = coaching.get("baseline_summary", "pose-derived movement measurements")
    else:
        attempts = int(own.get("attacks", {}).get("attempts", 0))
        combinations = int(own.get("combinations", {}).get("count", 0))
        accuracy = own.get("attacks", {}).get("accuracy")
        baseline = f"{count_of(attempts, 'attempt')}, {count_of(combinations, 'combination')}"
        if accuracy is not None:
            baseline += f", {_pct(accuracy)}% conversion"
    def measured_goal(drill: dict) -> str:
        """A target tied to the number the drill was chosen for.

        Reads the metric off the drill rather than matching words in its name.
        The old string matching had no branch for pressure or footwork, so the
        two dimensions that separate fighters most got a generic sentence.
        """
        key = drill.get("metric")
        current = drill.get("measured")
        theirs = drill.get("opponent")
        if key is None or current is None:
            return (
                f"Beat what Fighter {fighter} did in this fight ({baseline}) "
                "without giving up what already went well."
            )
        current = float(current)
        target, show = _metric_progress(key, current)
        if key == "pressure_index":
            goal = f"Move your pressure from {show(current)} to {show(target)}."
        elif key == "footwork_body_lengths_per_second":
            goal = f"Move your feet more: {show(current)} to {show(target)}."
        else:
            goal = f"Raise {drill.get('label', 'this number').lower()} from {show(current)} to {show(target)}."
        if theirs is not None:
            if key == "pressure_index":
                theirs_shown = f"{(float(theirs) + 1) / 2 * 100:.0f}"
            elif key == "footwork_body_lengths_per_second":
                theirs_shown = f"{float(theirs):.1f}"
            else:
                theirs_shown = f"{_pct(float(theirs))}%"
            goal += f" Your opponent was at {theirs_shown}."
        return goal

    plan = []
    for index, drill in enumerate(drills[:4], start=1):
        plan.append({
            "session_block": index,
            "focus": f"Block {index}: {drill['name']}",
            # The prescription names the fighter where it needs to, and the plan
            # is headed with the name: a "Fighter A: " prefix printed the name
            # twice in every line once the page put names in (QA, 2026-10-07).
            "work": drill["prescription"],
            "goal": measured_goal(drill),
            "baseline": baseline,
        })
    return plan


def drop_work_prefix(report: dict) -> dict:
    """Remove the "Fighter A: " that stored plans put before every drill line.

    Reports saved before the prefix was dropped (build_training_plan) still
    carry it; the render-time gate removes it so they read like new ones.
    Idempotent.
    """
    for fighter in ("A", "B"):
        prefix = f"Fighter {fighter}: "
        for item in ((report.get("training_plan") or {}).get(fighter) or []):
            if isinstance(item, dict) and str(item.get("work") or "").startswith(prefix):
                item["work"] = item["work"][len(prefix):]
        for week in ((report.get("training_progression") or {}).get(fighter) or []):
            if isinstance(week, dict) and isinstance(week.get("work"), list):
                week["work"] = [line[len(prefix):] if isinstance(line, str) and line.startswith(prefix) else line
                                for line in week["work"]]
    return report


# What the report says where coaching found nothing to name - the "Keep
# doing", "Fix next" and training-target cards - and why.
#
# QA, 2026-10-07 (/result/b0bc06c741a7): "this fight did not give enough
# evidence" beside "100% seen". The plan was empty because nothing measured
# was clearly behind the opponent - not because too little was seen - and the
# page had one sentence for every empty plan.
TOO_LITTLE = "too_little_measured"
NO_DIRECTION = "nothing_ranked"
NO_CLEAR_GAP = "no_clear_gap"
NOTHING_UNUSUAL = "nothing_unusual"
WITHHELD = "withheld"


def _listed(labels: list[str]) -> str:
    labels = [label.lower() for label in labels]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def coaching_gaps(report: dict, fighter: str, sport: str | None = None) -> dict:
    """Why this fighter has no strength, no fix or no training target.

    ``code`` is one of WITHHELD, TOO_LITTLE, NO_DIRECTION, NO_CLEAR_GAP or
    NOTHING_UNUSUAL; ``keep``, ``fix`` and ``plan`` are the sentences for the
    three cards. Only "too little measured" says anything about the footage.
    """
    report = report or {}
    coaching = (report.get("coaching") or {}).get(fighter) or {}
    mode = (report.get("integrity") or {}).get("coaching_evidence_mode")
    if mode in ("withheld_identity_failure", "withheld_not_a_fight") and coaching.get("note"):
        note = str(coaching["note"])
        return {"code": WITHHELD, "keep": note, "fix": note, "plan": note}

    metrics = report.get("metrics") or {}
    own = metrics.get(fighter) or {}
    other = metrics.get("B" if fighter == "A" else "A") or {}
    measured = [(key, label) for key, label, *_rest in POSE_DIMENSIONS if own.get(key) is not None]
    if not measured:
        seen = own.get("pose_coverage")
        why = own.get("pose_note") or own.get("guard_note")
        text = ("Too little of the body was measured to name anything"
                + (f": the camera saw you for {float(seen) * 100:.0f}% of the video, but guard, balance, "
                   "centre, pressure and movement could not be measured from it" if seen else "")
                + "." + (f" {why}" if why else " A longer clip, filmed head to toe, gives more to measure."))
        return {"code": TOO_LITTLE, "keep": text, "fix": text, "plan": text}

    ranked = [(key, label) for key, label in measured if _has_better_direction(key, sport)]
    names = _listed([label for _, label in measured])
    if not ranked:
        text = (f"{names.capitalize()} were measured, but none has a better direction to aim for here "
                "- they depend on how you fight - so there is nothing to rank as a strength or a fix.")
        return {"code": NO_DIRECTION, "keep": text, "fix": text,
                "plan": text + " A drill is only set from a number you were behind on."}

    ranked_names = _listed([label for _, label in ranked])
    if any(other.get(key) is not None for key, _ in ranked):
        return {
            "code": NO_CLEAR_GAP,
            "keep": f"On {ranked_names} you were not clearly ahead of your opponent - the gaps were too small to call.",
            "fix": f"On {ranked_names} you were not clearly behind your opponent either.",
            "plan": (f"No target from this fight: on {ranked_names} you were not clearly behind your "
                     "opponent, so there was no measured gap to set one from."),
        }
    return {
        "code": NOTHING_UNUSUAL,
        "keep": f"With no opponent to compare against, {ranked_names} sat in their usual range.",
        "fix": f"With no opponent to compare against, {ranked_names} sat in their usual range.",
        "plan": (f"No target from this fight: with no opponent to compare against, {ranked_names} "
                 "did not sit far enough from their usual range to set one."),
    }
