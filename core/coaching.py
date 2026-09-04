from __future__ import annotations

from collections.abc import Callable

from core.types import StrikeEvent


def _pct(value):
    return None if value is None else round(float(value) * 100, 1)


def _event_times(events: list[StrikeEvent], fighter: str, predicate, limit: int = 5) -> list[float]:
    return [round(e.peak_time, 2) for e in events if e.fighter == fighter and predicate(e)][:limit]


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
            f"Targets the measured guard index ({_pct(guard) if guard is not None else 'insufficient pose evidence'}%).",
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
            f"5 x 1 min: Fighter {fighter} must link every first attack to a second legal technique and exceed the detected baseline of {combinations} combinations.",
            "Targets the fighter's measured combination volume.",
        ),
        (
            1.0 / (1.0 + defenses),
            "Defend-and-return rounds",
            f"4 x 90 sec: Fighter {fighter} earns a repetition only after a visible defense followed by an immediate legal return; beat the detected baseline of {defenses} defenses.",
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


def build_pose_coaching(fighter: str, own: dict, opponent: dict | None = None) -> dict:
    """Build useful coaching only from identity-safe pose measurements.

    This path deliberately ignores action attempts, contacts and technique
    labels.  It keeps the report useful while the temporal action model is not
    release-validated without laundering its candidates into fight facts.
    """
    # Pressure and footwork are the two that actually differ between fighters
    # on real footage, and both were missing from coaching entirely.
    dimensions = [
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
            "Center-line movement rounds",
            "3 x 2 min: use a marked center lane; exit every exchange at an angle and recover the lane before restarting.",
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

    comparable = [item for item in measured if item[0] is not None]
    # Ranked on the comparison that exists: against the opponent when there is
    # one, against the reference band when there is not.
    ranked = sorted(measured, key=lambda item: item[6], reverse=True)
    if comparable:
        ordered = sorted(comparable, key=lambda item: item[0], reverse=True)
        strongest = ordered[0]
        # Only things the fighter is actually behind on. Taking the bottom two
        # regardless told a fighter who led on nearly everything to work on a
        # number they were winning, which reads as though nobody looked.
        behind = [item for item in ordered if item[0] < 0]
        weakest = behind[-2:][::-1]
    else:
        # Only one fighter was analysed. Rank against the reference band.
        strongest = ranked[0]
        weakest = ranked[-2:][::-1]

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
        side = "better than" if gap > 0 else ("level with" if abs(gap) < 0.02 else "behind")
        return (
            f"{label} {shown}",
            f"You {shown}{unit}, them {theirs_shown} - {side} your opponent here.",
        )

    strength_title, strength_detail = _phrase(strongest)
    strengths = [{
        "title": strength_title,
        "detail": strength_detail,
        "evidence_times": [],
    }]
    improvements = []
    drills = []
    if comparable and not weakest:
        improvements.append({
            "title": "Nothing behind your opponent",
            "detail": (
                "On every movement number measured you matched or beat them. "
                "The next gain is in the striking, which WarriorIQ cannot "
                "score yet."
            ),
            "evidence_times": [],
        })
    for item in weakest:
        _gap, _mine, _key, label, drill, prescription, _rank = item
        title, detail = _phrase(item)
        improvements.append({
            "title": f"Work on: {title}",
            "detail": detail,
            "evidence_times": [],
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
                "title": f"Consistent guard · {_pct(guard)}% index",
                "detail": f"Guard-position index was {_pct(guard)}% on frames with sufficient pose evidence.",
                "evidence_times": [],
            })
        elif guard < 0.42:
            improvements.append({
                "title": f"Guard recovery · {_pct(guard)}% index",
                "detail": f"Guard-position index was {_pct(guard)}%. Hands frequently remained far from the head line after movement/attacks.",
                "evidence_times": [],
            })
            drills.append({
                "name": f"Guard recovery · {_pct(guard)}% baseline",
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
            "why": "Reduces over-rotation and makes follow-up defense faster.",
        })

    defense_counts = own.get("defenses", {})
    total_defenses = sum(defense_counts.values())
    if total_defenses >= 3:
        best_defense = max(defense_counts, key=defense_counts.get)
        strengths.append({
            "title": f"Active defense · {total_defenses} actions",
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
                "name": f"{target.title()} defense · {count}-action baseline",
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
            "title": f"Measured activity · {attacks.get('attempts', 0)} attempts",
            "detail": f"WarriorIQ tracked {attacks.get('attempts', 0)} attack attempts and {own.get('combinations', {}).get('count', 0)} combinations in the usable evidence window.",
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
            "work": [f"Fighter {fighter}: {drill['prescription']}" for drill in drills],
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
        baseline = f"{attempts} attempts, {combinations} combinations"
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
                f"Improve on Fighter {fighter}'s measured baseline ({baseline}) "
                "without sacrificing the strongest measured area."
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
            "work": f"Fighter {fighter}: {drill['prescription']}",
            "goal": measured_goal(drill),
            "baseline": baseline,
        })
    return plan
