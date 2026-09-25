from __future__ import annotations

import json
from html import escape
from pathlib import Path

from core.coaching import (
    POSE_DIMENSIONS,
    build_coaching, build_pose_coaching, build_training_plan, build_training_progression,
)
from core.sport_profiles import build_sport_coaching
from core.config import SETTINGS
from core.evidence_trust import automated_evidence_trust
from core.scoring import (
    event_legality, is_legal_event, is_verified_scoring_event,
    minimum_kicks_per_round, score_fight,
)
from core.types import AnalysisRequest, DefenseEvent, RoundSpec, StrikeEvent


# What each index should be read against, keyed by metric name. Built from
# coaching's POSE_DIMENSIONS so there is one source of truth: if the reference
# for guard changes, the coaching sentence and the report card change together.
# Outcomes where the striking limb actually arrived at the opponent.
#
# Hand-checking all 60 events of athens_hd against the video put a number on
# each of the analyser's own outcomes, and they are not equally trustworthy:
#
#     likely_landed   100% real     blocked   100% real
#     clean            80%          checked    50%
#     missed           42%          uncertain  12%
#
# `missed` is also exactly the set whose limb never entered the opponent's box
# - 0 of 19 reached - so the report was filling its evidence list with the two
# categories the analyser is worst at. Publishing only the arrived ones takes
# the timeline from 59% to 91% correct on that fight.
#
# **This is one fight, 54 labelled events.** It filters what the report shows;
# it deliberately does not touch `events`, so the underlying stream is intact
# for training and for the next fight that can be labelled to check this.
ARRIVED_OUTCOMES = frozenset({"clean", "likely_landed", "blocked"})

# How tall a fighter must be in the network's input before punches are counted.
#
# Punches were withheld outright, on this evidence from hand-checking three
# bouts against the video:
#
#     fight 1   punches reported 11, actually thrown 0
#     fight 2   punches reported  3, actually thrown 2
#     fight 3   punches reported 14, actually thrown 3
#
# All three are messenger copies where a fighter is 56-76 px in the source and
# reaches the network about 86-101 px tall. On the iPhone original, where a
# fighter is 268 px and reaches the network at 201 px, the same detector and
# the same hand-checking give a different answer:
#
#     everything called a punch               27 events   56% real
#       of those, only the ones that arrived  13 events   85% real
#          outcome blocked                     5 events  100% real
#          outcome likely_landed               3 events  100% real
#          outcome uncertain                   8 events   12% real
#
# and not one punch proposal turned out to be a kick misnamed. So a punch is
# not untrustworthy in itself - it is untrustworthy when the hand throwing it
# is a dozen pixels wide. Withholding every one of them threw away eleven real
# punches on that fight to avoid two false ones.
#
# The threshold sits between the two groups rather than at the edge of either,
# because this is one fight's evidence on the permissive side against three
# fights' on the strict side. core/preflight.py measures the number on every
# analysis, so no new measurement is needed to apply it.
PUNCHES_NEED_THIS_MANY_PIXELS = 150.0

# A punch has to have been stopped or seen to land; "clean" is not enough.
#
# Kicks may use the full ARRIVED set - hand-checked, all three of its outcomes
# are 100% real on the fight with labels. Punches are not the same. Broken out
# by outcome on that fight:
#
#     blocked         5 real,  0 wrong   100%
#     likely_landed   3 real,  0 wrong   100%
#     clean           3 real,  2 wrong    60%
#
# Published with `clean` included, the evidence list ran 17 of 21 correct and
# BOTH of the wrong rows were clean punches - one on a fighter standing apart
# doing nothing, one crediting B for a kick A threw. Dropping that one outcome
# costs three real punches and removes both errors. A coach who reads "clean
# punch" about a moment where nothing happened stops believing the rest of the
# page, so the cheaper mistake is to list fewer.
PUNCH_OUTCOMES = frozenset({"likely_landed", "blocked"})


_TYPICAL = {key: reference[0] for key, _label, reference, *_rest in POSE_DIMENSIONS}


def _metric_row(label: str, value, key: str) -> str:
    """One card row, with the figure this number should be compared against.

    A bare "Guard 0.110" cannot be read by anybody. It is not a percentage of
    anything a coach knows, and four of the six rows on each fighter card were
    exactly that. The reference is not a target and is not presented as one -
    it is what the measurement tends to sit at on real footage, which is enough
    for a reader to tell whether 0.110 is unusual.
    """
    if value is None:
        return f"<tr><td>{escape(label)}</td><td>Unavailable</td><td class='muted'></td></tr>"
    typical = _TYPICAL.get(key)
    context = "" if typical is None else f"typical ≈ {typical:.2f}"
    return (f"<tr><td>{escape(label)}</td><td>{float(value):.3f}</td>"
            f"<td class='muted'>{context}</td></tr>")


def _timeline_event_reliable(event: StrikeEvent) -> bool:
    """Keep only actions whose motion, anatomy and outcome all have evidence."""
    # A public "key moment" must be directly supported. Ambiguous likely
    # contacts stay out. A miss appears only when its distance evidence passes
    # the same strict action/contact/identity gates as landed moments.
    if event.outcome not in {"clean", "blocked", "checked", "missed"}:
        return False
    if float(event.confidence) < 0.84:
        return False
    if float(event.contact_confidence) < 0.86:
        return False
    conf = event.evidence.get("contact_attacker_conf") or event.evidence.get("peak_attacker_conf") or []
    endpoint = {"left_hand": 9, "right_hand": 10, "left_leg": 15, "right_leg": 16, "left_knee": 13, "right_knee": 14}.get(event.limb)
    if endpoint is None or len(conf) <= endpoint or float(conf[endpoint]) < 0.50:
        return False
    if event.family == "punch" and event.target == "leg":
        return False
    if float(event.metadata.get("attacker_identity_confidence", 1.0)) < 0.70:
        return False
    if float(event.metadata.get("opponent_identity_confidence", 1.0)) < 0.70:
        return False
    return True


def _identity_seed_safe(tracking: dict, fighter: str) -> bool:
    """Reject legacy/invalid detector seeds without distrusting manual anchors."""
    source_key = f"fighter_{fighter}_seed_source"
    iou_key = f"initial_iou_{fighter}"
    source = tracking.get(source_key)
    overlap = tracking.get(iou_key)
    # Older synthetic reports and tests predate seed diagnostics. Their caller
    # remains responsible for identity trust; real analyses now always record
    # both fields.
    if source is None and overlap is None:
        return True
    if source == "manual_anchor":
        return True
    return source == "pose_detector" and float(overlap or 0.0) >= SETTINGS.min_initial_iou


MIN_COVERAGE_TO_REPORT_OBSERVED = 0.15


# Has anybody established how often a flagged action really happened?
#
# No. Measured once, by hand, on 178 clips across the three reference fights:
# **about a third** of displayed actions were real. The rest were the
# opponent's strike credited to the defender, a moment where nothing happened,
# or the referee. Three fights labelled by one reader is a finding, not a
# validated precision figure - but it is far more than enough to stop the
# report claiming these counts are minimums, and to stop it confirming a
# federation obligation from them.
#
# Flip this to True only when precision has been measured on held-out footage
# nobody tuned against. See project-detector-measured-on-178-clips.
STRIKE_COUNTS_PRECISION_VALIDATED = False


def observed_summary(report: dict) -> dict | None:
    """What we can stand behind when the scorecard cannot be given.

    A scorecard is a comparative claim - this fighter beat that one - and it
    needs both fighters followed well enough to compare. That bar is often not
    met, and the page then said "Not scored" and nothing else, throwing away
    everything the analysis did establish.

    This is the middle setting the report never had. It is deliberately not a
    score and never a comparison:

      * **the count is not a floor.** It was written as one - "at least N,
        so the real number is higher" - and 178 hand-checked clips across the
        three reference fights say the opposite: of the actions the report
        displayed, roughly a third were real. Two in three were the opponent's
        strike credited to the defender, a moment where nothing happened, or
        the referee. A count that is inflated must never be presented as a
        minimum, so the wording claims neither direction now.
      * the denominator travels with the number. "Twelve actions" is a claim
        about the fight; "twelve in the 40% we could follow you" is a claim
        about the footage, and only the second one is true.
      * a fighter followed too little to have a meaningful denominator is left
        out entirely rather than given a small number that reads as a quiet one.
      * **kicks only.** Punches are counted internally and deliberately not
        reported. Hand-checking every proposed event in three fights against
        the video gave this:

              fight 1   punches reported 11, actually thrown  0
              fight 2   punches reported  3, actually thrown  2
              fight 3   punches reported 14, actually thrown  3

              fight 1   kicks   reported  7, actually thrown  7
              fight 2   kicks   reported  4, actually thrown  4
              fight 3   kicks   reported  9, actually thrown  8

        The kick count is right in all three bouts and the punch count is
        inflated by eleven in two of them. Individual kick events are only 27%
        precise, but the errors cancel almost exactly - false kicks are offset
        by missed ones - so the *count* survives even though the *events* do
        not. Punches have no such luck: at this framing an arm extension is a
        few pixels and the detector proposes them from noise.

        Naming a jab against a cross is a separate and even weaker claim, which
        is why `action_labels_available` is false and the technique breakdown
        is already empty.
      * no outcomes. Whether a strike landed rests on contact classification
        that is not validated, so the unit is the action attempted.

    The counts come from the same statistics block the rest of the report uses
    rather than being recounted from the raw event list. Two honest numbers for
    the same thing on one page is worse than either of them alone, and the raw
    list has not been through the confidence bar the statistics apply.
    """
    statistics = (report.get("statistics") or {}).get("fighters") or {}
    if not statistics:
        return None
    out = {}
    for fighter in ("A", "B"):
        item = statistics.get(fighter) or {}
        coverage = float(item.get("observation_coverage") or 0.0)
        if coverage < MIN_COVERAGE_TO_REPORT_OBSERVED:
            continue
        # Knees are not counted, and the reason they used to be is worth
        # keeping: a knee and a round kick are both a leg arriving, so a
        # misnamed knee was still a leg and the count survived. Family-level
        # labelling of the HD bout says the confusion does not stop at the
        # leg. Of five proposed knees, none was a knee - three were kicks and
        # **two were punches**.
        #
        # A bucket that is 40% the family this report deliberately withholds
        # cannot be published, whatever the label on it says. Dropping it
        # loses three real kicks per five proposals, which is an under-count,
        # and under-counting is the direction everything here already errs in.
        kicks = int(item.get("kick_attempts") or 0)
        punches_withheld = int(item.get("punch_attempts") or 0)
        knees_withheld = int(item.get("knee_attempts") or 0)
        if kicks <= 0:
            continue
        out[fighter] = {
            "followed_share": coverage,
            "actions_evidenced": kicks,
            "families": {"kick": kicks},
            # Surfaced so the page can say the omission is deliberate rather
            # than leaving a coach wondering why their boxer threw nothing.
            "punches_withheld": punches_withheld,
            "knees_withheld": knees_withheld,
        }
    if not out:
        return None
    return {
        "fighters": out,
        "basis": "leg strikes the analysis flagged while it had sight of that fighter",
        "punches_reported": False,
        # Precision has been measured once, by hand, on three fights: about a
        # third of displayed actions were real. That is too small a sample to
        # publish as a product claim and far too weak to call the count a
        # minimum. Neither direction is asserted until there is a validated
        # number to assert. See project-detector-measured-on-178-clips.
        "is_a_floor": STRIKE_COUNTS_PRECISION_VALIDATED,
        "precision_validated": STRIKE_COUNTS_PRECISION_VALIDATED,
    }


def unattributed_kick_total(report: dict) -> dict | None:
    """One kick total for the whole fight, for when identity failed.

    A failed identity check means the report cannot say whose strikes were
    whose, and the page used to answer that with three different totals from
    three different sources: kicks per fighter (31), leg strikes including
    knees (34) and arrived kicks from the raw event list (24). This is the one
    number that survives - the statistics block's kick attempts, summed - with
    landed kept separate and given only when the statistics trusted outcomes.
    Never split by fighter, because that split is the thing that failed.
    """
    fighters = (report.get("statistics") or {}).get("fighters") or {}
    if not fighters:
        return None
    rows = [fighters.get(fighter) or {} for fighter in ("A", "B")]
    attempts = sum(int(row.get("kick_attempts") or 0) for row in rows)
    landed_values = [row.get("kicks_landed") for row in rows]
    landed = (sum(int(value) for value in landed_values)
              if all(value is not None for value in landed_values) else None)
    return {"attempts": attempts, "landed": landed}


def kick_minimum_check(report: dict) -> dict | None:
    """Whether each round meets its discipline's obligatory kick count.

    WAKO Full Contact, Chapter 8 Article 6: a kickboxer "is obliged to deliver
    a minimum of 6 kicks per round", and a shortfall not made up in the next
    round costs a minus point. It is the only WAKO obligation this analysis can
    check, because it is a count of kicks *thrown* - the rule asks only that
    the kickboxer "clearly show the intention to hit the opponent by kicking" -
    and attempts are what the model produces.

    **This can confirm compliance and can never allege a shortfall**, and the
    asymmetry is the whole design. Every count here is a floor: it is what we
    could evidence while we had sight of that fighter, and coverage on real
    tournament footage runs well under half. Six or more evidenced means the
    obligation was met, whatever we missed. Three evidenced means three that we
    saw, and the fighter may well have thrown nine. Reporting that as a
    shortfall would be accusing an athlete of a penalty on the strength of our
    own dropped frames.

    Knees do NOT count toward the kick total, and used to. The old reasoning
    was that in Full Contact a knee is illegal, so a leg arriving must be a
    kick the family classifier misnamed. Family-level labelling of the HD bout
    says the misnaming is wider than that: of five proposed knees, three were
    kicks and two were punches. Counting them would credit a kickboxer with
    kicks they did not throw, against a rule that carries a minus point.

    Excluding them only lowers a floor that can confirm compliance and can
    never allege a shortfall, so it costs nothing a fighter can be penalised
    for. Same measurement, same direction, as `observed_summary`.
    """
    ruleset = ((report.get("scorecard") or {}).get("ruleset")
               or (report.get("request") or {}).get("ruleset"))
    if not ruleset:
        return None
    try:
        minimum = minimum_kicks_per_round(ruleset)
    except ValueError:
        return None
    if not minimum:
        return None
    rounds = (report.get("statistics") or {}).get("rounds") or []
    if not rounds:
        return None

    out = []
    for item in rounds:
        fighters = {}
        for fighter in ("A", "B"):
            families = ((item.get("fighters") or {}).get(fighter) or {}).get("families") or {}
            evidenced = int((families.get("kick") or {}).get("attempts") or 0)
            fighters[fighter] = {
                "kicks_evidenced": evidenced,
                # True only when the floor alone clears the bar. False here
                # means "not established from this footage", never "failed",
                # which is why the key is not called `met` on its own.
                #
                # And it stays False entirely while the count is unvalidated.
                # "You met the minimum" is an affirmative claim about a rule,
                # and it was safe only because the count was assumed to be a
                # floor. Measurement says the count is inflated, not
                # conservative, so confirming compliance from it could tell a
                # kickboxer they satisfied WAKO Article 6 when they did not.
                # Refusing to confirm costs a feature; confirming wrongly
                # costs somebody a bout.
                "minimum_confirmed_met": (
                    STRIKE_COUNTS_PRECISION_VALIDATED and evidenced >= minimum
                ),
            }
        out.append({"round": item.get("round"), "fighters": fighters})

    return {
        "minimum": minimum,
        "rule": "WAKO Full Contact, Chapter 8 Article 6: minimum 6 kicks per round, 18 per bout.",
        "rounds": out,
        "basis": "kicks the analysis flagged while it had sight of that fighter",
        "is_a_floor": STRIKE_COUNTS_PRECISION_VALIDATED,
        "precision_validated": STRIKE_COUNTS_PRECISION_VALIDATED,
        "note": (
            "A round can be confirmed as meeting the minimum but never shown to have "
            "missed it: an unconfirmed round is one we could not follow closely enough, "
            "not a shortfall by the fighter."
        ),
    }


def refresh_identity_integrity(report: dict) -> dict:
    """Apply the current identity safety gate to new and legacy reports.

    This prevents an older high-coverage wrong-person track from remaining
    usable after the lock policy improves.  It also upgrades safe legacy
    reports with pose-only coaching when action labels are still unvalidated.
    """
    tracking = report.setdefault("tracking", {})
    # Two fighters who cannot be told apart in this video fail identity however
    # well they were followed. Coverage answers "was somebody tracked", never
    # "was it the right somebody", and this is the one case where the analysis
    # can know the answer is no before it starts.
    separable = tracking.get("fighters_separable")
    identity_ready = {
        fighter: (
            _identity_seed_safe(tracking, fighter)
            and float(tracking.get(f"fighter_{fighter}_coverage", 0.0)) >= 0.45
            and separable is not False
        )
        for fighter in ("A", "B")
    }
    tracking["fighter_A_initial_lock_safe"] = identity_ready["A"]
    tracking["fighter_B_initial_lock_safe"] = identity_ready["B"]
    target = report.get("video", {}).get("analysis_target", "BOTH")
    required = ("A", "B") if target == "BOTH" else (target,)
    identity_safe = all(identity_ready.get(fighter, False) for fighter in required)
    integrity = report.setdefault("integrity", {})
    integrity["identity_evidence_trusted"] = identity_safe
    integrity["fighter_identity_trusted"] = identity_ready

    if not identity_safe:
        integrity["action_metrics_trusted"] = False
        integrity["coaching_evidence_mode"] = "withheld_identity_failure"
        failed = ", ".join(f"Fighter {fighter}" for fighter in required if not identity_ready.get(fighter, False))
        scorecard = report.setdefault("scorecard", {})
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": ("fighters_not_separable" if separable is False
                       else "identity_integrity_failed"),
            "disclaimer": (
                (
                    "Scorecard withheld because the two fighters look too alike in this "
                    "video to tell apart reliably. Their kit matches at "
                    f"{float(tracking.get('fighter_pair_similarity') or 0.0):.0%} where a "
                    "readable bout is usually nearer 60%, so any per-fighter total risks "
                    "crediting the wrong athlete."
                ) if separable is False else (
                    f"Scorecard withheld because {failed} did not pass the fighter-identity gate. "
                    "Return to fighter selection and analyze again; person coverage alone cannot prove identity."
                )
            ),
        })
        report["key_moments"] = []
        report["illegal_moves"] = []
        metrics = report.get("metrics", {})
        for fighter in ("A", "B"):
            if identity_ready[fighter] and fighter in metrics:
                pose_coaching = build_pose_coaching(fighter, metrics[fighter], metrics.get("B" if fighter == "A" else "A"))
                report.setdefault("coaching", {})[fighter] = pose_coaching
                report.setdefault("training_plan", {})[fighter] = build_training_plan(
                    pose_coaching, fighter, metrics[fighter]
                )
                report.setdefault("training_progression", {})[fighter] = build_training_progression(
                    pose_coaching, fighter, metrics[fighter]
                )
            elif not identity_ready[fighter]:
                report.setdefault("coaching", {})[fighter] = {
                    "strengths": [], "improvements": [], "drills": [],
                    "note": "Coaching withheld because this fighter did not pass the identity-integrity gate.",
                }
                report.setdefault("training_plan", {})[fighter] = []
                report.setdefault("training_progression", {})[fighter] = []
        return report

    if not bool(integrity.get("action_metrics_trusted", False)):
        integrity["coaching_evidence_mode"] = "pose_only"
        metrics = report.get("metrics", {})
        for fighter in required:
            if fighter not in metrics:
                continue
            pose_coaching = build_pose_coaching(fighter, metrics[fighter], metrics.get("B" if fighter == "A" else "A"))
            report.setdefault("coaching", {})[fighter] = pose_coaching
            report.setdefault("training_plan", {})[fighter] = build_training_plan(
                pose_coaching, fighter, metrics[fighter]
            )
            report.setdefault("training_progression", {})[fighter] = build_training_progression(
                pose_coaching, fighter, metrics[fighter]
            )
    return report


def build_preliminary_scorecard(
    events: list[StrikeEvent],
    ruleset: str,
    round_numbers: list[int],
    tracking: dict,
    analysis_target: str,
) -> dict:
    """Build a visible, explicitly unvalidated estimate from action candidates.

    Tracking coverage can establish that both selected people were observed; it
    cannot validate the rule engine's action labels.  This score therefore has
    a separate status and vocabulary from validated or human-confirmed facts.
    """
    coverage_a = float(tracking.get("fighter_A_coverage", 0))
    coverage_b = float(tracking.get("fighter_B_coverage", 0))
    minimum_coverage = min(coverage_a, coverage_b)
    coverage_ok = minimum_coverage >= SETTINGS.min_tracking_coverage_for_score
    scorecard = score_fight(events, ruleset, round_numbers, [], reliable=coverage_ok)
    candidate_count = int(scorecard.get("verified_actions_counted", 0))
    scorecard["evidence"] = {
        "required_tracking_coverage_each": SETTINGS.min_tracking_coverage_for_score,
        "fighter_A_tracking_coverage": coverage_a,
        "fighter_B_tracking_coverage": coverage_b,
        "scoring_action_candidates": candidate_count,
        "duplicate_action_candidates_removed": int(scorecard.get("duplicate_action_candidates_removed", 0)),
        "action_confidence_required": 0.72,
        "contact_confidence_required": 0.62,
        "evidence_source": "unvalidated_action_candidates",
    }
    if analysis_target != "BOTH":
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": "both_fighters_required",
            "disclaimer": "To receive an estimated scorecard, choose Analyze both fighters. A one-fighter analysis does not count the opponent's points.",
        })
    elif not coverage_ok:
        scorecard["status"] = "insufficient_observation_coverage"
        scorecard["disclaimer"] = (
            f"A preliminary score requires at least {SETTINGS.min_tracking_coverage_for_score*100:.0f}% observation coverage "
            f"for both fighters. This analysis produced A {coverage_a*100:.1f}% and B {coverage_b*100:.1f}%."
        )
    elif candidate_count < SETTINGS.min_verified_actions_for_score:
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": "insufficient_scoring_actions",
            "disclaimer": (
                "No score is shown. A round estimate needs at least "
                f"{SETTINGS.min_verified_actions_for_score} scoring actions with enough evidence, and this "
                f"analysis verified {candidate_count}. Movement, coverage and the detected "
                "action timeline below are unaffected."
                if candidate_count
                else "No preliminary score is shown because the automatic action engine found no scoring candidates with enough evidence."
            ),
        })
    elif not STRIKE_COUNTS_PRECISION_VALIDATED:
        # **A kickboxing round cannot be scored on kicks alone.**
        #
        # Every ruleset here scores hands and feet, and K-1 weights a punch at
        # 1.0 against a kick at 1.15 - so a punch is most of what decides a
        # close round. Checked against video on three bouts, the punch count
        # was overstated by eleven in two of them and the family is 14%
        # precise; the report stopped publishing it for that reason.
        #
        # Scoring from it anyway would be the same number wearing a different
        # hat. Dropping punches from the maths is no better: it would score a
        # boxing-heavy round as though nobody threw a hand, which is a
        # different wrong answer rather than a right one.
        #
        # So no score while punches cannot be counted. The kick count, the
        # movement numbers and the action timeline are unaffected - they are
        # measured, and none of them claims to say who won.
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": "punch_counting_unavailable",
            "disclaimer": (
                "No score is shown. Scoring a round needs both hands and feet counted, and "
                "WarriorIQ's punch counting is not accurate enough yet - checked against video, "
                "the kick count came out right and the punch count did not. Scoring on kicks "
                "alone would understate anyone who boxes. Movement, coverage, the leg-strike "
                "count and the action timeline below are unaffected."
            ),
        })
    else:
        scorecard["available"] = True
        scorecard["status"] = "preliminary_unvalidated"
        scorecard["disclaimer"] = (
            "Preliminary computer estimate from high-confidence action candidates. The actions are not verified fight facts; "
            "the report does not ask you to label or correct them. This is not an official judges' score."
        )
    return scorecard


def build_report(
    req: AnalysisRequest,
    original_name: str,
    rounds: list[RoundSpec],
    events: list[StrikeEvent],
    defenses: list[DefenseEvent],
    metrics: dict,
    tracking: dict,
    performance: dict,
    classifier: dict,
) -> dict:
    evidence_trust = automated_evidence_trust(classifier)
    automated_evidence_trusted = bool(evidence_trust["automated_evidence_trusted"])
    tracking = dict(tracking)
    identity_ready = {
        fighter: (
            _identity_seed_safe(tracking, fighter)
            and float(tracking.get(f"fighter_{fighter}_coverage", 0.0)) >= 0.45
        )
        for fighter in ("A", "B")
    }
    tracking["fighter_A_initial_lock_safe"] = identity_ready["A"]
    tracking["fighter_B_initial_lock_safe"] = identity_ready["B"]
    required_fighters = ("A", "B") if req.analysis_target == "BOTH" else (req.analysis_target,)
    identity_evidence_trusted = all(identity_ready[fighter] for fighter in required_fighters)
    action_metrics_trusted = automated_evidence_trusted and identity_evidence_trusted
    round_numbers = [r.number for r in rounds if r.selected]
    minimum_coverage = min(float(tracking.get("fighter_A_coverage", 0)), float(tracking.get("fighter_B_coverage", 0)))
    scoring_reliable = action_metrics_trusted and minimum_coverage >= SETTINGS.min_tracking_coverage_for_score
    verified_events = [e for e in events if action_metrics_trusted and is_verified_scoring_event(e, req.ruleset)]
    key_events: list[StrikeEvent] = []
    timeline_candidates = [
        event for event in events
        if action_metrics_trusted and _timeline_event_reliable(event) and is_legal_event(event, req.ruleset)
    ]
    for event in sorted(timeline_candidates, key=lambda e: (-e.contact_confidence, -e.confidence, e.peak_time)):
        if all(abs(event.peak_time - kept.peak_time) >= 2.5 for kept in key_events):
            key_events.append(event)
        if len(key_events) >= 8:
            break
    key_events.sort(key=lambda e: e.peak_time)
    illegal_events: list[dict] = []
    for event in sorted(events, key=lambda item: (-item.confidence, item.peak_time)):
        if not action_metrics_trusted:
            break
        legal, reason = event_legality(event, req.ruleset)
        if legal or not _timeline_event_reliable(event):
            continue
        if any(event.fighter == kept["fighter"] and abs(event.peak_time - kept["peak_time"]) < 0.45 for kept in illegal_events):
            continue
        item = event.to_dict()
        item["legality_reason"] = reason
        illegal_events.append(item)
        if len(illegal_events) >= 12:
            break
    illegal_events.sort(key=lambda item: item["peak_time"])
    if action_metrics_trusted:
        scorecard = score_fight(events, req.ruleset, round_numbers, [], reliable=scoring_reliable)
        scorecard["evidence"] = {
            "required_tracking_coverage_each": SETTINGS.min_tracking_coverage_for_score,
            "fighter_A_tracking_coverage": float(tracking.get("fighter_A_coverage", 0)),
            "fighter_B_tracking_coverage": float(tracking.get("fighter_B_coverage", 0)),
            "verified_scoring_actions": int(scorecard.get("verified_actions_counted", len(verified_events))),
            "duplicate_action_candidates_removed": int(scorecard.get("duplicate_action_candidates_removed", 0)),
            "action_confidence_required": 0.72,
            "contact_confidence_required": 0.62,
            "evidence_source": "validated_model",
        }
    else:
        scorecard = build_preliminary_scorecard(events, req.ruleset, round_numbers, tracking, req.analysis_target)
        if req.analysis_target == "BOTH" and not identity_evidence_trusted:
            failed = ", ".join(f"Fighter {fighter}" for fighter in ("A", "B") if not identity_ready[fighter])
            scorecard.update({
                "available": False,
                "totals": {"A": None, "B": None},
                "rounds": [],
                "winner_estimate": None,
                "status": "identity_integrity_failed",
                "disclaimer": (
                    f"Scorecard withheld because {failed} did not pass the fighter-identity gate. "
                    "Return to fighter selection and analyze again; person coverage alone cannot prove identity."
                ),
            })
    if req.analysis_target != "BOTH":
        scorecard["available"] = False
        scorecard["totals"] = {"A": None, "B": None}
        scorecard["rounds"] = []
        scorecard["winner_estimate"] = None
        scorecard["disclaimer"] = "To receive an estimated scorecard, choose Analyze both fighters. A one-fighter analysis does not count the opponent's points."
    elif action_metrics_trusted and not scoring_reliable:
        scorecard["disclaimer"] = (
            f"An estimated score requires at least {SETTINGS.min_tracking_coverage_for_score*100:.0f}% verified tracking "
            f"for both fighters. This analysis produced A {tracking.get('fighter_A_coverage', 0)*100:.1f}% and "
            f"B {tracking.get('fighter_B_coverage', 0)*100:.1f}%."
        )
    insufficient_coaching = {
        "strengths": [], "improvements": [], "drills": [],
        "note": "WarriorIQ did not invent coaching advice from unverified action candidates.",
    }
    # Validated actions support technique coaching. Until that release gate is
    # passed, identity-safe pose measurements still support useful movement
    # work without presenting candidate strikes as facts.
    coaching: dict[str, dict] = {}
    for fighter in ("A", "B"):
        if action_metrics_trusted and identity_ready[fighter]:
            coaching[fighter] = build_coaching(fighter, metrics, events)
        elif identity_ready[fighter]:
            coaching[fighter] = build_pose_coaching(fighter, metrics[fighter], metrics.get("B" if fighter == "A" else "A"))
        else:
            coaching[fighter] = dict(insufficient_coaching)
            coaching[fighter]["note"] = "Coaching withheld because this fighter did not pass the identity-integrity gate."
    coaching_a, coaching_b = coaching["A"], coaching["B"]

    # Reading a weapon mix against what the ruleset rewards needs the family
    # counts, so it rides the same gate as the rest of the action coaching: no
    # trusted actions, no sport-specific reading.
    sport_coaching = {
        fighter: (
            build_sport_coaching(fighter, metrics, events, req.ruleset)
            if action_metrics_trusted and identity_ready[fighter] else None
        )
        for fighter in ("A", "B")
    }

    report = {
        "product": {"name": "WarriorIQ", "version": "1.0"},
        "video": {
            "label": "Fight analysis",
            "fight_type": req.fight_type,
            "analysis_target": req.analysis_target,
            "focus_fighter": req.focus_fighter or req.analysis_target,
        },
        "setup": {
            "ruleset": req.ruleset,
            "round_count": req.round_count,
            "round_duration_seconds": req.round_duration_seconds,
            "break_duration_seconds": req.break_duration_seconds,
            "selected_rounds": req.selected_rounds,
            "start_seconds": req.start_seconds,
            "end_seconds": req.end_seconds,
            "fighter_a_box": req.fighter_a_box,
            "fighter_b_box": req.fighter_b_box,
        },
        "rounds": [
            {
                "number": r.number,
                "start_seconds": r.start_seconds,
                "end_seconds": r.end_seconds,
                "selected": r.selected,
            }
            for r in rounds
        ],
        "performance": performance,
        "tracking": tracking,
        "classifier": classifier,
        "metrics": metrics,
        "scorecard": scorecard,
        "events": [e.to_dict() for e in events],
        "key_moments": [e.to_dict() for e in key_events],
        "illegal_moves": illegal_events,
        "defenses": [d.to_dict() for d in defenses],
        "coaching": {"A": coaching_a, "B": coaching_b},
        "sport_coaching": sport_coaching,
        "training_plan": {
            "A": build_training_plan(coaching_a, "A", metrics["A"]),
            "B": build_training_plan(coaching_b, "B", metrics["B"]),
        },
        "training_progression": {
            "A": build_training_progression(coaching_a, "A", metrics["A"]),
            "B": build_training_progression(coaching_b, "B", metrics["B"]),
        },
        "integrity": {
            **evidence_trust,
            "identity_evidence_trusted": identity_evidence_trusted,
            "fighter_identity_trusted": identity_ready,
            "action_metrics_trusted": action_metrics_trusted,
            "coaching_evidence_mode": "validated_actions" if action_metrics_trusted else "pose_only" if identity_evidence_trusted else "withheld_identity_failure",
            "human_review_complete": False,
            "no_demo_statistics": True,
            "uncertainty_policy": "WarriorIQ leaves a fighter/action unavailable or uncertain when evidence is insufficient rather than inventing a result.",
            "scoring_status": scorecard["disclaimer"],
            "minimum_fighter_coverage_for_score": SETTINGS.min_tracking_coverage_for_score,
            "model_validation_status": "A custom WarriorIQ temporal checkpoint is used only when present. Otherwise the multi-frame deterministic classifier is labeled as fallback.",
            "rules_reference": "WAKO Rules revision 25.10.2022; K-1 2026 amendment takes effect 01.01.2027 and is not applied before that date.",
        },
    }
    return report


def write_report(job_dir: Path, report: dict) -> tuple[Path, Path]:
    job_dir.mkdir(parents=True, exist_ok=True)
    json_path = job_dir / "report.json"
    html_path = job_dir / "report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(value):
        if value is None:
            return "Unavailable"
        if isinstance(value, float):
            return f"{value:.3f}"
        return escape(str(value))

    # This file is written beside the analysis on the machine that ran it and
    # is not served to anybody. It still has to say the same thing the web
    # report says. Twice today a claim was corrected in one surface and left
    # standing in another - the scorecard against the live feed, then the
    # report against the live view - and a stale copy of a retracted number is
    # exactly how a wrong figure gets quoted back later.
    trusted = bool((report.get("integrity") or {}).get("action_metrics_trusted", False))

    # One source for the headline and for the table under it. They disagreed
    # once: the card counted every flagged kick while the timeline listed only
    # the ones that arrived, so a reader saw "15" above a list of six and had
    # to do arithmetic to find out nothing was broken. Worse, the largest
    # number on the page was the least reliable one - flagged kicks are 63%
    # real on the fight that was hand-checked, the arrived ones 100%.
    # Which families can be counted on THIS video. Kicks always; punches only
    # where the fighters are big enough in the network's input for a hand to be
    # readable - see PUNCHES_NEED_THIS_MANY_PIXELS.
    recording = (report.get("tracking") or {}).get("recording") or {}
    subject_pixels = float(recording.get("subject_px_in_network") or 0.0)
    punches_are_readable = subject_pixels >= PUNCHES_NEED_THIS_MANY_PIXELS
    countable_families = {"kick"} | ({"punch"} if punches_are_readable else set())

    kick_events = [e for e in (report.get("events") or [])
                   if (e.get("family") or "") in countable_families]
    def _arrived(event: dict) -> bool:
        outcome = event.get("outcome") or ""
        if (event.get("family") or "") == "punch":
            return outcome in PUNCH_OUTCOMES
        return outcome in ARRIVED_OUTCOMES

    arrived_kicks = [e for e in kick_events if _arrived(e)]

    def _per_fighter(items: list) -> dict:
        counts = {"A": 0, "B": 0}
        for item in items:
            side = str(item.get("fighter") or "")
            if side in counts:
                counts[side] += 1
        return counts

    flagged_kicks = _per_fighter(kick_events)
    reached_kicks = _per_fighter(arrived_kicks)

    def fighter_card(name: str) -> str:
        m = report["metrics"][name]
        attacks = m["attacks"]
        # Leg strikes only, and no outcomes unless the analysis passed its own
        # integrity gate. Checked against the video on three bouts: the kick
        # count was right in all three and the punch count overstated by eleven
        # in two, so punches are not shown here either.
        # On an untrusted run every leg-strike number on the card comes from
        # the same events the timeline is built from, so the headline, this row
        # and the table cannot contradict each other however the metrics
        # pipeline counts. A trusted run keeps the metrics figure, which is
        # what its own timeline lists.
        kicks = (int((attacks.get("families") or {}).get("kick") or 0) if trusted
                 else flagged_kicks.get(name, 0))
        # Every index gets the figure it should be read against. "Guard 0.110"
        # alone is unreadable - a coach cannot tell whether it is good, and the
        # report was showing four such numbers per fighter. The reference is
        # coaching's own POSE_DIMENSIONS, not a scale invented here, so the
        # card and the coaching text cannot disagree about what normal is.
        rows = [
            f"<tr><td>Leg strikes flagged</td><td>{kicks}</td><td class='muted'></td></tr>",
            f"<tr><td>Pose coverage</td><td>{m['pose_coverage']*100:.1f}%</td><td class='muted'>of analysed frames</td></tr>",
            _metric_row("Footwork (body lengths/s)", m.get("footwork_body_lengths_per_second"), "footwork_body_lengths_per_second"),
            _metric_row("Guard", m.get("guard_index"), "guard_index"),
            _metric_row("Balance", m.get("balance_index"), "balance_index"),
            _metric_row("Centre control", m.get("ring_center_control"), "ring_center_control"),
        ]
        if trusted:
            accuracy = "Unavailable" if attacks["accuracy"] is None else f"{attacks['accuracy']*100:.1f}%"
            rows += [
                f"<tr><td>Landed / attempts</td><td>{attacks['landed']} / {attacks['attempts']}</td></tr>",
                f"<tr><td>Accuracy</td><td>{accuracy}</td></tr>",
                f"<tr><td>Strongest weapon</td><td>{fmt(m['strongest_weapon'])}</td></tr>",
                f"<tr><td>Combinations</td><td>{m['combinations']['count']}</td></tr>",
                f"<tr><td>Counters</td><td>{m['counters']['count']}</td></tr>",
            ]
        # Written for a coach, not for whoever built the gate. The previous
        # wording named an "identity and action integrity gate", which tells a
        # customer nothing except that something failed. What they need to know
        # is which numbers they can rely on and which are missing, and why.
        measured = ("Strikes that reached, movement and coverage above are measured."
                    if punches_are_readable else
                    "Kicks, knees, movement and coverage above are measured.")
        withheld = ("Named techniques and accuracy are not shown - WarriorIQ can see "
                    "that a strike arrived but cannot yet tell you reliably which "
                    "punch or kick it was, so it does not guess."
                    if punches_are_readable else
                    "Punch counts, accuracy and named techniques are not shown for this "
                    "fight - the fighters are too small in the picture for WarriorIQ to "
                    "count hands reliably.")
        note = "" if trusted else (
            "<div class='muted'>%s %s</div>" % (measured, withheld))
        # On a trusted run the timeline lists every key moment, so the flagged
        # count is what the table shows and the two already agree.
        if trusted:
            headline, caption = kicks, "leg strikes flagged (kicks and knees)"
        else:
            headline = reached_kicks.get(name, 0)
            caption = "%s that reached, of %d flagged" % (
                "strikes" if punches_are_readable else "kicks",
                flagged_kicks.get(name, 0))
        return f"""
        <section class='card'>
          <h2>Fighter {name}</h2>
          <div class='big'>{headline}</div>
          <div class='muted'>{escape(caption)}</div>
          <table>{''.join(rows)}</table>
          {note}
        </section>
        """

    # Named techniques and outcomes only where the analysis earned them. A
    # "jab" on footage that cannot resolve an arm is a guess with a confident
    # label on it.
    # `key_moments` is filtered on outcome, and outcomes are only classified
    # when the analysis is trusted - so on an untrusted run the table rendered
    # its headers over nothing at all. That is the worst of both: it withholds
    # the punch counts it does not trust AND the leg strikes it does, leaving a
    # reader with an empty table under a card that says 15.
    #
    # So when nothing is trusted, fall back to the leg strikes, which is
    # exactly what the card counts and what the scorecard note says came out
    # right when checked against video. Technique names stay withheld.
    moments = report.get("key_moments") or []
    withheld_candidates = 0
    if not trusted:
        moments = arrived_kicks
        withheld_candidates = len(kick_events) - len(arrived_kicks)
    # Outcome and Target are only ever filled on a trusted run, so on every
    # other run they were two columns of "not classified" and "-" - two thirds
    # of the table saying nothing, which reads as broken rather than careful.
    # The columns are dropped instead of filled with placeholders.
    if trusted:
        timeline_head = "<tr><th>Round</th><th>Time</th><th>Fighter</th><th>Technique</th><th>Outcome</th><th>Target</th></tr>"
        event_rows = "".join(
            f"<tr><td>{e['round_number'] or '-'}</td><td>{e['peak_time']:.2f}</td><td>{escape(e['fighter'])}</td>"
            f"<td>{escape(e['technique'].replace('_',' '))}</td>"
            f"<td>{escape(e['outcome'])}</td>"
            f"<td>{escape(str(e['target']))}</td></tr>"
            for e in moments)
    else:
        timeline_head = "<tr><th>Round</th><th>Time</th><th>Fighter</th><th>What</th></tr>"
        event_rows = "".join(
            f"<tr><td>{e['round_number'] or '-'}</td><td>{e['peak_time']:.2f}</td><td>{escape(e['fighter'])}</td>"
            f"<td>{escape(e.get('family') or 'action')}</td></tr>"
            for e in moments)
    extra = ("" if not withheld_candidates else
             " %d more were seen but not shown, because the strike never "
             "reached the opponent and those are the ones WarriorIQ gets wrong "
             "most often." % withheld_candidates)
    punch_note = ("" if punches_are_readable else
                  " Punches are left out of this video - the fighters are too "
                  "small in the picture for WarriorIQ to count hands reliably. "
                  "The advice at the bottom of this page is how to change that.")
    timeline_note = "" if trusted else (
        "<div class='muted'>%s that reached the other fighter, with the second "
        "each one happened, so you can find it on the video.%s%s</div>"
        % ("Strikes" if punches_are_readable else "Kicks and knees", extra, punch_note))

    coaching_html = ""
    for fighter in ("A", "B"):
        c = report["coaching"][fighter]
        coaching_html += f"<section class='card'><h2>Coaching · Fighter {fighter}</h2>"
        coaching_html += "<h3>Strengths</h3><ul>" + "".join(
            f"<li><strong>{escape(x['title'])}</strong> — {escape(x['detail'])}</li>" for x in c["strengths"]
        ) + "</ul>"
        coaching_html += "<h3>Improvements</h3><ul>" + "".join(
            f"<li><strong>{escape(x['title'])}</strong> — {escape(x['detail'])}</li>" for x in c["improvements"]
        ) + "</ul></section>"

    # The preflight probe measures the recording on every analysis - how tall
    # the fighters are in frame, how many people are in shot, how much the
    # camera moves - and writes plain advice for the person holding it. None of
    # it was ever rendered, so the one thing a customer can actually change
    # between this analysis and a better one was computed and thrown away.
    #
    # It is last on the page on purpose: it explains the numbers above rather
    # than competing with them.
    recording = (report.get("tracking") or {}).get("recording") or {}
    recording_html = ""
    if recording.get("measured"):
        source = recording.get("source") or {}
        facts = [
            ("Video", "%s×%s at %s fps" % (source.get("width"), source.get("height"),
                                           source.get("fps"))),
            ("Fighter height in frame", "%.0f%% of the picture"
             % (100.0 * float(recording.get("subject_share_of_height") or 0.0))),
            ("People in shot", "about %.0f" % float(recording.get("people_in_frame") or 0)),
            ("Camera movement", "%.1f%% of the frame"
             % float(recording.get("camera_shift_percent") or 0.0)),
        ]
        rows = "".join("<tr><td>%s</td><td>%s</td></tr>" % (escape(k), escape(str(v)))
                       for k, v in facts)
        notes = list(recording.get("blocking") or []) + list(recording.get("warnings") or [])
        advice = list(recording.get("advice") or [])
        notes_html = "".join("<li>%s</li>" % escape(n) for n in notes)
        advice_html = "".join("<li>%s</li>" % escape(a) for a in advice)
        recording_html = (
            "<section class='card'><h2>Your recording</h2>"
            "<p class='muted'>How the footage was filmed decides most of what "
            "WarriorIQ can tell you about it. This is what it measured.</p>"
            "<table>%s</table>" % rows
            + ("<h3>What limited this analysis</h3><ul>%s</ul>" % notes_html if notes_html else "")
            + ("<h3>For a better result next time</h3><ul>%s</ul>" % advice_html if advice_html else "")
            + "</section>")

    score = report["scorecard"]
    scorecard_html = (
        f"<p class='big'>Fighter A {score['totals']['A']} · Fighter B {score['totals']['B']}</p>"
        if score.get("available") else
        f"<p>{escape(score['disclaimer'])}</p>"
    )
    # integrity.scoring_status is a copy of the scorecard disclaimer, so when
    # there is no score to show the same paragraph was printed twice on one
    # page - once as the scorecard section, once under Integrity. Repeating a
    # caveat does not make it more believable, it makes the page look generated.
    # It is kept under Integrity only where the scorecard section shows totals
    # instead, or where the two texts have diverged and dropping one would
    # withhold something.
    scoring_status = str(report["integrity"].get("scoring_status") or "")
    already_shown = (not score.get("available")) and scoring_status == score.get("disclaimer")
    integrity_scoring = "" if (already_shown or not scoring_status) else (
        f"<p>{escape(scoring_status)}</p>")

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>WarriorIQ Report</title>
<style>
body{{font-family:Inter,Arial,sans-serif;background:#090c12;color:#eef2f7;max-width:1150px;margin:0 auto;padding:32px}}
header{{display:flex;justify-content:space-between;align-items:end;margin-bottom:24px}}h1{{font-size:38px;margin:0}}.muted{{color:#8b98aa}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{background:#121824;border:1px solid #283346;border-radius:18px;padding:20px;margin-bottom:18px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #263144;text-align:left}}.big{{font-size:32px;font-weight:800}}.pill{{background:#1d2838;padding:7px 10px;border-radius:999px}}a{{color:#8cc8ff}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div><h1>WarriorIQ</h1><div class='muted'>Fight analysis</div></div><div class='pill'>{escape(report['scorecard']['ruleset_label'])}</div></header>
<div class='grid'>{fighter_card('A')}{fighter_card('B')}</div>
<section class='card'><h2>Performance</h2><p>Segment analysed: {report['performance']['segment_duration_seconds']:.1f}s · Processing time: {report['performance']['analysis_seconds']:.1f}s</p></section>
<section class='card'><h2>Estimated scorecard</h2>{scorecard_html}</section>
<section class='card'><h2>Evidence timeline</h2>{timeline_note}<table><thead>{timeline_head}</thead><tbody>{event_rows}</tbody></table></section>
{coaching_html}
<section class='card'><h2>Integrity</h2><p>{escape(report['integrity']['uncertainty_policy'])}</p>{integrity_scoring}</section>
{recording_html}
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return json_path, html_path
