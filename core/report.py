from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path

from core.count_plausibility import counts_implausible
from core.coaching import (
    POSE_DIMENSIONS,
    build_coaching, build_pose_coaching, build_training_plan, build_training_progression, drop_work_prefix,
)
from core.features import is_on as feature_on
from core.guard import reconcile_report_guard
from core.sport_profiles import build_sport_coaching
from core.config import SETTINGS
from core.evidence_trust import accepted_model_event, automated_evidence_trust
from core.scoring import (
    SAME_INSTANT_SECONDS, event_legality, is_legal_event, is_verified_scoring_event,
    minimum_kicks_per_round, score_fight, sport_counted_families, sport_of,
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


def _one_label_per_instant(events: list[dict]) -> list[dict]:
    """The report's copy of scoring.collapse_simultaneous_labels, for dicts.

    The detector files one moment several times - once per limb that moved -
    and contact classification then moves each copy onto the same impact
    frame. On the hand-labelled Kick Light fight, 42 proposed events were 33
    moments, and one of them was listed four times (a jab, a cross, an
    uppercut and a low kick, all at 87.67s). The statistics block and the
    scorecard already keep one label per fighter per instant; the report card
    counted the raw list, so that moment was four of fighter A's "29 flagged".
    Collapsed the same way, it is 21.
    """
    kept: list[dict] = []
    for fighter in ("A", "B"):
        own = sorted((e for e in events if e.get("fighter") == fighter),
                     key=lambda e: float(e.get("peak_time") or 0.0))
        groups: list[list[dict]] = []
        for event in own:
            if groups and (float(event.get("peak_time") or 0.0)
                           - float(groups[-1][0].get("peak_time") or 0.0)) <= SAME_INSTANT_SECONDS:
                groups[-1].append(event)
            else:
                groups.append([event])
        kept.extend(max(group, key=lambda e: (float(e.get("contact_confidence") or 0.0),
                                             float(e.get("confidence") or 0.0)))
                    for group in groups)
    return sorted(kept, key=lambda e: float(e.get("peak_time") or 0.0))


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


def identity_churned(tracking: dict) -> dict[str, bool]:
    """Whether each fighter had to be found again too often to vouch for.

    Coverage says somebody was followed; it cannot say it was the same
    somebody. An identity handed between tracks many times a minute - a
    panning handheld camera in a crowded hall - lands on spectators and the
    opponent while coverage stays high. Absent on older reports, which are
    judged as before. See SETTINGS.max_identity_handoffs_per_minute.

    Reports that also count the suspicious hand-offs - the ones where the
    person picked up looks different or is somewhere else - are judged on
    those alone: a handheld phone re-creates tracks constantly, and most of
    them land back on the right fighter. See
    SETTINGS.max_suspicious_handoffs_per_minute.
    """
    return {fighter: churn_rate(tracking, fighter)[1] for fighter in ("A", "B")}


def churn_rate(tracking: dict, fighter: str) -> tuple[float, bool]:
    """(hand-offs a minute the gate reads, whether that is over its limit)."""
    suspicious = tracking.get(f"fighter_{fighter}_suspicious_handoffs_per_minute")
    if suspicious is not None:
        rate = float(suspicious)
        return rate, rate > SETTINGS.max_suspicious_handoffs_per_minute
    rate = float(tracking.get(f"fighter_{fighter}_handoffs_per_minute") or 0.0)
    return rate, rate > SETTINGS.max_identity_handoffs_per_minute


def _legacy_kit_check(tracking: dict) -> bool:
    """Analysed before the current kit check existed (core/kit.py).

    Only those reports used the old histogram on purpose. A current analysis
    whose kit could not be measured also carries the histogram method, and is
    not an "earlier" check.
    """
    return (tracking.get("pair_similarity_method") != "kit_regions_lab_v1"
            and not tracking.get("kit_check_attempted"))


def lookalike_blocks_identity(tracking: dict) -> bool:
    """Whether matching kit is what stops this fight's identity being trusted.

    Kit used to decide it outright: two fighters "too alike" failed identity
    however well they were followed, so a bout in identical kit could never be
    scored - and the old histogram called a white-trunks/black-trunks pair 82%
    alike, so plenty of bouts in different kit could not be either. Kit is now
    measured properly (core/kit.py) and only says *how* identity must hold:
    for a matching pair, by position and motion, which is judged by how often
    the identity manager could not tell which was which. Reports from before
    that measure are judged as they always were.
    """
    if tracking.get("fighters_separable") is not False:
        return False
    if _legacy_kit_check(tracking):
        return True
    rate = tracking.get("identity_confusions_per_minute")
    return rate is None or float(rate) > SETTINGS.max_lookalike_confusions_per_minute


def times(count: int) -> str:
    """"once", "twice", "3 times" - never "1 times"."""
    count = int(count)
    return "once" if count == 1 else "twice" if count == 2 else f"{count} times"


def identity_failure(tracking: dict, required: tuple[str, ...] = ("A", "B")) -> dict | None:
    """Why identity failed, as one cause with one recommendation.

    A failed report used to give two at once - "picking the fighters again will
    not change that, a steadier recording will" beside "pick the two fighters
    again on a frame where their kit differs most... Show me who is who" -
    because the notice, the score box and the stored disclaimer each read a
    different signal. They all read this now, and the first cause that applies
    wins: a bad seed, then a camera that kept losing them, then kit that
    matches, then a fighter out of view. `repick` says whether picking the
    fighters again can help, and nothing on the page offers it when it cannot.
    """
    ready = identity_ready_by_fighter(tracking)
    failed = [fighter for fighter in required if not ready.get(fighter, False)]
    if not failed:
        return None
    who = " and ".join(f"Fighter {fighter}" for fighter in failed)
    seed_bad = [fighter for fighter in failed if not _identity_seed_safe(tracking, fighter)]
    if seed_bad:
        named = " and ".join(f"Fighter {fighter}" for fighter in seed_bad)
        return {"cause": "seed", "failed": failed, "repick": True,
                "headline": (f"WarriorIQ could not find {named} where the box was drawn, so it cannot be "
                             "sure it followed the right person."),
                "advice": "Pick the fighters again on a frame where both are fully visible and apart."}
    churned = [fighter for fighter in failed if identity_churned(tracking).get(fighter)]
    if churned:
        named = " and ".join(f"Fighter {fighter}" for fighter in churned)
        rate = max(churn_rate(tracking, fighter)[0] for fighter in churned)
        return {"cause": "camera", "failed": failed, "repick": False,
                "headline": (f"WarriorIQ kept losing {named}: it had to find them again {times(round(rate))} a "
                             "minute, so it cannot be sure the numbers belong to them. That happens when "
                             "the camera moves a lot or other people are close to the fighters."),
                "advice": ("Picking the fighters again will not change that; a steadier recording will. "
                           "Film from one fixed spot - a tripod, or the phone held still - with the fighters "
                           "filling most of the picture.")}
    if lookalike_blocks_identity(tracking):
        kit = tracking.get("kit_similarity") or {}
        confusions = int(tracking.get("identity_confusions") or 0)
        lost = (f", and WarriorIQ could not tell which was which {times(confusions)}" if confusions else "")
        if not kit and not _legacy_kit_check(tracking):
            # The current check ran and could not compare the two kits at all.
            return {"cause": "kit_unmeasured", "failed": failed, "repick": True,
                    "headline": ("WarriorIQ could not compare the two fighters' kit in this video, so it could "
                                 f"only tell them apart by position and movement{lost}."),
                    "advice": ("Pick the fighters again on a frame where both are fully visible and apart, "
                               "so their kit can be compared.")}
        if _legacy_kit_check(tracking):
            # Analysed before the kit check was replaced. Its percentage came
            # from a histogram that could not tell black from white, so it is
            # not repeated here as if it were a measurement.
            return {"cause": "lookalike", "failed": failed, "repick": True,
                    "headline": ("WarriorIQ's earlier kit check judged the two fighters too alike to tell "
                                 f"apart{lost}. That check could not separate some different kits - black "
                                 "from white, for one - so it may have been wrong about this fight."),
                    "advice": ("Analyse the fight again to use the current check; if they still cannot be "
                               "told apart, pick them on a frame where their kit differs most.")}
        if kit.get("small"):
            return {"cause": "small", "failed": failed, "repick": False,
                    "headline": f"The fighters are too small in the picture to tell apart by their kit{lost}.",
                    "advice": "Film closer, so the fighters fill more of the picture."}
        if kit.get("achromatic"):
            return {"cause": "black_and_white", "failed": failed, "repick": False,
                    "headline": ("This footage has no colour, so the two fighters could only be told apart by "
                                 f"position and movement{lost}."),
                    "advice": ("Picking the fighters again will not add colour. Footage where the two cross "
                               "each other less - a side angle - is what helps.")}
        similarity = kit.get("similarity")
        matched = f" ({float(similarity):.0%} alike across head, top and shorts)" if similarity is not None else ""
        return {"cause": "kit", "failed": failed, "repick": True,
                "headline": (f"The two fighters' kit matches{matched}, so they could only be told apart by "
                             f"position and movement{lost}."),
                "advice": ("If their kit differs anywhere - headgear, gloves, shorts - pick the fighters again "
                           "on a frame where that difference is visible. If it is identical, a side angle "
                           "where they cross each other less helps.")}
    low = [fighter for fighter in failed if float(tracking.get(f"fighter_{fighter}_coverage", 0.0)) < 0.45]
    if low:
        named = " and ".join(
            f"Fighter {fighter} ({float(tracking.get(f'fighter_{fighter}_coverage', 0.0)) * 100:.0f}%)"
            for fighter in low)
        return {"cause": "coverage", "failed": failed, "repick": True,
                "headline": (f"WarriorIQ only followed {named} for part of the fight, too little to be sure "
                             "it was always the same person."),
                "advice": ("Footage where both fighters stay in the picture helps most; picking them again "
                           "on a clearer frame helps if they were hard to see where you chose.")}
    return {"cause": "unknown", "failed": failed, "repick": True,
            "headline": f"WarriorIQ could not stay certain that it was following {who}.",
            "advice": "Pick both fighters again on a clearer frame, then analyse it again."}


def identity_ready_by_fighter(tracking: dict) -> dict[str, bool]:
    """The identity gate, per fighter. The one definition build_report and
    refresh_identity_integrity both use: two copies drifted apart once, and
    the page then disowned a fight the saved report called trusted."""
    churned = identity_churned(tracking)
    alike_blocks = lookalike_blocks_identity(tracking)
    # The forward pass starts at the beginning of the video even when the
    # fighters could not be followed back from the chosen frame; arriving there
    # with A and B on the wrong boxes means the identities before it are not
    # the ones the person picked. Absent (None) on reports without a check.
    seed_unconfirmed = tracking.get("identity_seed_confirmed") is False
    return {
        fighter: (
            not seed_unconfirmed
            and _identity_seed_safe(tracking, fighter)
            and float(tracking.get(f"fighter_{fighter}_coverage", 0.0)) >= 0.45
            and not alike_blocks
            and not churned[fighter]
        )
        for fighter in ("A", "B")
    }


# The tracking fields the identity gate reads. Saved with each fight's
# progress snapshot so the Progress page can apply the same gate the report
# page applies, without reopening the full report.
IDENTITY_TRACKING_KEYS = (
    "fighter_A_seed_source", "fighter_B_seed_source", "initial_iou_A", "initial_iou_B",
    "fighter_A_coverage", "fighter_B_coverage", "fighters_separable", "fighter_pair_similarity",
    "pair_similarity_method", "kit_check_attempted", "identity_confusions_per_minute",
    "identity_confusions", "fighter_A_handoffs_per_minute", "fighter_B_handoffs_per_minute",
    "fighter_A_suspicious_handoffs_per_minute", "fighter_B_suspicious_handoffs_per_minute",
    "identity_seed_confirmed",
)


def identity_tracking(tracking: dict) -> dict:
    return {key: tracking[key] for key in IDENTITY_TRACKING_KEYS if key in (tracking or {})}


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

# Whether punch, kick and knee counts are shown at all - a separate question
# from whether they are validated, which is why it is a separate switch.
#
# Owner decision, 2026-09-27: every strike family a sport scores is shown,
# because a report that counts kicks only was no use to a kickboxer, a boxer
# or a Muay Thai fighter. The counts stay what they are - automatic, and not
# validated - so they are labelled as estimates wherever they appear, with the
# measured accuracy beside them (ESTIMATE_NOTE), and nothing that needs a
# validated count is switched on by this: no "at least N", no confirmed
# federation minimum. Those still follow STRIKE_COUNTS_PRECISION_VALIDATED.
#
# The measurement behind the note: 72 clips from a Kick Light bout, labelled
# by a competitor, kept in dataset/regression/kicklight_stavrou_ceschia and
# scored by tools/benchmark_labelled_fight.py. Of the 16 moments a report
# counts on that fight, 11 were real strikes and 6 of those the right type.
# Re-run it and update this note whenever the numbers move.
#
# Switched off again on 2026-10-04, as an environment flag rather than a code
# edit. QA on production found a waist-up boxing clip counted 33 kicks and 14
# knees for one fighter, and the Accuracy Lab shows 10 real of 26 counted. Off,
# no punch, kick or knee count appears on the report summary, the live
# progress page, the replay chapter list or the story card; movement, guard,
# balance, centre and pressure stay. Set WARRIORIQ_PUBLISH_STRIKE_COUNTS=1 to
# show them again once strike classification meets the release targets on
# /validation.
#
# Read through core/features.py, which every page's copy reads as well, so the
# report and the marketing cannot disagree about what is switched on.
STRIKE_COUNTS_PUBLISHED = feature_on("strike_counts")

# The kickboxing sentence of core.sport_policy, kept here under its old name
# for callers that predate the per-sport policy; a test holds them equal. Every
# surface that knows the sport uses counting_policy(sport).estimate_note.
ESTIMATE_NOTE = (
    "Automatic counts, not checked by a person. WarriorIQ has not yet shown that it counts "
    "strikes accurately, so treat these as rough estimates - it also often mixes up "
    "punches, kicks and knees."
)

ESTIMATED_SCORE_NOTE = (
    "Estimated score, not an official judges' score. It is built from every strike "
    "WarriorIQ marked as landed, counted automatically and not checked by a person, and "
    "WarriorIQ has not yet shown that it counts strikes accurately."
)

_FAMILY_OF_PLURAL = {"punches": "punch", "kicks": "kick", "knees": "knee"}


def published_families(sport: str | None, report: dict | None = None) -> tuple[str, ...]:
    """The strike families a report shows for this sport, singular.

    Only what the sport scores - a boxing report does not list kicks the
    detector proposed - and none at all while counts are not published, or
    when ``report``'s counts failed the plausibility check
    (core/count_plausibility.py).
    """
    if counts_implausible(report):
        return ()
    try:
        scored = tuple(_FAMILY_OF_PLURAL[f] for f in sport_counted_families(sport or "kickboxing"))
    except (KeyError, ValueError):
        scored = ("punch", "kick", "knee")
    if STRIKE_COUNTS_PUBLISHED or STRIKE_COUNTS_PRECISION_VALIDATED:
        return scored
    # Otherwise only what this sport's accuracy exam passed, for the exact
    # strike model that made this report (core/strike_exam.py). No verdict,
    # or another model, means nothing.
    from core.strike_exam import passed_families

    exam = passed_families(sport, report)
    return tuple(family for family in scored if family in exam)


SHARE_CARD_NOTE = "Automatic estimate by WarriorIQ, not checked by a person."


def _sport_estimate_note(sport: str | None) -> str:
    from core.sport_policy import counting_policy

    return counting_policy(sport).estimate_note or ESTIMATE_NOTE


MOMENT_TIMES_PER_CARD = 4


def coaching_moments(report: dict, fighters: list[str], tier: str, items: int | None) -> list[dict]:
    """The report's own coaching points that point at moments, as replay cards.

    One card per point, never one per second: the same advice repeated over a
    run of cards reads as filler. Only what the result page itself shows - the
    same tier cut, nothing when the identity check failed (the page withholds
    coaching then) - and only points that carry evidence times, because a card
    that cannot take you to the moment is just the report again.
    """
    if not (report.get("integrity") or {}).get("identity_evidence_trusted", True):
        return []
    cards = []
    for fighter in fighters:
        coaching = (report.get("coaching") or {}).get(fighter) or {}
        strengths = list(coaching.get("strengths") or [])
        improvements = list(coaching.get("improvements") or [])
        if tier == "compact":
            strengths, improvements = [], improvements[:items or 1]
        elif tier != "full" and items is not None:
            strengths, improvements = strengths[:items], improvements[:items]
        for kind, points in (("keep", strengths), ("fix", improvements)):
            for point in points:
                times = sorted({round(float(t), 2) for t in point.get("evidence_times") or []
                                if isinstance(t, (int, float))})
                if not times:
                    continue
                cards.append({
                    "kind": kind, "fighter": fighter,
                    "title": str(point.get("title") or "").removeprefix("Work on: "),
                    "detail": str(point.get("detail") or ""),
                    "times": times[:MOMENT_TIMES_PER_CARD],
                })
    return sorted(cards, key=lambda card: card["times"][0])


def share_card(report: dict) -> dict | None:
    """What a stats-only story card may show, per fighter; None when it may not.

    Built from the numbers the result page already shows - the strike
    attempts in the statistics block, the estimated score, the first
    strength and improvement in the coaching - so the card never says
    anything the page does not. No names and no video: the opponent did not
    agree to be posted, and a still frame could show a minor.

    Per-fighter numbers are attributions, so when the identity check failed
    there is no card at all rather than one that may be the other fighter's.
    """
    if not (report.get("integrity") or {}).get("identity_evidence_trusted", True):
        return None
    # Too little fight footage to measure: the page shows no numbers, so the
    # card has none to post either.
    if (report.get("integrity") or {}).get("fight_footage_sufficient") is False:
        return None
    # Not recognised as a fight: nothing on it would be a fighter's result.
    if not_a_fight(report):
        return None
    statistics = (report.get("statistics") or {}).get("fighters") or {}
    if not statistics:
        return None
    scorecard = report.get("scorecard") or {}
    sport = scorecard.get("sport")
    families = published_families(sport, report)
    totals = scorecard.get("totals") or {}
    scored = bool(scorecard.get("available")) and None not in (totals.get("A"), totals.get("B"))
    metrics = report.get("metrics") or {}
    fighters = {}
    for fighter in ("A", "B"):
        item = statistics.get(fighter) or {}
        strikes = {family: int(item.get(f"{family}_attempts") or 0) for family in families}
        coaching = (report.get("coaching") or {}).get(fighter) or {}
        strengths = coaching.get("strengths") or []
        improvements = coaching.get("improvements") or []
        # Only a real weakness, one the analysis could point to in the fight:
        # "Nothing behind your opponent" is a sentence, not a thing to work on.
        weak = next((item for item in improvements if item.get("evidence_times")), None)
        working_on = None
        if weak is not None:
            working_on = str(weak.get("title") or "").removeprefix("Work on: ") or None
        fighters[fighter] = {
            # None, not zero, while strike counts are switched off: the card
            # then shows movement instead of "0 strikes thrown".
            "strikes": strikes if families else None,
            "total": sum(strikes.values()) if families else None,
            "movement": _card_movement(metrics.get(fighter) or {}),
            "strength": strengths[0].get("title") if strengths else None,
            "working_on": working_on,
        }
    return {
        "sport": scorecard.get("sport_label") or (sport or "").replace("_", " ").title() or "Fight",
        "fighters": fighters,
        # The estimated score is built from the strike counts, so it goes
        # wherever they go.
        "score": {"A": totals["A"], "B": totals["B"]} if scored and families else None,
        "note": SHARE_CARD_NOTE,
    }


def _card_movement(metrics: dict) -> list[dict]:
    """Guard, balance, centre and pressure as story-card rows, measured only.

    Each is a 0-100 value in the units the report page prints: shares of the
    round for the first three, and pressure "of 100" (core.squad.movement_value),
    which maps the stored -1..1 reading onto 0..100.
    """
    rows = []
    for key, label, unit in (("guard_index", "Guard up", "%"), ("balance_index", "Balanced", "%"),
                             ("ring_center_control", "Held the centre", "%"),
                             ("pressure_index", "Pressure", " of 100")):
        value = metrics.get(key)
        if not isinstance(value, (int, float)):
            continue
        share = (float(value) + 1.0) / 2.0 if key == "pressure_index" else float(value)
        rows.append({"label": label, "value": int(round(max(0.0, min(1.0, share)) * 100)), "unit": unit})
    return rows


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
    shown = published_families((report.get("scorecard") or {}).get("sport"), report)
    out = {}
    for fighter in ("A", "B"):
        item = statistics.get(fighter) or {}
        coverage = float(item.get("observation_coverage") or 0.0)
        if coverage < MIN_COVERAGE_TO_REPORT_OBSERVED:
            continue
        # Knees were not counted, and the reason they used to be is worth
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
        #
        # Both of those paragraphs still describe the detector. What changed is
        # STRIKE_COUNTS_PUBLISHED: the families are shown as estimates, with
        # the measured accuracy beside them, rather than withheld.
        counts = {family: int(item.get("%s_attempts" % family) or 0) for family in shown}
        total = sum(counts.values())
        if total <= 0:
            continue
        out[fighter] = {
            "followed_share": coverage,
            "actions_evidenced": total,
            "families": counts,
            # Surfaced so the page can say the omission is deliberate rather
            # than leaving a coach wondering why their boxer threw nothing.
            "punches_withheld": 0 if "punch" in shown else int(item.get("punch_attempts") or 0),
            "knees_withheld": 0 if "knee" in shown else int(item.get("knee_attempts") or 0),
        }
    if not out:
        return None
    return {
        "fighters": out,
        "basis": ("strikes the analysis flagged while it had sight of that fighter"
                  if "punch" in shown or "knee" in shown else
                  "leg strikes the analysis flagged while it had sight of that fighter"),
        "families_shown": list(shown),
        "punches_reported": "punch" in shown,
        "estimate_note": _sport_estimate_note((report.get("scorecard") or {}).get("sport")),
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
    # With counts published this is every family the sport scores; the
    # function keeps its name so callers and stored reports stay compatible.
    shown = published_families((report.get("scorecard") or {}).get("sport"), report)
    if not shown:
        # Strike counts are switched off: a total of none would read as zero.
        return None
    landed_key = {"punch": "punches_landed", "kick": "kicks_landed", "knee": "knees_landed"}
    attempts = sum(int(row.get("%s_attempts" % family) or 0) for row in rows for family in shown)
    landed_values = [row.get(landed_key[family]) for row in rows for family in shown]
    landed = (sum(int(value) for value in landed_values)
              if all(value is not None for value in landed_values) else None)
    return {"attempts": attempts, "landed": landed,
            "label": "Kicks" if shown == ("kick",) else "Strikes"}


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
    if not STRIKE_COUNTS_PUBLISHED and not STRIKE_COUNTS_PRECISION_VALIDATED:
        # "N kicks seen" is a kick count, and strike counts are switched off.
        return None
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


def _report_sport(report: dict) -> str | None:
    """The sport a stored report was analysed as, for sport-aware coaching."""
    scorecard = report.get("scorecard") or {}
    if scorecard.get("sport"):
        return str(scorecard["sport"])
    ruleset = scorecard.get("ruleset") or (report.get("request") or {}).get("ruleset")
    try:
        return sport_of(ruleset) if ruleset else None
    except (KeyError, ValueError):
        return None


def identity_verdict(report: dict) -> dict:
    """The one answer to "is this report about the people the user picked?".

    QA, 2026-10-04: one report said "Not scored, we lost sight of a fighter"
    in one section and "Good observation evidence / Identity stability:
    Stable" in another, and still offered a training plan and a success
    target. Each section had its own rule. Every section now reads this:
    integrity.identity_evidence_trusted (written from it by
    refresh_identity_integrity), the evidence-quality summary, the score
    explanation and the coaching. When ``trusted`` is False no section
    attributes anything to a fighter.

    ``followed_enough_to_score`` is the coverage the score needs. It is not
    part of identity: a fighter can be the right person and still be out of
    sight too often to score. The quality label reads it too, so "good
    evidence" never sits beside "we lost sight of a fighter".
    """
    tracking = report.get("tracking") or {}
    ready = identity_ready_by_fighter(tracking)
    target = (report.get("video") or {}).get("analysis_target", "BOTH")
    required = ("A", "B") if target == "BOTH" else (target,)
    trusted = all(ready.get(fighter, False) for fighter in required)
    coverage = {fighter: max(0.0, min(1.0, float(tracking.get(f"fighter_{fighter}_coverage", 0.0) or 0.0)))
                for fighter in ("A", "B")}
    return {
        "trusted": trusted,
        "by_fighter": ready,
        "required": required,
        "coverage": coverage,
        "followed_enough_to_score": min(coverage[f] for f in required) >= SETTINGS.min_tracking_coverage_for_score,
        "cause": None if trusted else identity_failure(tracking, required),
    }


def not_a_fight(report: dict) -> dict | None:
    """The plausibility verdict when the footage did not move like a fight.

    core/fight_presence.py FightPresence.plausibility, stored with the video.
    None for a fight, and for a saved report with no verdict and no track.
    """
    verdict = (report.get("video") or {}).get("plausibility")
    if isinstance(verdict, dict) and verdict.get("plausible") is False:
        return verdict
    return None


def not_a_fight_reason(verdict: dict) -> str:
    reasons = [str(text) for text in (verdict.get("text") or []) if text]
    why = "; and ".join(reasons) if reasons else "the movement did not look like two people fighting"
    return f"This clip was not recognised as a fight: {why}."


# Shown where guard and balance are withheld because the poses never changed.
RIGID_POSE_NOTE = ("Not measured: the bodies' poses never changed, so there was no guard or "
                   "stance to follow.")


def withhold_for_not_a_fight(report: dict, verdict: dict) -> None:
    """Measured movement only: no coaching, no plan, no score, no key moments.

    QA, 2026-10-07: two photo cut-outs got "Strong observation evidence",
    strengths and a four-week plan. Where the poses never changed at all,
    guard and balance are not readings of a body either, and go too.
    """
    reason = not_a_fight_reason(verdict)
    integrity = report.setdefault("integrity", {})
    integrity["fight_plausible"] = False
    integrity["action_metrics_trusted"] = False
    integrity["coaching_evidence_mode"] = "withheld_not_a_fight"
    scorecard = report.setdefault("scorecard", {})
    scorecard.update({"available": False, "totals": {"A": None, "B": None}, "rounds": [],
                      "winner_estimate": None, "status": "not_a_fight",
                      "disclaimer": f"Not scored. {reason}"})
    report["key_moments"] = []
    report["illegal_moves"] = []
    for fighter in ("A", "B"):
        report.setdefault("coaching", {})[fighter] = {
            "strengths": [], "improvements": [], "drills": [],
            "note": f"Coaching and the training plan are withheld. {reason}",
        }
        report.setdefault("training_plan", {})[fighter] = []
        report.setdefault("training_progression", {})[fighter] = []
    if "rigid" in (verdict.get("reasons") or []):
        withhold_pose_figures(report, RIGID_POSE_NOTE, "poses never changed")


def withhold_pose_figures(report: dict, note: str, short: str) -> None:
    """Guard and balance, and everything derived from them, become "Not measured".

    For footage where they are not readings of an upright, live body: frozen
    poses (core/fight_presence.py) or a video still lying on its side.
    Movement, centre and pressure come from where the body was, not its
    shape, and stay.
    """
    metrics = report.get("metrics") or {}
    for fighter in ("A", "B"):
        own = metrics.get(fighter)
        if not isinstance(own, dict):
            continue
        own["guard_index"] = None
        own["balance_index"] = None
        own["guard_note"] = note
        own["guard_note_short"] = short
        own["pose_note"] = note
        numbers = own.get("numbers")
        if isinstance(numbers, dict):
            numbers.update({"hands_up_share": None, "longest_hands_down_seconds": None,
                            "off_balance_count": None})
        for key in ("moments", "spread"):
            if isinstance(own.get(key), dict):
                own[key].pop("guard_index", None)
                own[key].pop("balance_index", None)
        availability = own.get("availability")
        if isinstance(availability, dict):
            for name in ("guard", "balance"):
                availability[name] = {**(availability.get(name) or {}), "available": False, "reason": note}


SIDEWAYS_NOTE = ("Guard and balance are not measured: this video was still filmed sideways, so "
                 "WarriorIQ could not tell up from down for the body. Turn it with \"Rotate 90°\" on "
                 "the fighter-selection page and analyse it again.")


def still_sideways(report: dict) -> bool:
    return bool(((report.get("video") or {}).get("orientation") or {}).get("sideways"))


def withhold_for_sideways(report: dict) -> dict:
    """QA, 2026-10-07: "filmed sideways, could not turn it" at upload, then a
    solo report with guard 39%. Hands "up by the face" and a balanced stance
    are directions on an upright body; read from a body lying across the
    picture they are not measurements."""
    if still_sideways(report):
        withhold_pose_figures(report, SIDEWAYS_NOTE, "video still sideways")
    return report


def refresh_identity_integrity(report: dict) -> dict:
    """Apply the current identity safety gate to new and legacy reports.

    This prevents an older high-coverage wrong-person track from remaining
    usable after the lock policy improves.  It also upgrades safe legacy
    reports with pose-only coaching when action labels are still unvalidated.
    """
    # Before anything reads a guard figure: older reports carried three guard
    # summaries that disagreed with each other (core/guard.py).
    reconcile_report_guard(report)
    drop_work_prefix(report)
    withhold_for_sideways(report)
    tracking = report.setdefault("tracking", {})
    # Two fighters who cannot be told apart in this video fail identity however
    # well they were followed. Coverage answers "was somebody tracked", never
    # "was it the right somebody", and this is the one case where the analysis
    # can know the answer is no before it starts.
    verdict = identity_verdict(report)
    identity_ready = verdict["by_fighter"]
    tracking["fighter_A_initial_lock_safe"] = identity_ready["A"]
    tracking["fighter_B_initial_lock_safe"] = identity_ready["B"]
    required = verdict["required"]
    identity_safe = verdict["trusted"]
    integrity = report.setdefault("integrity", {})
    integrity["identity_evidence_trusted"] = identity_safe
    integrity["fighter_identity_trusted"] = identity_ready

    if not identity_safe:
        integrity["action_metrics_trusted"] = False
        integrity["coaching_evidence_mode"] = "withheld_identity_failure"
        failed = ", ".join(f"Fighter {fighter}" for fighter in required if not identity_ready.get(fighter, False))
        # One cause, one recommendation, shared with the page.
        cause = identity_failure(tracking, required)
        scorecard = report.setdefault("scorecard", {})
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": ("fighters_not_separable" if lookalike_blocks_identity(tracking)
                       else "identity_integrity_failed"),
            "disclaimer": ("Scorecard withheld. " + (
                f"{cause['headline']} {cause['advice']}" if cause
                else f"{failed} did not pass the fighter-identity gate.")),
        })
        report["key_moments"] = []
        report["illegal_moves"] = []
        # No coaching for either fighter. It used to be kept for whichever one
        # passed on their own, so a report headed "identity check failed"
        # still offered a training plan and a success target. Coaching
        # compares a fighter with their opponent, so it cannot stand once the
        # report cannot say who the opponent was.
        for fighter in ("A", "B"):
            report.setdefault("coaching", {})[fighter] = {
                "strengths": [], "improvements": [], "drills": [],
                "note": "Coaching withheld because WarriorIQ could not confirm who was who in this fight.",
            }
            report.setdefault("training_plan", {})[fighter] = []
            report.setdefault("training_progression", {})[fighter] = []
        return report

    # The footage did not move like a fight (core/fight_presence.py): the
    # movement measured is shown, nothing that interprets it as fighting.
    implausible = not_a_fight(report)
    if implausible:
        withhold_for_not_a_fight(report, implausible)
        return report

    if not bool(integrity.get("action_metrics_trusted", False)):
        integrity["coaching_evidence_mode"] = "pose_only"
        metrics = report.get("metrics", {})
        for fighter in required:
            if fighter not in metrics:
                continue
            pose_coaching = build_pose_coaching(fighter, metrics[fighter], metrics.get("B" if fighter == "A" else "A"),
                                                _report_sport(report))
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
    # With counts published but not validated, the score is built the same
    # way - from the landed candidates - and labelled an estimate.
    estimated = STRIKE_COUNTS_PUBLISHED and not STRIKE_COUNTS_PRECISION_VALIDATED
    scorecard = score_fight(events, ruleset, round_numbers, [], reliable=coverage_ok,
                            estimated=estimated)
    candidate_count = int(scorecard.get("verified_actions_counted", 0))
    scorecard["evidence"] = {
        "required_tracking_coverage_each": SETTINGS.min_tracking_coverage_for_score,
        "fighter_A_tracking_coverage": coverage_a,
        "fighter_B_tracking_coverage": coverage_b,
        "scoring_action_candidates": candidate_count,
        "duplicate_action_candidates_removed": int(scorecard.get("duplicate_action_candidates_removed", 0)),
        "action_confidence_required": None if estimated else 0.72,
        "contact_confidence_required": None if estimated else 0.62,
        "evidence_source": ("estimated_landed_candidates" if estimated
                            else "unvalidated_action_candidates"),
    }
    if analysis_target != "BOTH":
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": "both_fighters_required",
            "disclaimer": "To receive an estimated scorecard, choose Analyse both fighters. A one-fighter analysis does not count the opponent's points.",
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
                # "verified" was wrong: these are unverified candidates, and
                # the same page lists them as such.
                "No score is shown. A round estimate needs at least "
                f"{SETTINGS.min_verified_actions_for_score} scoring actions with enough evidence, and this "
                f"analysis found {candidate_count} candidate{'' if candidate_count == 1 else 's'}. "
                "Movement, coverage, guard and balance below are unaffected."
                if candidate_count
                else "No preliminary score is shown because the automatic action engine found no scoring candidates with enough evidence."
            ),
        })
    elif not STRIKE_COUNTS_PRECISION_VALIDATED and not STRIKE_COUNTS_PUBLISHED:
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
                # Boxing has no kicks, so the general wording - hands and
                # feet, the leg-strike count - described a sport it is not.
                "No score is shown. Boxing is scored on punches, and WarriorIQ's punch "
                "counting is not accurate enough yet - checked against video, the punch "
                "count was overstated. Movement, guard, balance and coverage below are "
                "unaffected."
                if scorecard.get("sport") == "boxing" else
                "No score is shown. Scoring a round needs both hands and feet counted, and "
                "WarriorIQ's punch counting is not accurate enough yet - checked against video, "
                "the kick count came out right and the punch count did not. Scoring on kicks "
                "alone would understate anyone who boxes. Movement, coverage, the leg-strike "
                "count and the action timeline below are unaffected."
            ),
        })
    elif estimated:
        scorecard["available"] = True
        scorecard["status"] = "estimated_from_detector"
        scorecard["disclaimer"] = ESTIMATED_SCORE_NOTE
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
    # Separability as well, exactly as refresh_identity_integrity applies it.
    # Without it the report page (which refreshes) said "not safe to use"
    # while the integrity stored here - and saved into the Progress snapshot -
    # said trusted, so Progress charted a fight its own report disowned.
    identity_ready = identity_ready_by_fighter(tracking)
    tracking["fighter_A_initial_lock_safe"] = identity_ready["A"]
    tracking["fighter_B_initial_lock_safe"] = identity_ready["B"]
    required_fighters = ("A", "B") if req.analysis_target == "BOTH" else (req.analysis_target,)
    identity_evidence_trusted = all(identity_ready[fighter] for fighter in required_fighters)
    # A release-ready model still has to have confirmed each event itself
    # (core.evidence_trust.accepted_model_event); rule-only candidates are not
    # promoted to verified facts because the model is loaded.
    action_metrics_trusted = (
        automated_evidence_trusted and identity_evidence_trusted
        and all(accepted_model_event(event) for event in events)
    )
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
                    "Return to fighter selection and analyse again; person coverage alone cannot prove identity."
                ),
            })
    if req.analysis_target != "BOTH":
        scorecard["available"] = False
        scorecard["totals"] = {"A": None, "B": None}
        scorecard["rounds"] = []
        scorecard["winner_estimate"] = None
        scorecard["disclaimer"] = "To receive an estimated scorecard, choose Analyse both fighters. A one-fighter analysis does not count the opponent's points."
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
        # Coaching for neither fighter unless the report as a whole is trusted
        # (identity_verdict): one fighter passing on their own is not enough
        # when the coaching compares them with an opponent nobody confirmed.
        if action_metrics_trusted and identity_evidence_trusted and identity_ready[fighter]:
            coaching[fighter] = build_coaching(fighter, metrics, events)
        elif identity_evidence_trusted and identity_ready[fighter]:
            coaching[fighter] = build_pose_coaching(fighter, metrics[fighter], metrics.get("B" if fighter == "A" else "A"),
                                                    sport_of(req.ruleset))
        else:
            coaching[fighter] = dict(insufficient_coaching)
            coaching[fighter]["note"] = "Coaching withheld because WarriorIQ could not confirm who was who in this fight."
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
            "model_validation_status": "A custom WarriorIQ temporal checkpoint is used only when present. Otherwise the multi-frame deterministic classifier is labelled as fallback.",
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
    # With counts published every scored family is shown as an estimate,
    # whatever the size of the fighters; ESTIMATE_NOTE says how far to trust it.
    punches_are_readable = (STRIKE_COUNTS_PUBLISHED
                            or subject_pixels >= PUNCHES_NEED_THIS_MANY_PIXELS)
    if STRIKE_COUNTS_PUBLISHED:
        countable_families = set(published_families((report.get("scorecard") or {}).get("sport"), report))
    else:
        countable_families = {"kick"} | ({"punch"} if punches_are_readable else set())

    kick_events = [e for e in _one_label_per_instant(report.get("events") or [])
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
        # On an untrusted run with readable punches this count includes them,
        # and the row used to call it "leg strikes" regardless.
        kicks_label = ("Strikes flagged" if not trusted and punches_are_readable
                       else "Leg strikes flagged")
        # Every index gets the figure it should be read against. "Guard 0.110"
        # alone is unreadable - a coach cannot tell whether it is good, and the
        # report was showing four such numbers per fighter. The reference is
        # coaching's own POSE_DIMENSIONS, not a scale invented here, so the
        # card and the coaching text cannot disagree about what normal is.
        rows = [
            f"<tr><td>{kicks_label}</td><td>{kicks}</td><td class='muted'></td></tr>",
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
        estimate = (" " + ESTIMATE_NOTE) if STRIKE_COUNTS_PUBLISHED else ""
        note = "" if trusted else (
            "<div class='muted'>%s %s%s</div>" % (measured, withheld, escape(estimate)))
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


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def fix_first(report: dict, fighter: str, *, training_items: int | None = None) -> dict | None:
    """The one thing to fix first, where it happened, and the drill for it.

    The report's top card, after mmagpt.app's "Fix this first": one finding
    first, the proof next to it, something to do about it under it. Built
    only from what the analysis already measured - the fighter's first
    improvement (core/coaching.py), its own evidence times, and the drill made
    for the same measurement - so nothing here is new or guessed. None when
    there is no measured fault to name (or the identity check failed, when the
    page shows no coaching at all), and the page falls back to its usual rows.

    ``training_items`` is the plan's allowance (core/payments.py): 0 keeps the
    drill off, like the training plan it comes from.
    """
    if not (report.get("integrity") or {}).get("identity_evidence_trusted", True):
        return None
    coaching = (report.get("coaching") or {}).get(fighter) or {}
    improvements = [item for item in coaching.get("improvements") or [] if isinstance(item, dict)]
    if not improvements:
        return None
    first = improvements[0]
    drills = [d for d in coaching.get("drills") or [] if isinstance(d, dict)]
    # Matched by measurement on new reports, by the sentence on older ones
    # (a drill's "why" is its finding's detail, core/coaching.py).
    drill = (next((d for d in drills if first.get("metric") and d.get("metric") == first.get("metric")), None)
             or next((d for d in drills if d.get("why") and d.get("why") == first.get("detail")), None))
    times = sorted(float(t) for t in first.get("evidence_times") or [] if isinstance(t, (int, float)))
    if drill is None and not times:
        # "Nothing behind your opponent" and the like: no fault, no moment.
        return None
    card = {
        "title": str(first.get("title") or "").removeprefix("Work on: "),
        "detail": str(first.get("detail") or ""),
        "moment_seconds": times[0] if times else None,
        "moment_clock": _clock(times[0]) if times else None,
        "more_moments": max(0, len(times) - 1),
        "drill": None,
    }
    if drill is not None and training_items != 0:
        plan = next((row for row in (report.get("training_plan") or {}).get(fighter) or []
                     if isinstance(row, dict) and str(row.get("work") or "").endswith(str(drill.get("prescription")))),
                    None)
        card["drill"] = {
            "name": str(drill.get("name") or "").split(" · ", 1)[-1],
            "prescription": str(drill.get("prescription") or ""),
            "goal": str((plan or {}).get("goal") or ""),
        }
    return card


def did_well(report: dict, fighter: str) -> dict | None:
    """What the fighter did well, beside "Fix this first": their first strength.

    The same item the report's "Keep doing" row shows (core/coaching.py), with
    the first moment it was seen, so the top of the report gives one thing
    to keep as well as one to fix. Built only from what the analysis measured;
    None when there is no strength to name, or the identity check failed.
    """
    if not (report.get("integrity") or {}).get("identity_evidence_trusted", True):
        return None
    coaching = (report.get("coaching") or {}).get(fighter) or {}
    strengths = [item for item in coaching.get("strengths") or [] if isinstance(item, dict) and item.get("title")]
    if not strengths:
        return None
    first = strengths[0]
    times = sorted(float(t) for t in first.get("evidence_times") or [] if isinstance(t, (int, float)))
    return {
        "title": str(first["title"]),
        "detail": str(first.get("detail") or ""),
        "moment_seconds": times[0] if times else None,
        "moment_clock": _clock(times[0]) if times else None,
    }
