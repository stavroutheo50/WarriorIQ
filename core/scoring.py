from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from core.action import CONFIDENCE_CEILING, CONFIDENCE_FLOOR
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
    # Is a round judged as a whole on a ten-point-must card, or is the bout
    # decided by counting scoring techniques? This was called `ring_sport`,
    # and the name caused the error it was named after: WAKO's three ring
    # disciplines were set True because they are fought in a ring, when the
    # WAKO rules count points and never score a round 10-9 (Chapter 7,
    # Articles 7.1 and 8.1). Boxing, Muay Thai and MMA are the ten-point-must
    # sports here; the venue has nothing to do with it.
    ten_point_must: bool
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
    # Federations that publish a fixed point value per technique and target get
    # that table verbatim, as (family, target, points). Sports judged round by
    # round on a ten-point-must basis have no such table and keep the weighted
    # comparison instead - that is a difference in how the sport is scored, not
    # a gap in the data.
    point_table: tuple[tuple[str, str | None, int], ...] = ()
    # What a kick is worth when the athlete turned their back into it. Several
    # federations pay a turning kick more than the same kick thrown square -
    # WT taekwondo scores a turning head kick 5 against 3 - so it is a separate
    # value per target rather than a multiplier. Empty means this ruleset does
    # not distinguish them, and the ordinary table applies.
    turning_kick_points: tuple[tuple[str, int], ...] = ()
    # How a ten-point-must federation turns a lead into a round score, as
    # (largest difference in scoring actions, points the loser gets). IFMA
    # publishes exactly this table for muaythai, so where a federation states
    # it we use its numbers instead of the generic heuristic below - which is
    # a pair of fitted constants (a 5.5 lead and a doubling) that no rulebook
    # contains. Empty means the federation does not publish one.
    round_margins: tuple[tuple[int, int], ...] = ()
    # Actions this discipline scores that the model cannot observe at all.
    unobserved: tuple[str, ...] = ()


def _weights(profile: RuleProfile) -> dict[str, float]:
    return dict(profile.family_value)


# Every WAKO discipline below was checked line by line on 2026-09-07 against
# the two documents WAKO publishes at wako.sport/rules-overview: the "WAKO
# Rules" book (190 pages) and the "WAKO Referee Rules" (50 pages). Article
# numbers in the comments are that book's, so a future reader can disagree
# with the source rather than with me.
#
# The finding that mattered: **WAKO does not use the ten-point-must system in
# any discipline.** Chapter 7 (ring rules) Article 8.1 says "Each legal
# technique will be scored as 1 point", and Article 7.1 decides the bout by
# "the kickboxer who scored more points". So a WAKO ring bout is counted, and
# every legal technique counts the same - a head kick is not worth more than a
# jab in Full Contact, Low Kick or K-1. That is the opposite of what was
# encoded here, which scored those three on a 10-9 card with kicks weighted
# 1.15 against punches. The weighting was plausible and invented.
JUMP_BONUS = ("jumping-kick bonuses, which score 2 to the body and 3 to the head",)

RULESETS: dict[str, RuleProfile] = {
    # ---- Kickboxing (WAKO disciplines) -------------------------------------
    # The three ring disciplines share Chapter 7. Every legal technique is one
    # point, to any legal target, so the tables differ only in which targets
    # each discipline makes legal - which is Article 3 of each chapter.
    #
    # K-1: legs are legal in full ("Legs (all parts including joints)"), knees
    # are a listed technique (Article 4.3), and the spinning back fist is a
    # listed hand technique.
    "K1": RuleProfile(
        "K1", "K-1", False, True, True, True, False, frozenset({"spinning_backfist"}),
        point_table=(
            ("punch", "head", 1), ("punch", "body", 1),
            ("kick", "head", 1), ("kick", "body", 1), ("kick", "leg", 1),
            ("knee", "head", 1), ("knee", "body", 1), ("knee", "leg", 1),
        ),
    ),
    # Low Kick: thigh only ("below the waist and above the knee"), no knees,
    # no back fist. Kicks to the knee and below are explicitly illegal, which
    # `event_legality` cannot separate from a thigh kick on this footage - the
    # table therefore values a leg kick, and the legality check is what refuses
    # the ones it can recognise by technique name.
    "LOW_KICK": RuleProfile(
        "LOW_KICK", "Low Kick", False, True, False, True, False, frozenset(),
        point_table=(
            ("punch", "head", 1), ("punch", "body", 1),
            ("kick", "head", 1), ("kick", "body", 1), ("kick", "leg", 1),
        ),
    ),
    # Full Contact: above the waist only; the foot is a target at ankle level
    # for sweeping alone, which is not a kick this model emits, so there is no
    # leg row. Article 6 of Chapter 8 also obliges each kickboxer to throw a
    # minimum of six kicks per round - see MINIMUM_KICKS_PER_ROUND.
    "FULL_CONTACT": RuleProfile(
        "FULL_CONTACT", "Full Contact", False, False, False, True, False, frozenset(),
        point_table=(
            ("punch", "head", 1), ("punch", "body", 1),
            ("kick", "head", 1), ("kick", "body", 1),
        ),
    ),
    # Point Fighting: Chapter 2, Article 8.3 "Points", verbatim -
    #   Punch 1 pt / Kick to the body 1 pt / Foot sweep 1 pt /
    #   Kick to head 2 pts / Jumping kick to body 2 pts /
    #   Jumping kick to head 3 pts.
    # All four values already here were right. The back fist is legal and the
    # spinning back fist is not, which Article 4.1 states in those words.
    "POINT_FIGHTING": RuleProfile(
        "POINT_FIGHTING", "Point Fighting", False, False, False, False, True,
        frozenset({"backfist"}),
        point_table=(
            ("punch", "head", 1), ("punch", "body", 1),
            ("kick", "body", 1), ("kick", "head", 2),
            ("kick", "leg", 1),          # foot sweeps score 1
        ),
        unobserved=JUMP_BONUS,
    ),
    # Light Contact and Kick Light are scored by judges pressing a button, and
    # the rules say how many times: Chapter 3 Article 6.3 and Chapter 5
    # Article 6, in identical words -
    #   once   for a hand and leg technique to body, and hand technique to head
    #   twice  for a jump kick to body or head kick
    #   three  for a jump kick to head
    # so a head kick is 2 in both, exactly as in point fighting. Neither had a
    # table at all before; both were being scored by the generic weighting.
    "LIGHT_CONTACT": RuleProfile(
        "LIGHT_CONTACT", "Light Contact", False, False, False, False, False, frozenset(),
        point_table=(
            ("punch", "head", 1), ("punch", "body", 1),
            ("kick", "body", 1), ("kick", "head", 2),
            ("kick", "leg", 1),          # foot sweeps, as in point fighting
        ),
        unobserved=JUMP_BONUS,
    ),
    # Kick Light is the one tatami discipline where the thigh is a legal
    # target ("Legs - Thigh, inside, outside and back"). The button rule does
    # not name a thigh kick, and the only techniques it lifts above one point
    # are head kicks and jumping kicks, so a thigh kick scores 1. That is a
    # reading of the rule rather than a quotation of it, and is marked as such.
    "KICK_LIGHT": RuleProfile(
        "KICK_LIGHT", "Kick Light", False, True, False, False, False, frozenset(),
        point_table=(
            ("punch", "head", 1), ("punch", "body", 1),
            ("kick", "body", 1), ("kick", "head", 2),
            ("kick", "leg", 1),          # inferred: not lifted above 1 by the button rule
        ),
        unobserved=JUMP_BONUS,
    ),

    # ---- Boxing -------------------------------------------------------------
    # Checked against World Boxing Competition Rules, in force November 2024,
    # from worldboxing.org on 2026-09-07. Rule 7.1.2 confirms the Ten Point
    # Must system, and Rule 7.2.1 gives three criteria in order of importance:
    # number of scoring blows to the target area, then technical and tactical
    # superiority, then competitiveness. Rule 7.2.2.1 is explicit that
    # "quantity of the scoring blows should be considered as the most important
    # factor", which is the one criterion a count can speak to.
    #
    # `unobserved` stays empty and that is correct: boxing has no scoring
    # *action* outside the punch, so there is no family to declare missing and
    # nothing to warn an uploader about before they spend an hour on a bout.
    #
    # But the comment that used to sit here - "the one discipline the model
    # observes completely" - was wrong twice over, and is worth naming rather
    # than deleting. Rule 7.2.2 says a blow scores only if it "must connect
    # with the knuckle surface of the glove" and "must have the weight of the
    # body or shoulder behind it"; neither is measurable on 480x220 footage, so
    # what this counts is punches thrown, not scoring blows. And the claim that
    # "every punch class is already detected" is contradicted by the report
    # itself, which sets `action_labels_available` false because a jab is not
    # tellable from a cross at this resolution.
    "BOXING": RuleProfile(
        "BOXING", "Boxing", True, False, False, True, False, frozenset(),
        sport="boxing", sport_label="Boxing",
        allow_kick=False,
        family_value=(("punch", 1.0),),
    ),

    # ---- Muay Thai ----------------------------------------------------------
    # Checked against IFMA Muaythai Rules & Regulations v3.057, revised 11 May
    # 2026, from muaythai.sport on 2026-09-07. IFMA is the international
    # federation, and three things in it contradicted what was encoded here.
    #
    # 1. **Every skill scores the same.** Article 29.1: "A Muaythai skill is a
    #    punch, kick, knee or elbow applied with force and intent to cause
    #    effect. One score will be awarded for each Muaythai skill that strikes
    #    against a scoring target". Punches were weighted 0.9 against 1.25 for
    #    kicks and knees, which is the folk wisdom about Muay Thai and is not
    #    IFMA's rule. Those weights also set the coaching emphasis, so the
    #    advice inherited the invented ratio.
    # 2. **The round margins are published**, in Article 29.2.1, as a count of
    #    scoring skills: a lead of 7 or fewer is a small margin, 8 to 14 a
    #    large one, 15 to 21 total domination, scoring 10-9, 10-8 and 10-7.
    # 3. **A sweep is a foul, not a score.** Article 31.2.7 makes tripping an
    #    opponent without a Muaythai skill a prohibited act, and 29.2.2 refuses
    #    a score for "throwing the opponent without striking". "Sweeps and
    #    dumps" was listed here as something the sport scores and we cannot
    #    see, which overstated what the report was missing. Clinch dominance is
    #    likewise not itself scored - a knee thrown from the clinch scores as a
    #    knee. Both still matter under professional stadium scoring, which is a
    #    different rulebook, so they are named as that rather than dropped.
    #
    # Article 29.1.1 makes the target "any part of the body except the groin
    # and cervical spine", which is why leg attacks are legal here.
    "MUAY_THAI": RuleProfile(
        "MUAY_THAI", "Full rules (elbows allowed)", True, True, True, True, False, frozenset({"spinning_backfist"}),
        sport="muay_thai", sport_label="Muay Thai",
        family_value=(("punch", 1.0), ("kick", 1.0), ("knee", 1.0)),
        round_margins=((7, 9), (14, 8), (21, 7)),
        unobserved=("elbow strikes", "clinch dominance and sweeps under professional stadium scoring"),
    ),
    # Amateur and many promotional cards bar elbows outright. That is not a
    # cosmetic variant: with elbows barred they stop being a scoring action the
    # analysis is blind to, so banning them measurably improves what the report
    # can honestly claim. It is the biggest coverage lever in this sport.
    "MUAY_THAI_NO_ELBOWS": RuleProfile(
        "MUAY_THAI_NO_ELBOWS", "No elbows", True, True, True, True, False, frozenset({"spinning_backfist"}),
        sport="muay_thai", sport_label="Muay Thai",
        family_value=(("punch", 1.0), ("kick", 1.0), ("knee", 1.0)),
        round_margins=((7, 9), (14, 8), (21, 7)),
        unobserved=("clinch dominance and sweeps under professional stadium scoring",),
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
    # Checked against the Official ITF Rules of Competition, Version 2022v1,
    # T 34 "Point Awards": one point for any legal hand attack to mid or high
    # section, two for a foot attack to mid-section, three for a foot attack to
    # high-section. T 33 "Target Area" limits targets to the head (front, sides
    # and top, excluding the neck and the back) and the frontal trunk from
    # shoulder to navel - so a leg is not a target and scores nothing.
    #
    # There is no jumping or spinning bonus in the championship rules. Those
    # exist in some national bodies, which is why they are named below as
    # something this ruleset does not model rather than invented into the table.
    "ITF_TAEKWONDO": RuleProfile(
        "ITF_TAEKWONDO", "ITF · International Taekwon-Do Federation", False, False, False, False, False, frozenset(),
        sport="taekwondo", sport_label="Taekwondo",
        allow_head_punch=True,
        family_value=(("punch", 1.0), ("kick", 2.0)),
        point_table=(
            ("punch", "head", 1), ("punch", "body", 1),
            ("kick", "body", 2), ("kick", "head", 3),
            ("kick", "leg", 0),          # legs are not a legal ITF target
        ),
        unobserved=("jumping and spinning bonuses used by some national bodies "
                    "but not by the ITF championship rules",),
    ),
    # Checked against WT Competition Rules and Interpretation, in force as of
    # 1 June 2026, downloaded from worldtaekwondo.org on 2026-09-07.
    #
    # Article 12.3 "Valid Points", quoted: one point for a punch to the trunk
    # protector, two for a kick to the trunk protector, three for a kick to the
    # head, and "when a valid turning kick is delivered to the trunk protector
    # or the head, the awarded points shall be doubled: four (4) points for a
    # valid turning kick to the trunk protector, and six (6) points for a valid
    # turning kick to the head". The head value was **5** here, which is not a
    # number the rules contain - the rule is a doubling, so it is 6.
    #
    # Article 11.2 makes the head a foot-only target and the trunk the only
    # other one, so a leg scores nothing; Article 14.4.1.8 forbids hitting the
    # head with the hand, 14.4.1.9 forbids the knee, and 14.4.1.6 forbids
    # kicking below the waist. Those three flags were already right.
    #
    # A caution on the turning bonus: Explanation #1 to Article 12 requires
    # head *and shoulder* rotation before a back kick counts as a turning kick.
    # `_turned_into_it` reads a `spinning` flag from the pose evidence, which
    # is a proxy for that and not a measurement of it.
    "WT_TAEKWONDO": RuleProfile(
        "WT_TAEKWONDO", "WT · World Taekwondo (Olympic)", False, False, False, True, False, frozenset(),
        sport="taekwondo", sport_label="Taekwondo",
        allow_head_punch=False,
        family_value=(("punch", 0.5), ("kick", 1.6)),
        point_table=(
            ("punch", "body", 1),        # head punches are illegal, so score 0
            ("kick", "body", 2), ("kick", "head", 3),
            ("kick", "leg", 0),
        ),
        turning_kick_points=(("body", 4), ("head", 6)),
        unobserved=(
            "electronic body and head protector scoring",
            # Article 12.3.5: a Gam-jeom against one athlete is a point to the
            # other, and two points inside the last ten seconds of a round.
            # Penalties decide close WT rounds and none of them are visible
            # here - stepping out, falling, grabbing, avoiding.
            "points awarded from penalties against the opponent",
        ),
    ),

    # ---- MMA ----------------------------------------------------------------
    # Standing exchanges only. Takedowns, ground position, control time,
    # ground-and-pound and submissions decide most MMA rounds and none of them
    # are in the detector's vocabulary; the two-fighter pose model is built for
    # upright athletes. This profile is deliberately explicit that a standing
    # striking read is all it offers.
    #
    # Checked against the ABC Unified Rules of Mixed Martial Arts, amended
    # July 2024, on 2026-09-07. Judging Criteria (A)(b) confirms the ten-point
    # must system, and (c) orders the criteria: effective striking/grappling
    # first, then effective aggressiveness, then control of the fighting area,
    # with the latter two "not taken into consideration unless Plan A is
    # weighed as being even". `round_margins` stays empty because the Unified
    # Rules describe the bands in words - a close margin, a large margin by
    # "damage, dominance, and duration" - and publish no strike differential
    # to put a number on, unlike IFMA.
    #
    # Damage is added to the unobserved list because the rules make it the
    # thing a 10-8 turns on, and define it as "visible evidence such as
    # swelling and lacerations" - which is not something 480x220 video shows,
    # and not something this analysis attempts.
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
            "damage, which decides a dominant round",
        ),
    ),
}


# WAKO Chapter 8, Article 6 "Number of kicks per round": in Full Contact each
# kickboxer "is obliged to deliver a minimum of 6 kicks per round", 18 across
# the bout, and a shortfall not made up in the following round costs a minus
# point. It is the one WAKO obligation this analysis can genuinely check, and
# it is a count of kicks *thrown* - the rule asks only that they "clearly show
# the intention to hit the opponent by kicking" - so it is measured against
# attempts, not landed strikes.
#
# Only Full Contact carries it. Searched the whole rulebook: "minimum of N
# kicks" appears once, in Chapter 8.
MINIMUM_KICKS_PER_ROUND: dict[str, int] = {"FULL_CONTACT": 6}


def minimum_kicks_per_round(ruleset: str) -> int | None:
    """How many kicks a round this discipline obliges, if it obliges any."""
    return MINIMUM_KICKS_PER_ROUND.get(normalize_ruleset(ruleset))


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


# How much of the rule-based confidence range a scoring action must clear.
#
# Half of it: a displayed score is a stronger claim than "something was thrown"
# and this sits well above the attempt tier, which asks a quarter. It is
# written as a share for the same reason that one is - the constant here was
# 0.72, fitted to a confidence formula that saturated near 1.0, and it outlived
# a rescale of that formula. Afterwards the formula ran 0.30 to 0.94 with a
# median near 0.50, so 0.72 meant two thirds of maximum evidence on a landed,
# legal, in-range strike. On a taekwondo bout that left 5 verified actions out
# of 19 good ones, and on a kickboxing bout it left none at all: the four that
# landed peaked at 0.70, just under the bar. No score could be shown for either.
SCORING_CONFIDENCE = CONFIDENCE_FLOOR + 0.50 * (CONFIDENCE_CEILING - CONFIDENCE_FLOOR)


# A scored kick has to have taken a foot off the floor, measured against the
# athlete's own torso. Half a torso is far less than any real kick and far more
# than a step.
#
# This exists because footwork was being scored as landed kicks. On a taekwondo
# bout every one of 33 detected kicks had the kicking foot level with or below
# the standing foot - median lift -0.01 torsos - and nine of them were recorded
# as landing. The cause is upstream and cannot be tuned away: across 487
# fighter-frames of that same bout the pose model never once placed a foot more
# than 0.6 torsos above the other, with high ankle confidence throughout, while
# the video plainly shows kicks. It confidently returns a standing pose during
# a kick at this resolution, so what the action stage sees as a fast leg is
# usually a step.
#
# Attempts are unaffected: they are already published as unvalidated candidates.
# This only stops a footstep reaching a scorecard as a landed strike.
MIN_SCORING_FOOT_LIFT = 0.5


def is_verified_scoring_event(event: StrikeEvent, ruleset: str) -> bool:
    """Only evidence strong enough to support a displayed score is counted."""
    if event.family == "kick":
        lift = (event.evidence or {}).get("foot_lift_torsos")
        # None means the pose could not say, which is not evidence of a kick.
        if lift is None or float(lift) < MIN_SCORING_FOOT_LIFT:
            return False
    return (
        is_legal_event(event, ruleset)
        and event.outcome in {"clean", "likely_landed"}
        and float(event.confidence) >= SCORING_CONFIDENCE
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


def _turned_into_it(event: StrikeEvent) -> bool:
    """Was this kick thrown with the athlete's back turning through it?

    Read from the pose evidence recorded when the action was detected, not
    from the technique name: "right_round_kick" is the same label whether it
    was thrown square or off a full spin, and the two are worth different
    numbers of points.
    """
    return bool((event.evidence or {}).get("spinning"))


def _table_points(event: StrikeEvent, profile: RuleProfile) -> int | None:
    """The federation's own value for this technique and target, if it has one.

    Returns None when the ruleset publishes no table, so the caller falls back
    to the generic scorer rather than inventing a value.
    """
    if not profile.point_table:
        return None
    if event.outcome not in {"clean", "likely_landed"}:
        return 0
    if event.family == "kick" and profile.turning_kick_points and _turned_into_it(event):
        for target, points in profile.turning_kick_points:
            if event.target == target:
                return int(points)
    for family, target, points in profile.point_table:
        if event.family == family and event.target == target:
            return int(points)
    # A legal family landing on a target the table does not list scores
    # nothing: the table is the sport's complete account of what counts.
    return 0


def _loser_points(lead: float, profile: RuleProfile, loser_value: float, winner_value: float) -> int:
    """What the losing side of a ten-point-must round scores.

    Where the federation publishes the margins, they are used verbatim: the
    first band the lead falls inside decides it, and a lead past the last band
    takes the last band's value, because a rulebook that stops at "total
    domination" has no wider category to promote it to.

    Everywhere else this is the generic heuristic, kept because boxing and MMA
    genuinely do not publish a table - a judge weighs the round. Its two
    constants were fitted rather than sourced, which is worth knowing when
    reading a 10-8 it produced.
    """
    if profile.round_margins:
        for limit, points in profile.round_margins:
            if lead <= limit:
                return int(points)
        return int(profile.round_margins[-1][1])
    return 8 if lead >= 5.5 and winner_value >= loser_value * 2.0 + 2.0 else 9


def _one_point_per_landed_action(event: StrikeEvent) -> int:
    """Last resort for a counted discipline that publishes no table.

    There is no such discipline today - `test_every_counted_ruleset_publishes_a_table`
    holds that line - and this exists so that adding one cannot silently
    inherit somebody else's point values. Two hand-written mappings used to
    live here, one per tatami discipline, duplicating numbers that the
    profiles above already state; they disagreed with the published tables for
    light contact and kick light for as long as they existed, because nobody
    reading a profile could see that a second copy was what actually ran.
    """
    return 1 if event.outcome in {"clean", "likely_landed"} else 0


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
        "mode": "estimated_10_point_must" if profile.ten_point_must else "estimated_points",
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

    if profile.ten_point_must:
        total_a_rounds = total_b_rounds = 0
        for r in sorted(by_round):
            if profile.round_margins:
                # The federation publishes how a lead becomes a round score,
                # and it counts scoring actions rather than weighing them - so
                # the count is what the margin is read from. IFMA Article
                # 29.2.1 is a table of differences in "scoring Muaythai
                # skills", not of anything weighted, and using the weighted
                # value here would be applying its thresholds to a different
                # quantity than the one it defines them over.
                a_value = float(len(by_round[r]["A"]))
                b_value = float(len(by_round[r]["B"]))
            else:
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
            elif (diff == 0 if profile.round_margins else abs(diff) < 0.35):
                # Equal on count, and IFMA 29.2.1 then separates the round on
                # which athlete used "more forceful" skills. Force is not
                # something this measures, so the round stays even rather than
                # being decided on a tiebreak we cannot see.
                a_score, b_score, winner = 10, 10, "EVEN"
            elif diff > 0:
                a_score, b_score = 10, _loser_points(diff, profile, b_value, a_value)
                winner = "A"
                total_a_rounds += 1
            else:
                a_score, b_score = _loser_points(-diff, profile, a_value, b_value), 10
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
        def scorer(event, _profile=profile):
            table = _table_points(event, _profile)
            return _one_point_per_landed_action(event) if table is None else table
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
