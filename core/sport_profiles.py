"""What each sport is, and what that means for a fighter's report.

Two things live here. The first is identity: an accent, a crest letterform and
one honest line about what decides a bout, so a Muay Thai report does not look
and read exactly like a boxing one.

The second is the part that matters. A ruleset already declares what each
family of strike is worth (``RuleProfile.family_value``), which means the gap
between *what a sport rewards* and *what a fighter actually threw* is a real,
measured quantity rather than an opinion. A fighter who throws 88% punches in
WT taekwondo is not being given a style note here; they are being shown that
they spent the bout on the sport's least valuable action. That comparison is
the whole basis of the sport-specific coaching below, and it is why none of it
is hand-written advice: every observation is computed from detected attempts
and carries the timestamps that produced it.

Nothing here invents a claim the detector cannot support. Where a sport scores
something WarriorIQ cannot observe, the profile says so and the panel stays
quiet about it rather than filling the space.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.scoring import RULESETS, normalize_ruleset
from core.types import StrikeEvent


# Enough attempts that a share is a tendency rather than an accident. Below
# this the weapon-mix comparison is suppressed entirely.
MIN_ATTEMPTS_FOR_MIX = 8

# A family is "central" to a sport when it carries at least this share of the
# available scoring value. Below it, silence on that family is not a finding.
CENTRAL_FAMILY_SHARE = 0.34


@dataclass(frozen=True)
class SportIdentity:
    """The visual and editorial character of one sport."""

    key: str
    label: str
    # Two-letter crest. Drawn rather than an emoji so it stays legible at 22px
    # and does not change shape between platforms.
    mark: str
    accent: str            # rgb triplet, consumed as `rgb(var(--sport-accent))`
    accent_soft: str
    # One line on what actually decides a bout. Shown under the sport's name.
    decided_by: str
    # The reading a WarriorIQ report gives in this sport, stated plainly.
    report_frame: str


SPORT_IDENTITIES: dict[str, SportIdentity] = {
    "kickboxing": SportIdentity(
        "kickboxing", "Kickboxing", "KB", "92 130 239", "150 178 255",
        decided_by="Hands and feet score together; the ruleset decides which legs and knees count.",
        report_frame="Every action that scores in kickboxing is one WarriorIQ observes directly.",
    ),
    "boxing": SportIdentity(
        "boxing", "Boxing", "BX", "233 106 106", "246 168 168",
        decided_by="Punches, and only punches. Output, accuracy and defence carry the round.",
        report_frame="The one sport read in full: its entire scoring vocabulary is punches.",
    ),
    "muay_thai": SportIdentity(
        "muay_thai", "Muay Thai", "MT", "226 154 74", "244 197 138",
        decided_by="Kicks and knees outscore hands, and the clinch decides close rounds.",
        report_frame="Punches, kicks and knees are read. Elbows and clinch work are not.",
    ),
    "taekwondo": SportIdentity(
        "taekwondo", "Taekwondo", "TK", "85 198 223", "150 224 240",
        decided_by="Kicks decide it. Height and turning multiply what a kick is worth.",
        report_frame="Kick and punch counts are read; electronic and rotation bonuses are not.",
    ),
    "mma": SportIdentity(
        "mma", "MMA", "MM", "160 138 232", "196 180 244",
        decided_by="Standing exchanges are one part of it. Most rounds turn on the ground.",
        report_frame="A standing striking read only — takedowns and ground work are unread.",
    ),
}


# "punch" does not pluralise by adding an s, and the report says these words a
# dozen times per page. Kept here rather than in the template so the coaching
# text and the chart labels cannot disagree.
FAMILY_PLURAL = {"punch": "Punches", "kick": "Kicks", "knee": "Knees"}


def family_plural(family: str) -> str:
    return FAMILY_PLURAL.get(family, f"{family.title()}s")


def sport_identity(sport: str) -> SportIdentity:
    return SPORT_IDENTITIES.get(sport, SPORT_IDENTITIES["kickboxing"])


def reward_shares(ruleset: str) -> dict[str, float]:
    """How the sport's scoring value is split across the families it allows.

    Derived from the ruleset's own weights rather than restated, so a change to
    scoring cannot silently disagree with the coaching built on top of it.
    """
    profile = RULESETS[normalize_ruleset(ruleset)]
    allowed = {
        "punch": profile.allow_punch,
        "kick": profile.allow_kick,
        "knee": profile.allow_knee,
    }
    weights = {family: value for family, value in profile.family_value if allowed.get(family)}
    total = sum(weights.values())
    if not total:
        return {}
    return {family: value / total for family, value in weights.items()}


def observed_shares(families: dict[str, int]) -> dict[str, float]:
    total = sum(families.values())
    if not total:
        return {}
    return {family: count / total for family, count in families.items()}


def _times(events: list[StrikeEvent], fighter: str, family: str, limit: int = 5) -> list[float]:
    return [
        round(float(event.peak_time), 2)
        for event in events
        if event.fighter == fighter and event.family == family
    ][:limit]


def build_sport_coaching(
    fighter: str, metrics: dict, events: list[StrikeEvent], ruleset: str,
) -> dict:
    """Read a fighter's weapon mix against what this sport actually rewards.

    Returns the same observation shape the general coaching uses, so the report
    renders one kind of thing. An empty ``observations`` list is a valid and
    honest result: too few attempts to read a tendency.
    """
    key = normalize_ruleset(ruleset)
    profile = RULESETS[key]
    identity = sport_identity(profile.sport)
    attacks = metrics.get(fighter, {}).get("attacks", {}) or {}
    families = {
        family: int(count)
        for family, count in (attacks.get("families") or {}).items()
        if count
    }
    attempts = sum(families.values())
    rewards = reward_shares(key)
    shares = observed_shares(families)

    observations: list[dict] = []

    # An action the sport does not score at all. In WT taekwondo a punch to the
    # head is a penalty; in boxing a kick is not a shot, it is a foul. This is
    # the one finding worth reporting on a single occurrence.
    for family, count in families.items():
        legal = {"punch": profile.allow_punch, "kick": profile.allow_kick,
                 "knee": profile.allow_knee}.get(family, True)
        if not legal:
            observations.append({
                "tone": "improvement",
                "title": f"{family_plural(family)} do not score in {identity.label} · {count} detected",
                "detail": (
                    f"{count} {family} attempt(s) were detected under "
                    f"{profile.label}, where they carry no scoring value."
                ),
                "evidence_times": _times(events, fighter, family),
            })

    if attempts >= MIN_ATTEMPTS_FOR_MIX and rewards:
        central = [f for f, share in rewards.items() if share >= CENTRAL_FAMILY_SHARE]
        for family in sorted(central, key=lambda f: -rewards[f]):
            observed = shares.get(family, 0.0)
            expected = rewards[family]
            if observed < expected * 0.5:
                observations.append({
                    "tone": "improvement",
                    "title": f"Under-using the {family} · {observed * 100:.0f}% of attempts",
                    "detail": (
                        f"{identity.label} puts {expected * 100:.0f}% of its scoring value on "
                        f"{family}s, and {observed * 100:.0f}% of Fighter {fighter}'s "
                        f"{attempts} detected attempts were {family}s."
                    ),
                    "evidence_times": _times(events, fighter, family),
                })
            elif observed >= expected:
                observations.append({
                    "tone": "strength",
                    "title": f"Weapon mix suits the ruleset · {observed * 100:.0f}% {family}s",
                    "detail": (
                        f"{observed * 100:.0f}% of {attempts} detected attempts were {family}s, "
                        f"against the {expected * 100:.0f}% of scoring value {identity.label} "
                        f"places there."
                    ),
                    "evidence_times": _times(events, fighter, family),
                })

    return {
        "sport": profile.sport,
        "sport_label": identity.label,
        "mark": identity.mark,
        "ruleset_label": profile.label,
        "decided_by": identity.decided_by,
        "report_frame": identity.report_frame,
        "unobserved": list(profile.unobserved),
        "attempts_read": attempts,
        "reward_shares": rewards,
        "observed_shares": shares,
        "family_labels": {family: family_plural(family) for family in rewards},
        "observations": observations,
        # Said plainly when there is not enough to read, rather than padding
        # the panel with advice the footage does not support.
        "insufficient_evidence": attempts < MIN_ATTEMPTS_FOR_MIX,
    }
