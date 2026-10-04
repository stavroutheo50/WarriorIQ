"""What a WarriorIQ report counts in each sport, and how far to trust it.

QA, 2026-09, found every surface telling a different story about the same
counts: the kickboxing upload page said "your report counts punches, kicks and
knees", the live page said "leg strikes only... punches are not shown here or
in the report" while its own feed listed punch attempts, a taekwondo report
showed "15 punches - 13 kicks", the boxing upload page warned that "it often
mixed up punches and kicks" in a sport with no kicks, and the replay said no
action passed verification while the report offered "Watch every counted
strike (57)".

Each of those sentences was written in a different file at a different time,
against a different policy. They now come from here, built from the two facts
that decide them:

  * what the sport scores - ``core.scoring.sport_counted_families`` and
    ``sport_unobserved``, from the ruleset definitions themselves;
  * whether counts are published and whether their precision is validated -
    ``core.report.STRIKE_COUNTS_PUBLISHED`` and
    ``STRIKE_COUNTS_PRECISION_VALIDATED``, read at call time so a change to
    either moves every sentence at once.

The accuracy sentence is the one measurement there is: 72 clips of one Kick
Light bout, labelled by a competitor (dataset/regression/kicklight_stavrou_
ceschia, tools/benchmark_labelled_fight.py). Of the 16 moments counted, 11 were
real strikes and 6 of those the right type. Nothing else has been checked by
hand, so a sport other than kickboxing says exactly that rather than borrowing
a kickboxing number as if it were its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.scoring import sport_counted_families, sport_unobserved

SPORT_LABELS = {
    "kickboxing": "Kickboxing", "boxing": "Boxing", "muay_thai": "Muay Thai",
    "taekwondo": "Taekwondo", "mma": "MMA",
}

# The accuracy sentence, per sport. Kickboxing is the one sport with a hand
# check; every other sport says so rather than borrowing that number as if it
# were its own, and nothing here names a strike the sport does not have.
_OTHER_SPORT_CHECK = ("Counting has been checked by hand on one bout of another sport so far, "
                      "where about two in three counted strikes were real; ")
_ACCURACY = {
    "kickboxing": ("On a kickboxing fight we checked by hand, about two in three of the strikes "
                   "WarriorIQ counted were real, and it often mixed up punches, kicks and knees, "
                   "so treat these as estimates."),
    "boxing": (_OTHER_SPORT_CHECK + "boxing footage has not been checked by hand yet, so treat "
               "these as estimates. Only punches are counted; footwork never is."),
    "muay_thai": (_OTHER_SPORT_CHECK + "Muay Thai footage has not been checked by hand yet, and "
                  "the strike type is often wrong, so treat these as estimates."),
    "taekwondo": (_OTHER_SPORT_CHECK + "taekwondo footage has not been checked by hand yet, and "
                  "the strike type is often wrong, so treat these as estimates."),
    "mma": (_OTHER_SPORT_CHECK + "MMA footage has not been checked by hand yet, so treat these "
            "as estimates."),
}

# Whole areas of the sport a report never reads, in the reader's words. The
# ruleset's own `unobserved` list is the authority; these are the short forms
# for a single line on the upload page.
_NOT_ANALYSED = {
    "kickboxing": (),
    "boxing": (),
    "muay_thai": ("elbows", "clinch work and sweeps"),
    "taekwondo": ("electronic-protector scoring", "height and turning bonuses"),
    "mma": ("takedowns", "ground work", "submissions", "elbows"),
}

_MEASURED_FROM_BODY = "Movement, pressure, ring centre, guard and balance are measured from the fighters' bodies."


def _prose(items) -> str:
    items = [str(item) for item in items if item]
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} and {items[-1]}"


@dataclass(frozen=True)
class CountingPolicy:
    sport: str
    label: str
    # Plural family names, as people read them: ("punches", "kicks").
    counted: tuple[str, ...]
    withheld: tuple[str, ...]
    estimates: bool
    estimate_note: str
    not_analysed: tuple[str, ...]
    setup_line: str
    live_note: str
    report_frame: str
    replay_note: str
    badge: str

    @property
    def counted_singular(self) -> tuple[str, ...]:
        return tuple({"punches": "punch", "kicks": "kick", "knees": "knee"}.get(f, f) for f in self.counted)

    @property
    def not_analysed_line(self) -> str:
        if not self.not_analysed:
            return ""
        line = f"Not analysed: {_prose(self.not_analysed)}."
        if self.sport == "mma":
            line += " Most MMA rounds turn on those, so this report covers standing exchanges only."
        return line

    def as_dict(self) -> dict:
        return {
            "sport": self.sport, "label": self.label,
            "counted": list(self.counted), "counted_text": _prose(self.counted),
            "withheld": list(self.withheld), "withheld_text": _prose(self.withheld),
            "estimates": self.estimates, "estimate_note": self.estimate_note,
            "not_analysed": list(self.not_analysed), "not_analysed_line": self.not_analysed_line,
            "setup_line": self.setup_line, "live_note": self.live_note,
            "report_frame": self.report_frame, "replay_note": self.replay_note, "badge": self.badge,
        }


def counting_policy(sport: str | None, *, published: bool | None = None,
                    validated: bool | None = None) -> CountingPolicy:
    """The one answer to "what does this sport's report count, and how well"."""
    from core import report as _report

    sport = sport if sport in SPORT_LABELS else "kickboxing"
    label = SPORT_LABELS[sport]
    published = _report.STRIKE_COUNTS_PUBLISHED if published is None else published
    validated = _report.STRIKE_COUNTS_PRECISION_VALIDATED if validated is None else validated
    scored = tuple(sport_counted_families(sport))
    if published or validated:
        counted, withheld = scored, ()
    else:
        # Switched off, nothing is counted - the same answer as
        # core.report.published_families.
        counted, withheld = (), scored
    estimates = bool(counted) and not validated
    not_analysed = _NOT_ANALYSED.get(sport, ()) if sport_unobserved(sport) else ()

    if estimates:
        estimate_note = f"Automatic counts, not checked by a person. {_ACCURACY[sport]}"
    elif counted:
        estimate_note = "Counts are validated against hand-labelled footage."
    else:
        estimate_note = ""

    if counted:
        setup_line = (f"Your report counts {_prose(counted)}"
                      + (" as automatic estimates." if estimates else "."))
        if withheld:
            setup_line += f" {_prose(withheld).capitalize()} are not counted yet."
    else:
        setup_line = f"{label} reports have no strike counts yet. {_MEASURED_FROM_BODY}"

    if counted:
        live_note = (f"<strong>Live attempts:</strong> {_prose(counted)} as they are detected - "
                     "automatic, not checked by a person. Landed, missed, blocked, target and "
                     "scoring are not shown here; the report adds them as estimates once the "
                     "whole fight is read.")
    else:
        live_note = ("<strong>Live view:</strong> no strike counts yet. "
                     "Identity and movement are being measured.")

    if counted:
        report_frame = (f"{label} reports count {_prose(counted)}"
                        + (", as estimates." if estimates else ".")
                        + (f" {_prose(withheld).capitalize()} are not counted yet." if withheld else "")
                        + (f" Not analysed: {_prose(not_analysed)}." if not_analysed else ""))
    else:
        report_frame = f"{label} reports have no strike counts yet. {_MEASURED_FROM_BODY}"

    replay_note = ("Each timestamp is a strike WarriorIQ counted automatically - an estimate, not "
                   "checked by a person. Selecting it starts one second earlier so you can judge it "
                   "yourself.")

    if not counted:
        badge = "No punch counts yet" if sport == "boxing" else "No strike counts yet"
    elif sport == "mma":
        badge = "Strikes only, no grappling yet"
    elif withheld:
        badge = "Kick counts only"
    else:
        badge = f"Counts {_prose(counted)}"

    return CountingPolicy(
        sport=sport, label=label, counted=counted, withheld=withheld, estimates=estimates,
        estimate_note=estimate_note, not_analysed=not_analysed, setup_line=setup_line,
        live_note=live_note, report_frame=report_frame, replay_note=replay_note, badge=badge,
    )
