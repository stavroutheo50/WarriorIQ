"""One name, one definition and one reference point per measurement.

The report showed roughly six distinct numbers and restated each of them up to
four times under different names. `ring_center_control` appeared as "Centre",
"Center position", "Ring position" and "Centre control"; `pose_coverage` as
"Seen", "Pose evidence", "Observed" and "Observation coverage". A reader
comparing two sections had no way to know they were looking at the same
measurement twice rather than at two that happened to agree.

Worse, most of them were bare percentages. "Guard 14%" says nothing without
knowing whether 14% is good, and only the training block ever supplied a
target.

So each measurement is defined once here, with:

  * **one name**, used everywhere it appears;
  * **a direction**, saying which way is better, or that nobody knows;
  * **a band**, when the codebase genuinely holds one.

Where the bands come from
-------------------------
They are not invented for this file. `core.coaching` already decides what to
tell an athlete to work on, and those thresholds are the only numbers in the
project that say what good looks like:

    guard_index    >= 0.62 is called a strength, < 0.42 is called a weakness
    balance_index  < 0.48 is called a weakness

They are coaching heuristics rather than validated population norms, and they
are labelled that way in `band_source` so the report cannot present them as
more than they are.

`ring_center_control`, `pressure_index` and `footwork_body_lengths_per_second`
have no such threshold anywhere. Rather than inventing one, their band is None
and the report says the reference is not known - which is the honest answer
and the one the audit asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

# The thresholds core/coaching.py acts on. Imported as literals rather than
# from that module to avoid a circular import; the test in
# tests/test_metric_catalog.py asserts they still match its source.
GUARD_STRENGTH = 0.62
GUARD_WEAKNESS = 0.42
BALANCE_WEAKNESS = 0.48

# How reliable the rest of a report is. Below this the page already warns, and
# the number is the one the analyse-time gate reads.
POSE_COVERAGE_USABLE = 0.70


@dataclass(frozen=True)
class Metric:
    """A measurement as the reader meets it."""

    key: str
    # The one name. Not "Centre" here and "Ring position" three sections down.
    name: str
    # One sentence, in the second person, saying what was measured.
    definition: str
    unit: str                      # "percent" or "rate"
    direction: str                 # "higher", "lower" or "unknown"
    # The band, as (needs_work_below, doing_well_at_or_above). Either may be
    # None. The whole band is None when nothing in the project knows.
    band: tuple[float | None, float | None] | None
    band_source: str

    def reading(self, value: float | None) -> dict:
        """Where one measured value sits against its band."""
        if value is None:
            return {"metric": self, "value": None, "standing": "not_measured",
                    "reference": "Not measured in this fight."}
        if self.band is None:
            return {
                "metric": self, "value": value, "standing": "no_reference",
                # The audit's instruction, taken literally: say so rather than
                # showing a bare percentage.
                "reference": "WarriorIQ has no reference for this yet, so the "
                             "number is a measurement rather than a verdict.",
            }
        low, high = self.band
        if high is not None and value >= high:
            standing = "doing_well"
        elif low is not None and value < low:
            standing = "needs_work"
        else:
            standing = "middle"
        return {"metric": self, "value": value, "standing": standing,
                "reference": self._sentence(standing)}

    def _sentence(self, standing: str) -> str:
        """Word the band that actually exists.

        Most of these have a lower bound and no upper one - there is a level
        below which WarriorIQ says "train this", and nothing above it that the
        project is willing to call excellent. Describing that as "between the
        two" invented a second bound: 94% coverage read as "between the two:
        under 70% is worth training", which is both bounds' worth of confidence
        about one.
        """
        low, high = self.band
        if standing == "doing_well":
            return f"{self._as_text(high)} or better is where this stops being "\
                   f"something to work on. {self.band_source}".strip()
        if standing == "needs_work":
            return f"Under {self._as_text(low)} is what WarriorIQ flags as worth "\
                   f"training. {self.band_source}".strip()
        if low is not None and high is None:
            return f"Above {self._as_text(low)}, which is where WarriorIQ stops "\
                   f"flagging this. Nothing here says what excellent looks "\
                   f"like. {self.band_source}".strip()
        if low is None and high is not None:
            return f"Under {self._as_text(high)}, the level WarriorIQ calls a "\
                   f"strength. {self.band_source}".strip()
        return (f"Between {self._as_text(low)}, under which WarriorIQ says train "
                f"this, and {self._as_text(high)}, at which it calls it a "
                f"strength. {self.band_source}").strip()

    def _as_text(self, value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value * 100:.0f}%" if self.unit == "percent" else f"{value:.2f}"


COACHING_BAND = "Taken from the thresholds WarriorIQ's own coaching acts on, " \
                "not from a population of athletes."

CATALOG: tuple[Metric, ...] = (
    Metric(
        key="pose_coverage",
        name="Seen",
        definition="The share of the video in which the camera could see you "
                   "clearly enough to measure anything.",
        unit="percent", direction="higher",
        band=(POSE_COVERAGE_USABLE, None),
        band_source="Under 70% the measurements below rest on less of the fight.",
    ),
    Metric(
        key="guard_index",
        name="Guard",
        definition="The share of the fight your hands stayed up near your head.",
        unit="percent", direction="higher",
        band=(GUARD_WEAKNESS, GUARD_STRENGTH),
        band_source=COACHING_BAND,
    ),
    Metric(
        key="balance_index",
        name="Balance",
        definition="The share of the fight you stayed in a balanced stance.",
        unit="percent", direction="higher",
        band=(BALANCE_WEAKNESS, None),
        band_source=COACHING_BAND,
    ),
    Metric(
        key="ring_center_control",
        name="Centre",
        definition="The share of the fight you spent in the middle rather than "
                   "on the outside.",
        unit="percent",
        # Holding the centre is usually read as pressure, but a counter-fighter
        # giving ground on purpose is not doing worse. Nothing here measures
        # which of the two this is.
        direction="unknown", band=None,
        band_source="",
    ),
    Metric(
        key="footwork_body_lengths_per_second",
        name="Movement",
        definition="How far you travel each second, in your own body lengths.",
        unit="rate", direction="unknown", band=None,
        band_source="",
    ),
    Metric(
        key="pressure_index",
        name="Pressure",
        definition="Whether you walked at your opponent or gave ground.",
        # Stored as -1..1 and shown as 0-100 by the template, which is the one
        # measurement here whose displayed number is not its stored one. Marked
        # "rate" so nothing in this file formats it as a percentage of the
        # stored value; the band is None, so nothing does.
        unit="rate", direction="unknown", band=None,
        band_source="",
    ),
)

BY_KEY: dict[str, Metric] = {metric.key: metric for metric in CATALOG}

# Names that used to appear for the same measurement, kept so a test can prove
# none of them came back. Every one of these was on the page at the same time
# as the canonical name above.
RETIRED_NAMES: dict[str, tuple[str, ...]] = {
    "pose_coverage": ("Pose evidence", "Observation coverage", "Observed", "Pose coverage"),
    "guard_index": ("Guard baseline",),
    "balance_index": ("Balance baseline",),
    "ring_center_control": ("Center position", "Ring position", "Centre control"),
    "footwork_body_lengths_per_second": ("Footwork pace",),
}


def readings(metrics: dict | None) -> list[dict]:
    """Every catalogued measurement for one fighter, in reading order."""
    values = metrics or {}
    return [metric.reading(values.get(metric.key)) for metric in CATALOG]
