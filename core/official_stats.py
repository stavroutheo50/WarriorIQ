"""Measure WarriorIQ's strike counts against a fight's official statistics.

Nobody labels anything here. For UFC bouts the official per-round numbers are
public (ufcstats.com: strikes landed of attempted, by head/body/leg and by
distance/clinch/ground, knockdowns, takedowns), and a public export of them is
kept current at github.com/Greco1899/scrape_ufc_stats (``ufc_fight_stats.csv``).
Analyse a bout in WarriorIQ, give this the result and the bout's names, and it
says round by round how far each count is from the official one.

Two definitions differ and both are reported rather than one chosen:

  * ``TOTAL STR.`` is every strike the official statisticians counted;
  * ``SIG.STR.`` leaves out light strikes (mostly in the clinch and on the
    ground). WarriorIQ has no "significant" rule, so neither is exactly its
    definition; the gap between the two official numbers is the honest margin.

Pure functions over plain rows and events, so the comparison is tested
without a video. Strike statistics are only as right as who-is-who: run this on
results whose identity verdict is trusted, or the comparison measures swaps.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from core.fight_stats import normalize_outcome

TARGETS = ("head", "body", "leg")
STRIKE_FAMILIES = ("punch", "kick", "knee", "elbow")


@dataclass
class Count:
    landed: int = 0
    attempted: int = 0

    def as_dict(self) -> dict:
        return {"landed": self.landed, "attempted": self.attempted}


@dataclass
class RoundLine:
    """One fighter's numbers in one round."""

    total: Count = field(default_factory=Count)
    significant: Count = field(default_factory=Count)
    by_target: dict[str, Count] = field(default_factory=lambda: {t: Count() for t in TARGETS})
    knockdowns: int = 0
    takedowns: Count = field(default_factory=Count)


def _of(text: str) -> Count:
    """``"29 of 73"`` -> Count(29, 73). A dash or blank is no data, not zero."""
    match = re.fullmatch(r"\s*(\d+)\s+of\s+(\d+)\s*", str(text or ""))
    if not match:
        raise ValueError(f"not an 'X of Y' count: {text!r}")
    return Count(int(match.group(1)), int(match.group(2)))


def _name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def official_rounds(rows: list[dict], *, bout: str, fighter: str, event: str | None = None) -> dict[int, RoundLine]:
    """One fighter's official lines in one bout, keyed by round number.

    ``bout`` and ``fighter`` match ignoring case, accents and punctuation; a
    bout that matches more than once (a rematch) needs ``event`` too.
    """
    wanted_bout, wanted_fighter = _name(bout), _name(fighter)
    wanted_event = _name(event) if event else None
    matched = [
        row for row in rows
        if _name(row.get("BOUT", "")) == wanted_bout and _name(row.get("FIGHTER", "")) == wanted_fighter
        and (wanted_event is None or _name(row.get("EVENT", "")) == wanted_event)
    ]
    if not matched:
        raise LookupError(f"no official statistics for {fighter!r} in {bout!r}")
    events = {_name(row.get("EVENT", "")) for row in matched}
    if len(events) > 1:
        raise LookupError(f"{bout!r} was fought more than once; give the event name")
    lines: dict[int, RoundLine] = {}
    for row in matched:
        number = re.search(r"\d+", str(row.get("ROUND", "")))
        if not number:
            continue
        lines[int(number.group())] = RoundLine(
            total=_of(row["TOTAL STR."]),
            significant=_of(row["SIG.STR."]),
            by_target={target: _of(row[target.upper()]) for target in TARGETS},
            knockdowns=int(float(row.get("KD") or 0)),
            takedowns=_of(row["TD"]),
        )
    return lines


def load_official(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def warrioriq_rounds(events: list[dict], fighter: str) -> dict[int, RoundLine]:
    """WarriorIQ's strikes for ``fighter`` ("A" or "B"), on the official lines' terms."""
    lines: dict[int, RoundLine] = {}
    for event in events:
        if event.get("fighter") != fighter or str(event.get("family")) not in STRIKE_FAMILIES:
            continue
        try:
            number = int(event.get("round_number"))
        except (TypeError, ValueError):
            continue
        line = lines.setdefault(number, RoundLine())
        landed = normalize_outcome(event.get("outcome")) == "landed"
        for count in (line.total, line.significant):
            count.attempted += 1
            count.landed += int(landed)
        target = str(event.get("target") or "")
        if target in line.by_target:
            line.by_target[target].attempted += 1
            line.by_target[target].landed += int(landed)
    return lines


def _error(ours: int, official: int) -> dict:
    return {"warrioriq": ours, "official": official, "difference": ours - official,
            "relative": None if official == 0 else round((ours - official) / official, 3)}


def compare(ours: dict[int, RoundLine], official: dict[int, RoundLine]) -> dict:
    """Round-by-round differences, plus the totals over every official round.

    A round WarriorIQ did not analyse is listed as missing rather than counted
    as zero strikes, which would read as a huge undercount it did not make.
    """
    rounds = []
    totals = {key: [0, 0] for key in ("attempted_vs_total", "landed_vs_total",
                                     "attempted_vs_significant", "landed_vs_significant")}
    missing = []
    for number in sorted(official):
        if number not in ours:
            missing.append(number)
            continue
        mine, theirs = ours[number], official[number]
        rounds.append({
            "round": number,
            "attempted_vs_total": _error(mine.total.attempted, theirs.total.attempted),
            "landed_vs_total": _error(mine.total.landed, theirs.total.landed),
            "attempted_vs_significant": _error(mine.total.attempted, theirs.significant.attempted),
            "landed_vs_significant": _error(mine.total.landed, theirs.significant.landed),
            "landed_by_target": {t: _error(mine.by_target[t].landed, theirs.by_target[t].landed)
                                 for t in TARGETS},
        })
        for key, (a, b) in (("attempted_vs_total", (mine.total.attempted, theirs.total.attempted)),
                            ("landed_vs_total", (mine.total.landed, theirs.total.landed)),
                            ("attempted_vs_significant", (mine.total.attempted, theirs.significant.attempted)),
                            ("landed_vs_significant", (mine.total.landed, theirs.significant.landed))):
            totals[key][0] += a
            totals[key][1] += b
    return {
        "rounds": rounds,
        "rounds_missing_from_warrioriq": missing,
        "totals": {key: _error(a, b) for key, (a, b) in totals.items()},
    }
