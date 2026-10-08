"""Which strike counts may be published, sport by sport, and on what evidence.

Strike counts were one global switch (WARRIORIQ_PUBLISH_STRIKE_COUNTS): on for
every sport or off for every sport, decided by hand. This module adds a second
way for counts to go live: an exam, run by tools/strike_exam.py, whose verdict
is committed to ``dataset/strike_exam_verdict.json`` and read here.

A family of one sport (boxing punches, MMA kicks ...) passes only when two
independent kinds of human-made evidence agree with the model that made the
report:

* **Clip check, per family.** Held-out clips that people labelled completely
  (public datasets such as TKD-Kick3): of the windows the model calls this
  family, at least ``min_precision`` really are, over at least
  ``min_true_strikes`` real ones. A sparse dataset - one that marks only
  some strikes - cannot measure precision and is not accepted here.
* **Count check, per sport.** Whole fights against totals people counted -
  official per-round statistics today (core/official_stats.py): WarriorIQ's
  count is within ``max_round_error`` of the official total in a typical
  round (median), over at least ``min_rounds`` rounds from ``min_bouts``
  bouts.

Labels WarriorIQ made for itself (tools/auto_label.py, fight ids starting
``auto_``) are training material only; the exam refuses them as answers,
because a model graded against its own guesses cannot fail.

The verdict names the strike model it passed with (``checkpoint_sha256``) and
a report qualifies only if it was made by that exact model
(``classifier.temporal_checkpoint_sha256``, recorded by the worker). Every
report made before this existed, or by any other model, keeps counts off.
Without a verdict file nothing changes at all.

Scores, key moments and illegal-move flags stay behind the global switch: they
depend on whether strikes landed, which this exam does not check yet.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

VERDICT_PATH = Path(__file__).resolve().parents[1] / "dataset" / "strike_exam_verdict.json"
SCHEMA = "warrioriq.strike_exam.v1"

# The bar, as agreed on 2026-10-08. Changing it is a product decision.
BAR = {
    "min_precision": 0.80,
    "min_true_strikes": 50,
    "max_round_error": 0.20,
    "min_rounds": 10,
    "min_bouts": 3,
}

FAMILIES = ("punch", "kick", "knee")
AUTO_PREFIX = "auto_"
# Fight ids that are training material only, never exam answers: WarriorIQ's
# own labels, and datasets that mark only some strikes (an unmarked window may
# hold a strike, so precision cannot be measured on them).
NOT_ANSWER_KEYS = {
    AUTO_PREFIX: "an auto-label; WarriorIQ's own labels are never answers",
    "strikemetrics_": "StrikeMetrics marks only some strikes, so precision cannot be measured on it",
}


def not_an_answer_key(fight_id: str) -> str | None:
    """Why ``fight_id`` may not be used as exam evidence, or None if it may."""
    return next((why for prefix, why in NOT_ANSWER_KEYS.items() if str(fight_id).startswith(prefix)), None)


def family_of(action_class: str | None) -> str | None:
    """core.temporal_model.ACTION_CLASSES name -> strike family (None for "none")."""
    name = str(action_class or "")
    if not name or name == "none":
        return None
    if "knee" in name:
        return "knee"
    if "kick" in name:
        return "kick"
    return "punch"


@dataclass(frozen=True)
class WindowCheck:
    family: str
    true_strikes: int
    predicted: int
    correct: int
    sources: tuple[str, ...]

    @property
    def precision(self) -> float | None:
        return None if self.predicted == 0 else self.correct / self.predicted

    @property
    def recall(self) -> float | None:
        return None if self.true_strikes == 0 else self.correct / self.true_strikes

    def passed(self, bar: dict = BAR) -> bool:
        return (self.true_strikes >= bar["min_true_strikes"] and self.precision is not None
                and self.precision >= bar["min_precision"])

    def as_dict(self, bar: dict = BAR) -> dict:
        return {"family": self.family, "true_strikes": self.true_strikes, "predicted": self.predicted,
                "correct": self.correct, "precision": _round(self.precision), "recall": _round(self.recall),
                "sources": list(self.sources), "passed": self.passed(bar)}


def window_checks(pairs, sources_by_family: dict[str, set] | None = None) -> dict[str, WindowCheck]:
    """Per-family precision and recall from (true class, predicted class) pairs.

    A window counts for a family's precision when the model predicted that
    family, and is correct when the truth is the same family. Side and exact
    technique are not required: a count is per family.
    """
    true_n = {family: 0 for family in FAMILIES}
    predicted = {family: 0 for family in FAMILIES}
    correct = {family: 0 for family in FAMILIES}
    for truth, guess in pairs:
        t, g = family_of(truth), family_of(guess)
        if t in true_n:
            true_n[t] += 1
        if g in predicted:
            predicted[g] += 1
            if g == t:
                correct[g] += 1
    sources_by_family = sources_by_family or {}
    return {family: WindowCheck(family, true_n[family], predicted[family], correct[family],
                                tuple(sorted(sources_by_family.get(family, ()))))
            for family in FAMILIES}


@dataclass(frozen=True)
class CountCheck:
    sport: str
    rounds: tuple[tuple[str, int, int], ...]   # (bout, ours, truth)
    truth_source: str

    @property
    def errors(self) -> list[float]:
        return [abs(ours - truth) / truth for _, ours, truth in self.rounds if truth > 0]

    @property
    def median_error(self) -> float | None:
        return statistics.median(self.errors) if self.errors else None

    @property
    def bouts(self) -> int:
        return len({bout for bout, _, _ in self.rounds})

    def passed(self, bar: dict = BAR) -> bool:
        return (len(self.errors) >= bar["min_rounds"] and self.bouts >= bar["min_bouts"]
                and self.median_error is not None and self.median_error <= bar["max_round_error"])

    def as_dict(self, bar: dict = BAR) -> dict:
        within = [e for e in self.errors if e <= bar["max_round_error"]]
        return {"sport": self.sport, "truth_source": self.truth_source, "rounds": len(self.errors),
                "bouts": self.bouts, "median_round_error": _round(self.median_error),
                "rounds_within_bar": len(within), "passed": self.passed(bar)}


def decide(windows: dict[str, WindowCheck], counts: dict[str, CountCheck], sports, *,
           checkpoint_sha256: str, bar: dict = BAR, scored_families=None) -> dict:
    """The verdict: a family of a sport passes when its clip check and its sport's count check both pass."""
    verdict = {"schema": SCHEMA, "checkpoint_sha256": checkpoint_sha256, "bar": dict(bar),
               "families": {f: w.as_dict(bar) for f, w in windows.items()}, "sports": {}}
    for sport in sports:
        count = counts.get(sport)
        families = scored_families(sport) if scored_families else FAMILIES
        entry = {"count_check": count.as_dict(bar) if count else None, "passed_families": []}
        if count is not None and count.passed(bar):
            entry["passed_families"] = [f for f in families if f in windows and windows[f].passed(bar)]
        verdict["sports"][sport] = entry
    return verdict


def _round(value):
    return None if value is None else round(float(value), 3)


# --- reading the verdict -----------------------------------------------------

_cache: dict = {"key": None, "verdict": None}


def load_verdict(path: Path | None = None) -> dict | None:
    """The committed verdict, or None when there is none or it is unreadable."""
    path = Path(path or VERDICT_PATH)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _cache["key"] != key:
        try:
            verdict = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            verdict = None
        if not isinstance(verdict, dict) or verdict.get("schema") != SCHEMA:
            verdict = None
        _cache.update(key=key, verdict=verdict)
    return _cache["verdict"]


def report_model(report: dict | None) -> str | None:
    """The strike model that made ``report``, as the worker recorded it."""
    classifier = (report or {}).get("classifier") or {}
    value = classifier.get("temporal_checkpoint_sha256")
    return str(value) if value else None


def passed_families(sport: str | None, report: dict | None, *, verdict: dict | None = None) -> tuple[str, ...]:
    """Families of ``sport`` the exam lets ``report`` show. Empty unless every condition holds."""
    verdict = verdict if verdict is not None else load_verdict()
    model = report_model(report)
    if not verdict or not model or verdict.get("checkpoint_sha256") != model:
        return ()
    entry = (verdict.get("sports") or {}).get(str(sport or ""))
    if not isinstance(entry, dict):
        return ()
    return tuple(f for f in entry.get("passed_families") or () if f in FAMILIES)


def exam_note(sport: str | None, *, verdict: dict | None = None) -> str:
    """One sentence on what the exam measured for ``sport``, for the report."""
    verdict = verdict if verdict is not None else load_verdict()
    entry = ((verdict or {}).get("sports") or {}).get(str(sport or "")) or {}
    count = entry.get("count_check") or {}
    passed = entry.get("passed_families") or []
    if not passed or not count:
        return ""
    precisions = [((verdict.get("families") or {}).get(f) or {}).get("precision") for f in passed]
    lowest = min(p for p in precisions if p is not None) if any(p is not None for p in precisions) else None
    parts = []
    if lowest is not None:
        parts.append(f"at least {lowest * 100:.0f}% of counted strikes were real strikes on clips people labelled")
    if count.get("median_round_error") is not None:
        parts.append(f"a typical round was within {count['median_round_error'] * 100:.0f}% of the "
                     f"official count over {count.get('rounds')} rounds")
    return ("Automatic counts that passed WarriorIQ's accuracy check: " + " and ".join(parts) + "."
            if parts else "")
