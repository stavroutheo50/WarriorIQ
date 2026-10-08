"""Which report features are switched on, and how every page talks about them.

QA, 2026-10-07: the homepage hero, the meta description and "Confidence-gated
scorecards" promised a round-by-round score estimate, and every current report
said "Not scored", because strikes cannot be read reliably yet. The pricing
cards promised verified key moments and a legality review that no report
showed. Each surface had its own wording and nothing tied it to the switch
that actually decides what a report contains.

So each feature has one status here, and the marketing copy, the meta
description, the pricing cards and the report sections all read it: while a
feature is off its copy says it is coming, or the copy leaves it out.

Strike counts keep their existing switch, WARRIORIQ_PUBLISH_STRIKE_COUNTS
(see core/report.py STRIKE_COUNTS_PUBLISHED for the measurements behind it).
The other three are built from those counts - a score adds them up, a key
moment is a counted strike, a legality flag judges one - so each can be on
only while strike counts are, and each can also be held off on its own with
WARRIORIQ_FEATURE_<NAME>=0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    key: str
    name: str
    on: bool
    # A short phrase for lists ("Round-by-round score estimate").
    label: str
    # What copy says while the feature is off.
    coming: str


def _env_on(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _load() -> dict[str, Feature]:
    counts = _env_on("WARRIORIQ_PUBLISH_STRIKE_COUNTS", "0")

    def built_on_counts(key: str) -> bool:
        return counts and _env_on(f"WARRIORIQ_FEATURE_{key.upper()}", "1")

    features = (
        Feature("strike_counts", "Strike counts", counts,
                "Punch, kick and knee counts",
                "Strike counts are coming: counting punches, kicks and knees automatically is not "
                "accurate enough yet, so reports show movement, guard, balance, centre and pressure."),
        Feature("scoring", "Score estimate", built_on_counts("scoring"),
                "Round-by-round score estimate under your sport's rules",
                "Round-by-round score estimates are coming: a score is built from strike counts, "
                "which are not accurate enough yet."),
        Feature("key_moments", "Key moments", built_on_counts("key_moments"),
                "Verified key moments to replay",
                "Key moments are coming: they are counted strikes, which are not accurate enough yet."),
        Feature("illegal_moves", "Illegal-move flags", built_on_counts("illegal_moves"),
                "Illegal-move flags under your ruleset",
                "Illegal-move flags are coming: they judge counted strikes, which are not accurate "
                "enough yet."),
    )
    return {feature.key: feature for feature in features}


FEATURES: dict[str, Feature] = _load()


def is_on(key: str) -> bool:
    feature = FEATURES.get(key)
    return bool(feature and feature.on)


def status() -> dict[str, dict]:
    """Every feature as plain data, for templates."""
    return {key: {"on": f.on, "name": f.name, "label": f.label, "coming": f.coming}
            for key, f in FEATURES.items()}
