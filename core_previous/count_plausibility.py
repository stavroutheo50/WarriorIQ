"""Strike counts that cannot be true are not shown.

Nothing here says a count is right. It catches counts that are impossible for
a real fight - the kind a tracking or detection failure produces - so they are
withheld with a reason instead of published as a fighter's statistics.

The limits come from the official per-round statistics of 40,773 UFC
fighter-rounds (ufcstats.com, via github.com/Greco1899/scrape_ufc_stats,
fetched 2026-10-04 by tools/fetch_public_datasets.py), per fighter per round:

    strikes thrown per 5-minute round    1st pct 6     median 42    99th pct 120
    share of thrown strikes that landed  1st pct 14%   median 53%   99th pct 93%

Each limit sits well outside that range, so a real but unusually busy or
accurate fighter is never withheld: 36 a minute is 1.5 times the 99th
percentile's 24, and the landed shares allow for the smaller samples a short
clip gives. Kickboxing and boxing rounds are no busier per minute than the
busiest MMA rounds.
"""

from __future__ import annotations

MAX_ATTEMPTS_PER_MINUTE = 36.0
# The rate needs time to mean anything: a ten-second clip of a flurry is busy.
MIN_SECONDS_FOR_RATE = 60.0
# Landed share is only judged on enough strikes for a share to be a share.
MIN_ATTEMPTS_FOR_SHARE = 20
MIN_LANDED_SHARE = 0.05
MAX_LANDED_SHARE = 0.97


def check(fighters: dict, analysed_seconds: float | None) -> dict:
    """{"implausible": bool, "reasons": [...]} for one fight's per-fighter counts.

    ``fighters`` is the statistics block's per-fighter dict ("A"/"B" ->
    attempts, landed). A landed of None (outcomes withheld) is not judged.
    """
    reasons = []
    minutes = (float(analysed_seconds) / 60.0) if analysed_seconds else 0.0
    for name in ("A", "B"):
        item = fighters.get(name) or {}
        attempts = int(item.get("attempts") or 0)
        if minutes * 60.0 >= MIN_SECONDS_FOR_RATE and attempts / minutes > MAX_ATTEMPTS_PER_MINUTE:
            reasons.append(f"Fighter {name}: {attempts / minutes:.0f} strikes a minute, more than any "
                           f"real fight shows (limit {MAX_ATTEMPTS_PER_MINUTE:.0f})")
        landed = item.get("landed")
        if landed is not None and attempts >= MIN_ATTEMPTS_FOR_SHARE:
            share = float(landed) / attempts
            if not MIN_LANDED_SHARE <= share <= MAX_LANDED_SHARE:
                reasons.append(f"Fighter {name}: {share:.0%} of {attempts} strikes landed, outside what "
                               f"real fights show ({MIN_LANDED_SHARE:.0%}-{MAX_LANDED_SHARE:.0%})")
    return {"implausible": bool(reasons), "reasons": reasons}


def counts_implausible(report: dict | None) -> bool:
    """Did this report's statistics fail the check? False when it was never run."""
    statistics = (report or {}).get("statistics") or {}
    return bool((statistics.get("plausibility") or {}).get("implausible"))
