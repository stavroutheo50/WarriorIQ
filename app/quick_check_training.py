"""Quick-check answers as strike-training data, from policy version 2026-11-25.

Until then the privacy policy said quick-check answers were "not used as
training data". From version 2026-11-25 (owner, 2026-11-25: strike counting
can only improve on labelled real strikes, and these are the ones people give)
an answer may be used to train strike counting, with the body measurements
of that moment from the fight's tracking file - never its video - when all of
these hold:

- it was given on or after TRAINING_SINCE, and after the fight's owner had
  accepted that policy version: an answer given under the old promise is
  never used;
- the fight's owner is not an under-18 account (they are never asked);
- the owner has not turned it off in Settings. Turning it off also takes out
  answers already given.

The choice is recorded like every other consent choice (legal_acceptances,
newest first), so there is no new table. In app/ rather than core/: it is not
analysis, and core/*.py is the worker's analysis-code fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINING_SINCE = "2026-11-25"
KIND = "quick_check_training"
WINDOW_SECONDS = 1.0


def objected(profile_id: int) -> bool:
    """True when the account's newest choice is "do not use my answers"."""
    from core.db import list_legal_acceptances

    for item in list_legal_acceptances(profile_id=int(profile_id)):
        if item.get("kind") == KIND:
            return item.get("current_status") == "declined"
    return False


def set_choice(profile_id: int, allowed: bool, policy_version: str) -> None:
    from core.db import record_legal_acceptance

    record_legal_acceptance(KIND, policy_version, profile_id=int(profile_id),
                            metadata={"source": "settings", "allowed": bool(allowed)},
                            current_status="accepted" if allowed else "declined")


def usable(answer: dict, fight: dict | None, account: dict | None) -> bool:
    """Whether one strike-check answer may be used for training (see the module docstring)."""
    from core.db import is_strike_check
    from core.social import is_minor_account

    if not is_strike_check(answer) or not fight or not account:
        return False
    given = str(answer.get("created_at") or "")
    accepted_at = str(account.get("policies_accepted_at") or "")
    if given < TRAINING_SINCE or str(account.get("terms_version") or "") < TRAINING_SINCE:
        return False
    if not accepted_at or given < accepted_at:
        return False
    if is_minor_account(account):
        return False
    return not objected(int(fight["profile_id"]))


def training_rows(outputs: Path) -> list[dict]:
    """Every usable answer, with the tracking of both fighters around that moment."""
    from core.db import get_account_by_profile, get_fight, list_annotations

    rows, cache = [], {}
    for answer in list_annotations():
        job_id = str(answer.get("job_id") or "")
        if job_id not in cache:
            fight = get_fight(job_id)
            cache[job_id] = (fight, get_account_by_profile(int(fight["profile_id"])) if fight else None)
        fight, account = cache[job_id]
        if not usable(answer, fight, account):
            continue
        seconds = float(answer["event_time"])
        corrected = answer.get("corrected") or {}
        rows.append({"job_id": job_id, "ruleset": answer.get("ruleset"), "seconds": seconds,
                     "fighter": corrected.get("fighter"), "family": (answer.get("predicted") or {}).get("family"),
                     "verdict": corrected.get("verdict"), "window": _window(outputs / job_id / "tracking.jsonl", seconds)})
    return rows


def _window(tracking: Path, seconds: float) -> list[dict]:
    """Both fighters' boxes and keypoints within WINDOW_SECONDS of the moment: measurements, no pictures."""
    if not tracking.is_file():
        return []
    out = []
    with tracking.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            t = float(record.get("time_seconds") or 0.0)
            if t > seconds + WINDOW_SECONDS:
                break
            if t >= seconds - WINDOW_SECONDS:
                out.append({"time_seconds": t, **{
                    f"fighter_{side}": {key: ((record.get(f"fighter_{side}") or {}).get("observation") or {}).get(key)
                                        for key in ("box", "keypoints", "keypoint_conf")}
                    for side in ("A", "B")}})
    return out
