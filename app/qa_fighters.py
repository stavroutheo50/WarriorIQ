"""The QA pass's test fighters, and taking them off a roster from /admin (2026-11-22).

Audit 2026-10-14 (#32) left test fighters in the owner's roster.
tools/delete_qa_test_data.py removes or archives them, but the host has no
shell to run it, so the admin page offers the reversible half: archiving. The
fighters leave the roster and keep their fights; adding a fighter with the same
name brings one back (core.db.create_fighter un-archives). Deleting stays a deliberate operator step in the tool.

Matching is the tool's: a fighter whose name is EXACTLY one of
QA_FIGHTER_NAMES, compared case-insensitively after collapsing whitespace.
Never a substring, so a real athlete called "Qasim" is never caught. Only the
signed-in administrator's own roster is touched.
"""

from __future__ import annotations

from core import db

# The names the QA pass created (audit 2026-10-14, #32), exactly as entered.
QA_FIGHTER_NAMES = (
    "QA Test Boxer",
    "QA Test MMA",
    "QA Test MT",
    "QA Test TKD",
    "QA <img src=x onerror=alert(1)> \"O'Neil\" & Ωmega",
)


def key(name: str) -> str:
    return " ".join(str(name or "").split()).casefold()


def on_roster(profile_id: int) -> list[dict]:
    """QA fighters still on this roster (not archived), each with how many fights are filed under them."""
    wanted = {key(name) for name in QA_FIGHTER_NAMES}
    found = [dict(f) for f in db.list_fighters(profile_id) if key(f["name"]) in wanted and not f.get("archived")]
    if found:
        counts: dict[int, int] = {}
        for fight in db.list_fights(profile_id):
            if fight.get("fighter_id") is not None:
                counts[int(fight["fighter_id"])] = counts.get(int(fight["fighter_id"]), 0) + 1
        for fighter in found:
            fighter["fights"] = counts.get(int(fighter["id"]), 0)
    return found


def archive_all(profile_id: int, account_id: int) -> int:
    """Archive every QA fighter on this roster. Their fights stay. Returns how many were archived."""
    archived = 0
    for fighter in on_roster(profile_id):
        if db.archive_fighter(profile_id, int(fighter["id"])):
            archived += 1
            db.record_security_event("qa_test_data_archived", account_id=account_id, resource_type="fighter",
                                     resource_id=str(fighter["id"]), metadata={"tool": "admin_page"})
    return archived
