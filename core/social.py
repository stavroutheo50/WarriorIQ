"""Who can see an athlete's profile, and who can follow it.

Decided with the owner (2026-10-05):

  * every profile is **private by default**; the athlete chooses public;
  * an account under the minimum age (it has a guardian approval status) is
    **always private**, and in this first version is not visible to, or
    followable by, anyone but its owner - follow requests from strangers to
    children need guardian controls that do not exist yet;
  * opponents are never shown: a profile carries only the athlete's own
    numbers (Fight Camp level, streak, analysed-fight count), and shared fights,
    when they arrive, are the stats-only share card with no names.

Pure rules, so they are tested without a database.
"""

from __future__ import annotations

import re

HANDLE_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_.]{1,28})[a-z0-9]$")
# Paths and words a handle must never be, so /athlete/<handle> can never look
# like an official page.
RESERVED_HANDLES = frozenset({
    "admin", "administrator", "warrioriq", "warrior_iq", "support", "help", "staff", "official",
    "moderator", "settings", "profile", "login", "signup", "about", "api", "root", "system",
})
VISIBILITIES = ("private", "public")


def normalize_handle(raw: str | None) -> str | None:
    """A lowercase handle, or None for "no handle". Raises ValueError with a reason."""
    handle = (raw or "").strip().lstrip("@").lower()
    if not handle:
        return None
    if not HANDLE_PATTERN.match(handle):
        raise ValueError("A username is 3-30 characters: letters, numbers, dots and underscores, "
                         "starting and ending with a letter or number.")
    if handle in RESERVED_HANDLES:
        raise ValueError("That username is reserved. Choose another.")
    return handle


def is_minor_account(account: dict | None) -> bool:
    """An under-age account is the one that needed a guardian's approval."""
    if not account:
        return False
    return str(account.get("guardian_approval_status") or "not_applicable") != "not_applicable"


def effective_visibility(profile: dict, minor: bool) -> str:
    if minor:
        return "private"
    return "public" if profile.get("visibility") == "public" else "private"


def can_view(*, viewer_profile_id: int | None, profile: dict, minor: bool, follow: str | None) -> bool:
    """May this viewer see the profile's details (level, streak, fights)?"""
    if viewer_profile_id is not None and int(viewer_profile_id) == int(profile["id"]):
        return True
    if minor:
        return False
    return effective_visibility(profile, minor) == "public" or follow == "accepted"


def is_discoverable(*, viewer_profile_id: int | None, profile: dict, minor: bool) -> bool:
    """Does the profile page exist at all for this viewer? Not for a minor's, to strangers."""
    if not profile.get("handle"):
        return False
    if viewer_profile_id is not None and int(viewer_profile_id) == int(profile["id"]):
        return True
    return not minor


def can_follow(*, viewer_profile_id: int | None, profile: dict, minor: bool) -> bool:
    return (viewer_profile_id is not None and int(viewer_profile_id) != int(profile["id"])
            and not minor and bool(profile.get("handle")))
