"""Catch mistyped email providers at sign-up (owner, 2026-11-22).

A new account typed "gmaill.com": the confirmation link could never arrive, and
the account was lost. This lists misspellings of the big providers that are
near-certain mistakes - nobody's real mailbox lives at them - and suggests the
intended address. It is deliberately a short, curated list rather than fuzzy
matching: a guess that "corrects" a real but unusual domain would refuse a
genuine address, which is worse than missing a typo.
"""

from __future__ import annotations

_PROVIDERS = {
    "gmail.com": ("gmaill.com", "gmial.com", "gmai.com", "gmal.com", "gamil.com", "gnail.com", "gmail.co",
                  "gmail.con", "gmail.cm", "gmail.om", "gmail.comm", "gmaik.com", "gmsil.com", "gmali.com",
                  "gmaul.com", "gmeil.com", "gmail.c", "gmail.cmo", "gmail.vom", "gmail.xom", "gimail.com"),
    "googlemail.com": ("googlemail.con", "goglemail.com"),
    "yahoo.com": ("yaho.com", "yahooo.com", "yahoo.con", "yhoo.com", "yahho.com", "yaoo.com", "yahoo.cm"),
    "hotmail.com": ("hotmial.com", "hotmal.com", "hotmai.com", "hotmail.con", "hotmil.com", "hotmaill.com",
                    "homtail.com", "hotamil.com", "hotmail.cm", "hptmail.com"),
    "outlook.com": ("outlok.com", "outloo.com", "outlook.con", "otulook.com", "outlool.com", "outllook.com"),
    "icloud.com": ("iclod.com", "icould.com", "icloud.con", "iclould.com", "icloud.cm", "icloud.co", "iclud.com"),
    "live.com": ("live.con", "liv.com"),
    "aol.com": ("aol.con", "aoll.com"),
}
TYPO_DOMAINS: dict[str, str] = {typo: right for right, typos in _PROVIDERS.items() for typo in typos}


def suggestion(email: str) -> str | None:
    """The address the person most likely meant, or None when nothing looks mistyped."""
    local, at, domain = (email or "").strip().rpartition("@")
    if not at or not local:
        return None
    right = TYPO_DOMAINS.get(domain.lower())
    return f"{local}@{right}" if right else None
