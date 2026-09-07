"""Cross-site request forgery tokens.

What this adds, and what was already here. The session cookie is
`SameSite=Lax`, every mutation is a POST, and there is no state-changing GET -
checked route by route. That combination already refuses the classic attack,
because a browser will not attach a Lax cookie to a cross-site POST. So this is
not plugging an open door; it is the second lock the launch checklist asks for,
and it closes two things Lax does not:

  * **Login CSRF.** Lax stops the attacker borrowing *your* session. It does
    not stop them giving you *theirs*: a cross-site POST to /login needs no
    cookie sent, only one set, and the victim then uploads their footage into
    the attacker's library believing they are signed in as themselves. On a
    product whose whole content is private video of identifiable people, that
    is the one worth closing.
  * **Any browser that does not default to Lax**, and any future route added
    without noticing that the protection was ambient rather than explicit.

The scheme is double submit: a random token in a cookie, and the same value
returned in a form field or the `X-CSRF-Token` header. An attacker on another
origin can cause the request but cannot read the cookie to echo it back.

The cookie stays `httponly`. The usual double-submit recipe makes it readable
so scripts can copy it, which hands any injected script the token as well;
instead the value is rendered into the page (a hidden input, and a meta tag for
fetch) from server state, and the cookie is never exposed to JavaScript.

**The limit, stated plainly:** double submit trusts that nobody else can set a
cookie on this domain. Anyone who can - a takeover of a sibling subdomain, or a
network attacker on plain HTTP - can plant a token and echo it. The first is
why warrioriq.eu should not gain untrusted subdomains; the second is what HSTS
is for, and it is already sent. Binding the token to the session id instead
would remove that assumption, and is the upgrade to make if subdomains ever
become untrusted.
"""

from __future__ import annotations

import hmac
import secrets

# 32 bytes, so guessing is not a strategy. urlsafe_b64 of 32 bytes is 43
# characters; the bounds below are a shape check, not a security boundary.
TOKEN_BYTES = 32
MIN_TOKEN_LENGTH = 32
MAX_TOKEN_LENGTH = 128


def issue_token() -> str:
    """A fresh token for a visitor who does not have one."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def usable_token(value: str | None) -> str | None:
    """The cookie's token if it is the right shape, else None.

    A cookie of the wrong shape is treated as absent and replaced rather than
    rejected, so a visitor carrying a stale or truncated value gets a working
    one on their next page load instead of a wall of 403s they cannot clear.
    """
    if not value:
        return None
    value = value.strip()
    if not MIN_TOKEN_LENGTH <= len(value) <= MAX_TOKEN_LENGTH:
        return None
    # urlsafe base64 only. Anything else did not come from issue_token.
    if not all(c.isalnum() or c in "-_=" for c in value):
        return None
    return value


def tokens_match(expected: str | None, submitted: str | None) -> bool:
    """Constant-time comparison that treats either side missing as a failure.

    `hmac.compare_digest` raises on None, and an empty submitted token must
    never compare equal to an empty cookie - which is exactly the state a
    visitor with no cookie at all would be in.
    """
    if not expected or not submitted:
        return False
    return hmac.compare_digest(str(expected), str(submitted))
