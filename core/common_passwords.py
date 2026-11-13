"""Passwords too easy to guess for a new account (owner, 2026-11-13).

The length rule alone let `password123456` and `qwertyuiop` through. Attackers
try the most used passwords first, so these are refused when a password is
chosen (sign-up, reset, change). Never applied at sign-in: an existing
password keeps working.

Two lists, kept short on purpose and served to the sign-up page as well, so
the page and the server refuse the same things:

  * BASES: words people pad with digits or symbols. A password whose letters,
    once leading and trailing digits and symbols are stripped, are only one
    of these is refused: "Password2024!", "qwerty1234567", "!!dragon!!2026".
  * FULL: whole passwords of 10+ characters that are not a word plus padding,
    mostly keyboard walks.

A password made only of digits and symbols, or of one or two characters
repeated, is refused too.
"""

from __future__ import annotations

import re

BASES = frozenset("""
password passw0rd passwort motdepasse contrasena senha parola
qwerty qwertz azerty qwertyui qwertyuiop asdf asdfgh asdfghjk asdfghjkl zxcvbn zxcvbnm qazwsx qwer
abc abcd abcde abcdef abcdefg abcdefgh abcdefghi abcdefghij abcdefghijk
letmein welcome welcomeback iloveyou loveyou lovely love admin administrator root login user guest
test testing changeme default secret hello hellohello whatever nothing trustno freedom
monkey dragon shadow master superman batman spiderman ironman starwars pokemon minecraft fortnite
sunshine princess flower butterfly angel baby babygirl cookie chocolate summer winter
football baseball basketball soccer hockey tennis golf boxing boxer fighter fight champion
warrior warriors warrioriq mma ufc kickboxing muaythai jiujitsu karate judo wrestling
michael jordan jennifer daniel thomas charlie jessica ashley
liverpool arsenal chelsea manchester barcelona realmadrid juventus
computer internet samsung apple google iphone android facebook instagram
matrix ninja killer hunter tiger lion eagle diamond silver golden
""".split())

FULL = frozenset("""
1q2w3e4r5t 1q2w3e4r5t6y 1qaz2wsx3edc 1qaz2wsx q1w2e3r4t5 q1w2e3r4t5y6 zaq12wsx zaq1zaq1
1qazxsw2 qweasdzxc qweasdzxc123 1234qwer 1234qwerty 123qweasd qwe123qwe
""".split())

_PADDING = re.compile(r"^[\W\d_]+|[\W\d_]+$")


def is_common(password: str) -> bool:
    lower = (password or "").lower()
    if lower in FULL or len(set(lower)) <= 2:
        return True
    core = _PADDING.sub("", lower)
    return not core or core in BASES


def page_rules() -> dict:
    """The same lists for the sign-up page's live hint."""
    return {"bases": sorted(BASES), "full": sorted(FULL)}
