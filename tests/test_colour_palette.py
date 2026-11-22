"""The stylesheets' colours come from one palette (2026-11-22).

They held 619 distinct hard-coded colours, most of them near-copies of the
colour tokens in product.css and glass.css (--wiq-*, --glass-*). Every colour
within delta-E 20 of a token, and of similar saturation so a tinted colour is
never turned grey, now uses the token. Screenshots of 62 pages moved by under
0.3% of pixels, mostly the footer text, which got a little lighter.

The count is RATCHETED, like the breakpoints in style.css: it may go down
freely, but a new one-off colour fails here. Use a token (var(--wiq-...)) or,
if a new colour is really needed, add a token for it and raise the number on
purpose.
"""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
MAX_DISTINCT_HEX = 259


def _distinct_hex() -> set[str]:
    seen = set()
    for path in STATIC.glob("*.css"):
        text = re.sub(r"/\*.*?\*/|url\([^)]*\)", "", path.read_text(encoding="utf-8"), flags=re.S)
        for match in re.finditer(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b", text):
            value = match.group(1).lower()
            seen.add("".join(c * 2 for c in value) if len(value) == 3 else value)
    return seen


def test_no_new_one_off_colours():
    count = len(_distinct_hex())
    assert count <= MAX_DISTINCT_HEX, (
        f"{count} distinct hard-coded colours (limit {MAX_DISTINCT_HEX}). Use a palette token instead.")


def test_the_palette_tokens_are_defined_once_on_root():
    css = "\n".join(p.read_text(encoding="utf-8") for p in STATIC.glob("*.css"))
    for token in ("--wiq-canvas", "--wiq-line", "--wiq-text", "--glass-text-soft", "--glass-text-faint",
                  "--wiq-a", "--wiq-b", "--wiq-ink-on-bright"):
        assert len(re.findall(re.escape(token) + r"\s*:", css)) == 1, token
