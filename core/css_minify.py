"""Shrink the stylesheet bundles at import, with no build step.

Two stylesheets reach every page and together they are 216 KB of source, on a
site whose heaviest page carries one image. They are already concatenated in
memory at import by `_build_css_bundle`, so this is the same pass doing a
little more work - nothing is generated on disk, nothing has to be run before
a deploy, and the source files stay readable.

**Deliberately timid.** A minifier that is clever about CSS it does not fully
parse breaks pages in ways nobody notices until a phone loads them. This one
does four things and refuses to do anything else:

  * drops comments;
  * collapses runs of whitespace to a single space;
  * removes whitespace around the delimiters that cannot carry meaning - the
    braces, the semicolon and the comma;
  * drops the last semicolon before a closing brace.

What it deliberately leaves alone:

  * **`:` and whitespace between selectors.** `li :first-child` and
    `li:first-child` are different selectors, and telling them apart needs a
    real parse.
  * **`+` and `-`.** `calc(100% - 20px)` requires those spaces; removing them
    is the classic way to break a layout silently.
  * **anything inside a string or a `url()`.** `content:"a, b"` is text, and
    a data URI is full of commas and braces that mean nothing to CSS.

The saving is smaller than a full minifier would manage. It is also a saving
that cannot change a single computed style, which is the trade worth making
for a file that is never reviewed after it ships.
"""

from __future__ import annotations

import re

# A quoted string, a url(...), or a comment - the three regions where the
# characters this pass rewrites are not delimiters.
_PROTECTED = re.compile(
    r"""
      "(?:[^"\\]|\\.)*"      # a double-quoted string
    | '(?:[^'\\]|\\.)*'      # a single-quoted string
    | url\([^)]*\)           # a url(), which may hold an unquoted data URI
    | /\*.*?\*/              # a comment
    """,
    re.VERBOSE | re.DOTALL,
)

_AROUND_DELIMITERS = re.compile(r"\s*([{};,])\s*")
_TRAILING_SEMICOLON = re.compile(r";}")
_WHITESPACE = re.compile(r"\s+")


def minify(css: str) -> str:
    """Return `css` with the safe whitespace removed.

    The pass runs until it stops changing anything, which takes two goes in
    practice and is the honest fix for a problem the first version papered
    over. Each chunk is squeezed on its own, so whitespace that ends up
    adjacent to a delimiter only after the chunks are joined - the space a
    removed comment leaves between `}` and the next selector - survives the
    first pass. Running the *protected-aware* pass again collects it.

    A single tidy-up sweep over the joined string was tried instead and was
    wrong: it ran across the strings and url()s that had just been put back,
    so `content:"a, b"` came out as `content:"a,b"`.
    """
    previous = css
    for _ in range(4):
        current = _one_pass(previous)
        if current == previous:
            break
        previous = current
    return previous.strip()


def _one_pass(css: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(css):
        pieces.append(_squeeze(css[cursor:match.start()]))
        token = match.group(0)
        # A comment goes entirely; a string or url() survives untouched. The
        # space is what a comment leaves behind - `a/* x */b` is two tokens,
        # so removing the comment outright would join them.
        pieces.append(" " if token.startswith("/*") else token)
        cursor = match.end()
    pieces.append(_squeeze(css[cursor:]))
    return "".join(pieces)


def _squeeze(chunk: str) -> str:
    """Every rewrite happens here, on unprotected text only.

    A final pass over the joined result was tried first and was wrong twice
    over: it ran across the protected tokens that had just been put back, so
    `content:"a, b"` came out as `content:"a,b"` and `content:";}"` lost its
    semicolon - the minifier quietly editing somebody's text.

    The joins do not need such a pass. A delimiter beside a protected token
    always sits at the edge of an unprotected chunk, where this already sees
    it.
    """
    chunk = _WHITESPACE.sub(" ", chunk)
    chunk = _AROUND_DELIMITERS.sub(r"\1", chunk)
    return _TRAILING_SEMICOLON.sub("}", chunk)
