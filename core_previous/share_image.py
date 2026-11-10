"""The picture a shared fight link shows in WhatsApp, Facebook and the rest.

A link preview needs an image at an address, so this one is drawn on the
server from core.report.share_card - the same numbers as the story card and
the result page - rather than taken from the browser. The owner chooses what
to post; they do not get to choose what the numbers say.

Drawn with OpenCV, which the web host already has. Its built-in fonts are
ASCII only, which is one reason the owner's name is left to the page (where
any alphabet works) and never drawn here.
"""

from __future__ import annotations

import cv2
import numpy as np

W, H = 1200, 630
# BGR, matching static/share_card.js.
BACKGROUND = (22, 10, 7)
INK = (255, 247, 244)
MUTED = (194, 168, 154)
CYAN = (255, 215, 79)
TRACK = (83, 54, 36)
CORNER_COLOURS = {"red": (143, 113, 255), "blue": (255, 160, 90)}
FAMILY_LABELS = {"punch": "Punches", "kick": "Kicks", "knee": "Knees"}
FONT = cv2.FONT_HERSHEY_DUPLEX


def _glow(canvas: np.ndarray, centre: tuple[int, int], radius: float, colour: tuple[int, int, int], strength: float) -> None:
    ys, xs = np.mgrid[0:H, 0:W]
    distance = np.sqrt((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) / radius
    weight = (np.clip(1.0 - distance, 0.0, 1.0) ** 2 * strength)[..., None]
    canvas[:] = (canvas * (1 - weight) + np.array(colour, dtype=np.float32) * weight).astype(np.uint8)


_background: np.ndarray | None = None


def _backdrop() -> np.ndarray:
    """The glows are the same on every card, and most of the drawing time: made once."""
    global _background
    if _background is None:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = BACKGROUND
        _glow(canvas, (120, 90), 620, CYAN, 0.22)
        _glow(canvas, (W - 60, H - 40), 560, CORNER_COLOURS["red"], 0.16)
        _background = canvas
    return _background.copy()


def _text(canvas, text, x, y, scale, colour, thickness=2, right=False):
    width = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    cv2.putText(canvas, text, (x - width if right else x, y), FONT, scale, colour, thickness, cv2.LINE_AA)
    return width


def _ascii(text: str) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 else "-" if ch in "–—" else "" for ch in str(text))


def preview_png(card: dict, side: str, corner: str | None) -> bytes:
    """1200x630 PNG for one fighter of one fight's share card.

    `side` is the fighter the owner was ("A" or "B") and `corner` their corner
    ("red", "blue" or None when nobody said).
    """
    other = "B" if side == "A" else "A"
    me = card["fighters"][side]
    canvas = _backdrop()

    x = 64
    # One word, as the site header spells it: a trailing space after WARRIOR
    # left a gap that read as two words (QA, 2026-10-07).
    width = _text(canvas, "WARRIOR", x, 92, 1.5, INK, 3)
    _text(canvas, "IQ", x + width, 92, 1.5, CYAN, 3)
    _text(canvas, _ascii(f"{card.get('sport') or 'Fight'} - fight analysis").upper(), x, 138, 0.8, MUTED, 2)

    label = f"{corner.upper()} CORNER" if corner in CORNER_COLOURS else f"FIGHTER {side}"
    colour = CORNER_COLOURS.get(corner or "", INK)
    size = cv2.getTextSize(label, FONT, 0.8, 2)[0]
    right = W - 64
    cv2.rectangle(canvas, (right - size[0] - 36, 58), (right, 108), colour, 2, cv2.LINE_AA)
    _text(canvas, label, right - 18, 93, 0.8, colour, 2, right=True)

    left, bar_right, y = 560, W - 64, 230
    if me.get("strikes") is not None:
        # The total, big, then what it is made of.
        _text(canvas, str(int(me["total"])), x - 6, 380, 6.0, INK, 14)
        _text(canvas, "strikes thrown", x, 440, 1.0, MUTED, 2)
        families = [family for family in ("punch", "kick", "knee") if family in me["strikes"]]
        rows = [(FAMILY_LABELS[family], int(me["strikes"][family]), str(int(me["strikes"][family])))
                for family in families]
        top = max([1] + [value for _, value, _ in rows])
    else:
        # Strike counts switched off: movement only, each on a 0-100 scale.
        rows = [(_ascii(row["label"]), int(row["value"]), f"{int(row['value'])}{_ascii(row.get('unit') or '')}")
                for row in me.get("movement") or []]
        top = 100
    for label, value, shown in rows:
        _text(canvas, label, left, y, 1.0, INK, 2)
        _text(canvas, shown, bar_right, y, 1.0, INK, 2, right=True)
        cv2.rectangle(canvas, (left, y + 18), (bar_right, y + 30), TRACK, -1, cv2.LINE_AA)
        filled = left + max(12, int((bar_right - left) * value / top))
        cv2.rectangle(canvas, (left, y + 18), (filled, y + 30), CYAN, -1, cv2.LINE_AA)
        y += 84

    if card.get("score"):
        score = f"{card['score'][side]} - {card['score'][other]}"
        _text(canvas, "Estimated score", left, 505, 0.8, MUTED, 2)
        _text(canvas, score, bar_right, 510, 1.6, INK, 3, right=True)

    _text(canvas, "warrioriq.eu", x, 580, 1.1, INK, 2)
    _text(canvas, "Automatic estimate, not checked by a person" if me.get("strikes") is not None
          else "Measured automatically from movement", W - 64, 578, 0.6, MUTED, 1, right=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("could not encode the preview image")
    return encoded.tobytes()


# The site's own link preview (base.html og:image). QA, 2026-10-07: links to
# warrioriq.eu showed the bare square logo with twitter:card "summary", so
# WhatsApp, X and LinkedIn drew a small thumbnail. This is a 1200x630 card in
# the fight card's look. Its words describe what every report contains today
# (core/features.py: strike counts and scores are not switched on), so it
# makes no claim a report cannot back.
SITE_CARD_LINES = (
    "Combat-sports fight video analysis",
    "Movement, guard, balance and pressure,",
    "measured from your own fight video.",
)


def site_card_png(logo: np.ndarray | None = None) -> bytes:
    """1200x630 PNG for links to the site itself."""
    canvas = _backdrop()
    x = 96
    if logo is not None:
        size = 260
        mark = cv2.resize(logo, (size, size), interpolation=cv2.INTER_AREA)
        top, left = (H - size) // 2, W - size - 110
        if mark.shape[2] == 4:
            alpha = mark[..., 3:4].astype(np.float32) / 255.0
            region = canvas[top:top + size, left:left + size].astype(np.float32)
            canvas[top:top + size, left:left + size] = (
                region * (1 - alpha) + mark[..., :3].astype(np.float32) * alpha).astype(np.uint8)
        else:
            canvas[top:top + size, left:left + size] = mark[..., :3]
    width = _text(canvas, "WARRIOR", x, 250, 2.6, INK, 5)
    _text(canvas, "IQ", x + width, 250, 2.6, CYAN, 5)
    _text(canvas, SITE_CARD_LINES[0].upper(), x, 318, 0.9, MUTED, 2)
    _text(canvas, SITE_CARD_LINES[1], x, 410, 1.0, INK, 2)
    _text(canvas, SITE_CARD_LINES[2], x, 456, 1.0, INK, 2)
    _text(canvas, "warrioriq.eu", x, 548, 0.9, CYAN, 2)
    ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("could not encode the site card")
    return encoded.tobytes()
