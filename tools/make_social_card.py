"""Redraw app/static/warrioriq-social-card.png, the site's link-preview image.

Run after changing core/share_image.site_card_png or the logo:
    python tools/make_social_card.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from core.share_image import site_card_png  # noqa: E402

OUTPUT = ROOT / "app" / "static" / "warrioriq-social-card.png"


def main() -> int:
    logo = cv2.imread(str(ROOT / "app" / "static" / "warrioriq-logo-512.png"), cv2.IMREAD_UNCHANGED)
    OUTPUT.write_bytes(site_card_png(logo))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
