"""The upload page offers short solo training clips before the upload (after mmagpt.app)."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SoloHintTests(unittest.TestCase):
    def test_upload_page_offers_solo_clips(self):
        page = (ROOT / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        self.assertIn('id="setupSolo"', page)
        self.assertIn("Solo session", page)
        # The name it gives must be the toggle the next step really shows.
        select = (ROOT / "app" / "templates" / "select.html").read_text(encoding="utf-8")
        self.assertIn("<strong>Solo session</strong>", select)

    def test_the_length_it_suggests_is_accepted(self):
        from core.preflight import MIN_USABLE_SECONDS

        self.assertLessEqual(MIN_USABLE_SECONDS, 20.0)

    def test_it_promises_only_what_a_solo_report_has(self):
        from core.solo import SOLO_METRICS

        page = (ROOT / "app" / "templates" / "analyze.html").read_text(encoding="utf-8")
        line = page[page.index('id="setupSolo"'):].split("</p>", 1)[0]
        self.assertIn("no opponent numbers or score", line)
        for measured in ("guard_index", "balance_index"):
            self.assertIn(measured, SOLO_METRICS)


if __name__ == "__main__":
    unittest.main()
