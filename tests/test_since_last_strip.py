"""Compare to last video, near the top of the report - arrows, never a verdict."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROGRESS = {"available": True, "headline": "Better than your last fight.",
            "changes": [{"key": "pressure", "label": "Pressure", "direction": "up", "better": True, "now": 0.4, "before": 0.2},
                        {"key": "centre", "label": "Centre", "direction": "down", "better": False, "now": 0.3, "before": 0.5},
                        {"key": "footwork", "label": "Footwork", "direction": "steady", "better": True, "now": 1.0, "before": 1.0}],
            "previous_name": "IMG_1.mov", "previous_date": "2026-10-01", "previous_job_id": "prevjob123"}


class SinceLastStripTests(unittest.TestCase):
    def _page(self, progress, trusted=True):
        from tests_support import render_result

        report = json.loads((ROOT / "tests" / "fixtures" / "report_sample.json").read_text(encoding="utf-8"))
        report.setdefault("integrity", {})["identity_evidence_trusted"] = trusted
        return render_result(report=report, numbers={"state": "ok"}, job_id="thisjob",
                             progress_since_last=progress)

    def test_arrows_and_both_links(self):
        page = self._page(PROGRESS)
        strip = re.search(r'<nav class="since-last".*?</nav>', page, re.S).group(0)
        self.assertIn("Pressure</b> <span aria-hidden=\"true\">▲</span> higher", strip)
        self.assertIn("Centre</b> <span aria-hidden=\"true\">▼</span> lower", strip)
        self.assertIn("steady", strip)
        self.assertIn('href="#report-progress"', strip)
        self.assertIn('href="/compare?a=prevjob123&amp;b=thisjob"', strip)
        # Pressure, centre and movement have no better direction.
        for verdict in ("Better", "better", "worse", "Down on"):
            self.assertNotIn(verdict, strip)

    def test_hidden_without_a_comparison_or_when_identity_failed(self):
        self.assertNotIn('class="since-last"', self._page({"available": False}))
        self.assertNotIn('class="since-last"', self._page(PROGRESS, trusted=False))

    def test_comparison_names_the_previous_fight(self):
        source = (ROOT / "core" / "squad.py").read_text(encoding="utf-8")
        self.assertIn('"previous_job_id": previous.get("job_id")', source)


if __name__ == "__main__":
    unittest.main()
