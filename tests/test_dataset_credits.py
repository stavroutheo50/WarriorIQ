"""Every public dataset the strike model trains on is credited, as its licence asks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class CreditTests(unittest.TestCase):
    def test_transparency_page_credits_the_training_data(self):
        from fastapi.testclient import TestClient

        import app.main as webapp

        page = TestClient(webapp.app).get("/ai-transparency").text
        self.assertIn("Training data", page)
        for credit in ("BoxingVI", "arXiv:2511.16524", "R. Hegde", "TKD-Kick3", "CC BY 4.0",
                       "StrikeMetrics", "MIT licence"):
            self.assertIn(credit, page)

    def test_boxingvi_licence_is_recorded_with_its_source(self):
        import fetch_public_datasets

        boxingvi = next(s for s in fetch_public_datasets.SOURCES if s.name == "boxingvi")
        self.assertEqual(boxingvi.commercial_use, "yes")
        self.assertIn("attribution", boxingvi.licence)
        source = (ROOT / "tools" / "fetch_public_datasets.py").read_text(encoding="utf-8")
        self.assertIn("Please use it with\n    # attribution. It is creative commons.", source)

    def test_pipeline_fetches_boxingvi_unless_told_not_to(self):
        import strike_pipeline

        for argv, expected in ((["--rounds", "1"], True), (["--rounds", "1", "--no-boxingvi"], False)):
            calls = []
            with patch.object(strike_pipeline, "step", lambda title, command: calls.append(command) or False), \
                    patch.object(strike_pipeline, "DATASET", Path("/nonexistent")):
                strike_pipeline.main(argv)
            fetch = calls[0]
            self.assertEqual("boxingvi" in fetch, expected, argv)


if __name__ == "__main__":
    unittest.main()
