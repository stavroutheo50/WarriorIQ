"""tools/exam_ufc_fight.py: one UFC fight into the exam's manifest, refused early on wrong names."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import exam_ufc_fight as tool  # noqa: E402

from core.official_stats import load_official  # noqa: E402


def _official(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["EVENT", "BOUT", "ROUND", "FIGHTER", "KD", "SIG.STR.",
                                                    "TOTAL STR.", "HEAD", "BODY", "LEG", "TD"])
        writer.writeheader()
        for fighter in ("Ann Lee", "Bea Cruz"):
            writer.writerow({"EVENT": "UFC 1", "BOUT": "Ann Lee vs. Bea Cruz", "ROUND": "Round 1",
                             "FIGHTER": fighter, "KD": "0", "SIG.STR.": "5 of 9", "TOTAL STR.": "6 of 11",
                             "HEAD": "0 of 0", "BODY": "0 of 0", "LEG": "0 of 0", "TD": "0 of 0"})
    return path


class ExamUfcFightTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.rows = load_official(_official(self.root / "ufc.csv"))

    def test_find_lists_the_exact_names_to_copy(self):
        self.assertEqual(tool.find_bouts(self.rows, "cruz"),
                         [("UFC 1", "Ann Lee vs. Bea Cruz", ["Ann Lee", "Bea Cruz"])])
        self.assertEqual(tool.find_bouts(self.rows, "nobody"), [])

    def test_wrong_names_are_refused_before_any_analysis(self):
        tool.check_official(self.rows, "Ann Lee vs. Bea Cruz", ("Ann Lee", "Bea Cruz"), None)
        with self.assertRaises(SystemExit) as refused:
            tool.check_official(self.rows, "Ann Lee vs. Bea Cruz", ("Ann Lee", "Bea Crux"), None)
        self.assertIn("Bea Crux", str(refused.exception))
        self.assertIn("--find", str(refused.exception))

    def test_manifest_replaces_a_rerun_of_the_same_fight(self):
        manifest = self.root / "exam" / "ufc_bouts.json"
        tool.update_manifest(manifest, {"result": "outputs/ufcexam_x", "bout": "X", "fighter_a": "a", "fighter_b": "b"})
        tool.update_manifest(manifest, {"result": "outputs/ufcexam_y", "bout": "Y", "fighter_a": "a", "fighter_b": "b"})
        tool.update_manifest(manifest, {"result": "outputs/ufcexam_x", "bout": "X2", "fighter_a": "a", "fighter_b": "b"})
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual([e["bout"] for e in entries], ["Y", "X2"])

    def test_the_candidate_is_set_only_for_this_process(self):
        source = (ROOT / "tools" / "exam_ufc_fight.py").read_text(encoding="utf-8")
        # Set before core.analyzer (and so core.config) is imported, never written to a settings file.
        self.assertLess(source.index('os.environ["WARRIORIQ_TEMPORAL_MODEL"]'), source.index("from core import analyzer"))
        self.assertNotIn("dotenv", source)
        self.assertNotIn("'.env'", source)


if __name__ == "__main__":
    unittest.main()
