"""Which proposals are worth labelling, and getting a report out of the app.

Two things were blocking the one decision anybody is waiting on - whether
punch counting can be switched back on - and neither was a missing
measurement.

The first was volume: 131 unlabelled proposals is hours of watching video,
and most of that work moves no decision. The second was that a report could
not leave the application, so an analysis that came out wrong had to be
described rather than sent.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from browser_client import BrowserClient as TestClient

from app.main import app
from core.report import ARRIVED_OUTCOMES
from tools.label_queue import DECIDES, build

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.queue = build()
        if not self.queue:
            self.skipTest("no label packs in this checkout")

    def test_it_asks_only_for_answers_that_decide_something(self):
        """The report publishes only strikes whose limb arrived, so a missed
        proposal changes the detector's overall precision and changes nothing
        a reader sees. Labelling it cannot settle the open question."""
        for pack, rows in self.queue.items():
            for row in rows:
                with self.subTest(pack=pack, id=row["id"]):
                    self.assertEqual((row["family"], "arrived" if row["arrived"] else "missed"),
                                     DECIDES)

    def test_it_is_much_shorter_than_labelling_everything(self):
        deciding = sum(len(rows) for rows in self.queue.values())
        everything = sum(len(rows) for rows in build(everything=True).values())
        self.assertLess(deciding, everything / 2,
                        "a queue that is most of the work is not a queue")

    def test_each_pack_is_in_time_order(self):
        """Labelling in time order means watching the round once instead of
        scrubbing back and forth."""
        for pack, rows in self.queue.items():
            with self.subTest(pack=pack):
                self.assertEqual([r["at"] for r in rows],
                                 sorted(r["at"] for r in rows))

    def test_nothing_already_answered_is_asked_again(self):
        from tools.evaluate_strike_counts import CONFIRMED, REJECTED, _pack_path, discover

        answered = {}
        for labels_path in discover():
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            pack = _pack_path(str(data.get("pack") or "")).name
            answered[pack] = {
                int(l["id"]) for l in data.get("labels", []) if "id" in l
                and str(l.get("verdict") or "").lower() in {CONFIRMED, REJECTED}
            }
        for pack, rows in self.queue.items():
            for row in rows:
                with self.subTest(pack=pack, id=row["id"]):
                    self.assertNotIn(row["id"], answered.get(pack, set()))

    def test_the_written_queue_is_readable_by_the_label_server(self):
        for path in (PROJECT_ROOT / "labelpack").glob("*/queue.json"):
            with self.subTest(pack=path.parent.name):
                written = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(written["schema"], "warrioriq.label_queue.v1")
                self.assertTrue(written["ids"])
                self.assertTrue(written["why"])

    def test_the_server_says_how_many_decide(self):
        source = (PROJECT_ROOT / "tools" / "serve_label_pack.py").read_text(encoding="utf-8")
        self.assertIn("queue.json", source)
        self.assertIn("decide", source)


class ReportExportTests(unittest.TestCase):
    """The report is the evidence. It has to be able to leave."""

    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_a_stranger_gets_nothing(self):
        response = self.client.get("/result/athens_hd/report.json")
        self.assertIn(response.status_code, {403, 404},
                      "somebody else's measured footage must not be downloadable")

    def test_it_does_not_advertise_itself_for_a_job_that_does_not_exist(self):
        self.assertEqual(
            self.client.get("/result/doesnotexist/report.json").status_code, 404)

    def test_it_is_owner_gated_by_the_same_check_as_the_report_page(self):
        source = (PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8")
        start = source.index("def download_report_json")
        # To the next route, rather than a character count the docstring can
        # outgrow - which it did.
        body = source[start:source.index("@app.", start)]
        self.assertIn("_authorized_job(request, job_id)", body)
        self.assertIn("no-store", body)
        self.assertIn("attachment", body)

    def test_the_route_is_private(self):
        """/result/ is already in the private prefixes, so this inherits
        noindex and no-store rather than needing its own rule."""
        from app.main import PRIVATE_ROUTE_PREFIXES

        self.assertTrue("/result/".startswith(PRIVATE_ROUTE_PREFIXES)
                        or "/result/" in PRIVATE_ROUTE_PREFIXES)


class ArrivedIsTheDecidingSetTests(unittest.TestCase):
    def test_the_queue_uses_the_reports_own_definition_of_arrived(self):
        """If core.report widens what it publishes, the queue has to follow,
        or it will be asking about clips that no longer decide anything."""
        import tools.label_queue as queue_module

        self.assertIs(queue_module.ARRIVED_OUTCOMES, ARRIVED_OUTCOMES)


class SavingAnswersTests(unittest.TestCase):
    """Answers on disk are the only copy: labelpack/ is gitignored.

    The save path replaced the file with whatever the page was holding, and
    the page restored its state from localStorage alone. A browser that had
    never opened that pack started empty, so the first keystroke wrote a file
    containing one answer. Reproduced against the real cropmotion pack, which
    held 91: one POST left one.
    """

    def test_a_page_answering_a_subset_keeps_the_rest(self):
        from tools.serve_label_pack import merge_answers

        existing = {"labels": [{"id": 1, "technique": "jab"},
                               {"id": 2, "technique": "none"}],
                    "unsure": [3], "wrong_person": [{"id": 4}]}
        incoming = {"scope": [2], "labels": [{"id": 2, "technique": "cross"}],
                    "unsure": [], "wrong_person": []}
        merged = merge_answers(existing, incoming)
        self.assertEqual(sorted(l["id"] for l in merged["labels"]), [1, 2])
        self.assertEqual([l["technique"] for l in merged["labels"] if l["id"] == 2],
                         ["cross"], "the answer inside the scope must be the new one")
        self.assertEqual([l["technique"] for l in merged["labels"] if l["id"] == 1],
                         ["jab"], "an answer outside the scope must survive untouched")
        self.assertEqual(merged["unsure"], [3])
        self.assertEqual(merged["wrong_person"], [{"id": 4}])

    def test_an_answer_can_still_change_family_inside_its_scope(self):
        """Merging must not mean "can never be corrected"."""
        from tools.serve_label_pack import merge_answers

        merged = merge_answers(
            {"labels": [{"id": 7, "technique": "jab"}], "unsure": [], "wrong_person": []},
            {"scope": [7], "labels": [], "unsure": [7], "wrong_person": []})
        self.assertEqual(merged["labels"], [])
        self.assertEqual(merged["unsure"], [7])

    def test_a_page_that_does_not_declare_a_scope_still_writes_whole(self):
        """Older pages always covered the entire pack, so replacing is right
        for them. Kept so a pack built before this change keeps working."""
        from tools.serve_label_pack import merge_answers

        merged = merge_answers({"labels": [{"id": 1, "technique": "jab"}]},
                               {"labels": [{"id": 9, "technique": "cross"}]})
        self.assertEqual([l["id"] for l in merged["labels"]], [9])

    def test_the_header_of_an_existing_file_is_not_thrown_away(self):
        from tools.serve_label_pack import merge_answers

        merged = merge_answers(
            {"labelled_by": "claude-opus-5, 2026-09-07", "labels": []},
            {"scope": [], "labels": [], "unsure": [], "wrong_person": []})
        self.assertEqual(merged["labelled_by"], "claude-opus-5, 2026-09-07")

    def test_ids_compare_the_same_whether_they_arrive_as_text_or_number(self):
        """The page sends numbers; some files on disk hold strings."""
        from tools.serve_label_pack import merge_answers

        merged = merge_answers({"labels": [{"id": "5", "technique": "jab"}]},
                               {"scope": [5], "labels": [{"id": 5, "technique": "cross"}],
                                "unsure": [], "wrong_person": []})
        self.assertEqual(len(merged["labels"]), 1, "5 and \"5\" are the same clip")
        self.assertEqual(merged["labels"][0]["technique"], "cross")


if __name__ == "__main__":
    unittest.main()
