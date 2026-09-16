"""A failed analysis has to leave something a person can look at.

The worker already caught failures, released the visitor's allowance and
marked the job `error`, so one bad upload never stalled the queue. What it did
not do was keep the reason anywhere but the log - which rotates, and which
needs shell access to read on a host that does not grant it.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import core.db as database


class DeadLetterQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = database.DB_PATH
        self.addCleanup(lambda: setattr(database, "DB_PATH", previous))
        database.DB_PATH = Path(self.temp.name) / "dlq.sqlite3"
        database.init_db()

    def test_a_failure_is_recorded_with_its_reason(self):
        database.record_analysis_failure(
            "job-1", "ValueError", detail="could not decode the video track", account_id=7)
        rows = database.list_analysis_failures()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["job_id"], "job-1")
        self.assertEqual(rows[0]["reason"], "ValueError")
        self.assertIn("decode", rows[0]["detail"])
        self.assertEqual(rows[0]["account_id"], 7)

    def test_a_retry_counts_rather_than_duplicating(self):
        """A video that fails every attempt is one problem, not five."""
        for _ in range(4):
            database.record_analysis_failure("job-1", "ValueError", detail="same fault")
        rows = database.list_analysis_failures()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["attempts"], 4)

    def test_different_jobs_are_different_entries(self):
        database.record_analysis_failure("job-1", "ValueError")
        database.record_analysis_failure("job-2", "MemoryError", detail="CUDA out of memory")
        self.assertEqual({r["job_id"] for r in database.list_analysis_failures()}, {"job-1", "job-2"})

    def test_reviewing_takes_it_off_the_queue_without_deleting_it(self):
        database.record_analysis_failure("job-1", "ValueError")
        failure_id = database.list_analysis_failures()[0]["id"]
        self.assertTrue(database.mark_analysis_failure_reviewed(failure_id))
        self.assertEqual(database.list_analysis_failures(), [])
        kept = database.list_analysis_failures(include_reviewed=True)
        self.assertEqual(len(kept), 1)
        self.assertIsNotNone(kept[0]["reviewed_at"])

    def test_reviewing_twice_is_not_an_error_and_changes_nothing(self):
        database.record_analysis_failure("job-1", "ValueError")
        failure_id = database.list_analysis_failures()[0]["id"]
        self.assertTrue(database.mark_analysis_failure_reviewed(failure_id))
        self.assertFalse(database.mark_analysis_failure_reviewed(failure_id))

    def test_a_reviewed_job_that_fails_again_opens_a_new_entry(self):
        """Closed means dealt with. A recurrence is news, not a tally."""
        database.record_analysis_failure("job-1", "ValueError")
        database.mark_analysis_failure_reviewed(database.list_analysis_failures()[0]["id"])
        database.record_analysis_failure("job-1", "ValueError")
        open_rows = database.list_analysis_failures()
        self.assertEqual(len(open_rows), 1)
        self.assertEqual(open_rows[0]["attempts"], 1)
        self.assertEqual(len(database.list_analysis_failures(include_reviewed=True)), 2)

    def test_long_reasons_are_truncated_rather_than_rejected(self):
        """A traceback must not be able to fail the thing recording it."""
        database.record_analysis_failure("job-1", "X" * 400, detail="Y" * 4000)
        row = database.list_analysis_failures()[0]
        self.assertLessEqual(len(row["reason"]), 120)
        self.assertLessEqual(len(row["detail"]), 500)


class MalformedVideoTests(unittest.TestCase):
    """The case the queue exists for."""

    def test_a_file_that_is_not_a_video_cannot_be_opened(self):
        import cv2

        broken = Path(tempfile.mkdtemp()) / "corrupt.mp4"
        # A valid ftyp box and nothing else: the header says mp4, there is no
        # moov atom, and every decoder refuses it. This is what a truncated
        # phone upload looks like.
        broken.write_bytes(b"\x00\x00\x00\x20ftypmp42" + os.urandom(4096))
        capture = cv2.VideoCapture(str(broken))
        opened = capture.isOpened()
        read_ok, _ = capture.read()
        capture.release()
        self.assertFalse(read_ok, "a headerless mp4 should not yield a frame")
        self.assertFalse(opened and read_ok)


if __name__ == "__main__":
    unittest.main()
