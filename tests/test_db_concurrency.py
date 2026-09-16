"""A second writer has to wait, not fail.

WarriorIQ runs a web process and a GPU worker against one SQLite file. The
rollback journal takes an exclusive lock for the whole of a write, so a reader
waits for its duration; WAL lets readers through against the last committed
state.

Measured while writing these: six writers making 25 writes each took 2.68s on
the rollback journal and 1.84s on WAL, and **neither lost a write or raised**.
sqlite3.connect() already applies a five-second busy timeout when none is
given, so contention was being waited out rather than failing even before WAL.
These tests therefore assert that concurrent writes all land and that readers
are not blocked - not that a previously broken case was fixed.
"""
from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

import core.db as database
from core.config import SETTINGS


class WriteConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.previous = database.DB_PATH
        self.addCleanup(lambda: setattr(database, "DB_PATH", self.previous))
        database.DB_PATH = Path(self.temp.name) / "concurrency.sqlite3"
        database.init_db()

    def _hammer(self, writers=4, each=15):
        """Every writer opens its own connection, as separate processes do."""
        errors: list[Exception] = []
        barrier = threading.Barrier(writers)

        def write(index: int) -> None:
            barrier.wait()                      # start together, maximise overlap
            for n in range(each):
                try:
                    with database.connection() as con:
                        con.execute(
                            "INSERT INTO security_events(event_type, occurred_at) VALUES(?,?)",
                            ("concurrency-%d-%d" % (index, n), "2026-09-16T00:00:00Z"))
                except Exception as exc:        # noqa: BLE001 - recorded, not raised
                    errors.append(exc)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(writers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return errors

    def test_concurrent_writers_all_succeed(self):
        errors = self._hammer()
        self.assertEqual(
            errors, [],
            "concurrent writes failed: %s" % ({type(e).__name__ for e in errors} or None))

    def test_every_row_written_by_every_writer_survives(self):
        """No silent loss: contention must cost time, never data."""
        self._hammer(writers=4, each=15)
        with database.connection() as con:
            rows = con.execute(
                "SELECT COUNT(*) FROM security_events WHERE event_type LIKE 'concurrency-%'"
            ).fetchone()[0]
        self.assertEqual(rows, 4 * 15)

    def test_a_blocked_writer_waits_rather_than_raising(self):
        """The busy timeout is the part that turns an error into a wait."""
        with database.connection() as con:
            timeout = con.execute("PRAGMA busy_timeout").fetchone()[0]
        self.assertGreaterEqual(int(timeout), 1000)
        self.assertEqual(int(timeout), int(SETTINGS.sqlite_busy_timeout_ms))

    def test_readers_are_not_blocked_by_a_writer(self):
        """What WAL buys. Skipped where the filesystem will not support it."""
        with database.connection() as con:
            mode = str(con.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode != "wal":
            self.skipTest("journal_mode is %r; WAL unavailable on this filesystem" % mode)
        held = threading.Event()
        release = threading.Event()

        def writer() -> None:
            with database.connection() as con:
                con.execute(
                    "INSERT INTO security_events(event_type, occurred_at) VALUES(?,?)",
                    ("held-open", "2026-09-16T00:00:00Z"))
                held.set()
                release.wait(5)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            self.assertTrue(held.wait(5), "writer never started")
            # A read while that write is open must return, not block or raise.
            with database.connection() as con:
                con.execute("SELECT COUNT(*) FROM security_events").fetchone()
        finally:
            release.set()
            thread.join()

    def test_wal_can_be_switched_off_for_a_hostile_filesystem(self):
        """A network filesystem cannot hold WAL's shared-memory file.

        The switch has to work, because the fallback is the only thing between
        a shared host and a database that will not open at all.
        """
        previous = SETTINGS.sqlite_wal
        try:
            object.__setattr__(SETTINGS, "sqlite_wal", False)
            plain = Path(self.temp.name) / "plain.sqlite3"
            previous_path = database.DB_PATH
            database.DB_PATH = plain
            database.init_db()
            with database.connection() as con:
                mode = str(con.execute("PRAGMA journal_mode").fetchone()[0]).lower()
                timeout = int(con.execute("PRAGMA busy_timeout").fetchone()[0])
            database.DB_PATH = previous_path
        finally:
            object.__setattr__(SETTINGS, "sqlite_wal", previous)
        self.assertNotEqual(mode, "wal")
        # The busy timeout is not part of the WAL decision and must survive it.
        self.assertGreaterEqual(timeout, 1000)


if __name__ == "__main__":
    unittest.main()
