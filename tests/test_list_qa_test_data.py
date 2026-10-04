"""tools/list_qa_test_data.py lists, filters and never writes (item 23).

Exercised only against a throwaway database made here; the tool itself is for
the operator to run against the live database when they choose.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE accounts (id INTEGER PRIMARY KEY, email TEXT, profile_id INTEGER);
        CREATE TABLE fighters (id INTEGER PRIMARY KEY, profile_id INTEGER, name TEXT, archived INTEGER, created_at TEXT);
        CREATE TABLE fights (id INTEGER PRIMARY KEY, job_id TEXT, profile_id INTEGER, original_name TEXT,
            video_path TEXT, report_path TEXT, ruleset TEXT, created_at TEXT, summary_json TEXT, fighter_id INTEGER);
        INSERT INTO accounts VALUES (1, 'qa@example.com', 10), (2, 'real@example.com', 20);
        INSERT INTO fighters VALUES (1, 10, 'QA Tester', 0, '2026-10-04T10:00:00'), (2, 20, 'Real Fighter', 0, '2026-09-01T10:00:00');
        INSERT INTO fights VALUES (1, 'aaa111', 10, 'qa-clip.mp4', '', '/r/a.json', 'K1', '2026-10-04T11:00:00', '{}', 1),
                                  (2, 'bbb222', 20, 'bout.mp4', '', '/r/b.json', 'K1', '2026-09-02T11:00:00', '{}', 2);
    """)
    con.commit()
    con.close()


def _run(db: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "tools" / "list_qa_test_data.py"), "--db", str(db), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_it_lists_only_what_the_filters_select_and_changes_nothing(tmp_path):
    db = tmp_path / "copy.sqlite3"
    _db(db)
    before = db.read_bytes()
    out = _run(db, "--email", "qa@example.com", "--csv", str(tmp_path / "review.csv"))
    assert out.returncode == 0, out.stderr
    assert "aaa111" in out.stdout and "QA Tester" in out.stdout
    assert "bbb222" not in out.stdout and "Real Fighter" not in out.stdout
    assert db.read_bytes() == before
    assert "aaa111" in (tmp_path / "review.csv").read_text(encoding="utf-8")


def test_it_refuses_to_list_everything_by_accident(tmp_path):
    db = tmp_path / "copy.sqlite3"
    _db(db)
    assert _run(db).returncode != 0


def test_it_opens_the_database_read_only():
    source = (ROOT / "tools" / "list_qa_test_data.py").read_text(encoding="utf-8")
    assert "?mode=ro" in source
    import re

    sql = " ".join(re.findall(r'connection\.execute\(\s*((?:"[^"]*"\s*)+)', source)).upper()
    assert "SELECT" in sql
    for statement in ("DELETE ", "UPDATE ", "INSERT ", "DROP ", "ALTER "):
        assert statement not in sql
