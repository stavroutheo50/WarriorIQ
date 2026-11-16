"""Backups and server checks without a shell (owner, 2026-11-16).

The host has no shell, so the website makes its own daily database backup and
the admin page runs the server checks and the leftover-copy clean-up.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

import app.main as web
from app import site_backup
from browser_client import BrowserClient
from core import db
from core.auth import hash_password, issue_session
from core.config import SETTINGS
from tools import backup_runtime


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "data" / "warrioriq.sqlite3"
    path.parent.mkdir()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE t (x)")
        connection.execute("INSERT INTO t VALUES (1)")
    monkeypatch.setattr(site_backup, "DB_PATH", path)
    monkeypatch.setattr(backup_runtime, "DB_PATH", path)
    monkeypatch.setenv("WARRIORIQ_AUTO_BACKUP", "1")
    monkeypatch.delenv("WARRIORIQ_BACKUP_DIR", raising=False)
    return path


def test_a_day_gets_one_checked_copy_beside_the_database(database):
    first = site_backup.make_backup(scheduled=True)
    assert first["outcome"] == "done"
    folder = database.parent / "backups"
    copy = folder / first["name"]
    assert copy.is_file() and copy.with_suffix(".json").is_file()
    with sqlite3.connect(copy) as connection:
        assert connection.execute("SELECT x FROM t").fetchall() == [(1,)]
    assert site_backup.latest()["integrity_check"] == "ok"
    # Again the same day: not due. A day later: due.
    assert site_backup.make_backup(scheduled=True)["outcome"] == "not_due"
    tomorrow = datetime.now(timezone.utc) + timedelta(hours=25)
    assert site_backup.make_backup(scheduled=True, now=tomorrow)["outcome"] in {"done"}


def test_a_manual_copy_right_after_another_is_refused(database):
    assert site_backup.make_backup(scheduled=False)["outcome"] == "done"
    assert site_backup.make_backup(scheduled=False)["outcome"] == "recent"


def test_only_the_newest_are_kept(database, monkeypatch):
    monkeypatch.setenv("WARRIORIQ_BACKUP_KEEP", "2")
    folder = database.parent / "backups"
    folder.mkdir()
    for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        (folder / f"warrioriq-{stamp}.sqlite3").write_bytes(b"old")
    (folder / "notes.sqlite3").write_bytes(b"not ours")
    assert site_backup.make_backup(scheduled=True)["outcome"] == "done"
    names = sorted(path.name for path in folder.glob("*.sqlite3"))
    assert len([name for name in names if name.startswith("warrioriq-")]) == 2
    assert "warrioriq-20260103T000000Z.sqlite3" in names and "notes.sqlite3" in names


def test_two_processes_never_copy_at_once(database):
    with site_backup._exclusive(database.parent / "backups") as held:
        assert held
        assert site_backup.make_backup(scheduled=True)["outcome"] == "busy"


def test_no_database_means_no_copy(tmp_path, monkeypatch):
    missing = tmp_path / "nowhere" / "warrioriq.sqlite3"
    monkeypatch.setattr(site_backup, "DB_PATH", missing)
    assert site_backup.make_backup(scheduled=True)["outcome"] == "no_database"
    assert not missing.parent.exists()


def test_switched_off_means_no_scheduled_copy(database, monkeypatch):
    monkeypatch.setenv("WARRIORIQ_AUTO_BACKUP", "0")
    assert site_backup.daily_backup_if_due() is None
    assert not (database.parent / "backups").exists()


def test_only_current_leftovers_can_be_deleted(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    old = time.time() - 3 * 86400
    leftover = uploads / "gone_web.mp4"
    kept_original, kept_copy = uploads / "live.mp4", uploads / "live_web.mp4"
    young = uploads / "fresh_web.mp4"
    for path in (leftover, kept_original, kept_copy, young):
        path.write_bytes(b"x")
    for path in (leftover, kept_original, kept_copy):
        os.utime(path, (old, old))
    outside = tmp_path / "secret_web.mp4"
    outside.write_bytes(b"x")
    os.utime(outside, (old, old))
    monkeypatch.setattr(site_backup, "UPLOADS", uploads)

    assert [item["name"] for item in site_backup.leftover_copies()] == ["gone_web.mp4"]
    deleted = site_backup.delete_leftover_copies(
        ["gone_web.mp4", "live_web.mp4", "fresh_web.mp4", "../secret_web.mp4", "live.mp4"])
    assert deleted == ["gone_web.mp4"]
    assert not leftover.exists()
    assert kept_original.exists() and kept_copy.exists() and young.exists() and outside.exists()


def test_the_maintenance_sweep_makes_the_daily_copy(database):
    saved = (web._last_guest_cleanup, web._last_saved_video_cleanup)
    try:
        web._run_periodic_maintenance(now=time.monotonic() + 10 ** 7)
    finally:
        web._last_guest_cleanup, web._last_saved_video_cleanup = saved
    assert site_backup.latest() is not None
    assert any(event["event_type"] == "database_backup_created" for event in db.list_security_events())


def _admin_client():
    email = f"backup-admin-{uuid.uuid4().hex[:10]}@example.test"
    created = db.create_account(email, hash_password("Strong-Local-Password"))
    client = BrowserClient(web.app)
    client.cookies.set(web.SESSION_COOKIE, issue_session(created["id"]))
    web._rate_windows.clear()
    return client, email


def test_the_admin_page_backs_up_and_cleans_up_for_an_admin_only(database, tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    leftover = uploads / "gone_web.mp4"
    leftover.write_bytes(b"x")
    old = time.time() - 3 * 86400
    os.utime(leftover, (old, old))
    monkeypatch.setattr(site_backup, "UPLOADS", uploads)

    client, email = _admin_client()
    assert client.post("/admin/backup").status_code == 404           # not an administrator
    previous = SETTINGS.admin_emails
    object.__setattr__(SETTINGS, "admin_emails", (email.lower(),))
    try:
        page = client.get("/admin").text
        assert "No database backup in the last two days." in page   # none made yet
        assert 'id="adminBackups"' in page and "gone_web.mp4" in page
        done = client.post("/admin/backup", follow_redirects=False)
        assert done.status_code == 303 and "tool=backup_done" in done.headers["location"]
        page = client.get("/admin?tool=backup_done").text
        assert "Backup made and checked." in page and "No database backup in the last two days." not in page
        # Free text in the address is never shown.
        assert "Injected" not in client.get("/admin?tool=Injected").text
        # Deleting needs the confirmation box.
        refused = client.post("/admin/leftovers/delete", data={"name": "gone_web.mp4"}, follow_redirects=False)
        assert "tool=leftovers_none" in refused.headers["location"] and leftover.exists()
        client.post("/admin/leftovers/delete", data={"name": "gone_web.mp4", "confirm": "yes"})
        assert not leftover.exists()
    finally:
        object.__setattr__(SETTINGS, "admin_emails", previous)
