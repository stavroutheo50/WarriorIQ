"""Backups and server checks that need no shell (owner, 2026-11-16).

The host has no shell access ("Shell access is not enabled on your account"),
so a cPanel cron job for tools/backup_runtime.py could not be set up and the
server checks in OPS_TODO.md could not be run. The website does them itself:

- once a day the periodic maintenance sweep (app/main.py) makes an
  integrity-checked copy of the database with the same code as
  tools/backup_runtime.py, and keeps the newest KEEP copies;
- the admin page shows the last backup, lets an administrator make one now,
  says whether ffprobe and ffmpeg are installed, and lists the browser copies
  of videos whose original is already gone (tools/delete_orphan_web_videos.py)
  with a confirmed delete.

Backups hold the database only, never videos. They go next to the database
in a "backups" folder unless WARRIORIQ_BACKUP_DIR names another; that folder
is not served by any route. Passenger can run several website processes, so
making a backup takes a file lock and checks again inside it: two processes
never copy at once and a day never gets two scheduled copies.

In app/ rather than core/: every core/*.py is part of the analysis-code
fingerprint the worker must match, and none of this is analysis.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.config import DB_PATH, UPLOADS
from tools.backup_runtime import _BACKUP_NAME, backup_database, prune_backups
from tools.delete_orphan_web_videos import orphan_copies

DAY = timedelta(hours=24)
# A second copy within this time of the last one is refused: each copy counts
# toward KEEP, so repeated presses would otherwise push out older days.
MANUAL_GAP = timedelta(minutes=5)
LEFTOVER_MIN_AGE_HOURS = 24.0

try:                                                    # Linux hosting
    import fcntl
except ImportError:                                     # pragma: no cover - Windows development
    fcntl = None

_process_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("WARRIORIQ_AUTO_BACKUP", "1").strip().lower() not in {"0", "false", "no", "off"}


def keep() -> int:
    try:
        return max(1, int(os.getenv("WARRIORIQ_BACKUP_KEEP", "14")))
    except ValueError:
        return 14


def backup_dir() -> Path:
    configured = os.getenv("WARRIORIQ_BACKUP_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path(DB_PATH).parent / "backups"


def _stamp(name: str) -> datetime | None:
    try:
        return datetime.strptime(name[len("warrioriq-"):-len(".sqlite3")], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None


def backups() -> list[dict]:
    """This folder's backups, newest first, with what each manifest recorded."""
    folder = backup_dir()
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("warrioriq-*.sqlite3"), reverse=True):
        created = _stamp(path.name) if _BACKUP_NAME.fullmatch(path.name) else None
        if created is None:
            continue
        try:
            manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        found.append({"name": path.name, "created_at": created, "bytes": path.stat().st_size,
                      "integrity_check": manifest.get("integrity_check"),
                      "sha256": manifest.get("sha256")})
    return found


def latest() -> dict | None:
    copies = backups()
    return copies[0] if copies else None


@contextmanager
def _exclusive(folder: Path):
    """Yield True while this process holds the backup lock, False if another does."""
    if not _process_lock.acquire(blocking=False):
        yield False
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if fcntl is None:                               # pragma: no cover
            yield True
            return
        with open(folder / ".backup.lock", "a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        _process_lock.release()


def make_backup(*, scheduled: bool, now: datetime | None = None) -> dict:
    """Make one backup if it is due; say what happened.

    Returns {"outcome": "done" | "not_due" | "recent" | "busy" | "no_database" |
    "failed", ...}. A scheduled run is due a day after the last copy; a manual
    one is refused within MANUAL_GAP of the last copy.
    """
    now = now or datetime.now(timezone.utc)
    if not Path(DB_PATH).is_file() or Path(DB_PATH).stat().st_size == 0:
        return {"outcome": "no_database"}
    folder = backup_dir()
    with _exclusive(folder) as held:
        if not held:
            return {"outcome": "busy"}
        # Checked again inside the lock: another process may just have made it.
        last = latest()
        if last is not None:
            age = now - last["created_at"]
            if scheduled and age < DAY:
                return {"outcome": "not_due"}
            if not scheduled and age < MANUAL_GAP:
                return {"outcome": "recent"}
        try:
            result = backup_database(folder)
        except (SystemExit, RuntimeError, OSError) as error:
            return {"outcome": "failed", "error": str(error)[:300]}
        removed = prune_backups(folder, keep())
        return {"outcome": "done", "name": Path(result["path"]).name, "bytes": result["bytes"],
                "removed": removed}


def daily_backup_if_due() -> dict | None:
    """The scheduled copy, from the periodic maintenance sweep."""
    if not enabled():
        return None
    return make_backup(scheduled=True)


def server_checks() -> dict:
    """What the host has installed, found the way the upload check finds it."""
    return {"ffprobe": shutil.which("ffprobe"), "ffmpeg": shutil.which("ffmpeg")}


def leftover_copies() -> list[dict]:
    """Browser copies of videos whose original is gone (see tools/delete_orphan_web_videos.py)."""
    found = []
    for path in orphan_copies(Path(UPLOADS), LEFTOVER_MIN_AGE_HOURS):
        try:
            found.append({"name": path.name, "bytes": path.stat().st_size})
        except OSError:
            continue
    return found


def delete_leftover_copies(names: list[str]) -> list[str]:
    """Delete the named copies that are leftovers right now, and nothing else.

    Each name must be on the current leftover list, worked out again here, so
    a stale page or a hand-made request can never name another file.
    """
    wanted = set(names)
    deleted = []
    for path in orphan_copies(Path(UPLOADS), LEFTOVER_MIN_AGE_HOURS):
        if path.name in wanted:
            path.unlink(missing_ok=True)
            deleted.append(path.name)
    return deleted
