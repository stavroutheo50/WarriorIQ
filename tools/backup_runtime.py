from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DB_PATH


def backup_database(destination_root: Path) -> dict:
    # sqlite3.connect creates a missing file, so a wrong path (a cron job that
    # did not get WARRIORIQ_DATA_DIR) would "back up" a new empty database and
    # report success. Refuse instead.
    if not Path(DB_PATH).is_file() or Path(DB_PATH).stat().st_size == 0:
        raise SystemExit(f"No WarriorIQ database at {DB_PATH}. Set WARRIORIQ_DATA_DIR (or "
                         "WARRIORIQ_DB_PATH) to the same value the website uses.")
    destination_root = destination_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_root / f"warrioriq-{stamp}.sqlite3"
    with closing(sqlite3.connect(DB_PATH)) as source, closing(sqlite3.connect(destination)) as target:
        source.backup(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Backup integrity check failed: {integrity}")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database_file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "integrity_check": integrity,
        "contains_original_videos": False,
    }
    manifest_path = destination.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "path": str(destination), "manifest_path": str(manifest_path)}


def prune_backups(destination_root: Path, keep: int) -> list[str]:
    """Delete all but the newest `keep` backups this tool made in `destination_root`.

    For a scheduled run (a cPanel cron job), so daily copies do not fill the
    account's disk. Only files named like this tool's own output are touched:
    warrioriq-<UTC stamp>.sqlite3 and its .json manifest. The stamp sorts by
    time, so the newest are kept. Nothing is deleted unless keep is at least 1.
    """
    if keep < 1:
        return []
    destination_root = destination_root.expanduser().resolve()
    copies = sorted(path for path in destination_root.glob("warrioriq-*.sqlite3")
                    if _BACKUP_NAME.fullmatch(path.name))
    removed = []
    for old in copies[:-keep]:
        for path in (old, old.with_suffix(".json")):
            if path.exists():
                path.unlink()
                removed.append(path.name)
    return removed


_BACKUP_NAME = re.compile(r"warrioriq-\d{8}T\d{6}Z\.sqlite3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an integrity-checked WarriorIQ database backup")
    parser.add_argument("destination", help="Existing protected backup root or a directory to create")
    parser.add_argument("--keep", type=int, default=0,
                        help="After a good backup, keep only the newest N backups in the destination "
                             "(0, the default, deletes nothing)")
    args = parser.parse_args()
    result = backup_database(Path(args.destination))
    result["removed_old_backups"] = prune_backups(Path(args.destination), args.keep)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
