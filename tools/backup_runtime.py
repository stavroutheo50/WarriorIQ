from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an integrity-checked WarriorIQ database backup")
    parser.add_argument("destination", help="Existing protected backup root or a directory to create")
    args = parser.parse_args()
    print(json.dumps(backup_database(Path(args.destination)), indent=2))


if __name__ == "__main__":
    main()
