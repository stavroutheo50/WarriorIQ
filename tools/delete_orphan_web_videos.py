"""Delete browser-playable video copies whose original video is already gone.

Until 2026-11-15 the 30-day retention sweep and Settings -> "delete original
video" removed a fight's original upload but left its browser-playable copy,
"<job>_web.mp4", in the uploads folder. Both paths now delete it; this tool
clears the copies left behind before that fix.

A dry run unless --confirm is given: it lists exactly what it would delete and
touches nothing else.

    python tools/delete_orphan_web_videos.py
    python tools/delete_orphan_web_videos.py --confirm

What it matches, and nothing more:

  * files directly in the uploads folder named "<job>_web.mp4";
  * whose original - any other file in the uploads folder named "<job>.<ext>" -
    no longer exists, so no page can play the copy any more (every media route
    serves it only through its original);
  * last changed more than --min-age-hours ago (default 24), so a copy being
    made for an upload in progress is never caught.

Run it with the same WARRIORIQ_DATA_DIR the website uses (in cPanel's Setup
Python App), after a database backup. Each deletion is written to
security_events.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUFFIX = "_web"


def orphan_copies(uploads: Path, min_age_hours: float, now: float | None = None) -> list[Path]:
    """The "<job>_web.mp4" files in `uploads` whose original is gone."""
    now = time.time() if now is None else now
    if not uploads.is_dir():
        return []
    files = [path for path in uploads.iterdir() if path.is_file()]
    originals = {path.stem for path in files if not path.stem.endswith(SUFFIX)}
    found = []
    for path in files:
        if path.suffix.lower() != ".mp4" or not path.stem.endswith(SUFFIX):
            continue
        job = path.stem[: -len(SUFFIX)]
        if not job or job in originals:
            continue
        if now - path.stat().st_mtime < min_age_hours * 3600:
            continue
        found.append(path)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-age-hours", type=float, default=24.0,
                        help="only copies older than this (default 24)")
    parser.add_argument("--confirm", action="store_true", help="actually delete; without it nothing changes")
    args = parser.parse_args()

    from core.config import DB_PATH, UPLOADS

    # Run with the wrong data folder, sqlite would create an empty database
    # beside an empty uploads folder and report nothing to do.
    if not Path(DB_PATH).is_file():
        print(f"No WarriorIQ database at {DB_PATH}. Set WARRIORIQ_DATA_DIR to the same value the website uses.")
        return 1
    copies = orphan_copies(Path(UPLOADS), args.min_age_hours)
    total = sum(path.stat().st_size for path in copies)
    print(f"Uploads folder: {UPLOADS}")
    for path in copies:
        print(f"  {path.name}  {path.stat().st_size / 1e6:.1f} MB")
    print(f"{len(copies)} leftover cop{'y' if len(copies) == 1 else 'ies'}, {total / 1e6:.1f} MB in all.")
    if not args.confirm:
        print("\nDry run: nothing was changed. Run again with --confirm to delete exactly the files above.")
        return 0

    from core.db import init_db, record_security_event

    init_db()
    for path in copies:
        path.unlink(missing_ok=True)
        record_security_event("orphan_web_video_deleted", resource_type="fight",
                              resource_id=path.stem[: -len(SUFFIX)], metadata={"tool": "delete_orphan_web_videos"})
    print(f"Deleted {len(copies)} file{'s' if len(copies) != 1 else ''}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
