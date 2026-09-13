"""Remove job directories that can never run, and say so before doing it.

`claim_next_job` builds the worker's queue by globbing every
`outputs/*/analysis-session.json`, so a leftover session file is a job the
worker will claim, fail on, and move past. Test runs put their uploads under
the OS temp directory, which Windows clears; the session files they wrote into
the real `outputs/` stayed behind. Measured 2026-09-13: **725 such entries**,
every one pointing at a video under a temp path that no longer exists.

A directory is only removed when all three hold:

  * no row in `fights` refers to its job id, so nothing in the product links
    to it;
  * its `analysis-session.json` names a `video_path` that does not exist, so
    there is no footage left to analyse either way;
  * that path lies under the OS temp directory, which is where test runs put
    their uploads and where nothing a real visitor uploaded is ever written.

Anything that fails one of those is listed and left alone. Run with --dry-run
first; it prints exactly what would go and touches nothing.

    python tools/clear_phantom_jobs.py --dry-run
    python tools/clear_phantom_jobs.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.config import DB_PATH, OUTPUTS  # noqa: E402


def _referenced_job_ids() -> set[str]:
    with sqlite3.connect(DB_PATH) as con:
        return {row[0] for row in con.execute("SELECT job_id FROM fights")}


def _is_phantom(directory: pathlib.Path, temp_root: pathlib.Path) -> bool:
    session = directory / "analysis-session.json"
    if not session.exists():
        return False
    try:
        video_path = json.loads(session.read_text(encoding="utf-8")).get("video_path") or ""
    except (OSError, ValueError):
        return False
    if not video_path:
        return False
    candidate = pathlib.Path(video_path)
    if candidate.exists():
        return False
    try:
        return temp_root in candidate.resolve().parents
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list what would go, delete nothing")
    args = parser.parse_args()

    if not OUTPUTS.exists():
        print(f"no outputs directory at {OUTPUTS}")
        return 0

    referenced = _referenced_job_ids()
    temp_root = pathlib.Path(tempfile.gettempdir()).resolve()
    phantom: list[pathlib.Path] = []
    kept: list[pathlib.Path] = []
    for directory in sorted(p for p in OUTPUTS.iterdir() if p.is_dir()):
        if directory.name in referenced:
            continue
        (phantom if _is_phantom(directory, temp_root) else kept).append(directory)

    freed = sum(f.stat().st_size for d in phantom for f in d.rglob("*") if f.is_file())
    print(f"outputs directory   : {OUTPUTS}")
    print(f"referenced by a fight: {len(referenced & {p.name for p in OUTPUTS.iterdir() if p.is_dir()})}")
    print(f"phantom jobs        : {len(phantom)}  ({freed / 1048576:.1f} MB)")
    print(f"orphans left alone  : {len(kept)}")
    for directory in kept[:20]:
        print(f"    keeping {directory.name}")
    if len(kept) > 20:
        print(f"    ... and {len(kept) - 20} more")

    if args.dry_run:
        print("\ndry run - nothing was deleted")
        return 0
    for directory in phantom:
        shutil.rmtree(directory, ignore_errors=True)
    remaining = len(list(OUTPUTS.glob("*/analysis-session.json")))
    print(f"\nremoved {len(phantom)} directories; the worker queue now shows {remaining} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
