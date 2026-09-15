"""Give existing uploads a browser-friendly copy without re-encoding them.

Phone footage arrives in a QuickTime container. Measured on a real 101 MB
iPhone upload, the streams inside are already H.264 High and AAC - only the
wrapper is QuickTime - so copying them into a standard MP4 takes about a fifth
of a second and loses nothing. Re-encoding the same file took 22 seconds and
discarded three quarters of the data to reach a format it was already in.

Originals are never modified or removed: analysis reads them, and the replay
page offers them for download when playback fails. This only adds a sibling
file, and only where one is missing.

Files whose streams cannot be copied - ProRes, HEVC - are reported and skipped.
Those need a real re-encode, which is a deliberate decision about CPU time and
not something a backfill should start on its own.

    python tools/backfill_web_video.py --dry-run
    python tools/backfill_web_video.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import SETTINGS
from core.video import _PLAYABLE_CONTAINERS, derivative_for, normalize_container


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uploads", default=str(PROJECT_ROOT / "uploads"),
        help="Directory holding the uploaded fight videos.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be converted and stop.")
    args = parser.parse_args()

    uploads = Path(args.uploads)
    if not uploads.is_dir():
        print(f"No uploads directory at {uploads}")
        raise SystemExit(1)

    candidates = sorted(
        path for path in uploads.iterdir()
        if path.is_file()
        and path.suffix.lower() not in _PLAYABLE_CONTAINERS
        and not path.stem.endswith("_web")
        and not derivative_for(path).exists()
    )
    if not candidates:
        print("Nothing to do: every upload already plays as it is, or already "
              "has a copy beside it.")
        return

    print(f"{len(candidates)} upload(s) without a browser-friendly copy:")
    for path in candidates:
        print(f"  {path.name}  {path.stat().st_size / 1048576:.1f} MB")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    converted = skipped = 0
    started = time.perf_counter()
    for path in candidates:
        result = normalize_container(path)
        if result is None:
            skipped += 1
            print(f"  skipped {path.name} - streams cannot be copied; this one "
                  f"needs a real re-encode")
        else:
            converted += 1
            print(f"  wrote   {result.name}  {result.stat().st_size / 1048576:.1f} MB")
    print(f"\n{converted} converted, {skipped} skipped, "
          f"{time.perf_counter() - started:.1f}s total.")
    if skipped:
        print("Skipped files still play wherever their codec is supported, and "
              "the replay page offers the original for download when it is not.")


if __name__ == "__main__":
    main()
