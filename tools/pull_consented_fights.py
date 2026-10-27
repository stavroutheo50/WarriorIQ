"""Bring down the fights their owners shared to improve WarriorIQ (2026-10-27).

Customers who tick "Help WarriorIQ get better" on upload have their fight's
video, report and tracking kept on the web server (core/training_consent.py).
This copies them to the training machine, into dataset/consented_fights/<job>/,
where tools/benchmark_dense_strikes.py and tools/fit_strike_check.py can use
them once they are labelled.

    .venv/Scripts/python.exe tools/pull_consented_fights.py

Reads the same WARRIORIQ_WORKER_REMOTE_URL and WARRIORIQ_WORKER_TOKEN the worker
uses. Files already here are skipped. A fight deleted on the website since the
last pull is deleted here too, so a customer's deletion reaches every copy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import DATASET, SETTINGS  # noqa: E402

DESTINATION = DATASET / "consented_fights"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the fights customers shared for training")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    base, token = SETTINGS.worker_remote_url, SETTINGS.worker_token
    if not base or not token:
        print("Set WARRIORIQ_WORKER_REMOTE_URL and WARRIORIQ_WORKER_TOKEN in .env first.")
        return 2
    headers = {"Authorization": f"Bearer {token}"}
    try:
        listing = requests.get(f"{base}/api/worker/consented", headers=headers, timeout=60)
    except requests.RequestException as exc:
        print(f"could not reach the server: {exc}")
        return 1
    if listing.status_code == 404:
        print("this WarriorIQ has no /api/worker/consented route yet - deploy the latest commit first")
        return 1
    if listing.status_code != 200:
        print(f"unexpected reply {listing.status_code}: {listing.text[:200]}")
        return 1

    fights = listing.json().get("fights") or []
    DESTINATION.mkdir(parents=True, exist_ok=True)
    shared = {str(fight.get("job_id")) for fight in fights}
    # Deleted on the website: delete here too.
    for folder in DESTINATION.iterdir():
        if folder.is_dir() and folder.name not in shared:
            shutil.rmtree(folder, ignore_errors=True)
            print(f"removed {folder.name} (deleted by its owner)")
    downloaded = 0
    for fight in fights:
        job_id = str(fight["job_id"])
        folder = DESTINATION / job_id
        folder.mkdir(exist_ok=True)
        (folder / "consent.json").write_text(json.dumps(fight, indent=1), encoding="utf-8")
        for name, size in (fight.get("files") or {}).items():
            target = folder / name
            if target.exists() and target.stat().st_size == size:
                continue
            with requests.get(f"{base}/api/worker/consented/{job_id}/{name}", headers=headers,
                              timeout=args.timeout, stream=True) as response:
                if response.status_code != 200:
                    print(f"{job_id}/{name}: {response.status_code}")
                    continue
                partial = target.with_suffix(target.suffix + ".part")
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
                partial.replace(target)
                downloaded += 1
    print(f"{len(fights)} shared fights, {downloaded} files downloaded into {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
