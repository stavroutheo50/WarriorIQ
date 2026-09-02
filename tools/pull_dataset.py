"""Bring the labelled training set down from the web server.

Labels are made in the browser and saved on the web server. The model is
trained here, on the machine with the GPU. Nothing joined the two, so a
labelling session sat on the server with no way to reach the trainer - and this
host has no file manager to fetch it by hand.

    .venv/Scripts/python.exe tools/pull_dataset.py

Reads the same WARRIORIQ_WORKER_REMOTE_URL and WARRIORIQ_WORKER_TOKEN the
worker already uses, so there is nothing new to configure. Existing files are
left alone unless --overwrite is passed; labelling is additive and re-running
this should never lose work.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import DATASET, SETTINGS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download labelled sequences from the web server")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace local files that already exist")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    base = SETTINGS.worker_remote_url
    token = SETTINGS.worker_token
    if not base or not token:
        print("Set WARRIORIQ_WORKER_REMOTE_URL and WARRIORIQ_WORKER_TOKEN in .env first.")
        return 2

    url = f"{base}/api/worker/dataset"
    print(f"asking {base} for the labelled set")
    try:
        response = requests.get(
            url, headers={"Authorization": f"Bearer {token}"}, timeout=args.timeout,
        )
    except requests.RequestException as exc:
        print(f"could not reach the server: {exc}")
        return 1

    if response.status_code == 404:
        # A missing route and an empty dataset both answer 404, and they need
        # opposite actions, so tell them apart by our own error text.
        detail = ""
        try:
            detail = str(response.json().get("detail", ""))
        except ValueError:
            pass
        if "labelled sequences" in detail:
            print("the server has no labelled sequences yet - label some fights first")
        else:
            print("this WarriorIQ has no /api/worker/dataset route yet - deploy the")
            print("latest commit in cPanel first, then run this again")
        return 1
    if response.status_code == 401:
        print("the server rejected the worker token; check it matches the one on the web server")
        return 1
    if response.status_code != 200:
        print(f"unexpected reply {response.status_code}: {response.text[:200]}")
        return 1

    destination = DATASET / "sequences"
    destination.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        for name in bundle.namelist():
            # Never let a served name escape the dataset folder.
            safe = Path(name).name
            if not safe.endswith(".npz"):
                continue
            target = destination / safe
            if target.exists() and not args.overwrite:
                skipped += 1
                continue
            target.write_bytes(bundle.read(name))
            written += 1

    total = len(list(destination.glob("*.npz")))
    print(f"downloaded {written} new, left {skipped} already here")
    print(f"{total} labelled sequences now in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
