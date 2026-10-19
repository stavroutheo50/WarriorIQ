"""Check that the website runs the same release as this checkout.

QA, 2026-10-14 (#23): the worker and the website were deployed separately,
drifted apart, and fights sat at "warming up" because the website refuses work
to a worker running other analysis code. Both now report one release string,
core/build_info.release_version() - the analysis version and the analysis code
fingerprint - which is exactly what the website compares when a worker claims a
fight. Equal strings mean the worker will be given fights.

    python deploy/release.py                    # print the release, wait for the site to match
    python deploy/release.py --site https://warrioriq.eu --wait 900

Order: the website deploys itself from main on Render (render.yaml autoDeploy).
Update the GPU worker PC to the same main (deploy/update-worker.ps1), then run
this from that folder: it waits until the site's /health reports the same
release. An uncommitted change is a different release from what Render builds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def site_release(site: str, timeout: float = 10.0) -> str | None:
    try:
        with urllib.request.urlopen(f"{site.rstrip('/')}/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")).get("release")
    except Exception:                                               # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    from core.build_info import build_commit, release_version

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="https://warrioriq.eu")
    parser.add_argument("--wait", type=float, default=600.0, help="Seconds to wait for the site to match")
    args = parser.parse_args(argv)

    release = release_version()
    print(f"release {release} (commit {build_commit()})")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True, check=False).stdout.strip()
    if dirty:
        print("warning: uncommitted changes here make this a different release from the site")
    deadline = time.time() + max(0.0, args.wait)
    while True:
        live = site_release(args.site)
        if live == release:
            print(f"site {args.site} reports release {live}: worker and website match")
            return 0
        if time.time() >= deadline:
            print(f"site {args.site} reports release {live!r}, expected {release}: "
                  "the worker will be refused until the website runs this commit")
            return 1
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
