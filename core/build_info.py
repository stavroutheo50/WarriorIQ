"""Which analysis code produced a result.

QA, 2026-10-04: reports made after the whole-video fix still started at the
fighter-selection frame. The web app had the fix; the GPU worker on Modal had
been deployed from an older checkout and was running the old analyser. Nothing
in a report said which code made it, so the two could disagree silently.

Every result now carries a stamp from here. ``ANALYSIS_VERSION`` is a plain
integer, raised by hand whenever a change alters what an analysis measures or
covers; the web app refuses work to, and flags results from, a worker whose
number is lower than its own. The commit is for people reading a report: it is
not ordered, so it is shown and never compared.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

# 2: the whole video is analysed from 0:00; the selection frame only seeds
#    identity (core/analyzer.py _analyze_from_seed).
# 3: solo sessions (core/solo.py). An older worker would read a solo job as a
#    two-fighter one with no Fighter B.
# 4: a quarter turn applied while decoding (AnalysisRequest.rotate_clockwise),
#    with the fighter boxes drawn on the turned frame; and guard measured as
#    wrist-to-chin over shoulder width (core/guard.py). An older worker would
#    analyse a turned job unturned, with its boxes on the wrong pixels.
ANALYSIS_VERSION = 4

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def build_commit() -> str:
    """The commit this code was built from, or "unknown".

    A Modal image ships without .git, so the deploy bakes the commit into
    WARRIORIQ_BUILD_COMMIT; a checkout answers from git directly.
    """
    baked = os.getenv("WARRIORIQ_BUILD_COMMIT", "").strip()
    if baked:
        return baked[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def stamp() -> dict:
    return {"analysis_version": ANALYSIS_VERSION, "commit": build_commit()}


def result_check(report: dict) -> dict:
    """Whether a result came from analysis code at least as new as this one.

    A report with no stamp predates stamping, so it is older by definition.
    """
    produced = (report or {}).get("analysis_build") or {}
    try:
        version = int(produced.get("analysis_version"))
    except (TypeError, ValueError):
        version = None
    return {
        "analysis_version": version,
        "commit": produced.get("commit") or "unknown",
        "current_version": ANALYSIS_VERSION,
        "outdated": version is None or version < ANALYSIS_VERSION,
    }
