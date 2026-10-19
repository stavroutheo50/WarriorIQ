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
# 5: fighters re-identified by appearance after leaving the frame, camera
#    motion compensated before matching, officials held as a third-person
#    class, and a fighter followed for 20-45% of the fight reported at low
#    confidence instead of failing (audit 2026-10-14, #2).
# 6: guard needs both wrists up by the chin (core/guard.py hands_up_share/3;
#    audit 2026-10-14, #5).
ANALYSIS_VERSION = 6

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def build_commit() -> str:
    """The commit this code was built from, or "unknown".

    A Modal image ships without .git, so the deploy bakes the commit into
    WARRIORIQ_BUILD_COMMIT; a checkout answers from git directly.

    QA, 2026-10-07: reports said "build unknown". The cPanel deploy copies the
    code without .git and records the commit in DEPLOYED_COMMIT (.cpanel.yml),
    and Render sets RENDER_GIT_COMMIT; neither was read here, so an analysis
    run on either host could not name its own code. Both are now asked before
    git.
    """
    for name in ("WARRIORIQ_BUILD_COMMIT", "RENDER_GIT_COMMIT"):
        baked = os.getenv(name, "").strip()
        if baked:
            return baked[:12]
    try:
        deployed = (ROOT / "DEPLOYED_COMMIT").read_text(encoding="utf-8").strip()
    except OSError:
        deployed = ""
    if deployed:
        return deployed[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


@lru_cache(maxsize=1)
def analysis_fingerprint() -> str:
    """A hash of the analysis code itself: every file in core/.

    ANALYSIS_VERSION only moves when somebody remembers to raise it, and it was
    not raised through a week of changes to what an analysis measures - strike
    counts, re-crediting by shorts, timing - so a worker running a copy of the
    code from before them passed the check (2026-10-13: a worker run from a
    separate folder, reporting "commit unknown"). The fingerprint needs nobody
    to remember: any difference in core/ between the web app and the worker
    and the worker gets no fights until both run the same code.

    Line endings are normalised, so a Windows checkout and a Linux host of the
    same commit agree.
    """
    import hashlib

    digest = hashlib.sha256()
    for path in sorted((ROOT / "core").glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def analysis_file_hashes() -> dict[str, str]:
    """Each file in core/ with a hash of its own, so a mismatch can name the files.

    2026-10-14: a worker refused for running other code sat refused for twenty
    minutes after an update, and neither side could say what differed. With
    these the web app names the files that differ, and tells a stray extra file
    left in the worker's folder (never imported by code the same as the web
    app's) apart from code that really is different.
    """
    import hashlib

    return {path.name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
            for path in sorted((ROOT / "core").glob("*.py"))}


def code_differences(worker_files: dict[str, str]) -> dict[str, list[str]]:
    """How a worker's core/ differs from this one's, by file name."""
    own = analysis_file_hashes()
    return {"changed": sorted(name for name in own if name in worker_files and worker_files[name] != own[name]),
            "missing_on_worker": sorted(name for name in own if name not in worker_files),
            "extra_on_worker": sorted(name for name in worker_files if name not in own)}


def release_version() -> str:
    """The one version string the website and the worker must both report.

    QA, 2026-10-14: the two were deployed separately and drifted, and fights
    waited at "warming up". The string is the analysis version and the code
    fingerprint together - exactly what the claim route compares - so the two
    sides agree on this string if and only if the worker will be given fights.
    deploy/release.py prints it and checks both sides report it.
    """
    return f"{ANALYSIS_VERSION}.{analysis_fingerprint()[:12]}"


def stamp() -> dict:
    return {"analysis_version": ANALYSIS_VERSION, "commit": build_commit(),
            "fingerprint": analysis_fingerprint()}


def engine_name(report: dict | None) -> str:
    """The engine that made a report, for notices about older reports.

    "Older reports" said nothing a fighter could check, and a report three days
    old was called one (QA, 2026-10-07). Stamping began with version 2, so a
    report without a stamp came from before it.
    """
    version = result_check(report or {})["analysis_version"]
    return f"analysis engine v{version}" if version else "an analysis engine from before v2"


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
