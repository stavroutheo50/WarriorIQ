"""Keep the test suite out of the development runtime.

There was no conftest at all, so every run wrote straight into the real
database and the real job queue. By 2026-09-13 that had left **196 test
accounts** among two real ones, and **725 phantom jobs** in `outputs/` - each
one a session file `claim_next_job` hands the worker, naming a video under a
temp directory Windows had already cleared. The worker claims each, fails, and
moves on.

Only the three paths the build writes are moved. `MODELS` and `DATASET` stay
where they are: they are read rather than written, and a test that needs the
pose engine still has to find it.

The environment has to be set before `core.config` computes those paths at
import time, which is why this happens at module level rather than in a
fixture - pytest imports conftest before it collects anything. If that ever
stops being true the guard below fails the run loudly, because the failure it
is guarding against is silent: the suite would simply go back to filling the
development database, and nobody would notice for another three weeks.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile

if "core.config" in sys.modules:                                    # pragma: no cover
    raise RuntimeError(
        "core.config was imported before tests/conftest.py could redirect the "
        "runtime paths, so this run would write to the development database "
        "and job queue. Something now imports it at collection time; move that "
        "import inside the test that needs it."
    )

_RUNTIME = pathlib.Path(tempfile.mkdtemp(prefix="warrioriq-tests-"))
(_RUNTIME / "uploads").mkdir(parents=True, exist_ok=True)
(_RUNTIME / "outputs").mkdir(parents=True, exist_ok=True)
os.environ["WARRIORIQ_DB_PATH"] = str(_RUNTIME / "warrioriq.sqlite3")
os.environ["WARRIORIQ_UPLOADS_DIR"] = str(_RUNTIME / "uploads")
os.environ["WARRIORIQ_OUTPUTS_DIR"] = str(_RUNTIME / "outputs")


def pytest_report_header(config) -> str:
    """Say where the run is writing, so an isolated run is visible, not assumed."""
    return f"warrioriq runtime (scratch): {_RUNTIME}"


def pytest_sessionfinish(session, exitstatus) -> None:
    shutil.rmtree(_RUNTIME, ignore_errors=True)
