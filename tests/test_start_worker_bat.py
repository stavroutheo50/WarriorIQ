"""start-worker.bat stops retrying when the background worker is already running."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_window_closes_on_the_already_running_exit_code():
    source = (ROOT / "worker.py").read_text(encoding="utf-8")
    code = int(re.search(r"^EXIT_ANOTHER_WORKER_IS_RUNNING = (\d+)$", source, re.M).group(1))
    bat = (ROOT / "start-worker.bat").read_text(encoding="utf-8")
    # "errorlevel N" means N or more, so the pair matches exactly this code.
    assert f"if errorlevel {code} if not errorlevel {code + 1} goto already_running" in bat
    assert ":already_running" in bat and "exit /b 0" in bat


def test_setup_removes_only_startup_entries_that_launch_the_bat():
    setup = (ROOT / "deploy" / "setup-always-on.ps1").read_text(encoding="utf-8")
    assert "GetFolderPath('Startup')" in setup
    assert "-like '*start-worker.bat'" in setup
