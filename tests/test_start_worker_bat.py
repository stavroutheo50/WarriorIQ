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


def test_update_worker_replaces_core_whole_and_leaves_private_files():
    """2026-10-14: the worker ran from a folder nobody updated, and was refused for twenty minutes."""
    bat = (ROOT / "update-worker.bat").read_text(encoding="utf-8")
    assert "deploy\\update-worker.ps1" in bat
    script = (ROOT / "deploy" / "update-worker.ps1").read_text(encoding="utf-8")
    # A stray file left in core/ changes the fingerprint, so core is mirrored, not merged.
    assert re.search(r"robocopy \(Join-Path \$source 'core'\) \(Join-Path \$root 'core'\) /MIR", script)
    assert "--branch main" in script and "analysis_fingerprint" in script
    assert "worker.py" in script and "start-worker.bat" in script
    # Only code is copied: .env, .venv and outputs are never named as targets.
    for private in (".env'", ".venv'", "'outputs'"):
        assert f"Join-Path $root {private}" not in script.replace('"', "'")


def test_the_worker_will_not_run_under_another_programs_python(tmp_path, monkeypatch):
    """2026-10-14: a worker started with another app's Python held the lock, so no fight started."""
    import sys

    import worker

    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(worker, "__file__", str(tmp_path / "worker.py"))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
    assert "start-worker.bat" in worker._foreign_python(windows=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv"))
    assert worker._foreign_python(windows=True) is None
    # Linux hosts and machines without a project .venv are not affected.
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
    assert worker._foreign_python(windows=False) is None
    (tmp_path / ".venv").rmdir()
    assert worker._foreign_python(windows=True) is None
    bat = (ROOT / "start-worker.bat").read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in bat
