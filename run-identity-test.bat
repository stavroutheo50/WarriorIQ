@echo off
REM Double-click to test the fighter-lock fix on this PC (20-40 minutes).
REM The worker is paused while it runs and started again afterwards.
REM What it does is described in tools\run_identity_test.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_identity_test.ps1"
pause
