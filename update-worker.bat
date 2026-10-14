@echo off
REM WarriorIQ - update this folder's worker to the code on GitHub main and
REM restart it. Run it after every website deploy: the website refuses a worker
REM running different analysis code. See deploy\update-worker.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\update-worker.ps1"
pause
