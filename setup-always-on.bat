@echo off
REM Double-click to make this PC an always-on WarriorIQ analysis machine:
REM the worker runs as a background service and the PC stops sleeping.
REM Windows will ask for administrator permission - say Yes.
REM What it does is described in deploy\setup-always-on.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\setup-always-on.ps1"
