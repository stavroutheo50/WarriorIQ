@echo off
REM WarriorIQ analysis worker.
REM
REM Nothing is analysed unless this is running: the website only queues jobs,
REM and every upload waits here until this machine claims it. Uses %~dp0 so it
REM works wherever the project lives.
REM
REM Close the window to stop it. Restarts itself if the analysis crashes, so a
REM single bad video cannot leave the queue stalled.
title WarriorIQ worker
cd /d "%~dp0"
:loop
".venv\Scripts\python.exe" worker.py
echo.
echo Worker stopped. Restarting in 10 seconds - close this window to stop for good.
timeout /t 10 /nobreak >nul
goto loop
