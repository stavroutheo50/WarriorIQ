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
REM 3 = another worker already holds the lock (worker.py
REM EXIT_ANOTHER_WORKER_IS_RUNNING): usually the background service that
REM setup-always-on.bat installs. Nothing is wrong, and retrying every ten
REM seconds forever only fills the screen with errors, so this window closes.
if errorlevel 3 if not errorlevel 4 goto already_running
echo.
echo Worker stopped. Restarting in 10 seconds - close this window to stop for good.
timeout /t 10 /nobreak >nul
goto loop

:already_running
echo.
echo The WarriorIQ worker is already running in the background - this window is not needed.
echo It closes in 15 seconds. Fights are still being analysed.
timeout /t 15 /nobreak >nul
exit /b 0
