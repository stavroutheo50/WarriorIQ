# WarriorIQ - test the fighter-lock fix on this PC, in one go.
#
# Runs tools/run_identity_benchmark.py: every hand-marked clip is analysed
# twice, with the setting under test off and on (WARRIORIQ_BYSTANDER_MEMORY:
# remembering the referee who stands beside the fighters), and scored against the frames
# a person marked. The answer decides whether the setting is switched on.
#
#   1. stops the worker, so the benchmark has the GPU to itself;
#   2. pulls the latest main (skipped, with a note, if this PC cannot);
#   3. runs the benchmark - downloads the clips once, then 20-40 minutes;
#   4. starts the worker again, whatever happened in step 3.
#
# Double-click run-identity-test.bat in the project folder. The results are
# written to logs\identity-benchmark.txt. The real fight database and outputs
# are not touched (see the benchmark script).

param([string]$Root = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Continue'
$root   = (Resolve-Path $Root).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
$out    = Join-Path $logDir 'identity-benchmark.txt'
$task   = 'WarriorIQ Analysis Worker'
Set-Location $root
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
if (-not (Test-Path $python)) { Write-Host "STOPPED: the project .venv is missing ($python)"; exit 1 }

# 1. Stop the worker (same as tools\update_worker.ps1).
$scheduled = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
if ($scheduled) { Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue; Write-Host "Stopped scheduled task '$task'" }
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*worker.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host "Stopped worker process $($_.ProcessId)" }
Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -like '*start-worker.bat*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host "Closed start-worker.bat window $($_.ProcessId)" }

try {
    # 2. Latest code. A PC with local edits keeps them; update_worker.ps1 sorts those out.
    git pull --ff-only origin main 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { Write-Host "Could not pull main - testing the code already on this PC" }
    if (-not (Test-Path (Join-Path $root 'tools\run_identity_benchmark.py'))) {
        Write-Host "STOPPED: this PC's code is too old for the test. Run tools\update_worker.ps1 first."
        exit 1
    }

    # 3. The benchmark. Output to the screen and to the results file.
    Write-Host "Running the fighter-lock test. Leave this window open: 20-40 minutes."
    "WarriorIQ fighter-lock test, $(Get-Date -Format 'yyyy-MM-dd HH:mm'), commit $(git rev-parse --short HEAD)" |
        Set-Content -Path $out -Encoding UTF8
    & $python -u (Join-Path $root 'tools\run_identity_benchmark.py') 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $out -Append | Out-Host
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "FINISHED. Send the two TOTAL lines above (they are also in logs\identity-benchmark.txt)."
    } else {
        Write-Host "The test stopped with an error - send logs\identity-benchmark.txt"
    }
}
finally {
    # 4. Worker back on, the way it was being run.
    if ($scheduled) {
        Start-ScheduledTask -TaskName $task
        Write-Host "Started scheduled task '$task'"
    } else {
        Start-Process -FilePath (Join-Path $root 'start-worker.bat') -WorkingDirectory $root
        Write-Host "Started start-worker.bat in a new window"
    }
}
