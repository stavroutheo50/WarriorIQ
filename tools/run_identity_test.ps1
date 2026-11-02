# WarriorIQ - test the fighter-lock fix on this PC, in one go.
#
# Runs tools/run_identity_benchmark.py once per setting under test: every
# hand-marked clip is analysed twice, with the setting off and on, and scored
# against the frames a person marked. The answer decides whether each setting
# is switched on. Tested by default (2026-11-02):
#   WARRIORIQ_OFFICIAL_KIT_OVERRIDE - a fighter in their own kit is not
#       refused as the referee (on since PR #193; checks it does not let a
#       real referee in on the pankration bouts);
#   WARRIORIQ_BYSTANDER_MEMORY - refuse people already seen standing beside
#       the fighters (a coach at the ropes took a fighter's box on S01).
#
#   1. stops the worker, so the benchmark has the GPU to itself;
#   2. pulls the latest main (skipped, with a note, if this PC cannot);
#   3. runs the benchmark - downloads the clips once, then 20-40 minutes;
#   4. starts the worker again, whatever happened in step 3.
#
# Double-click run-identity-test.bat in the project folder. The results are
# written to logs\identity-benchmark.txt. The real fight database and outputs
# are not touched (see the benchmark script).

param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [string[]]$Settings = @('WARRIORIQ_OFFICIAL_KIT_OVERRIDE', 'WARRIORIQ_BYSTANDER_MEMORY')
)

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
    Write-Host "Running the fighter-lock test. Leave this window open: about 20-40 minutes per setting."
    "WarriorIQ fighter-lock test, $(Get-Date -Format 'yyyy-MM-dd HH:mm'), commit $(git rev-parse --short HEAD)" |
        Set-Content -Path $out -Encoding UTF8
    $failed = $false
    foreach ($setting in $Settings) {
        "" | Add-Content -Path $out
        "=== $setting off vs on ===" | Tee-Object -FilePath $out -Append | Out-Host
        & $python -u (Join-Path $root 'tools\run_identity_benchmark.py') --setting $setting 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath $out -Append | Out-Host
        if ($LASTEXITCODE -ne 0) { $failed = $true }
    }
    if (-not $failed) {
        Write-Host ""
        Write-Host "FINISHED. Send logs\identity-benchmark.txt (or the TOTAL lines above)."
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
