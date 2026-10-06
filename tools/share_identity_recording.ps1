# WarriorIQ - record what fighter identity saw on the test fights, and send it.
#
# Runs tools/identity_recording.py record on this PC (the GPU, the pose refiner
# and the recovery step that a cloud machine does not have), then pushes the
# recordings to their own GitHub branch so an identity fix can be replayed
# against this PC's real detections in seconds, instead of one full test run
# per idea. The recordings hold only the detections of the public test clips
# and the numeric settings - no secrets, no customer data.
#
#   1. stops the worker, so the recording has the GPU to itself;
#   2. pulls the latest main;
#   3. records every test fight (about 10 minutes);
#   4. starts the worker again, whatever happened in step 3;
#   5. pushes the recordings to a new branch "identity-recording-<date>" and
#      switches back to main. Local edits on this PC are not touched.
#
# Run from the project folder:
#     powershell -ExecutionPolicy Bypass -File tools\share_identity_recording.ps1

param([string]$Root = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Continue'
$root   = (Resolve-Path $Root).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$task   = 'WarriorIQ Analysis Worker'
$folder = 'dataset\public\identity_recordings'
Set-Location $root
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

$recorded = $false
try {
    # 2. Latest code.
    git pull --ff-only origin main 2>&1 | Out-Host
    if (-not (Test-Path (Join-Path $root 'tools\identity_recording.py'))) {
        Write-Host "STOPPED: this PC's code is too old. Run: git pull origin main"
        exit 1
    }
    # 3. Record.
    Write-Host "Recording the test fights. Leave this window open: about 10 minutes."
    & $python -u (Join-Path $root 'tools\identity_recording.py') record 2>&1 | ForEach-Object { "$_" } | Out-Host
    $recorded = ($LASTEXITCODE -eq 0)
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
if (-not $recorded) { Write-Host "The recording stopped with an error - send a photo of this window."; exit 1 }

# 5. Push the recordings to their own branch, then back to main.
$branch = "identity-recording-$(Get-Date -Format 'yyyyMMdd-HHmm')"
git switch -c $branch 2>&1 | Out-Host
git add -f "$folder\*.pkl.gz" 2>&1 | Out-Host
git -c user.name="WarriorIQ PC" -c user.email="worker@warrioriq.eu" commit -m "Identity recordings of the benchmark clips from the analysis PC" 2>&1 | Out-Host
git push -u origin $branch 2>&1 | Out-Host
$pushed = ($LASTEXITCODE -eq 0)
git switch main 2>&1 | Out-Host
if ($pushed) {
    Write-Host ""
    Write-Host "SENT. The recordings are on GitHub as branch $branch - tell Claude it is done." -ForegroundColor Green
} else {
    Write-Host "Could not push the branch $branch - send a photo of this window." -ForegroundColor Red
}
