# WarriorIQ - the always-on half of the auto-wake.
#
# Waking the machine and starting the analysis are two different problems.
# A magic packet brings the PC back in about eighteen seconds, but if nothing
# is watching the queue the fight then sits until the next scheduled drain,
# which is up to five minutes later. That wait was the whole delay.
#
# This script is the fix: a worker that never exits. A live process is
# suspended with the machine when it sleeps and resumes the instant it wakes,
# so the queue is polled about a second after the packet lands.
#
# Deliberately NOT done here: taking a sleep hold. worker.py takes one only
# while a fight is actually being analysed. If this script held sleep open the
# machine could never suspend at all, and the whole point is that it sleeps
# until a fight arrives.
#
# Registered by install-worker-service.ps1. To run it by hand for debugging:
#     .\deploy\run-worker.ps1

$ErrorActionPreference = 'Continue'

$root   = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$worker = Join-Path $root 'worker.py'
$logDir = Join-Path $root 'logs'
$log    = Join-Path $logDir 'worker-service.log'

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Write-Log($message) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Add-Content -Path $log -Value $line -Encoding utf8
}

# This process is meant to live for months, so the log needs a ceiling.
function Limit-Log {
    if (-not (Test-Path $log)) { return }
    if ((Get-Item $log).Length -lt 5MB) { return }
    $old = "$log.1"
    if (Test-Path $old) { Remove-Item $old -Force }
    Move-Item $log $old
    Write-Log 'log rotated'
}

if (-not (Test-Path $python)) { Write-Log "python not found at $python"; exit 1 }
if (-not (Test-Path $worker)) { Write-Log "worker.py not found at $worker"; exit 1 }

Write-Log '--- worker service starting ---'
Set-Location $root

# Restart on crash, backing off so a permanently broken install does not spin
# the CPU. Resets as soon as a run survives a couple of minutes, so the common
# case - a long healthy run that dies once - retries immediately.
$delay = 5
$max   = 300

while ($true) {
    Limit-Log
    $started = Get-Date
    try {
        & $python $worker 2>&1 | ForEach-Object { Write-Log $_ }
        $code = $LASTEXITCODE
    } catch {
        $code = -1
        Write-Log "worker threw: $($_.Exception.Message)"
    }

    $ran = [int]((Get-Date) - $started).TotalSeconds
    Write-Log "worker exited with code $code after ${ran}s"

    if ($ran -gt 120) { $delay = 5 }

    Write-Log "restarting in ${delay}s"
    Start-Sleep -Seconds $delay

    $delay = [Math]::Min($delay * 2, $max)
}
