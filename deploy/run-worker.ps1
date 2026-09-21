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

# Someone else already doing the job is not a crash.
#
# start-worker.bat exists as well, so both supervisors can be up at once - and
# then this one started a worker every five minutes that immediately declined
# the lock, wrote an ERROR, and exited. Two bad outcomes: a log that reads like
# a fault when nothing is wrong, and a queue left up to five minutes from being
# picked up if the other worker ever stopped.
#
# So while another live worker holds the lock, stand by instead: no process is
# started, one line is logged on the way in and one on the way out, and the
# check runs often enough to take over quickly.
$standbyPoll = 30
$standingBy  = $false

# worker.py's own guard is what actually decides - this only avoids starting a
# process that would immediately decline. It mirrors LOCK_STALE_SECONDS there.
# If the two ever drift apart the worst case is the old behaviour: a worker
# starts, declines, and exits with EXIT_ANOTHER_WORKER_IS_RUNNING below.
$lockStaleSeconds = 90
$lockFile = Join-Path $root 'worker.lock'

function Get-LiveLockHolder {
    if (-not (Test-Path $lockFile)) { return $null }
    try { $parts = (Get-Content $lockFile -ErrorAction Stop) -split '\s+' } catch { return $null }
    if ($parts.Count -lt 2) { return $null }
    $holderPid = 0; $beat = 0.0
    if (-not [int]::TryParse($parts[0], [ref]$holderPid)) { return $null }
    if (-not [double]::TryParse($parts[1], [ref]$beat)) { return $null }
    $age = ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) - $beat
    if ($age -ge $lockStaleSeconds) { return $null }
    if (-not (Get-Process -Id $holderPid -ErrorAction SilentlyContinue)) { return $null }
    return $holderPid
}

while ($true) {
    Limit-Log

    $holder = Get-LiveLockHolder
    if ($null -ne $holder) {
        if (-not $standingBy) {
            Write-Log "another worker (process $holder) holds the lock; standing by"
            $standingBy = $true
        }
        Start-Sleep -Seconds $standbyPoll
        continue
    }
    if ($standingBy) {
        Write-Log 'the lock is free again; starting the worker'
        $standingBy = $false
        $delay = 5
    }

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

    # It raced someone to the lock between the check above and starting. Same
    # situation, so say so once and stand by rather than counting it a crash.
    if ($code -eq 3) {
        if (-not $standingBy) {
            Write-Log 'another worker claimed the lock first; standing by'
            $standingBy = $true
        }
        Start-Sleep -Seconds $standbyPoll
        continue
    }

    if ($ran -gt 120) { $delay = 5 }

    Write-Log "restarting in ${delay}s"
    Start-Sleep -Seconds $delay

    $delay = [Math]::Min($delay * 2, $max)
}
