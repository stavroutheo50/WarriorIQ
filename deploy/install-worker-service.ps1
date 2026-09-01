# WarriorIQ - keep the analysis worker running so a fight never waits for a timer.
#
# The problem this solves: waking the machine is not the same as starting the
# worker. A magic packet brings the PC back in about eighteen seconds, but if
# nothing is polling the queue the fight then sits until the next scheduled
# drain - up to five minutes later. That was the whole delay.
#
# The fix is a worker that simply never stops. A running process is suspended
# with the machine when it sleeps and resumes the moment it wakes, so the queue
# is polled within a second of the magic packet landing. worker.py takes a
# Windows sleep hold only while a fight is actually being analysed, so an idle
# worker still lets the machine sleep normally.
#
# Run once, from an elevated PowerShell:
#     .\deploy\install-worker-service.ps1
#
# To remove it:
#     Unregister-ScheduledTask -TaskName 'WarriorIQ Analysis Worker' -Confirm:$false

$ErrorActionPreference = 'Stop'

$taskName = 'WarriorIQ Analysis Worker'
$root     = Split-Path -Parent $PSScriptRoot
$python   = Join-Path $root '.venv\Scripts\pythonw.exe'
$fallback = Join-Path $root '.venv\Scripts\python.exe'
$runner   = Join-Path $PSScriptRoot 'run-worker.ps1'

if (-not (Test-Path $python)) {
    if (-not (Test-Path $fallback)) { throw "No Python found in $root\.venv\Scripts" }
    $python = $fallback
}
if (-not (Test-Path $runner)) { throw "Missing $runner" }

Write-Host "Installing '$taskName'"
Write-Host "  project : $root"
Write-Host "  python  : $python"

# Replace any previous copy so re-running is safe.
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host '  removing the previous registration'
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`"" `
    -WorkingDirectory $root

# At boot, and again on logon, so the worker is up whether or not anyone signs
# in. Resume from sleep needs no trigger: the process is still alive.
$triggers = @(
    (New-ScheduledTaskTrigger -AtStartup),
    (New-ScheduledTaskTrigger -AtLogOn)
)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -RestartCount 999 `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

# SYSTEM so it runs with no one logged in. The worker needs no desktop.
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers `
    -Settings $settings -Principal $principal `
    -Description 'Runs the WarriorIQ analysis worker continuously so a queued fight starts within seconds of the machine waking.' | Out-Null

Write-Host '  registered'
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
$state = (Get-ScheduledTask -TaskName $taskName).State
Write-Host "  state: $state"

Write-Host ''
Write-Host 'Done. What changes:'
Write-Host '  - the worker starts at boot and stays running'
Write-Host '  - it is suspended with the machine on sleep and resumes on wake'
Write-Host '  - a fight now starts seconds after the wake, not on the 5-minute drain'
Write-Host '  - the machine can still sleep when idle; the hold is taken only mid-analysis'
Write-Host ''
Write-Host "Log: $root\logs\worker-service.log"
Write-Host 'Keep the existing 5-minute drain task as the backstop - it costs nothing'
Write-Host 'and it still collects a fight if this worker is ever not running.'
