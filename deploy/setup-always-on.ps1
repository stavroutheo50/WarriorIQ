# WarriorIQ - make this PC an always-on analysis machine. Run by double-clicking
# setup-always-on.bat in the project folder; it asks for administrator rights.
#
# What it does, and why:
#   1. installs the worker as a background service (install-worker-service.ps1):
#      it starts at boot, needs no one signed in, has no window to close by
#      mistake, and restarts itself within a minute if it ever crashes;
#   2. stops the PC sleeping while plugged in. Waking it by magic packet
#      worked most of the time; the time it did not, a fight waited 17 hours.
#      The screen can still switch off;
#   3. closes the old start-worker.bat window and removes its Startup entry,
#      so only the service runs.
#
# To undo: Unregister-ScheduledTask -TaskName 'WarriorIQ Analysis Worker' -Confirm:$false
# and set sleep back in Windows Settings -> System -> Power.

$ErrorActionPreference = 'Continue'

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Start-Process powershell -Verb RunAs -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"")
    exit
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$ok = $true

Write-Host 'Closing the old worker window, if it is open...'
Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -like '*start-worker.bat*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*worker.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# The old way to start the worker was an entry in the Startup folder that
# opened start-worker.bat at sign-in. With the service also running, that
# window found the worker already there and complained every ten seconds.
# Only entries that launch start-worker.bat are removed.
Write-Host 'Removing the old start-worker.bat entry from Startup, if there is one...'
$shell = New-Object -ComObject WScript.Shell
foreach ($folder in @([Environment]::GetFolderPath('Startup'), [Environment]::GetFolderPath('CommonStartup'))) {
    if (-not $folder -or -not (Test-Path $folder)) { continue }
    Get-ChildItem -LiteralPath $folder -File | ForEach-Object {
        $launches = $false
        if ($_.Extension -eq '.lnk') {
            $launches = ($shell.CreateShortcut($_.FullName).TargetPath -like '*start-worker.bat')
        } elseif ($_.Extension -in '.bat', '.cmd') {
            $launches = [bool](Select-String -LiteralPath $_.FullName -Pattern 'start-worker.bat' -SimpleMatch -Quiet)
        }
        if ($launches) { Remove-Item -LiteralPath $_.FullName -Force; Write-Host "  removed $($_.Name)" }
    }
}

Write-Host 'Installing the always-on worker service...'
& (Join-Path $PSScriptRoot 'install-worker-service.ps1')
if (-not $?) { $ok = $false }

Write-Host 'Stopping the PC from sleeping while plugged in...'
powercfg /change standby-timeout-ac 0
if ($LASTEXITCODE -ne 0) { $ok = $false }
powercfg /change hibernate-timeout-ac 0
if ($LASTEXITCODE -ne 0) { $ok = $false }

$state = (Get-ScheduledTask -TaskName 'WarriorIQ Analysis Worker' -ErrorAction SilentlyContinue).State
Write-Host ''
if ($ok -and $state -eq 'Running') {
    Write-Host 'DONE. The WarriorIQ worker now runs in the background, starts with Windows,' -ForegroundColor Green
    Write-Host 'restarts itself if it crashes, and the PC no longer sleeps while plugged in.' -ForegroundColor Green
    Write-Host 'You do not need start-worker.bat any more.'
} else {
    Write-Host "NOT EVERYTHING WORKED (service state: $state). Send a photo of this window." -ForegroundColor Red
}
Write-Host ''
Read-Host 'Press Enter to close'
