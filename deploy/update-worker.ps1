# WarriorIQ - bring this PC's analysis worker up to the code on GitHub main.
# Run by double-clicking update-worker.bat in the worker's folder.
#
# Why: the website refuses a worker whose analysis code differs from its own
# (core/build_info.analysis_fingerprint). On 2026-10-14 the website was
# updated, `git pull` ran in a different folder from the one the worker ran
# from, and fights sat refused for twenty minutes. This updates the folder the
# worker actually runs from, in one step:
#   1. stops the workers running now (the start-worker.bat window and any
#      python worker.py), so no old code keeps asking for fights;
#   2. downloads main from GitHub into a temporary folder;
#   3. replaces core/ whole - a leftover file in it changes the fingerprint -
#      keeping the previous one as core_previous; copies app, tools, deploy,
#      models and the top-level scripts over the rest;
#   4. prints the new fingerprint and starts the worker again.
# .env, .venv, outputs and downloaded model weights are never touched.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "No .venv in $root - run this from the folder the worker runs from."
}

Write-Host 'Stopping the workers running now...'
Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -like '*start-worker.bat*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*worker.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
$service = Get-ScheduledTask -TaskName 'WarriorIQ Analysis Worker' -ErrorAction SilentlyContinue
if ($service) { Stop-ScheduledTask -TaskName 'WarriorIQ Analysis Worker' }

Write-Host 'Downloading the latest code from GitHub...'
$source = Join-Path $env:TEMP 'warrioriq_main'
if (Test-Path $source) { Remove-Item $source -Recurse -Force }
git clone --quiet --depth 1 --branch main https://github.com/stavroutheo50/WarriorIQ.git $source
if ($LASTEXITCODE -ne 0) { throw 'git clone failed - is Git installed and signed in to GitHub?' }

Write-Host 'Copying it into this folder...'
$previous = Join-Path $root 'core_previous'
if (Test-Path $previous) { Remove-Item $previous -Recurse -Force }
if (Test-Path (Join-Path $root 'core')) { Rename-Item (Join-Path $root 'core') 'core_previous' }
# robocopy exit codes below 8 mean success.
robocopy (Join-Path $source 'core') (Join-Path $root 'core') /MIR /NJH /NJS /NFL /NDL | Out-Null
if ($LASTEXITCODE -ge 8) { throw 'Copying core failed.' }
foreach ($folder in 'app', 'tools', 'deploy', 'models') {
    robocopy (Join-Path $source $folder) (Join-Path $root $folder) /E /NJH /NJS /NFL /NDL | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying $folder failed." }
}
robocopy $source $root *.py *.bat requirements*.txt /XF update-worker.bat /NJH /NJS /NFL /NDL | Out-Null
if ($LASTEXITCODE -ge 8) { throw 'Copying the top-level files failed.' }

$commit = git -C $source rev-parse --short HEAD
$fingerprint = & $python -c "from core.build_info import analysis_fingerprint as f; print(f())"
Write-Host ''
Write-Host "Worker updated to GitHub main $commit (code fingerprint $fingerprint)."
Write-Host 'The website must be deployed from the same commit, or it will refuse this worker.'

if ($service) {
    Start-ScheduledTask -TaskName 'WarriorIQ Analysis Worker'
    Write-Host 'The background worker has been restarted.'
} else {
    Start-Process (Join-Path $root 'start-worker.bat') -WorkingDirectory $root
    Write-Host 'The worker window has been started.'
}
