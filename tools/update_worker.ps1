# WarriorIQ - bring the analysis PC up to date in one go.
#
# Why this exists: the website refuses an analysis worker whose code is older
# than its own (core/build_info.ANALYSIS_VERSION). A PC left on old code is
# turned away and sits idle while uploads wait. This script brings the PC
# level with GitHub and installs the GPU speed-ups, in one run:
#
#   1. stops the running worker (start-worker.bat window or the scheduled task);
#   2. saves any local edits to their own branch - committed, and pushed to
#      GitHub when this PC can push - so nothing is lost;
#   3. switches to main and pulls;
#   4. installs the requirements, TensorRT and the RTMPose GPU runtime into .venv;
#   5. builds the TensorRT pose engine for this GPU (once; reused afterwards);
#   6. checks the GPU, TensorRT and RTMPose actually work;
#   7. starts the worker again.
#
# Run it from the project folder:
#     powershell -ExecutionPolicy Bypass -File tools\update_worker.ps1
# On a PC still on old code, which does not have this file yet, fetch it from
# GitHub and run that copy against the project folder (one line, in the folder):
#     git fetch origin main; git show origin/main:tools/update_worker.ps1 > $env:TEMP\wiq_update.ps1; powershell -ExecutionPolicy Bypass -File $env:TEMP\wiq_update.ps1 -Root "$PWD"
# Everything it does is written to logs\update-worker.log.

param([string]$Root = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Continue'
$root   = (Resolve-Path $Root).Path
if (-not (Test-Path (Join-Path $root 'worker.py'))) { Write-Host "STOPPED: $root is not the WarriorIQ folder (no worker.py)"; exit 1 }
$python = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
$log    = Join-Path $logDir 'update-worker.log'
$task   = 'WarriorIQ Analysis Worker'
Set-Location $root
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Say($message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $message"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

function Run($what, [scriptblock]$command) {
    Say $what
    & $command 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    return ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE)
}

# Python code goes through a file, never `python -c "..."`: Windows PowerShell
# 5.1 strips the double quotes inside an argument to a native program, so
# print("ENGINE") arrived as print(ENGINE) and every step that used it failed.
function Run-Python($what, [string]$code) {
    $step = Join-Path $logDir 'update-worker-step.py'
    $prelude = "import os, sys`nsys.path.insert(0, os.getcwd())`n"
    [System.IO.File]::WriteAllText($step, $prelude + $code, (New-Object System.Text.UTF8Encoding $false))
    return (Run $what { & $python $step })
}

function Stop-Here($why) {
    Say "STOPPED: $why"
    Say "Nothing after this step was done. Send logs\update-worker.log to get it fixed."
    exit 1
}

if (-not (Test-Path $python)) { Stop-Here "the project .venv is missing ($python)" }
$problems = @()

# 1. Stop the worker, so files are not replaced under a running analysis.
$scheduled = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
if ($scheduled) { Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue; Say "Stopped scheduled task '$task'" }
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*worker.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Say "Stopped worker process $($_.ProcessId)" }
Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -like '*start-worker.bat*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Say "Closed start-worker.bat window $($_.ProcessId)" }

# 2. Keep local edits: their own branch, never thrown away.
$changes = git status --porcelain
if ($changes) {
    $branch = "pc-local-changes-$(Get-Date -Format 'yyyy-MM-dd-HHmm')"
    if (-not (Run "Saving local changes to branch $branch" { git switch -c $branch })) { Stop-Here "could not create branch $branch" }
    Run "Adding changed files" { git add -A } | Out-Null
    if (-not (Run "Committing local changes" { git -c user.name="WarriorIQ PC" -c user.email="worker@warrioriq.eu" commit -m "Analysis PC local changes, saved before updating to main" })) {
        Stop-Here "could not commit the local changes (they are untouched on branch $branch)"
    }
    if (Run "Backing up $branch to GitHub" { git push -u origin $branch }) {
        Say "Local changes are on GitHub as branch $branch"
    } else {
        Say "Could not push $branch - the changes are still saved on this PC in that branch"
    }
}

# 3. Latest main.
if (-not (Run "Switching to main" { git switch main })) { Stop-Here "could not switch to main" }
if (-not (Run "Pulling the latest code" { git pull --ff-only origin main })) { Stop-Here "git pull failed" }

# 4. Packages. PyTorch must be the CUDA 12.8 build for an RTX 50-series card.
Run "Upgrading pip" { & $python -m pip install --upgrade pip } | Out-Null
$torchOk = & $python -c "import torch; print(torch.version.cuda or 0)" 2>$null
if (-not ("$torchOk" -match '^12\.(8|9)|^13')) {
    if (-not (Run "Installing PyTorch for CUDA 12.8" { & $python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 })) { Stop-Here "PyTorch install failed" }
}
if (-not (Run "Installing requirements.txt" { & $python -m pip install -r requirements.txt })) { Stop-Here "requirements.txt install failed" }
if (-not (Run "Installing TensorRT" { & $python -m pip install -r requirements-trt-cuda12.txt })) { Stop-Here "TensorRT install failed" }
# The CPU and GPU onnxruntime packages install the same module and must not
# be installed together (requirements-rtm-cuda12.txt). The GPU package also
# runs on the CPU, so nothing that used the CPU one loses anything.
$cpuOrt = & $python -m pip show onnxruntime 2>$null
if ($cpuOrt) { Run "Removing CPU-only onnxruntime" { & $python -m pip uninstall -y onnxruntime } | Out-Null }
if (-not (Run "Installing the RTMPose GPU runtime" { & $python -m pip install -r requirements-rtm-cuda12.txt })) { Stop-Here "onnxruntime-gpu install failed" }
if (-not (Run "Installing RTMPose" { & $python -m pip install --no-deps rtmlib==0.0.16 })) { Stop-Here "rtmlib install failed" }

# 5. TensorRT engine for this GPU, built once and reused (core/trt_engine.py).
# An engine that exists is not one that loads: one built by another TensorRT
# version is rejected at the first frame, and every analysis then ran on
# PyTorch while "ALL CHECKS PASSED" (2026-10-06). So each engine is run once
# here. One that cannot load is renamed to .rejected (kept, not deleted), and
# this GPU's own engine is rebuilt if it is the one refused.
$build = @'
import logging
from pathlib import Path
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(message)s")
from core.config import MODELS, SETTINGS
from core.trt_engine import ensure_pose_engine
from ultralytics import YOLO

def loads(path):
    try:
        result = YOLO(str(path), task="pose").predict(np.zeros((640, 640, 3), np.uint8), device=0, imgsz=640, verbose=False)
        return result[0].keypoints is not None
    except Exception as exc:
        print("ENGINE CANNOT LOAD", path, type(exc).__name__, str(exc)[:160])
        return False

def set_aside(path):
    target = Path(str(path) + ".rejected")
    if target.exists():
        target.unlink()
    Path(path).rename(target)
    print("RENAMED", path, "->", target.name)

engine = ensure_pose_engine(MODELS)
if engine and not loads(engine):
    set_aside(engine)
    engine = ensure_pose_engine(MODELS)
    if engine and not loads(engine):
        engine = None
configured = Path(SETTINGS.pose_model_engine)
if engine and configured.exists() and configured.resolve() != Path(engine).resolve() and not loads(configured):
    set_aside(configured)
print("ENGINE", engine)
raise SystemExit(0 if engine else 1)
'@
if (-not (Run-Python "Building the TensorRT engine (first time takes several minutes)" $build)) {
    Say "TensorRT engine build failed - analyses still run, just slower; the reason is above"
    $problems += "TensorRT engine not built"
}

# 6. Prove it works rather than assume.
$check = @'
import torch  # before onnxruntime: see core/rtm_pose.py
print("GPU", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0), "CUDA", torch.version.cuda)
from core.build_info import ANALYSIS_VERSION, build_commit
print("analysis version", ANALYSIS_VERSION, "commit", build_commit())
import numpy as np
from core.pose_tracker import PoseTracker
tracker = PoseTracker()
tracker.warmup(np.zeros((720, 1280, 3), np.uint8))
print("pose model", tracker.model_path)
assert tracker.model_path.endswith(".engine"), "the TensorRT engine did not load - analyses would run on PyTorch, slower"
# A pose model run as plain detection finds people but no body keypoints, and
# the analysis has nothing to measure (2026-10-06): prove keypoints come back.
probe = tracker.model.predict(np.zeros((720, 1280, 3), np.uint8), device=0, imgsz=640, verbose=False)
print("pose task", tracker.model.task, "keypoints returned", probe[0].keypoints is not None)
assert probe[0].keypoints is not None, "the pose model returned no body keypoints"
from core import rtm_pose
refiner = rtm_pose._get()
assert refiner is not None, "RTMPose did not load - see the warning above"
session = getattr(refiner.model, "session", None)
providers = session.get_providers() if session is not None else []
print("RTMPose providers", providers)
assert "CUDAExecutionProvider" in providers, "RTMPose would run on the CPU"
print("ALL CHECKS PASSED")
'@
if (-not (Run-Python "Checking GPU, RTMPose and the pose model" $check)) {
    Say "The check reported a problem above - the worker is restarted anyway; send logs\update-worker.log"
    $problems += "GPU / RTMPose / pose model check failed"
}

# 7. Start the worker again, the way it was being run.
if ($scheduled) {
    Start-ScheduledTask -TaskName $task
    Say "Started scheduled task '$task'"
} else {
    Start-Process -FilePath (Join-Path $root 'start-worker.bat') -WorkingDirectory $root
    Say "Started start-worker.bat in a new window"
}
if ($problems.Count -eq 0) {
    Say "Done. The worker is on the latest code. ALL CHECKS PASSED"
} else {
    Say "Done, BUT NOT EVERYTHING WORKED: $($problems -join '; '). Send logs\update-worker.log"
}
