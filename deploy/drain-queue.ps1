# WarriorIQ - drain the analysis queue, then let the machine go back to sleep.
#
# Run by Windows Task Scheduler on a repeating schedule with "Wake the computer
# to run this task". The machine wakes from S3, claims whatever fights are
# waiting, analyses them on the local GPU and exits; Windows' own idle timeout
# returns it to sleep.
#
# It deliberately does NOT force sleep afterwards. Forcing it would suspend the
# machine mid-use whenever the timer happened to fire while someone was working
# on it; letting the normal idle timer do the job is both safer and correct.

$ErrorActionPreference = 'Stop'

$root   = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$worker = Join-Path $root 'worker.py'
$logDir = Join-Path $root 'logs'
$log    = Join-Path $logDir 'drain-queue.log'

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Write-Log($message) {
    $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Add-Content -Path $log -Value $line -Encoding utf8
}

if (-not (Test-Path $python)) { Write-Log "python not found at $python"; exit 1 }
if (-not (Test-Path $worker)) { Write-Log "worker.py not found at $worker"; exit 1 }

# Keep the machine awake for as long as the analysis actually runs. Without
# this the idle timer can suspend the box mid-fight, and the job's lease then
# has to expire before anyone else can pick it up.
$power = Add-Type -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -Name 'Power' -Namespace 'WarriorIQ' -PassThru

$ES_CONTINUOUS = [uint32]'0x80000000'
$ES_SYSTEM_REQUIRED = [uint32]'0x00000001'
[void]$power::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

Write-Log 'waking - draining the WarriorIQ queue'
try {
    Push-Location $root
    # --once claims fights until the queue is empty, then exits. A non-zero code
    # means the server was unreachable, which the next scheduled run retries.
    & $python $worker --once 2>&1 | ForEach-Object { Write-Log $_ }
    $code = $LASTEXITCODE
    Write-Log "worker finished with exit code $code"
} catch {
    Write-Log "drain failed: $($_.Exception.Message)"
} finally {
    Pop-Location
    # Release the sleep block so the normal idle timeout can suspend again.
    [void]$power::SetThreadExecutionState($ES_CONTINUOUS)
}
