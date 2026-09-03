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

# A continuously running worker already claims fights within a second, so this
# scheduled drain has nothing to do and must not start a second one: two
# workers means the pose model loaded twice on one GPU, which is how a machine
# ended up holding 3.3GB doing nothing. The drain stays installed as the
# fallback for when the continuous worker is not running.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'worker\.py' -and $_.CommandLine -notmatch '--once' }
if ($running) {
    Write-Log "continuous worker already running (pid $($running[0].ProcessId)); nothing to drain"
    exit 0
}

Write-Log 'waking - draining the WarriorIQ queue'
try {
    Push-Location $root
    # --once claims fights until the queue is empty, then exits. A non-zero code
    # means the server was unreachable, which the next scheduled run retries.
    # $ErrorActionPreference is deliberately relaxed across this one call.
    #
    # In Windows PowerShell 5.1, "2>&1" on a NATIVE executable wraps every
    # stderr line in an ErrorRecord. Python's logging writes INFO to stderr, so
    # under 'Stop' the worker's own first log line - "Worker running analysis
    # code ..." - was raised as a terminating error and caught below. Every
    # scheduled drain aborted on that line before analysing anything, and the
    # log recorded "drain failed:" followed by an ordinary INFO message.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python $worker --once 2>&1 | ForEach-Object { Write-Log "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    Write-Log "worker finished with exit code $code"
} catch {
    Write-Log "drain failed: $($_.Exception.Message)"
} finally {
    Pop-Location
    # Release the sleep block so the normal idle timeout can suspend again.
    [void]$power::SetThreadExecutionState($ES_CONTINUOUS)
}
