# WarriorIQ - check that this machine can actually be woken for a fight.
#
# There are two halves to waking the analysis GPU and they fail differently:
#
#   1. This machine's own configuration - the network card must be allowed to
#      wake the system from sleep, and the scheduled drain must be allowed to
#      wake it on a timer. Both are checked here, because both are local and
#      both are things Windows silently turns off after a driver update.
#
#   2. Whether a magic packet sent from the website's server survives the trip
#      through the home router. No script on this machine can test that: the
#      packet either arrives or it does not, and the only honest answer comes
#      from the observed wake latency at /ready after a real fight.
#
# Run it from an elevated PowerShell:  .\deploy\wake-selftest.ps1

$ErrorActionPreference = 'Continue'

function Say($label, $ok, $detail) {
    $mark = if ($ok -eq $true) { '[ ok ]' } elseif ($ok -eq $false) { '[FAIL]' } else { '[ ?? ]' }
    Write-Host ("{0} {1}" -f $mark, $label)
    if ($detail) { Write-Host ("       {0}" -f $detail) }
}

Write-Host ''
Write-Host 'WarriorIQ wake self-test'
Write-Host '========================'
Write-Host ''

# --- 1. Which adapter carries the traffic, and what is its MAC ---------------
$adapter = Get-NetAdapter -Physical | Where-Object { $_.Status -eq 'Up' } | Select-Object -First 1
if (-not $adapter) {
    Say 'Active network adapter' $false 'No physical adapter is up. Nothing can be woken.'
    exit 1
}
Say 'Active network adapter' $true ("{0} - MAC {1}" -f $adapter.Name, $adapter.MacAddress)
Write-Host ("       Set WARRIORIQ_WOL_MAC to this MAC on the website." )
Write-Host ''

# --- 2. Is the card allowed to wake the machine at all -----------------------
try {
    $power = Get-NetAdapterPowerManagement -Name $adapter.Name
    Say 'Wake on magic packet' ($power.WakeOnMagicPacket -eq 'Enabled') ("WakeOnMagicPacket = {0}" -f $power.WakeOnMagicPacket)
    if ($power.WakeOnMagicPacket -ne 'Enabled') {
        Write-Host '       Fix: Device Manager > the adapter > Power Management >'
        Write-Host '            tick "Allow this device to wake the computer".'
    }
    Say 'Device may wake the computer' ($power.DeviceSleepOnDisconnect -ne $null) ''
} catch {
    Say 'Wake on magic packet' $null 'Could not read adapter power settings (run as Administrator).'
}
Write-Host ''

# --- 3. Does the firmware still allow wake from the sleep state in use -------
$wakeArmed = (powercfg /devicequery wake_armed) 2>$null
if ($wakeArmed -and ($wakeArmed -join ' ') -match [regex]::Escape($adapter.InterfaceDescription)) {
    Say 'Adapter is armed to wake the system' $true ''
} else {
    Say 'Adapter is armed to wake the system' $false 'powercfg /devicequery wake_armed does not list this adapter.'
    Write-Host ("       Fix: powercfg /deviceenablewake `"{0}`"" -f $adapter.InterfaceDescription)
}

# Fast Startup leaves the machine in a hybrid shutdown that many cards will not
# wake from, and it is on by default on Windows 11.
$hiberboot = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power' -Name HiberbootEnabled -ErrorAction SilentlyContinue).HiberbootEnabled
if ($hiberboot -eq 1) {
    Say 'Fast Startup disabled' $false 'Fast Startup is ON; a shut-down machine will not wake.'
    Write-Host '       Fix: powercfg /hibernate off   (or turn off Fast Startup in Power Options)'
} else {
    Say 'Fast Startup disabled' $true 'Sleep and shutdown can both be woken.'
}
Write-Host ''

# --- 4. The guaranteed path: the scheduled drain -----------------------------
$task = Get-ScheduledTask -TaskName 'WarriorIQ*' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($task) {
    $settings = $task.Settings
    Say 'Scheduled drain task exists' $true $task.TaskName
    Say 'Task may wake the computer' ($settings.WakeToRun -eq $true) ("WakeToRun = {0}" -f $settings.WakeToRun)
    if (-not $settings.WakeToRun) {
        Write-Host '       Fix: Task Scheduler > the task > Conditions >'
        Write-Host '            tick "Wake the computer to run this task".'
    }
    $trigger = $task.Triggers | Select-Object -First 1
    if ($trigger -and $trigger.Repetition -and $trigger.Repetition.Interval) {
        Say 'Repeat interval' $true ("{0} - this is the longest a fight can wait" -f $trigger.Repetition.Interval)
    } else {
        Say 'Repeat interval' $false 'No repetition set; the queue would only drain once.'
    }
} else {
    Say 'Scheduled drain task exists' $false 'No WarriorIQ task found. This is the guaranteed wake path.'
    Write-Host '       Fix: create a task running deploy\drain-queue.ps1 every 5 minutes,'
    Write-Host '            with "Wake the computer to run this task" ticked.'
}

Write-Host ''
Write-Host 'The half this cannot test'
Write-Host '-------------------------'
Write-Host 'Whether a magic packet from the website survives the home router is not'
Write-Host 'knowable from here. After the next real fight, open:'
Write-Host '    https://warrioriq.eu/ready'
Write-Host 'and read the "wake" section:'
Write-Host '    median_seconds well under the drain interval -> the packet is arriving'
Write-Host '    median_seconds near the drain interval       -> the packet is being dropped,'
Write-Host '                                                   and the timer is doing the work'
Write-Host ''
