<#
.SYNOPSIS
  Registers the watcher and the search UI to start at logon (current user, no admin needed).
  Run:    powershell -ExecutionPolicy Bypass -File install_task.ps1
  Remove: powershell -ExecutionPolicy Bypass -File install_task.ps1 -Uninstall
#>
param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$names = @("VectorEmbed Watcher", "VectorEmbed Search")

if ($Uninstall) {
    foreach ($n in $names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    Write-Host "Removed."
    return
}

$root   = $PSScriptRoot
$python = (Get-Command python).Source
$pythonw = Join-Path (Split-Path $python) "pythonw.exe"
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found next to $python" }

# Critical: without these two flags Task Scheduler kills the task the moment you unplug the laptop,
# and the watcher must keep running on battery to record changes (it only *indexes* on AC).
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

$tasks = @(
    @{ Name = $names[0]; Args = "`"$root\watcher.py`"" },
    @{ Name = $names[1]; Args = "-m ui.app" }
)
foreach ($t in $tasks) {
    $action = New-ScheduledTaskAction -Execute $pythonw -Argument $t.Args -WorkingDirectory $root
    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Force | Out-Null
    Write-Host "Registered '$($t.Name)'"
}
Write-Host "Start now with:  Start-ScheduledTask -TaskName '$($names[0])'; Start-ScheduledTask -TaskName '$($names[1])'"
