<#
.SYNOPSIS
  Registers the watcher and the search UI to start at logon (current user, no admin needed).
  Thin wrapper over `ldf autostart`; the logic (and the battery flags) live in
  src\localdoc_finder\core\autostart.py so the app, the installer and this script share it.
  Run:    powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
  Remove: powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1 -Uninstall
#>
param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "python.exe not found in $root\.venv - create the venv first (uv sync)." }

$state = if ($Uninstall) { "off" } else { "on" }
& $python -m localdoc_finder.cli autostart $state
if ($LASTEXITCODE -ne 0) { throw "ldf autostart $state failed" }
if (-not $Uninstall) {
    Write-Host "Start now with:  Start-ScheduledTask -TaskName 'LocalDocFinder Watcher'; Start-ScheduledTask -TaskName 'LocalDocFinder Search'"
}
