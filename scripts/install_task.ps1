<#
.SYNOPSIS
  Registers the watcher and the search UI to start at logon (current user, no admin needed).
  Thin wrapper over `ve autostart`; the logic (and the battery flags) live in
  src\vector_embed\core\autostart.py so the app, the installer and this script share it.
  Run:    powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
  Remove: powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1 -Uninstall
#>
param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "python.exe not found in $root\.venv - create the venv first (uv sync)." }

$state = if ($Uninstall) { "off" } else { "on" }
& $python -m vector_embed.cli autostart $state
if ($LASTEXITCODE -ne 0) { throw "ve autostart $state failed" }
if (-not $Uninstall) {
    Write-Host "Start now with:  Start-ScheduledTask -TaskName 'VectorEmbed Watcher'; Start-ScheduledTask -TaskName 'VectorEmbed Search'"
}
