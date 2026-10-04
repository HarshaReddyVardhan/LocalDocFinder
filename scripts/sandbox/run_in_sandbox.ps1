# Runs inside Windows Sandbox: install, check, launch, uninstall. Results go to C:\Test\result.txt
# (the mapped scripts\sandbox folder on the host). The wizard itself needs a human: after this
# script finishes, the app is left running so you can click through it.
$ErrorActionPreference = "Continue"
$log = "C:\Test\result.txt"
Set-Content $log "LocalDoc Finder sandbox test $(Get-Date -Format s)"
function Note($text) { Add-Content $log $text; Write-Host $text }
function Check($name, $ok) { Note ("[{0}] {1}" -f ($(if ($ok) { "PASS" } else { "FAIL" }), $name)) }

Check "no Python on this machine" (-not (Get-Command python -ErrorAction SilentlyContinue))
Check "no Ollama on this machine" (-not (Get-Command ollama -ErrorAction SilentlyContinue))

$setup = "C:\Release\LocalDocFinder-win-Setup.exe"
Check "Setup.exe present" (Test-Path $setup)
Start-Process $setup -ArgumentList "--silent" -Wait
Start-Sleep -Seconds 10   # Velopack installs, runs the after-install hook, then launches the app

$root = Join-Path $env:LOCALAPPDATA "LocalDocFinder"
$dataRoot = Join-Path $env:LOCALAPPDATA "LocalDocFinderData"
$exe = Join-Path $root "current\LocalDocFinder.exe"
$ldf = Join-Path $root "current\ldf.exe"
Check "installed per-user, no admin ($exe)" (Test-Path $exe)
Check "ldf.exe present" (Test-Path $ldf)

$tasks = Get-ScheduledTask -TaskName "LocalDocFinder*" -ErrorAction SilentlyContinue
Check "startup tasks registered (Watcher + Search)" (($tasks | Measure-Object).Count -eq 2)
foreach ($t in $tasks) {
    $s = $t.Settings
    Check "$($t.TaskName): runs on battery" ($s.DisallowStartIfOnBatteries -eq $false -and $s.StopIfGoingOnBatteries -eq $false)
}

$doctor = & $ldf doctor 2>&1 | Out-String
Add-Content $log $doctor
Check "ldf.exe doctor runs (python line)" ($doctor -match "\[ok\] python")
Check "doctor reports Ollama missing" ($doctor -match "\[FAIL\] ollama")
Check "doctor reports setup not run" ($doctor -match "\[FAIL\] setup")

$app = Get-Process LocalDocFinder -ErrorAction SilentlyContinue
Check "tray app is running after install" (($app | Measure-Object).Count -ge 1)
Note "Now click through the setup wizard by hand (Ollama install consent, model picks, download)."
Note "When done, press Enter in the PowerShell window to run the uninstall check."
Read-Host "Press Enter after you finished the wizard"

$update = Join-Path $root "Update.exe"
Start-Process $update -ArgumentList "--uninstall --silent" -Wait
Start-Sleep -Seconds 8
Check "app folder removed after uninstall" (-not (Test-Path $exe))
Check "startup tasks removed after uninstall" (($(Get-ScheduledTask -TaskName "LocalDocFinder*" -ErrorAction SilentlyContinue) | Measure-Object).Count -eq 0)
Check "no LocalDocFinder.exe still running" (-not (Get-Process LocalDocFinder -ErrorAction SilentlyContinue))
Check "user data survives uninstall" (Test-Path (Join-Path $dataRoot state.sqlite))
Note "Done."
