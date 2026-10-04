<#
.SYNOPSIS
  Builds the Windows installer: PyInstaller one-folder app, then Velopack Setup.exe + update packages.
  Needs the .NET SDK on the build machine for `vpk` (end users need nothing). vpk is found on PATH,
  or in .tools\ (install there with:  dotnet tool install vpk --tool-path .tools).

  Output (Releases\):  VectorEmbed-win-Setup.exe, the full package and, when an earlier release can
  be downloaded, a delta package. Unsigned builds show a SmartScreen warning.

  Examples:
    powershell -ExecutionPolicy Bypass -File scripts\build.ps1
    powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -RepoUrl https://github.com/OWNER/REPO -Upload
#>
param(
    [string]$Version,          # default: the version in pyproject.toml
    [string]$RepoUrl = "",     # GitHub repo that hosts releases; baked in as the update source
    [switch]$Upload            # publish to GitHub Releases (needs `gh auth login` or GITHUB_TOKEN)
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "No .venv found. Create it first (uv sync)." }
function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe failed with exit code $LASTEXITCODE" }
}

if (-not $Version) {
    $Version = & $python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
}
$sourceFile = Join-Path $root "src\vector_embed\update_source.txt"

Step "1/5 Build dependencies"
if (Get-Command uv -ErrorAction SilentlyContinue) { Run "uv" @("sync", "--group", "build") }
else { Run $python @("-m", "pip", "install", "--quiet", "pyinstaller>=6.10") }

Step "2/5 PyInstaller (version $Version)"
if ($RepoUrl) { Set-Content -Path $sourceFile -Value $RepoUrl -NoNewline -Encoding ascii }
elseif (Test-Path $sourceFile) { Remove-Item $sourceFile }
Run $python @("packaging\make_icon.py")
Run $python @("-m", "PyInstaller", "packaging\vector_embed.spec", "--noconfirm", "--clean",
    "--distpath", "dist", "--workpath", "build")

Step "3/5 Smoke test: ve.exe doctor"
# doctor's exit code reflects this machine's setup (Ollama, models); we only need it to run.
# Windows PowerShell 5.1 turns native stderr into terminating errors under "Stop"; relax it here.
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$smoke = & "dist\VectorEmbed\ve.exe" doctor 2>&1 | Out-String
$ErrorActionPreference = $previous
Write-Host $smoke
if ($smoke -notmatch "\[ok\] python" -or $smoke -match "Traceback") {
    throw "The packaged ve.exe doctor did not run cleanly."
}
# Proves the extractor modules were bundled (they are found by scanning the package at run time).
if ($smoke -notmatch "\[ok\] extractors: .*pdf.*") {
    throw "The packaged build is missing document extractors."
}

Step "4/5 Velopack pack"
$vpk = (Get-Command vpk -ErrorAction SilentlyContinue).Source
if (-not $vpk) { $vpk = Join-Path $root ".tools\vpk.exe" }
if (-not (Test-Path $vpk)) {
    throw "vpk not found. Install the .NET SDK, then: dotnet tool install vpk --tool-path .tools"
}
if ($RepoUrl) {
    # Fetching the previous release lets vpk build a small delta package.
    $downloadArgs = @("download", "github", "--repoUrl", $RepoUrl, "--outputDir", "Releases")
    if ($env:GITHUB_TOKEN) { $downloadArgs += @("--token", $env:GITHUB_TOKEN) }
    & $vpk @downloadArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "No earlier release to base a delta on (first release?)." }
}
Run $vpk @("pack", "--packId", "VectorEmbed", "--packVersion", $Version,
    "--packDir", "dist\VectorEmbed", "--mainExe", "VectorEmbed.exe",
    "--packTitle", "Vector Embed", "--runtime", "win-x64", "--icon", "packaging\vector_embed.ico",
    "--outputDir", "Releases")

Step "5/5 Publish"
if ($Upload) {
    if (-not $RepoUrl) { throw "-Upload needs -RepoUrl." }
    $uploadArgs = @("upload", "github", "--repoUrl", $RepoUrl, "--outputDir", "Releases",
        "--tag", "v$Version", "--releaseName", "Vector Embed $Version", "--publish")
    if ($env:GITHUB_TOKEN) { $uploadArgs += @("--token", $env:GITHUB_TOKEN) }
    Run $vpk $uploadArgs
} else {
    Write-Host "Skipped (add -Upload -RepoUrl <repo> to publish)."
}
Write-Host "`nDone. Installer: Releases\VectorEmbed-win-Setup.exe" -ForegroundColor Green
