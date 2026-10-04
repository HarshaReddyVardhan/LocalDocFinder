<#
.SYNOPSIS
  One-line installer: downloads the latest LocalDocFinder-win-Setup.exe from GitHub Releases and runs it.

  irm https://raw.githubusercontent.com/HarshaReddyVardhan/LocalDocFinder/main/scripts/install.ps1 | iex

  To pin a version:  & ([scriptblock]::Create((irm <url>))) -Version 0.2.0
#>
param(
    [string]$Repo = "HarshaReddyVardhan/LocalDocFinder",
    [string]$Version = ""   # default: latest release
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$api = if ($Version) { "https://api.github.com/repos/$Repo/releases/tags/v$Version" }
       else { "https://api.github.com/repos/$Repo/releases/latest" }
$release = Invoke-RestMethod -Uri $api -Headers @{ "User-Agent" = "LocalDocFinder-installer" }
$asset = $release.assets | Where-Object { $_.name -eq "LocalDocFinder-win-Setup.exe" } | Select-Object -First 1
if (-not $asset) { throw "Release $($release.tag_name) has no LocalDocFinder-win-Setup.exe." }

$target = Join-Path $env:TEMP "LocalDocFinder-win-Setup.exe"
Write-Host "Downloading LocalDoc Finder $($release.tag_name)..."
$ProgressPreference = "SilentlyContinue"   # the progress bar slows Invoke-WebRequest a lot on 5.1
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $target
Write-Host "Running the installer (the build is unsigned, so SmartScreen may ask)..."
Start-Process -FilePath $target -Wait
Remove-Item $target -ErrorAction SilentlyContinue
