$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
cargo build --release -p hocmesh
$Dest = Join-Path $env:LOCALAPPDATA "hocMESH\bin"
New-Item -ItemType Directory -Force $Dest | Out-Null
Copy-Item target\release\hocmesh.exe (Join-Path $Dest "hocmesh.exe") -Force
Write-Host "Installed hocMESH participant client to $Dest\hocmesh.exe"
Write-Host "Add $Dest to PATH or run the executable directly."
