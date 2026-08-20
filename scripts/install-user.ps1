$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
cargo build --release -p mesh
$Dest = Join-Path $env:LOCALAPPDATA "MESH\bin"
New-Item -ItemType Directory -Force $Dest | Out-Null
Copy-Item target\release\mesh.exe (Join-Path $Dest "mesh.exe") -Force
Write-Host "Installed MESH participant client to $Dest\mesh.exe"
Write-Host "Add $Dest to PATH or run the executable directly."
