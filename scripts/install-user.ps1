$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
cargo build --release -p hocmesh -p hocmesh-coordinator -p hocmesh-validator
$Dest = Join-Path $env:LOCALAPPDATA "hocMESH\bin"
New-Item -ItemType Directory -Force $Dest | Out-Null
foreach ($exe in @("hocmesh.exe", "hocmesh-coordinator.exe", "hocmesh-validator.exe")) {
    Copy-Item (Join-Path "target\release" $exe) (Join-Path $Dest $exe) -Force
}
Write-Host "Installed the hocMESH peer to $Dest"
Write-Host "Add $Dest to PATH or run the executable directly."
