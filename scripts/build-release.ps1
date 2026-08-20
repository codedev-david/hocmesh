$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
cargo build --release --workspace
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
New-Item -ItemType Directory -Force dist, dist\docs, dist\config | Out-Null
Copy-Item target\release\mesh.exe dist\
Copy-Item target\release\mesh-coordinator.exe dist\
Copy-Item target\release\mesh-validator.exe dist\
Copy-Item README.md, CODEX_HANDOFF.md, LICENSE dist\
Copy-Item docs\*.md dist\docs\
Copy-Item config\*.json dist\config\
Write-Host "Release folder: $((Resolve-Path dist).Path)"
