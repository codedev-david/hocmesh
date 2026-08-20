$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
cargo build --release --workspace
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
New-Item -ItemType Directory -Force dist, dist\docs, dist\config | Out-Null
Copy-Item target\release\mesh.exe dist\
Copy-Item target\release\mesh-coordinator.exe dist\
Copy-Item target\release\mesh-validator.exe dist\
Copy-Item README.md, CODEX_HANDOFF.md, LICENSE, VERSION dist\
Copy-Item docs\*.md dist\docs\
Copy-Item config\*.json dist\config\
$archive = "mesh-$((Get-Content VERSION).Trim())-windows-x86_64.zip"
if (Test-Path $archive) { Remove-Item -Force $archive }
Compress-Archive -Path dist\* -DestinationPath $archive
Get-FileHash -Algorithm SHA256 $archive |
    ForEach-Object { "$($_.Hash.ToLowerInvariant())  $archive" } |
    Set-Content -Encoding ascii "$archive.sha256"
Write-Host "Release folder: $((Resolve-Path dist).Path)"
Write-Host "Release archive: $((Resolve-Path $archive).Path)"
