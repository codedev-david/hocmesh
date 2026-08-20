$ErrorActionPreference = "Stop"

cargo build --release --workspace
Remove-Item -Force mesh-demo.db,mesh-demo.db-shm,mesh-demo.db-wal -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force node-a,node-b,node-c -ErrorAction SilentlyContinue

.\target\release\mesh-coordinator.exe seed --db mesh-demo.db --start 2 --end 5000000 --shards 64

Write-Host ""
Write-Host "Next, open four PowerShell windows:"
Write-Host "1: .\target\release\mesh-coordinator.exe serve --db mesh-demo.db --listen 127.0.0.1:8080"
Write-Host "2: .\target\release\mesh.exe --home node-a init; .\target\release\mesh.exe --home node-a daemon --workers 2"
Write-Host "3: .\target\release\mesh.exe --home node-b init; .\target\release\mesh.exe --home node-b daemon --workers 2"
Write-Host "4: .\target\release\mesh.exe --home node-c init; .\target\release\mesh.exe --home node-c daemon --workers 2"
