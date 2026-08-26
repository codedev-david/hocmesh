$ErrorActionPreference = "Stop"

cargo build --release --workspace
Remove-Item -Force hocmesh-demo.db,hocmesh-demo.db-shm,hocmesh-demo.db-wal -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force node-a,node-b,node-c -ErrorAction SilentlyContinue

.\target\release\hocmesh-coordinator.exe seed --db hocmesh-demo.db --start 2 --end 5000000 --shards 64

Write-Host ""
Write-Host "Next, open four PowerShell windows:"
Write-Host "1: .\target\release\hocmesh-coordinator.exe serve --db hocmesh-demo.db --listen 127.0.0.1:8080"
Write-Host "2: .\target\release\hocmesh.exe --home node-a init; .\target\release\hocmesh.exe --home node-a daemon --workers 2"
Write-Host "3: .\target\release\hocmesh.exe --home node-b init; .\target\release\hocmesh.exe --home node-b daemon --workers 2"
Write-Host "4: .\target\release\hocmesh.exe --home node-c init; .\target\release\hocmesh.exe --home node-c daemon --workers 2"
