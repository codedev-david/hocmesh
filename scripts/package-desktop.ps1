[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Binary,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$TauriCliVersion = "2.11.4"
)

# Builds the desktop installers: an MSI and an NSIS setup executable, each
# carrying a whole hocMESH peer -- the node, the coordinator, the validator,
# and the window that drives them.
#
# There is no smaller "desktop-only" install. A hocMESH peer serves before it
# consumes, and a machine that can join a mesh but not start or validate one is
# a half-install that looks complete. package-windows.ps1 builds exactly the
# same three binaries without the window, for a machine with no screen.
#
# The binaries are passed in rather than built here so that the window and the
# daemons it drives always come from one build.

$ErrorActionPreference = "Stop"
$binaryPath = (Resolve-Path -LiteralPath $Binary).Path
$binaryDir = Split-Path -Parent $binaryPath
$peers = @{
    "hocmesh-coordinator" = Join-Path $binaryDir "hocmesh-coordinator.exe"
    "hocmesh-validator"   = Join-Path $binaryDir "hocmesh-validator.exe"
}
foreach ($peer in $peers.Values) {
    if (-not (Test-Path -LiteralPath $peer)) {
        throw "expected to package $peer alongside $binaryPath; build the whole peer first"
    }
}
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$desktopDir = Join-Path $repositoryRoot "crates\hocmesh-desktop"

$cleanVersion = $Version.TrimStart("v")
if ($cleanVersion -notmatch '^\d+\.\d+\.\d+$') {
    throw "installer version must be numeric major.minor.patch: $cleanVersion"
}

# The bundler takes its version from tauri.conf.json, so a tree whose config
# disagrees with its VERSION file would ship an installer named for one
# release and reporting another.
$configuredVersion = (Get-Content (Join-Path $desktopDir "tauri.conf.json") -Raw | ConvertFrom-Json).version
$repositoryVersion = (Get-Content (Join-Path $repositoryRoot "VERSION") -Raw).Trim()
if ($configuredVersion -ne $repositoryVersion) {
    throw "tauri.conf.json says $configuredVersion but VERSION says $repositoryVersion"
}

# Tauri names a sidecar for the triple it was built for and strips that suffix
# when it bundles, which is what lands each binary next to the app under its
# real name -- `hocmesh.exe` being the first place supervisor::candidate_paths
# looks, and the command an operator types in a terminal.
$hostTriple = (& rustc -vV | Select-String -Pattern '^host: (.+)$').Matches[0].Groups[1].Value
$sidecarDir = Join-Path $desktopDir "binaries"
New-Item -ItemType Directory -Force -Path $sidecarDir | Out-Null
Copy-Item -LiteralPath $binaryPath -Destination (Join-Path $sidecarDir "hocmesh-$hostTriple.exe") -Force
foreach ($name in $peers.Keys) {
    Copy-Item -LiteralPath $peers[$name] -Destination (Join-Path $sidecarDir "$name-$hostTriple.exe") -Force
}

if (-not (Get-Command cargo-tauri -ErrorAction SilentlyContinue)) {
    # Out-Host, here and below, so that the only thing this script writes to
    # the pipeline is the artifact paths a caller wants to capture.
    & cargo install tauri-cli --version $TauriCliVersion --locked | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "installing tauri-cli failed with exit code $LASTEXITCODE" }
}

Push-Location $desktopDir
try {
    & cargo tauri build --config tauri.bundle.json -- --locked | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "cargo tauri build failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$bundleDir = Join-Path $repositoryRoot "target\release\bundle"

function Copy-Bundle {
    param([string]$Subdirectory, [string]$Filter, [string]$ArtifactName)

    $source = Get-ChildItem (Join-Path $bundleDir $Subdirectory) -Filter $Filter -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $source) { throw "the bundler produced no $Filter in $Subdirectory" }
    if ($source.Length -le 0) { throw "$($source.FullName) is empty" }
    $artifact = Join-Path $outputPath $ArtifactName
    Copy-Item -LiteralPath $source.FullName -Destination $artifact -Force
    return $artifact
}

$msi = Copy-Bundle -Subdirectory "msi" -Filter "*.msi" -ArtifactName "hocmesh-desktop-$cleanVersion-x86_64.msi"
$nsis = Copy-Bundle -Subdirectory "nsis" -Filter "*.exe" -ArtifactName "hocmesh-desktop-$cleanVersion-x86_64-setup.exe"

# An installer that lays down the window without the daemon would install an
# app that cannot start anything, and it would look perfectly fine from the
# outside. Open it and check for both.
$extract = Join-Path ([System.IO.Path]::GetTempPath()) "hocmesh-desktop-msi-$cleanVersion"
if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
$process = Start-Process msiexec.exe -Wait -PassThru -ArgumentList @("/a", $msi, "/qn", "TARGETDIR=$extract")
if ($process.ExitCode -ne 0) { throw "MSI administrative extraction failed: $($process.ExitCode)" }
foreach ($expected in @("hocmesh.exe", "hocmesh-coordinator.exe", "hocmesh-validator.exe", "hocmesh-desktop.exe")) {
    if (-not (Get-ChildItem $extract -Filter $expected -Recurse -File | Select-Object -First 1)) {
        throw "$expected is absent from $msi"
    }
}
Remove-Item -Recurse -Force $extract

Write-Output $msi
Write-Output $nsis
