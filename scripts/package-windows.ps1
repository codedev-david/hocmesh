[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Binary,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$binaryPath = (Resolve-Path -LiteralPath $Binary).Path

# The coordinator and the validator are taken from wherever the node binary
# came from, so that a caller cannot accidentally package three binaries from
# three different builds. If they are missing, the build failed and the MSI
# should not be produced at all.
$binaryDirectory = Split-Path -Parent $binaryPath
$suffix = [System.IO.Path]::GetExtension($binaryPath)
$coordinatorPath = Join-Path $binaryDirectory "hocmesh-coordinator$suffix"
$validatorPath = Join-Path $binaryDirectory "hocmesh-validator$suffix"
foreach ($companion in @($coordinatorPath, $validatorPath)) {
    if (-not (Test-Path -LiteralPath $companion)) {
        throw "expected to package $companion alongside $binaryPath; build the whole workspace first"
    }
}
$cleanVersion = $Version.TrimStart("v")
if ($cleanVersion -notmatch '^\d+\.\d+\.\d+$') {
    throw "MSI version must be numeric major.minor.patch: $cleanVersion"
}
if (-not (Get-Command wix -ErrorAction SilentlyContinue)) {
    throw "WiX must be installed: dotnet tool install --global wix --version 6.0.2"
}

$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$artifact = Join-Path $outputPath "hocmesh-$cleanVersion-x86_64.msi"
$source = Join-Path $PSScriptRoot "..\packaging\windows\hocmesh.wxs"

& wix build --nologo -arch x64 `
    -d "HocMeshExe=$binaryPath" `
    -d "HocMeshCoordinatorExe=$coordinatorPath" `
    -d "HocMeshValidatorExe=$validatorPath" `
    -d "HocMeshVersion=$cleanVersion" `
    -pdbtype none -o $artifact $source
if ($LASTEXITCODE -ne 0) { throw "WiX build failed with exit code $LASTEXITCODE" }
& wix msi validate --nologo $artifact
if ($LASTEXITCODE -ne 0) { throw "MSI validation failed with exit code $LASTEXITCODE" }
Write-Output $artifact
