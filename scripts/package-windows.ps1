[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Binary,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$binaryPath = (Resolve-Path -LiteralPath $Binary).Path
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

& wix build --nologo -arch x64 -d "HocMeshExe=$binaryPath" -d "HocMeshVersion=$cleanVersion" -pdbtype none -o $artifact $source
if ($LASTEXITCODE -ne 0) { throw "WiX build failed with exit code $LASTEXITCODE" }
& wix msi validate --nologo $artifact
if ($LASTEXITCODE -ne 0) { throw "MSI validation failed with exit code $LASTEXITCODE" }
Write-Output $artifact
