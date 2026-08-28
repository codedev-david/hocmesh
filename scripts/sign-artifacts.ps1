<#
.SYNOPSIS
    Authenticode-sign the Windows artifacts in a directory.

.DESCRIPTION
    Signing proves the bytes came from the holder of the certificate and have
    not been altered since. It does not stop anyone copying the installer, and
    nothing can: see docs/DISTRIBUTION.md for what this does and does not buy.

    With no certificate configured this reports that the artifacts are unsigned
    and succeeds, so a fork or a local build still works. Pass -Required (CI
    does, for tagged releases) to turn a missing certificate into a failure
    rather than a silently unsigned release.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [string]$CertificateBase64 = $env:WINDOWS_CERT_PFX_BASE64,
    [string]$CertificatePassword = $env:WINDOWS_CERT_PASSWORD,
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$Required
)

$ErrorActionPreference = "Stop"

$artifacts = Get-ChildItem -Path $Directory -Recurse -File |
    Where-Object { $_.Extension -in @(".exe", ".msi", ".dll") }

if (-not $artifacts) {
    throw "no signable artifacts found under $Directory"
}

if (-not $CertificateBase64 -or -not $CertificatePassword) {
    $message = "No signing certificate configured; $($artifacts.Count) artifact(s) are UNSIGNED."
    if ($Required) { throw $message }
    Write-Host $message
    Write-Host "Set WINDOWS_CERT_PFX_BASE64 and WINDOWS_CERT_PASSWORD to sign."
    exit 0
}

# signtool ships with the Windows SDK and is not on PATH by default.
$signtool = Get-Command signtool.exe -ErrorAction SilentlyContinue
if (-not $signtool) {
    $found = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "x64" } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if (-not $found) { throw "signtool.exe was not found; install the Windows SDK" }
    $signtool = $found.FullName
} else {
    $signtool = $signtool.Source
}

$pfx = Join-Path ([System.IO.Path]::GetTempPath()) "hocmesh-signing-$([System.Guid]::NewGuid()).pfx"
try {
    [System.IO.File]::WriteAllBytes($pfx, [System.Convert]::FromBase64String($CertificateBase64))

    foreach ($artifact in $artifacts) {
        Write-Host "Signing $($artifact.Name)"
        # A timestamp is what keeps the signature valid after the certificate
        # expires. Without it every shipped installer stops verifying on the
        # certificate's expiry date.
        & $signtool sign /f $pfx /p $CertificatePassword /fd SHA256 `
            /tr $TimestampUrl /td SHA256 $artifact.FullName
        if ($LASTEXITCODE -ne 0) { throw "signtool failed on $($artifact.Name)" }
    }

    foreach ($artifact in $artifacts) {
        & $signtool verify /pa /v $artifact.FullName
        if ($LASTEXITCODE -ne 0) { throw "the signature on $($artifact.Name) does not verify" }
    }
    Write-Host "Signed and verified $($artifacts.Count) artifact(s)."
} finally {
    if (Test-Path $pfx) { Remove-Item $pfx -Force }
}
