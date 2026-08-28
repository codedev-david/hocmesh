#Requires -Version 5.1
<#
.SYNOPSIS
Stand up a whole hocMESH locally, put artificial load through it, and check the
economy survived.

.DESCRIPTION
The Windows peer of scripts/loadtest-local.sh, and the same argument for
existing: the failures worth catching in this system are races, not slow
responses. Two proposers on the same head, a reward applied before its head is
readable, a claim key settled twice -- none of them are reachable by one person
clicking through a demo, and all of them are reachable by a dozen jobs landing
at once.

So this creates the contention and then asks the ledger to prove nothing
leaked. Pass or fail is decided by whether the work settled and whether the CU
add up, never by how fast this machine happened to be: a speed threshold in CI
is a flaky test, and a flaky test teaches people to ignore red.

.EXAMPLE
./scripts/loadtest-local.ps1

.EXAMPLE
./scripts/loadtest-local.ps1 -Jobs 40 -Concurrency 10 -Shards 8
#>
[CmdletBinding()]
param(
    [int]$Jobs = 12,
    [int]$Concurrency = 4,
    [int]$Shards = 4,
    [long]$Size = 50000,
    [ValidateSet('collatz', 'prime', 'matrix')]
    [string]$Workload = 'collatz',
    # Worker threads per node, not the number of nodes. Two nodes always run.
    [int]$Workers = 2,
    [int]$DurationSecs = 0,
    [string]$Json = '',
    # Bigger than any plausible run needs. Community work is minted by the
    # sitting validators and is the only way a new account gets its first CU,
    # so the seed has to cover the run with room to spare -- an account that
    # runs out halfway through fails in a way that looks exactly like the
    # settlement bug this script exists to detect.
    [long]$SeedEnd = 4000000,
    [int]$SeedShards = 64,
    [switch]$Keep,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$bin = Join-Path $root 'target\release'
$node = Join-Path $bin 'hocmesh.exe'
$coordBin = Join-Path $bin 'hocmesh-coordinator.exe'
$validatorBin = Join-Path $bin 'hocmesh-validator.exe'

if (-not $SkipBuild) {
    Write-Host '==> building release binaries'
    cargo build --release -p hocmesh -p hocmesh-coordinator -p hocmesh-validator
    if ($LASTEXITCODE -ne 0) { throw 'build failed' }
}
foreach ($b in @($node, $coordBin, $validatorBin)) {
    if (-not (Test-Path $b)) { throw "missing binary: $b (drop -SkipBuild?)" }
}

$work = Join-Path $root 'target\loadtest'
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force $work | Out-Null

$procs = New-Object System.Collections.ArrayList
$coordPort = 0

function Stop-Everything {
    if ($Keep) {
        Write-Host ''
        Write-Host "-Keep: leaving the network up. Coordinator: http://127.0.0.1:$coordPort"
        return
    }
    # Reverse order, so workers stop before the coordinator they report to and
    # the coordinator stops before the ledger it settles against. Tearing down
    # the other way produces a screenful of connection errors that look like
    # failures and are not.
    for ($i = $procs.Count - 1; $i -ge 0; $i--) {
        try { Stop-Process -Id $procs[$i] -Force -ErrorAction Stop } catch {}
    }
}

# No BOM. Every file this writes is read by a Rust program through serde.
function Write-Utf8([string]$path, [string]$content) {
    [System.IO.File]::WriteAllText($path, $content, (New-Object System.Text.UTF8Encoding $false))
}

function Get-FreePort([int]$from) {
    $p = $from
    while ($true) {
        $listener = $null
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
            $listener.Start()
            $listener.Stop()
            return $p
        } catch { $p++ } finally { if ($listener) { try { $listener.Stop() } catch {} } }
    }
}

function Start-Bg([string]$exe, [string[]]$exeArgs, [string]$log) {
    $p = Start-Process -FilePath $exe -ArgumentList $exeArgs -PassThru -NoNewWindow `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    [void]$procs.Add($p.Id)
    return $p
}

function Wait-Health([int]$port, [string]$what) {
    for ($i = 0; $i -lt 200; $i++) {
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -UseBasicParsing | Out-Null
            return
        } catch { Start-Sleep -Milliseconds 100 }
    }
    $log = Join-Path $work "$what.log"
    if (Test-Path $log) { Get-Content $log -Tail 20 | Write-Host }
    throw "$what never became healthy on port $port"
}

# Windows PowerShell 5.1 turns any line a native program writes to stderr into
# an ErrorRecord, and with $ErrorActionPreference = 'Stop' that aborts the
# script -- even when the program went on to exit 0. These binaries warn on
# stderr about unsealed keys, which is correct of them and not a failure here,
# so native calls go through this and are judged on their exit code alone.
function Invoke-Native([string]$exe, [string[]]$exeArgs) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $exe @exeArgs 2>&1 } finally { $ErrorActionPreference = $previous }
}

function Run-Ok([string]$exe, [string[]]$exeArgs, [string]$what) {
    Invoke-Native $exe $exeArgs | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "$what failed with exit code $LASTEXITCODE" }
}

try {
    # -------------------------------------------------------------- ledger --
    # Four validators, threshold three. The set refuses any threshold that is
    # not more than two thirds of its membership, which is what makes the
    # ledger a quorum rather than a database with extra steps: it survives one
    # seat being wrong or gone, and no smaller group can move a balance. Four
    # is the smallest membership where that leaves a spare seat.
    Write-Host '==> creating a 4-validator set (threshold 3)'
    $members = @()
    $valPorts = @()
    $port = 9301
    foreach ($i in 0..3) {
        $port = Get-FreePort $port
        $valPorts += $port
        # $home is a read-only automatic variable in PowerShell.
        $valHome = Join-Path $work "validator-$i"
        $out = Invoke-Native $validatorBin @('id', '--home', $valHome)
        if ($LASTEXITCODE -ne 0) { throw "validator $i id failed" }
        $vid = ($out | Select-String '^validator_id=(.*)$').Matches.Groups[1].Value
        $pk = ($out | Select-String '^public_key_b64=(.*)$').Matches.Groups[1].Value
        if (-not $vid -or -not $pk) { throw "could not read validator $i identity" }
        $members += [ordered]@{
            validator_id   = $vid
            url            = "http://127.0.0.1:$port"
            public_key_b64 = $pk
        }
        $port++
    }

    $validators = Join-Path $work 'validators.json'
    # WriteAllText, not Out-File: `-Encoding utf8` in Windows PowerShell 5.1
    # emits a BOM, and serde refuses a BOM with "expected value at line 1
    # column 1" -- which reads like a malformed set rather than an encoding.
    Write-Utf8 $validators ([ordered]@{
        threshold                   = 3
        community_issuance_limit_mcu = 1000000000
        members                     = $members
    } | ConvertTo-Json -Depth 5)

    foreach ($i in 0..3) {
        Start-Bg $validatorBin @(
            'serve',
            '--home', (Join-Path $work "validator-$i"),
            '--db', (Join-Path $work "validator-$i.db"),
            '--listen', "127.0.0.1:$($valPorts[$i])",
            '--validators', $validators
        ) (Join-Path $work "validator-$i.log") | Out-Null
    }
    foreach ($i in 0..3) { Wait-Health $valPorts[$i] "validator-$i" }
    Write-Host "    validators up on $($valPorts -join ' ')"

    # ---------------------------------------------------------- the seed --
    # The sponsorships have to come off the keys that actually sit in the
    # validators' homes: minting is the set's decision, and the coordinator can
    # only carry signatures it was handed. Signing any other way here would
    # prove something no operator ever does.
    Write-Host "==> minting community work (2..$SeedEnd, $SeedShards shards)"
    $seedJob = 'job_loadtest_seed'
    $vouches = foreach ($i in 0..3) {
        $out = Invoke-Native $node @(
            '--home', (Join-Path $work "validator-$i"), 'community-vouch',
            '--validators', $validators, '--job-id', $seedJob,
            '--start', '2', '--end', "$SeedEnd", '--shards', "$SeedShards"
        )
        if ($LASTEXITCODE -ne 0) { throw "community-vouch $i failed" }
        ($out | Select-Object -Last 1).Trim()
    }
    $sponsors = Join-Path $work 'sponsors.json'
    Write-Utf8 $sponsors "[$($vouches -join ',')]"

    $coordDb = Join-Path $work 'coordinator.db'
    Run-Ok $coordBin @(
        'seed', '--db', $coordDb, '--validators', $validators,
        '--job-id', $seedJob, '--sponsors', $sponsors,
        '--start', '2', '--end', "$SeedEnd", '--shards', "$SeedShards"
    ) 'seed community job'

    $coordPort = Get-FreePort 9401
    Start-Bg $coordBin @(
        'serve', '--db', $coordDb, '--listen', "127.0.0.1:$coordPort",
        '--validators', $validators
    ) (Join-Path $work 'coordinator.log') | Out-Null
    Wait-Health $coordPort 'coordinator'
    $coordinator = "http://127.0.0.1:$coordPort"
    Write-Host "    coordinator up on $coordPort"

    # ---------------------------------------------------------- funding --
    # What the run will cost is knowable before it starts, from the same
    # pricing function the ledger charges with, so wait for exactly that much
    # rather than guessing a sleep. A run that begins underfunded fails at
    # settlement and looks indistinguishable from a real bug.
    $nodeA = Join-Path $work 'node-a'
    $dry = Invoke-Native $node @(
        '--home', $nodeA, 'loadtest', '--coordinator', $coordinator, '--dry-run',
        '--jobs', "$Jobs", '--concurrency', "$Concurrency", '--shards', "$Shards",
        '--workload', $Workload, '--size', "$Size"
    )
    if ($LASTEXITCODE -ne 0) { throw 'could not price the run' }
    $needMcu = [long]($dry | Select-String '^total_mcu=(\d+)$').Matches.Groups[1].Value
    Write-Host "==> this run will cost $needMcu mCU; earning it first"

    Run-Ok $node @('--home', $nodeA, 'init', '--coordinator', $coordinator) 'node-a init'
    $earner = Start-Bg $node @(
        '--home', $nodeA, 'daemon', '--coordinator', $coordinator,
        '--workers', "$Workers", '--no-control'
    ) (Join-Path $work 'node-a.log')

    # Only the requester works this stage, so it takes the whole seed and
    # reaches solvency in a handful of shards rather than racing other nodes.
    $banked = 0
    for ($i = 0; $i -lt 600; $i++) {
        $out = Invoke-Native $node @('--home', $nodeA, 'balance', '--coordinator', $coordinator)
        $m = $out | Select-String '^Banked: ([0-9.]+) CU$'
        if ($m) { $banked = [long]([double]$m.Matches.Groups[1].Value * 1000) }
        if ($banked -ge $needMcu) { break }
        Start-Sleep -Milliseconds 500
    }
    if ($banked -lt $needMcu) {
        Get-Content (Join-Path $work 'node-a.log') -Tail 20 | Write-Host
        throw "node-a only earned $banked of the $needMcu mCU it needs"
    }
    Write-Host "    node-a banked $banked mCU"

    # Stop earning before spending. During the load test node-a is a requester,
    # and a requester whose own daemon is draining leftover community work in
    # the background makes the accounting harder to read for no benefit -- the
    # CU invariants would still hold, but a human staring at the report would
    # have to work out why.
    try { Stop-Process -Id $earner.Id -Force -ErrorAction Stop } catch {}

    # ---------------------------------------------------------- workers --
    Write-Host "==> starting 2 worker nodes, $Workers threads each"
    foreach ($n in @('b', 'c')) {
        $workerHome = Join-Path $work "node-$n"
        Run-Ok $node @('--home', $workerHome, 'init', '--coordinator', $coordinator) "node-$n init"
        Start-Bg $node @(
            '--home', $workerHome, 'daemon', '--coordinator', $coordinator,
            '--workers', "$Workers", '--no-control'
        ) (Join-Path $work "node-$n.log") | Out-Null
    }

    # ------------------------------------------------------------- load --
    Write-Host '==> running the load test'
    Write-Host ''
    $loadArgs = @(
        '--home', $nodeA, 'loadtest', '--coordinator', $coordinator,
        '--jobs', "$Jobs", '--concurrency', "$Concurrency",
        '--shards', "$Shards", '--workload', $Workload, '--size', "$Size"
    )
    if ($DurationSecs -gt 0) { $loadArgs += @('--duration-secs', "$DurationSecs") }
    if ($Json) { $loadArgs += @('--json', $Json) }
    Invoke-Native $node $loadArgs | Write-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host "LOAD TEST FAILED. Logs are in $work"
        exit $LASTEXITCODE
    }

    # The run passing means the coordinator's arithmetic held. Auditing the
    # ledger from genesis is a different claim, and the stronger one: every
    # entry the quorum certified re-verified from the first, against the
    # validator set that was sitting at the time.
    Write-Host ''
    Write-Host '==> auditing the ledger from genesis'
    Invoke-Native $node @(
        '--home', $nodeA, 'ledger-sync', '--validators', $validators,
        '--db', (Join-Path $work 'mirror.db'), '--coordinator', $coordinator
    ) | Write-Host
    if ($LASTEXITCODE -ne 0) {
        throw 'ledger audit FAILED after a passing load test -- that is the bad case'
    }

    Write-Host ''
    Write-Host 'PASSED: work settled, CU conserved, ledger audits from genesis.'
} finally {
    Stop-Everything
}
