# Distribution, signing, and what actually protects the network

hocMESH is closed source and shipped from a private repository. This document
says what that decision buys, what the signing does, and — because the
difference matters more here than in most projects — what neither of them can
do.

---

## 1. What is actually enforced

| Claim | True? | Enforced by |
|---|---|---|
| The source is not public | Yes | The repository is private; the licence forbids redistribution and reverse engineering |
| An installer is provably ours and unaltered | Yes, once signing keys are configured | Authenticode (Windows), Developer ID + notarisation (macOS), a GPG-signed checksum list (Linux) |
| A tampered installer is detectable | Yes | The signature fails to verify, and the published SHA-256 does not match |
| A node cannot be made to run arbitrary code | Yes | The work allow-list. A node executes named workloads with typed parameters; there is no path that runs a binary somebody sent it |
| A forged balance cannot be introduced | Yes | Every ledger entry needs threshold signatures from a validator quorum, and any peer can replay the chain and re-derive every balance |
| The installer cannot be copied or redistributed | **No** | Nothing. See below |
| Keeping the source private is what secures the ledger | **No** | The ledger is secured by signatures and quorum. See below |

## 2. Two things that are worth being blunt about

**No installer can be made technically un-redistributable.** A file that runs on
a user's machine can be copied off it. Signing, licence keys, obfuscation and
integrity checks all raise the effort required and make tampering *detectable*;
none of them make redistribution *impossible*, and any vendor claiming otherwise
is describing a legal deterrent rather than a technical one. What is enforceable
here is the combination of a private repository, a licence that forbids
redistribution, signed artifacts so a modified copy is identifiable, and — the
part that actually matters — a network that does not care whether a binary is
authentic.

**The ledger's security does not come from the source being secret.** It comes
from four properties that hold whether or not an attacker has read every line:

- Every transaction is signed by a key the coordinator never sees.
- Every ledger height needs threshold signatures from a quorum of independent
  validators; one compromised validator changes nothing.
- Every price is a closed-form function of the work *spec*
  (`work_cost_mcu`), so any peer can recompute it, and a validator refuses an
  entry whose arithmetic does not check out.
- Every peer can mirror the whole chain and audit it (`hocmesh audit`), and an
  ordinary transaction that does not sum to zero is rejected.

That is the right way round. A system that would be exploitable if its source
were known is exploitable already — the attacker who matters will decompile the
binary. Keeping the source private is a **commercial** decision, and a
reasonable one; it is not a security control, and it should not be relied on as
one. The build's protection against a modified peer is that a modified peer
cannot forge a signature, cannot reach quorum, and cannot make arithmetic that
does not check out get accepted.

## 3. Membership is the real gate

The strongest protection against a hostile peer is not that it cannot get the
software — it is that having the software does not admit it to anything.

New validators join by **vouching**: sitting validators sign a threshold vouch,
recorded on the chain itself (`hocmesh membership-vouch`,
`hocmesh membership-commit`). Every identity starts at zero CU, the only
issuance source is a bounded community account, and CU is never bought, sold or
transferred. So a copied installer yields a peer that can serve work and earn,
which is exactly what the network wants more of, and nothing else.

See `docs/SECURITY.md` for the trust boundaries in full.

## 4. Signing: how it is set up

Signing is off until keys are configured, and the scripts say so out loud rather
than quietly producing unsigned artifacts. Set
`HOCMESH_SIGNING_REQUIRED=1` (or pass `-Required` on Windows) to turn a missing
key into a build failure — do that for anything published.

| Platform | Script | Secrets |
|---|---|---|
| Windows | `scripts/sign-artifacts.ps1` | `WINDOWS_CERT_PFX_BASE64`, `WINDOWS_CERT_PASSWORD` |
| macOS | `scripts/sign-artifacts.sh` | `MACOS_CERT_P12_BASE64`, `MACOS_CERT_PASSWORD`, `MACOS_SIGN_IDENTITY`, and for notarisation `MACOS_NOTARY_APPLE_ID`, `MACOS_NOTARY_PASSWORD`, `MACOS_NOTARY_TEAM_ID` |
| Linux | `scripts/sign-artifacts.sh` | `GPG_PRIVATE_KEY` (base64 of an ASCII-armoured private key), `GPG_PASSPHRASE` |

`.github/workflows/release.yml` runs these. Windows and macOS artifacts are
signed **before** checksums are taken, so the published digest is the digest of
the file a user runs. Linux has no per-file signature format that every tool
checks, so the checksum list is signed instead: verifying the list verifies
every artifact it names.

Two details that are easy to get wrong and expensive to discover later:

- **Timestamp the Windows signature** (`/tr`). Without a timestamp, every
  installer ever shipped stops verifying on the day the certificate expires.
- **Notarise, then staple** on macOS. Without stapling, Gatekeeper has to reach
  Apple to approve the app, so a first run offline fails.

## 5. Verifying a download

```bash
# Linux / macOS
sha256sum -c hocmesh-<version>-<target>.tar.gz.sha256
gpg --verify hocmesh-<version>-<target>.tar.gz.sha256.asc
```

```powershell
# Windows
(Get-FileHash -Algorithm SHA256 .\hocmesh-desktop-<version>.msi).Hash
Get-AuthenticodeSignature .\hocmesh-desktop-<version>.msi | Format-List
```

A `Status` of `Valid` and a matching digest together mean the bytes are the ones
that were built and published. Neither means the network trusts the peer that
runs them — that is what identity, vouching and quorum are for.
