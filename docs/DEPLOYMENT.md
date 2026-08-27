# Deploying hocMESH on real machines

This is the runbook for taking hocMESH off one developer's laptop and onto two
or more machines that talk to each other over a network. It assumes nothing has
been set up yet.

The README's "Quick local MVP mode" runs everything in one directory on one
host and uses a coordinator-owned SQLite ledger. That mode is for development
only: the ledger answers to whoever is running the coordinator. Everything below
runs the quorum ledger, where the coordinator schedules work but is never the
authority for CU.

---

## 0. What has to be true before you start

| Requirement | Why |
| --- | --- |
| One machine reachable by the others on a known address | The coordinator and the validators listen on TCP; workers dial out to them. |
| Outbound HTTPS from every machine that will run AI work | `runtime-install` fetches from `github.com`; `model-pull` fetches from `huggingface.co`. |
| A clock within a minute or so of correct on every machine | Signed requests carry timestamps and are rejected outside a freshness window. |
| Rust 1.97+ **or** a release artifact for the platform | See "Installing" below. |

Nothing else. There is no database server, no message broker, no container
runtime, and no external identity provider.

---

## 1. Installing

Every hocMESH deployment needs all three binaries available *somewhere*, though
not on every machine:

| Binary | Runs on | Purpose |
| --- | --- | --- |
| `hocmesh-validator` | The ledger quorum (3 or more hosts) | Authoritative for CU. Signs entries. |
| `hocmesh-coordinator` | One host | Schedules work. Holds no authority over balances. |
| `hocmesh` | Every participating machine | Contributes hardware, submits jobs, runs inference. |

### From a release artifact

The `.msi`, `.pkg`, `.deb` and the plain archive all install **all three**
binaries. That is deliberate: a machine with `hocmesh` but no
`hocmesh-coordinator` can only join a mesh somebody else started, and an
installer that quietly leaves you unable to start one is a half-install that
looks complete.

```bash
# Linux
sudo dpkg -i hocmesh_0.3.0_amd64.deb

# macOS
sudo installer -pkg hocmesh-0.3.0.pkg -target /
```

```powershell
# Windows (elevated)
msiexec /i hocmesh-0.3.0-x86_64.msi /qb
```

### From source

```bash
cargo build --release --workspace --locked
```

The three binaries land in `target/release/`. Copy them onto each machine, or
run `scripts/install-user.sh` / `scripts/install-user.ps1` to put `hocmesh` on
the current user's PATH.

---

## 2. Stand up the validator quorum

The validators are the ledger. Run at least three, ideally on hosts that do not
fail together. `validators.json` names the sitting set and states the
`threshold` — how many of them must sign before any CU moves. Start from
`config/validators.example.json`, which sets four members and a threshold of
three.

On each validator host, generate an identity:

```bash
hocmesh-validator id --home .hocmesh-validator-1
```

The identity is created on first use, so this both generates and prints it.

Collect the printed ids and public keys into a single `validators.json`,
replacing the `REPLACE` placeholders in the example, and distribute **the same
file** to every validator and to the coordinator. See `docs/LEDGER.md` for the
file's shape and `README.md` §"Recommended quorum-ledger mode" for a worked
three-validator example.

Then start each one, binding to an address the others can reach:

```bash
hocmesh-validator serve \
  --home .hocmesh-validator-1 \
  --db validator-1.db \
  --listen 0.0.0.0:9101 \
  --validators validators.json
```

Seal the signing keys before doing this on anything you care about:

```bash
export HOCMESH_IDENTITY_PASSPHRASE='...'
```

Without it, `identity.json` holds an unsealed Ed25519 private key and hocMESH
says so on every command. The ledger itself is deliberately not encrypted — see
`docs/SECURITY.md` for why the key is sealed and the ledger is not.

---

## 3. Start the coordinator

On the host the workers will dial:

```bash
hocmesh-coordinator serve \
  --db hocmesh.db \
  --listen 0.0.0.0:8080 \
  --validators validators.json
```

`--db` here is scheduling state — leases, shards, job metadata. Balances live
with the validators. If this database is lost, no CU is lost with it.

Put a TLS terminator in front of it for anything public. hocMESH signs its
requests, so an attacker cannot forge them, but without TLS they are readable.

---

## 4. Join a machine to the mesh

On every participating machine:

```bash
hocmesh --coordinator https://coordinator.example.org --home .hocmesh init
```

Decide what share of the box is being lent. A contributor lends a share, not the
whole machine:

```bash
hocmesh --home .hocmesh limits --cpu-percent 50 --memory-percent 40 --gpu-percent 0
```

Then run the daemon:

```bash
hocmesh --coordinator https://coordinator.example.org --home .hocmesh daemon --workers 4
```

Confirm from another machine that it is visible:

```bash
hocmesh --coordinator https://coordinator.example.org --home .hocmesh status
```

---

## 5. Add local AI inference

This is the part that needs no external setup at all. Two commands.

### 5a. Install the inference runtime

```bash
hocmesh runtime-install
```

This downloads the llama.cpp build pinned in `crates/hocmesh-gpu/src/runtime.rs`
for this OS and architecture, checks it against a SHA-256 compiled into the
binary, unpacks it under `<home>/runtime/<build>/`, and records the executable
so `infer` and `daemon` find it without a flag.

The digest is the point. hocMESH's safety property is that nodes execute
allow-listed work and never a binary somebody sent them, so an installer that
fetched "the latest llama.cpp" by name would be handing that property away — a
name is a mutable pointer and a digest is not. If the bytes do not hash to the
compiled-in value, the download is discarded and nothing is installed.

Check what it will do, or what it did, without downloading anything:

```bash
hocmesh runtime-status
```

To use a llama.cpp you built yourself instead, skip this and pass
`--runtime /path/to/llama-cli` to `infer`, or `--ai-runtime` to `daemon`.

### 5b. Pull a model

```bash
hocmesh model-catalog          # what is known by name
hocmesh model-pull qwen2.5-0.5b-instruct
```

`model-pull` resolves the file on the Hub, downloads it with resume, verifies
the SHA-256 the Hub published for it, reads the architecture out of the GGUF
header rather than making you assert it, chunks it into the content-addressed
store, and registers it. Then:

```bash
hocmesh infer --model-id qwen2.5-0.5b-instruct --prompt "hello"
```

No `--runtime`, no `--architecture`, no manual download.

The catalogue is a convenience index from a memorable id to a repository and a
preferred quantisation. It carries no digests — those are resolved per pull and
verified against the bytes that arrive, because a digest shipped in the binary
that nobody could re-verify would look like a guarantee it is not. Anything not
catalogued works the same way:

```bash
hocmesh model-pull --repository TheBloke/Llama-2-7B-Chat-GGUF --quantisation q4_k_m
```

Or pin a specific file yourself, from anywhere:

```bash
hocmesh model-pull \
  --url https://example.org/model.gguf \
  --sha256 <64 hex characters>
```

`--sha256` is optional when the source publishes a digest and **required** with
`--url`, because there is otherwise nothing to check the bytes against.

### 5c. Serve the model to the rest of the mesh

Once one machine holds a model, others can fetch it from that machine rather
than from the internet:

```bash
# On the machine that has the model
hocmesh --home .hocmesh daemon \
  --workers 4 \
  --model-seed-listen 0.0.0.0:8090 \
  --model-seed-url http://this-host.example.org:8090

# On a machine that wants it
hocmesh --home .hocmesh model-seed \
  --peer http://this-host.example.org:8090 \
  --model-id qwen2.5-0.5b-instruct
```

Chunks are addressed and verified by content, so a seeding peer cannot serve
anything other than the model you asked for.

### 5d. Offer AI work to the mesh

A daemon advertises AI readiness when it has a runtime **and** the operator has
agreed to serve inference. Agreeing is its own switch, because running a model
for yourself is not the same decision as running one for strangers:

```bash
hocmesh --home .hocmesh limits --ai on
hocmesh --home .hocmesh daemon --workers 4
```

`--ai` takes `on`, `off`, or `auto`. `auto` is the default and means "offer it
when a GPU is lent", which is what every node did before the switch existed —
so an existing `limits.json` keeps its behaviour exactly.

**A machine with no GPU can serve inference.** With `--ai on` the node
advertises its shared CPU slice as a device and takes AI work on it. That is
what the CPU build from `runtime-install` is for. It will be slow, and the size
of model it will be offered is bounded by `--memory-percent`, because the
advertised device reports the lent slice rather than the whole machine.

On a GPU box, `--gpu-percent` still governs the accelerator: at `0` the GPU is
not advertised at all, and `--ai on` then offers CPU inference instead.

`--ai-runtime` overrides which executable is used. `--no-ai` declines AI work
for one run without changing the stored limits.

---

## 6. Prove it works across the two machines

From a machine that is **not** running the daemon under test:

```bash
hocmesh --coordinator https://coordinator.example.org --home .hocmesh submit-prime \
  --start 2 --end 2000000 --shards 8
hocmesh --coordinator https://coordinator.example.org --home .hocmesh balance
```

A requester cannot execute their own paid shards — both the scheduler and the
validators enforce that — so this genuinely exercises the network path rather
than a local loop. Watch the other machine's daemon log pick up leases.

For inference across machines:

```bash
hocmesh ai-submit --model-id qwen2.5-0.5b-instruct --prompt "..." --layers 28
hocmesh ai-job <job-id>
hocmesh ai-receipt <job-id> <assignment-id>
hocmesh ai-settle <job-id> <assignment-id>            # accept, and pay the provider
hocmesh ai-settle <job-id> <assignment-id> --dispute  # return the CU to the commons
```

`--layers` is the model's layer count, which is a property of the model you
are running, not something hocMESH picks. `hocmesh ai-plan --model-id <id>
--layers <n>` shows how that would be split across the nodes currently
advertising AI readiness before you spend anything.

Settlement is two-stage on purpose: the receipt moves escrow into a holding
account, and only a signed acceptance pays the provider. Disputing returns the
same CU to the commons, so neither answer is the cheaper one.

---

## 7. Verify without trusting the coordinator

```bash
hocmesh-validator audit --db validator-1.db --validators validators.json
```

This recomputes the hash chain and checks the quorum certificates. It does not
ask the coordinator anything. See `docs/VERIFICATION.md`.

---

## Where things live

Everything a node owns is under `--home` (default `.hocmesh`):

```
.hocmesh/
  identity.json          Ed25519 node identity (seal it)
  limits.json            what share of this machine is lent (written by `limits`)
  model-cache/           content-addressed model chunks
  model-registry.db      which models are registered
  models/                materialised .gguf files, named by manifest digest
  runtime/
    <build>/             the unpacked llama.cpp install
    current.txt          which executable infer/daemon use
  downloads/             transient; emptied unless --keep-download
```

Deleting `runtime/` and re-running `runtime-install` is always safe. Deleting
`identity.json` gives the machine a new identity and forfeits its history.

---

## Troubleshooting

**`runtime-install` fails with a TLS or handshake error.** Corporate TLS
interception. hocMESH pins the digest of what it downloads, so it cannot accept
an intercepted response even if the proxy re-signs it. Fetch the release asset
named by `runtime-status` on a machine that can reach `github.com`, unpack it,
and pass `--runtime <path>` instead.

**`model-pull` fails with `received fatal alert: HandshakeFailure`.** The same
cause, against `huggingface.co`. Download the `.gguf` elsewhere and import it:

```bash
hocmesh model-import ./model.gguf --model-id my-model --format gguf --architecture llama
```

**`model-pull` says a repository could not be listed.** Catalogue entries are
unverified pointers to repositories, and repositories move. Pass `--repository`
directly; the error includes the URL that was tried.

**A model is refused because it is sharded.** Multi-part GGUF
(`-00001-of-00003.gguf`) is not supported. The error lists the files it found so
you can pick a single-file quantisation.

**A node never receives work.** Check `hocmesh status` from the node itself,
then confirm the operator limits are not zero for the resource the job needs.
For AI work specifically, run `hocmesh limits` and read the `ai:` line: `auto`
with no GPU lent means the node is not offering inference, whatever is
installed. `hocmesh limits --ai on` offers it.

**The daemon logs an unsealed-key warning on every command.** Set
`HOCMESH_IDENTITY_PASSPHRASE`. On platforms without file-mode enforcement there
is no other protection for the key.
