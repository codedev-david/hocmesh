# Changelog

## 0.5.1

### The forward pass is now checked against an implementation nobody here wrote

0.5.0 shipped the engine and proved that splitting a model changes nothing.
That proof compared hocMESH against hocMESH: if the attention layout or a
dequantiser had been wrong, a split model would have reproduced a whole model's
mistake exactly and every assertion would still have passed. This release
closes that gap, and closing it found a real bug.

- **New test `reference_parity.rs`** drives llama.cpp as a server and feeds it
  token ids rather than text, so no tokeniser sits between the two
  implementations and a tokenising difference cannot be mistaken for an
  arithmetic one. On f32 weights the unsplit engine generates exactly what
  llama.cpp generates; so does a three-stage split, compared against llama.cpp
  running the model whole. Every quantised format — `q4_0`, `q4_1`, `q5_0`,
  `q5_1`, `q8_0`, `f16`, `bf16` — decodes bit-identically to llama.cpp's own
  decoding, checked element by element on every tensor.
- **CI installs the reference implementation and sets
  `HOCMESH_REQUIRE_REFERENCE=1`**, so a missing llama.cpp fails the build rather
  than skipping the comparison. A skip and a pass look identical in a log.

### Fixed: a model with a tied output head could not be split

Most small models — SmolLM2, Qwen3 0.6B, Llama 3.2 1B — have no
`output.weight` and reuse the embedding table as the output head. The engine
refused to load the last stage of such a model unless that same stage also held
block 0, which for any real split it does not. The whole class was
unsplittable, and the fixture never caught it because the fixture writes a
separate head by default.

The check was asking the wrong question: it tested *where the stage sits*
rather than *whether the table is there*. A shard already carries the shared
tensors whichever end of the model it holds, so the table was present the whole
time. `Stage::load` now reads it, and the refusal is kept only for a file that
genuinely has neither a head nor an embedding table.

### Also

- **`model-fixture --weights`** writes the fixture in any supported format
  (`f32`, `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`) rather than
  only f32, which is what lets the parity test cover the quantised paths.
- **The fixture now carries tokenizer metadata.** The engine never reads it —
  it takes token ids directly — but llama.cpp refuses to load a model without
  it, so without these entries the fixture could not be handed to the reference
  implementation at all.

## 0.5.0

### A model now runs across machines none of which hold it whole

This is the thing the rest of the system existed to make possible, and until
this release it was the one piece missing. hocMESH could already say which
layers a machine should hold, price the work, settle it and move the bytes. A
stage was then executed by an external llama.cpp process, which loads whole
models — so a model too large for any single participating machine could be
planned and paid for but never run.

- **New crate `hocmesh-engine`** — loads blocks `[start, end)` out of a GGUF file
  and runs a forward pass over an activation handed to it. RMS norm, RoPE,
  grouped-query attention and SwiGLU, over F32, F16, BF16, Q8\_0, Q4\_0, Q4\_1,
  Q5\_0 and Q5\_1. Nothing in a stage reaches for a weight outside its own range,
  which is what lets it run on a machine that holds only its own layers.
- **The architecture is read, never asserted.** An architecture the engine has
  not been told about is refused at load. A wrong RoPE layout does not fail — it
  generates fluent nonsense — so guessing is not on the table.
- **`hocmesh stage-serve`** puts one layer range behind an HTTP port with the
  address of the next stage; the tail turns the activation into logits and the
  answer walks back down the chain. **`hocmesh stage-run`** drives a chain, or
  runs the same model whole in one process to compare against.
- **`hocmesh model-shard`** materialises only the bytes a stage's layers need.
  The file is created at the model's full declared length so every tensor sits
  where the header says it does, and the rest is a hole. A `.shard.json` sidecar
  records which bytes are real, because **a hole reads back as zeros, and zeros
  are a valid weight matrix** — a stage with a zeroed layer does not crash, it
  generates confident nonsense. `stage-serve` checks the sidecar before reading a
  single weight and refuses to start with the missing range named.
- **`hocmesh model-fixture`** writes a small but genuine GGUF file — the real
  format, through the real header reader, tensor directory and dequantiser — so
  the split can be exercised without downloading gigabytes.

### The proof, rather than the description

- **`split_matches_whole.rs`** — the same model cut two, three, four and eight
  ways produces **bit-identical** logits, for every supported weight format and
  at several grouped-query head ratios. Not approximately: each block's
  arithmetic depends only on the activation it was handed and its own weights, so
  there is no rounding difference to tolerate, and tolerating one would hide a
  real divergence. Activations round-trip through the wire encoding on every hop,
  so the test exercises the bytes a real pipeline sends.
- **`distributed_inference.rs`** — three separate OS processes, three shards each
  holding about 41% of the file (asserted by reading it: bytes present below
  bytes total, chunks kept below chunks total, share under half), chained
  together, generating output identical to the same model run whole in one
  process, down to the SHA-256 of the logits. A second test points a stage at a
  shard that does not hold its layers and asserts it refuses to start.
- **A guard on the guard.** `inf == inf`, and two identical NaN bit patterns
  compare equal — so a model that had saturated would pass every bit-exactness
  assertion above while computing nothing. This was not hypothetical: the fixture
  generated weights in `[-0.25, 8191.75]` because a 24-bit value was divided by
  2^11, the attention softmax saturated, the residual stream ran to infinity, and
  the first end-to-end run "passed" with every argmax pinned at token 0. The
  divisor is fixed, the harness asserts finiteness at every step, and a dedicated
  test asserts the fixture produces a real logit spread and more than one winning
  token across positions.

### A validator that missed a commit now catches itself up

A quorum certificate only applies on top of the head it names, so a seat that
missed a single commit — a dropped connection, a process killed mid-round, a
moment of being busy — refused every entry after it as well, **for as long as it
ran**. The only cure was an operator noticing and running `validator sync` by
hand. In the meantime that seat kept answering balance queries: not with an
error, but with a stale number signed as though it were current. Two seats in
that state deny a quorum on an account nobody disagrees about, which is how it
surfaced — a load test failing on the first thing it does, reading the balance
the run is about to be measured against.

- **`catch_up_to`** in the validator fetches and applies what a seat missed,
  verifying every certificate against the set that governed its height, exactly
  as `sync` and an offline audit do. It runs when a certificate lands above the
  local head, and on a one-second heartbeat so a gap no later commit reveals
  still closes.
- **`fetch_certificates` no longer stops at the first seat that answers.** A seat
  that is itself behind returns an empty page for a height it has not reached,
  and taking that as the answer meant a validator catching up on the entry it
  missed could be sent away by another validator missing the same entry. Empty
  from *every* seat still means end of chain, which is what callers walking the
  chain forward rely on. Each refusal is now reported, because "no validator
  could provide ledger entries" is not something anybody can act on.
- **`balance_quorum` waits briefly for the set to agree with itself.** An entry
  commits the moment `threshold` seats have stored it, so for a short while the
  rest are still attesting the balance from just before it. Lag closes; a real
  split does not, and the error after the budget names which case it was and what
  each camp of seats was holding.
- **The quorum tests now prove the seat heals itself.** A killed validator is
  restarted with nothing done to it — no `sync`, no operator — and must be
  observed behind and then converge. `sync` is still exercised, as the path for a
  seat that is not running.

### Scarcity is a ranking term, never a price

A 48 GB card and an 8 GB card earned the same for the same shard, and they have
to: the reward is `work_cost_mcu` of the spec, every validator recomputes it from
`split_work` before signing, and a coordinator paying a premium for scarce
hardware would be proposing a settlement the quorum rejects on sight. Pricing is
not available as a lever and should not be.

- **A fifth scheduling axis, `scarcity`** (weight 0.10), scores a shard's declared
  working set against the machine offering to hold it, preferring the smallest
  machine that fits and ranking a GPU node down for work that cannot use a GPU.
  The large machine is still offered anything nobody else can take.
- **Reputation was already in the score** and the README said otherwise.
  `standing()` folds a Laplace-smoothed acceptance ratio and the node's current
  audit rate into the reliability axis. The stale paragraph is corrected.

### Packaging and documentation

- **RPM packaging**, the one installer format the release workflow was missing:
  `packaging/linux/hocmesh.spec.in`, `scripts/package-linux-rpm.sh`, an `rpm`
  target on the desktop bundle, and both in the release job with checksums and
  signatures. The machines most likely to lend spare capacity are servers, and a
  large share of servers are not Debian. Both scripts open the finished package
  and check its contents and its relationship fields before handing it over.
- **`docs/ROADMAP.md`** claimed "signed protocol v4 requests" while
  `PROTOCOL_VERSION` was 6. Priority 6 and the north-star section are rewritten
  against what now exists, including what it is *not*: three processes on one
  host is the protocol proved, not a deployment, and the Priority 1 multi-host
  run still stands open.
- **`README.md`'s runtime boundary** described a single llama.cpp path. There
  are two paths now and they do different jobs, so it says which is which and
  that they do not yet meet — accelerated means whole models, split means CPU.
- **`docs/IMPLEMENTATION_STATUS.md`** gained rows for the engine, exact split
  execution, distributed inference and partial materialisation, and one row for
  what is *not* done: the forward pass is not compared against llama.cpp on a
  converted model. The split is proved to change nothing; that the unsplit
  result matches another implementation is not, and no document here should
  imply otherwise.
- **A temporary-directory collision** in the quorum integration tests. The
  directory name was a nanosecond timestamp, and Windows advances the system
  clock in ~15 ms steps — so two tests starting in the same tick shared a
  directory and one deleted the other's ledger, keys and databases on drop. It
  now carries the process id and a counter, matching the engine tests.

### A model is split by memory bandwidth, not by stage count

`plan_parallelism` divided a model's layers evenly across pipeline stages, so a
fast machine paired with a slow one ran the whole pipeline at the slow one's
pace. Decode re-reads every weight in a stage per token, so stage time is
`bytes / bandwidth` and the stages finish together only when layers are
proportional to bandwidth.

- **Added `benchmark_memory_bandwidth()`** (`hocmesh-core/src/hardware.rs`) — a
  sequential read over a buffer larger than any last-level cache, four
  accumulators so the loop waits on memory and not on the add chain,
  `black_box` so it cannot be optimised away. Best of three passes, since every
  error source makes the figure look worse than the hardware is. Returns `None`
  rather than a guess if it cannot measure.
- **`memory_bandwidth_bytes_per_second`** added to `NodeCapabilities`,
  `DeviceCapability`, `hocmesh-ai`'s `NodeProfile` and `CandidateScore`, all
  `#[serde(default)]` so existing `capabilities_json` rows still load.
- **The GPU figure now reaches the planner.** `protocol_gpu_to_device` measured
  `benchmark_bytes_per_second` at registration and then dropped it, leaving the
  one component that needs it unable to see it. A device without its own
  measurement falls back to its node's.
- **`layer_spans`** allocates layers by largest remainder, ties broken on stage
  index so two coordinators plan identically. Any unmeasured stage sends the
  whole split back to uniform — a default would be an unpredictable error, not
  a smaller one. Every stage keeps at least one layer, as a repair after the
  proportional split rather than a reservation before it.

### Fast nodes get a bounded head start on contended work

Pull-based scheduling answers whoever polls, so the previous release's
inclusion fix let a slow node take work a fast node would have finished sooner.
A coordinator cannot reserve a shard for a peer that has not asked — but it can
answer "not yet".

- **`Unfit::StillReservedForFasterNodes`** defers a node for
  `HEAD_START_SECONDS * (1 - hardware)`, capped at 30s and zero for the fastest
  machine, so the mesh cannot deadlock with everyone deferring.
- **Only under contention.** `Scale::contended()` is `recent_pollers >
  pending_shards`; with a shard for everyone, holding one back delays the job
  for no gain. Measured from shard creation, so an aging queue opens to all.
- **`recent_pollers`** is one indexed `COUNT(*)` over `nodes.last_seen`, which
  every poll already writes; new index `idx_nodes_last_seen`. Passing `0` means
  "not measured" and reproduces the previous behaviour exactly.

### Modest hardware is scheduled, not excluded

A node predicted to take longer than the flat 900-second lease was refused the
shard outright (`Unfit::SlowerThanLease`). The lease is a timeout, not a price —
nothing on the chain reads it, and a shard is worth the same mCU however long
its holder took — so a constant that knew nothing about the machine was deciding
which machines were allowed to contribute at all. On a network whose premise is
lending hardware you already own, that is the wrong answer.

The lease is now sized to the node taking the shard: `predicted × 1.5`, floored
at the old default so nobody loses time they had before, and capped at
`MAX_LEASE_SECONDS` (3× the default). A machine roughly three times slower than
a current laptop — a decade-old workstation, say — now earns on the same shards
at the same price, and simply takes longer over them.

Past the ceiling `SlowerThanLease` still applies, because at that point the mesh
really is better off giving the shard to somebody else.

`SETTLEMENT_WINDOW_SECS` is unchanged and so is `PROTOCOL_VERSION`: the new
ceiling was chosen to fit inside the existing window, which is consensus-visible.
A compile-time assertion now enforces that the longest lease stays shorter than
the window it settles in, so widening it later has to be a deliberate protocol
change rather than a quiet break in settlement for exactly the slow nodes this
was meant to help.

Shard *sizing* is deliberately untouched. Validators recompute a job's price
from `(work, shards)`, so the shard count is part of what quorum signs — cutting
smaller shards for slower machines would have the coordinator charge a number
the validators reject.

### `loadtest-local.ps1` can build its own binaries again

The one native call in the script that was not routed through `Invoke-Native`
was the `cargo build` at the top, so an ordinary `Compiling hocmesh-protocol`
on stderr became a terminating `NativeCommandError` and the run died before it
started. It only ever surfaced without `-SkipBuild`. Moving the helper above its
first use was part of the fix — a PowerShell script runs top to bottom, and the
function was declared eighty lines below the call.

### A quorum can agree about a balance from different heights

`balance_quorum` grouped validator proofs on the whole tuple including
`head.sequence` and `head.entry_hash`, so a quorum had to be at the byte-identical
ledger head at the same instant. A validator's head moves whenever *any* account
transacts, so under concurrent submits the same balance is routinely attested
from different heights, and those genuine agreements were thrown away — the
coordinator returned `409 no quorum-agreed balance` for an account no validator
disagreed about. The new load test reproduced it at roughly one job in
twenty-four with eight concurrent submitters.

Proofs are now grouped on `(balance, earned, spent)`, which is the claim being
agreed on; the head stays inside each signed proof as provenance for when that
validator said it. Where two groups both reach threshold the freshest wins, so
a lagging quorum cannot hold the answer back. The threshold itself is unchanged.

### An account can move to a new machine

Nothing about an account was ever tied to the hardware it was made on — the
balance is what the ledger implies for a public key, not a row on your disk —
but until now the only way to act on that was to copy `identity.json` by hand
and hope. `hocmesh identity show | export | import | inspect` makes it a
supported operation.

- A backup is **always** sealed (XChaCha20-Poly1305 under Argon2id), even when
  the node it came from stores its own key unsealed, because the copy people
  actually make ends up in cloud sync or a chat message.
- The account id and public key stay readable, so `identity inspect` answers
  "whose account is this?" without a passphrase — and a backup whose header
  disagrees with the key inside it is refused, so that readable header can never
  become attacker-controlled text you trusted.
- Importing over a *different* account needs `--force`, and so does importing
  over an identity that will not open. Under `--force` the displaced key is
  renamed to `identity.json.replaced-<timestamp>`, never deleted.
- Restoring the account already on the machine is a no-op, so "did my backup
  work?" is a safe question.
- `hocmesh identity …` dispatches **before** an identity can be created, so
  asking about an account no longer mints one as a side effect.
- Machine binding was considered and rejected: it would trade a recoverable loss
  for an unrecoverable one and protect nothing, because the ledger's safety is
  signatures and quorum, not the location of a key.

### Artificial load is a shipped command

Every hard bug this ledger has had was a race, and a race needs contention that
a single developer machine never produces by hand.

- `hocmesh loadtest` submits concurrent jobs and then **audits the economy it
  just stressed**: reserved CU must equal recorded spend, the account's banked,
  earned and consumed figures must agree, and ledger height must not go
  backwards. It fails on unsettled work or CU that do not add up — never on
  being slow, because a latency threshold on shared CI is a flaky test.
- `--dry-run` prices a plan through the same function the ledger charges with,
  so a harness can wait for exactly enough CU instead of guessing at a sleep.
- `scripts/loadtest-local.sh` and `scripts/loadtest-local.ps1` stand up a whole
  network — four validators at threshold three, a coordinator, worker nodes —
  mint the community work that funds the run, apply the load, and finish by
  re-auditing the ledger from genesis.
- CI runs it on every push and keeps the JSON report and process logs as
  artifacts, including when the run fails.

## v0.4.0 — one install, one peer

### There is no client and no server

Every hocMESH install is now a whole peer: node, coordinator and validator, on
every machine. The model is a torrent swarm run in the other order — you seed
hardware first, and what that earns is what lets you later borrow somebody
else's. The only difference between the two installers is whether the machine
has a screen, so **they replace each other rather than sitting side by side**.

- Both installers carry all three binaries; the desktop one adds the window.
- Both claim `/usr/bin/hocmesh`, and both declare it: the desktop `.deb` carries
  `Provides`/`Conflicts`/`Replaces: hocmesh`, the headless one `Conflicts` and
  `Replaces: hoc-mesh-desktop`. Installing either over the other swaps it
  cleanly instead of failing on a dpkg file collision. `package-desktop.sh`
  rewrites those three fields into the built artifact and then asks dpkg to
  read them back: the bundler was asked for them in `tauri.conf.json` and
  shipped a package dpkg did not agree carried them, so the packaging script
  no longer trusts it. It rewrites rather than adds, because Debian field
  names are case-insensitive and appending to a field already present under
  another casing produced a duplicate that failed the rebuild. It also reads
  the desktop package's own name off the artifact and fails unless the
  headless `control` names that exact string, so a rename cannot quietly leave
  two packages that both own `/usr/bin/hocmesh`.
- Client/server language is gone from packages, scripts, CI and docs.
- `scripts/install-user.sh` / `.ps1` install all three binaries for one user.

### Models can now be addressed by layer

`hocmesh-model`'s GGUF reader went from metadata to the tensor directory.

- `gguf::tensor_directory` reads every tensor's name, type code, shape and
  offset, plus the declared alignment and where the data section starts.
- `tensors_for_layers` selects a stage's tensors, `shared_tensors` returns the
  embeddings and output head that belong to no block, `extents_for_layers`
  merges a selection into byte spans, and `chunks_for_extents` turns those into
  the chunk indexes to fetch.
- `block_layout` gives bytes per block for every GGML type this build knows and
  answers `None` — never a guess — for one it does not.
- New command: `hocmesh model-inspect <file> --stages N`. It reads the header
  only, so it works on a partly-fetched file.

This is what lets a peer fetch the layers it will run instead of the whole file.
On a 32-block, 26 MB model split four ways, a middle stage pulls 3.5 MB in one
span — one chunk of seven.

### A lost ledger race no longer settles as a rejection

A `Rejected` is final — the round loop returns it to the caller without
retrying — so a batch that was never actually refused must never be reported as
one. Three windows could do that under load.

The first: a proposer that lost a height race saw a threshold of refusals while
`head_quorum()` still reported the old height. One signed head at or past the
contested sequence is now enough to conclude the race was lost, because
retrying is always safe — a round that fell short applied nothing — while
*settling* needs a quorum.

The second is narrower and was caught by a coverage run, where instrumentation
slows everything enough to widen it: the winner's entry is applied but its
signed head has not come back yet, so every seat refuses and no head reads as
taken. A vote is now only counted as a verdict on the batch if the seat was
building on the same head the proposer was — `ProposalVote` carries
`head_sequence`, and a seat at another height, or one that signed an entry
other than the one put to it, is telling the proposer its head is stale rather
than refusing the transactions. Below a threshold of judging seats the round
defers and re-reads the head.

The third is the ordinary end of a round nobody observed. The transaction is
applied, the certificate never gets back, the proposer climbs, and the
validators turn it away with `claim already settled` — its own success,
refusing it. Retrying cannot help here and neither can deferring: the work is
done. Told "rejected" about work the ledger did, a caller runs a job that is
already paid for and reports a reservation that exists as missing. A refused
round is now resolved against the ledger before the error is handed back, and a
transaction the quorum can show is already committed comes back with the
certificate of the entry that carries it, at the height it landed at. The match
is by transaction hash, not by claim key: a claim key is shared by every
transaction settling the same claim, so a key-only match would hand back a
certificate for somebody else's transaction — including the one case this
ledger exists to refuse, a second reward for one assignment under different
numbers. That coordinator is still told no. Anything that does not resolve
keeps its original error.

Refusals also carry their reasons out of the round now. The old message was
`received only 0 valid votes; threshold is 3`, which said nothing about why;
it now names the validators and quotes what each said, or states plainly that
every seat accepted a different entry than the one proposed. Ten new tests in
`hocmesh-ledger/src/network.rs` cover the first two rules, and
`a_settled_transaction_resolves_to_its_entry_and_an_impostor_does_not` in
`hocmesh-integration-tests` covers the third against four live validators.
`head_sequence` is `#[serde(default)]`, so a validator that predates it still
votes and is read as in step.

### Release signing

- `scripts/sign-artifacts.ps1` (Authenticode, timestamped) and
  `scripts/sign-artifacts.sh` (macOS Developer ID + notarisation, or a
  GPG-signed checksum list on Linux), wired into `release.yml`.
- Windows and macOS artifacts are signed **before** checksums are taken, so the
  published digest is the digest of the file a user runs.
- Signing is off until keys are configured and says so out loud;
  `HOCMESH_SIGNING_REQUIRED=1` turns a missing key into a build failure.

### Licence

hocMESH is now closed source under a proprietary licence
(`LicenseRef-hocMESH-Proprietary`). The repository is private. Third-party open
source components keep their own licences and ship as a CycloneDX SBOM.

### Documentation

- `docs/DISTRIBUTED_INFERENCE.md` — what the physics allows: the arithmetic for
  tensor, pipeline and MoE splitting, why speculative decoding is the highest-
  leverage piece, and the seven-step build order. Step 1 is done.
- `docs/DISTRIBUTION.md` — what signing and a private repository do and do not
  protect, stated plainly.
- README gained an Architecture section: the torrent model, the three planes,
  how proximity picks a machine, where hardware inequality is handled, and what
  an account actually is.

### Still not true, and said so in the README

**Nothing here performs distributed inference of a single model across
machines.** The planner cuts a model into layer ranges and the transport carries
activations, but no engine loads a *layer range* and runs a forward pass — stock
llama.cpp loads whole models. That is step 2 of the build order and it is the
load-bearing gap.

## v0.3.0

Distributed hocMESH AI and Compute Core runtime: model registry and
content-addressed chunks, peer-to-peer seeding, CUDA/ROCm/Metal backends, GGUF
and safetensors manifests, GPU benchmarking, latency-aware scheduling,
pipeline/model/batch parallelism, tensor transport, failure-aware rerouting, and
the Tauri desktop app.
