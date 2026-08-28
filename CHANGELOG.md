# Changelog

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
one. Two windows could do that under load.

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

Refusals also carry their reasons out of the round now. The old message was
`received only 0 valid votes; threshold is 3`, which said nothing about why;
it now names the validators and quotes what each said, or states plainly that
every seat accepted a different entry than the one proposed. Ten new tests in
`hocmesh-ledger/src/network.rs` cover both rules. `head_sequence` is
`#[serde(default)]`, so a validator that predates it still votes and is read as
in step.

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
