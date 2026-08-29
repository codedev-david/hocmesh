# Distributed inference: what the physics allows

The goal is that an ordinary laptop runs a large model by borrowing the rest of
the machine from peers nearby. This document states what is built, what is
missing, and — with the arithmetic rather than the hope — which ways of
splitting a model can actually survive a network between the pieces.

Everything here is arithmetic anyone can redo. Where a number depends on a model
shape, the shape is stated.

---

## 1. What already exists

| Piece | Where | State |
|---|---|---|
| Proximity map (Vivaldi coordinates, predicted RTT between any two peers) | `hocmesh-core/src/proximity.rs` | Working |
| Ranking by distance **to the requester** | `hocmesh-coordinator` `scoring_latency_ms` | Working |
| Layer-range planner (`PipelineStage { layer_start, layer_end }`) | `hocmesh-ai` `plan_parallelism` | Working, validated for gaps and overlaps |
| Tensor-group and batch planners | `hocmesh-ai` | Working |
| Ordered, checksum-bound tensor framing with replay rejection and route failover | `hocmesh-transport` | Working |
| Content-addressed model chunks, rarest-first peer seeding | `hocmesh-model`, node peer server | Working |
| Two-stage inference settlement through the ledger | `hocmesh-ai`, `hocmesh-ledger` | Working |
| GGUF **metadata** reader (architecture, name, layer count) | `hocmesh-model/src/gguf.rs` | Working |
| GGUF **tensor directory** reader: name, type, shape, offset; byte extents and chunk indexes per layer range | `hocmesh-model/src/gguf.rs`, `hocmesh model-inspect` | Working |
| **Layer-range execution**: load blocks [start, end), run a forward pass over an activation handed in | `hocmesh-engine`, `hocmesh stage-serve` | Working, CPU |
| **Distributed inference of one model across processes**, bit-identical to running it whole | `hocmesh stage-serve` + `stage-run`, proved in `distributed_inference.rs` | Working |

## 2. What was missing, and is not any more

For every release up to 0.4 this section named one gap, and it was the middle of
the sandwich:

> **An execution engine that loads a layer range and runs a forward pass over an
> activation it was handed.**

A stage was executed by an external llama.cpp process. llama.cpp loads *whole
models*; it has no interface for "load layers 24–47, take this hidden state, give
me the hidden state after layer 47." So the planner could cut a model into stages
that nothing could run.

That engine is `hocmesh-engine`, added in 0.5.0. The three sub-pieces:

1. ~~A GGUF **tensor** reader~~ — **done.** `gguf::tensor_directory` reads the
   directory, `tensors_for_layers` selects a stage's tensors, and
   `extents_for_layers` turns them into merged byte spans and chunk indexes.
   `hocmesh model-inspect <file> --stages N` prints them.
2. ~~A forward pass~~ — **done.** `hocmesh-engine`'s `Stage` holds blocks
   `[start, end)`, owns the KV cache for exactly those blocks, and runs RMS norm,
   RoPE, grouped-query attention and SwiGLU over an activation handed to it.
   Weights are read through `WeightFile` one tensor at a time, so a stage never
   touches a byte outside its own range. F32, F16, BF16, Q8\_0, Q4\_0, Q4\_1,
   Q5\_0, Q5\_1 and the k-quants Q2\_K through Q6\_K are decoded; anything else
   is refused at load rather than guessed at, as is any file carrying a tensor
   or a metadata key this build would not act on.
3. ~~The stage protocol~~ — **done for the single-sequence case.** `stage-serve`
   exposes `GET /stage/info`, `POST /stage/token`, `POST /stage/forward` and
   `POST /stage/reset`; each stage forwards to its `--next` and the tail's logits
   return down the chain. Position travels with the activation rather than being
   counted by the stage, so a stage cannot silently disagree with its neighbours
   about where in the sequence it is, and a gap in positions is an error rather
   than a hole.

### What is still open

- **Model families the engine does not implement.** The block shape it runs is
  `attn_norm -> q,k,v -> rope -> attention -> out, ffn_norm -> gate*up -> down`
  with RMS norm and SwiGLU, plus optional per-head query/key norms and
  projection biases. Mixture-of-experts models are not among them: routing
  means the weights a token needs are chosen per token, which a static
  layer-range split does not express.

  What decides whether a file loads is now the file, not its architecture
  string. Every tensor of every block is enumerated and one this build would
  not read is refused; every tensor that is read must have the shape the header
  implies; every metadata key that would change the maths -- sliding windows,
  expert counts, logit softcapping, ALiBi, rotary scaling -- is refused when it
  is set. The name list survives for one question only, and it is the one a
  GGUF file cannot answer: which pairs of elements a rotary embedding rotates
  together, which llama.cpp fixes per architecture in its own source. A model
  published under a name this build has not been told about is refused with
  that named as the missing fact, and `HOCMESH_ASSUME_ARCHITECTURE=llama` (or
  `qwen3`, ...) supplies it -- after which every check above still has to pass.

  Two axes stay unanswerable and the override says so: a LayerNorm model with
  no norm bias, and a GELU feed-forward, carry the same tensors under the same
  names as the shapes this build runs. That is why the override is opt-in
  rather than a default guess. `stablelm` is the cautionary case -- it was on
  the known list and generated fluent, wrong text, because llama.cpp normalises
  it with LayerNorm.
- **Parity on a downloaded checkpoint, on every push.** The forward pass is
  checked against llama.cpp -- see below -- on a generated fixture, because a
  fixture is what CI can afford to download and run. The same comparison has
  been run by hand against a real checkpoint (SmolLM2-135M: 30 blocks,
  grouped-query attention, a tied output head), whole and split three ways, and
  it matched on every prompt tried. What is missing is not the evidence but the
  automation: nothing re-runs that on each commit.
- **Resuming a sequence elsewhere.** A stage that drops mid-sequence takes its KV
  cache with it. Re-running the prompt against a replacement is correct and
  costs the prompt again; migrating the cache is not implemented.
- **Batching across sequences.** One sequence at a time per chain.
- **GPU execution of a stage.** The engine is CPU. GPU inference still goes
  through the llama.cpp adapters, which means it still loads whole models — so
  the *distributed* path and the *accelerated* path do not currently meet.
- **Throughput.** The engine is written for correctness and for splitting. A
  model that fits on one machine runs faster through llama.cpp.

### The property that makes it worth having

Split execution is **bit-identical** to whole execution, not approximately equal.
Each block's arithmetic depends only on the activation it was handed and its own
weights, so a correct split has no rounding difference to tolerate — and
tolerating one would hide a real divergence. That is asserted, over the wire
encoding, in `crates/hocmesh-engine/tests/split_matches_whole.rs` for two, three,
four and eight-way cuts, every supported weight format, and several
grouped-query head ratios; and end to end across three OS processes holding
disjoint shards in
`crates/hocmesh-integration-tests/tests/distributed_inference.rs`.

---

## 3. The arithmetic that decides the design

Three ways to split a model. They are not equally viable over a network, and
the difference is large enough to determine the whole design.

Assume a dense model with hidden size `H = 4096`, `L = 64` layers, activations
in fp16 (2 bytes). Substitute your own; the conclusions move slowly.

### 3.1 Tensor parallelism — split every matmul

Each layer needs **two all-reduces** of the hidden state: one after attention,
one after the FFN. An all-reduce is a *serialized* collective — every rank waits
for every other.

```
round trips per token = 2 x L = 128
bytes per token       = 2 x L x H x 2 B  ~=  1 MB   (ring, 2 ranks)
```

128 serialized round trips per token:

| Interconnect | RTT | Added latency per token | Ceiling |
|---|---|---|---|
| NVLink / PCIe in one box | ~5 us | 0.6 ms | ~1500 tok/s |
| Datacenter RDMA | ~50 us | 6 ms | ~160 tok/s |
| Good LAN (1 GbE) | ~1 ms | 128 ms | **~8 tok/s** |
| Same city, internet | ~20 ms | 2.6 s | **0.4 tok/s** |

**Verdict: tensor parallelism is unusable outside a single machine or an RDMA
fabric.** It is not a tuning problem — it is two orders of magnitude. The
`ModelTensor` planner in `hocmesh-ai` should be understood as multi-GPU-in-one-box
support, not as a mesh strategy.

### 3.2 Pipeline parallelism — split by layer range

Each token crosses each stage boundary **once**, carrying one hidden state.

```
bytes per token per boundary = H x 2 B = 8 KB
round trips per token        = S - 1     (S = stages)
```

Bandwidth is a non-issue: 4 stages at 100 tok/s is 2.4 MB/s, about 19 Mbps.
Latency is the whole story, because decoding is autoregressive — token *n+1*
cannot start until token *n* has been all the way through.

| Peers' distance | RTT | S=2 | S=4 | S=8 |
|---|---|---|---|---|
| Same LAN | 1 ms | ~1000 tok/s | ~330 tok/s | ~140 tok/s |
| Same city / same ISP | 10 ms | ~100 tok/s | ~33 tok/s | ~14 tok/s |
| Same country | 40 ms | ~25 tok/s | ~8 tok/s | ~3.5 tok/s |
| Cross-continent | 150 ms | ~7 tok/s | ~2 tok/s | ~1 tok/s |

(Network ceiling only; real speed is `1 / (network + compute)`.)

**Verdict: pipeline parallelism works, and proximity is exactly the right
lever.** The difference between "same city" and "cross-continent" is 15x. This
is precisely what the Vivaldi coordinate system was built to exploit, and it is
why ranking is done against the *requester's* position rather than the
coordinator's.

Two corollaries worth stating plainly:

- **Fewer stages beat more stages.** Every extra peer buys memory and costs a
  round trip per token. Use the smallest number of stages that fits the model.
- **Batching helps throughput, never single-stream latency.** A pipeline with
  microbatches keeps every stage busy and serves many users well. It does not
  make one user's cursor blink faster. Do not conflate the two when reporting.

### 3.3 MoE expert offload — split by expert

For a Mixture-of-Experts model, only a few experts fire per token. If peers hold
experts, the laptop keeps the attention stack and routes to whoever holds the
active experts.

```
bytes per token per expert = H x 2 B x 2 (in and out) = 16 KB
round trips per token      = 1  (experts are queried in parallel, not in series)
```

The active experts for a token are known at once, so they are fetched
**concurrently**: one round trip per token regardless of how many experts fire.
That is the same cost as a 2-stage pipeline, for a much larger model.

**Verdict: MoE is the best structural fit for a mesh that exists.** It converts
"model too big for this laptop" into "hold the router and the attention stack
locally, borrow the experts" — and the experts a given user hits are stable
enough that caching works.

### 3.4 Speculative decoding — the actual latency answer

The technique that breaks the round-trip-per-token barrier. The laptop runs a
**small draft model locally at full local speed** and proposes *K* tokens. One
nearby peer holding the big model **verifies all K in a single forward pass**,
because verification is parallel over positions in a way generation is not. Every
accepted prefix token is exactly what the big model would have produced — this
is lossless, not an approximation.

```
round trips per K tokens = 1
effective tokens per round trip = expected accepted prefix ~= 4-6 for K=8
```

| Peer distance | RTT | Naive (1 RT/token) | Speculative, K=8, ~5 accepted |
|---|---|---|---|
| Same LAN, 1 ms | 1 ms | ~1000 tok/s | ~5000 tok/s |
| Same city, 10 ms | 10 ms | ~100 tok/s | ~500 tok/s |
| Same country, 40 ms | 40 ms | ~25 tok/s | ~125 tok/s |
| Cross-continent, 150 ms | 150 ms | ~7 tok/s | ~33 tok/s |

**Verdict: this is the single highest-leverage piece of the whole vision.** It
turns latency from a per-token tax into a per-*chunk* tax, and it composes with
pipeline parallelism and with MoE offload rather than competing with them.

### 3.5 The honest summary

> Splitting *matmuls* across a network does not work and never will. Splitting
> *layers* works and is bounded by round trips, so proximity buys 10-15x.
> Splitting *experts* works better still. Amortising round trips with
> speculative decoding buys another 5x on top, and is the difference between a
> usable cursor and a slideshow.
>
> The vision is sound. The route to it is pipeline + MoE + speculation over
> proximity-ranked peers — not tensor parallelism.

There is also a benefit that needs no parallel speedup at all and is worth
shipping first: **a laptop that cannot hold a model can still run it** if peers
hold the weights and stream them. Slower than a datacenter, and still the
difference between running the model and not running it.

---

## 4. Build order

Each step is independently testable, and each is worthless without the one
before it. Do not reorder. Steps 1 to 4 shipped in 0.5.0; the rest are still in
this order and still for the same reasons.

**Step 1 — GGUF tensor reader. Done.** `gguf::tensor_directory` reads name,
type code, shape and offset for every tensor, plus the declared alignment and
where the data section starts. `TensorInfo::layer_index` reads the block a
tensor belongs to from its name, `tensors_for_layers` selects a stage's set,
`shared_tensors` returns the embeddings and output head that belong to no
block, `extents_for_layers` merges the selection into byte spans, and
`chunks_for_extents` turns those into the chunk indexes to fetch.
`block_layout` gives bytes per block for every GGML type this build knows and
answers `None` — never a guess — for one it does not.

*Tested:* every tensor lands inside the file; layer ranges partition the block
tensors with nothing claimed twice and nothing left over but the shared set;
two stages ask for disjoint bytes and neither asks for the whole file; a header
cut short reads as absent rather than corrupt; overlapping or past-the-end
tensors are refused; an unknown type reports no length rather than a wrong one.
The end-to-end check is `hocmesh model-inspect` against a GGUF written by a
separate implementation of the spec, where the four stage extents plus the
shared set sum to exactly the data section — a partition, verified rather than
asserted.

This is also what lets the chunk store fetch *only the layers a stage needs*.
On a 26 MB, 32-block file split four ways, a middle stage pulls 3.5 MB in one
span — 1 chunk of 7 — instead of the whole file.

**Step 2 — CPU forward pass over a layer range. Done.** `hocmesh-engine`'s
`Stage::load(&mut file, start..end)` reads only the tensors for blocks
`[start, end)` plus whatever shared tensors that range genuinely owns, and
`Stage::forward(&activation)` runs RMS norm, RoPE, grouped-query attention and
SwiGLU over a hidden state it was handed. F32, F16, BF16, Q8\_0, Q4\_0, Q4\_1,
Q5\_0, Q5\_1 and the k-quants Q2\_K through Q6\_K are decoded; anything else is
refused. What a file is allowed to be is decided by the tensors and metadata it
actually holds; the architecture string in the header answers one question the
file cannot, the RoPE pair layout, because a wrong RoPE layout does not crash —
it generates fluent nonsense.

*Tested:* `crates/hocmesh-engine/tests/split_matches_whole.rs`. The same model
run whole and run in two, three, four and eight pieces produces **bit-identical**
logits — every supported weight format, several grouped-query head ratios,
activations round-tripped through the wire encoding on every hop. An empty
range, a range past the end, and a tied output head split across stages are all
refused at load. And because `inf == inf` and identical NaN patterns compare
equal, a separate test asserts the fixture produces finite logits with a real
spread and more than one winning token — otherwise every comparison above would
pass while computing nothing.

*Also tested, and it is the comparison everything else here leans on:*
`reference_parity.rs` checks the forward pass against llama.cpp. llama.cpp runs
as a server and is fed token ids rather than text, so no tokeniser sits between
the two implementations and a tokenising difference cannot be mistaken for an
arithmetic one. On f32 weights the unsplit engine generates exactly what
llama.cpp generates; so does a three-stage split, compared against llama.cpp
running the model whole. Separately, every quantised format this engine reads
— `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `f16`, `bf16`, and the k-quants
`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` — decodes bit-identically to llama.cpp's
own decoding of the same bytes, checked element by element on every tensor.

The k-quants are checked separately and on a wider fixture, because a k-quant
super-block is 256 elements across and llama.cpp will not store a row that is
not a multiple of that — it quietly picks another type instead. The assertion
is over *types decoded* rather than tensors compared, since asking for
`q4_k_m` gets a per-tensor mixture: a count could not tell "Q4_K decoded
correctly" apart from "nothing in this file was Q4_K".

This build reads the k-quants and deliberately does not write them.
Quantising well is a search for per-sub-block scales, not a formula, and a
worse encoder would produce files that load, run at full speed, and generate
measurably worse text than the same model quantised by llama.cpp — with
nothing to report as an error. `quantize` therefore refuses them by name and
says what to use instead.

Two limits are deliberate. Quantised *generation* is not compared, because
llama.cpp does not decode to f32 and multiply there; it quantises the
activations too and takes integer dot products. That is a different arithmetic
path, so small differences are nobody's bug — and the fixture's random weights
leave its logits nearly tied, so an argmax over them is decided by rounding.
Comparing generated tokens under those conditions measures noise, which is why
what is asserted for quantised weights is the decoding rather than the output.
And the comparison runs on a generated fixture rather than a downloaded
checkpoint; see *What is still open*.

**Step 3 — Stage runner and KV-cache ownership. Done, with the recovery
question answered one way.** A stage owns the KV cache for the blocks it holds
and for the sequence routed to it. Position travels with the activation instead
of being counted locally, so two stages cannot silently disagree about where in
the sequence they are, and a gap in positions is an error rather than a hole.

The choice on a stage dying mid-sequence is **replay from the prompt on a
replacement**, not cache checkpointing. `POST /stage/reset` clears the chain and
the sequence is re-run; correctness costs the prompt again. Migrating a cache is
not implemented and is not pretended to be.

**Step 4 — Activation transport. Done.** `hocmesh stage-serve` puts one layer
range behind an HTTP port with `--next` naming the stage after it; the tail
turns the activation into logits and the answer walks back down the chain.
`hocmesh stage-run` drives a chain, or runs the same model whole in one process
to compare against. `hocmesh model-shard` materialises only the bytes a stage's
layers need, at the model's full declared length so every tensor sits where the
header says, with a `.shard.json` sidecar recording which bytes are real —
because a hole reads back as zeros and zeros are a valid weight matrix.

*Tested:* `crates/hocmesh-integration-tests/tests/distributed_inference.rs`.
Three separate OS processes, three shards each holding about 41% of the file
(asserted by reading it, not by assuming it), chained together, produce output
identical to the whole model in one process down to the SHA-256 of the logits.
A stage pointed at a shard that does not hold its layers refuses to start and
names the missing range.

**Step 5 (next) — Speculative decoding.** Draft locally, verify remotely, accept
the longest matching prefix.
*Test:* output is **token-identical** to non-speculative decoding — that is the
whole correctness claim — and measure the acceptance rate, since it is what
determines the speedup.

**Step 6 — MoE expert routing.** Route by active expert instead of by layer
range; fetch concurrently; cache hot experts locally.
*Test:* identical outputs to a single-process run, and assert the concurrent
fetch really is one round trip and not N.

**Step 7 — Pricing a token.** The rule to keep: the price is a closed-form
function of the *spec* (`work_cost_mcu`), never a measurement of the machine, so
any peer can recheck it and no validator has to trust a self-report. Inference
already settles this way — escrow, receipt, then a signed acceptance or dispute
— so what is left is a spec for the work a stage does, not a new mechanism.

---

## 5. Things that will bite

- **KV cache is the real memory cost, and it is per-sequence.** For long
  contexts it dwarfs the weights on a stage. Plan for it before measuring how
  many peers a model needs.
- **Quantisation must match across stages.** Two peers running different quants
  of the same layer range produce different numerics. Bind the quantisation into
  the model manifest hash so a mismatch is impossible rather than merely
  unlikely.
- **A peer that leaves mid-token is normal, not exceptional.** Home machines
  sleep. Step 3 is not optional hardening; it is the feature.
- **Do not report throughput as latency.** Microbatched pipeline throughput and
  single-stream tokens-per-second are different numbers and only one of them is
  what a user feels.
- **Trust boundary.** A stage that returns a plausible but wrong activation is
  undetectable by checksums — the bytes are intact, the values are lies. The
  existing verification story recomputes deterministic CPU work; it does not
  cover this. Options are redundant execution on a sampled fraction, or
  restricting inference to vouched peers. Decide before opening it up, not
  after.
