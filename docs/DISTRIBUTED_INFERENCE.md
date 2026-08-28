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
| GGUF **tensor directory** reader: name, type, shape, offset; byte extents and chunk indexes per layer range | `hocmesh-model/src/gguf.rs`, `hocmesh model-inspect` | Working (build order step 1) |

## 2. What is missing

One thing, and it is the middle of the sandwich:

> **An execution engine that loads a layer range and runs a forward pass over an
> activation it was handed.**

Today a stage is executed by an external llama.cpp process. llama.cpp loads
*whole models*; it has no interface for "load layers 24–47, take this hidden
state, give me the hidden state after layer 47." So the planner can cut a model
into stages that nothing can run.

Concretely, three sub-pieces are absent:

1. ~~A GGUF **tensor** reader~~ — **done.** `gguf::tensor_directory` reads the
   directory, `tensors_for_layers` selects a stage's tensors, and
   `extents_for_layers` turns them into merged byte spans and chunk indexes.
   `hocmesh model-inspect <file> --stages N` prints them.
2. A forward pass — attention and FFN over a layer range, with a KV cache that
   belongs to that stage. **This is the load-bearing gap.**
3. The stage protocol — who owns the KV cache across tokens, what happens when a
   stage drops mid-sequence, how a sequence is resumed elsewhere.

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
before it. Do not reorder.

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

**Step 2 (next) — CPU reference forward pass, one architecture, one dtype.** Correctness
first, speed never. `forward(layer_range, hidden_state, kv_cache) -> hidden_state`.
*Test:* run the full layer range in one process and assert the output matches
llama.cpp for the same prompt within tolerance. Without this comparison the rest
is unfalsifiable.

**Step 3 — Stage runner and KV-cache ownership.** A stage owns the KV cache for
the sequences routed to it. Define what happens when a stage dies mid-sequence:
either the sequence is replayed from the prompt on a replacement peer, or caches
are checkpointed. Pick one and write it down.
*Test:* kill a stage mid-sequence, assert the sequence completes and the answer
is identical to the uninterrupted run.

**Step 4 — Activation transport.** Carry the hidden state over the framing that
already exists in `hocmesh-transport` (checksums, ordering, replay rejection,
failover are all done). This step is mostly plumbing, which is the point of
having built the transport first.
*Test:* two stages on two processes produce the same tokens as one process.

**Step 5 — Speculative decoding.** Draft locally, verify remotely, accept the
longest matching prefix.
*Test:* output is **token-identical** to non-speculative decoding — that is the
whole correctness claim — and measure the acceptance rate, since it is what
determines the speedup.

**Step 6 — MoE expert routing.** Route by active expert instead of by layer
range; fetch concurrently; cache hot experts locally.
*Test:* identical outputs to a single-process run, and assert the concurrent
fetch really is one round trip and not N.

**Step 7 — Pricing.** Only now can a token be priced. Keep the existing rule:
the price is a closed-form function of the *spec* (`work_cost_mcu`), never a
measurement of the machine, so any peer can recheck it and no validator has to
trust a self-report.

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
