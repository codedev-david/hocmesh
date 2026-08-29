# Performance

This document answers one question, because it is the question that decides
whether any of the rest matters:

> Can a normal laptop, joined to a mesh, run a frontier-class open model at
> frontier-class speed?

The short answer is that half of it is reachable and half of it is not, and the
half that is not is blocked by arithmetic rather than by engineering. What
follows is the arithmetic, the measurements taken on this machine, and the
ordered plan that gets as close to the goal as the arithmetic allows.

## Decoding is a bandwidth problem

Generating one token requires reading every weight the model will use for that
token, once, from memory into the processor. Almost nothing is reused between
tokens. The processor is idle most of the time waiting for the memory bus, so
the speed of generation is not set by how fast the machine can multiply. It is
set by how fast it can read:

```
tokens per second  =  memory bandwidth  ÷  active bytes per token
```

Everything below is that formula with different numbers in it.

Two quantities matter, and only two. **Active bytes per token** is the size of
the weights actually touched for one token — for a dense model that is the whole
model; for a mixture-of-experts model it is the shared trunk plus whichever few
experts the router picked. **Memory bandwidth** is what the machine can sustain
reading, which on this laptop measures at 19.5 GB/s under llama.cpp and is a
property of the DDR5 bus, not of the software.

## What one machine can do

At 19.5 GB/s measured, with weights at roughly 0.55 bytes per parameter (q4_K
including its scales):

| Model                    | Active params | Bytes/token | Ceiling  | Resident (q4) |
| ------------------------ | ------------- | ----------- | -------- | ------------- |
| Qwen3-30B-A3B            | 3 B           | 1.7 GB      | 11 tok/s | 17 GB         |
| GLM-4.5-Air 106B-A12B    | 12 B          | 6.6 GB      | 3.0 tok/s| 58 GB         |
| Qwen3-235B-A22B          | 22 B          | 12 GB       | 1.6 tok/s| 129 GB        |
| Qwen3-Coder-480B-A35B    | 35 B          | 19 GB       | 1.0 tok/s| 264 GB        |
| DeepSeek-V3 671B-A37B    | 37 B          | 20 GB       | 1.0 tok/s| 369 GB        |
| A dense 70B              | 70 B          | 38 GB       | 0.5 tok/s| 38 GB         |

The "Ceiling" column is the speed a *perfect* implementation reaches on this
laptop. It is not a target to beat; it is a wall to stand next to.

Read the last two columns together, because they are the whole design problem in
one place. The 480B model needs 264 GB resident, which no laptop has — that is
the capacity problem, and it is what the mesh exists to solve. But even with
infinite RAM it would generate one token per second, because 19 GB has to cross
the memory bus for each one. **Those are two different problems and only one of
them is solved by adding machines.**

## What the mesh changes, and what it does not

hocMESH splits a model by layer range: node A holds blocks 0..k, node B holds
k..2k, and the hidden state hops from one to the next. The hidden state is small
— 8192 values in bf16 is 16 KB — so each hop moves 16 KB regardless of how large
the model is. At 1 Gb/s that serialises in 0.13 ms. **Network bandwidth is not a
constraint on this design and never will be.**

What the split does solve is capacity. Fourteen nodes with 20 GB of usable RAM
each hold the 480B model between them, and it runs. Nothing else makes it run.

What the split does **not** solve is single-stream speed, and this is the part
that is easy to get wrong. For one token to be produced, node A must finish
before node B starts, because B's input is A's output. The stages are
sequential, not concurrent. Time per token is therefore the *sum* of each node's
streaming time, which is the same total as one impossibly-large machine would
take:

```
14 nodes × 1.36 GB each ÷ 19.5 GB/s  =  975 ms   — the same 19 GB, just spread out
       + 14 hops × 0.1 ms one-way    =    1.4 ms — LAN, negligible
       + 14 hops × 15 ms one-way     =  210 ms   — open internet, a 21% tax
```

Adding a fifteenth node makes the model fit better. It does not make the stream
faster. A single interactive conversation on a 35B-active model runs at about
one token per second whether it is spread over fourteen machines or would fit on
one.

The consolation is real, though: because per-token compute is already close to a
second, even 30 ms of internet round-trip per hop is a 21% tax rather than a
catastrophe. **The pipeline design survives the open internet.** That is a
genuine and non-obvious property of choosing pipeline parallelism, and it is why
the split is by layer range rather than within layers.

### Why not split inside the layers

Tensor parallelism cuts each weight matrix across machines, so every node
streams 1/N of every layer and they work at the same time. That *does* divide
single-stream time by N — it is what datacenter serving does. The cost is an
all-reduce of the hidden state twice per layer: roughly 124 synchronised round
trips per token on a 62-layer model.

- On a 10 GbE LAN at 0.1 ms: 12 ms of round trips per token. Viable.
- On a 1 GbE LAN at 0.2 ms: 25 ms of round trips plus 32 ms of serialisation.
  Marginal — it caps you near 17 tok/s before any compute.
- On the open internet at 30 ms: 3.7 seconds of round trips per token. Dead.

So tensor parallelism is not universally dead — it is dead for the deployment
hocMESH targets, which is heterogeneous machines on ordinary connections. On a
rack with 10 GbE it would be the right answer. That distinction is worth keeping
straight, because if the product ever grows a LAN-cluster mode, this is the door
it goes through.

### What the mesh is genuinely good at

Throughput, not latency. Fourteen pipelined nodes with fourteen requests in
flight deliver roughly fourteen tokens per second in aggregate while each
individual stream still sees one. For agents, batch evaluation, overnight runs
and anything with many independent requests, that scales close to linearly. For
one person typing at one model, it does not help at all.

## Where the engine actually is today

Measured on this machine against llama.cpp, same file, same prompts,
SmolLM2-135M in f32 (540 MB, so 0.54 GB/token):

| Implementation | tok/s        | Effective bandwidth |
| -------------- | ------------ | ------------------- |
| hocMESH engine | 3.5          | 1.9 GB/s            |
| llama.cpp      | 36.1 ± 5.5   | 19.5 GB/s           |

**We are a factor of ten off the machine's own ceiling on f32, and roughly a
factor of eighty off on quantised weights.** The parity work in 0.5.1 proved the
engine computes the right answer; it also made it impossible to pretend it
computes it at a reasonable speed. Two causes account for essentially all of it,
and neither is subtle.

**Every weight is expanded to f32 in RAM.** `crates/hocmesh-engine/src/weights.rs`
defines `Tensor` as "one tensor, reconstructed to `f32`" with a `values: Vec<f32>`
behind it. A q4 model on disk becomes an f32 model in memory — an eight-fold
blowup in exactly the quantity the formula divides by. A 17 GB model becomes 135
GB resident, which is both the reason quantised models are slow and a large part
of the reason big ones do not fit.

**The inner loop is scalar and single-threaded.** In
`crates/hocmesh-engine/src/stage.rs`:

```rust
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
```

That is one multiply-add per iteration on one core. AVX2 does eight lanes, AVX-512
does sixteen, and there are cores sitting idle. There is no `rayon` anywhere in
the workspace and no SIMD intrinsics in the engine.

Both of these are ordinary engineering with no research risk attached. They are
also worth more than every other item on the list combined, because a 10–80×
multiplier applied first makes every subsequent multiplier meaningful.

## The plan, in order

**0. Fix the engine.** Store weights in their on-disk format and dequantise per
block inside the dot product, so a q4 model occupies q4 bytes and streams q4
bytes. Write SIMD kernels for the quantised dot products. Thread the matrix-vector
product across cores. Expected: 10–80× depending on format, landing the engine
within sight of llama.cpp. No research risk. Do this before anything else.

**1. Execute stages on the GPU.** A single consumer card has 1000 GB/s of memory
bandwidth against the laptop's 19.5 — a fifty-fold jump in the only number that
matters. The constraint is VRAM: 24 GB holds a 30B-A3B model comfortably and
nothing larger. The mesh's layer split is exactly the right shape for spreading
a model across several machines' GPUs, and this is where the split path and the
accelerated path finally meet, which today they do not.

**2. Support mixture-of-experts.** The engine implements dense SwiGLU only —
`ffn_gate_exps` and `ffn_gate_inp` appear nowhere in the codebase, so every model
in the table above is currently unloadable. This is the difference between running
a 30B model at 30B cost and running it at 3B cost. Everything worth targeting is
MoE. This is the largest single change in the list and the one that unlocks the
rest.

**3. Place stages by latency.** Order the pipeline so that hops follow the
cheapest links, keep co-located nodes adjacent, and prefer a longer chain of
nearby machines to a shorter chain of distant ones. Worth ~20% on a WAN mesh,
worth nothing on a LAN. Cheap to implement once measurement exists.

**4. Speculative decoding.** A small draft model proposes k tokens; the large
model verifies all k in a single forward pass. Verification reads the weights
once for k tokens instead of once per token, which attacks the bandwidth bound
directly rather than working around it. At a 65% acceptance rate this is a
2.5–3× multiplier, it composes with everything above, and it is the only item on
this list that improves *single-stream* latency on a distributed model. After
item 0, this is the highest-value work.

**5. Continuous batching.** Keeps every stage busy with multiple requests in
flight. Multiplies aggregate throughput, does nothing for one stream. Right for
the agent and batch workloads, not for the interactive one.

## What this adds up to

Being honest about the destination, because the point of the arithmetic is that
it does not negotiate.

**Reachable, on one good machine.** A 30B-A3B model at q4 needs 1.7 GB per token
and 17 GB resident. After item 0 that is roughly 11 tok/s on CPU; on a 24 GB GPU
(item 1) it is several hundred; with speculation (item 4) faster still. This is
a genuinely capable coding model running locally at a speed nobody would call
slow. **It is also the single most valuable thing on this page, and it needs
items 0, 1, 2 and 4 — none of which require the mesh at all.**

**Reachable, on a mesh.** The 480B and 671B models become *runnable* on fourteen
to nineteen ordinary machines, at roughly 1 tok/s per stream and near-linear
aggregate throughput. That is a real capability that does not otherwise exist at
this price, and it is a good fit for agents, evaluation runs and anything
asynchronous. It is not a good fit for typing at a chat box.

**Not reachable.** A 35B-active model at 50 tok/s for a single interactive stream
on commodity hardware over ordinary internet. That needs 950 GB/s of bandwidth
applied to one sequential chain, and the only ways to get it are tensor
parallelism over a fast local fabric or a card with that much bandwidth on it.
Distribution over the open internet cannot produce it, because pipeline stages
run in sequence and the bytes still have to move. Any plan that claims otherwise
has an error in it.

So the answer to the original question is: **not with a 480B model, and yes with
a 30B-A3B one** — and the work that makes the second true is worth doing first,
because it is the case most people actually have.
