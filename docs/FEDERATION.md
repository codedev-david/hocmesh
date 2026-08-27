# Federated coordinators and scheduling

Two problems are solved here, and they are worth keeping apart.

**Federation** decides *which coordinator hands out a given job's shards*. It
exists because one coordinator is a single point of unavailability, and because
a mesh that spans regions wants a control plane near each of them.

**Scheduling** decides *which shard a particular worker is offered next*. It
exists because handing out work in arrival order ignores everything the mesh
already knows about the machine asking for it.

Neither decides who gets paid. Payment is settled by the validator quorum
against evidence the worker signed, under a claim key derived from the job, so
the worst a wrong decision in this file can cost is duplicated or misplaced
effort. That bound is what lets both mechanisms be this simple.

---

## Federation

### The configuration file

`hocmesh-coordinator serve --federation <path>`:

```json
{
  "coordinator_id": "eu-1",
  "region": "eu",
  "advertise": "https://eu-1.example.org",
  "peers": [
    { "coordinator_id": "us-1", "region": "us", "url": "https://us-1.example.org" },
    { "coordinator_id": "ap-1", "region": "ap", "url": "https://ap-1.example.org" }
  ],
  "probe_interval_secs": 10,
  "misses_before_down": 3
}
```

- `coordinator_id` must be **stable across restarts**. Ownership is hashed
  against it, so renaming a coordinator reshuffles every job in the mesh.
- `region` is a label, not a location. It is used to gate clustering and to
  discount a shard's network score across a boundary. It is never turned into a
  latency; inventing one would be inventing data.
- `advertise` is published so an operator can read the topology back out of any
  member. Nothing dials it.
- `peers` lists the other coordinators. Each must appear exactly once, and none
  may reuse this coordinator's id.
- `probe_interval_secs` and `misses_before_down` set how quickly a silent peer
  is treated as gone: `interval x misses` is the worst-case detection time.

Without `--federation` a coordinator runs unfederated and owns every job it can
see, which is the previous behaviour exactly.

### Who owns a job

```
owner(job) = argmax over live coordinators of  hash(coordinator_id \0 job_id)
```

Rendezvous hashing, with ties broken on the coordinator id. Two properties
matter:

1. **No agreement is required.** Every coordinator computes the same answer
   from the same inputs, so there is no election, no lease, no lock, and no
   handover protocol.
2. **Removing one coordinator moves only its share.** Jobs owned by the
   survivors keep their owner, so a failure does not reshuffle the mesh.

The live set always contains this coordinator. A coordinator that cannot reach
*anyone* therefore keeps serving its own share rather than stopping.

### Peers start down

A newly started coordinator marks every peer down until a probe succeeds. This
is deliberate and it is the opposite of the convenient default: if peers
started up, a coordinator that came back after a network outage would spend the
first probe interval refusing to hand out jobs whose owners are, as far as
anyone can tell, not there. Refusing work you could do is worse than briefly
doing work someone else might also be doing -- and the latter is bounded by the
claim key, which is the whole reason this asymmetry is safe.

### What happens when a peer dies

1. `misses_before_down` consecutive probe failures move the peer out of the
   live set.
2. Ownership of that peer's jobs moves to whichever surviving coordinator now
   wins the hash. Nothing is copied and nothing is announced.
3. Leases the departed coordinator had handed out are **shortened to a 60
   second grace window**, not cancelled. A worker that is still running the
   shard can report in and be paid; only after the grace window is the shard
   offered to someone else. Cancelling immediately would throw away good work
   whenever the failure was in the coordinator-to-coordinator link rather than
   in the worker.

If the coordinator's *database* is lost as well, that is a different problem
with a different tool: `hocmesh-coordinator rebuild` replays certified entries
into an empty database. See `docs/CRASH_RECOVERY.md`.

### Shared job state

Federated coordinators serve **one job store**. The ownership rule is what
stops two of them handing the same shard to two different workers.

State that is not in the store comes back from the chain. `sync_from_ledger`
resumes replay from a persisted watermark recording both the sequence number
reached and the validator set in force at that point, so entries are always
verified against the set that actually signed them rather than the set sitting
today. A coordinator that has been down for a while catches up from where it
stopped instead of replaying from genesis.

### Reading it back

- `GET /v1/federation/status` — this coordinator's id, region, live set, and
  per-peer health including consecutive misses and the last error.
- `GET /v1/federation/jobs/{job_id}` — who owns that job right now, whether
  that is this coordinator, and where the owner answers.

Both are unauthenticated reads of scheduling state. Neither exposes CU.

---

## Scheduling

`poll_work` no longer takes the first pending row. It scores a bounded
oldest-first window of shards this coordinator owns and offers the best one.
The window is bounded so polling stays cheap under a long backlog, and ordered
by age so the bound can never hide the shards closest to starving.

### Hard gates, applied before any scoring

A shard is not scored at all if the worker shares no CPU, has less shareable
memory than the shard's working set, or could not finish the shard inside a
lease. That last one is arithmetic rather than a guess: the node's own
benchmark is candidates per second at a fixed problem size, and the workspace
prices work in units of `REFERENCE_OPS_PER_MCU`, so the two convert exactly.
Hardware is a filter here, not a preference.

### The four axes

| Axis | Weight | What it measures |
|---|---|---|
| Hardware | 0.25 | Safety margin between the shard's cost and what this machine does in a lease |
| Network | 0.20 | Measured round trip amortised against the size of the shard |
| Reliability | 0.25 | Audit standing, and how much an unproven node is exposed to at once |
| Locality | 0.30 | Shards of this job already done here, adjacency to them, and model manifests already cached |

The axes interact with shard *size* on purpose, which is what makes them
discriminate between candidates rather than just between workers. The
interaction is a productive tension: the network axis pulls toward larger
shards, because a fixed round trip is cheaper per unit of work when the unit is
bigger, while the reliability axis pulls unproven nodes toward smaller ones,
because the mesh should not stake a large shard on a machine that has not
earned it yet.

Absent measurements score mid-axis, never well. An unplaced network coordinate
is not near, an unbenchmarked machine is not fast, and two machines that have
measured nothing about each other are at an explicit placeholder distance, not
at zero.

### Starvation

A shard that has waited past the starvation window gets a bonus added
*outside* the weighted mean, and the bonus exceeds the maximum possible fit.
So a starved shard outranks every fresh candidate by construction rather than
by tuning -- it is a guarantee, not a preference. The constant is checked at
compile time.

### Reading it back

`GET /v1/topology` describes the machines the scheduler is choosing from:
who is online, what they can do, whether they have placed themselves, and their
standing. `?cluster=N` additionally asks for the tightest N machines that pass
a gate, for work that has to land on several machines at once;
`&region=`, `&min_memory_bytes=` and `&min_standing=` constrain it. An
unsatisfiable request says so instead of returning a short set.

Per-decision detail is emitted at `debug` level with every axis, so a placement
that looks wrong can be explained rather than guessed at.
