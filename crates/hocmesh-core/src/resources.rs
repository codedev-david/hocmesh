//! Making an operator's limits mean something.
//!
//! A node advertises how many cores, how much RAM and how much of its GPU it
//! lends. Until now nothing consulted those numbers when work arrived: a
//! machine that offered four of its sixteen cores would run a job across all
//! sixteen and report, accurately and uselessly, that it was lending four. A
//! setting that reads as a control and controls nothing is worse than no
//! setting, because the operator stops watching it.
//!
//! What this is
//! ------------
//!
//! Admission control with counted permits. Work asks for what it is about to
//! use, waits until that much is free, and gives it back when it finishes.
//! Nothing starts that has not been counted, so what is in flight cannot
//! exceed what was lent.
//!
//! What this is not
//! ----------------
//!
//! It is not a sandbox, and that is worth being blunt about rather than
//! leaving for someone to discover.
//!
//! * The worker count bounds *hocMESH's own* concurrent work. It does not stop
//!   the operating system scheduling those workers across every core on the
//!   machine -- and it should not: four workers spread over sixteen cores still
//!   consume four cores' worth of time, which is what was lent. What it cannot
//!   do is constrain anything outside this process.
//! * The byte budgets account for what work *declares* it is about to
//!   allocate, checked before it starts. They are not resident-set limits. A
//!   path that allocates without asking is invisible here, which is why the
//!   asking belongs where the size is already known exactly -- a model file's
//!   tensor directory -- rather than estimated.
//! * Device memory is counted the same way. Time on the GPU is not divided and
//!   cannot be from user space without a scheduler to divide it. A build
//!   claiming otherwise would be claiming a guarantee it has no way to keep.
//!
//! Refusing rather than queueing
//! -----------------------------
//!
//! A claim larger than the whole budget is refused at once instead of waiting.
//! Waiting would be waiting for capacity that cannot arrive however long the
//! queue drains -- a deadlock wearing the clothes of a slow node, and diagnosed
//! as the network being busy. The refusal names both numbers.

use std::{
    fmt,
    sync::{Arc, Condvar, Mutex},
};

use hocmesh_protocol::NodeCapabilities;

/// What an operator has put up for other people's work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lent {
    /// Workers that may run at once. Zero means this node lends nothing.
    pub logical_cpus: usize,
    /// Host memory a workload may occupy.
    pub memory_bytes: u64,
    /// Device memory a workload may occupy, across every lent GPU.
    pub device_memory_bytes: u64,
}

impl Lent {
    /// Read the operator's limits out of what this node advertises.
    ///
    /// `shared_gpu_percent` is applied to the memory the GPUs actually report,
    /// which is the one thing about a GPU that can be divided and held to. It
    /// used to be read as a boolean -- lending 1% and lending 100% did the same
    /// thing -- so a number the operator chose carefully was rounded to "yes".
    ///
    /// A card that never reported its memory contributes nothing rather than a
    /// guess. Unknown VRAM is not a large amount of VRAM.
    #[must_use]
    pub fn from_capabilities(caps: &NodeCapabilities) -> Self {
        // A CPU-only node that lends inference advertises its shared host slice
        // as a device, so that entry's `memory_mb` is main memory wearing a
        // device's clothes. Summing it here would lend the same bytes twice --
        // once as the host budget and again as a device budget that no work can
        // ever draw on, because nothing offloads layers to a CPU.
        let vram: u64 = caps
            .gpus
            .iter()
            .filter(|gpu| gpu.backend != "cpu")
            .filter_map(|gpu| gpu.memory_mb)
            .map(|mb| mb.saturating_mul(1 << 20))
            .sum();
        let share = u128::from(caps.shared_gpu_percent.min(100));
        // Divide last. Dividing first throws away the remainder before the
        // share is applied, so a quarter of a 16 GiB card comes out 21 bytes
        // short -- harmless here, but the same shape of mistake is how a
        // budget quietly stops matching the number the operator set. The wider
        // type makes the multiply exact rather than merely unlikely to
        // overflow.
        let device_memory_bytes = (u128::from(vram) * share / 100) as u64;
        Lent {
            logical_cpus: caps.shared_logical_cpus,
            memory_bytes: caps.shared_memory_bytes,
            device_memory_bytes,
        }
    }

    /// Whether this node lends anything usable at all.
    #[must_use]
    pub fn is_nothing(&self) -> bool {
        self.logical_cpus == 0 || self.memory_bytes == 0
    }
}

/// What one piece of work is about to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Claim {
    pub logical_cpus: usize,
    pub memory_bytes: u64,
    pub device_memory_bytes: u64,
}

impl Claim {
    /// A claim for `logical_cpus` workers and `memory_bytes` of host memory.
    #[must_use]
    pub fn host(logical_cpus: usize, memory_bytes: u64) -> Self {
        Claim {
            logical_cpus,
            memory_bytes,
            device_memory_bytes: 0,
        }
    }

    /// The same, plus device memory.
    #[must_use]
    pub fn with_device(mut self, device_memory_bytes: u64) -> Self {
        self.device_memory_bytes = device_memory_bytes;
        self
    }
}

/// Why a claim can never be granted, however long its caller waits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooLarge {
    pub claim: Claim,
    pub lent: Lent,
}

impl fmt::Display for TooLarge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this work needs {} worker(s), {} MiB of memory and {} MiB on a GPU, and this \
             node lends {} worker(s), {} MiB and {} MiB. Waiting would not help: no amount \
             of other work finishing makes room that was never offered",
            self.claim.logical_cpus,
            self.claim.memory_bytes >> 20,
            self.claim.device_memory_bytes >> 20,
            self.lent.logical_cpus,
            self.lent.memory_bytes >> 20,
            self.lent.device_memory_bytes >> 20,
        )
    }
}

impl std::error::Error for TooLarge {}

/// How much of what was lent is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub logical_cpus: usize,
    pub memory_bytes: u64,
    pub device_memory_bytes: u64,
}

#[derive(Debug, Default)]
struct Held {
    cpus: usize,
    memory: u64,
    device_memory: u64,
}

/// The counted permits behind one node's advertised limits.
///
/// One per process. Cloning shares it -- a single accounting is the whole
/// point, so a second pool would be a second set of limits.
#[derive(Debug, Clone)]
pub struct ResourcePool {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    lent: Lent,
    held: Mutex<Held>,
    freed: Condvar,
}

impl ResourcePool {
    #[must_use]
    pub fn new(lent: Lent) -> Self {
        ResourcePool {
            inner: Arc::new(Inner {
                lent,
                held: Mutex::new(Held::default()),
                freed: Condvar::new(),
            }),
        }
    }

    /// The limits this pool enforces.
    #[must_use]
    pub fn lent(&self) -> Lent {
        self.inner.lent
    }

    /// Take a claim if there is room now.
    ///
    /// `Ok(None)` means it would fit eventually but does not fit yet, so a
    /// caller with something better to do than wait can go and do it.
    pub fn try_claim(&self, claim: Claim) -> Result<Option<Lease>, TooLarge> {
        self.refuse_impossible(claim)?;
        let mut held = self.held();
        if !self.fits(&held, claim) {
            return Ok(None);
        }
        Self::take(&mut held, claim);
        drop(held);
        Ok(Some(self.lease(claim)))
    }

    /// Take a claim, waiting until there is room for it.
    ///
    /// Blocks the calling thread. Refuses at once -- rather than waiting
    /// forever -- for a claim larger than everything lent.
    pub fn claim(&self, claim: Claim) -> Result<Lease, TooLarge> {
        self.refuse_impossible(claim)?;
        let mut held = self.held();
        while !self.fits(&held, claim) {
            held = match self.inner.freed.wait(held) {
                Ok(held) => held,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        Self::take(&mut held, claim);
        drop(held);
        Ok(self.lease(claim))
    }

    /// How much is in use right now.
    #[must_use]
    pub fn usage(&self) -> Usage {
        let held = self.held();
        Usage {
            logical_cpus: held.cpus,
            memory_bytes: held.memory,
            device_memory_bytes: held.device_memory,
        }
    }

    /// How busy this node is, per mille, as the fullest of its budgets.
    ///
    /// The fullest rather than the average, because a node with every worker
    /// busy is full whatever its spare memory says, and sending it more work on
    /// the strength of that memory only lengthens a queue.
    ///
    /// A node that lends nothing reads as completely full, which is what it is:
    /// there is no room on it for anyone else's work. A node that lends no GPU
    /// is not full on that axis -- it simply has no GPU to offer, and reading
    /// that as saturation would make every CPU-only node look overloaded.
    #[must_use]
    pub fn load_permille(&self) -> u16 {
        let usage = self.usage();
        let lent = self.inner.lent;
        let ratio = |used: u64, total: u64| -> u64 {
            if total == 0 {
                return 1000;
            }
            (used.min(total) * 1000) / total
        };
        let cpus = ratio(usage.logical_cpus as u64, lent.logical_cpus as u64);
        let memory = ratio(usage.memory_bytes, lent.memory_bytes);
        let device = if lent.device_memory_bytes == 0 {
            0
        } else {
            ratio(usage.device_memory_bytes, lent.device_memory_bytes)
        };
        cpus.max(memory).max(device).min(1000) as u16
    }

    fn held(&self) -> std::sync::MutexGuard<'_, Held> {
        // A poisoned lock means another thread panicked while holding it. Its
        // lease released on unwind, so the counts are sound; refusing here
        // would turn one panicked job into a node that never works again.
        match self.inner.held.lock() {
            Ok(held) => held,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn refuse_impossible(&self, claim: Claim) -> Result<(), TooLarge> {
        let lent = self.inner.lent;
        let impossible = claim.logical_cpus > lent.logical_cpus
            || claim.memory_bytes > lent.memory_bytes
            || claim.device_memory_bytes > lent.device_memory_bytes;
        if impossible {
            return Err(TooLarge { claim, lent });
        }
        Ok(())
    }

    fn fits(&self, held: &Held, claim: Claim) -> bool {
        let lent = self.inner.lent;
        held.cpus + claim.logical_cpus <= lent.logical_cpus
            && held.memory + claim.memory_bytes <= lent.memory_bytes
            && held.device_memory + claim.device_memory_bytes <= lent.device_memory_bytes
    }

    fn take(held: &mut Held, claim: Claim) {
        held.cpus += claim.logical_cpus;
        held.memory += claim.memory_bytes;
        held.device_memory += claim.device_memory_bytes;
    }

    fn lease(&self, claim: Claim) -> Lease {
        Lease {
            pool: self.clone(),
            claim,
        }
    }

    fn release(&self, claim: Claim) {
        let mut held = self.held();
        held.cpus = held.cpus.saturating_sub(claim.logical_cpus);
        held.memory = held.memory.saturating_sub(claim.memory_bytes);
        held.device_memory = held.device_memory.saturating_sub(claim.device_memory_bytes);
        drop(held);
        self.inner.freed.notify_all();
    }
}

/// Permits held for as long as the work runs.
///
/// Released on drop, including while a panic unwinds. A lease returned by hand
/// would leak whatever the one path that forgot it was holding, and the node
/// would shrink by that much for the rest of its life with nothing to show why.
#[derive(Debug)]
pub struct Lease {
    pool: ResourcePool,
    claim: Claim,
}

impl Lease {
    /// What this lease holds.
    #[must_use]
    pub fn claim(&self) -> Claim {
        self.claim
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.release(self.claim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn pool(cpus: usize, memory: u64, device: u64) -> ResourcePool {
        ResourcePool::new(Lent {
            logical_cpus: cpus,
            memory_bytes: memory,
            device_memory_bytes: device,
        })
    }

    #[test]
    fn work_beyond_what_was_lent_does_not_start() {
        let pool = pool(4, 8 * GIB, 0);
        let _first = pool.claim(Claim::host(3, 6 * GIB)).expect("fits");
        assert!(
            pool.try_claim(Claim::host(2, GIB))
                .expect("not impossible")
                .is_none(),
            "a fifth worker started on a node lending four, so the operator's \
             limit is decoration"
        );
    }

    #[test]
    fn a_claim_nothing_could_ever_satisfy_is_refused_rather_than_queued() {
        let pool = pool(4, 8 * GIB, 0);
        let refusal = pool
            .claim(Claim::host(4, 32 * GIB))
            .expect_err("a claim four times the lent memory was accepted")
            .to_string();
        assert!(
            refusal.contains("Waiting would not help"),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn finishing_work_gives_its_room_back() {
        let pool = pool(4, 8 * GIB, 0);
        {
            let _lease = pool.claim(Claim::host(4, 8 * GIB)).expect("fits");
            assert!(
                pool.try_claim(Claim::host(1, 1))
                    .expect("possible")
                    .is_none()
            );
        }
        pool.claim(Claim::host(4, 8 * GIB))
            .expect("the whole node was free again");
    }

    #[test]
    fn a_panic_does_not_shrink_the_node() {
        let pool = pool(4, 8 * GIB, 0);
        let unwound = std::panic::catch_unwind({
            let pool = pool.clone();
            move || {
                let _lease = pool.claim(Claim::host(4, 8 * GIB)).expect("fits");
                panic!("work failed half way through");
            }
        });
        assert!(unwound.is_err());
        assert_eq!(
            pool.usage(),
            Usage::default(),
            "a job that panicked kept its permits, so every crash costs the \
             node capacity it never gets back"
        );
    }

    #[test]
    fn a_waiting_claim_wakes_when_room_appears() {
        let pool = pool(2, 2 * GIB, 0);
        let held = pool.claim(Claim::host(2, 2 * GIB)).expect("fits");
        let waiter = std::thread::spawn({
            let pool = pool.clone();
            move || pool.claim(Claim::host(2, 2 * GIB)).map(|lease| lease.claim())
        });
        drop(held);
        let taken = waiter.join().expect("thread").expect("claimed");
        assert_eq!(taken.logical_cpus, 2);
    }

    #[test]
    fn lending_a_share_of_a_gpu_lends_that_share_and_not_all_of_it() {
        let mut caps = crate::hardware::detect_capabilities(false);
        caps.shared_logical_cpus = 4;
        caps.shared_memory_bytes = 8 * GIB;
        caps.shared_gpu_percent = 25;
        caps.gpus = vec![hocmesh_protocol::GpuCapability {
            stable_id: "gpu-0".into(),
            vendor: "test".into(),
            name: "test".into(),
            backend: "test".into(),
            memory_mb: Some(16 * 1024),
            driver_version: None,
            compute_version: None,
            supports_fp16: true,
            supports_bf16: false,
            supports_int8: false,
            benchmark_bytes_per_second: None,
            benchmark_p95_micros: None,
        }];
        let lent = Lent::from_capabilities(&caps);
        assert_eq!(
            lent.device_memory_bytes,
            4 * GIB,
            "a quarter of a 16 GiB card was not read as 4 GiB, which is what \
             made --gpu-percent a yes/no switch"
        );

        caps.shared_gpu_percent = 1;
        let sliver = Lent::from_capabilities(&caps);
        assert!(
            sliver.device_memory_bytes < lent.device_memory_bytes,
            "lending 1% and lending 25% came to the same thing"
        );
    }

    #[test]
    fn a_card_that_never_said_how_much_memory_it_has_lends_none() {
        let mut caps = crate::hardware::detect_capabilities(false);
        caps.shared_logical_cpus = 4;
        caps.shared_memory_bytes = 8 * GIB;
        caps.shared_gpu_percent = 100;
        caps.gpus = vec![hocmesh_protocol::GpuCapability {
            stable_id: "gpu-0".into(),
            vendor: "apple".into(),
            name: "unknown vram".into(),
            backend: "metal".into(),
            memory_mb: None,
            driver_version: None,
            compute_version: None,
            supports_fp16: true,
            supports_bf16: false,
            supports_int8: false,
            benchmark_bytes_per_second: None,
            benchmark_p95_micros: None,
        }];
        assert_eq!(
            Lent::from_capabilities(&caps).device_memory_bytes,
            0,
            "a card whose memory was never discovered was treated as though it \
             had some, so work would be admitted onto it and fail there"
        );
    }

    #[test]
    fn load_is_the_fullest_budget_not_the_average() {
        let pool = pool(4, 64 * GIB, 0);
        let _lease = pool.claim(Claim::host(4, GIB)).expect("fits");
        assert_eq!(
            pool.load_permille(),
            1000,
            "every worker was busy and the node still advertised itself as \
             nearly idle because it had memory to spare"
        );
    }

    #[test]
    fn the_shared_cpu_slice_is_not_lent_a_second_time_as_a_card() {
        let mut caps = crate::hardware::detect_capabilities(false);
        caps.shared_logical_cpus = 4;
        caps.shared_memory_bytes = 8 << 30;
        caps.shared_gpu_percent = 100;
        // What a CPU-only node with --ai on advertises: its host slice, in the
        // `gpus` list, because that is how it offers inference without a card.
        caps.gpus = vec![hocmesh_protocol::GpuCapability {
            stable_id: "cpu:0".into(),
            vendor: "cpu".into(),
            name: "a processor".into(),
            backend: "cpu".into(),
            memory_mb: Some(8 << 10),
            driver_version: None,
            compute_version: None,
            supports_fp16: false,
            supports_bf16: false,
            supports_int8: true,
            benchmark_bytes_per_second: None,
            benchmark_p95_micros: None,
        }];
        let lent = Lent::from_capabilities(&caps);
        assert_eq!(lent.memory_bytes, 8 << 30);
        assert_eq!(
            lent.device_memory_bytes, 0,
            "the shared host slice was lent again as device memory, so the node \
             advertised twice the memory it has"
        );
    }

    #[test]
    fn a_node_with_no_gpu_is_not_reported_as_a_full_one() {
        let pool = pool(4, 8 * GIB, 0);
        assert_eq!(pool.load_permille(), 0);
    }

    #[test]
    fn a_node_that_lends_nothing_has_no_room_for_anyone() {
        let pool = pool(0, 0, 0);
        assert_eq!(pool.load_permille(), 1000);
        assert!(pool.claim(Claim::host(1, 1)).is_err());
    }
}
