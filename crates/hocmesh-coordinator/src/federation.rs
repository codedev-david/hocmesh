//! Regional coordinators, and what happens when one of them dies.
//!
//! A federated deployment runs several coordinators over one validator set.
//! The thing that makes that cheap is that they do not have to agree about
//! anything that matters: the ledger already holds every job, every shard and
//! every settlement, and `rebuild` already replays it. Two coordinators
//! reading the same chain hold the same job state without exchanging a byte,
//! so there is no shared database to keep consistent and no consensus to run
//! between them.
//!
//! What is left is an efficiency question -- *who hands out which shard* --
//! and it is answered without agreement, by rendezvous hashing. Each
//! coordinator ranks itself against the live set for a given job id, and the
//! highest rank owns it. Every coordinator computes the same answer from the
//! same inputs, so ownership needs no election, no lock and no lease.
//!
//! Failure is the same mechanism seen from the other side. A coordinator that
//! stops answering probes leaves the live set; ownership of its jobs
//! re-derives onto the survivors on the next poll, with no state transfer,
//! because there was never any state to transfer. That is the whole of
//! failover: it is automatic because ownership is a pure function of who is
//! alive.
//!
//! Getting it wrong is survivable by construction. If two coordinators both
//! believe they own a job -- during a partition, say -- they both hand out its
//! shards, and the worst case is that a shard is computed twice: assignment
//! ids are derived, so the second reward carries the same claim key and the
//! ledger refuses it. Duplicated effort, never duplicated CU.

use anyhow::{Context, Result, bail};
use hocmesh_protocol::hash_bytes;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;

fn default_probe_interval() -> u64 {
    10
}

fn default_misses() -> u32 {
    3
}

/// One coordinator in the federation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerConfig {
    pub coordinator_id: String,
    #[serde(default)]
    pub region: String,
    pub url: String,
}

/// The file an operator hands to `--federation`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// This coordinator's name in the federation. Stable across restarts:
    /// ownership is hashed against it, so changing it reshuffles every job.
    pub coordinator_id: String,
    #[serde(default)]
    pub region: String,
    /// Where this coordinator answers, for peers to probe. Advisory: nothing
    /// here dials it, it is published so an operator can read the topology
    /// back out of any member.
    #[serde(default)]
    pub advertise: String,
    #[serde(default)]
    pub peers: Vec<PeerConfig>,
    #[serde(default = "default_probe_interval")]
    pub probe_interval_secs: u64,
    /// Consecutive failed probes before a peer is treated as gone.
    #[serde(default = "default_misses")]
    pub misses_before_down: u32,
}

/// What is currently believed about one peer.
#[derive(Debug, Clone, Serialize)]
pub struct PeerHealth {
    pub coordinator_id: String,
    pub region: String,
    pub url: String,
    pub up: bool,
    pub consecutive_misses: u32,
    pub last_ok_unix: Option<i64>,
    pub last_error: Option<String>,
}

/// A peer crossing the line between alive and gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    CameUp,
    WentDown,
}

/// The federation as this coordinator sees it, for `/v1/federation/status`.
#[derive(Debug, Clone, Serialize)]
pub struct FederationStatus {
    pub coordinator_id: String,
    pub region: String,
    pub advertise: String,
    pub probe_interval_secs: u64,
    pub misses_before_down: u32,
    /// Coordinators believed alive, including this one. Ownership is derived
    /// from exactly this list.
    pub live: Vec<String>,
    pub peers: Vec<PeerHealth>,
}

struct Inner {
    me: PeerConfig,
    advertise: String,
    probe_interval_secs: u64,
    misses_before_down: u32,
    health: RwLock<BTreeMap<String, PeerHealth>>,
}

/// A handle on the federation, cheap to clone into request handlers.
#[derive(Clone)]
pub struct Federation {
    inner: Arc<Inner>,
}

impl Federation {
    pub fn new(config: FederationConfig) -> Result<Self> {
        if config.coordinator_id.trim().is_empty() {
            bail!("federation config needs a coordinator_id");
        }
        if config.misses_before_down == 0 {
            bail!("misses_before_down must be at least 1, or a single dropped probe evicts a peer");
        }
        if config.probe_interval_secs == 0 {
            bail!("probe_interval_secs must be at least 1");
        }
        let mut seen = BTreeSet::new();
        seen.insert(config.coordinator_id.clone());
        let mut health = BTreeMap::new();
        for peer in &config.peers {
            if peer.coordinator_id.trim().is_empty() {
                bail!("every federation peer needs a coordinator_id");
            }
            if !seen.insert(peer.coordinator_id.clone()) {
                bail!(
                    "coordinator id {} appears twice in the federation; ownership would be ambiguous",
                    peer.coordinator_id
                );
            }
            if peer.url.trim().is_empty() {
                bail!("federation peer {} needs a url", peer.coordinator_id);
            }
            health.insert(
                peer.coordinator_id.clone(),
                PeerHealth {
                    coordinator_id: peer.coordinator_id.clone(),
                    region: peer.region.clone(),
                    url: peer.url.trim_end_matches('/').to_string(),
                    // Peers start gone, not present. Until one answers, this
                    // coordinator owns everything -- which duplicates work at
                    // worst, where assuming a dead peer is alive strands its
                    // jobs until the misses run out.
                    up: false,
                    consecutive_misses: config.misses_before_down,
                    last_ok_unix: None,
                    last_error: None,
                },
            );
        }
        Ok(Self {
            inner: Arc::new(Inner {
                me: PeerConfig {
                    coordinator_id: config.coordinator_id,
                    region: config.region,
                    url: config.advertise.trim_end_matches('/').to_string(),
                },
                advertise: config.advertise,
                probe_interval_secs: config.probe_interval_secs,
                misses_before_down: config.misses_before_down,
                health: RwLock::new(health),
            }),
        })
    }

    pub fn load(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading federation config {path}"))?;
        let config: FederationConfig = serde_json::from_str(&raw)
            .with_context(|| format!("parsing federation config {path}"))?;
        Self::new(config)
    }

    pub fn coordinator_id(&self) -> &str {
        &self.inner.me.coordinator_id
    }

    /// `None` when the operator named no region, so nothing downstream scores
    /// an unnamed region as a match.
    pub fn region(&self) -> Option<&str> {
        Some(self.inner.me.region.as_str()).filter(|r| !r.is_empty())
    }

    pub fn probe_interval(&self) -> Duration {
        Duration::from_secs(self.inner.probe_interval_secs)
    }

    pub fn peers(&self) -> Vec<PeerHealth> {
        self.inner
            .health
            .read()
            .expect("federation health lock")
            .values()
            .cloned()
            .collect()
    }

    /// Coordinators believed alive, this one always included.
    ///
    /// A coordinator never evicts itself: it is the one member whose liveness
    /// it can observe directly, and a federation where every member had
    /// declared itself gone would hand out nothing at all.
    pub fn live(&self) -> Vec<String> {
        let mut live = vec![self.inner.me.coordinator_id.clone()];
        live.extend(
            self.inner
                .health
                .read()
                .expect("federation health lock")
                .values()
                .filter(|p| p.up)
                .map(|p| p.coordinator_id.clone()),
        );
        live.sort();
        live
    }

    /// Which coordinator should be handing out this job's shards.
    pub fn owner_of(&self, job_id: &str) -> String {
        owner(job_id, &self.live()).unwrap_or_else(|| self.inner.me.coordinator_id.clone())
    }

    /// Whether this coordinator should be handing out this job's shards.
    pub fn owns(&self, job_id: &str) -> bool {
        self.owner_of(job_id) == self.inner.me.coordinator_id
    }

    pub fn status(&self) -> FederationStatus {
        FederationStatus {
            coordinator_id: self.inner.me.coordinator_id.clone(),
            region: self.inner.me.region.clone(),
            advertise: self.inner.advertise.clone(),
            probe_interval_secs: self.inner.probe_interval_secs,
            misses_before_down: self.inner.misses_before_down,
            live: self.live(),
            peers: self.peers(),
        }
    }

    /// Fold one probe result into a peer's health.
    ///
    /// Pure apart from the lock, so the state machine that decides when a peer
    /// is gone can be tested without a network. Returns the transition, if
    /// this probe was the one that crossed the line.
    pub fn record_probe(
        &self,
        coordinator_id: &str,
        outcome: Result<(), String>,
        now: i64,
    ) -> Option<Transition> {
        let mut health = self.inner.health.write().expect("federation health lock");
        let peer = health.get_mut(coordinator_id)?;
        let was_up = peer.up;
        match outcome {
            Ok(()) => {
                peer.consecutive_misses = 0;
                peer.last_ok_unix = Some(now);
                peer.last_error = None;
                // One good answer is enough to come back. Recovery is cheap
                // and wrong in only one direction: a peer wrongly believed
                // alive costs a delay, a peer wrongly believed dead costs
                // duplicated work on every job it owns.
                peer.up = true;
            }
            Err(e) => {
                peer.consecutive_misses = peer.consecutive_misses.saturating_add(1);
                peer.last_error = Some(e);
                peer.up = peer.consecutive_misses < self.inner.misses_before_down;
            }
        }
        match (was_up, peer.up) {
            (false, true) => Some(Transition::CameUp),
            (true, false) => Some(Transition::WentDown),
            _ => None,
        }
    }

    /// Probe every peer once and record what came back.
    pub async fn probe_all(&self, http: &reqwest::Client, now: i64) -> Vec<(String, Transition)> {
        let targets: Vec<(String, String)> = self
            .peers()
            .into_iter()
            .map(|p| (p.coordinator_id, p.url))
            .collect();
        let mut transitions = Vec::new();
        for (coordinator_id, url) in targets {
            let outcome = match http.get(format!("{url}/health")).send().await {
                Ok(r) if r.status().is_success() => Ok(()),
                Ok(r) => Err(format!("health returned {}", r.status())),
                Err(e) => Err(e.to_string()),
            };
            if let Some(t) = self.record_probe(&coordinator_id, outcome, now) {
                transitions.push((coordinator_id, t));
            }
        }
        transitions
    }

    /// The url a peer answers on, for an operator-facing report.
    pub fn peer_url(&self, coordinator_id: &str) -> Option<String> {
        self.inner
            .health
            .read()
            .expect("federation health lock")
            .get(coordinator_id)
            .map(|p| p.url.clone())
    }
}

/// This coordinator's rendezvous weight for a job.
///
/// Highest-random-weight hashing, so ownership is a pure function of the job
/// id and the live set. Its property is the one that matters when a
/// coordinator dies: only the jobs owned by the departed member move, and they
/// spread across the survivors rather than all landing on one. Nothing else
/// is reshuffled, so a failure does not restripe the whole deployment.
pub fn weight(coordinator_id: &str, job_id: &str) -> u64 {
    // A separator, so ("ab", "c") and ("a", "bc") are different jobs.
    let digest = hash_bytes(format!("{coordinator_id}\u{0}{job_id}").as_bytes());
    u64::from_str_radix(&digest[..16], 16).unwrap_or(0)
}

/// The highest-weighted member of `live` for this job.
pub fn owner(job_id: &str, live: &[String]) -> Option<String> {
    live.iter()
        .max_by(|a, b| {
            weight(a, job_id)
                .cmp(&weight(b, job_id))
                // A hash collision must not make ownership depend on the order
                // the live set happened to be built in.
                .then_with(|| a.as_str().cmp(b.as_str()))
        })
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(id: &str, peers: &[(&str, &str)]) -> FederationConfig {
        FederationConfig {
            coordinator_id: id.into(),
            region: "eu".into(),
            advertise: format!("http://{id}:8080"),
            peers: peers
                .iter()
                .map(|(pid, region)| PeerConfig {
                    coordinator_id: (*pid).into(),
                    region: (*region).into(),
                    url: format!("http://{pid}:8080/"),
                })
                .collect(),
            probe_interval_secs: 10,
            misses_before_down: 3,
        }
    }

    fn federation(id: &str, peers: &[(&str, &str)]) -> Federation {
        Federation::new(config(id, peers)).expect("valid config")
    }

    #[test]
    fn a_duplicate_coordinator_id_is_refused_rather_than_left_ambiguous() {
        let mut c = config("a", &[("b", "eu")]);
        c.peers.push(PeerConfig {
            coordinator_id: "a".into(),
            region: "eu".into(),
            url: "http://elsewhere:8080".into(),
        });
        assert!(Federation::new(c).is_err());
        let mut twice = config("a", &[("b", "eu"), ("b", "us")]);
        twice.probe_interval_secs = 10;
        assert!(Federation::new(twice).is_err());
    }

    #[test]
    fn a_config_that_would_evict_on_one_dropped_probe_is_refused() {
        let mut c = config("a", &[("b", "eu")]);
        c.misses_before_down = 0;
        assert!(Federation::new(c).is_err());
    }

    #[test]
    fn peers_start_gone_so_a_lone_coordinator_still_hands_out_work() {
        let f = federation("a", &[("b", "eu"), ("c", "eu")]);
        assert_eq!(f.live(), vec!["a".to_string()]);
        for job in ["job-1", "job-2", "job-3", "job-4", "job-5"] {
            assert!(
                f.owns(job),
                "{job} should be owned while the peers are silent"
            );
        }
    }

    #[test]
    fn one_answer_brings_a_peer_back_but_it_takes_the_full_count_to_lose_it() {
        let f = federation("a", &[("b", "eu")]);
        assert_eq!(f.record_probe("b", Ok(()), 100), Some(Transition::CameUp));
        assert_eq!(f.live(), vec!["a".to_string(), "b".to_string()]);
        // Two misses out of three: still alive, and no transition reported.
        assert_eq!(f.record_probe("b", Err("timeout".into()), 110), None);
        assert_eq!(f.record_probe("b", Err("timeout".into()), 120), None);
        assert_eq!(f.live(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            f.record_probe("b", Err("timeout".into()), 130),
            Some(Transition::WentDown)
        );
        assert_eq!(f.live(), vec!["a".to_string()]);
        // And a single success is enough to be back.
        assert_eq!(f.record_probe("b", Ok(()), 140), Some(Transition::CameUp));
        let peer = &f.peers()[0];
        assert_eq!(peer.last_ok_unix, Some(140));
        assert!(peer.last_error.is_none());
        assert_eq!(peer.consecutive_misses, 0);
    }

    #[test]
    fn a_recovered_peer_stops_reporting_a_transition_on_every_further_success() {
        let f = federation("a", &[("b", "eu")]);
        assert_eq!(f.record_probe("b", Ok(()), 100), Some(Transition::CameUp));
        assert_eq!(f.record_probe("b", Ok(()), 110), None);
        assert_eq!(f.record_probe("b", Ok(()), 120), None);
    }

    #[test]
    fn a_probe_for_a_coordinator_nobody_configured_is_ignored() {
        let f = federation("a", &[("b", "eu")]);
        assert_eq!(f.record_probe("ghost", Ok(()), 100), None);
        assert_eq!(f.live(), vec!["a".to_string()]);
    }

    #[test]
    fn every_coordinator_computes_the_same_owner_from_the_same_live_set() {
        let live: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        for job in ["job-1", "job-2", "job-3", "job-4", "job-5", "job-6"] {
            let from_a = owner(job, &live).unwrap();
            let mut shuffled = live.clone();
            shuffled.reverse();
            assert_eq!(from_a, owner(job, &shuffled).unwrap());
        }
    }

    #[test]
    fn ownership_is_shared_out_rather_than_pooling_on_one_member() {
        let live: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for i in 0..600 {
            *counts
                .entry(owner(&format!("job-{i}"), &live).unwrap())
                .or_default() += 1;
        }
        assert_eq!(counts.len(), 3, "every coordinator should own something");
        for (id, n) in &counts {
            assert!(
                (100..300).contains(n),
                "{id} owns {n} of 600, which is not a share"
            );
        }
    }

    #[test]
    fn losing_a_coordinator_moves_only_the_jobs_it_owned() {
        let before: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let after: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let mut moved = 0;
        let mut moved_from_c = 0;
        for i in 0..600 {
            let job = format!("job-{i}");
            let was = owner(&job, &before).unwrap();
            let now = owner(&job, &after).unwrap();
            if was != now {
                moved += 1;
                assert_eq!(
                    was, "c",
                    "{job} moved off a coordinator that is still alive"
                );
                moved_from_c += 1;
            }
        }
        assert!(moved > 0, "c owned nothing, so the test proved nothing");
        assert_eq!(moved, moved_from_c);
    }

    #[test]
    fn a_dead_peers_jobs_are_adopted_without_anyone_being_told_where_to_put_them() {
        // Two coordinators, each holding the same view. This is the whole of
        // failover: b stops answering, and a's answer to "do I own this job"
        // changes for exactly b's jobs, with no message between them.
        let a = federation("a", &[("b", "eu")]);
        a.record_probe("b", Ok(()), 100);
        let jobs: Vec<String> = (0..200).map(|i| format!("job-{i}")).collect();
        let bs: Vec<&String> = jobs.iter().filter(|j| !a.owns(j)).collect();
        assert!(
            !bs.is_empty(),
            "b owned nothing, so the test proved nothing"
        );
        let mine_before: Vec<&String> = jobs.iter().filter(|j| a.owns(j)).collect();

        for t in 0..3 {
            a.record_probe("b", Err("connection refused".into()), 110 + t);
        }
        assert_eq!(a.live(), vec!["a".to_string()]);
        for job in &jobs {
            assert!(a.owns(job), "{job} was not adopted after b went silent");
        }
        // And when b comes back, a gives back exactly what it borrowed.
        a.record_probe("b", Ok(()), 200);
        let mine_after: Vec<&String> = jobs.iter().filter(|j| a.owns(j)).collect();
        assert_eq!(mine_before, mine_after);
    }

    #[test]
    fn an_unnamed_region_is_no_region_rather_than_the_empty_one() {
        let mut c = config("a", &[]);
        c.region = String::new();
        let f = Federation::new(c).unwrap();
        assert_eq!(f.region(), None);
        assert_eq!(federation("a", &[]).region(), Some("eu"));
    }

    #[test]
    fn a_peer_url_keeps_its_shape_whatever_the_operator_typed() {
        let f = federation("a", &[("b", "eu")]);
        assert_eq!(f.peer_url("b").as_deref(), Some("http://b:8080"));
        assert_eq!(f.peer_url("nobody"), None);
    }

    #[test]
    fn a_separator_keeps_two_different_pairs_from_hashing_alike() {
        assert_ne!(weight("ab", "c"), weight("a", "bc"));
    }

    #[test]
    fn status_reports_the_live_set_ownership_is_actually_derived_from() {
        let f = federation("a", &[("b", "eu"), ("c", "us")]);
        f.record_probe("c", Ok(()), 100);
        let s = f.status();
        assert_eq!(s.coordinator_id, "a");
        assert_eq!(s.live, vec!["a".to_string(), "c".to_string()]);
        assert_eq!(s.peers.len(), 2);
        let owners: BTreeSet<String> = (0..200).map(|i| f.owner_of(&format!("job-{i}"))).collect();
        assert!(
            owners.iter().all(|o| s.live.contains(o)),
            "ownership escaped the live set: {owners:?}"
        );
    }
}
