//! What a machine is allowed to be asked to do.
//!
//! A pipeline is not made of interchangeable parts. The stage holding the
//! first layers reads the whole prompt and pushes a full set of activations
//! downstream before a single token comes back, so the link out of that
//! machine sets the time to first token for everyone waiting on it. The
//! stages after it pass one narrow vector per token and are almost
//! indifferent to bandwidth. Requiring a fast uplink of *every* participant
//! would therefore buy nothing on most of the pipeline while excluding most of
//! the machines that could have run it.
//!
//! So the requirement attaches to the job, not to the door. Anyone may join.
//! Anyone may seed a model, serve a decode stage, or take a batch of prompts,
//! on whatever connection they have. A fast uplink is asked for in one place
//! only: hosting the prefill stage.
//!
//! Roles are *derived* from measured capabilities rather than declared. A node
//! cannot claim to be prefill-class; it can only report its hardware and its
//! measured uplink, which is the surface the rest of the system already has to
//! treat as untrusted. Adding a declared role would have added a second thing
//! to disbelieve without removing the first.
//!
//! The uplink figure must have been *measured* ([`crate::bandwidth`]). Zero
//! means nothing measured it, and an unmeasured link is not a slow one and is
//! certainly not a fast one -- so it does not qualify for prefill, and does not
//! disqualify the node from anything else.

use hocmesh_protocol::NodeCapabilities;
use std::collections::BTreeSet;

/// Uplink a machine must have shown before it may host a prefill stage.
///
/// One gigabit. A prefill stage pushes `prompt_tokens * hidden_size * 4` bytes
/// downstream in one go -- for a long prompt into a large model that is
/// hundreds of megabytes, and it is on the critical path to the first token,
/// with nothing to overlap it against. Below roughly this figure the transfer
/// dominates whatever the machine saved by being fast at arithmetic.
pub const PREFILL_UPLINK_KBPS: u64 = 1_000_000;

/// A kind of work a node can be asked to do.
///
/// Not a hierarchy: [`NodeRole::Seed`] is not a lesser [`NodeRole::Prefill`].
/// A machine on a slow link with a large disk is genuinely the right seed and
/// genuinely the wrong prefill host, and one on a fast link with no accelerator
/// is the reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeRole {
    /// Hold the first layers: read the prompt, push activations downstream.
    Prefill,
    /// Hold a later slice of layers: one narrow vector in, one out, per token.
    Decode,
    /// Serve model chunks to peers that do not have them yet.
    Seed,
    /// Run whole prompts alone, with no stage to talk to.
    Batch,
}

impl NodeRole {
    /// Every role, in a stable order, for iterating.
    pub const ALL: [NodeRole; 4] = [
        NodeRole::Prefill,
        NodeRole::Decode,
        NodeRole::Seed,
        NodeRole::Batch,
    ];

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            NodeRole::Prefill => "prefill",
            NodeRole::Decode => "decode",
            NodeRole::Seed => "seed",
            NodeRole::Batch => "batch",
        }
    }
}

/// The uplink this node has actually shown, or `None` if nothing measured it.
///
/// The wire field is a plain integer with no null, so zero carries "unmeasured"
/// -- and every caller that must tell the two apart should come through here
/// rather than comparing against zero and hoping the next reader remembers why.
#[must_use]
pub fn measured_uplink_kbps(caps: &NodeCapabilities) -> Option<u64> {
    (caps.model_bandwidth_kbps > 0).then_some(caps.model_bandwidth_kbps)
}

/// Whether this node may be asked to do `role`.
#[must_use]
pub fn can_serve(caps: &NodeCapabilities, role: NodeRole) -> bool {
    // Lending nothing is the one answer that rules out everything. It is the
    // operator's own setting, not a judgement about their hardware.
    if caps.shared_logical_cpus == 0 {
        return false;
    }
    match role {
        // Open to anyone who lends a core. Seeding is how a machine on an
        // unknown link earns the measurement that might later qualify it for
        // prefill, so gating it on bandwidth would close the only door to it.
        NodeRole::Seed => true,
        NodeRole::Batch => caps.ai_runtime_ready,
        NodeRole::Decode => caps.ai_runtime_ready,
        NodeRole::Prefill => {
            caps.ai_runtime_ready
                && measured_uplink_kbps(caps).is_some_and(|kbps| kbps >= PREFILL_UPLINK_KBPS)
        }
    }
}

/// Everything this node may be asked to do.
#[must_use]
pub fn roles_for(caps: &NodeCapabilities) -> BTreeSet<NodeRole> {
    NodeRole::ALL
        .into_iter()
        .filter(|&role| can_serve(caps, role))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contributor() -> NodeCapabilities {
        let mut caps = crate::hardware::detect_capabilities(false);
        caps.shared_logical_cpus = 4;
        caps.ai_runtime_ready = true;
        caps
    }

    #[test]
    fn a_slow_link_is_not_a_reason_to_turn_a_machine_away() {
        let mut caps = contributor();
        caps.model_bandwidth_kbps = 2_000; // 2 Mbps, a poor domestic upload.
        let roles = roles_for(&caps);
        assert!(
            !roles.contains(&NodeRole::Prefill),
            "a 2 Mbps uplink was accepted for the one stage whose whole cost is \
             pushing bytes"
        );
        for role in [NodeRole::Decode, NodeRole::Seed, NodeRole::Batch] {
            assert!(
                roles.contains(&role),
                "{} was refused for want of bandwidth it does not need; the gate \
                 is supposed to be on the role, not on joining",
                role.as_str()
            );
        }
    }

    #[test]
    fn an_unmeasured_link_is_not_read_as_a_fast_one() {
        let mut caps = contributor();
        caps.model_bandwidth_kbps = 0;
        assert_eq!(measured_uplink_kbps(&caps), None);
        assert!(!can_serve(&caps, NodeRole::Prefill));
        assert!(
            can_serve(&caps, NodeRole::Seed),
            "a node with no measurement cannot seed, so it can never earn one"
        );
    }

    #[test]
    fn a_measured_gigabit_link_may_host_the_head_of_a_pipeline() {
        let mut caps = contributor();
        caps.model_bandwidth_kbps = PREFILL_UPLINK_KBPS;
        assert_eq!(roles_for(&caps).len(), NodeRole::ALL.len());
    }

    #[test]
    fn a_node_lending_no_cores_is_asked_for_nothing() {
        let mut caps = contributor();
        caps.model_bandwidth_kbps = 10 * PREFILL_UPLINK_KBPS;
        caps.shared_logical_cpus = 0;
        assert!(
            roles_for(&caps).is_empty(),
            "an operator who lends nothing was still handed work because the \
             machine happened to have a fast link"
        );
    }

    #[test]
    fn a_machine_with_no_inference_runtime_can_still_seed() {
        let mut caps = contributor();
        caps.ai_runtime_ready = false;
        caps.model_bandwidth_kbps = 10 * PREFILL_UPLINK_KBPS;
        assert_eq!(roles_for(&caps), BTreeSet::from([NodeRole::Seed]));
    }
}
