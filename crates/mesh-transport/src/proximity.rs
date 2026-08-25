//! Node-to-node latency probing.
//!
//! A probe is deliberately the smallest useful exchange: a tiny body out, a
//! tiny body back, timed by the caller. What is measured must be the network
//! path, so nothing here reads a file, touches a database, or serialises
//! anything larger than a coordinate.
//!
//! Probing is *outbound*, which is what makes it usable on real contributor
//! machines: a node behind NAT can fit its own coordinate perfectly well by
//! measuring others. Serving probes is the opt-in half, for nodes that happen
//! to be reachable.

use anyhow::{Context, Result, ensure};
use axum::{Json, Router, extract::State, routing::post};
use mesh_core::proximity::Vivaldi;
use mesh_protocol::{NetworkCoordinate, ProbeRequest, ProbeResponse};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// How long a single probe may take before it is written off.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
/// Timed samples taken per peer, on top of one untimed warm-up.
const PROBE_SAMPLES: usize = 3;

/// The position this node serves to probing peers, shared with the loop that
/// fits it.
#[derive(Clone)]
pub struct ProbeState {
    node_id: String,
    tracker: Arc<Mutex<Vivaldi>>,
}

impl ProbeState {
    pub fn new(node_id: impl Into<String>, tracker: Arc<Mutex<Vivaldi>>) -> Self {
        Self {
            node_id: node_id.into(),
            tracker,
        }
    }
}

/// Serve latency probes for other nodes.
pub fn probe_router(state: ProbeState) -> Router {
    Router::new()
        .route("/v1/proximity/probe", post(serve_probe))
        .with_state(state)
}

/// Answer a probe, and fold in the caller's report of an earlier round trip.
///
/// The responder cannot time the round trip itself - it only ever sees one
/// direction - so a peer that has already measured us tells us what it saw.
/// That doubles the observations in the network for no extra probes. A caller
/// that lies about the round trip is bounded by the same step and confidence
/// limits as any other observation, which is why accepting it is safe.
async fn serve_probe(
    State(state): State<ProbeState>,
    Json(req): Json<ProbeRequest>,
) -> Json<ProbeResponse> {
    let mut coordinate = None;
    if let Ok(mut tracker) = state.tracker.lock() {
        if let (Some(remote), Some(rtt)) = (req.coordinate, req.measured_rtt_micros) {
            tracker.observe(&remote, rtt);
        }
        // The provisional position, not the advertised one: a peer needs
        // somewhere to fit against even before we are confident, and the
        // confidence it should place in us rides along in `error_permille`.
        coordinate = Some(tracker.provisional_coordinate());
    }
    Json(ProbeResponse {
        node_id: state.node_id.clone(),
        coordinate,
    })
}

/// What one round of probing a peer produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeOutcome {
    pub node_id: String,
    /// The peer's position as it stands, confident or not. `None` only from a
    /// peer that declined to give one, which cannot be fitted against.
    pub coordinate: Option<NetworkCoordinate>,
    /// The smallest round trip observed, in microseconds.
    pub rtt_micros: u64,
}

/// Measure the round trip to `endpoint`, reporting our own position to it.
///
/// The minimum of several samples is returned rather than the mean: latency
/// noise is one-sided - queueing and scheduling only ever *add* delay - so the
/// minimum is the closest estimate of the path itself. The first exchange is
/// discarded because it pays for connection setup, which is not path latency.
pub async fn probe_peer(
    http: &reqwest::Client,
    endpoint: &str,
    ours: Option<NetworkCoordinate>,
    last_measured_rtt_micros: Option<u64>,
) -> Result<ProbeOutcome> {
    let url = format!("{}/v1/proximity/probe", endpoint.trim_end_matches('/'));
    let request = ProbeRequest {
        coordinate: ours,
        measured_rtt_micros: last_measured_rtt_micros,
    };

    // Warm-up: pays for DNS, TCP and TLS so the timed samples do not.
    let warm = exchange(http, &url, &request).await?;

    let mut best = u64::MAX;
    let mut latest = warm;
    for _ in 0..PROBE_SAMPLES {
        let started = Instant::now();
        latest = exchange(http, &url, &request).await?;
        let elapsed = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        best = best.min(elapsed.max(1));
    }
    ensure!(best < u64::MAX, "probe produced no samples");

    Ok(ProbeOutcome {
        node_id: latest.node_id,
        coordinate: latest.coordinate,
        rtt_micros: best,
    })
}

async fn exchange(
    http: &reqwest::Client,
    url: &str,
    request: &ProbeRequest,
) -> Result<ProbeResponse> {
    let response = http
        .post(url)
        .timeout(PROBE_TIMEOUT)
        .json(request)
        .send()
        .await
        .with_context(|| format!("probing {url}"))?
        .error_for_status()
        .with_context(|| format!("probing {url}"))?;
    response
        .json::<ProbeResponse>()
        .await
        .with_context(|| format!("decoding probe response from {url}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn serve(state: ProbeState) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, probe_router(state)).await.unwrap();
        });
        format!("http://{address}")
    }

    /// A node with no measurements is still a usable probe target.
    ///
    /// It must not be advertised as placed, but it must be measurable, or a
    /// network where nobody has started could never start.
    #[tokio::test]
    async fn an_unfitted_peer_is_measurable_but_says_it_is_unsure() {
        let tracker = Arc::new(Mutex::new(Vivaldi::seeded(b"peer")));
        let endpoint = serve(ProbeState::new("mesh_peer", tracker)).await;
        let outcome = probe_peer(&reqwest::Client::new(), &endpoint, None, None)
            .await
            .unwrap();
        assert_eq!(outcome.node_id, "mesh_peer");
        let coordinate = outcome
            .coordinate
            .expect("a peer must be measurable before it is confident");
        assert_eq!(
            coordinate.error_permille, 1000,
            "an unfitted peer must report no confidence at all"
        );
        assert!(outcome.rtt_micros > 0, "a round trip takes non-zero time");
    }

    /// A caller's report of a round trip it measured moves the responder.
    #[tokio::test]
    async fn a_reported_round_trip_fits_the_responder() {
        let tracker = Arc::new(Mutex::new(Vivaldi::seeded(b"peer")));
        let endpoint = serve(ProbeState::new("mesh_peer", tracker.clone())).await;
        let http = reqwest::Client::new();
        // A caller with no observations has no coordinate to report, so start
        // from one that has already been placed somewhere.
        let caller = NetworkCoordinate {
            vector_micros: [20_000, 0, 0],
            height_micros: 1_000,
            error_permille: 200,
        };

        for _ in 0..5 {
            probe_peer(&http, &endpoint, Some(caller), Some(40_000))
                .await
                .unwrap();
        }
        let fitted = tracker.lock().unwrap().coordinate();
        assert!(
            fitted.is_some(),
            "the responder should have a position after reported round trips"
        );
    }

    /// An unreachable peer is a failed probe, never a coordinate of zero.
    #[tokio::test]
    async fn an_unreachable_peer_is_an_error() {
        let result = probe_peer(&reqwest::Client::new(), "http://127.0.0.1:1", None, None).await;
        assert!(result.is_err());
    }
}
