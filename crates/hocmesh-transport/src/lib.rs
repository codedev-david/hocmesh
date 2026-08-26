mod proximity;
pub use proximity::{ProbeOutcome, ProbeState, probe_peer, probe_router};

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path as AxumPath, State},
    http::{HeaderValue, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
    routing::get,
};
use hocmesh_model::{ChunkRef, ChunkStore, ModelManifest, sha256};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorFrame {
    pub job_id: String,
    pub stream_id: String,
    pub sequence: u64,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub payload_sha256: String,
    pub payload: Vec<u8>,
}

pub struct TensorAssembler {
    next_sequence: u64,
    max_reorder_window: u64,
    pending: BTreeMap<u64, TensorFrame>,
}

impl TensorAssembler {
    pub fn new(max_reorder_window: u64) -> Result<Self> {
        ensure!(max_reorder_window > 0, "reorder window must be positive");
        Ok(Self {
            next_sequence: 0,
            max_reorder_window,
            pending: BTreeMap::new(),
        })
    }

    pub fn accept(&mut self, frame: TensorFrame) -> Result<Vec<TensorFrame>> {
        frame.validate()?;
        ensure!(
            frame.sequence >= self.next_sequence,
            "replayed tensor frame"
        );
        ensure!(
            frame.sequence < self.next_sequence.saturating_add(self.max_reorder_window),
            "tensor frame exceeds reorder window"
        );
        ensure!(
            !self.pending.contains_key(&frame.sequence),
            "duplicate tensor frame"
        );
        self.pending.insert(frame.sequence, frame);
        let mut ready = Vec::new();
        while let Some(frame) = self.pending.remove(&self.next_sequence) {
            ready.push(frame);
            self.next_sequence += 1;
        }
        Ok(ready)
    }
}

#[derive(Clone, Default)]
pub struct TensorInbox {
    assemblers: Arc<tokio::sync::Mutex<BTreeMap<(String, String), TensorAssembler>>>,
    delivered: Arc<tokio::sync::Mutex<Vec<TensorFrame>>>,
}

impl TensorInbox {
    pub async fn take_delivered(&self) -> Vec<TensorFrame> {
        std::mem::take(&mut *self.delivered.lock().await)
    }
}

pub fn tensor_router(inbox: TensorInbox) -> Router {
    Router::new()
        .route("/v1/tensors", axum::routing::post(receive_tensor))
        .with_state(inbox)
}

async fn receive_tensor(
    State(inbox): State<TensorInbox>,
    Json(frame): Json<TensorFrame>,
) -> Result<StatusCode, StatusCode> {
    frame.validate().map_err(|_| StatusCode::BAD_REQUEST)?;
    let key = (frame.job_id.clone(), frame.stream_id.clone());
    let ready = {
        let mut assemblers = inbox.assemblers.lock().await;
        let assembler = assemblers
            .entry(key)
            .or_insert_with(|| TensorAssembler::new(1024).unwrap());
        assembler.accept(frame).map_err(|_| StatusCode::CONFLICT)?
    };
    inbox.delivered.lock().await.extend(ready);
    Ok(StatusCode::ACCEPTED)
}

impl TensorFrame {
    pub fn new(
        job_id: impl Into<String>,
        stream_id: impl Into<String>,
        sequence: u64,
        shape: Vec<u64>,
        dtype: impl Into<String>,
        payload: Vec<u8>,
    ) -> Result<Self> {
        ensure!(!payload.is_empty(), "tensor payload is empty");
        let frame = Self {
            job_id: job_id.into(),
            stream_id: stream_id.into(),
            sequence,
            shape,
            dtype: dtype.into(),
            payload_sha256: sha256(&payload),
            payload,
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.job_id.is_empty() && !self.stream_id.is_empty(),
            "tensor route is empty"
        );
        ensure!(
            !self.shape.is_empty() && self.shape.iter().all(|n| *n > 0),
            "invalid tensor shape"
        );
        ensure!(!self.dtype.is_empty(), "tensor dtype is empty");
        ensure!(
            sha256(&self.payload) == self.payload_sha256,
            "tensor checksum mismatch"
        );
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(serde_json::to_vec(self)?)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let frame: Self = serde_json::from_slice(bytes)?;
        frame.validate()?;
        Ok(frame)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerInventory {
    pub peer: String,
    pub chunks: BTreeSet<String>,
    pub latency_ms: u64,
    pub failures: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SeedAssignment {
    pub chunk: ChunkRef,
    pub peer: String,
}

/// Assigns missing chunks rarest-first and distributes requests across healthy
/// peers. This prevents a fast peer from becoming the only model seed.
pub fn plan_seeding(
    manifest: &ModelManifest,
    local: &BTreeSet<String>,
    peers: &[PeerInventory],
) -> Result<Vec<SeedAssignment>> {
    manifest.validate()?;
    let mut assignments = Vec::new();
    let mut peer_load: BTreeMap<&str, usize> = BTreeMap::new();
    let mut missing: Vec<_> = manifest
        .chunks
        .iter()
        .filter(|chunk| !local.contains(&chunk.sha256))
        .collect();
    missing.sort_by_key(|chunk| {
        peers
            .iter()
            .filter(|peer| peer.chunks.contains(&chunk.sha256))
            .count()
    });
    for chunk in missing {
        let peer = peers
            .iter()
            .filter(|peer| peer.chunks.contains(&chunk.sha256))
            .min_by_key(|peer| {
                (
                    peer.failures,
                    *peer_load.get(peer.peer.as_str()).unwrap_or(&0),
                    peer.latency_ms,
                )
            })
            .with_context(|| format!("no peer has chunk {}", chunk.sha256))?;
        *peer_load.entry(&peer.peer).or_default() += 1;
        assignments.push(SeedAssignment {
            chunk: chunk.clone(),
            peer: peer.peer.clone(),
        });
    }
    Ok(assignments)
}

#[async_trait]
pub trait PeerSource: Send + Sync {
    async fn manifest(&self, model_id: &str, revision: &str) -> Result<ModelManifest>;
    async fn chunk(&self, digest: &str) -> Result<Vec<u8>>;
}

#[derive(Clone)]
pub struct HttpPeerSource {
    base_url: String,
    client: reqwest::Client,
}

impl HttpPeerSource {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        ensure!(
            base_url.starts_with("http://") || base_url.starts_with("https://"),
            "peer URL must be HTTP(S)"
        );
        Ok(Self {
            base_url,
            client: reqwest::Client::new(),
        })
    }
}

#[async_trait]
impl PeerSource for HttpPeerSource {
    async fn manifest(&self, model_id: &str, revision: &str) -> Result<ModelManifest> {
        let response = self
            .client
            .get(format!(
                "{}/v1/models/{model_id}/{revision}/manifest",
                self.base_url
            ))
            .send()
            .await?
            .error_for_status()?;
        let manifest: ModelManifest = response.json().await?;
        manifest.validate()?;
        Ok(manifest)
    }

    async fn chunk(&self, digest: &str) -> Result<Vec<u8>> {
        ensure!(
            digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid chunk digest"
        );
        let bytes = self
            .client
            .get(format!("{}/v1/chunks/{digest}", self.base_url))
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?
            .to_vec();
        ensure!(sha256(&bytes) == digest, "peer returned corrupt chunk");
        Ok(bytes)
    }
}

pub async fn seed_from_peer(
    source: &dyn PeerSource,
    store: &ChunkStore,
    model_id: &str,
    revision: &str,
) -> Result<ModelManifest> {
    let manifest = source.manifest(model_id, revision).await?;
    for chunk in &manifest.chunks {
        if !store.contains(chunk) {
            let bytes = source.chunk(&chunk.sha256).await?;
            ensure!(
                bytes.len() as u64 == chunk.size_bytes,
                "peer chunk size mismatch"
            );
            ensure!(
                store.put(&bytes)? == chunk.sha256,
                "stored chunk digest mismatch"
            );
        }
    }
    Ok(manifest)
}

#[derive(Clone)]
pub struct SeedServerState {
    store: Arc<ChunkStore>,
    manifests: Arc<BTreeMap<(String, String), ModelManifest>>,
}

impl SeedServerState {
    pub fn new(
        store: Arc<ChunkStore>,
        manifests: impl IntoIterator<Item = ModelManifest>,
    ) -> Result<Self> {
        let mut map = BTreeMap::new();
        for manifest in manifests {
            manifest.validate()?;
            map.insert(
                (manifest.model_id.clone(), manifest.revision.clone()),
                manifest,
            );
        }
        Ok(Self {
            store,
            manifests: Arc::new(map),
        })
    }
}

pub fn seed_router(state: SeedServerState) -> Router {
    Router::new()
        .route(
            "/v1/models/{model}/{revision}/manifest",
            get(serve_manifest),
        )
        .route("/v1/chunks/{digest}", get(serve_chunk))
        .with_state(state)
}

async fn serve_manifest(
    State(state): State<SeedServerState>,
    AxumPath((model, revision)): AxumPath<(String, String)>,
) -> Result<Json<ModelManifest>, StatusCode> {
    state
        .manifests
        .get(&(model, revision))
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn serve_chunk(
    State(state): State<SeedServerState>,
    AxumPath(digest): AxumPath<String>,
) -> Response {
    match state.store.read(&digest) {
        Ok(bytes) => {
            let mut response = Bytes::from(bytes).into_response();
            response.headers_mut().insert(
                CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            response
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

pub async fn send_tensor(endpoint: &str, frame: &TensorFrame) -> Result<()> {
    frame.validate()?;
    reqwest::Client::new()
        .post(endpoint)
        .json(frame)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

pub async fn send_tensor_with_failover(
    endpoints: &[String],
    frame: &TensorFrame,
) -> Result<String> {
    ensure!(!endpoints.is_empty(), "tensor route has no endpoints");
    let mut failures = Vec::new();
    for endpoint in endpoints {
        match send_tensor(endpoint, frame).await {
            Ok(()) => return Ok(endpoint.clone()),
            Err(error) => failures.push(format!("{endpoint}: {error}")),
        }
    }
    anyhow::bail!("all tensor routes failed: {}", failures.join("; "))
}

pub fn verify_file(path: impl AsRef<Path>, expected_sha256: &str) -> Result<()> {
    let bytes = std::fs::read(path)?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes)) == expected_sha256,
        "file checksum mismatch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hocmesh_model::{ModelFormat, sha256};

    fn manifest() -> ModelManifest {
        ModelManifest {
            schema_version: 1,
            model_id: "m".into(),
            revision: "1".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: None,
            tensor_dtype: None,
            total_size_bytes: 2,
            chunks: vec![
                ChunkRef {
                    index: 0,
                    sha256: sha256(b"a"),
                    size_bytes: 1,
                },
                ChunkRef {
                    index: 1,
                    sha256: sha256(b"b"),
                    size_bytes: 1,
                },
            ],
            metadata: Default::default(),
        }
    }

    #[test]
    fn tensor_corruption_is_rejected() {
        let frame = TensorFrame::new("job", "stream", 0, vec![1], "u8", vec![1]).unwrap();
        let mut bytes = frame.encode().unwrap();
        let last = bytes.len() - 2;
        bytes[last] ^= 1;
        assert!(TensorFrame::decode(&bytes).is_err());
    }

    #[test]
    fn tensor_assembler_reorders_and_rejects_replay() {
        let frame0 = TensorFrame::new("job", "stream", 0, vec![1], "u8", vec![0]).unwrap();
        let frame1 = TensorFrame::new("job", "stream", 1, vec![1], "u8", vec![1]).unwrap();
        let mut assembler = TensorAssembler::new(4).unwrap();
        assert!(assembler.accept(frame1).unwrap().is_empty());
        let ready = assembler.accept(frame0.clone()).unwrap();
        assert_eq!(
            ready.iter().map(|frame| frame.sequence).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(assembler.accept(frame0).is_err());
    }

    #[test]
    fn tensor_validation_enforces_routes_shapes_payloads_and_window() {
        assert!(TensorAssembler::new(0).is_err());
        assert!(TensorFrame::new("", "stream", 0, vec![1], "u8", vec![1]).is_err());
        assert!(TensorFrame::new("job", "stream", 0, vec![0], "u8", vec![1]).is_err());
        assert!(TensorFrame::new("job", "stream", 0, vec![1], "", vec![1]).is_err());
        assert!(TensorFrame::new("job", "stream", 0, vec![1], "u8", vec![]).is_err());
        let mut assembler = TensorAssembler::new(2).unwrap();
        let outside = TensorFrame::new("job", "stream", 2, vec![1], "u8", vec![1]).unwrap();
        assert!(assembler.accept(outside).is_err());
    }

    #[tokio::test]
    async fn tensor_transport_fails_over_to_healthy_route() {
        let inbox = TensorInbox::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_inbox = inbox.clone();
        let server = tokio::spawn(async move {
            axum::serve(listener, tensor_router(server_inbox))
                .await
                .unwrap()
        });
        let frame = TensorFrame::new("job", "stream", 0, vec![1], "u8", vec![7]).unwrap();
        let healthy = format!("http://{address}/v1/tensors");
        let selected = send_tensor_with_failover(
            &["http://127.0.0.1:1/v1/tensors".into(), healthy.clone()],
            &frame,
        )
        .await
        .unwrap();
        assert_eq!(selected, healthy);
        assert_eq!(inbox.take_delivered().await, vec![frame]);
        server.abort();
    }

    #[test]
    fn seeding_is_rarest_first_and_balanced() {
        let m = manifest();
        let peers = vec![
            PeerInventory {
                peer: "a".into(),
                chunks: [m.chunks[0].sha256.clone(), m.chunks[1].sha256.clone()]
                    .into_iter()
                    .collect(),
                latency_ms: 2,
                failures: 0,
            },
            PeerInventory {
                peer: "b".into(),
                chunks: [m.chunks[1].sha256.clone()].into_iter().collect(),
                latency_ms: 1,
                failures: 0,
            },
        ];
        let plan = plan_seeding(&m, &BTreeSet::new(), &peers).unwrap();
        assert_eq!(plan[0].chunk.index, 0);
        assert_ne!(plan[0].peer, plan[1].peer);
    }

    #[test]
    fn seeding_requires_a_source_for_every_missing_chunk() {
        let m = manifest();
        assert!(plan_seeding(&m, &BTreeSet::new(), &[]).is_err());
        let local = m.chunks.iter().map(|chunk| chunk.sha256.clone()).collect();
        assert!(plan_seeding(&m, &local, &[]).unwrap().is_empty());
        assert!(HttpPeerSource::new("file:///tmp/peer").is_err());
    }

    #[tokio::test]
    async fn all_failed_tensor_routes_return_an_aggregate_error() {
        let frame = TensorFrame::new("job", "stream", 0, vec![1], "u8", vec![7]).unwrap();
        assert!(send_tensor_with_failover(&[], &frame).await.is_err());
        let error = send_tensor_with_failover(&["http://127.0.0.1:1/v1/tensors".into()], &frame)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("all tensor routes failed"));
    }

    #[tokio::test]
    async fn http_peer_seeds_and_verifies_all_chunks() {
        let unique = format!("{}-{}", std::process::id(), hocmesh_protocol_time());
        let source_root = std::env::temp_dir().join(format!("hocmesh-seed-source-{unique}"));
        let target_root = std::env::temp_dir().join(format!("hocmesh-seed-target-{unique}"));
        let source_store = Arc::new(ChunkStore::open(&source_root).unwrap());
        let a = source_store.put(b"a").unwrap();
        let b = source_store.put(b"b").unwrap();
        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "m".into(),
            revision: "1".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: None,
            tensor_dtype: None,
            total_size_bytes: 2,
            chunks: vec![
                ChunkRef {
                    index: 0,
                    sha256: a,
                    size_bytes: 1,
                },
                ChunkRef {
                    index: 1,
                    sha256: b,
                    size_bytes: 1,
                },
            ],
            metadata: Default::default(),
        };
        let state = SeedServerState::new(source_store.clone(), [manifest.clone()]).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, seed_router(state)).await.unwrap() });
        let target = ChunkStore::open(&target_root).unwrap();
        let seeded = seed_from_peer(
            &HttpPeerSource::new(format!("http://{address}")).unwrap(),
            &target,
            "m",
            "1",
        )
        .await
        .unwrap();
        assert_eq!(seeded, manifest);
        assert!(seeded.chunks.iter().all(|chunk| target.contains(chunk)));
        server.abort();
        drop(target);
        drop(source_store);
        std::fs::remove_dir_all(source_root).unwrap();
        std::fs::remove_dir_all(target_root).unwrap();
    }

    fn hocmesh_protocol_time() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }
}
