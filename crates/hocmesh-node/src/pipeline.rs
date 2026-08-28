//! Running one model across several machines, none of which holds it whole.
//!
//! This is the piece the rest of the project was built to make possible. A node
//! is handed a contiguous range of a model's transformer blocks and the address
//! of whoever holds the next range. It loads only its own weights — from a file
//! that is genuinely missing everybody else's — runs its blocks, and hands the
//! activation on. The last stage turns the activation into logits and the
//! answer travels back down the chain the way it came.
//!
//! Two properties make it worth having rather than merely interesting:
//!
//! * The output is *identical* to running the same model in one process. Not
//!   close: identical, bit for bit. Every block's arithmetic depends only on the
//!   activation it was given and the weights it holds, so there is nothing for a
//!   split to change. `stage-run --model` produces the reference locally and
//!   `stage-run --head` produces it over the network; the test asserts the two
//!   digests match.
//! * A stage physically cannot read a neighbour's weights. It is checked before
//!   any weight is loaded, against the byte ranges the node actually fetched —
//!   because a hole in a sparse file reads back as zeros, and zeros are a valid
//!   weight matrix that produces confident nonsense.

use anyhow::{Context, Result, anyhow, bail, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use hocmesh_engine::{Activation, ModelConfig, Stage, WeightFile};
use hocmesh_model::gguf::ByteExtent;
use serde::{Deserialize, Serialize};
use std::{ops::Range, path::Path, sync::Arc, time::Duration};
use tokio::sync::Mutex;

/// How long a stage waits on the stage after it.
///
/// Long enough for a large block range on a slow machine, short enough that a
/// dead next hop surfaces as an error rather than as a request that never
/// returns.
const HOP_TIMEOUT: Duration = Duration::from_secs(120);

/// What a stage says about itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageInfo {
    pub architecture: String,
    /// The blocks this stage holds, as `[start, end)`.
    pub blocks: [u32; 2],
    /// Blocks in the whole model, which every stage agrees on.
    pub block_count: u32,
    pub is_first: bool,
    pub is_last: bool,
    /// The next stage's address, absent on the tail.
    pub next: Option<String>,
    /// Bytes of the model file this stage actually holds, and the file's full
    /// length. A stage serving a real shard holds a fraction of the second
    /// number, and that gap is the whole point.
    pub bytes_present: u64,
    pub bytes_total: u64,
}

/// A prompt handed to the head of a chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRequest {
    pub token: u32,
    pub position: u32,
}

/// The sidecar written beside a partially materialised model.
///
/// It records which byte ranges were fetched. Without it a stage would have no
/// way to tell a hole from a weight that happens to be zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    pub model_id: String,
    pub revision: String,
    pub blocks: [u32; 2],
    pub chunks_kept: usize,
    pub chunks_total: usize,
    pub bytes_present: u64,
    pub bytes_total: u64,
    /// Present ranges as `[start, end)` pairs, in file order.
    pub present: Vec<[u64; 2]>,
}

impl ShardManifest {
    #[must_use]
    pub fn extents(&self) -> Vec<ByteExtent> {
        self.present
            .iter()
            .map(|pair| ByteExtent {
                start: pair[0],
                end: pair[1],
            })
            .collect()
    }

    /// Where the sidecar for a model file lives.
    #[must_use]
    pub fn path_for(model: &Path) -> std::path::PathBuf {
        let mut name = model.as_os_str().to_os_string();
        name.push(".shard.json");
        std::path::PathBuf::from(name)
    }
}

/// Parse `4..12`, the way a layer range is written everywhere else.
pub fn parse_blocks(text: &str) -> Result<Range<u32>> {
    let (start, end) = text
        .split_once("..")
        .with_context(|| format!("layer range {text:?} is not written as start..end"))?;
    let start: u32 = start
        .trim()
        .parse()
        .with_context(|| format!("{start:?} is not a block index"))?;
    let end: u32 = end
        .trim()
        .parse()
        .with_context(|| format!("{end:?} is not a block index"))?;
    ensure!(
        start < end,
        "{text}: a stage holding no blocks is a network hop that computes nothing"
    );
    Ok(start..end)
}

/// Which chunks of a model a stage needs in order to run its own blocks.
///
/// The shared tensors — the embedding table, the final norm, the output head —
/// come along too, because whichever stage holds an end of the model needs
/// them. Everything else in the file is somebody else's problem.
pub fn chunks_for_blocks(
    manifest: &hocmesh_model::ModelManifest,
    header: &[u8],
    blocks: Range<u32>,
) -> Result<Vec<u32>> {
    let directory = hocmesh_model::gguf::tensor_directory(header)?
        .context("model has no readable tensor directory")?;
    let total = manifest.total_size_bytes;
    let mut extents = directory.extents_for_layers(blocks, total);
    extents.extend(directory.extents_of(&directory.shared_tensors(), total));
    // The header itself: without it there is no directory, and so no way to
    // find anything. It is the one region every stage holds.
    extents.push(ByteExtent {
        start: 0,
        end: directory.data_start.min(total),
    });
    let chunk_size = manifest
        .chunks
        .first()
        .map(|chunk| chunk.size_bytes)
        .filter(|size| *size > 0)
        .context("manifest declares no chunks")?;
    let indexes = hocmesh_model::gguf::TensorDirectory::chunks_for_extents(&extents, chunk_size)?;
    Ok(indexes.into_iter().map(|index| index as u32).collect())
}

/// One stage, ready to serve.
#[derive(Clone)]
pub struct StageState {
    stage: Arc<Mutex<Stage>>,
    next: Option<String>,
    info: StageInfo,
}

impl StageState {
    /// Load a stage from a model file, refusing to start if the file is missing
    /// any byte the stage's own blocks need.
    pub fn load(model: &Path, blocks: Range<u32>, next: Option<String>) -> Result<Self> {
        let mut file = WeightFile::open(model)?;
        let total = std::fs::metadata(model)?.len();
        let shard_path = ShardManifest::path_for(model);
        let (present, bytes_present) = if shard_path.exists() {
            let shard: ShardManifest = serde_json::from_slice(&std::fs::read(&shard_path)?)
                .with_context(|| format!("reading {}", shard_path.display()))?;
            (shard.extents(), shard.bytes_present)
        } else {
            // No sidecar means the file was materialised whole, and a whole
            // file really does hold every byte.
            (
                vec![ByteExtent {
                    start: 0,
                    end: total,
                }],
                total,
            )
        };
        file.assert_layers_present(blocks.clone(), &present)?;

        let config = ModelConfig::from_header(&file.header)?;
        let stage = Stage::load(&mut file, blocks.clone())?;
        let info = StageInfo {
            architecture: config.architecture.clone(),
            blocks: [blocks.start, blocks.end],
            block_count: config.block_count,
            is_first: stage.is_first(),
            is_last: stage.is_last(),
            next: next.clone(),
            bytes_present,
            bytes_total: total,
        };
        ensure!(
            stage.is_last() == next.is_none(),
            "a stage holding blocks {}..{} of {} {} a next hop",
            blocks.start,
            blocks.end,
            config.block_count,
            if stage.is_last() {
                "ends the model and must not be given"
            } else {
                "does not end the model and needs"
            }
        );
        Ok(StageState {
            stage: Arc::new(Mutex::new(stage)),
            next,
            info,
        })
    }

    #[must_use]
    pub fn info(&self) -> &StageInfo {
        &self.info
    }
}

/// An error on the wire, which is an error a caller can act on rather than a
/// panic that takes the stage down with it.
struct StageError(anyhow::Error);

impl IntoResponse for StageError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, format!("{:#}", self.0)).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for StageError {
    fn from(error: E) -> Self {
        StageError(error.into())
    }
}

pub fn stage_router(state: StageState) -> Router {
    Router::new()
        .route("/stage/info", get(info))
        .route("/stage/token", post(token))
        .route("/stage/forward", post(forward))
        .route("/stage/reset", post(reset))
        .with_state(state)
}

async fn info(State(state): State<StageState>) -> Json<StageInfo> {
    Json(state.info.clone())
}

/// Embed a token and start it down the chain. Only the head can do this: it is
/// the only stage that holds the embedding table.
async fn token(
    State(state): State<StageState>,
    Json(request): Json<TokenRequest>,
) -> Result<Vec<u8>, StageError> {
    let activation = {
        let stage = state.stage.lock().await;
        if !stage.is_first() {
            return Err(StageError(anyhow!(
                "this stage holds blocks {}..{} and has no embedding table; \
                 send tokens to the head of the chain",
                state.info.blocks[0],
                state.info.blocks[1]
            )));
        }
        stage.embed(&[request.token], request.position)?
    };
    advance(&state, activation).await
}

/// Take an activation from the stage before, run this stage's blocks, and pass
/// it on.
async fn forward(State(state): State<StageState>, body: Bytes) -> Result<Vec<u8>, StageError> {
    let activation = Activation::from_bytes(&body)?;
    advance(&state, activation).await
}

/// Run the blocks and either hand the result to the next stage or, at the tail,
/// turn it into logits.
async fn advance(state: &StageState, input: Activation) -> Result<Vec<u8>, StageError> {
    let output = {
        let mut stage = state.stage.lock().await;
        stage.forward(&input)?
    };
    match &state.next {
        Some(address) => {
            let body = output.to_bytes();
            Ok(post_activation(address, body).await?)
        }
        None => {
            let stage = state.stage.lock().await;
            let logits = stage.logits(&output)?;
            Ok(encode_f32(&logits))
        }
    }
}

/// Clear the attention caches along the whole chain.
///
/// A stage that reset while its neighbours did not would silently answer from
/// half a conversation, so this propagates rather than being something an
/// operator has to remember to do on every machine.
async fn reset(State(state): State<StageState>) -> Result<StatusCode, StageError> {
    state.stage.lock().await.reset();
    if let Some(address) = &state.next {
        let response = client()?
            .post(format!("{}/stage/reset", address.trim_end_matches('/')))
            .send()
            .await
            .with_context(|| format!("resetting the stage at {address}"))?;
        ensure_ok(response, address).await?;
    }
    Ok(StatusCode::NO_CONTENT)
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(HOP_TIMEOUT)
        .build()
        .context("building the inter-stage HTTP client")
}

async fn post_activation(address: &str, body: Vec<u8>) -> Result<Vec<u8>> {
    let response = client()?
        .post(format!("{}/stage/forward", address.trim_end_matches('/')))
        .header("content-type", "application/octet-stream")
        .body(body)
        .send()
        .await
        .with_context(|| format!("handing the activation to the stage at {address}"))?;
    let response = ensure_ok(response, address).await?;
    Ok(response.bytes().await?.to_vec())
}

async fn ensure_ok(response: reqwest::Response, address: &str) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    bail!("the stage at {address} answered {status}: {body}")
}

/// Logits on the wire, as raw little-endian `f32`.
///
/// Not JSON. A JSON round trip of a float is not guaranteed to return the same
/// bits, and the entire claim being made here is that a split model returns the
/// same bits.
fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        bytes.len().is_multiple_of(4),
        "a logits response of {} bytes is not a whole number of floats",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// What a run produced, in a form two runs can be compared by.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Generated {
    /// The prompt followed by everything greedily generated from it.
    pub tokens: Vec<u32>,
    /// How many forward passes were run.
    pub steps: usize,
    /// SHA-256 over the exact bytes of every logit of every step.
    ///
    /// The comparison that matters is over bit patterns, not values: two runs
    /// that agree to six decimal places have already diverged, and a tolerance
    /// would hide it. A digest compares all of them at once and is the same
    /// length whatever the vocabulary is.
    pub logits_sha256: String,
    /// The single most likely token at each step, kept so a mismatch says
    /// something more useful than "the digests differ".
    pub argmax_per_step: Vec<u32>,
}

struct Trace {
    bytes: Vec<u8>,
    argmax: Vec<u32>,
}

impl Trace {
    fn new() -> Self {
        Trace {
            bytes: Vec::new(),
            argmax: Vec::new(),
        }
    }

    fn record(&mut self, logits: &[f32]) -> Result<u32> {
        self.bytes.extend_from_slice(&encode_f32(logits));
        let best = argmax(logits)?;
        self.argmax.push(best);
        Ok(best)
    }

    fn finish(self, tokens: Vec<u32>, steps: usize) -> Generated {
        Generated {
            tokens,
            steps,
            logits_sha256: hocmesh_model::sha256(&self.bytes),
            argmax_per_step: self.argmax,
        }
    }
}

/// The most likely next token.
///
/// Ties go to the lower index, deterministically, so two runs of the same model
/// cannot disagree about a tie.
fn argmax(logits: &[f32]) -> Result<u32> {
    let mut best = 0_usize;
    let mut best_value = f32::NEG_INFINITY;
    ensure!(!logits.is_empty(), "the model produced no logits");
    for (index, value) in logits.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    u32::try_from(best).context("token index does not fit in u32")
}

/// Drive a prompt through a chain of stages over the network.
pub async fn generate_over_chain(
    head: &str,
    prompt: &[u32],
    max_new_tokens: usize,
) -> Result<Generated> {
    ensure!(!prompt.is_empty(), "the prompt has no tokens");
    let head = head.trim_end_matches('/').to_string();
    let http = client()?;
    let response = http
        .post(format!("{head}/stage/reset"))
        .send()
        .await
        .with_context(|| format!("resetting the chain at {head}"))?;
    ensure_ok(response, &head).await?;

    let mut trace = Trace::new();
    let mut tokens = prompt.to_vec();
    let mut steps = 0_usize;
    let mut position = 0_u32;
    let mut cursor = 0_usize;
    loop {
        let token = tokens[cursor];
        let response = http
            .post(format!("{head}/stage/token"))
            .json(&TokenRequest { token, position })
            .send()
            .await
            .with_context(|| format!("sending token {token} at position {position}"))?;
        let body = ensure_ok(response, &head).await?.bytes().await?;
        let next = trace.record(&decode_f32(&body)?)?;
        steps += 1;
        position += 1;
        cursor += 1;
        if cursor >= tokens.len() {
            if tokens.len() - prompt.len() >= max_new_tokens {
                break;
            }
            tokens.push(next);
        }
    }
    Ok(trace.finish(tokens, steps))
}

/// Run the same prompt through the same model in this process.
///
/// The reference the distributed run is measured against. It reads the whole
/// file, which is exactly what a machine that can hold the whole model does and
/// exactly what the distributed path exists to avoid needing.
pub fn generate_locally(model: &Path, prompt: &[u32], max_new_tokens: usize) -> Result<Generated> {
    ensure!(!prompt.is_empty(), "the prompt has no tokens");
    let mut file = WeightFile::open(model)?;
    let config = ModelConfig::from_header(&file.header)?;
    let mut stage = Stage::load(&mut file, 0..config.block_count)?;

    let mut trace = Trace::new();
    let mut tokens = prompt.to_vec();
    let mut steps = 0_usize;
    let mut position = 0_u32;
    let mut cursor = 0_usize;
    loop {
        let activation = stage.embed(&[tokens[cursor]], position)?;
        let output = stage.forward(&activation)?;
        let next = trace.record(&stage.logits(&output)?)?;
        steps += 1;
        position += 1;
        cursor += 1;
        if cursor >= tokens.len() {
            if tokens.len() - prompt.len() >= max_new_tokens {
                break;
            }
            tokens.push(next);
        }
    }
    Ok(trace.finish(tokens, steps))
}

/// Parse `3,17,5` — a prompt as token ids, so a run needs no tokenizer.
///
/// Tokenising is a separate problem with its own correctness argument. Keeping
/// it out means this path proves the thing it is meant to prove and nothing
/// else.
pub fn parse_tokens(text: &str) -> Result<Vec<u32>> {
    text.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .with_context(|| format!("{part:?} is not a token id"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_range_is_read_the_way_it_is_written() {
        assert_eq!(parse_blocks("4..12").unwrap(), 4..12);
        assert_eq!(parse_blocks(" 0 .. 1 ").unwrap(), 0..1);
        assert!(
            parse_blocks("4..4").is_err(),
            "an empty range computes nothing"
        );
        assert!(
            parse_blocks("12..4").is_err(),
            "a backwards range is not a range"
        );
        assert!(parse_blocks("4-12").is_err(), "the separator is ..");
    }

    #[test]
    fn a_prompt_is_a_list_of_token_ids() {
        assert_eq!(parse_tokens("3,17,5").unwrap(), vec![3, 17, 5]);
        assert_eq!(parse_tokens("7").unwrap(), vec![7]);
        assert!(parse_tokens("3,cat").is_err());
    }

    #[test]
    fn logits_survive_the_wire_unchanged() {
        // Including the values a JSON round trip is least likely to preserve.
        let values = vec![0.1_f32, -0.0, f32::MIN_POSITIVE, 1.0 / 3.0, 1e30, -2.5e-8];
        let round_tripped = decode_f32(&encode_f32(&values)).unwrap();
        assert_eq!(
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            round_tripped
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_truncated_logits_response_is_refused() {
        assert!(decode_f32(&[0, 1, 2]).is_err());
    }

    #[test]
    fn the_most_likely_token_breaks_ties_towards_the_lower_index() {
        assert_eq!(argmax(&[0.0, 5.0, 5.0, 1.0]).unwrap(), 1);
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]).unwrap(), 1);
        assert!(argmax(&[]).is_err());
    }

    #[test]
    fn a_shard_sidecar_sits_beside_its_model() {
        let path = ShardManifest::path_for(Path::new("/models/llama.gguf"));
        assert!(path.to_string_lossy().ends_with("llama.gguf.shard.json"));
    }
}
