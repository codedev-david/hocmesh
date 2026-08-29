//! One contiguous range of a model's transformer blocks, executed.
//!
//! This is the piece that was missing. A stage holds blocks `[start, end)` and
//! nothing else, is handed the activation the previous stage produced, and
//! hands on the activation the next one needs. The first stage additionally
//! turns token ids into the first activation; the last additionally turns the
//! final activation into logits.
//!
//! Everything here is `f32` and sequential on purpose. The arithmetic a block
//! performs depends only on the activation it was given and the weights it
//! holds, so running blocks `0..8` and `8..16` in two processes produces the
//! same bits as running `0..16` in one — the property the whole design rests
//! on, and the one [`crate::tests`] checks rather than assumes.

use anyhow::{Context, Result, ensure};
use std::ops::Range;

use crate::config::{ModelConfig, RopeStyle};
use crate::weights::{Tensor, WeightFile};

/// The activation passed between stages.
///
/// Carries the positions it covers because a stage has no other way to know
/// them: rotary embeddings and the causal mask are both functions of absolute
/// position, and a stage that inferred position from its own cache length
/// could not be restarted, retried, or moved to another node mid-sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Activation {
    /// Absolute position of the first token in this step.
    pub position: u32,
    /// Row-major `[tokens][embedding_length]`.
    pub hidden: Vec<f32>,
    pub tokens: usize,
}

impl Activation {
    /// Encode for the wire, exactly.
    ///
    /// Little-endian `f32` bits, not decimal. A JSON round trip of a float is
    /// not guaranteed to return the same bits, and an activation that changes
    /// in its last bit on the way between two machines would make a split run
    /// differ from a whole one for no reason anybody could find.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.hidden.len() * 4);
        out.extend_from_slice(&self.position.to_le_bytes());
        out.extend_from_slice(&(self.tokens as u32).to_le_bytes());
        for value in &self.hidden {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= 8,
            "activation frame is too short for a header"
        );
        let position = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let tokens = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let body = &bytes[8..];
        ensure!(
            body.len().is_multiple_of(4),
            "activation frame holds {} bytes, not a whole number of f32",
            body.len()
        );
        let hidden = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>();
        ensure!(
            tokens > 0 && hidden.len().is_multiple_of(tokens),
            "{} values do not divide into {tokens} tokens",
            hidden.len()
        );
        Ok(Activation {
            position,
            hidden,
            tokens,
        })
    }
}

/// The nine tensors one transformer block is made of.
struct BlockWeights {
    attn_norm: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
    attn_output: Tensor,
    ffn_norm: Tensor,
    ffn_gate: Tensor,
    ffn_up: Tensor,
    ffn_down: Tensor,
}

/// Keys and values already computed for earlier positions in this sequence.
#[derive(Default)]
struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// A loaded range of blocks, ready to run.
pub struct Stage {
    pub config: ModelConfig,
    pub blocks: Range<u32>,
    embedding: Option<Tensor>,
    output_norm: Option<Tensor>,
    output_head: Option<Tensor>,
    layers: Vec<BlockWeights>,
    cache: Vec<KvCache>,
}

impl Stage {
    /// Load blocks `[blocks.start, blocks.end)` out of an open model file.
    ///
    /// Loads the embedding table only for a stage that starts at block 0, and
    /// the output head only for one that ends at the last block. A middle
    /// stage therefore never reads either, which is what lets it run on a
    /// machine that does not hold them.
    pub fn load(file: &mut WeightFile, blocks: Range<u32>) -> Result<Self> {
        let mut config = ModelConfig::from_header(&file.header)?;
        config.validate_range(&blocks)?;
        let first = blocks.start == 0;
        let last = blocks.end == config.block_count;

        let embedding = first.then(|| file.load("token_embd.weight")).transpose()?;
        if let Some(table) = &embedding {
            table.expect_shape(
                "token_embd.weight",
                &[u64::from(config.embedding_length), table.dimensions[1]],
            )?;
            // The file's own vocabulary, which is the one that matters: a
            // metadata key that disagrees with the table would index past it.
            config.vocab_size = table.dimensions[1] as u32;
        }

        let embed = u64::from(config.embedding_length);
        let ffn = u64::from(config.feed_forward_length);
        let kv = u64::from(config.kv_width());
        let mut layers = Vec::with_capacity(blocks.len());
        for index in blocks.clone() {
            let get = |file: &mut WeightFile, suffix: &str, shape: &[u64]| -> Result<Tensor> {
                let name = format!("blk.{index}.{suffix}");
                let tensor = file.load(&name)?;
                tensor.expect_shape(&name, shape)?;
                Ok(tensor)
            };
            layers.push(BlockWeights {
                attn_norm: get(file, "attn_norm.weight", &[embed])?,
                attn_q: get(file, "attn_q.weight", &[embed, embed])?,
                attn_k: get(file, "attn_k.weight", &[embed, kv])?,
                attn_v: get(file, "attn_v.weight", &[embed, kv])?,
                attn_output: get(file, "attn_output.weight", &[embed, embed])?,
                ffn_norm: get(file, "ffn_norm.weight", &[embed])?,
                ffn_gate: get(file, "ffn_gate.weight", &[embed, ffn])?,
                ffn_up: get(file, "ffn_up.weight", &[embed, ffn])?,
                ffn_down: get(file, "ffn_down.weight", &[ffn, embed])?,
            });
        }

        let output_norm = last.then(|| file.load("output_norm.weight")).transpose()?;
        // A model with no `output.weight` ties its head to its embedding
        // table: one matrix, read as a lookup at the front of the model and as
        // rows of output weights at the back. The stage holding the last block
        // therefore has to read that table even when it does not hold block 0
        // and will never embed anything with it. It can: a shard carries the
        // shared tensors whichever end of the model it holds, because either
        // end needs them. Tied embeddings are the rule rather than the
        // exception among the small models worth splitting, so refusing them
        // would rule out most of the point.
        let output_head = if last {
            match file.load_optional("output.weight")? {
                Some(head) => Some(head),
                None if embedding.is_some() => embedding.clone(),
                None => Some(file.load("token_embd.weight").context(
                    "this model ties its output head to its embedding table, but this \
                     file holds neither, so the stage with the last block has nothing \
                     to turn its activation into logits with",
                )?),
            }
        } else {
            None
        };
        if let Some(head) = &output_head {
            // Rows of the head are vocabulary entries, its columns the model
            // width. Taking the count from the tensor rather than from
            // metadata stops a header that disagrees with the file from
            // indexing past the end of it.
            head.expect_shape(
                "output head",
                &[u64::from(config.embedding_length), head.dimensions[1]],
            )?;
            config.vocab_size = head.dimensions[1] as u32;
        }
        let depth = layers.len();
        Ok(Stage {
            config,
            blocks,
            embedding,
            output_norm,
            output_head,
            layers,
            cache: (0..depth).map(|_| KvCache::default()).collect(),
        })
    }

    #[must_use]
    pub fn is_first(&self) -> bool {
        self.blocks.start == 0
    }

    #[must_use]
    pub fn is_last(&self) -> bool {
        self.blocks.end == self.config.block_count
    }

    /// Forget the sequence so far. Called between prompts, never within one.
    pub fn reset(&mut self) {
        for cache in &mut self.cache {
            cache.keys.clear();
            cache.values.clear();
        }
    }

    /// Turn token ids into the activation the first block consumes.
    pub fn embed(&self, tokens: &[u32], position: u32) -> Result<Activation> {
        let table = self
            .embedding
            .as_ref()
            .context("only the stage holding block 0 can embed tokens")?;
        let width = self.config.embedding_length as usize;
        let mut hidden = Vec::with_capacity(tokens.len() * width);
        for token in tokens {
            ensure!(
                (*token as usize) < table.rows(),
                "token {token} is outside a vocabulary of {}",
                table.rows()
            );
            hidden.extend_from_slice(table.row(*token as usize));
        }
        Ok(Activation {
            position,
            hidden,
            tokens: tokens.len(),
        })
    }

    /// Run this stage's blocks over an activation.
    pub fn forward(&mut self, input: &Activation) -> Result<Activation> {
        let width = self.config.embedding_length as usize;
        ensure!(
            input.tokens > 0 && input.hidden.len() == input.tokens * width,
            "activation is {} values, expected {} x {width}",
            input.hidden.len(),
            input.tokens
        );
        let mut hidden = input.hidden.clone();
        for depth in 0..self.layers.len() {
            self.run_block(depth, &mut hidden, input.position, input.tokens)?;
        }
        Ok(Activation {
            position: input.position,
            hidden,
            tokens: input.tokens,
        })
    }

    /// Turn the final activation into a score per vocabulary entry, for the
    /// last position only — the only one a sampler ever looks at.
    pub fn logits(&self, input: &Activation) -> Result<Vec<f32>> {
        let norm = self
            .output_norm
            .as_ref()
            .context("only the stage holding the last block can produce logits")?;
        let head = self
            .output_head
            .as_ref()
            .context("this stage holds no output head")?;
        let width = self.config.embedding_length as usize;
        let last = &input.hidden[(input.tokens - 1) * width..input.tokens * width];
        let mut normed = vec![0.0f32; width];
        rms_norm(last, &norm.values, self.config.rms_norm_eps, &mut normed);
        let mut logits = vec![0.0f32; head.rows()];
        mat_vec(head, &normed, &mut logits);
        Ok(logits)
    }

    fn run_block(
        &mut self,
        depth: usize,
        hidden: &mut [f32],
        position: u32,
        tokens: usize,
    ) -> Result<()> {
        let config = &self.config;
        let width = config.embedding_length as usize;
        let head_dim = config.head_dim() as usize;
        let kv_width = config.kv_width() as usize;
        let group = config.group_size() as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let layer = &self.layers[depth];
        let cache = &mut self.cache[depth];

        let mut normed = vec![0.0f32; width];
        let mut q = vec![0.0f32; width];
        let mut k = vec![0.0f32; kv_width];
        let mut v = vec![0.0f32; kv_width];
        let mut attended = vec![0.0f32; width];
        let mut projected = vec![0.0f32; width];
        let mut gate = vec![0.0f32; config.feed_forward_length as usize];
        let mut up = vec![0.0f32; config.feed_forward_length as usize];

        for token in 0..tokens {
            let at = position as usize + token;
            let row = &mut hidden[token * width..(token + 1) * width];

            // -- attention --
            rms_norm(
                row,
                &layer.attn_norm.values,
                config.rms_norm_eps,
                &mut normed,
            );
            mat_vec(&layer.attn_q, &normed, &mut q);
            mat_vec(&layer.attn_k, &normed, &mut k);
            mat_vec(&layer.attn_v, &normed, &mut v);
            rope(&mut q, head_dim, config, at);
            rope(&mut k, head_dim, config, at);

            // The cache is append-only within a sequence, so a position is
            // written exactly once and every later step reads the same bytes.
            ensure!(
                cache.keys.len() == at * kv_width,
                "position {at} arrived out of order: the cache holds {} positions",
                cache.keys.len() / kv_width.max(1)
            );
            cache.keys.extend_from_slice(&k);
            cache.values.extend_from_slice(&v);

            let mut scores = vec![0.0f32; at + 1];
            for head in 0..config.head_count as usize {
                let kv_head = head / group;
                let query = &q[head * head_dim..(head + 1) * head_dim];
                for (past, score) in scores.iter_mut().enumerate() {
                    let key = &cache.keys[past * kv_width + kv_head * head_dim..][..head_dim];
                    *score = dot(query, key) * scale;
                }
                softmax(&mut scores);
                let out = &mut attended[head * head_dim..(head + 1) * head_dim];
                out.fill(0.0);
                for (past, weight) in scores.iter().enumerate() {
                    let value = &cache.values[past * kv_width + kv_head * head_dim..][..head_dim];
                    for (slot, element) in out.iter_mut().zip(value) {
                        *slot += weight * element;
                    }
                }
            }
            mat_vec(&layer.attn_output, &attended, &mut projected);
            for (slot, delta) in row.iter_mut().zip(&projected) {
                *slot += delta;
            }

            // -- feed forward --
            rms_norm(
                row,
                &layer.ffn_norm.values,
                config.rms_norm_eps,
                &mut normed,
            );
            mat_vec(&layer.ffn_gate, &normed, &mut gate);
            mat_vec(&layer.ffn_up, &normed, &mut up);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = silu(*g) * u;
            }
            mat_vec(&layer.ffn_down, &gate, &mut projected);
            for (slot, delta) in row.iter_mut().zip(&projected) {
                *slot += delta;
            }
        }
        Ok(())
    }
}

/// `x * w / sqrt(mean(x^2) + eps)`, the normalisation the llama family uses.
fn rms_norm(input: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_square = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((slot, value), w) in out.iter_mut().zip(input).zip(weight) {
        *slot = value * scale * w;
    }
}

/// `out[r] = weight_row(r) . input`.
///
/// GGUF stores a weight matrix with its input dimension fastest-varying, so a
/// row is contiguous and this is a sequence of dot products in a fixed order —
/// which is what makes the result reproducible across machines.
fn mat_vec(weight: &Tensor, input: &[f32], out: &mut [f32]) {
    let row_len = weight.row_len();
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = dot(
            &weight.values[index * row_len..(index + 1) * row_len],
            input,
        );
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softmax, shifted by the maximum so a large score cannot overflow `exp`.
fn softmax(scores: &mut [f32]) {
    let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for score in scores.iter_mut() {
        *score = (*score - peak).exp();
        total += *score;
    }
    // `total` is at least 1, because the peak contributes exp(0).
    for score in scores.iter_mut() {
        *score /= total;
    }
}

/// Rotate each head's leading `rope_dimension_count` elements by an angle that
/// grows with absolute position.
///
/// Which two elements form a pair is the whole of the difference between the
/// two conventions, and getting it wrong is not detectable from the output
/// shape — see [`RopeStyle`].
fn rope(vector: &mut [f32], head_dim: usize, config: &ModelConfig, position: usize) {
    let rotated = config.rope_dimension_count as usize;
    let half = rotated / 2;
    for head in vector.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let frequency = config.rope_theta.powf(-((2 * i) as f32) / rotated as f32);
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let (a, b) = match config.rope_style {
                RopeStyle::Interleaved => (2 * i, 2 * i + 1),
                RopeStyle::Halved => (i, i + half),
            };
            let (x, y) = (head[a], head[b]);
            head[a] = x * cos - y * sin;
            head[b] = x * sin + y * cos;
        }
    }
}
