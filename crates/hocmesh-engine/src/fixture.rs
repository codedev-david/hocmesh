//! Building a real GGUF file small enough to run in a test.
//!
//! The property that matters — that a model split across stages computes the
//! same thing as the same model run whole — cannot be checked against a
//! downloaded model without downloading one, and a test that needs a
//! multi-gigabyte file is a test that does not run in CI. So this writes a
//! genuine GGUF file, byte-for-byte in the real format, with a handful of
//! layers and deterministic weights.
//!
//! It is not a mock. The bytes go through the same header reader, the same
//! tensor directory, the same dequantiser and the same forward pass as a real
//! model does. Only the size is unusual.

use anyhow::Result;
use std::io::Write;
use std::path::Path;

use crate::dequant;

/// The shape of a model to generate.
#[derive(Debug, Clone)]
pub struct Recipe {
    pub architecture: String,
    pub block_count: u32,
    pub embedding_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub feed_forward_length: u32,
    pub vocab_size: u32,
    /// GGML type code the weight matrices are stored as.
    pub weight_kind: u32,
    /// Whether to emit `output.weight`, or leave the head tied to the
    /// embedding table.
    pub separate_output_head: bool,
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe {
            architecture: "llama".into(),
            block_count: 4,
            embedding_length: 32,
            head_count: 4,
            head_count_kv: 2,
            feed_forward_length: 64,
            vocab_size: 48,
            weight_kind: dequant::F32,
            separate_output_head: true,
        }
    }
}

/// A deterministic value stream, so two runs of a test build the same model.
///
/// Weights are small and centred on zero: a transformer with large random
/// weights saturates its own softmax, and every position then attends to one
/// token, which would let a wrong causal mask pass unnoticed.
fn weight_at(seed: u64, index: u64) -> f32 {
    let mut state = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(index.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    state ^= state >> 30;
    state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^= state >> 31;
    // A stable name-derived value in [-0.25, 0.25). The top 24 bits over 2^25
    // give [0, 0.5); the shift down centres it.
    //
    // The range is not cosmetic. Weights an order of magnitude larger saturate
    // the attention softmax into a one-hot distribution and then run the
    // residual stream up to infinity, and a model whose logits are all `inf`
    // compares bit-identical to itself however wrongly it was split. That is
    // why `logits_are_finite_and_not_all_the_same` exists.
    ((state >> 40) as f32 / 33_554_432.0) - 0.25
}

fn seed_of(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

struct Pending {
    name: String,
    dimensions: Vec<u64>,
    kind: u32,
    data: Vec<u8>,
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

/// The metadata block, which counts its own entries.
///
/// The header states how many key/value pairs follow, and a count that
/// disagrees with the bytes does not fail: the reader stops early and then
/// parses the remaining metadata as the tensor directory, which produces a
/// tensor named after several kilobytes of weights. Keeping the count with the
/// writes is the only way that stays impossible.
#[derive(Default)]
struct Metadata {
    bytes: Vec<u8>,
    entries: u64,
}

impl Metadata {
    fn u32(&mut self, key: &str, value: u32) {
        string(&mut self.bytes, key);
        self.bytes.extend_from_slice(&4u32.to_le_bytes()); // ValueType::U32
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self.entries += 1;
    }

    fn f32(&mut self, key: &str, value: f32) {
        string(&mut self.bytes, key);
        self.bytes.extend_from_slice(&6u32.to_le_bytes()); // ValueType::F32
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self.entries += 1;
    }

    fn string(&mut self, key: &str, value: &str) {
        string(&mut self.bytes, key);
        self.bytes.extend_from_slice(&8u32.to_le_bytes()); // ValueType::String
        string(&mut self.bytes, value);
        self.entries += 1;
    }
}

impl Recipe {
    fn tensor(&self, name: &str, dimensions: Vec<u64>) -> Result<Pending> {
        let count: u64 = dimensions.iter().product();
        let seed = seed_of(name);
        let values: Vec<f32> = (0..count).map(|i| weight_at(seed, i)).collect();
        // Norm weights multiply rather than add, so they sit near one; every
        // other tensor is a projection and sits near zero.
        let values = if name.ends_with("norm.weight") {
            values.iter().map(|v| 1.0 + v).collect()
        } else {
            values
        };
        let mut data = Vec::new();
        dequant::quantize(self.weight_kind_for(name), &values, &mut data)?;
        Ok(Pending {
            name: name.to_string(),
            dimensions,
            kind: self.weight_kind_for(name),
            data,
        })
    }

    /// Norms are always `F32`, as they are in every real conversion: they are
    /// one value per channel and quantising them saves nothing.
    fn weight_kind_for(&self, name: &str) -> u32 {
        if name.ends_with("norm.weight") {
            dequant::F32
        } else {
            self.weight_kind
        }
    }

    /// Write a complete GGUF file at `path`.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let embed = u64::from(self.embedding_length);
        let ffn = u64::from(self.feed_forward_length);
        let head_dim = embed / u64::from(self.head_count);
        let kv = u64::from(self.head_count_kv) * head_dim;

        let mut tensors =
            vec![self.tensor("token_embd.weight", vec![embed, u64::from(self.vocab_size)])?];
        for block in 0..self.block_count {
            for (suffix, dimensions) in [
                ("attn_norm.weight", vec![embed]),
                ("attn_q.weight", vec![embed, embed]),
                ("attn_k.weight", vec![embed, kv]),
                ("attn_v.weight", vec![embed, kv]),
                ("attn_output.weight", vec![embed, embed]),
                ("ffn_norm.weight", vec![embed]),
                ("ffn_gate.weight", vec![embed, ffn]),
                ("ffn_up.weight", vec![embed, ffn]),
                ("ffn_down.weight", vec![ffn, embed]),
            ] {
                tensors.push(self.tensor(&format!("blk.{block}.{suffix}"), dimensions)?);
            }
        }
        tensors.push(self.tensor("output_norm.weight", vec![embed])?);
        if self.separate_output_head {
            tensors.push(self.tensor("output.weight", vec![embed, u64::from(self.vocab_size)])?);
        }

        let mut metadata = Metadata::default();
        let arch = &self.architecture;
        metadata.string("general.architecture", arch);
        metadata.string("general.name", "hocmesh-fixture");
        metadata.u32(&format!("{arch}.block_count"), self.block_count);
        metadata.u32(&format!("{arch}.embedding_length"), self.embedding_length);
        metadata.u32(&format!("{arch}.attention.head_count"), self.head_count);
        metadata.u32(
            &format!("{arch}.attention.head_count_kv"),
            self.head_count_kv,
        );
        metadata.u32(
            &format!("{arch}.feed_forward_length"),
            self.feed_forward_length,
        );
        metadata.u32(&format!("{arch}.context_length"), 256);
        metadata.f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5);
        metadata.f32(&format!("{arch}.rope.freq_base"), 10_000.0);

        // Offsets are relative to the start of the tensor data section and
        // each tensor is aligned, exactly as a real converter emits them.
        let alignment = hocmesh_model::gguf::DEFAULT_ALIGNMENT;
        let mut directory = Vec::new();
        let mut offset = 0u64;
        for tensor in &tensors {
            string(&mut directory, &tensor.name);
            directory.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
            for extent in &tensor.dimensions {
                directory.extend_from_slice(&extent.to_le_bytes());
            }
            directory.extend_from_slice(&tensor.kind.to_le_bytes());
            directory.extend_from_slice(&offset.to_le_bytes());
            offset += tensor.data.len() as u64;
            offset = offset.next_multiple_of(alignment);
        }

        let mut out = Vec::new();
        out.extend_from_slice(&hocmesh_model::gguf::MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&metadata.entries.to_le_bytes());
        out.extend_from_slice(&metadata.bytes);
        out.extend_from_slice(&directory);
        while !(out.len() as u64).is_multiple_of(alignment) {
            out.push(0);
        }
        for tensor in &tensors {
            out.extend_from_slice(&tensor.data);
            while !(out.len() as u64).is_multiple_of(alignment) {
                out.push(0);
            }
        }

        let mut file = std::fs::File::create(path)?;
        file.write_all(&out)?;
        file.sync_all()?;
        Ok(())
    }
}
