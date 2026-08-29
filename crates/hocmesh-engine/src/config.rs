//! The handful of numbers that decide what a forward pass does.
//!
//! Read from the file rather than configured, for the same reason
//! `hocmesh-model` reads the architecture rather than trusting an operator's
//! `--architecture` flag: a hyperparameter typed by hand is a field nobody can
//! check, and getting one wrong here does not fail, it generates nonsense.

use anyhow::{Context, Result, bail, ensure};
use hocmesh_model::gguf;

/// Which pairs of elements a rotary embedding rotates together.
///
/// Not in the file. llama.cpp fixes it per architecture, so it is derived from
/// the architecture the file declares and an architecture this build has not
/// been told about is refused. Guessing costs nothing at load time and
/// produces a model that generates fluent, wrong text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Adjacent pairs `(0,1), (2,3), ...` — llama, mistral, and the families
    /// converted from them.
    Interleaved,
    /// Halves `(i, i + d/2)` — the GPT-NeoX layout, used by qwen2, phi and the
    /// stablelm line.
    Halved,
}

/// The architectures this build knows the rotary pairing for.
///
/// Read this as an answer to one question, not as a list of supported models.
/// Which pairs of elements a rotary embedding rotates together is the single
/// fact a GGUF file does not carry -- llama.cpp fixes it per architecture in
/// its own source -- so it is the single fact a name has to supply.
///
/// It is deliberately no longer the gate. A name here is not sufficient: the
/// list cannot see inside the file, so it cannot tell a Qwen3 this build
/// computes correctly from one carrying a tensor this build would ignore, and
/// `Stage::load` enumerates what the file actually holds and refuses anything
/// it would not consume. It is not necessary either: the shape below --
/// `attn_norm -> q,k,v -> rope -> attention -> out, ffn_norm -> gate*up ->
/// down`, RMS norm and SwiGLU, with optional per-head query/key norms and
/// projection biases as additional terms rather than different ones -- is what
/// most published models are, under names that did not exist when this list was
/// last edited. [`ASSUME_ARCHITECTURE`] lets an operator point one of those at
/// the entry that answers its rotary question, and every other check still
/// applies to it.
///
/// `stablelm` was on this list and should not have been: llama.cpp's stablelm
/// normalises with LayerNorm, which subtracts a mean and adds a bias that
/// `rms_norm` does neither of. It did not fail, it generated fluent, wrong
/// text -- the exact outcome the doc comment above `RopeStyle` promises to
/// avoid. Putting it back means implementing LayerNorm first.
///
/// That is also the honest limit of [`ASSUME_ARCHITECTURE`]. A LayerNorm model
/// that carries no norm bias, or one that puts GELU where this build puts
/// SiLU, has the same tensors under the same names, so nothing in the file
/// contradicts the operator who names it as llama. Everything a file can be
/// asked is asked; the two axes it has no way to answer are why the override
/// is opt-in and says so.
const KNOWN: &[(&str, RopeStyle)] = &[
    ("llama", RopeStyle::Interleaved),
    ("mistral", RopeStyle::Interleaved),
    ("qwen2", RopeStyle::Halved),
    ("qwen3", RopeStyle::Halved),
];

/// Read a file whose declared architecture this build does not know by name.
///
/// Set it to a name on [`KNOWN`] -- `HOCMESH_ASSUME_ARCHITECTURE=llama` -- to
/// say "this file is that shape, so use that rotary pairing".
///
/// Narrower than it looks, because the declared name is not what decides
/// whether a file can be run. Everything that is actually *in* the file is
/// checked against what this engine implements whatever the header claims:
/// every tensor of every block is enumerated and one this build would not read
/// is refused, every tensor that is read must have the shape the configuration
/// implies, and every metadata key that would change the maths is refused when
/// it is set. A file that is not this shape stays refused with this variable
/// set, on the evidence rather than on the name.
///
/// What the variable supplies is the one fact that is genuinely not in a GGUF
/// file: which pairs of elements a rotary embedding rotates together.
/// llama.cpp fixes that per architecture in its own source, so a name this
/// build has not been told about carries no answer to it. Getting it wrong
/// does not fail -- it generates fluent, wrong text -- which is why the
/// default is to refuse rather than to guess, and why supplying it is a
/// deliberate act with an operator behind it.
pub const ASSUME_ARCHITECTURE: &str = "HOCMESH_ASSUME_ARCHITECTURE";

/// Which rotary pairing to use for a file declaring `architecture`.
fn rope_style_for(architecture: &str) -> Result<RopeStyle> {
    let lookup = |name: &str| {
        KNOWN
            .iter()
            .find(|(known, _)| *known == name)
            .map(|(_, style)| *style)
    };
    if let Some(style) = lookup(architecture) {
        return Ok(style);
    }
    let known = KNOWN
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ");
    let assumed = std::env::var(ASSUME_ARCHITECTURE)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let Some(assumed) = assumed else {
        bail!(
            "architecture {architecture:?} is not one this build knows the rotary pairing \
             for (it knows {known}). Everything else about this file is checked against \
             what this engine implements -- every tensor of every block, their shapes, and \
             the metadata keys that would change the maths -- but which pairs of elements a \
             rotary embedding rotates together is not written in a GGUF file at all, and \
             llama.cpp fixes it per architecture in its own source. If this model has one \
             of those shapes, say which with {ASSUME_ARCHITECTURE}=<name>; the rest of the \
             file will still have to prove itself. Guessing is not offered because a wrong \
             pairing does not fail, it generates fluent, wrong text."
        );
    };
    let Some(style) = lookup(&assumed) else {
        bail!(
            "{ASSUME_ARCHITECTURE} is set to {assumed:?}, which is not an architecture this \
             build knows either (it knows {known}). It names the shape to read \
             {architecture:?} as, not the shape the file already claims to be."
        );
    };
    Ok(style)
}

/// Everything the forward pass needs, and nothing else.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: String,
    pub rope_style: RopeStyle,
    /// Transformer blocks in the whole model, not in this stage.
    pub block_count: u32,
    pub embedding_length: u32,
    pub head_count: u32,
    /// Key/value heads. Fewer than `head_count` is grouped-query attention;
    /// equal is ordinary multi-head.
    pub head_count_kv: u32,
    pub feed_forward_length: u32,
    /// Width of one attention head's query and key.
    ///
    /// Read from `attention.key_length`, which most files omit because it is
    /// `embedding_length / head_count`. The families that write it -- Qwen3 at
    /// four of its six sizes, every Gemma -- are exactly the ones where that
    /// division gives the wrong answer, and deriving it there produces a
    /// tensor-shape complaint that blames the file for a key this engine did
    /// not read.
    pub key_length: u32,
    /// Width of one attention head's value. Usually equal to `key_length`;
    /// `attention.value_length` may say otherwise.
    pub value_length: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// How many of each head's dimensions are rotated. The rest are carried
    /// through unrotated, which some conversions rely on.
    pub rope_dimension_count: u32,
    pub context_length: u32,
    pub vocab_size: u32,
}

impl ModelConfig {
    /// Width of one attention head's query and key.
    #[must_use]
    pub fn head_dim(&self) -> u32 {
        self.key_length
    }

    /// How many query heads share each key/value head.
    #[must_use]
    pub fn group_size(&self) -> u32 {
        self.head_count / self.head_count_kv.max(1)
    }

    /// Width of the concatenated key vector for one position.
    #[must_use]
    pub fn k_width(&self) -> u32 {
        self.head_count_kv * self.key_length
    }

    /// Width of the concatenated value vector for one position.
    #[must_use]
    pub fn v_width(&self) -> u32 {
        self.head_count_kv * self.value_length
    }

    /// Width of the attention output before the output projection: every
    /// query head contributes one value.
    #[must_use]
    pub fn attention_width(&self) -> u32 {
        self.head_count * self.value_length
    }

    /// Width of the concatenated query vector for one position.
    #[must_use]
    pub fn q_width(&self) -> u32 {
        self.head_count * self.key_length
    }

    /// Read the configuration out of a GGUF header.
    ///
    /// `bytes` must cover the whole key/value block; the caller normally has
    /// the head of the file. A key that is present but unreadable is an error,
    /// and a key that is absent is either an error or a documented default —
    /// never a silent zero.
    pub fn from_header(bytes: &[u8]) -> Result<Self> {
        let architecture = gguf::architecture(bytes)?
            .context("GGUF file does not declare general.architecture")?;
        let rope_style = rope_style_for(&architecture)?;

        let required = |key: &str| -> Result<u64> {
            gguf::metadata_u64(bytes, &format!("{architecture}.{key}"))?
                .with_context(|| format!("GGUF file does not declare {architecture}.{key}"))
        };
        let block_count = required("block_count")? as u32;
        let embedding_length = required("embedding_length")? as u32;
        let head_count = required("attention.head_count")? as u32;
        let feed_forward_length = required("feed_forward_length")? as u32;
        // Absent means "as many key/value heads as query heads", which is what
        // every pre-GQA conversion means by leaving it out.
        let head_count_kv =
            gguf::metadata_u64(bytes, &format!("{architecture}.attention.head_count_kv"))?
                .unwrap_or(u64::from(head_count)) as u32;

        ensure!(head_count > 0, "attention.head_count is zero");
        ensure!(head_count_kv > 0, "attention.head_count_kv is zero");
        ensure!(
            head_count.is_multiple_of(head_count_kv),
            "{head_count} query heads do not group evenly over {head_count_kv} key/value heads"
        );

        let optional = |key: &str| -> Result<Option<u64>> {
            gguf::metadata_u64(bytes, &format!("{architecture}.{key}"))
        };
        // A file that states how wide a head is gets believed over the
        // division, because the two disagree in exactly the models where the
        // division is wrong. Only when the key is absent does
        // `embedding_length / head_count` have to come out whole -- demanding
        // that unconditionally refuses a model that told us the answer, and
        // the message blames the file for a key this engine never read.
        let key_length = match optional("attention.key_length")? {
            Some(width) => width as u32,
            None => {
                ensure!(
                    embedding_length.is_multiple_of(head_count),
                    "embedding_length {embedding_length} does not divide into {head_count} \
                     heads, and the file does not declare \
                     {architecture}.attention.key_length to say how wide one is"
                );
                embedding_length / head_count
            }
        };
        let value_length =
            optional("attention.value_length")?.unwrap_or(u64::from(key_length)) as u32;
        ensure!(key_length > 0, "attention.key_length is zero");
        ensure!(value_length > 0, "attention.value_length is zero");

        let rope_dimension_count =
            optional("rope.dimension_count")?.unwrap_or(u64::from(key_length)) as u32;
        ensure!(
            rope_dimension_count <= key_length && rope_dimension_count.is_multiple_of(2),
            "rope.dimension_count {rope_dimension_count} is not an even count within a \
             {key_length}-wide head"
        );

        Self::refuse_unimplemented_metadata(bytes, &architecture)?;

        Ok(ModelConfig {
            rope_style,
            block_count,
            embedding_length,
            head_count,
            head_count_kv,
            feed_forward_length,
            key_length,
            value_length,
            rope_dimension_count,
            // 1e-5 is the value llama.cpp uses when a file omits it.
            rms_norm_eps: gguf::metadata_f32(
                bytes,
                &format!("{architecture}.attention.layer_norm_rms_epsilon"),
            )?
            .unwrap_or(1e-5),
            // 10000 is the original RoPE base and the value every file that
            // omits the key was trained with.
            rope_theta: gguf::metadata_f32(bytes, &format!("{architecture}.rope.freq_base"))?
                .unwrap_or(10_000.0),
            context_length: gguf::metadata_u64(bytes, &format!("{architecture}.context_length"))?
                .unwrap_or(2048) as u32,
            // The vocabulary is the embedding table's own second dimension,
            // and the metadata key is optional. The caller fills it in from the
            // tensor shape when it loads one; zero here means "not yet known".
            vocab_size: gguf::metadata_u64(bytes, &format!("{architecture}.vocab_size"))?
                .unwrap_or(0) as u32,
            architecture,
        })
    }

    /// Refuse a file whose header asks for arithmetic this build does not do.
    ///
    /// Every key here changes the forward pass rather than describing it, and
    /// every one of them is silently ignorable: a model with a sliding window
    /// attends to its whole history instead of the last few thousand tokens
    /// and still produces fluent text, a model with a rope scaling factor
    /// places every position wrong and still produces fluent text. There is no
    /// output to inspect that says which happened. So the check is at load,
    /// where it costs one metadata read and can name what is missing.
    ///
    /// This list grows the other way from most: an entry is *removed* when the
    /// feature is implemented.
    fn refuse_unimplemented_metadata(bytes: &[u8], architecture: &str) -> Result<()> {
        const UNIMPLEMENTED_NUMERIC: &[(&str, &str)] = &[
            (
                "attention.sliding_window",
                "attends over a window rather than the whole history",
            ),
            ("expert_count", "routes each token to a subset of experts"),
            (
                "attention.attn_logit_softcapping",
                "bounds its attention scores with tanh",
            ),
            (
                "final_logit_softcapping",
                "bounds its output logits with tanh",
            ),
            (
                "attention.max_alibi_bias",
                "biases attention by distance instead of rotating positions",
            ),
        ];
        for (key, what) in UNIMPLEMENTED_NUMERIC {
            if let Some(value) = gguf::metadata_u64(bytes, &format!("{architecture}.{key}"))?
                && value != 0
            {
                bail!(
                    "this model {what} ({architecture}.{key} = {value}), which this engine \
                     does not implement; running it would ignore that and generate fluent, \
                     wrong text"
                );
            }
        }
        // Scaling is expressed either as a type string or, in older
        // conversions, as a bare linear factor. Both are refused; "none" and a
        // factor of 1 are the no-ops that mean the file wants nothing done.
        if let Some(kind) =
            gguf::metadata_string(bytes, &format!("{architecture}.rope.scaling.type"))?
            && kind != "none"
        {
            bail!(
                "this model scales rotary positions ({architecture}.rope.scaling.type = \
                 {kind:?}), which this engine does not implement; running it would place \
                 every position wrong and generate fluent, wrong text"
            );
        }
        if let Some(factor) =
            gguf::metadata_f32(bytes, &format!("{architecture}.rope.scale_linear"))?
            && (factor - 1.0).abs() > f32::EPSILON
        {
            bail!(
                "this model scales rotary positions ({architecture}.rope.scale_linear = \
                 {factor}), which this engine does not implement"
            );
        }
        Ok(())
    }

    /// Check that a stage's layer range is one this model has.
    pub fn validate_range(&self, blocks: &std::ops::Range<u32>) -> Result<()> {
        ensure!(
            blocks.start < blocks.end,
            "a stage holding no blocks ({}..{}) is a network hop that computes nothing",
            blocks.start,
            blocks.end
        );
        ensure!(
            blocks.end <= self.block_count,
            "blocks {}..{} run past the model's {} blocks",
            blocks.start,
            blocks.end,
            self.block_count
        );
        Ok(())
    }
}
