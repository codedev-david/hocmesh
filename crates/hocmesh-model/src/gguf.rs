//! Reading the few facts hocMESH needs out of a GGUF header.
//!
//! A model manifest records an architecture, and until now the operator had to
//! type it. That is a field nobody can check: `--architecture llama` on a Qwen
//! file is accepted, stored, and then used for cache-locality scoring, so the
//! mistake surfaces as slightly worse placement months later and is never
//! traced back. The file already states its own architecture, so ask the file.
//!
//! This is a reader, not a parser: it walks the key/value block far enough to
//! answer a question and stops. It never allocates from a length it has not
//! bounds-checked, never seeks past the buffer it was given, and returns
//! `Ok(None)` rather than an error when the answer simply is not in the bytes
//! available -- the caller holds the first chunk of a file that may be
//! gigabytes long, and "the header is longer than one chunk" is a normal
//! outcome, not a corrupt file.
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use anyhow::{Context, Result, bail, ensure};

/// `GGUF` little-endian.
pub const MAGIC: [u8; 4] = *b"GGUF";

/// Versions whose key/value encoding this reader understands.
///
/// Version 1 used 32-bit lengths and is not accepted, rather than being read
/// with 64-bit assumptions and silently producing nonsense.
const SUPPORTED_VERSIONS: std::ops::RangeInclusive<u32> = 2..=3;

/// Refuse to size a buffer from a length field larger than this. No real key or
/// string value approaches it, and it stops a corrupt length from turning into
/// an allocation.
const MAX_STRING_LEN: u64 = 1 << 20;

/// A hard stop on how many key/value pairs are walked, so a corrupt count
/// cannot spin.
const MAX_KV_PAIRS: u64 = 1 << 20;

/// The value types GGUF defines. Only the widths matter here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_code(code: u32) -> Result<Self> {
        Ok(match code {
            0 => ValueType::U8,
            1 => ValueType::I8,
            2 => ValueType::U16,
            3 => ValueType::I16,
            4 => ValueType::U32,
            5 => ValueType::I32,
            6 => ValueType::F32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::U64,
            11 => ValueType::I64,
            12 => ValueType::F64,
            other => bail!("unknown GGUF value type {other}"),
        })
    }

    /// Fixed width in bytes, or `None` for the variable-length types.
    fn fixed_width(self) -> Option<usize> {
        Some(match self {
            ValueType::U8 | ValueType::I8 | ValueType::Bool => 1,
            ValueType::U16 | ValueType::I16 => 2,
            ValueType::U32 | ValueType::I32 | ValueType::F32 => 4,
            ValueType::U64 | ValueType::I64 | ValueType::F64 => 8,
            ValueType::String | ValueType::Array => return None,
        })
    }
}

/// A cursor that cannot read past the end of what it was handed.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

/// Signals that the answer is not in the bytes available, which is different
/// from the bytes being wrong.
struct Truncated;

type Partial<T> = std::result::Result<T, Truncated>;

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, at: 0 }
    }

    fn take(&mut self, count: usize) -> Partial<&'a [u8]> {
        let end = self.at.checked_add(count).ok_or(Truncated)?;
        let slice = self.bytes.get(self.at..end).ok_or(Truncated)?;
        self.at = end;
        Ok(slice)
    }

    fn skip(&mut self, count: u64) -> Partial<()> {
        let count = usize::try_from(count).map_err(|_| Truncated)?;
        self.take(count).map(|_| ())
    }

    fn u32(&mut self) -> Partial<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Partial<u64> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// A length-prefixed string. The length is bounds-checked against the
    /// buffer before anything is taken, so a corrupt length reads as truncated.
    fn string(&mut self) -> Partial<&'a [u8]> {
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            return Err(Truncated);
        }
        self.take(usize::try_from(len).map_err(|_| Truncated)?)
    }
}

/// The value of a string-typed metadata key, if the header holds one.
///
/// `Ok(None)` means "not present in these bytes" -- either genuinely absent, or
/// beyond the end of the slice. An `Err` means the bytes are not a GGUF header
/// at all, which is worth telling the operator about.
pub fn metadata_string(bytes: &[u8], key: &str) -> Result<Option<String>> {
    let mut reader = Reader::new(bytes);

    let Ok(magic) = reader.take(4) else {
        bail!("file is too short to be a GGUF model");
    };
    ensure!(magic == MAGIC, "file does not start with the GGUF magic");

    let Ok(version) = reader.u32() else {
        bail!("truncated GGUF header: no version");
    };
    ensure!(
        SUPPORTED_VERSIONS.contains(&version),
        "GGUF version {version} is not supported (this build reads versions {}-{})",
        SUPPORTED_VERSIONS.start(),
        SUPPORTED_VERSIONS.end()
    );

    // Tensor count, then the number of key/value pairs.
    let Ok(_tensors) = reader.u64() else {
        return Ok(None);
    };
    let Ok(pairs) = reader.u64() else {
        return Ok(None);
    };
    if pairs > MAX_KV_PAIRS {
        bail!("GGUF header claims {pairs} metadata entries, which is not plausible");
    }

    for _ in 0..pairs {
        let Ok(found_key) = reader.string() else {
            return Ok(None);
        };
        let Ok(code) = reader.u32() else {
            return Ok(None);
        };
        let value_type = ValueType::from_code(code)?;
        let wanted = found_key == key.as_bytes();

        if wanted && value_type == ValueType::String {
            let Ok(value) = reader.string() else {
                return Ok(None);
            };
            return Ok(Some(String::from_utf8_lossy(value).into_owned()));
        }
        if wanted {
            // Present, but not a string. Reporting "absent" would be a lie and
            // guessing at a conversion would be worse.
            bail!("GGUF key {key} is not a string");
        }
        if skip_value(&mut reader, value_type).is_err() {
            return Ok(None);
        }
    }

    Ok(None)
}

/// A fixed-width metadata value, read without being converted.
///
/// Kept as the type the file declared so the caller decides what a narrowing
/// conversion means. A hyperparameter written as `i32` and read as `u32` is
/// fine at 4096 and catastrophic at -1, and the reader is not the place to
/// decide which one a given key is allowed to be.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Scalar {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
}

/// Read a fixed-width value of the given type. Strings and arrays are not
/// scalars and are refused rather than coerced.
fn read_scalar(reader: &mut Reader<'_>, value_type: ValueType) -> Partial<Option<Scalar>> {
    let width = match value_type.fixed_width() {
        Some(width) => width,
        None => return Ok(None),
    };
    let bytes = reader.take(width)?;
    let mut eight = [0u8; 8];
    eight[..width].copy_from_slice(bytes);
    Ok(Some(match value_type {
        ValueType::U8 | ValueType::U16 | ValueType::U32 | ValueType::U64 => {
            Scalar::Unsigned(u64::from_le_bytes(eight))
        }
        ValueType::Bool => Scalar::Bool(bytes[0] != 0),
        ValueType::I8 => Scalar::Signed(i64::from(bytes[0] as i8)),
        ValueType::I16 => Scalar::Signed(i64::from(i16::from_le_bytes([bytes[0], bytes[1]]))),
        ValueType::I32 => Scalar::Signed(i64::from(i32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]))),
        ValueType::I64 => Scalar::Signed(i64::from_le_bytes(eight)),
        ValueType::F32 => Scalar::Float(f64::from(f32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]))),
        ValueType::F64 => Scalar::Float(f64::from_le_bytes(eight)),
        ValueType::String | ValueType::Array => return Ok(None),
    }))
}

/// Walk the key/value block and hand every scalar to `wanted`, stopping at the
/// first one it accepts.
///
/// Shared by every typed accessor so there is one implementation of the walk
/// and one place where truncation is distinguished from absence.
fn find_scalar(
    bytes: &[u8],
    key: &str,
    mut wanted: impl FnMut(Scalar) -> Option<Scalar>,
) -> Result<Option<Scalar>> {
    let mut reader = Reader::new(bytes);

    let Ok(magic) = reader.take(4) else {
        bail!("file is too short to be a GGUF model");
    };
    ensure!(magic == MAGIC, "file does not start with the GGUF magic");
    let Ok(version) = reader.u32() else {
        bail!("truncated GGUF header: no version");
    };
    ensure!(
        SUPPORTED_VERSIONS.contains(&version),
        "GGUF version {version} is not supported (this build reads versions {}-{})",
        SUPPORTED_VERSIONS.start(),
        SUPPORTED_VERSIONS.end()
    );
    let (Ok(_tensors), Ok(pairs)) = (reader.u64(), reader.u64()) else {
        return Ok(None);
    };
    if pairs > MAX_KV_PAIRS {
        bail!("GGUF header claims {pairs} metadata entries, which is not plausible");
    }

    for _ in 0..pairs {
        let (Ok(found_key), Ok(code)) = (reader.string(), reader.u32()) else {
            return Ok(None);
        };
        let value_type = ValueType::from_code(code)?;
        if found_key != key.as_bytes() {
            if skip_value(&mut reader, value_type).is_err() {
                return Ok(None);
            }
            continue;
        }
        let Ok(Some(scalar)) = read_scalar(&mut reader, value_type) else {
            // Present and not a scalar. Reporting "absent" would be a lie and
            // guessing at a conversion would be worse.
            bail!("GGUF key {key} is not a number");
        };
        return match wanted(scalar) {
            Some(value) => Ok(Some(value)),
            None => bail!("GGUF key {key} holds {scalar:?}, which does not fit"),
        };
    }
    Ok(None)
}

/// A metadata value read as a count.
///
/// Every hyperparameter this reader wants -- head counts, embedding widths,
/// layer counts -- is a count, and converters disagree about whether to write
/// one as `u32` or `i32`. Both are accepted; a negative value is refused rather
/// than wrapped, because a wrapped count would size an allocation.
pub fn metadata_u64(bytes: &[u8], key: &str) -> Result<Option<u64>> {
    Ok(find_scalar(bytes, key, |scalar| match scalar {
        Scalar::Unsigned(value) => Some(Scalar::Unsigned(value)),
        Scalar::Signed(value) => u64::try_from(value).ok().map(Scalar::Unsigned),
        Scalar::Bool(value) => Some(Scalar::Unsigned(u64::from(value))),
        Scalar::Float(_) => None,
    })?
    .and_then(|scalar| match scalar {
        Scalar::Unsigned(value) => Some(value),
        _ => None,
    }))
}

/// A metadata value read as a real number, widening an integer where a file
/// wrote one (`rope.freq_base` is `f32` in most files and `u32` in a few).
pub fn metadata_f32(bytes: &[u8], key: &str) -> Result<Option<f32>> {
    Ok(find_scalar(bytes, key, |scalar| match scalar {
        Scalar::Float(value) => Some(Scalar::Float(value)),
        Scalar::Unsigned(value) => Some(Scalar::Float(value as f64)),
        Scalar::Signed(value) => Some(Scalar::Float(value as f64)),
        Scalar::Bool(_) => None,
    })?
    .and_then(|scalar| match scalar {
        Scalar::Float(value) => Some(value as f32),
        _ => None,
    }))
}

/// Step over a value of known type without interpreting it.
fn skip_value(reader: &mut Reader<'_>, value_type: ValueType) -> Partial<()> {
    match value_type {
        ValueType::String => reader.string().map(|_| ()),
        ValueType::Array => {
            let element = ValueType::from_code(reader.u32()?).map_err(|_| Truncated)?;
            let count = reader.u64()?;
            match element.fixed_width() {
                // Multiplied as u64 so a hostile count cannot wrap a usize.
                Some(width) => reader.skip(count.saturating_mul(width as u64)),
                None => {
                    // Strings and nested arrays have to be walked one by one.
                    for _ in 0..count {
                        skip_value(reader, element)?;
                    }
                    Ok(())
                }
            }
        }
        fixed => reader.skip(fixed.fixed_width().unwrap_or(0) as u64),
    }
}

/// The architecture a GGUF file declares, e.g. `llama`, `qwen2`, `phi3`.
pub fn architecture(bytes: &[u8]) -> Result<Option<String>> {
    metadata_string(bytes, "general.architecture")
}

/// The name a GGUF file gives itself, when it gives one.
pub fn model_name(bytes: &[u8]) -> Result<Option<String>> {
    metadata_string(bytes, "general.name")
}

/// The alignment GGUF uses for tensor data when a file declares none.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// The metadata key a file uses to override [`DEFAULT_ALIGNMENT`].
const ALIGNMENT_KEY: &str = "general.alignment";

/// An implausible tensor count is refused rather than allocated for.
const MAX_TENSORS: u64 = 1 << 20;

/// GGUF tensors have at most four dimensions. A little slack is allowed so a
/// future file reads as unsupported rather than as corrupt.
const MAX_DIMENSIONS: u32 = 8;

/// How a GGML type packs elements into bytes.
///
/// Quantised types store a whole block of elements together with shared scales,
/// so a byte length cannot be derived from an element count alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    /// Elements in one stored block.
    pub elements: u64,
    /// Bytes one stored block occupies.
    pub bytes: u64,
}

/// The block layout of a GGML type code, or `None` for a code this build does
/// not know.
///
/// `None` is a deliberate answer rather than a fallback. A wrong size here would
/// silently mis-slice a model and produce activations that are wrong without
/// being detectably wrong, so not knowing is reported instead of guessed.
#[must_use]
pub fn block_layout(kind: u32) -> Option<BlockLayout> {
    let (elements, bytes) = match kind {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        // 4 and 5 were Q4_2 and Q4_3 and no longer exist; 31-33 were the
        // repacked Q4_0 variants and were withdrawn. Anything else is newer
        // than this build.
        _ => return None,
    };
    Some(BlockLayout { elements, bytes })
}

/// One tensor, as the GGUF tensor directory describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// e.g. `blk.12.attn_q.weight`, or `token_embd.weight` for a shared tensor.
    pub name: String,
    /// Extents in GGML order, fastest-varying first.
    pub dimensions: Vec<u64>,
    /// The GGML type code. See [`block_layout`].
    pub kind: u32,
    /// Offset from the start of the tensor data section, **not** from the start
    /// of the file. Add [`TensorDirectory::data_start`] for a file offset.
    pub offset: u64,
}

impl TensorInfo {
    /// The number of elements, or `None` if the declared shape overflows.
    #[must_use]
    pub fn element_count(&self) -> Option<u64> {
        self.dimensions
            .iter()
            .try_fold(1u64, |total, extent| total.checked_mul(*extent))
    }

    /// The bytes this tensor occupies, or `None` when the type code is one this
    /// build does not know.
    #[must_use]
    pub fn data_len(&self) -> Option<u64> {
        let layout = block_layout(self.kind)?;
        let elements = self.element_count()?;
        let blocks = elements.div_ceil(layout.elements.max(1));
        blocks.checked_mul(layout.bytes)
    }

    /// The transformer block this tensor belongs to, for the `blk.N.` names
    /// every GGUF converter emits. Embeddings, the output head and the final
    /// norm belong to no block and answer `None`.
    #[must_use]
    pub fn layer_index(&self) -> Option<u32> {
        let rest = self.name.strip_prefix("blk.")?;
        let (index, _) = rest.split_once('.')?;
        index.parse().ok()
    }
}

/// A half-open span of a file, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteExtent {
    pub start: u64,
    pub end: u64,
}

impl ByteExtent {
    #[must_use]
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Every tensor a GGUF file holds, and where its data begins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDirectory {
    /// The alignment the file declares, or [`DEFAULT_ALIGNMENT`].
    pub alignment: u64,
    /// File offset at which tensor data starts. Tensor offsets are relative to
    /// this, so a file offset is `data_start + tensor.offset`.
    pub data_start: u64,
    /// In the order the directory lists them, which is not necessarily offset
    /// order.
    pub tensors: Vec<TensorInfo>,
}

/// Round `value` up to the next multiple of `alignment`.
fn align_up(value: u64, alignment: u64) -> Option<u64> {
    if alignment <= 1 {
        return Some(value);
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

/// Read the tensor directory: name, type, shape and offset for every tensor.
///
/// The header must be complete in `bytes`; the tensor *data* need not be
/// present, which is the point -- a peer can learn the layout of a model from
/// its first few hundred kilobytes and then fetch only the parts it will run.
///
/// `Ok(None)` means the bytes stop before the directory ends, exactly as in
/// [`metadata_string`]. An `Err` means the bytes are not a GGUF header, or
/// describe something this build refuses to believe.
pub fn tensor_directory(bytes: &[u8]) -> Result<Option<TensorDirectory>> {
    let mut reader = Reader::new(bytes);

    let Ok(magic) = reader.take(4) else {
        bail!("file is too short to be a GGUF model");
    };
    ensure!(magic == MAGIC, "file does not start with the GGUF magic");

    let Ok(version) = reader.u32() else {
        bail!("truncated GGUF header: no version");
    };
    ensure!(
        SUPPORTED_VERSIONS.contains(&version),
        "GGUF version {version} is not supported (this build reads versions {}-{})",
        SUPPORTED_VERSIONS.start(),
        SUPPORTED_VERSIONS.end()
    );

    let Ok(tensor_count) = reader.u64() else {
        return Ok(None);
    };
    if tensor_count > MAX_TENSORS {
        bail!("GGUF header claims {tensor_count} tensors, which is not plausible");
    }
    let Ok(pairs) = reader.u64() else {
        return Ok(None);
    };
    if pairs > MAX_KV_PAIRS {
        bail!("GGUF header claims {pairs} metadata entries, which is not plausible");
    }

    // One walk of the metadata, picking up the alignment on the way past. The
    // alignment sits behind the tokenizer vocabulary in most real files, so
    // reading it separately would mean walking the whole block twice.
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..pairs {
        let Ok(key) = reader.string() else {
            return Ok(None);
        };
        let Ok(code) = reader.u32() else {
            return Ok(None);
        };
        let value_type = ValueType::from_code(code)?;
        if key == ALIGNMENT_KEY.as_bytes() && value_type == ValueType::U32 {
            let Ok(declared) = reader.u32() else {
                return Ok(None);
            };
            ensure!(
                declared != 0 && declared.is_power_of_two(),
                "GGUF declares an alignment of {declared}, which is not a power of two"
            );
            alignment = u64::from(declared);
            continue;
        }
        if skip_value(&mut reader, value_type).is_err() {
            return Ok(None);
        }
    }

    let mut tensors = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0).min(4_096));
    for _ in 0..tensor_count {
        let Ok(name) = reader.string() else {
            return Ok(None);
        };
        let name = String::from_utf8_lossy(name).into_owned();
        let Ok(dimension_count) = reader.u32() else {
            return Ok(None);
        };
        ensure!(
            dimension_count <= MAX_DIMENSIONS,
            "tensor {name} claims {dimension_count} dimensions; GGUF allows at most 4"
        );
        let mut dimensions = Vec::with_capacity(dimension_count as usize);
        for _ in 0..dimension_count {
            let Ok(extent) = reader.u64() else {
                return Ok(None);
            };
            dimensions.push(extent);
        }
        let Ok(kind) = reader.u32() else {
            return Ok(None);
        };
        let Ok(offset) = reader.u64() else {
            return Ok(None);
        };
        tensors.push(TensorInfo {
            name,
            dimensions,
            kind,
            offset,
        });
    }

    let data_start = align_up(reader.at as u64, alignment)
        .context("GGUF tensor data would start past the end of a 64-bit file")?;

    Ok(Some(TensorDirectory {
        alignment,
        data_start,
        tensors,
    }))
}

impl TensorDirectory {
    /// One past the highest block index any tensor names, which is the layer
    /// count a pipeline plan has to partition.
    #[must_use]
    pub fn layer_count(&self) -> u32 {
        self.tensors
            .iter()
            .filter_map(TensorInfo::layer_index)
            .max()
            .map_or(0, |highest| highest + 1)
    }

    /// The tensors belonging to blocks in `layers`.
    #[must_use]
    pub fn tensors_for_layers(&self, layers: std::ops::Range<u32>) -> Vec<&TensorInfo> {
        self.tensors
            .iter()
            .filter(|tensor| {
                tensor
                    .layer_index()
                    .is_some_and(|index| layers.contains(&index))
            })
            .collect()
    }

    /// The tensors belonging to no block -- embeddings, the output head, the
    /// final norm. The first and last stage of a pipeline need these; the
    /// stages in between do not.
    #[must_use]
    pub fn shared_tensors(&self) -> Vec<&TensorInfo> {
        self.tensors
            .iter()
            .filter(|tensor| tensor.layer_index().is_none())
            .collect()
    }

    /// File offset and length of one tensor's data.
    ///
    /// The length comes from the declared type and shape when the type is
    /// known. When it is not, it is derived from where the next tensor starts,
    /// which over-reads by at most the alignment padding and is never short.
    /// `file_len` bounds the last tensor.
    #[must_use]
    pub fn extent_of(&self, tensor: &TensorInfo, file_len: u64) -> ByteExtent {
        let start = self.data_start.saturating_add(tensor.offset);
        let end = match tensor.data_len() {
            Some(len) => start.saturating_add(len),
            None => self
                .tensors
                .iter()
                .map(|other| other.offset)
                .filter(|offset| *offset > tensor.offset)
                .min()
                .map_or(file_len, |next| self.data_start.saturating_add(next)),
        };
        ByteExtent {
            start,
            end: end.min(file_len),
        }
    }

    /// The byte spans a stage must hold to run `layers`, merged so that
    /// neighbouring tensors become one request rather than many.
    ///
    /// This does not include the shared tensors; add
    /// [`Self::extents_of`]`(&self.shared_tensors(), ..)` for the first and last
    /// stage of a pipeline.
    #[must_use]
    pub fn extents_for_layers(
        &self,
        layers: std::ops::Range<u32>,
        file_len: u64,
    ) -> Vec<ByteExtent> {
        self.extents_of(&self.tensors_for_layers(layers), file_len)
    }

    /// [`Self::extent_of`] over a set of tensors, sorted and merged.
    #[must_use]
    pub fn extents_of(&self, tensors: &[&TensorInfo], file_len: u64) -> Vec<ByteExtent> {
        let mut spans: Vec<ByteExtent> = tensors
            .iter()
            .map(|tensor| self.extent_of(tensor, file_len))
            .filter(|span| !span.is_empty())
            .collect();
        spans.sort_unstable();

        let mut merged: Vec<ByteExtent> = Vec::with_capacity(spans.len());
        for span in spans {
            match merged.last_mut() {
                // Adjacent counts as contiguous: the gap between two tensors is
                // alignment padding, and asking for it costs less than a second
                // round trip.
                Some(last)
                    if span.start <= align_up(last.end, self.alignment).unwrap_or(last.end) =>
                {
                    last.end = last.end.max(span.end);
                }
                _ => merged.push(span),
            }
        }
        merged
    }

    /// The indexes of the fixed-size chunks that cover `extents`, so a peer can
    /// fetch the layers it will run instead of the whole file.
    ///
    /// Returns an error only for a zero chunk size, which is a caller bug.
    pub fn chunks_for_extents(extents: &[ByteExtent], chunk_size: u64) -> Result<Vec<u64>> {
        ensure!(chunk_size > 0, "chunk size must be greater than zero");
        let mut chunks: Vec<u64> = extents
            .iter()
            .filter(|extent| !extent.is_empty())
            .flat_map(|extent| (extent.start / chunk_size)..=((extent.end - 1) / chunk_size))
            .collect();
        chunks.sort_unstable();
        chunks.dedup();
        Ok(chunks)
    }

    /// Check the directory against the file it came from.
    ///
    /// Verifies that every tensor lands inside the file and that no two tensors
    /// claim the same bytes. Both are cheap, and both catch a truncated or
    /// tampered download before a stage runs on nonsense.
    pub fn validate(&self, file_len: u64) -> Result<()> {
        ensure!(
            self.data_start <= file_len,
            "GGUF tensor data starts at {} but the file is only {file_len} bytes",
            self.data_start
        );

        let mut spans: Vec<(ByteExtent, &str)> = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            let start = self
                .data_start
                .checked_add(tensor.offset)
                .with_context(|| format!("tensor {} has an offset that overflows", tensor.name))?;
            let len = tensor.data_len();
            let end = match len {
                Some(len) => start.checked_add(len).with_context(|| {
                    format!("tensor {} has a length that overflows", tensor.name)
                })?,
                None => start,
            };
            ensure!(
                end <= file_len,
                "tensor {} runs to byte {end} but the file is only {file_len} bytes",
                tensor.name
            );
            if len.is_some() {
                spans.push((ByteExtent { start, end }, tensor.name.as_str()));
            }
        }

        spans.sort_unstable_by_key(|(span, _)| *span);
        for pair in spans.windows(2) {
            let (first, first_name) = &pair[0];
            let (second, second_name) = &pair[1];
            ensure!(
                first.end <= second.start,
                "tensors {first_name} and {second_name} overlap in the file"
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builder for header bytes, so the tests state what they are testing
    /// rather than carrying a wall of hex.
    /// A tensor the builder lays out for real: offsets come from the declared
    /// shapes, so a test exercises the same arithmetic a converted file would.
    struct PlannedTensor {
        name: String,
        dimensions: Vec<u64>,
        kind: u32,
    }

    #[derive(Default)]
    struct Header {
        pairs: Vec<u8>,
        count: u64,
        tensors: Vec<PlannedTensor>,
        declared_alignment: Option<u32>,
    }

    impl Header {
        fn string_value(mut self, key: &str, value: &str) -> Self {
            self.key(key);
            self.pairs.extend_from_slice(&8u32.to_le_bytes());
            self.pairs
                .extend_from_slice(&(value.len() as u64).to_le_bytes());
            self.pairs.extend_from_slice(value.as_bytes());
            self.count += 1;
            self
        }

        fn u32_value(mut self, key: &str, value: u32) -> Self {
            self.key(key);
            self.pairs.extend_from_slice(&4u32.to_le_bytes());
            self.pairs.extend_from_slice(&value.to_le_bytes());
            self.count += 1;
            self
        }

        fn u32_array(mut self, key: &str, values: &[u32]) -> Self {
            self.key(key);
            self.pairs.extend_from_slice(&9u32.to_le_bytes());
            self.pairs.extend_from_slice(&4u32.to_le_bytes());
            self.pairs
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                self.pairs.extend_from_slice(&value.to_le_bytes());
            }
            self.count += 1;
            self
        }

        fn string_array(mut self, key: &str, values: &[&str]) -> Self {
            self.key(key);
            self.pairs.extend_from_slice(&9u32.to_le_bytes());
            self.pairs.extend_from_slice(&8u32.to_le_bytes());
            self.pairs
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                self.pairs
                    .extend_from_slice(&(value.len() as u64).to_le_bytes());
                self.pairs.extend_from_slice(value.as_bytes());
            }
            self.count += 1;
            self
        }

        fn tensor(mut self, name: &str, dimensions: &[u64], kind: u32) -> Self {
            self.tensors.push(PlannedTensor {
                name: name.to_string(),
                dimensions: dimensions.to_vec(),
                kind,
            });
            self
        }

        fn alignment(mut self, value: u32) -> Self {
            self.declared_alignment = Some(value);
            self.u32_value(ALIGNMENT_KEY, value)
        }

        fn key(&mut self, key: &str) {
            self.pairs
                .extend_from_slice(&(key.len() as u64).to_le_bytes());
            self.pairs.extend_from_slice(key.as_bytes());
        }

        fn build(self) -> Vec<u8> {
            self.build_version(3)
        }

        fn build_version(self, version: u32) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&MAGIC);
            out.extend_from_slice(&version.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&self.count.to_le_bytes());
            out.extend_from_slice(&self.pairs);
            out
        }

        /// Header, tensor directory, alignment padding and zeroed data, in the
        /// order and at the offsets a real GGUF file uses.
        fn build_file(self) -> Vec<u8> {
            let alignment = self
                .declared_alignment
                .map_or(DEFAULT_ALIGNMENT, u64::from)
                .max(1);

            let mut directory = Vec::new();
            let mut at = 0u64;
            for tensor in &self.tensors {
                let layout = block_layout(tensor.kind).expect("tests use known types");
                let elements: u64 = tensor.dimensions.iter().product();
                let len = elements.div_ceil(layout.elements) * layout.bytes;

                directory.extend_from_slice(&(tensor.name.len() as u64).to_le_bytes());
                directory.extend_from_slice(tensor.name.as_bytes());
                directory.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
                for extent in &tensor.dimensions {
                    directory.extend_from_slice(&extent.to_le_bytes());
                }
                directory.extend_from_slice(&tensor.kind.to_le_bytes());
                directory.extend_from_slice(&at.to_le_bytes());

                at = align_up(at + len, alignment).expect("test files are small");
            }
            let data_len = at as usize;

            let mut out = self.build();
            out.extend_from_slice(&directory);
            while !(out.len() as u64).is_multiple_of(alignment) {
                out.push(0);
            }
            out.resize(out.len() + data_len, 0);
            out
        }
    }

    #[test]
    fn the_architecture_is_read_from_the_first_key() {
        let bytes = Header::default()
            .string_value("general.architecture", "qwen2")
            .build();
        assert_eq!(architecture(&bytes).unwrap().as_deref(), Some("qwen2"));
    }

    /// The interesting case: the answer sits behind the tokenizer vocabulary,
    /// which is by far the largest thing in a real header.
    #[test]
    fn keys_of_every_type_are_stepped_over_to_reach_the_answer() {
        let vocabulary: Vec<String> = (0..2_000).map(|i| format!("token{i}")).collect();
        let vocabulary: Vec<&str> = vocabulary.iter().map(String::as_str).collect();
        let bytes = Header::default()
            .string_value("general.name", "Some Model")
            .u32_value("llama.block_count", 32)
            .u32_array("llama.head_counts", &[8; 512])
            .string_array("tokenizer.ggml.tokens", &vocabulary)
            .string_value("general.architecture", "phi3")
            .build();
        assert_eq!(architecture(&bytes).unwrap().as_deref(), Some("phi3"));
        assert_eq!(model_name(&bytes).unwrap().as_deref(), Some("Some Model"));
    }

    /// The caller holds one chunk of a file that may be gigabytes long. Running
    /// out of bytes is normal, and must not be reported as a corrupt file.
    #[test]
    fn a_header_cut_short_is_absent_rather_than_an_error() {
        let bytes = Header::default()
            .string_array("tokenizer.ggml.tokens", &["a", "b", "c"])
            .string_value("general.architecture", "llama")
            .build();
        for cut in [24, 40, bytes.len() - 1] {
            assert_eq!(
                architecture(&bytes[..cut]).unwrap(),
                None,
                "a header cut at {cut} should read as absent"
            );
        }
        assert_eq!(architecture(&bytes).unwrap().as_deref(), Some("llama"));
    }

    #[test]
    fn a_key_that_is_not_there_is_absent() {
        let bytes = Header::default()
            .string_value("general.name", "Some Model")
            .build();
        assert_eq!(architecture(&bytes).unwrap(), None);
    }

    #[test]
    fn something_that_is_not_a_gguf_file_is_an_error() {
        assert!(architecture(b"not a model at all").is_err());
        assert!(architecture(b"GG").is_err());
        assert!(architecture(b"").is_err());
    }

    #[test]
    fn an_unsupported_version_is_refused_rather_than_misread() {
        let bytes = Header::default()
            .string_value("general.architecture", "llama")
            .build_version(1);
        assert!(architecture(&bytes).is_err());
        let bytes = Header::default()
            .string_value("general.architecture", "llama")
            .build_version(99);
        assert!(architecture(&bytes).is_err());
    }

    /// A length field is attacker-controlled data in a file downloaded from the
    /// internet. It must never become an allocation or an out-of-bounds read.
    #[test]
    fn a_corrupt_length_is_bounded_rather_than_believed() {
        let mut bytes = Header::default()
            .string_value("general.architecture", "llama")
            .build();
        // Overwrite the first key's length with something enormous.
        let key_length_at = 4 + 4 + 8 + 8;
        bytes[key_length_at..key_length_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(architecture(&bytes).unwrap(), None);

        // And an array count that would overflow a usize multiply.
        let mut bytes = Header::default().u32_array("x", &[1, 2, 3]).build();
        let count_at = bytes.len() - 3 * 4 - 8;
        bytes[count_at..count_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(architecture(&bytes).unwrap(), None);
    }

    /// A metadata count in the billions is corruption, not a big model.
    #[test]
    fn an_implausible_pair_count_is_refused_immediately() {
        let mut bytes = Header::default()
            .string_value("general.architecture", "llama")
            .build();
        bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(architecture(&bytes).is_err());
    }

    /// Present but of the wrong type is neither "absent" nor a guess.
    #[test]
    fn a_key_of_the_wrong_type_is_reported_as_such() {
        let bytes = Header::default()
            .u32_value("general.architecture", 7)
            .build();
        let error = architecture(&bytes).expect_err("wrong type");
        assert!(error.to_string().contains("not a string"));
    }

    #[test]
    fn an_unknown_value_type_is_an_error() {
        let mut bytes = Header::default()
            .string_value("general.name", "x")
            .string_value("general.architecture", "llama")
            .build();
        // The first pair's type code, just past its 8-byte length and key.
        let type_at = 4 + 4 + 8 + 8 + 8 + "general.name".len();
        bytes[type_at..type_at + 4].copy_from_slice(&77u32.to_le_bytes());
        assert!(architecture(&bytes).is_err());
    }

    /// A small model shaped like a real one: an embedding, four transformer
    /// blocks of three tensors each, a final norm and an output head.
    fn four_block_model() -> Vec<u8> {
        let mut header = Header::default()
            .string_value("general.architecture", "llama")
            .u32_value("llama.block_count", 4)
            .tensor("token_embd.weight", &[64, 128], 12);
        for block in 0..4u32 {
            header = header
                .tensor(&format!("blk.{block}.attn_norm.weight"), &[64], 0)
                .tensor(&format!("blk.{block}.attn_q.weight"), &[64, 64], 12)
                .tensor(&format!("blk.{block}.ffn_down.weight"), &[256, 64], 14);
        }
        header
            .tensor("output_norm.weight", &[64], 0)
            .tensor("output.weight", &[64, 128], 12)
            .build_file()
    }

    #[test]
    fn every_tensor_offset_lands_inside_the_file() {
        let bytes = four_block_model();
        let directory = tensor_directory(&bytes).unwrap().expect("complete header");

        assert_eq!(directory.tensors.len(), 1 + 4 * 3 + 2);
        directory.validate(bytes.len() as u64).unwrap();

        for tensor in &directory.tensors {
            let extent = directory.extent_of(tensor, bytes.len() as u64);
            assert!(extent.start >= directory.data_start, "{}", tensor.name);
            assert!(extent.end <= bytes.len() as u64, "{}", tensor.name);
            assert_eq!(extent.len(), tensor.data_len().unwrap(), "{}", tensor.name);
        }
    }

    /// The claim a pipeline plan rests on: cutting by layer range covers every
    /// per-block tensor exactly once, and the tensors left over are exactly the
    /// ones no block owns.
    #[test]
    fn layer_ranges_partition_the_block_tensors_with_nothing_left_over() {
        let bytes = four_block_model();
        let directory = tensor_directory(&bytes).unwrap().unwrap();
        assert_eq!(directory.layer_count(), 4);

        let mut seen: Vec<&str> = Vec::new();
        for stage in [0..2u32, 2..4] {
            for tensor in directory.tensors_for_layers(stage) {
                assert!(!seen.contains(&tensor.name.as_str()), "{}", tensor.name);
                seen.push(&tensor.name);
            }
        }
        assert_eq!(seen.len(), 4 * 3, "every block tensor is claimed once");

        let shared: Vec<&str> = directory
            .shared_tensors()
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        assert_eq!(
            shared,
            ["token_embd.weight", "output_norm.weight", "output.weight"]
        );
        assert_eq!(seen.len() + shared.len(), directory.tensors.len());
    }

    /// The point of the whole exercise: a stage fetches its own layers, not the
    /// file.
    #[test]
    fn two_stages_ask_for_disjoint_bytes_and_neither_asks_for_the_whole_file() {
        let bytes = four_block_model();
        let file_len = bytes.len() as u64;
        let directory = tensor_directory(&bytes).unwrap().unwrap();

        let first = directory.extents_for_layers(0..2, file_len);
        let second = directory.extents_for_layers(2..4, file_len);
        assert!(!first.is_empty() && !second.is_empty());

        for a in &first {
            for b in &second {
                assert!(a.end <= b.start || b.end <= a.start, "{a:?} overlaps {b:?}");
            }
        }

        let asked: u64 = first.iter().chain(&second).map(ByteExtent::len).sum();
        assert!(asked < file_len, "{asked} should be less than {file_len}");
    }

    /// Neighbouring tensors are separated only by alignment padding, and one
    /// request for the pair beats two requests plus a round trip.
    #[test]
    fn tensors_that_sit_next_to_each_other_become_one_request() {
        let bytes = four_block_model();
        let directory = tensor_directory(&bytes).unwrap().unwrap();
        let extents = directory.extents_for_layers(0..4, bytes.len() as u64);
        assert_eq!(
            extents.len(),
            1,
            "twelve consecutive block tensors are one span, not twelve"
        );
    }

    #[test]
    fn a_stage_fetches_only_the_chunks_its_bytes_fall_in() {
        let bytes = four_block_model();
        let file_len = bytes.len() as u64;
        let directory = tensor_directory(&bytes).unwrap().unwrap();
        let extents = directory.extents_for_layers(2..4, file_len);

        let chunk_size = 4_096;
        let chunks = TensorDirectory::chunks_for_extents(&extents, chunk_size).unwrap();
        assert!(!chunks.is_empty());
        for extent in &extents {
            for byte in [extent.start, extent.end - 1] {
                assert!(
                    chunks.contains(&(byte / chunk_size)),
                    "byte {byte} uncovered"
                );
            }
        }
        let total_chunks = file_len.div_ceil(chunk_size);
        assert!(
            (chunks.len() as u64) < total_chunks,
            "fetching half the layers should not need every chunk"
        );
        assert!(TensorDirectory::chunks_for_extents(&extents, 0).is_err());
    }

    #[test]
    fn a_declared_alignment_moves_where_the_data_starts() {
        let loose = Header::default()
            .tensor("token_embd.weight", &[64, 64], 0)
            .build_file();
        let tight = Header::default()
            .alignment(1_024)
            .tensor("token_embd.weight", &[64, 64], 0)
            .build_file();

        let loose = tensor_directory(&loose).unwrap().unwrap();
        let tight = tensor_directory(&tight).unwrap().unwrap();
        assert_eq!(loose.alignment, DEFAULT_ALIGNMENT);
        assert_eq!(tight.alignment, 1_024);
        assert!(loose.data_start.is_multiple_of(DEFAULT_ALIGNMENT));
        assert!(tight.data_start.is_multiple_of(1_024));
    }

    #[test]
    fn an_alignment_that_is_not_a_power_of_two_is_refused() {
        let bytes = Header::default()
            .alignment(48)
            .tensor("token_embd.weight", &[8], 0)
            .build_file();
        let message = tensor_directory(&bytes).unwrap_err().to_string();
        assert!(message.contains("power of two"), "{message}");
    }

    /// Half a header is the normal case when a peer holds the first chunk of a
    /// file it is still fetching. That is absence, not corruption.
    #[test]
    fn a_directory_cut_short_is_absent_rather_than_an_error() {
        let bytes = four_block_model();
        let directory = tensor_directory(&bytes).unwrap().unwrap();
        for cut in [40, 64, 128, 200] {
            assert!(
                tensor_directory(&bytes[..cut]).unwrap().is_none(),
                "cut at {cut}"
            );
        }
        // The bytes up to the start of the data are enough on their own.
        let head = &bytes[..directory.data_start as usize];
        assert_eq!(tensor_directory(head).unwrap().unwrap(), directory);
    }

    #[test]
    fn a_tensor_that_runs_past_the_end_of_the_file_is_rejected() {
        let bytes = four_block_model();
        let directory = tensor_directory(&bytes).unwrap().unwrap();
        let message = directory
            .validate(directory.data_start + 8)
            .unwrap_err()
            .to_string();
        assert!(message.contains("only"), "{message}");
    }

    #[test]
    fn two_tensors_claiming_the_same_bytes_are_rejected() {
        let bytes = four_block_model();
        let mut directory = tensor_directory(&bytes).unwrap().unwrap();
        let stolen = directory.tensors[2].offset;
        directory.tensors[3].offset = stolen;
        let message = directory
            .validate(bytes.len() as u64)
            .unwrap_err()
            .to_string();
        assert!(message.contains("overlap"), "{message}");
    }

    /// A type this build does not know must not be given a made-up size. The
    /// span is then derived from where the next tensor starts, which is never
    /// short.
    #[test]
    fn an_unknown_type_reports_no_length_rather_than_a_wrong_one() {
        let unknown = TensorInfo {
            name: "blk.0.attn_q.weight".into(),
            dimensions: vec![64, 64],
            kind: 9_001,
            offset: 0,
        };
        assert_eq!(block_layout(9_001), None);
        assert_eq!(unknown.data_len(), None);
        assert_eq!(unknown.element_count(), Some(4_096));

        let directory = TensorDirectory {
            alignment: DEFAULT_ALIGNMENT,
            data_start: 1_024,
            tensors: vec![
                unknown.clone(),
                TensorInfo {
                    name: "blk.1.attn_q.weight".into(),
                    dimensions: vec![64, 64],
                    kind: 0,
                    offset: 8_192,
                },
            ],
        };
        let extent = directory.extent_of(&unknown, 100_000);
        assert_eq!(extent.start, 1_024);
        assert_eq!(extent.end, 1_024 + 8_192, "derived from the next tensor");
    }

    #[test]
    fn a_shape_that_overflows_is_reported_rather_than_wrapped() {
        let absurd = TensorInfo {
            name: "x".into(),
            dimensions: vec![u64::MAX, 2],
            kind: 0,
            offset: 0,
        };
        assert_eq!(absurd.element_count(), None);
        assert_eq!(absurd.data_len(), None);
    }

    #[test]
    fn block_names_are_read_and_everything_else_belongs_to_no_block() {
        let named = |name: &str| TensorInfo {
            name: name.into(),
            dimensions: vec![1],
            kind: 0,
            offset: 0,
        };
        assert_eq!(named("blk.0.attn_q.weight").layer_index(), Some(0));
        assert_eq!(named("blk.47.ffn_up.weight").layer_index(), Some(47));
        assert_eq!(named("token_embd.weight").layer_index(), None);
        assert_eq!(named("output.weight").layer_index(), None);
        assert_eq!(named("blk.weight").layer_index(), None);
        assert_eq!(named("blk.x.weight").layer_index(), None);
        assert_eq!(named("blk.12").layer_index(), None);
    }

    /// The block table is copied arithmetic, and a typo in it would mis-slice
    /// every model of that quantisation. Bits per weight is the invariant that
    /// catches one: a 4-bit quant that claims 9 bits, or 2, is wrong.
    #[test]
    fn every_known_block_layout_stores_a_plausible_number_of_bits_per_weight() {
        for kind in 0..=40u32 {
            let Some(layout) = block_layout(kind) else {
                continue;
            };
            assert!(layout.elements > 0 && layout.bytes > 0, "type {kind}");
            let bits = (layout.bytes as f64 * 8.0) / layout.elements as f64;
            let ceiling = match kind {
                0 | 26 => 32.0,      // F32, I32
                27 | 28 => 64.0,     // I64, F64
                1 | 25 | 30 => 16.0, // F16, I16, BF16
                24 => 8.0,           // I8
                // Every quantisation. The ceiling is above eight because the
                // eight-bit types carry per-block scales and, for Q8_K, block
                // sums as well.
                _ => 10.0,
            };
            assert!(
                bits >= 1.0 && bits <= ceiling,
                "type {kind} claims {bits} bits per weight"
            );
        }
    }
}
