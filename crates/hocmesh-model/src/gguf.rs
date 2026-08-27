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

use anyhow::{Result, bail, ensure};

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Builder for header bytes, so the tests state what they are testing
    /// rather than carrying a wall of hex.
    #[derive(Default)]
    struct Header {
        pairs: Vec<u8>,
        count: u64,
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
            out.extend_from_slice(&7u64.to_le_bytes()); // tensor count
            out.extend_from_slice(&self.count.to_le_bytes());
            out.extend_from_slice(&self.pairs);
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
}
