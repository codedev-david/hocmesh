//! Turning GGML's stored blocks back into `f32`.
//!
//! Every quantised GGML type packs a fixed number of elements into a fixed
//! number of bytes together with the scales needed to reconstruct them. The
//! layouts are exact and small, and they are reproduced here rather than
//! linked against so that a stage can be executed by this repository alone.
//!
//! Only the types whose layout is simple enough to state completely are
//! implemented. The k-quants (`Q2_K` through `Q8_K`) and the i-quants carry
//! per-sub-block scale codebooks, and a subtly wrong reconstruction of one of
//! them would produce activations that are wrong without being detectably
//! wrong -- the exact failure [`hocmesh_model::gguf::block_layout`] refuses to
//! risk by returning `None` for a type it does not know. So an unsupported
//! type is an error naming itself, never a best effort.
//!
//! Format reference: <https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h>

use anyhow::{Result, bail, ensure};

/// GGML type codes this module can reconstruct.
pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q4_1: u32 = 3;
pub const Q5_0: u32 = 6;
pub const Q5_1: u32 = 7;
pub const Q8_0: u32 = 8;
pub const BF16: u32 = 30;

/// Resolve the name a person would type to the code the format uses.
///
/// The error lists what is accepted rather than saying the input was wrong,
/// because the set is small, arbitrary, and impossible to guess.
pub fn kind_by_name(name: &str) -> anyhow::Result<u32> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "f32" => F32,
        "f16" => F16,
        "bf16" => BF16,
        "q4_0" => Q4_0,
        "q4_1" => Q4_1,
        "q5_0" => Q5_0,
        "q5_1" => Q5_1,
        "q8_0" => Q8_0,
        other => anyhow::bail!(
            "unknown weight format {other}; use one of \
             f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0"
        ),
    })
}

/// Whether [`dequantize`] can reconstruct this GGML type code.
#[must_use]
pub fn is_supported(kind: u32) -> bool {
    matches!(kind, F32 | F16 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | BF16)
}

/// The name a GGUF file would use for a type code, for error messages.
#[must_use]
pub fn type_name(kind: u32) -> &'static str {
    match kind {
        F32 => "F32",
        F16 => "F16",
        Q4_0 => "Q4_0",
        Q4_1 => "Q4_1",
        4 | 5 => "a withdrawn Q4_2/Q4_3",
        Q5_0 => "Q5_0",
        Q5_1 => "Q5_1",
        Q8_0 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16..=23 | 29 => "an i-quant",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        BF16 => "BF16",
        _ => "an unknown type",
    }
}

/// IEEE half precision to `f32`, including subnormals, infinities and NaN.
///
/// Written out rather than taken from a crate because it is fifteen lines and
/// this is the only floating-point conversion in the engine that a model file
/// can reach directly.
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    match exponent {
        // Zero or subnormal. A subnormal half is `mantissa * 2^-24` with no
        // implicit leading one, so it is renormalised by hand: the highest set
        // bit becomes the implicit one and the exponent follows from where it
        // was. `f32` has the range for every one of these, so none of them
        // stays subnormal after the widening.
        0 => {
            if mantissa == 0 {
                return f32::from_bits(sign);
            }
            let leading = mantissa.leading_zeros();
            let exponent = 134 - leading;
            let significand = (mantissa << (leading - 8)) & 0x007f_ffff;
            f32::from_bits(sign | (exponent << 23) | significand)
        }
        // Infinity or NaN: the f32 exponent is all ones and the mantissa
        // carries over, which keeps a signalling NaN signalling.
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)),
    }
}

/// bfloat16 to `f32`, which is the top half of the `f32` and nothing else.
#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

/// Reconstruct `count` elements of `kind` from `data` into `out`.
///
/// `out` is written in full and its length is the number of elements expected,
/// which is checked rather than trusted: a shape and a byte length that
/// disagree mean the tensor directory and the data section describe different
/// files, and reading on would mix one tensor's bytes into another's values.
pub fn dequantize(kind: u32, data: &[u8], out: &mut [f32]) -> Result<()> {
    let layout = hocmesh_model::gguf::block_layout(kind)
        .ok_or_else(|| anyhow::anyhow!("GGML type {kind} is not one this build knows"))?;
    ensure!(
        is_supported(kind),
        "GGML type {} ({kind}) is not one this engine can execute; \
         requantise the model to Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, F16, BF16 or F32",
        type_name(kind)
    );
    let per_block = layout.elements as usize;
    ensure!(
        out.len().is_multiple_of(per_block),
        "{} elements is not a whole number of {}-element blocks",
        out.len(),
        per_block
    );
    let blocks = out.len() / per_block;
    let needed = blocks * layout.bytes as usize;
    ensure!(
        data.len() >= needed,
        "tensor data is {} bytes, {needed} needed for {} elements of {}",
        data.len(),
        out.len(),
        type_name(kind)
    );

    match kind {
        F32 => {
            for (element, chunk) in out.iter_mut().zip(data.chunks_exact(4)) {
                *element = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        F16 => {
            for (element, chunk) in out.iter_mut().zip(data.chunks_exact(2)) {
                *element = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        BF16 => {
            for (element, chunk) in out.iter_mut().zip(data.chunks_exact(2)) {
                *element = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        // 32 elements as one f16 scale and 16 bytes holding two 4-bit values
        // each, biased by 8. The two nibbles of a byte are elements i and
        // i + 16, not i and i + 1 -- an interleave that is easy to get subtly
        // wrong and produces a model that merely generates badly.
        Q4_0 => {
            for (block, out) in data
                .chunks_exact(18)
                .zip(out.chunks_exact_mut(32))
                .take(blocks)
            {
                let scale = f16_to_f32(le_u16(block, 0));
                for (i, byte) in block[2..18].iter().enumerate() {
                    out[i] = ((byte & 0x0f) as f32 - 8.0) * scale;
                    out[i + 16] = ((byte >> 4) as f32 - 8.0) * scale;
                }
            }
        }
        // As Q4_0 but with a per-block minimum instead of a fixed bias, so the
        // stored value is unsigned and the block can represent a range that
        // does not straddle zero.
        Q4_1 => {
            for (block, out) in data
                .chunks_exact(20)
                .zip(out.chunks_exact_mut(32))
                .take(blocks)
            {
                let scale = f16_to_f32(le_u16(block, 0));
                let min = f16_to_f32(le_u16(block, 2));
                for (i, byte) in block[4..20].iter().enumerate() {
                    out[i] = (byte & 0x0f) as f32 * scale + min;
                    out[i + 16] = (byte >> 4) as f32 * scale + min;
                }
            }
        }
        // Q4_0 plus a fifth bit per element, held apart in a 32-bit field
        // where bit i belongs to element i.
        Q5_0 => {
            for (block, out) in data
                .chunks_exact(22)
                .zip(out.chunks_exact_mut(32))
                .take(blocks)
            {
                let scale = f16_to_f32(le_u16(block, 0));
                let high = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
                for (i, byte) in block[6..22].iter().enumerate() {
                    let low = ((high >> i) & 1) << 4;
                    let high_half = ((high >> (i + 16)) & 1) << 4;
                    out[i] = (((byte & 0x0f) as u32 | low) as f32 - 16.0) * scale;
                    out[i + 16] = (((byte >> 4) as u32 | high_half) as f32 - 16.0) * scale;
                }
            }
        }
        Q5_1 => {
            for (block, out) in data
                .chunks_exact(24)
                .zip(out.chunks_exact_mut(32))
                .take(blocks)
            {
                let scale = f16_to_f32(le_u16(block, 0));
                let min = f16_to_f32(le_u16(block, 2));
                let high = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
                for (i, byte) in block[8..24].iter().enumerate() {
                    let low = ((high >> i) & 1) << 4;
                    let high_half = ((high >> (i + 16)) & 1) << 4;
                    out[i] = ((byte & 0x0f) as u32 | low) as f32 * scale + min;
                    out[i + 16] = ((byte >> 4) as u32 | high_half) as f32 * scale + min;
                }
            }
        }
        // One f16 scale and 32 signed bytes, in element order.
        Q8_0 => {
            for (block, out) in data
                .chunks_exact(34)
                .zip(out.chunks_exact_mut(32))
                .take(blocks)
            {
                let scale = f16_to_f32(le_u16(block, 0));
                for (element, byte) in out.iter_mut().zip(&block[2..34]) {
                    *element = f32::from(*byte as i8) * scale;
                }
            }
        }
        _ => bail!("unreachable: {} passed the support check", type_name(kind)),
    }
    Ok(())
}

/// Quantise `values` into `kind`, for building test fixtures.
///
/// Only the types with no search in their encoder are offered, which is every
/// type [`dequantize`] handles: the scale follows from the block's extreme
/// value, so encoding is a formula rather than a choice. This is here so a
/// test can build a model file whose weights it knows exactly, and so
/// round-tripping is testable without shipping a fixture.
pub fn quantize(kind: u32, values: &[f32], out: &mut Vec<u8>) -> Result<()> {
    ensure!(is_supported(kind), "cannot quantise to {}", type_name(kind));
    let per_block = hocmesh_model::gguf::block_layout(kind)
        .map(|layout| layout.elements as usize)
        .unwrap_or(1);
    ensure!(
        values.len().is_multiple_of(per_block),
        "{} values is not a whole number of {per_block}-element blocks",
        values.len()
    );

    match kind {
        F32 => values
            .iter()
            .for_each(|value| out.extend_from_slice(&value.to_le_bytes())),
        F16 => values
            .iter()
            .for_each(|value| out.extend_from_slice(&f32_to_f16(*value).to_le_bytes())),
        BF16 => values.iter().for_each(|value| {
            out.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes())
        }),
        Q8_0 => {
            for block in values.chunks_exact(32) {
                let peak = block.iter().fold(0.0f32, |peak, v| peak.max(v.abs()));
                let scale = round_trip_scale(peak / 127.0);
                out.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
                for value in block {
                    out.push(quantise_signed(*value, scale, 127) as u8);
                }
            }
        }
        Q4_0 | Q5_0 => {
            let (bias, bytes_of_high) = if kind == Q4_0 { (8i32, 0) } else { (16i32, 4) };
            for block in values.chunks_exact(32) {
                // The extreme is taken with its sign: a symmetric encoding
                // reconstructs `(level - bias) * scale`, so the scale is set
                // by the value furthest from zero in whichever direction.
                let extreme = block
                    .iter()
                    .copied()
                    .fold(0.0f32, |far, v| if v.abs() > far.abs() { v } else { far });
                let scale = round_trip_scale(extreme / -(bias as f32));
                let levels: Vec<i32> = block
                    .iter()
                    .map(|value| {
                        (quantise_unsigned(*value, scale, bias) + bias).clamp(0, 2 * bias - 1)
                    })
                    .collect();
                out.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
                if bytes_of_high > 0 {
                    let mut high = 0u32;
                    for (i, level) in levels.iter().enumerate() {
                        high |= (((*level as u32) >> 4) & 1) << i;
                    }
                    out.extend_from_slice(&high.to_le_bytes());
                }
                for i in 0..16 {
                    out.push(((levels[i] & 0x0f) | ((levels[i + 16] & 0x0f) << 4)) as u8);
                }
            }
        }
        Q4_1 | Q5_1 => {
            let (levels_max, bytes_of_high) = if kind == Q4_1 { (15i32, 0) } else { (31i32, 4) };
            for block in values.chunks_exact(32) {
                let min = block.iter().copied().fold(f32::INFINITY, f32::min);
                let max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let min = round_trip_scale(min);
                let scale = round_trip_scale((max - min) / levels_max as f32);
                let levels: Vec<i32> = block
                    .iter()
                    .map(|value| {
                        if scale == 0.0 {
                            0
                        } else {
                            (((value - min) / scale).round() as i32).clamp(0, levels_max)
                        }
                    })
                    .collect();
                out.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
                out.extend_from_slice(&f32_to_f16(min).to_le_bytes());
                if bytes_of_high > 0 {
                    let mut high = 0u32;
                    for (i, level) in levels.iter().enumerate() {
                        high |= (((*level as u32) >> 4) & 1) << i;
                    }
                    out.extend_from_slice(&high.to_le_bytes());
                }
                for i in 0..16 {
                    out.push(((levels[i] & 0x0f) | ((levels[i + 16] & 0x0f) << 4)) as u8);
                }
            }
        }
        _ => bail!("unreachable: {} passed the support check", type_name(kind)),
    }
    Ok(())
}

/// A scale is stored as `f16`, so the encoder has to use the value the decoder
/// will read back or every element inherits the rounding error of the scale.
fn round_trip_scale(scale: f32) -> f32 {
    f16_to_f32(f32_to_f16(scale))
}

fn quantise_signed(value: f32, scale: f32, limit: i32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    ((value / scale).round() as i32).clamp(-limit - 1, limit) as i8
}

fn quantise_unsigned(value: f32, scale: f32, bias: i32) -> i32 {
    if scale == 0.0 {
        return 0;
    }
    ((value / scale).round() as i32).clamp(-bias, bias - 1)
}

/// `f32` to IEEE half precision, rounding to nearest even.
#[must_use]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        // Infinity stays infinity; NaN stays NaN with a mantissa that cannot
        // round down to zero and turn into an infinity.
        let payload = if mantissa == 0 { 0 } else { 0x0200 };
        return sign | 0x7c00 | payload;
    }
    let unbiased = exponent - 127 + 15;
    if unbiased >= 0x1f {
        return sign | 0x7c00;
    }
    if unbiased <= 0 {
        // Subnormal or zero. Shifting by more than 24 would discard the
        // implicit one along with everything else, so it is clamped.
        if unbiased < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = (14 - unbiased) as u32;
        let rounded = (mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    // Rounding can carry into the exponent, which the shift below absorbs.
    sign | (((unbiased as u32) << 10) as u16).wrapping_add((rounded >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(kind: u32, values: &[f32]) -> Vec<f32> {
        let mut bytes = Vec::new();
        quantize(kind, values, &mut bytes).expect("encode");
        let mut out = vec![0.0; values.len()];
        dequantize(kind, &bytes, &mut out).expect("decode");
        out
    }

    #[test]
    fn half_precision_survives_the_round_trip_it_can() {
        for value in [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 6.1035156e-5] {
            assert_eq!(f16_to_f32(f32_to_f16(value)), value, "{value}");
        }
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f16_to_f32(f32_to_f16(f32::INFINITY)), f32::INFINITY);
        assert_eq!(f16_to_f32(f32_to_f16(1e30)), f32::INFINITY);
        // Subnormal: representable in f16 only as a denormal.
        assert!((f16_to_f32(f32_to_f16(1e-7)) - 1.1920929e-7).abs() < 1e-9);
        assert_eq!(f16_to_f32(f32_to_f16(0.0)), 0.0);
        assert!(f16_to_f32(f32_to_f16(-0.0)).is_sign_negative());
    }

    #[test]
    fn bfloat16_keeps_the_top_half_of_the_float() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    /// Exactly representable values must come back exactly, which is what
    /// makes the quantised path testable at all: any error is then the codec's
    /// and not the fixture's.
    #[test]
    fn lossless_types_are_lossless() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 4.0).collect();
        for kind in [F32, F16] {
            assert_eq!(round_trip(kind, &values), values, "{}", type_name(kind));
        }
    }

    /// Every quantised type has a bounded error, and the bound is set by how
    /// many levels it has across the block's range.
    #[test]
    fn every_quantised_type_reconstructs_within_its_step() {
        let values: Vec<f32> = (0..32)
            .map(|i| ((i * 37) % 61) as f32 / 8.0 - 4.0)
            .collect();
        let span = 8.0f32;
        for (kind, levels) in [
            (Q4_0, 15.0),
            (Q4_1, 15.0),
            (Q5_0, 31.0),
            (Q5_1, 31.0),
            (Q8_0, 255.0),
        ] {
            let back = round_trip(kind, &values);
            let worst = values
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let step = span / levels;
            assert!(
                worst <= step,
                "{} was off by {worst}, more than one step of {step}",
                type_name(kind)
            );
        }
    }

    /// A block of one repeated value has zero range, which is the case where a
    /// scale of zero is correct and a division by it is not.
    #[test]
    fn a_constant_block_does_not_divide_by_zero() {
        for kind in [Q4_0, Q4_1, Q5_0, Q5_1, Q8_0] {
            let back = round_trip(kind, &[0.0; 32]);
            assert!(back.iter().all(|v| *v == 0.0), "{}", type_name(kind));
            let back = round_trip(kind, &[2.5; 32]);
            assert!(
                back.iter().all(|v| (v - 2.5).abs() < 0.2),
                "{} lost a constant block: {back:?}",
                type_name(kind)
            );
        }
    }

    /// The nibble interleave is the detail most easily got wrong, and getting
    /// it wrong still decodes -- to a permutation of the right values. Pinning
    /// one hand-built block catches that.
    #[test]
    fn the_low_nibble_is_element_i_and_the_high_nibble_is_element_i_plus_sixteen() {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        block[2] = 0x0f; // element 0 = 15 - 8 = 7, element 16 = 0 - 8 = -8
        let mut out = vec![0.0; 32];
        dequantize(Q4_0, &block, &mut out).expect("decode");
        assert_eq!(out[0], 7.0);
        assert_eq!(out[16], -8.0);
        assert_eq!(out[1], -8.0);
    }

    #[test]
    fn an_unsupported_type_says_so_rather_than_guessing() {
        let error = dequantize(12, &[0; 144], &mut [0.0; 256])
            .unwrap_err()
            .to_string();
        assert!(error.contains("Q4_K"), "{error}");
        assert!(!is_supported(12));
        assert!(is_supported(Q8_0));
    }

    #[test]
    fn a_short_buffer_is_refused_rather_than_read_past() {
        let error = dequantize(Q8_0, &[0; 33], &mut [0.0; 32])
            .unwrap_err()
            .to_string();
        assert!(error.contains("34 needed"), "{error}");
    }
}
