//! Reading only the tensors a stage is responsible for.
//!
//! The point of the whole exercise: a node holding blocks 8..16 opens a file
//! that may be missing every byte of blocks 0..8 and 16..32, reads its own
//! eight blocks, and never touches the rest. So loading is driven by an
//! explicit list of tensor names, and a name outside the stage's range is an
//! error rather than a read — a stage that could silently fall back to reading
//! a neighbour's weights would pass every test on one machine and fail only
//! once the model was really split.

use anyhow::{Context, Result, bail, ensure};
use hocmesh_model::gguf::{TensorDirectory, TensorInfo};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::dequant;

/// One tensor, reconstructed to `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Extents in GGML order, fastest-varying first. A weight matrix is
    /// `[inputs, outputs]`, stored as `outputs` contiguous rows of `inputs`.
    pub dimensions: Vec<u64>,
    pub values: Vec<f32>,
}

impl Tensor {
    /// Length of the fastest-varying dimension, which for a weight matrix is
    /// the number of inputs one output row consumes.
    #[must_use]
    pub fn row_len(&self) -> usize {
        self.dimensions.first().copied().unwrap_or(0) as usize
    }

    /// How many rows the tensor holds, which for a weight matrix is its number
    /// of outputs.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.values.len().checked_div(self.row_len()).unwrap_or(0)
    }

    /// One row, as a slice.
    #[must_use]
    pub fn row(&self, index: usize) -> &[f32] {
        let row = self.row_len();
        &self.values[index * row..(index + 1) * row]
    }

    pub fn expect_shape(&self, name: &str, dimensions: &[u64]) -> Result<()> {
        ensure!(
            self.dimensions == dimensions,
            "{name} is {:?}, expected {dimensions:?}",
            self.dimensions
        );
        Ok(())
    }
}

/// A GGUF file opened for tensor reads, with its directory already parsed.
pub struct WeightFile {
    path: PathBuf,
    file: File,
    directory: TensorDirectory,
    len: u64,
    /// The head of the file, kept so the configuration can be read back
    /// without a second open.
    pub header: Vec<u8>,
}

/// How much of a file's head is read to find the tensor directory.
///
/// A GGUF header is metadata plus one entry per tensor; a 70B model's
/// directory is a few hundred kilobytes. Reading a fixed slab and reporting
/// "the header is longer than this" beats growing a buffer against a length
/// the file itself supplied.
const HEADER_BUDGET: usize = 16 * 1024 * 1024;

impl WeightFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file =
            File::open(&path).with_context(|| format!("opening model file {}", path.display()))?;
        let len = file.metadata()?.len();
        let mut header = vec![0u8; HEADER_BUDGET.min(len as usize)];
        file.read_exact(&mut header)?;
        let directory = hocmesh_model::gguf::tensor_directory(&header)?.with_context(|| {
            format!(
                "{} has no readable tensor directory in its first {} bytes",
                path.display(),
                header.len()
            )
        })?;
        directory.validate(len)?;
        Ok(WeightFile {
            path,
            file,
            directory,
            len,
            header,
        })
    }

    #[must_use]
    pub fn directory(&self) -> &TensorDirectory {
        &self.directory
    }

    /// Whether the file declares a tensor of this name.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.info(name).is_some()
    }

    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.directory.tensors.iter().find(|t| t.name == name)
    }

    /// Read one tensor and reconstruct it to `f32`.
    ///
    /// Reads exactly the tensor's own extent. Everything else in the file may
    /// be absent, and on a stage that fetched only its own layers it will be.
    pub fn load(&mut self, name: &str) -> Result<Tensor> {
        let info = self
            .info(name)
            .with_context(|| format!("{} declares no tensor {name}", self.path.display()))?
            .clone();
        ensure!(
            dequant::is_supported(info.kind),
            "{name} is stored as {} ({}), which this engine cannot execute; \
             requantise the model",
            dequant::type_name(info.kind),
            info.kind
        );
        let elements = info
            .element_count()
            .with_context(|| format!("{name} declares a shape that overflows"))?;
        let extent = self.directory.extent_of(&info, self.len);
        let byte_len = usize::try_from(extent.len())
            .ok()
            .filter(|len| *len > 0)
            .with_context(|| format!("{name} has an unreadable extent"))?;

        let mut data = vec![0u8; byte_len];
        self.file.seek(SeekFrom::Start(extent.start))?;
        self.file
            .read_exact(&mut data)
            .with_context(|| format!("reading {name} at byte {}", extent.start))?;

        let mut values = vec![0.0f32; usize::try_from(elements)?];
        dequant::dequantize(info.kind, &data, &mut values)
            .with_context(|| format!("reconstructing {name}"))?;
        Ok(Tensor {
            dimensions: info.dimensions,
            values,
        })
    }

    /// Read a tensor only if the file declares it.
    ///
    /// For the two that are genuinely optional: `output.weight`, absent when a
    /// model ties its output head to its embedding table, and biases, which
    /// the llama family does not use but some conversions of it emit.
    pub fn load_optional(&mut self, name: &str) -> Result<Option<Tensor>> {
        if self.has(name) {
            self.load(name).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Confirm every byte a stage will read is actually present.
    ///
    /// A stage that fetched only its own chunks has a file full of holes, and
    /// a hole reads as zeros rather than as an error. Zeros are a valid weight
    /// matrix that produces a valid-looking activation, so a missing chunk
    /// would otherwise surface as a model that had quietly become worse. This
    /// checks the extents against the regions the caller says it holds, before
    /// anything is read.
    pub fn assert_layers_present(
        &self,
        blocks: std::ops::Range<u32>,
        present: &[hocmesh_model::gguf::ByteExtent],
    ) -> Result<()> {
        let mut needed = self.directory.extents_for_layers(blocks.clone(), self.len);
        needed.extend(
            self.directory
                .extents_of(&self.directory.shared_tensors(), self.len),
        );
        for extent in needed {
            let covered = present
                .iter()
                .any(|held| held.start <= extent.start && held.end >= extent.end);
            if !covered {
                bail!(
                    "{} is missing bytes {}..{}, which blocks {}..{} need; \
                     the stage would have read zeros and generated plausible nonsense",
                    self.path.display(),
                    extent.start,
                    extent.end,
                    blocks.start,
                    blocks.end
                );
            }
        }
        Ok(())
    }
}
