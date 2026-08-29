pub mod catalog;
pub mod gguf;

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

pub const DEFAULT_CHUNK_SIZE: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Gguf,
    Safetensors,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkRef {
    pub index: u32,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub revision: String,
    pub format: ModelFormat,
    pub architecture: String,
    pub parameter_count: Option<u64>,
    pub tensor_dtype: Option<String>,
    pub total_size_bytes: u64,
    pub chunks: Vec<ChunkRef>,
    #[serde(default)]
    pub metadata: std::collections::BTreeMap<String, String>,
}

/// Metadata key holding the number of transformer blocks in the model.
///
/// Optional, because a manifest can be written from a file listing alone. It is
/// the one number needed to turn `--gpu-layers N` into a quantity of bytes, so
/// where it is absent that conversion cannot be done and must not be guessed.
pub const LAYER_COUNT: &str = "layer_count";

impl ModelManifest {
    /// Transformer blocks in this model, where the manifest records them.
    ///
    /// `None` means the manifest was written without it, not that the model has
    /// no layers.
    #[must_use]
    pub fn layer_count(&self) -> Option<u32> {
        self.metadata.get(LAYER_COUNT)?.trim().parse().ok()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported manifest schema");
        ensure!(!self.model_id.trim().is_empty(), "model_id is empty");
        ensure!(!self.revision.trim().is_empty(), "revision is empty");
        ensure!(
            !self.architecture.trim().is_empty(),
            "architecture is empty"
        );
        ensure!(!self.chunks.is_empty(), "manifest has no chunks");
        let mut total = 0_u64;
        for (expected, chunk) in self.chunks.iter().enumerate() {
            ensure!(
                chunk.index as usize == expected,
                "chunk indexes are not contiguous"
            );
            ensure!(is_sha256(&chunk.sha256), "invalid chunk digest");
            ensure!(chunk.size_bytes > 0, "empty chunks are not permitted");
            // Two chunks may legitimately share a digest: a model file with the
            // same 4 MiB twice in it is a file that stores one chunk and
            // references it twice, which is the whole point of addressing chunks
            // by content. `index` is checked above, so order is still pinned,
            // and `materialize` reads by digest per index. Refusing duplicates
            // would make such a file unimportable for no security gain -- every
            // chunk still has to hash to its stated digest.
            total = total
                .checked_add(chunk.size_bytes)
                .context("model size overflow")?;
        }
        ensure!(
            total == self.total_size_bytes,
            "manifest size does not match chunks"
        );
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256(&serde_json::to_vec(self)?))
    }
}

pub struct ChunkStore {
    root: PathBuf,
}

impl ChunkStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("chunks"))?;
        Ok(Self { root })
    }

    pub fn import_reader<R: Read>(
        &self,
        mut reader: R,
        chunk_size: usize,
    ) -> Result<Vec<ChunkRef>> {
        ensure!(chunk_size > 0, "chunk size must be positive");
        let mut chunks = Vec::new();
        loop {
            let mut data = Vec::with_capacity(chunk_size);
            (&mut reader)
                .take(chunk_size as u64)
                .read_to_end(&mut data)?;
            if data.is_empty() {
                break;
            }
            let hash = self.put(&data)?;
            chunks.push(ChunkRef {
                index: chunks.len() as u32,
                sha256: hash,
                size_bytes: data.len() as u64,
            });
        }
        ensure!(!chunks.is_empty(), "cannot import an empty model");
        Ok(chunks)
    }

    pub fn put(&self, bytes: &[u8]) -> Result<String> {
        ensure!(!bytes.is_empty(), "cannot store an empty chunk");
        let hash = sha256(bytes);
        let path = self.chunk_path(&hash)?;
        if path.exists() {
            ensure!(
                self.read(&hash)? == bytes,
                "digest collision or corrupt chunk"
            );
            return Ok(hash);
        }
        let parent = path.parent().context("chunk path has no parent")?;
        fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(".{hash}.tmp-{}", std::process::id()));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, &path)?;
        Ok(hash)
    }

    pub fn read(&self, hash: &str) -> Result<Vec<u8>> {
        let path = self.chunk_path(hash)?;
        let bytes = fs::read(&path).with_context(|| format!("reading chunk {hash}"))?;
        ensure!(
            sha256(&bytes) == hash,
            "chunk {hash} failed integrity verification"
        );
        Ok(bytes)
    }

    pub fn contains(&self, chunk: &ChunkRef) -> bool {
        self.read(&chunk.sha256)
            .map(|data| data.len() as u64 == chunk.size_bytes)
            .unwrap_or(false)
    }

    pub fn materialize(&self, manifest: &ModelManifest, output: impl AsRef<Path>) -> Result<()> {
        manifest.validate()?;
        let mut file = fs::File::create(output)?;
        for chunk in &manifest.chunks {
            let data = self.read(&chunk.sha256)?;
            ensure!(data.len() as u64 == chunk.size_bytes, "chunk size mismatch");
            file.write_all(&data)?;
        }
        file.sync_all()?;
        Ok(())
    }

    /// Write a model file holding only the chunks this node was asked to keep.
    ///
    /// This is what makes "no machine holds the whole model" a fact about the
    /// disk rather than a claim about intent. The file is created at its full
    /// declared length so every tensor sits at the offset the header promises,
    /// but only the requested chunks are written; the rest is a hole. On any
    /// filesystem that supports sparse files the absent bytes occupy nothing.
    ///
    /// Returns the byte ranges that really are present, in file order, which is
    /// what [`crate::gguf::TensorDirectory`] consumers check a stage's layer
    /// range against before reading a single weight. A hole reads back as zeros
    /// rather than as an error, and zeros are a perfectly valid weight matrix,
    /// so the caller must check rather than assume.
    pub fn materialize_partial(
        &self,
        manifest: &ModelManifest,
        output: impl AsRef<Path>,
        keep: &[u32],
    ) -> Result<Vec<gguf::ByteExtent>> {
        manifest.validate()?;
        let file = fs::File::create(output)?;
        file.set_len(manifest.total_size_bytes)?;

        let mut present: Vec<gguf::ByteExtent> = Vec::new();
        let mut offset = 0_u64;
        for chunk in &manifest.chunks {
            let start = offset;
            offset += chunk.size_bytes;
            if !keep.contains(&chunk.index) {
                continue;
            }
            let data = self.read(&chunk.sha256)?;
            ensure!(data.len() as u64 == chunk.size_bytes, "chunk size mismatch");
            write_at(&file, start, &data)?;
            // Chunks are contiguous, so a run of kept chunks is one extent.
            match present.last_mut() {
                Some(last) if last.end == start => last.end = offset,
                _ => present.push(gguf::ByteExtent { start, end: offset }),
            }
        }
        file.sync_all()?;
        Ok(present)
    }

    fn chunk_path(&self, hash: &str) -> Result<PathBuf> {
        ensure!(is_sha256(hash), "invalid SHA-256 digest");
        Ok(self.root.join("chunks").join(&hash[..2]).join(hash))
    }
}

pub struct ModelRegistry {
    conn: Connection,
}

impl ModelRegistry {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS manifests (
               digest TEXT PRIMARY KEY,
               model_id TEXT NOT NULL,
               revision TEXT NOT NULL,
               format TEXT NOT NULL,
               manifest_json TEXT NOT NULL,
               created_at INTEGER NOT NULL DEFAULT (unixepoch()),
               UNIQUE(model_id, revision)
             );
             CREATE INDEX IF NOT EXISTS idx_manifests_model ON manifests(model_id);",
        )?;
        Ok(Self { conn })
    }

    pub fn register(&self, manifest: &ModelManifest) -> Result<String> {
        let digest = manifest.digest()?;
        let json = serde_json::to_string(manifest)?;
        let format = match manifest.format {
            ModelFormat::Gguf => "gguf",
            ModelFormat::Safetensors => "safetensors",
        };
        self.conn.execute(
            "INSERT INTO manifests(digest,model_id,revision,format,manifest_json)
             VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(model_id,revision) DO UPDATE SET
               digest=excluded.digest,format=excluded.format,manifest_json=excluded.manifest_json",
            params![digest, manifest.model_id, manifest.revision, format, json],
        )?;
        Ok(digest)
    }

    pub fn get(&self, model_id: &str, revision: &str) -> Result<Option<ModelManifest>> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT manifest_json FROM manifests WHERE model_id=?1 AND revision=?2",
                params![model_id, revision],
                |row| row.get(0),
            )
            .optional()?;
        json.map(|value| serde_json::from_str(&value).map_err(Into::into))
            .transpose()
    }

    pub fn list(&self) -> Result<Vec<ModelManifest>> {
        let mut statement = self
            .conn
            .prepare("SELECT manifest_json FROM manifests ORDER BY model_id,revision")?;
        statement
            .query_map([], |row| row.get::<_, String>(0))?
            .map(|row| Ok(serde_json::from_str(&row?)?))
            .collect()
    }
}

pub fn manifest_for_file(
    store: &ChunkStore,
    path: impl AsRef<Path>,
    model_id: impl Into<String>,
    revision: impl Into<String>,
    format: ModelFormat,
    architecture: impl Into<String>,
    chunk_size: usize,
) -> Result<ModelManifest> {
    let chunks = store.import_reader(fs::File::open(path)?, chunk_size)?;
    let total_size_bytes = chunks.iter().map(|chunk| chunk.size_bytes).sum();
    let manifest = ModelManifest {
        schema_version: 1,
        model_id: model_id.into(),
        revision: revision.into(),
        format,
        architecture: architecture.into(),
        parameter_count: None,
        tensor_dtype: None,
        total_size_bytes,
        chunks,
        metadata: Default::default(),
    };
    manifest.validate()?;
    Ok(manifest)
}

pub fn validate_format_header(format: ModelFormat, bytes: &[u8]) -> Result<()> {
    match format {
        ModelFormat::Gguf => ensure!(bytes.starts_with(b"GGUF"), "invalid GGUF header"),
        ModelFormat::Safetensors => {
            ensure!(bytes.len() >= 8, "safetensors header is truncated");
            let length = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            ensure!(
                length > 1 && length <= 100_000_000,
                "invalid safetensors header length"
            );
        }
    }
    Ok(())
}

/// Write at an absolute offset without disturbing a shared cursor.
///
/// Both platforms offer this; neither offers it through the same trait, and
/// seeking instead would make the operation order-dependent for no gain.
fn write_at(file: &fs::File, offset: u64, data: &[u8]) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(data, offset)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut written = 0_usize;
        while written < data.len() {
            let n = file.seek_write(&data[written..], offset + written as u64)?;
            ensure!(n > 0, "write made no progress");
            written += n;
        }
    }
    Ok(())
}

pub fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!("hocmesh-model-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn chunk_round_trip_is_content_addressed_and_deduplicated() {
        let root = temp_dir();
        let store = ChunkStore::open(&root).unwrap();
        let refs = store.import_reader(&b"abcdefghij"[..], 4).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!(store.put(b"abcd").unwrap(), refs[0].sha256);
        assert_eq!(store.read(&refs[2].sha256).unwrap(), b"ij");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registry_replaces_only_the_same_revision() {
        let root = temp_dir();
        let registry = ModelRegistry::open(root.join("registry.db")).unwrap();
        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "tiny".into(),
            revision: "v1".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: Some(7),
            tensor_dtype: Some("q4".into()),
            total_size_bytes: 3,
            chunks: vec![ChunkRef {
                index: 0,
                sha256: sha256(b"abc"),
                size_bytes: 3,
            }],
            metadata: Default::default(),
        };
        let digest = registry.register(&manifest).unwrap();
        assert_eq!(registry.get("tiny", "v1").unwrap(), Some(manifest));
        assert_eq!(registry.list().unwrap().len(), 1);
        assert_eq!(digest.len(), 64);
        drop(registry);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_manifest_is_rejected() {
        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "x".into(),
            revision: "v1".into(),
            format: ModelFormat::Gguf,
            architecture: "x".into(),
            parameter_count: None,
            tensor_dtype: None,
            total_size_bytes: 2,
            chunks: vec![ChunkRef {
                index: 1,
                sha256: sha256(b"x"),
                size_bytes: 1,
            }],
            metadata: Default::default(),
        };
        assert!(manifest.validate().is_err());
    }

    /// A file that contains the same chunk-sized run of bytes twice is a
    /// perfectly ordinary file, and a content-addressed store is exactly the
    /// thing that should store it once and reference it twice. Refusing the
    /// manifest would make the file unimportable and buy nothing: each chunk is
    /// still checked against its own digest on the way in and on the way out.
    #[test]
    fn a_repeated_chunk_is_stored_once_and_referenced_twice() {
        let root = temp_dir();
        let store = ChunkStore::open(root.join("chunks")).unwrap();
        let path = root.join("repeats.bin");
        let block = vec![7_u8; 32];
        let mut body = block.clone();
        body.extend_from_slice(&block);
        body.extend_from_slice(&block);
        fs::write(&path, &body).unwrap();

        let manifest = manifest_for_file(
            &store,
            &path,
            "repeats",
            "v1",
            ModelFormat::Gguf,
            "llama",
            32,
        )
        .expect("a file with repeated blocks is importable");

        assert_eq!(manifest.chunks.len(), 3);
        assert_eq!(manifest.chunks[0].sha256, manifest.chunks[2].sha256);
        assert_eq!(manifest.total_size_bytes, body.len() as u64);

        let restored = root.join("restored.bin");
        store.materialize(&manifest, &restored).unwrap();
        assert_eq!(fs::read(&restored).unwrap(), body);
    }

    #[test]
    fn gguf_and_safetensors_headers_are_validated() {
        validate_format_header(ModelFormat::Gguf, b"GGUFrest").unwrap();
        assert!(validate_format_header(ModelFormat::Gguf, b"nope").is_err());
        let mut safetensors = 16_u64.to_le_bytes().to_vec();
        safetensors.extend_from_slice(b"{\"metadata\":{}}");
        validate_format_header(ModelFormat::Safetensors, &safetensors).unwrap();
        assert!(validate_format_header(ModelFormat::Safetensors, &[0; 4]).is_err());
    }

    #[test]
    fn corrupt_chunks_are_never_reported_as_present() {
        let root = temp_dir();
        let store = ChunkStore::open(&root).unwrap();
        let digest = store.put(b"trusted").unwrap();
        let chunk = ChunkRef {
            index: 0,
            sha256: digest.clone(),
            size_bytes: 7,
        };
        fs::write(
            root.join("chunks").join(&digest[..2]).join(&digest),
            b"tampered",
        )
        .unwrap();
        assert!(store.read(&digest).is_err());
        assert!(!store.contains(&chunk));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn materialization_restores_original_byte_order() {
        let root = temp_dir();
        let source = root.join("model.gguf");
        let output = root.join("restored.gguf");
        fs::write(&source, b"GGUFabcdefgh").unwrap();
        let store = ChunkStore::open(root.join("store")).unwrap();
        let manifest =
            manifest_for_file(&store, &source, "tiny", "1", ModelFormat::Gguf, "llama", 3).unwrap();
        store.materialize(&manifest, &output).unwrap();
        assert_eq!(fs::read(output).unwrap(), b"GGUFabcdefgh");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_storage_inputs_are_rejected() {
        let root = temp_dir();
        let store = ChunkStore::open(&root).unwrap();
        assert!(store.put(&[]).is_err());
        assert!(store.import_reader(&b"x"[..], 0).is_err());
        assert!(store.import_reader(&b""[..], 1).is_err());
        assert!(store.read("../not-a-digest").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_manifest_without_a_layer_count_says_so_rather_than_reporting_none() {
        let mut manifest = ModelManifest {
            schema_version: 1,
            model_id: "m".into(),
            revision: "r".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: None,
            tensor_dtype: None,
            total_size_bytes: 0,
            chunks: Vec::new(),
            metadata: Default::default(),
        };
        assert_eq!(manifest.layer_count(), None);
        manifest
            .metadata
            .insert(LAYER_COUNT.into(), " 60 ".into());
        assert_eq!(manifest.layer_count(), Some(60));
        manifest
            .metadata
            .insert(LAYER_COUNT.into(), "many".into());
        assert_eq!(
            manifest.layer_count(),
            None,
            "an unparseable layer count was read as a number, which would size \
             a memory budget from nonsense"
        );
    }
}
