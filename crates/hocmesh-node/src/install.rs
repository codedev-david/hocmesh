//! The two things a fresh machine needs before it can run a model: an
//! inference runtime, and some weights.
//!
//! Both are downloads, and hocMESH treats a download as untrusted until it
//! hashes to something expected. The two differ in where that expectation comes
//! from, and the difference is deliberate:
//!
//! * The **runtime is executed**, so its digest is compiled into this binary
//!   ([`hocmesh_gpu::runtime`]). Nothing at run time can widen what is
//!   acceptable -- not a flag, not a config file, not a server response.
//! * The **weights are data**, so their digest is resolved from the repository
//!   that publishes them and then verified against the bytes that arrive. That
//!   catches a CDN serving the wrong object; it does not, and cannot, defend
//!   against a repository that publishes bad weights. An operator who wants a
//!   stronger claim pins `--sha256` themselves, which is required for `--url`.
//!
//! Weights land in the content-addressed chunk store, which means that once one
//! machine has pulled a model the rest of the mesh can seed from it and never
//! touch the internet again.

use anyhow::{Context, Result, ensure};
use hocmesh_gpu::runtime;
use hocmesh_model::{ChunkStore, ModelFormat, ModelRegistry, catalog, manifest_for_file};
use hocmesh_transport::{fetch, hub};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

/// Where downloads are parked before they are verified and unpacked.
///
/// Inside the hocMESH home rather than the system temp directory, so that a
/// resumable partial download survives a reboot and so that an operator who
/// wonders where their disk went finds it in the directory they chose.
fn downloads_dir(home: &Path) -> PathBuf {
    home.join("downloads")
}

/// The Hub credential, when the operator has set one.
///
/// Read from the environment rather than a flag so that it never reaches a
/// shell history, a process listing, or a log line.
fn hub_token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .find_map(|name| match std::env::var(name) {
            Ok(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
            _ => None,
        })
}

/// Draw a progress line on stderr, so stdout stays parseable by a script.
fn progress_reporter(label: String) -> impl FnMut(fetch::Progress) {
    move |progress| {
        let line = match progress.total {
            Some(total) if total > 0 => format!(
                "\r{label}: {} / {} ({:.0}%)",
                human_bytes(progress.downloaded),
                human_bytes(total),
                (progress.downloaded as f64 / total as f64) * 100.0
            ),
            _ => format!("\r{label}: {}", human_bytes(progress.downloaded)),
        };
        let mut stderr = std::io::stderr();
        let _ = write!(stderr, "{line}");
        let _ = stderr.flush();
    }
}

fn finish_progress() {
    let _ = writeln!(std::io::stderr());
}

/// Sizes people can compare against a disk, rather than digit counts.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

/// What an install did, so the caller can report it honestly.
#[derive(Debug, Clone)]
pub struct InstalledRuntime {
    pub executable: PathBuf,
    pub build: &'static str,
    pub asset: &'static str,
    pub sha256: &'static str,
    /// False when the runtime was already installed and nothing was fetched.
    pub installed_now: bool,
}

/// Fetch, verify and unpack the pinned llama.cpp build for this host.
pub async fn install_runtime(
    home: &Path,
    force: bool,
    keep_archive: bool,
) -> Result<InstalledRuntime> {
    let asset = runtime::asset_for_host()?;
    let target = runtime::runtime_dir(home);

    if !force && let Some(existing) = runtime::installed_runtime(home) {
        return Ok(InstalledRuntime {
            executable: existing,
            build: runtime::PINNED_BUILD,
            asset: asset.asset,
            sha256: asset.sha256,
            installed_now: false,
        });
    }

    let archive = downloads_dir(home).join(asset.asset);
    eprintln!(
        "Fetching llama.cpp {} for {}/{} ({})",
        runtime::PINNED_BUILD,
        asset.os,
        asset.arch,
        human_bytes(asset.size_bytes)
    );
    let client = reqwest::Client::builder()
        .user_agent(concat!("hocmesh/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let downloaded = fetch::download_verified(
        &client,
        &asset.url(),
        asset.sha256,
        &archive,
        progress_reporter(asset.asset.to_string()),
    )
    .await?;
    finish_progress();

    // Only now, after the digest matched, is any of this allowed near the
    // filesystem as code.
    if target.exists() {
        std::fs::remove_dir_all(&target)
            .with_context(|| format!("clearing {}", target.display()))?;
    }
    std::fs::create_dir_all(&target).with_context(|| format!("creating {}", target.display()))?;

    let archive_path = downloaded.path.clone();
    let kind = asset.archive;
    let extract_into = target.clone();
    tokio::task::spawn_blocking(move || extract(&archive_path, kind, &extract_into))
        .await
        .context("the extraction task panicked")??;

    let executable = runtime::locate_executable(&target).with_context(|| {
        format!(
            "{} unpacked but contained no {} executable",
            asset.asset,
            runtime::EXECUTABLE_STEM
        )
    })?;
    runtime::mark_executable(&executable)?;
    runtime::record_runtime(home, &executable)?;

    if !keep_archive {
        let _ = std::fs::remove_file(&downloaded.path);
    }

    Ok(InstalledRuntime {
        executable,
        build: runtime::PINNED_BUILD,
        asset: asset.asset,
        sha256: asset.sha256,
        installed_now: true,
    })
}

/// Unpack a verified archive.
///
/// Both unpackers are the upstream crates' own, which refuse entries whose
/// paths would escape the destination -- including the symlink-then-write form
/// that a hand-rolled loop gets wrong. `no_escape_is_possible` below asserts
/// that property against this code path rather than trusting the docs.
fn extract(archive: &Path, kind: runtime::ArchiveKind, into: &Path) -> Result<()> {
    let file =
        std::fs::File::open(archive).with_context(|| format!("opening {}", archive.display()))?;
    match kind {
        runtime::ArchiveKind::Zip => {
            let mut zip = zip::ZipArchive::new(file)
                .with_context(|| format!("reading {} as a zip archive", archive.display()))?;
            zip.extract(into)
                .with_context(|| format!("unpacking {} into {}", archive.display(), into.display()))
        }
        runtime::ArchiveKind::TarGz => {
            let decoder = flate2::read::GzDecoder::new(file);
            let mut tar = tar::Archive::new(decoder);
            tar.set_preserve_permissions(true);
            tar.set_overwrite(true);
            tar.unpack(into)
                .with_context(|| format!("unpacking {} into {}", archive.display(), into.display()))
        }
    }
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// Where a pull is being told to look.
#[derive(Debug, Clone)]
pub enum PullSource {
    /// A catalogued id: the repository and preferred quantisation come from
    /// [`hocmesh_model::catalog`].
    Catalogued(&'static catalog::CatalogEntry),
    /// An arbitrary Hugging Face repository.
    Repository {
        repository: String,
        quantisation: Option<String>,
    },
    /// A direct URL. Requires `--sha256`, because there is no listing to
    /// resolve a digest from and importing unverified weights is not a thing
    /// this command will do.
    Url(String),
}

/// Everything a pull needs that is not policy.
#[derive(Debug, Clone)]
pub struct PullRequest {
    pub source: PullSource,
    pub revision: Option<String>,
    /// Overrides the digest resolved from the repository, and is mandatory for
    /// [`PullSource::Url`].
    pub sha256: Option<String>,
    /// What to register the model as. Defaults to the catalogue id, or to a
    /// name derived from the repository.
    pub model_id: Option<String>,
    /// Overrides the architecture read out of the GGUF header.
    pub architecture: Option<String>,
    pub chunk_size: usize,
    /// Keep the downloaded file after importing. Off by default: the chunk
    /// store already holds the bytes, so the download is a second copy.
    pub keep_download: bool,
}

/// What a pull produced.
#[derive(Debug, Clone)]
pub struct PulledModel {
    pub model_id: String,
    pub revision: String,
    pub architecture: String,
    pub source_url: String,
    pub size_bytes: u64,
    pub chunks: usize,
    pub manifest_digest: String,
    pub sha256: String,
    /// False when the file was already on disk and already correct.
    pub downloaded: bool,
}

/// Resolve a request to a URL and, where possible, a digest.
#[derive(Debug)]
struct Resolved {
    url: String,
    sha256: Option<String>,
    file_name: String,
    revision: String,
    model_id: String,
    expected_bytes: Option<u64>,
    licence: Option<&'static str>,
}

async fn resolve(client: &reqwest::Client, request: &PullRequest) -> Result<Resolved> {
    match &request.source {
        PullSource::Catalogued(entry) => {
            let revision = request
                .revision
                .clone()
                .unwrap_or_else(|| entry.revision.to_string());
            let file = hub::resolve_gguf(
                client,
                entry.repository,
                &revision,
                Some(entry.quantisation),
                hub_token().as_deref(),
            )
            .await
            .with_context(|| {
                format!(
                    "resolving {} ({}) on the Hub. Catalogue entries are unverified pointers: if \
                     this repository has moved, pass --repository directly",
                    entry.id, entry.repository
                )
            })?;
            Ok(Resolved {
                file_name: file_name_of(&file.path),
                url: file.url,
                sha256: file.sha256,
                revision,
                model_id: request
                    .model_id
                    .clone()
                    .unwrap_or_else(|| entry.id.to_string()),
                expected_bytes: Some(file.size_bytes.max(entry.approx_bytes)),
                licence: Some(entry.license),
            })
        }
        PullSource::Repository {
            repository,
            quantisation,
        } => {
            let revision = request.revision.clone().unwrap_or_else(|| "main".into());
            let file = hub::resolve_gguf(
                client,
                repository,
                &revision,
                quantisation.as_deref(),
                hub_token().as_deref(),
            )
            .await?;
            Ok(Resolved {
                file_name: file_name_of(&file.path),
                url: file.url,
                sha256: file.sha256,
                revision,
                model_id: request
                    .model_id
                    .clone()
                    .unwrap_or_else(|| model_id_from_repository(repository)),
                expected_bytes: Some(file.size_bytes),
                licence: None,
            })
        }
        PullSource::Url(url) => {
            ensure!(
                request.sha256.is_some(),
                "--sha256 is required with --url: there is no listing to resolve a digest from, \
                 and hocMESH will not import weights it cannot verify"
            );
            let file_name = file_name_of(url);
            Ok(Resolved {
                model_id: request
                    .model_id
                    .clone()
                    .unwrap_or_else(|| model_id_from_file_name(&file_name)),
                file_name,
                url: url.clone(),
                sha256: None,
                revision: request.revision.clone().unwrap_or_else(|| "main".into()),
                expected_bytes: None,
                licence: None,
            })
        }
    }
}

/// Fetch a model, verify it, and import it into the chunk store and registry.
pub async fn pull_model(home: &Path, request: PullRequest) -> Result<PulledModel> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("hocmesh/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let resolved = resolve(&client, &request).await?;

    // An explicit pin outranks whatever the repository says, because it is the
    // only expectation in this flow that the operator chose themselves.
    let expected = match request.sha256.as_deref() {
        Some(pin) => fetch::normalise_digest(pin)?,
        None => resolved.sha256.clone().context(
            "that file publishes no LFS digest, so hocMESH cannot verify what it downloads; \
             pass --sha256 to pin it yourself",
        )?,
    };

    if let Some(licence) = resolved.licence {
        eprintln!("{} is {licence} licensed.", resolved.model_id);
    }
    match resolved.expected_bytes {
        Some(bytes) => eprintln!(
            "Fetching {} ({}) from {}",
            resolved.file_name,
            human_bytes(bytes),
            resolved.url
        ),
        None => eprintln!("Fetching {} from {}", resolved.file_name, resolved.url),
    }

    let destination = downloads_dir(home).join(&resolved.file_name);
    let downloaded = fetch::download_verified(
        &client,
        &resolved.url,
        &expected,
        &destination,
        progress_reporter(resolved.file_name.clone()),
    )
    .await?;
    finish_progress();

    let imported = import_file(
        home,
        &downloaded.path,
        &resolved.model_id,
        &resolved.revision,
        request.architecture.as_deref(),
        request.chunk_size,
    )?;

    if !request.keep_download {
        // The chunk store holds these bytes now; a second copy is just disk.
        let _ = std::fs::remove_file(&downloaded.path);
    }

    Ok(PulledModel {
        model_id: resolved.model_id,
        revision: resolved.revision,
        architecture: imported.architecture,
        source_url: resolved.url,
        size_bytes: imported.size_bytes,
        chunks: imported.chunks,
        manifest_digest: imported.manifest_digest,
        sha256: downloaded.sha256,
        downloaded: downloaded.transferred,
    })
}

/// The part of a pull that has nothing to do with the network, split out so
/// that `model-import` and `model-pull` agree about what an import is.
pub struct Imported {
    pub architecture: String,
    pub size_bytes: u64,
    pub chunks: usize,
    pub manifest_digest: String,
}

pub fn import_file(
    home: &Path,
    path: &Path,
    model_id: &str,
    revision: &str,
    architecture: Option<&str>,
    chunk_size: usize,
) -> Result<Imported> {
    let store = ChunkStore::open(home.join("model-cache"))?;

    // Read the header before chunking, so a file that is not a model is
    // rejected before it is copied into the store.
    let head = read_head(path)?;
    hocmesh_model::validate_format_header(ModelFormat::Gguf, &head)?;

    let architecture = match architecture {
        Some(explicit) => explicit.to_string(),
        None => hocmesh_model::gguf::architecture(&head)?.context(
            "this file does not declare general.architecture in its header; pass --architecture",
        )?,
    };

    let manifest = manifest_for_file(
        &store,
        path,
        model_id,
        revision,
        ModelFormat::Gguf,
        architecture.clone(),
        chunk_size,
    )?;
    std::fs::create_dir_all(home)?;
    let registry = ModelRegistry::open(home.join("model-registry.db"))?;
    let manifest_digest = registry.register(&manifest)?;

    Ok(Imported {
        architecture,
        size_bytes: manifest.total_size_bytes,
        chunks: manifest.chunks.len(),
        manifest_digest,
    })
}

/// Enough of the front of a file to hold a GGUF key/value block.
///
/// Bounded rather than "the whole file", because these run to gigabytes and the
/// header is the only part with anything to say.
fn read_head(path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    const HEAD: usize = 8 * 1024 * 1024;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut buffer = Vec::new();
    Read::by_ref(&mut file)
        .take(HEAD as u64)
        .read_to_end(&mut buffer)
        .with_context(|| format!("reading the header of {}", path.display()))?;
    Ok(buffer)
}

/// The last path segment of a URL or path, without any query string.
fn file_name_of(location: &str) -> String {
    let without_query = location.split(['?', '#']).next().unwrap_or(location);
    let name = without_query
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .unwrap_or("model.gguf");
    // Never let a remote name choose where a file lands.
    let cleaned: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
        .collect();
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        "model.gguf".to_string()
    } else {
        cleaned
    }
}

/// `Qwen/Qwen2.5-0.5B-Instruct-GGUF` becomes `qwen2.5-0.5b-instruct`.
fn model_id_from_repository(repository: &str) -> String {
    let name = repository.rsplit('/').next().unwrap_or(repository);
    let trimmed = name
        .strip_suffix("-GGUF")
        .or_else(|| name.strip_suffix("-gguf"))
        .unwrap_or(name);
    trimmed.to_ascii_lowercase()
}

fn model_id_from_file_name(file_name: &str) -> String {
    let stem = file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))
        .unwrap_or(file_name);
    stem.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hocmesh-install-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch directory");
        path
    }

    #[test]
    fn sizes_are_reported_in_units_people_compare_against_a_disk() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(4 * 1024 * 1024), "4.0 MiB");
        assert_eq!(human_bytes(1024 * 1024 * 1024), "1.0 GiB");
    }

    /// A remote name must never decide where a local file lands.
    #[test]
    fn a_remote_file_name_cannot_steer_the_download_path() {
        assert_eq!(file_name_of("https://h/a/b/model-q4.gguf"), "model-q4.gguf");
        assert_eq!(file_name_of("https://h/a/b/m.gguf?download=true"), "m.gguf");
        assert_eq!(file_name_of("https://h/a/b/m.gguf#frag"), "m.gguf");
        assert_eq!(file_name_of("https://h/a/b/"), "b");
        // Traversal, separators and shell characters are all stripped.
        assert_eq!(file_name_of("https://h/../../etc/passwd"), "passwd");
        assert_eq!(file_name_of("https://h/a/..%2f..%2fx"), "..2f..2fx");
        assert_eq!(file_name_of("https://h/a/$(whoami).gguf"), "whoami.gguf");
        assert_eq!(file_name_of("https://h/a/.."), "model.gguf");
        assert_eq!(file_name_of(""), "model.gguf");
    }

    #[test]
    fn a_model_id_is_derived_from_the_repository_when_none_is_given() {
        assert_eq!(
            model_id_from_repository("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            "qwen2.5-0.5b-instruct"
        );
        assert_eq!(
            model_id_from_repository("microsoft/Phi-3-mini-4k-instruct-gguf"),
            "phi-3-mini-4k-instruct"
        );
        assert_eq!(model_id_from_repository("owner/plain"), "plain");
        assert_eq!(model_id_from_file_name("Model-Q4_K_M.gguf"), "model-q4_k_m");
    }

    /// The archive is written by someone else. An entry that points outside the
    /// destination must not be written there -- this asserts it against the
    /// code path that actually ships, not against a dependency's changelog.
    #[test]
    fn no_escape_is_possible_when_unpacking_a_hostile_archive() {
        let root = scratch("escape");
        let into = root.join("into");
        std::fs::create_dir_all(&into).expect("destination");
        let archive = root.join("hostile.tar.gz");

        {
            let file = std::fs::File::create(&archive).expect("archive");
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
            let mut builder = tar::Builder::new(encoder);
            for name in ["../escaped.txt", "../../escaped-twice.txt", "/absolute.txt"] {
                let payload = b"owned";
                let mut header = tar::Header::new_gnu();
                header.set_size(payload.len() as u64);
                header.set_mode(0o644);
                // `set_path` refuses to write `..`, which is the tar crate
                // declining to *produce* this archive -- but a hostile archive
                // arrives over the network, not from this crate. Poke the name
                // field directly so the bytes on disk are the ones an attacker
                // would actually send, then let extraction meet them.
                let raw = header.as_old_mut();
                raw.name[..name.len()].copy_from_slice(name.as_bytes());
                header.set_cksum();
                builder
                    .append(&header, &payload[..])
                    .expect("appending a hostile entry");
            }
            let payload = b"fine";
            let mut header = tar::Header::new_gnu();
            header.set_size(payload.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, "build/bin/llama-cli", &payload[..])
                .expect("appending a legitimate entry");
            builder.into_inner().expect("tar").finish().expect("gzip");
        }

        // Guard the guard: if the hostile names never made it into the file,
        // everything below would pass without testing anything.
        {
            let file = std::fs::File::open(&archive).expect("archive");
            let mut listing = tar::Archive::new(flate2::read::GzDecoder::new(file));
            let names: Vec<String> = listing
                .entries()
                .expect("entries")
                .map(|entry| {
                    let entry = entry.expect("entry");
                    String::from_utf8_lossy(&entry.path_bytes()).into_owned()
                })
                .collect();
            assert!(
                names.iter().any(|name| name.starts_with("../")),
                "the archive under test is not hostile: {names:?}"
            );
            assert!(
                names.iter().any(|name| name.starts_with('/')),
                "the archive under test has no absolute path: {names:?}"
            );
        }

        // Extraction may refuse the archive outright or skip the bad entries;
        // either is acceptable. Writing outside `into` is not.
        let _ = extract(&archive, runtime::ArchiveKind::TarGz, &into);

        assert!(!root.join("escaped.txt").exists());
        assert!(!root.join("escaped-twice.txt").exists());
        assert!(
            !root
                .parent()
                .expect("temp dir")
                .join("escaped-twice.txt")
                .exists()
        );
        assert!(!Path::new("/absolute.txt").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The happy path of extraction plus discovery, without the network: a
    /// tarball shaped like the real one is unpacked and the runtime is found
    /// and recorded where `infer` will look for it.
    #[test]
    fn a_well_formed_archive_installs_and_becomes_the_default_runtime() {
        let root = scratch("unpack");
        let home = root.join("home");
        let target = runtime::runtime_dir(&home);
        std::fs::create_dir_all(&target).expect("target");
        let archive = root.join("llama.tar.gz");

        {
            let file = std::fs::File::create(&archive).expect("archive");
            let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
            let mut builder = tar::Builder::new(encoder);
            for (name, payload) in [
                ("build/bin/llama-cli", &b"#!/bin/sh\n"[..]),
                ("build/bin/libllama.so", &b"\x7fELF"[..]),
            ] {
                let mut header = tar::Header::new_gnu();
                header.set_size(payload.len() as u64);
                header.set_mode(0o755);
                header.set_cksum();
                builder
                    .append_data(&mut header, name, payload)
                    .expect("appending");
            }
            builder.into_inner().expect("tar").finish().expect("gzip");
        }

        extract(&archive, runtime::ArchiveKind::TarGz, &target).expect("extraction");
        // The sibling shared library has to come along, or the executable that
        // was extracted will not start.
        assert!(target.join("build/bin/libllama.so").is_file());

        let executable = runtime::locate_executable(&target).expect("the runtime");
        runtime::mark_executable(&executable).expect("permissions");
        runtime::record_runtime(&home, &executable).expect("recording");
        assert_eq!(
            runtime::require_runtime(&home)
                .expect("a default runtime")
                .file_name(),
            executable.file_name()
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A URL source without a pin is refused before anything is fetched: there
    /// is nothing to check the bytes against, and importing them anyway would
    /// put unverified weights in the store that other nodes then seed from.
    #[tokio::test]
    async fn a_direct_url_without_a_digest_is_refused() {
        let request = PullRequest {
            source: PullSource::Url("http://127.0.0.1:1/m.gguf".into()),
            revision: None,
            sha256: None,
            model_id: None,
            architecture: None,
            chunk_size: hocmesh_model::DEFAULT_CHUNK_SIZE,
            keep_download: false,
        };
        let error = resolve(&reqwest::Client::new(), &request)
            .await
            .expect_err("an unpinned url is refused");
        assert!(error.to_string().contains("--sha256 is required"));
    }

    #[tokio::test]
    async fn a_direct_url_with_a_digest_resolves_without_touching_the_network() {
        let request = PullRequest {
            source: PullSource::Url("https://example.test/weights/tiny-q4_k_m.gguf".into()),
            revision: None,
            sha256: Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into()),
            model_id: None,
            architecture: None,
            chunk_size: hocmesh_model::DEFAULT_CHUNK_SIZE,
            keep_download: false,
        };
        let resolved = resolve(&reqwest::Client::new(), &request)
            .await
            .expect("a pinned url resolves");
        assert_eq!(resolved.file_name, "tiny-q4_k_m.gguf");
        assert_eq!(resolved.model_id, "tiny-q4_k_m");
        assert_eq!(resolved.revision, "main");
        // The digest comes from the operator's pin, not from the resolver.
        assert!(resolved.sha256.is_none());
    }

    /// Importing reads the architecture out of the file rather than trusting a
    /// flag, and refuses anything that is not a GGUF model.
    #[test]
    fn an_import_derives_the_architecture_from_the_header() {
        let root = scratch("import");
        let home = root.join("home");
        std::fs::create_dir_all(&home).expect("home");

        let mut file = Vec::new();
        file.extend_from_slice(b"GGUF");
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes()); // tensors
        file.extend_from_slice(&1u64.to_le_bytes()); // kv pairs
        let key = b"general.architecture";
        file.extend_from_slice(&(key.len() as u64).to_le_bytes());
        file.extend_from_slice(key);
        file.extend_from_slice(&8u32.to_le_bytes()); // string
        file.extend_from_slice(&5u64.to_le_bytes());
        file.extend_from_slice(b"qwen2");
        file.extend_from_slice(&[0u8; 4096]);

        let path = root.join("tiny.gguf");
        std::fs::write(&path, &file).expect("model file");

        let imported = import_file(&home, &path, "tiny", "v1", None, 1024).expect("import");
        assert_eq!(imported.architecture, "qwen2");
        assert_eq!(imported.size_bytes, file.len() as u64);
        assert!(imported.chunks > 1);
        assert!(!imported.manifest_digest.is_empty());

        // An explicit architecture still wins, for a file that predates the key.
        let imported =
            import_file(&home, &path, "tiny", "v2", Some("llama"), 1024).expect("import");
        assert_eq!(imported.architecture, "llama");

        // Something that is not a model is refused before it is chunked.
        let junk = root.join("notes.txt");
        std::fs::write(&junk, b"this is not a model").expect("junk");
        assert!(import_file(&home, &junk, "junk", "v1", None, 1024).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }
}
