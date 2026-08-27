//! Downloading a file the mesh is going to trust, which means downloading it
//! by digest rather than by URL.
//!
//! Everything this module fetches is either executed (an inference runtime) or
//! imported into the content-addressed chunk store and served on to other
//! participants (a model). In both cases the URL is the least trustworthy part
//! of the transaction: it is a name, it can be redirected, and the machine on
//! the other end of it is not a member of the mesh. So a caller does not ask
//! for a URL here -- it asks for a digest and offers a URL as the place to look
//! for it. A body that hashes to something else is deleted, not returned.
//!
//! The two operational concessions are resume and progress. Model weights run
//! to gigabytes over connections that drop, and a download that has to start
//! over from zero is one people work around by fetching the file some other way
//! -- which is precisely the path that skips the digest check. So a partial
//! download is kept in a `.part` file and continued with a range request, and
//! the whole file is hashed at the end rather than incrementally: the bytes
//! already on disk were written by an earlier process that this one cannot
//! vouch for, and a resumed download must be exactly as suspicious of them as
//! it is of the ones still arriving.

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// How much has arrived, for a caller that wants to draw a progress line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Bytes on disk, including anything a previous attempt left behind.
    pub downloaded: u64,
    /// Total size, when the server was willing to say.
    pub total: Option<u64>,
    /// Bytes that were already present when this attempt started.
    pub resumed_from: u64,
}

/// What a completed download turned out to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Downloaded {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
    /// False when the file was already present and already correct, so a
    /// caller can say "already have it" instead of claiming to have fetched it.
    pub transferred: bool,
    /// Bytes carried over from an interrupted earlier attempt.
    pub resumed_from: u64,
}

/// Progress is reported on this boundary rather than per chunk, because a
/// multi-gigabyte download arrives in tens of thousands of pieces and a caller
/// that redraws a terminal line for each of them spends more time on the line
/// than on the download.
const PROGRESS_INTERVAL: u64 = 4 * 1024 * 1024;

/// Read buffer for hashing a file already on disk.
const HASH_BUFFER: usize = 1024 * 1024;

/// Accept a digest written either bare or with the `sha256:` prefix the OCI and
/// Hugging Face ecosystems both use, and reject anything that is not one.
///
/// Case is normalised rather than rejected: a digest pasted out of a checksum
/// file is frequently uppercase, and refusing it would teach people to edit
/// digests by hand, which is the one thing they must never do.
pub fn normalise_digest(value: &str) -> Result<String> {
    let trimmed = value.trim();
    let bare = trimmed.strip_prefix("sha256:").unwrap_or(trimmed);
    ensure!(
        bare.len() == 64 && bare.bytes().all(|b| b.is_ascii_hexdigit()),
        "expected a 64-character hex sha256 digest, got {value:?}"
    );
    Ok(bare.to_ascii_lowercase())
}

/// The SHA-256 of a file on disk, streamed rather than loaded.
pub async fn sha256_file(path: &Path) -> Result<String> {
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; HASH_BUFFER];
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

/// Fetch `url` into `destination`, and keep the result only if it hashes to
/// `expected_sha256`.
///
/// Returns without touching the network when `destination` already holds the
/// right bytes, so this is safe to call on every start-up. A file that exists
/// but hashes wrong is replaced, not trusted: a truncated or tampered cache
/// entry is worth less than no cache entry.
pub async fn download_verified(
    client: &reqwest::Client,
    url: &str,
    expected_sha256: &str,
    destination: &Path,
    mut on_progress: impl FnMut(Progress),
) -> Result<Downloaded> {
    let expected = normalise_digest(expected_sha256)?;

    if destination.is_file() {
        let actual = sha256_file(destination).await?;
        if actual == expected {
            let size_bytes = tokio::fs::metadata(destination).await?.len();
            on_progress(Progress {
                downloaded: size_bytes,
                total: Some(size_bytes),
                resumed_from: size_bytes,
            });
            return Ok(Downloaded {
                path: destination.to_path_buf(),
                size_bytes,
                sha256: actual,
                transferred: false,
                resumed_from: size_bytes,
            });
        }
    }

    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let part = part_path(destination);
    let mut already = match tokio::fs::metadata(&part).await {
        Ok(meta) if meta.is_file() => meta.len(),
        _ => 0,
    };

    let mut request = client.get(url);
    if already > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={already}-"));
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("requesting {url}"))?;
    let status = response.status();

    // 416 means the server thinks we already have at least the whole file,
    // which -- since we would not be here if the digest matched -- means what
    // is on disk is not this file at all. Start again rather than argue.
    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && already > 0 {
        let _ = tokio::fs::remove_file(&part).await;
        return Box::pin(download_verified(
            client,
            url,
            expected_sha256,
            destination,
            on_progress,
        ))
        .await;
    }
    ensure!(
        status.is_success(),
        "{url} returned HTTP {} {}",
        status.as_u16(),
        status.canonical_reason().unwrap_or("")
    );

    // A server is free to ignore a range request, and several CDNs do. If it
    // answered 200 to a ranged request it is sending the whole file, so what is
    // on disk is a prefix we must discard rather than prepend.
    let resuming = already > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    if already > 0 && !resuming {
        already = 0;
    }
    let resumed_from = already;

    let remaining = response.content_length();
    let total = remaining.map(|remaining| remaining + resumed_from);

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resuming)
        .truncate(!resuming)
        .open(&part)
        .await
        .with_context(|| format!("opening {}", part.display()))?;

    let mut downloaded = resumed_from;
    let mut last_report = downloaded;
    on_progress(Progress {
        downloaded,
        total,
        resumed_from,
    });

    let mut response = response;
    while let Some(chunk) = response
        .chunk()
        .await
        .with_context(|| format!("reading the response body from {url}"))?
    {
        file.write_all(&chunk)
            .await
            .with_context(|| format!("writing {}", part.display()))?;
        downloaded += chunk.len() as u64;
        if downloaded - last_report >= PROGRESS_INTERVAL {
            last_report = downloaded;
            on_progress(Progress {
                downloaded,
                total,
                resumed_from,
            });
        }
    }
    file.flush()
        .await
        .with_context(|| format!("flushing {}", part.display()))?;
    drop(file);
    on_progress(Progress {
        downloaded,
        total,
        resumed_from,
    });

    let actual = sha256_file(&part).await?;
    if actual != expected {
        // Poisoned, and worse, poisoned in a way a retry would resume from.
        let _ = tokio::fs::remove_file(&part).await;
        bail!(
            "{url} did not deliver what was asked for: expected sha256 {expected}, got {actual} \
             ({downloaded} bytes). The partial download has been discarded."
        );
    }

    if destination.exists() {
        tokio::fs::remove_file(destination)
            .await
            .with_context(|| format!("replacing {}", destination.display()))?;
    }
    tokio::fs::rename(&part, destination)
        .await
        .with_context(|| format!("moving {} into place", part.display()))?;

    Ok(Downloaded {
        path: destination.to_path_buf(),
        size_bytes: downloaded,
        sha256: actual,
        transferred: true,
        resumed_from,
    })
}

/// Where an interrupted download is parked.
///
/// Beside the destination rather than in a temporary directory, so that the
/// rename at the end stays on one filesystem and so that a user who wonders
/// where their disk went finds the answer next to what they asked for.
pub fn part_path(destination: &Path) -> PathBuf {
    let mut name = destination.as_os_str().to_os_string();
    name.push(".part");
    PathBuf::from(name)
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{HeaderMap, StatusCode, header},
        response::Response,
        routing::get,
    };
    use std::sync::Arc;

    const BODY: &[u8] = b"the quick brown fox jumps over the lazy dog, repeatedly and at length";

    fn digest_of(bytes: &[u8]) -> String {
        hex_lower(&Sha256::digest(bytes))
    }

    fn scratch(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hocmesh-fetch-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("scratch directory");
        path
    }

    /// The first byte of a `Range: bytes=N-` header, if there is one.
    fn range_start(headers: &HeaderMap) -> Option<u64> {
        headers
            .get(header::RANGE)?
            .to_str()
            .ok()?
            .strip_prefix("bytes=")?
            .split('-')
            .next()?
            .parse()
            .ok()
    }

    /// A server that honours ranges, one that ignores them, and one that lies
    /// about what it is serving -- the three behaviours a downloader meets in
    /// the wild and has to survive without producing a wrong file.
    async fn spawn_server() -> String {
        let ranged = |headers: HeaderMap| async move {
            match range_start(&headers) {
                Some(start) if start as usize >= BODY.len() => Response::builder()
                    .status(StatusCode::RANGE_NOT_SATISFIABLE)
                    .body(Body::empty())
                    .expect("response"),
                Some(start) => {
                    let rest = &BODY[start as usize..];
                    Response::builder()
                        .status(StatusCode::PARTIAL_CONTENT)
                        .header(header::CONTENT_LENGTH, rest.len())
                        .header(
                            header::CONTENT_RANGE,
                            format!("bytes {start}-{}/{}", BODY.len() - 1, BODY.len()),
                        )
                        .body(Body::from(rest.to_vec()))
                        .expect("response")
                }
                None => Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_LENGTH, BODY.len())
                    .body(Body::from(BODY.to_vec()))
                    .expect("response"),
            }
        };
        let ignores_range = || async {
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_LENGTH, BODY.len())
                .body(Body::from(BODY.to_vec()))
                .expect("response")
        };
        let wrong = || async { (StatusCode::OK, "not what you asked for") };
        let missing = || async { (StatusCode::NOT_FOUND, "no such object") };

        let app = Router::new()
            .route("/blob", get(ranged))
            .route("/norange", get(ignores_range))
            .route("/wrong", get(wrong))
            .route("/missing", get(missing));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binding a test server");
        let base = format!("http://{}", listener.local_addr().expect("local address"));
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        base
    }

    #[test]
    fn a_digest_is_accepted_bare_or_prefixed_and_never_malformed() {
        let bare = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(normalise_digest(bare).unwrap(), bare);
        assert_eq!(normalise_digest(&format!("sha256:{bare}")).unwrap(), bare);
        assert_eq!(normalise_digest(&bare.to_uppercase()).unwrap(), bare);
        assert_eq!(normalise_digest(&format!("  {bare}  ")).unwrap(), bare);

        assert!(normalise_digest("").is_err());
        assert!(normalise_digest("deadbeef").is_err());
        assert!(normalise_digest(&bare[..63]).is_err());
        assert!(normalise_digest(&format!("{bare}0")).is_err());
        assert!(normalise_digest(&format!("z{}", &bare[1..])).is_err());
        // A digest for a different algorithm is not silently accepted.
        assert!(normalise_digest(&format!("md5:{bare}")).is_err());
    }

    #[tokio::test]
    async fn a_file_hashes_the_same_streamed_as_in_memory() {
        let dir = scratch("hash");
        let path = dir.join("blob");
        // Larger than the read buffer, so the chunked loop is actually exercised.
        let bytes: Vec<u8> = (0..(HASH_BUFFER * 2 + 12345))
            .map(|i| (i % 251) as u8)
            .collect();
        tokio::fs::write(&path, &bytes).await.expect("writing");
        assert_eq!(sha256_file(&path).await.unwrap(), digest_of(&bytes));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_clean_download_verifies_and_lands_atomically() {
        let base = spawn_server().await;
        let dir = scratch("clean");
        let destination = dir.join("blob.bin");
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);

        let result = download_verified(
            &reqwest::Client::new(),
            &format!("{base}/blob"),
            &digest_of(BODY),
            &destination,
            move |progress| recorder.lock().expect("progress lock").push(progress),
        )
        .await
        .expect("the download succeeds");

        assert!(result.transferred);
        assert_eq!(result.resumed_from, 0);
        assert_eq!(result.size_bytes, BODY.len() as u64);
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), BODY);
        // The scratch file is gone, so a later run does not resume from it.
        assert!(!part_path(&destination).exists());

        let progress = seen.lock().expect("progress lock").clone();
        assert_eq!(
            progress.last().map(|p| p.downloaded),
            Some(BODY.len() as u64)
        );
        assert!(progress.iter().all(|p| p.total == Some(BODY.len() as u64)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_body_that_hashes_wrong_is_deleted_rather_than_returned() {
        let base = spawn_server().await;
        let dir = scratch("wrong");
        let destination = dir.join("blob.bin");

        let error = download_verified(
            &reqwest::Client::new(),
            &format!("{base}/wrong"),
            &digest_of(BODY),
            &destination,
            |_| {},
        )
        .await
        .expect_err("a mismatched body is refused");
        assert!(error.to_string().contains("did not deliver"));

        // Neither the destination nor a resumable fragment survives, because a
        // retry that resumed from poisoned bytes could never converge.
        assert!(!destination.exists());
        assert!(!part_path(&destination).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_interrupted_download_resumes_from_what_is_already_on_disk() {
        let base = spawn_server().await;
        let dir = scratch("resume");
        let destination = dir.join("blob.bin");
        let half = BODY.len() / 2;
        tokio::fs::write(part_path(&destination), &BODY[..half])
            .await
            .expect("a half-finished download");

        let result = download_verified(
            &reqwest::Client::new(),
            &format!("{base}/blob"),
            &digest_of(BODY),
            &destination,
            |_| {},
        )
        .await
        .expect("the download resumes");

        assert_eq!(result.resumed_from, half as u64);
        assert_eq!(result.size_bytes, BODY.len() as u64);
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), BODY);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A CDN that ignores `Range` answers 200 with the whole file. Appending
    /// that to the fragment on disk would double the body; the fragment has to
    /// be thrown away instead.
    #[tokio::test]
    async fn a_server_that_ignores_the_range_still_produces_the_right_file() {
        let base = spawn_server().await;
        let dir = scratch("norange");
        let destination = dir.join("blob.bin");
        tokio::fs::write(part_path(&destination), &BODY[..BODY.len() / 3])
            .await
            .expect("a half-finished download");

        let result = download_verified(
            &reqwest::Client::new(),
            &format!("{base}/norange"),
            &digest_of(BODY),
            &destination,
            |_| {},
        )
        .await
        .expect("the download restarts");

        assert_eq!(result.resumed_from, 0);
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), BODY);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Bytes left by an earlier attempt are not trusted just because they are
    /// local: the whole file is hashed, so a corrupt prefix is caught.
    #[tokio::test]
    async fn a_corrupt_fragment_is_caught_by_hashing_the_whole_file() {
        let base = spawn_server().await;
        let dir = scratch("poison");
        let destination = dir.join("blob.bin");
        let half = BODY.len() / 2;
        let mut poisoned = BODY[..half].to_vec();
        poisoned[0] ^= 0xff;
        tokio::fs::write(part_path(&destination), &poisoned)
            .await
            .expect("a corrupt fragment");

        let error = download_verified(
            &reqwest::Client::new(),
            &format!("{base}/blob"),
            &digest_of(BODY),
            &destination,
            |_| {},
        )
        .await
        .expect_err("a corrupt resume is refused");
        assert!(error.to_string().contains("did not deliver"));
        assert!(!part_path(&destination).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fragment longer than the resource means it is not the resource.
    #[tokio::test]
    async fn a_fragment_past_the_end_of_the_file_starts_over() {
        let base = spawn_server().await;
        let dir = scratch("overrun");
        let destination = dir.join("blob.bin");
        tokio::fs::write(part_path(&destination), vec![0u8; BODY.len() + 64])
            .await
            .expect("an oversized fragment");

        let result = download_verified(
            &reqwest::Client::new(),
            &format!("{base}/blob"),
            &digest_of(BODY),
            &destination,
            |_| {},
        )
        .await
        .expect("the download starts over");

        assert_eq!(result.resumed_from, 0);
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), BODY);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_file_already_present_and_correct_is_not_fetched_again() {
        let base = spawn_server().await;
        let dir = scratch("cached");
        let destination = dir.join("blob.bin");
        tokio::fs::write(&destination, BODY).await.expect("a cache");

        let result = download_verified(
            &reqwest::Client::new(),
            // Unreachable on purpose: a correct cache must not touch the network.
            &format!("{base}/missing"),
            &digest_of(BODY),
            &destination,
            |_| {},
        )
        .await
        .expect("the cache is accepted");

        assert!(!result.transferred);
        assert_eq!(result.sha256, digest_of(BODY));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cache entry that hashes wrong is worth less than no cache entry.
    #[tokio::test]
    async fn a_stale_file_already_present_is_replaced() {
        let base = spawn_server().await;
        let dir = scratch("stale");
        let destination = dir.join("blob.bin");
        tokio::fs::write(&destination, b"an older, wrong version")
            .await
            .expect("a stale cache");

        let result = download_verified(
            &reqwest::Client::new(),
            &format!("{base}/blob"),
            &digest_of(BODY),
            &destination,
            |_| {},
        )
        .await
        .expect("the stale file is replaced");

        assert!(result.transferred);
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), BODY);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_http_error_is_reported_with_its_status() {
        let base = spawn_server().await;
        let dir = scratch("404");
        let destination = dir.join("blob.bin");

        let error = download_verified(
            &reqwest::Client::new(),
            &format!("{base}/missing"),
            &digest_of(BODY),
            &destination,
            |_| {},
        )
        .await
        .expect_err("404 is an error");
        assert!(error.to_string().contains("404"));
        assert!(!destination.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_malformed_digest_is_refused_before_any_request_is_made() {
        let dir = scratch("baddigest");
        let destination = dir.join("blob.bin");
        assert!(
            download_verified(
                &reqwest::Client::new(),
                "http://127.0.0.1:1/never-reached",
                "not-a-digest",
                &destination,
                |_| {},
            )
            .await
            .is_err()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
