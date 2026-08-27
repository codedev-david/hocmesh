//! Turning "the 4-bit Qwen" into a URL and a digest.
//!
//! hocMESH will not import a model file it cannot name by content, and a model
//! repository names files by path. This module closes that gap and nothing
//! else: it asks the Hugging Face tree API what is actually in a repository at
//! a revision, picks the one GGUF the caller meant, and returns the file's LFS
//! object id -- which is its SHA-256 -- alongside the URL to fetch it from.
//!
//! Two decisions are worth keeping.
//!
//! The digest comes from the *listing*, not from the download. Resolving is a
//! separate request to a separate endpoint from the transfer, so a CDN edge
//! that serves the wrong bytes is caught by [`crate::fetch::download_verified`]
//! rather than trusted. It is a weak guarantee -- both requests go to the same
//! origin -- but it is a real one, and it is the strongest available without
//! signatures the Hub does not publish.
//!
//! A file with no LFS record has no digest here, and this module says so rather
//! than inventing one from the git object id: `oid` on a tree entry is a git
//! blob SHA-1 over a header plus the content, which is not the SHA-256 of the
//! file and would fail verification in a way nobody could diagnose. The caller
//! is expected to refuse the import and ask for an explicit `--sha256`.
//!
//! Nothing here has been exercised against the live Hub from the machine this
//! was written on -- corporate TLS interception makes `huggingface.co`
//! unreachable there -- so the tests drive a local server with recorded shapes.
//! The failure modes that survive that are the ones documented above: an
//! endpoint that moves, or a field that changes meaning.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// The Hub. A constant rather than a setting, because a configurable model
/// origin is a configurable answer to "what code is about to run here".
pub const HUB_HOST: &str = "https://huggingface.co";

/// A page of tree entries is at most this many files; the API caps it lower,
/// and the `Link` header carries the rest.
const MAX_PAGES: usize = 32;

/// One entry in a repository tree listing.
#[derive(Debug, Clone, Deserialize)]
pub struct TreeEntry {
    #[serde(rename = "type", default)]
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub size: u64,
    /// The git object id. A SHA-1 over a git blob header plus the content --
    /// deliberately *not* used as a content digest.
    #[serde(default)]
    pub oid: Option<String>,
    #[serde(default)]
    pub lfs: Option<LfsRecord>,
}

/// The large-file record, which is where a real content digest lives.
#[derive(Debug, Clone, Deserialize)]
pub struct LfsRecord {
    /// SHA-256 of the file, bare or `sha256:`-prefixed depending on endpoint.
    pub oid: String,
    #[serde(default)]
    pub size: u64,
}

impl TreeEntry {
    pub fn is_file(&self) -> bool {
        self.kind == "file"
    }

    /// The file name without directories.
    pub fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }

    pub fn is_gguf(&self) -> bool {
        self.name().to_ascii_lowercase().ends_with(".gguf")
    }

    /// The SHA-256 of the file, when the repository published one.
    pub fn sha256(&self) -> Option<String> {
        let record = self.lfs.as_ref()?;
        crate::fetch::normalise_digest(&record.oid).ok()
    }

    /// Bytes, preferring the LFS record because the tree `size` of a pointer
    /// file is the size of the pointer rather than of the model.
    pub fn size_bytes(&self) -> u64 {
        match &self.lfs {
            Some(record) if record.size > 0 => record.size,
            _ => self.size,
        }
    }
}

/// A resolved file: everything needed to fetch it and to prove it afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HubFile {
    pub repository: String,
    pub revision: String,
    pub path: String,
    pub size_bytes: u64,
    /// `None` when the repository published no LFS digest for this file. The
    /// caller must then be given an explicit digest or refuse the import.
    pub sha256: Option<String>,
    pub url: String,
}

/// Where a repository's file listing is read from.
pub fn tree_url(repository: &str, revision: &str) -> String {
    format!(
        "{HUB_HOST}/api/models/{}/tree/{}?recursive=1",
        encode_path(repository),
        encode_path(revision)
    )
}

/// Where a file in a repository is downloaded from.
pub fn download_url(repository: &str, revision: &str, path: &str) -> String {
    format!(
        "{HUB_HOST}/{}/resolve/{}/{}",
        encode_path(repository),
        encode_path(revision),
        encode_path(path)
    )
}

/// Read a repository's file listing, following pagination.
///
/// `token` is passed through as a bearer credential for gated or private
/// repositories; the caller sources it from the environment so that it is never
/// written into a manifest or a log line.
pub async fn list_tree(
    client: &reqwest::Client,
    repository: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<Vec<TreeEntry>> {
    list_tree_from(client, &tree_url(repository, revision), repository, token).await
}

async fn list_tree_from(
    client: &reqwest::Client,
    start: &str,
    repository: &str,
    token: Option<&str>,
) -> Result<Vec<TreeEntry>> {
    let mut entries = Vec::new();
    let mut next = Some(start.to_string());
    let mut pages = 0usize;

    while let Some(url) = next.take() {
        pages += 1;
        ensure!(
            pages <= MAX_PAGES,
            "{repository} listed more than {MAX_PAGES} pages of files; refusing to keep paging"
        );

        let mut request = client.get(&url);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("listing {repository} via {url}"))?;
        let status = response.status();
        if !status.is_success() {
            bail!("{}", listing_failure(repository, status, token.is_some()));
        }

        next = next_page(response.headers());
        let page: Vec<TreeEntry> = response
            .json()
            .await
            .with_context(|| format!("reading the file listing of {repository}"))?;
        entries.extend(page);
    }

    Ok(entries)
}

/// Resolve one GGUF in a repository, by quantisation when the caller named one.
pub async fn resolve_gguf(
    client: &reqwest::Client,
    repository: &str,
    revision: &str,
    quantisation: Option<&str>,
    token: Option<&str>,
) -> Result<HubFile> {
    let entries = list_tree(client, repository, revision, token).await?;
    let chosen = select_gguf(&entries, quantisation)?;
    Ok(HubFile {
        repository: repository.to_string(),
        revision: revision.to_string(),
        path: chosen.path.clone(),
        size_bytes: chosen.size_bytes(),
        sha256: chosen.sha256(),
        url: download_url(repository, revision, &chosen.path),
    })
}

/// Pick the GGUF the caller meant, or explain what was there instead.
///
/// The error paths matter more than the happy one. A user who asks for a
/// quantisation a repository does not publish is one keystroke from success,
/// and the only thing standing between them and it is knowing what *is*
/// published -- so every failure here lists the alternatives.
pub fn select_gguf<'a>(
    entries: &'a [TreeEntry],
    quantisation: Option<&str>,
) -> Result<&'a TreeEntry> {
    let mut candidates: Vec<&TreeEntry> = entries
        .iter()
        .filter(|entry| entry.is_file() && entry.is_gguf())
        .collect();
    candidates.sort_by(|a, b| a.path.cmp(&b.path));

    if candidates.is_empty() {
        bail!("that repository publishes no .gguf file at this revision");
    }

    // A model split across several files needs all of them concatenated in
    // order, and the chunk store addresses one file. Rejecting it plainly beats
    // importing the first shard and producing a model that loads and is wrong.
    let (whole, sharded): (Vec<&'a TreeEntry>, Vec<&'a TreeEntry>) = candidates
        .iter()
        .copied()
        .partition(|entry| shard_of(entry.name()).is_none());

    let matching: Vec<&'a TreeEntry> = match quantisation {
        Some(want) => whole
            .iter()
            .copied()
            .filter(|entry| quantisation_matches(entry.name(), want))
            .collect(),
        None => whole.clone(),
    };

    match matching.len() {
        1 => Ok(matching[0]),
        0 => {
            let sharded_match = sharded.iter().any(|entry| match quantisation {
                Some(want) => quantisation_matches(entry.name(), want),
                None => true,
            });
            if whole.is_empty() || sharded_match {
                bail!(
                    "the only matching weights are split across several files ({}), which \
                     hocMESH cannot import yet; pick a quantisation that ships as one file. \
                     Available: {}",
                    sharded.len(),
                    describe(&candidates)
                );
            }
            bail!(
                "no single-file .gguf matches quantisation {:?}. Available: {}",
                quantisation.unwrap_or(""),
                describe(&whole)
            )
        }
        _ => bail!(
            "several .gguf files match; pass --quantisation to choose one. Available: {}",
            describe(&whole)
        ),
    }
}

/// Whether a filename carries a quantisation token as its own component.
///
/// Bounded by `-` and `.` but *not* by `_`, so that asking for `q4` does not
/// silently hand back `q4_k_m`: those are different files with different
/// quality, and a request for one answered with the other is the kind of thing
/// nobody notices until the output is bad.
fn quantisation_matches(name: &str, want: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let want = want.trim().to_ascii_lowercase();
    if want.is_empty() {
        return false;
    }
    let stem = name.strip_suffix(".gguf").unwrap_or(&name);
    let bytes = stem.as_bytes();
    let boundary = |index: usize| -> bool {
        match bytes.get(index) {
            None => true,
            Some(b'-') | Some(b'.') => true,
            Some(_) => false,
        }
    };
    let mut from = 0;
    while let Some(offset) = stem[from..].find(&want) {
        let start = from + offset;
        let end = start + want.len();
        let before = start == 0 || boundary(start - 1);
        if before && boundary(end) {
            return true;
        }
        from = start + 1;
    }
    false
}

/// The `(part, of)` of a sharded GGUF, or `None` for a whole one.
///
/// Upstream names shards `...-00001-of-00003.gguf`. Parsed rather than pattern
/// matched so that a file which merely contains digits is not mistaken for one.
fn shard_of(name: &str) -> Option<(u32, u32)> {
    let stem = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))
        .unwrap_or(name);
    let (head, total) = stem.rsplit_once("-of-")?;
    let (_, part) = head.rsplit_once('-')?;
    if part.is_empty() || total.is_empty() {
        return None;
    }
    let part: u32 = part.parse().ok()?;
    let total: u32 = total.parse().ok()?;
    Some((part, total))
}

fn describe(entries: &[&TreeEntry]) -> String {
    entries
        .iter()
        .map(|entry| entry.name().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Say why a listing failed in terms of what the operator can do about it.
fn listing_failure(repository: &str, status: reqwest::StatusCode, authenticated: bool) -> String {
    match status {
        reqwest::StatusCode::NOT_FOUND => format!(
            "{repository} was not found at that revision. Check the repository name and that the \
             revision exists"
        ),
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN if !authenticated => {
            format!(
                "{repository} is gated or private. Accept its licence on the Hub and set HF_TOKEN \
                 to an access token, then try again"
            )
        }
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => format!(
            "the token in HF_TOKEN is not allowed to read {repository}; the licence may still need \
             accepting on the Hub"
        ),
        other => format!(
            "listing {repository} failed with HTTP {} {}",
            other.as_u16(),
            other.canonical_reason().unwrap_or("")
        ),
    }
}

/// The `rel="next"` target of a `Link` header, if the listing was paginated.
fn next_page(headers: &reqwest::header::HeaderMap) -> Option<String> {
    let link = headers.get(reqwest::header::LINK)?.to_str().ok()?;
    for part in link.split(',') {
        let mut pieces = part.split(';');
        let target = pieces.next()?.trim();
        let is_next = pieces.any(|piece| {
            let piece = piece.trim().replace(' ', "");
            piece == "rel=\"next\"" || piece == "rel=next"
        });
        if is_next {
            let target = target.trim_start_matches('<').trim_end_matches('>');
            if !target.is_empty() {
                return Some(target.to_string());
            }
        }
    }
    None
}

/// Percent-encode everything outside the unreserved set, leaving `/` alone so
/// that repository and file paths keep their structure.
fn encode_path(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, http::header, routing::get};

    fn file(path: &str, size: u64, oid: Option<&str>) -> TreeEntry {
        TreeEntry {
            kind: "file".into(),
            path: path.into(),
            size: if oid.is_some() { 135 } else { size },
            oid: Some("0123456789abcdef0123456789abcdef01234567".into()),
            lfs: oid.map(|oid| LfsRecord {
                oid: oid.into(),
                size,
            }),
        }
    }

    fn directory(path: &str) -> TreeEntry {
        TreeEntry {
            kind: "directory".into(),
            path: path.into(),
            size: 0,
            oid: None,
            lfs: None,
        }
    }

    const OID: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn a_repository_with_one_gguf_needs_no_quantisation() {
        let entries = vec![
            directory("."),
            file("README.md", 1024, None),
            file("model-q4_k_m.gguf", 400_000_000, Some(OID)),
        ];
        let chosen = select_gguf(&entries, None).expect("the only gguf");
        assert_eq!(chosen.path, "model-q4_k_m.gguf");
        assert_eq!(chosen.sha256().as_deref(), Some(OID));
        // The LFS size wins over the pointer-file size.
        assert_eq!(chosen.size_bytes(), 400_000_000);
    }

    #[test]
    fn several_gguf_files_require_a_quantisation() {
        let entries = vec![
            file("m-q4_k_m.gguf", 1, Some(OID)),
            file("m-q8_0.gguf", 2, Some(OID)),
        ];
        let error = select_gguf(&entries, None).expect_err("ambiguous");
        let message = error.to_string();
        assert!(message.contains("--quantisation"));
        // The error has to name the alternatives, or it is a dead end.
        assert!(message.contains("m-q4_k_m.gguf"));
        assert!(message.contains("m-q8_0.gguf"));

        assert_eq!(
            select_gguf(&entries, Some("q8_0")).expect("chosen").path,
            "m-q8_0.gguf"
        );
    }

    /// `q4` and `q4_k_m` are different files. Answering a request for one with
    /// the other is a quality regression nobody would trace back to here.
    #[test]
    fn a_quantisation_token_does_not_match_a_longer_one() {
        assert!(quantisation_matches("m-q4_k_m.gguf", "q4_k_m"));
        assert!(!quantisation_matches("m-q4_k_m.gguf", "q4"));
        assert!(quantisation_matches("m-q4.gguf", "q4"));
        // TheBloke's dot-separated convention.
        assert!(quantisation_matches("Mistral-7B.Q4_K_M.gguf", "q4_k_m"));
        assert!(quantisation_matches("Mistral-7B.Q4_K_M.gguf", "Q4_K_M"));
        // Not a component of the name at all.
        assert!(!quantisation_matches("m-fq4_k_m.gguf", "q4_k_m"));
        assert!(!quantisation_matches("m-q4_k_m.gguf", ""));
        assert!(!quantisation_matches("m-iq4_xs.gguf", "q4_xs"));
        assert!(quantisation_matches("m-iq4_xs.gguf", "iq4_xs"));
    }

    #[test]
    fn a_sharded_model_is_refused_with_its_reason() {
        let entries = vec![
            file("m-q4_k_m-00001-of-00003.gguf", 1, Some(OID)),
            file("m-q4_k_m-00002-of-00003.gguf", 1, Some(OID)),
            file("m-q4_k_m-00003-of-00003.gguf", 1, Some(OID)),
        ];
        let error = select_gguf(&entries, Some("q4_k_m")).expect_err("sharded");
        assert!(error.to_string().contains("split across several files"));

        let error = select_gguf(&entries, None).expect_err("sharded");
        assert!(error.to_string().contains("split across several files"));
    }

    /// A repository that publishes both keeps working: the whole file is the
    /// one that can be imported, so it is the one that is chosen.
    #[test]
    fn a_whole_file_is_preferred_over_shards_of_another_quantisation() {
        let entries = vec![
            file("m-q4_k_m.gguf", 1, Some(OID)),
            file("m-f16-00001-of-00002.gguf", 1, Some(OID)),
            file("m-f16-00002-of-00002.gguf", 1, Some(OID)),
        ];
        assert_eq!(
            select_gguf(&entries, None).expect("the whole file").path,
            "m-q4_k_m.gguf"
        );
    }

    #[test]
    fn an_unmatched_quantisation_lists_what_is_there() {
        let entries = vec![file("m-q4_k_m.gguf", 1, Some(OID))];
        let error = select_gguf(&entries, Some("q2_k")).expect_err("no such quantisation");
        let message = error.to_string();
        assert!(message.contains("q2_k"));
        assert!(message.contains("m-q4_k_m.gguf"));
    }

    #[test]
    fn a_repository_with_no_weights_says_so() {
        let entries = vec![file("README.md", 10, None), directory("configs")];
        assert!(
            select_gguf(&entries, None)
                .expect_err("no gguf")
                .to_string()
                .contains("no .gguf")
        );
    }

    /// A directory can be named `something.gguf`. It is not a file.
    #[test]
    fn a_directory_is_never_selected() {
        let entries = vec![directory("checkpoints.gguf")];
        assert!(select_gguf(&entries, None).is_err());
    }

    /// The git object id is a SHA-1 over a blob header. Reporting it as a
    /// content digest would produce verification failures nobody could explain.
    #[test]
    fn a_file_without_an_lfs_record_has_no_digest() {
        let entry = file("tiny.gguf", 900, None);
        assert!(entry.sha256().is_none());
        assert!(entry.oid.is_some());
    }

    #[test]
    fn an_lfs_oid_is_accepted_with_or_without_its_prefix() {
        assert_eq!(
            file("m.gguf", 1, Some(&format!("sha256:{OID}")))
                .sha256()
                .as_deref(),
            Some(OID)
        );
        // Garbage in the field is reported as "no digest", never passed on.
        assert!(file("m.gguf", 1, Some("not-a-digest")).sha256().is_none());
    }

    #[test]
    fn shards_are_parsed_rather_than_guessed() {
        assert_eq!(shard_of("m-00001-of-00003.gguf"), Some((1, 3)));
        assert_eq!(shard_of("m-1-of-2.gguf"), Some((1, 2)));
        assert_eq!(shard_of("qwen2.5-7b-instruct-q4_k_m.gguf"), None);
        assert_eq!(shard_of("m-of-.gguf"), None);
        assert_eq!(shard_of("m-x-of-y.gguf"), None);
        assert_eq!(shard_of("m.gguf"), None);
    }

    #[test]
    fn urls_are_built_from_the_repository_and_revision() {
        assert_eq!(
            tree_url("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "main"),
            "https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main?recursive=1"
        );
        assert_eq!(
            download_url("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "main", "a b.gguf"),
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/a%20b.gguf"
        );
        // A revision is frequently a commit sha, which must survive intact.
        assert!(tree_url("a/b", "0123abcd").ends_with("/tree/0123abcd?recursive=1"));
    }

    /// A path that tries to leave the repository is encoded, not obeyed.
    #[test]
    fn path_separators_survive_but_nothing_else_does() {
        assert_eq!(encode_path("a/b-c_d.e~f"), "a/b-c_d.e~f");
        assert_eq!(encode_path("a b"), "a%20b");
        assert_eq!(encode_path("a?b=c"), "a%3Fb%3Dc");
        assert_eq!(encode_path("a#b"), "a%23b");
    }

    #[test]
    fn pagination_follows_only_the_next_link() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::LINK,
            "<https://example.test/p2>; rel=\"next\", <https://example.test/p9>; rel=\"last\""
                .parse()
                .expect("header"),
        );
        assert_eq!(
            next_page(&headers).as_deref(),
            Some("https://example.test/p2")
        );

        let mut only_last = reqwest::header::HeaderMap::new();
        only_last.insert(
            reqwest::header::LINK,
            "<https://example.test/p9>; rel=\"last\""
                .parse()
                .expect("header"),
        );
        assert!(next_page(&only_last).is_none());
        assert!(next_page(&reqwest::header::HeaderMap::new()).is_none());
    }

    /// End to end against a local server that answers in the Hub's shape,
    /// including a second page, because the live Hub is not reachable from the
    /// machine this was developed on.
    #[tokio::test]
    async fn a_listing_is_read_and_paged_and_resolved() {
        let page_two = std::sync::Arc::new(std::sync::OnceLock::new());
        let base_for_handler = std::sync::Arc::clone(&page_two);

        let first = move || {
            let base = std::sync::Arc::clone(&base_for_handler);
            async move {
                let next: &String = base.get().expect("the base url");
                (
                    [(header::LINK, format!("<{next}/page2>; rel=\"next\""))],
                    axum::Json(serde_json::json!([
                        {"type": "directory", "path": "docs"},
                        {"type": "file", "path": "README.md", "size": 42,
                         "oid": "0123456789abcdef0123456789abcdef01234567"},
                    ])),
                )
            }
        };
        let second = || async {
            axum::Json(serde_json::json!([
                {"type": "file", "path": "m-q8_0.gguf", "size": 135,
                 "oid": "0123456789abcdef0123456789abcdef01234567",
                 "lfs": {"oid": OID, "size": 402_653_184}},
                {"type": "file", "path": "m-f16-00001-of-00002.gguf", "size": 135,
                 "lfs": {"oid": OID, "size": 1}},
            ]))
        };

        let app = Router::new()
            .route("/api/models/{*rest}", get(first))
            .route("/page2", get(second));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binding a test server");
        let base = format!("http://{}", listener.local_addr().expect("address"));
        page_two.set(base.clone()).expect("base url");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let entries = list_tree_from(
            &reqwest::Client::new(),
            &format!("{base}/api/models/owner/repo/tree/main?recursive=1"),
            "owner/repo",
            None,
        )
        .await
        .expect("the listing");
        assert_eq!(entries.len(), 4);

        let chosen = select_gguf(&entries, None).expect("the whole gguf");
        assert_eq!(chosen.path, "m-q8_0.gguf");
        assert_eq!(chosen.sha256().as_deref(), Some(OID));
        assert_eq!(chosen.size_bytes(), 402_653_184);
    }

    #[test]
    fn a_gated_repository_is_explained_rather_than_reported_as_a_status_code() {
        let anonymous = listing_failure("owner/repo", reqwest::StatusCode::FORBIDDEN, false);
        assert!(anonymous.contains("HF_TOKEN"));
        assert!(anonymous.contains("gated"));

        let with_token = listing_failure("owner/repo", reqwest::StatusCode::UNAUTHORIZED, true);
        assert!(with_token.contains("not allowed"));

        assert!(
            listing_failure("owner/repo", reqwest::StatusCode::NOT_FOUND, false)
                .contains("not found")
        );
        assert!(
            listing_failure("owner/repo", reqwest::StatusCode::BAD_GATEWAY, false).contains("502")
        );
    }
}
