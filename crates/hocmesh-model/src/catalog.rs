//! A short list of models that `hocmesh model-pull <id>` understands, so that
//! getting weights onto a machine is one command rather than a research task.
//!
//! What this is: a convenience index from a memorable id to a repository, a
//! preferred quantisation, and the licence the operator is agreeing to. It
//! saves typing `--repository Qwen/Qwen2.5-0.5B-Instruct-GGUF --quantisation
//! q4_k_m` and nothing more.
//!
//! What this deliberately is **not** is a digest. An entry names a repository
//! and a revision; the exact filename and its SHA-256 are resolved from the
//! repository at pull time and then verified against the bytes that arrive. Two
//! reasons, and the second is the important one:
//!
//! 1. Filenames churn far more than repository names do. A catalogue of
//!    filenames goes stale quietly; a catalogue of repositories does not.
//! 2. A digest written here that nobody could verify would be worse than no
//!    digest at all, because it would look like a guarantee. The machine this
//!    catalogue was written on cannot reach the Hub -- TLS interception -- so
//!    every entry below is an unverified pointer, and it says so. If an entry
//!    is wrong the pull fails with the repository's own file listing attached,
//!    which is a diagnosable failure. `--sha256` is available for anyone who
//!    wants to pin a file themselves, and is required for `--url`.
//!
//! `approx_bytes` is a rounded figure for the preferred quantisation, present
//! so the CLI can warn before a multi-gigabyte download rather than during one.
//! It is never compared against anything.
//!
//! Every entry is permissively licensed (Apache-2.0 or MIT). That is a hard
//! filter, not a preference: hocMESH ships this list, and shipping a pointer to
//! weights whose licence restricts redistribution would put an obligation on
//! operators that they never agreed to.

/// One catalogued model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogEntry {
    /// What the operator types.
    pub id: &'static str,
    /// The Hugging Face repository holding GGUF weights.
    pub repository: &'static str,
    /// Pinned to a branch rather than a commit, because a commit that turns out
    /// to be wrong cannot be corrected without a hocMESH release. The digest is
    /// resolved and verified per pull either way.
    pub revision: &'static str,
    /// The quantisation to pick when the operator does not name one.
    pub quantisation: &'static str,
    /// The llama.cpp architecture family, recorded in the model manifest.
    pub architecture: &'static str,
    /// SPDX identifier of the weights licence.
    pub license: &'static str,
    /// Human-readable parameter count.
    pub parameters: &'static str,
    /// Rounded size of the preferred quantisation, for a pre-download warning.
    pub approx_bytes: u64,
    /// One line on what it is for.
    pub summary: &'static str,
}

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

/// The catalogue, smallest first, so that `model-catalog` reads as a ladder
/// from "runs on anything" upwards.
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "smollm2-360m-instruct",
        repository: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
        revision: "main",
        quantisation: "q8_0",
        architecture: "llama",
        license: "Apache-2.0",
        parameters: "360M",
        approx_bytes: 390 * MB,
        summary: "Smallest useful instruct model; for proving a mesh works end to end.",
    },
    CatalogEntry {
        id: "qwen2.5-0.5b-instruct",
        repository: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        revision: "main",
        quantisation: "q4_k_m",
        architecture: "qwen2",
        license: "Apache-2.0",
        parameters: "0.5B",
        approx_bytes: 400 * MB,
        summary: "Fits in RAM on nearly anything; the default first pull.",
    },
    CatalogEntry {
        id: "qwen2.5-1.5b-instruct",
        repository: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        revision: "main",
        quantisation: "q4_k_m",
        architecture: "qwen2",
        license: "Apache-2.0",
        parameters: "1.5B",
        approx_bytes: 1100 * MB,
        summary: "Noticeably better than 0.5B and still comfortable on a laptop.",
    },
    CatalogEntry {
        id: "phi-3-mini-4k-instruct",
        repository: "microsoft/Phi-3-mini-4k-instruct-gguf",
        revision: "main",
        quantisation: "q4",
        architecture: "phi3",
        license: "MIT",
        parameters: "3.8B",
        approx_bytes: 2 * GB + 300 * MB,
        summary: "Strong reasoning for its size; MIT licensed.",
    },
    CatalogEntry {
        id: "mistral-7b-instruct-v0.2",
        repository: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        revision: "main",
        quantisation: "q4_k_m",
        architecture: "llama",
        license: "Apache-2.0",
        parameters: "7B",
        approx_bytes: 4 * GB + 100 * MB,
        summary: "A well-understood 7B baseline.",
    },
    CatalogEntry {
        id: "qwen2.5-7b-instruct",
        repository: "Qwen/Qwen2.5-7B-Instruct-GGUF",
        revision: "main",
        quantisation: "q4_k_m",
        architecture: "qwen2",
        license: "Apache-2.0",
        parameters: "7B",
        approx_bytes: 4 * GB + 700 * MB,
        summary: "The largest entry here; expect a long download and 8 GB of RAM.",
    },
];

/// Look up a catalogue id, case-insensitively.
pub fn lookup(id: &str) -> Option<&'static CatalogEntry> {
    let id = id.trim();
    CATALOG
        .iter()
        .find(|entry| entry.id.eq_ignore_ascii_case(id))
}

/// Ids close enough to `id` to be worth suggesting when a lookup misses.
///
/// Substring matching rather than edit distance: the ids are structured names,
/// and someone typing `qwen` or `7b` is filtering, not misspelling.
pub fn suggestions(id: &str) -> Vec<&'static str> {
    let needle = id.trim().to_ascii_lowercase();
    if needle.is_empty() {
        return CATALOG.iter().map(|entry| entry.id).collect();
    }
    CATALOG
        .iter()
        .filter(|entry| {
            let candidate = entry.id.to_ascii_lowercase();
            candidate.contains(&needle) || needle.contains(&candidate)
        })
        .map(|entry| entry.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_catalogue_id_resolves_to_a_repository() {
        let entry = lookup("qwen2.5-0.5b-instruct").expect("a catalogued model");
        assert_eq!(entry.repository, "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(entry.quantisation, "q4_k_m");
        assert_eq!(lookup("QWEN2.5-0.5B-INSTRUCT"), Some(entry));
        assert_eq!(lookup("  qwen2.5-0.5b-instruct  "), Some(entry));
        assert!(lookup("no-such-model").is_none());
    }

    /// Two entries sharing an id would make `lookup` order-dependent.
    #[test]
    fn every_id_is_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for entry in CATALOG {
            assert!(seen.insert(entry.id), "{} is listed twice", entry.id);
        }
    }

    /// hocMESH ships this list, so it may only point at weights an operator can
    /// redistribute without acquiring an obligation they were not told about.
    #[test]
    fn every_entry_is_permissively_licensed() {
        for entry in CATALOG {
            assert!(
                matches!(entry.license, "Apache-2.0" | "MIT"),
                "{} is {} licensed; only Apache-2.0 and MIT may be catalogued",
                entry.id,
                entry.license
            );
        }
    }

    /// An id is typed at a shell, and a repository is pasted into a URL.
    #[test]
    fn ids_are_lowercase_and_repositories_are_owner_slash_name() {
        for entry in CATALOG {
            assert_eq!(
                entry.id,
                entry.id.to_ascii_lowercase(),
                "{} is not lowercase",
                entry.id
            );
            assert!(!entry.id.contains(' '), "{} contains a space", entry.id);
            let mut parts = entry.repository.split('/');
            let owner = parts.next().unwrap_or_default();
            let name = parts.next().unwrap_or_default();
            assert!(
                !owner.is_empty() && !name.is_empty() && parts.next().is_none(),
                "{} is not owner/name",
                entry.repository
            );
            assert!(!entry.revision.is_empty());
            assert!(!entry.quantisation.is_empty());
            assert!(!entry.architecture.is_empty());
            assert!(!entry.summary.is_empty());
            assert!(entry.approx_bytes > 0, "{} has no size", entry.id);
        }
    }

    /// The list is presented smallest first; if it stops being sorted the
    /// "start here" reading of `model-catalog` quietly breaks.
    #[test]
    fn the_catalogue_is_ordered_smallest_first() {
        for pair in CATALOG.windows(2) {
            assert!(
                pair[0].approx_bytes <= pair[1].approx_bytes,
                "{} is listed before the smaller {}",
                pair[0].id,
                pair[1].id
            );
        }
    }

    #[test]
    fn a_near_miss_gets_suggestions_and_an_empty_query_gets_everything() {
        assert!(suggestions("qwen").contains(&"qwen2.5-0.5b-instruct"));
        assert!(suggestions("qwen").contains(&"qwen2.5-7b-instruct"));
        assert!(!suggestions("qwen").contains(&"phi-3-mini-4k-instruct"));
        assert_eq!(suggestions("").len(), CATALOG.len());
        assert!(suggestions("something-else-entirely").is_empty());
    }
}
