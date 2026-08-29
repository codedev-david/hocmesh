//! Getting an inference runtime onto the machine, without turning hocMESH into
//! a package manager.
//!
//! hocMESH neither builds nor vendors llama.cpp. [`LlamaCppBackend`] shells out
//! to an executable, and until this module existed the operator had to find one
//! themselves and pass its path on every command. That was the honest default
//! for a project whose safety property is a hard-coded allow-list of workloads:
//! "fetch a binary off the internet and execute it" is exactly the thing that
//! allow-list exists to prevent.
//!
//! So this installer is deliberately crippled in three ways. It cannot be
//! pointed at an arbitrary URL. It cannot resolve `latest`, or ask any registry
//! what the current version is -- a name is a mutable pointer, and following one
//! would put the choice of what executes on this machine in someone else's
//! hands. And it knows the SHA-256 of every archive it is willing to unpack,
//! compiled in below; anything that does not hash to one of those constants is
//! deleted instead of extracted.
//!
//! Upgrading the runtime is therefore a source change, a review and a release:
//! the same bar every other executable path in this repository has to clear.
//! That is the point, not an oversight.
//!
//! The digests were read from the GitHub release API for the pinned tag. These
//! are the CPU builds; a CUDA or ROCm archive is a separate download with its
//! own runtime dependencies, and choosing one for the operator would be
//! guessing at their machine. `--runtime` still accepts any path, so an
//! operator who has built their own is never blocked by this.
//!
//! [`LlamaCppBackend`]: crate::LlamaCppBackend

use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

/// The upstream project. Used only to build download URLs; nothing resolves
/// names against it at runtime.
pub const RUNTIME_REPOSITORY: &str = "ggml-org/llama.cpp";

/// The one llama.cpp release this build of hocMESH will install.
pub const PINNED_BUILD: &str = "b10657";

/// The executable inside the archive that the llama.cpp backend drives.
pub const EXECUTABLE_STEM: &str = "llama-cli";

/// How a release archive is packed. Extraction itself lives in the node crate,
/// which is the only place that needs the zip and tar dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveKind {
    Zip,
    TarGz,
}

impl ArchiveKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ArchiveKind::Zip => "zip",
            ArchiveKind::TarGz => "tar.gz",
        }
    }
}

/// One pinned release archive, identified by its content rather than its name.
///
/// `os` and `arch` are compared against [`std::env::consts`], so they use that
/// vocabulary (`windows`/`macos`/`linux`, `x86_64`/`aarch64`) rather than the
/// upstream file-naming one (`win`/`ubuntu`, `x64`/`arm64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RuntimeAsset {
    pub os: &'static str,
    pub arch: &'static str,
    pub asset: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    pub archive: ArchiveKind,
}

impl RuntimeAsset {
    /// Where the archive is downloaded from.
    ///
    /// The tag is a constant and the filename is a constant, so this URL is
    /// fully determined by the source. There is no field that a response body
    /// could steer, which is what makes the pinning worth anything.
    pub fn url(&self) -> String {
        format!(
            "https://github.com/{RUNTIME_REPOSITORY}/releases/download/{PINNED_BUILD}/{}",
            self.asset
        )
    }
}

/// Every archive of [`PINNED_BUILD`] this build will accept.
///
/// Generated from the GitHub release API rather than transcribed by hand.
pub const RUNTIME_ASSETS: &[RuntimeAsset] = &[
    RuntimeAsset {
        os: "macos",
        arch: "aarch64",
        asset: "llama-b10657-bin-macos-arm64.tar.gz",
        sha256: "e7ed4148d41d0d6de297c551977da1dcf7ab7086e82101852ce72d3482a2ee38",
        size_bytes: 10996431,
        archive: ArchiveKind::TarGz,
    },
    RuntimeAsset {
        os: "macos",
        arch: "x86_64",
        asset: "llama-b10657-bin-macos-x64.tar.gz",
        sha256: "17a9c196377282fae4a0f3362877dc10dcc8719792f495126e45c6a93e8af5af",
        size_bytes: 11059645,
        archive: ArchiveKind::TarGz,
    },
    RuntimeAsset {
        os: "linux",
        arch: "aarch64",
        asset: "llama-b10657-bin-ubuntu-arm64.tar.gz",
        sha256: "3104192608a8253eb469c535a8c1f9bf7a8d7f621c641d470c5345ae8f63cce0",
        size_bytes: 13078731,
        archive: ArchiveKind::TarGz,
    },
    RuntimeAsset {
        os: "linux",
        arch: "x86_64",
        asset: "llama-b10657-bin-ubuntu-x64.tar.gz",
        sha256: "e94605eaa0dd4a494c4091eb4e228a6c4f4acb6411bb97af9d0e5d1efa2ad1b7",
        size_bytes: 16330015,
        archive: ArchiveKind::TarGz,
    },
    RuntimeAsset {
        os: "windows",
        arch: "aarch64",
        asset: "llama-b10657-bin-win-cpu-arm64.zip",
        sha256: "cfd77ea6043cc5bfec53abab6e83b5f61ca6b5408291553bffd23e744e6a2900",
        size_bytes: 11865522,
        archive: ArchiveKind::Zip,
    },
    RuntimeAsset {
        os: "windows",
        arch: "x86_64",
        asset: "llama-b10657-bin-win-cpu-x64.zip",
        sha256: "8b6e836d608e0ef0fd2c881fb76d3ea682f6d2b6644726958aac0e05607f98c8",
        size_bytes: 18093183,
        archive: ArchiveKind::Zip,
    },
];

/// The archive for an explicit platform, or `None` if this build has no digest
/// for it. Missing is not the same as unsupported: it means nobody pinned one.
pub fn asset_for(os: &str, arch: &str) -> Option<&'static RuntimeAsset> {
    RUNTIME_ASSETS
        .iter()
        .find(|asset| asset.os == os && asset.arch == arch)
}

/// The archive for the machine this binary is running on.
pub fn asset_for_host() -> Result<&'static RuntimeAsset> {
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    asset_for(os, arch).with_context(|| {
        format!(
            "no pinned llama.cpp build for {os}/{arch}; build llama.cpp yourself and pass \
             --runtime <path to {EXECUTABLE_STEM}>"
        )
    })
}

/// `<home>/runtime` -- one directory per installed build.
pub fn runtime_root(home: &Path) -> PathBuf {
    home.join("runtime")
}

/// Where [`PINNED_BUILD`] is unpacked. Keyed by tag so that a future upgrade
/// lands beside the old build instead of half-overwriting it.
pub fn runtime_dir(home: &Path) -> PathBuf {
    runtime_root(home).join(PINNED_BUILD)
}

/// The file recording which executable `infer` and `daemon` default to.
///
/// A pointer file rather than a fixed path, because the layout inside the
/// archive differs per platform and because an operator may point this at a
/// runtime they built themselves.
pub fn pointer_path(home: &Path) -> PathBuf {
    runtime_root(home).join("current.txt")
}

/// An absolute path without Windows' verbatim `\\?\` prefix.
///
/// `canonicalize` returns verbatim paths on Windows. They work when handed
/// straight back to the OS, but they are what an operator sees in
/// `runtime-status` and in `current.txt`, and a path that cannot be pasted back
/// into a shell is a path that makes the tool look broken. The prefix is only
/// dropped for ordinary drive paths -- a UNC path keeps it, because there the
/// prefix is load-bearing rather than decoration.
pub fn plain_absolute(path: &Path) -> PathBuf {
    let absolute = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let text = absolute.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => absolute,
    }
}

/// Record an executable as this home's default runtime.
pub fn record_runtime(home: &Path, executable: &Path) -> Result<()> {
    let root = runtime_root(home);
    fs::create_dir_all(&root)
        .with_context(|| format!("creating runtime directory {}", root.display()))?;
    let absolute = plain_absolute(executable);
    let pointer = pointer_path(home);
    fs::write(&pointer, format!("{}\n", absolute.display()))
        .with_context(|| format!("writing {}", pointer.display()))?;
    Ok(())
}

/// The runtime this home defaults to, if one exists and is still there.
///
/// Falls back to scanning the pinned directory, so that deleting `current.txt`
/// -- or restoring a home from a backup that skipped it -- degrades to a slower
/// answer rather than to "not installed".
pub fn installed_runtime(home: &Path) -> Option<PathBuf> {
    if let Ok(recorded) = fs::read_to_string(pointer_path(home)) {
        let trimmed = recorded.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    locate_executable(&runtime_dir(home))
        .ok()
        .map(|found| plain_absolute(&found))
}

/// The runtime to use, or an error that says how to get one.
pub fn require_runtime(home: &Path) -> Result<PathBuf> {
    installed_runtime(home).with_context(|| {
        format!(
            "no inference runtime installed in {}; run `hocmesh runtime-install` to fetch the \
             pinned llama.cpp {PINNED_BUILD} build, or pass --runtime <path> to use your own",
            home.display()
        )
    })
}

/// Find `llama-cli` anywhere under `dir`.
pub fn locate_executable(dir: &Path) -> Result<PathBuf> {
    locate_named(dir, EXECUTABLE_STEM)
}

/// Find the executable called `stem` anywhere under `dir`.
///
/// The archives do not agree on layout -- the Windows zip puts binaries at the
/// root, the Linux and macOS tarballs put them under `build/bin` -- so the
/// installed path is discovered rather than assumed. Anything that wants a
/// sibling tool (`llama-server`, `llama-quantize`) has the same problem and
/// must not solve it by guessing a layout: a flat join finds the Windows
/// binaries and silently finds nothing on the other two platforms. Depth is
/// bounded, because this walks a tree an archive has just written into.
pub fn locate_named(dir: &Path, stem: &str) -> Result<PathBuf> {
    const MAX_DEPTH: usize = 8;
    let mut frontier = vec![(dir.to_path_buf(), 0usize)];
    let mut found: Option<PathBuf> = None;
    while let Some((current, depth)) = frontier.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if depth < MAX_DEPTH {
                    frontier.push((path, depth + 1));
                }
                continue;
            }
            if !is_named(&path, stem) {
                continue;
            }
            // Prefer the shallowest match, so a stray vendored or debug copy
            // deeper in the archive cannot decide what the mesh executes.
            let better = match &found {
                None => true,
                Some(existing) => path.components().count() < existing.components().count(),
            };
            if better {
                found = Some(path);
            }
        }
    }
    found.with_context(|| format!("no {stem} executable found under {}", dir.display()))
}

/// Whether a path is the llama.cpp CLI, under either naming convention.
///
/// Both are matched on every host rather than only the local one, so a home
/// directory synced between machines still reports honestly what it holds.
fn is_named(path: &Path, stem: &str) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower == stem || lower == format!("{stem}.exe")
}

/// Make an extracted file executable on unix. A no-op elsewhere.
///
/// Both zip and tar carry a mode, but the zip crate only applies one when the
/// archive was written on unix -- and the archive that matters here was not.
/// Setting the bit explicitly is cheaper than reasoning about that per archive.
#[cfg(unix)]
pub fn mark_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path)
        .with_context(|| format!("reading permissions of {}", path.display()))?
        .permissions();
    permissions.set_mode(permissions.mode() | 0o755);
    fs::set_permissions(path, permissions)
        .with_context(|| format!("marking {} executable", path.display()))
}

#[cfg(not(unix))]
pub fn mark_executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// Whether a string is 64 lowercase hex characters.
///
/// Applied to the compiled-in table by a test rather than at runtime: a typo in
/// a constant should fail this repository's tests, not a user's download.
pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// The digest hocMESH expects for a platform, for reporting.
pub fn expected_digest(os: &str, arch: &str) -> Result<&'static str> {
    match asset_for(os, arch) {
        Some(asset) => Ok(asset.sha256),
        None => bail!("no pinned llama.cpp build for {os}/{arch}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hocmesh-runtime-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&path);
        path
    }

    /// A mistyped digest would surface to a user as a download that fails for
    /// no visible reason. Catch it here instead.
    #[test]
    fn every_pinned_digest_is_a_sha256() {
        for asset in RUNTIME_ASSETS {
            assert!(
                is_sha256_hex(asset.sha256),
                "{} has a malformed digest: {}",
                asset.asset,
                asset.sha256
            );
            assert!(asset.size_bytes > 0, "{} has no size", asset.asset);
        }
    }

    /// Two entries for one platform would make the choice order-dependent.
    #[test]
    fn each_platform_appears_once() {
        let mut seen = std::collections::BTreeSet::new();
        for asset in RUNTIME_ASSETS {
            assert!(
                seen.insert((asset.os, asset.arch)),
                "{}/{} is pinned twice",
                asset.os,
                asset.arch
            );
        }
    }

    /// The filename has to belong to the pinned tag, or the URL points at an
    /// asset that does not exist.
    #[test]
    fn every_asset_name_carries_the_pinned_tag() {
        for asset in RUNTIME_ASSETS {
            assert!(
                asset.asset.contains(PINNED_BUILD),
                "{} is not an asset of {PINNED_BUILD}",
                asset.asset
            );
            let expected = match asset.archive {
                ArchiveKind::Zip => ".zip",
                ArchiveKind::TarGz => ".tar.gz",
            };
            assert!(
                asset.asset.ends_with(expected),
                "{} is not a {expected} archive",
                asset.asset
            );
        }
    }

    /// The URL is built from constants only. If this ever interpolates
    /// something a server controls, the pinning is worthless.
    #[test]
    fn the_download_url_is_fully_determined_by_the_pin() {
        let asset = asset_for("linux", "x86_64").expect("linux/x86_64 is pinned");
        assert_eq!(
            asset.url(),
            format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{PINNED_BUILD}/{}",
                asset.asset
            )
        );
    }

    /// Whatever machine CI runs on has to be able to install.
    #[test]
    fn the_host_this_test_runs_on_is_pinned() {
        asset_for_host().expect("the test host has a pinned runtime build");
    }

    #[test]
    fn an_unknown_platform_is_missing_rather_than_wrong() {
        assert!(asset_for("plan9", "x86_64").is_none());
        assert!(expected_digest("plan9", "x86_64").is_err());
    }

    #[test]
    fn a_recorded_runtime_survives_a_round_trip() {
        let home = scratch("record");
        fs::create_dir_all(runtime_dir(&home)).expect("home");
        assert!(installed_runtime(&home).is_none());
        assert!(require_runtime(&home).is_err());

        let bin = runtime_dir(&home).join("build").join("bin");
        fs::create_dir_all(&bin).expect("bin directory");
        let executable = bin.join(EXECUTABLE_STEM);
        fs::write(&executable, b"#!/bin/sh\n").expect("fake runtime");

        // Found by scanning, before anything was recorded.
        let scanned = installed_runtime(&home).expect("scanning finds the executable");
        assert!(is_named(&scanned, EXECUTABLE_STEM));

        record_runtime(&home, &executable).expect("recording");
        assert!(is_named(
            &installed_runtime(&home).expect("the pointer resolves"),
            EXECUTABLE_STEM
        ));
        assert!(require_runtime(&home).is_ok());

        // A pointer at something that is gone falls back rather than lying.
        fs::write(pointer_path(&home), "/nonexistent/llama-cli\n").expect("stale pointer");
        assert!(installed_runtime(&home).is_some());

        fs::remove_file(&executable).expect("removing the runtime");
        assert!(installed_runtime(&home).is_none());
        assert!(require_runtime(&home).is_err());

        let _ = fs::remove_dir_all(&home);
    }

    /// The shallowest copy wins, so an archive that also ships a vendored or
    /// debug duplicate does not decide which binary the mesh executes.
    #[test]
    fn the_shallowest_executable_wins() {
        let root = scratch("depth");
        let deep = root.join("vendor").join("copies").join("bin");
        fs::create_dir_all(&deep).expect("deep directory");
        fs::write(deep.join(EXECUTABLE_STEM), b"deep").expect("deep copy");
        let shallow = root.join("bin");
        fs::create_dir_all(&shallow).expect("shallow directory");
        fs::write(shallow.join(EXECUTABLE_STEM), b"shallow").expect("shallow copy");

        assert_eq!(
            locate_executable(&root).expect("an executable"),
            shallow.join(EXECUTABLE_STEM)
        );

        let _ = fs::remove_dir_all(&root);
    }

    /// The sibling tools live wherever the archive put them, which is not the
    /// same place on every platform: the Windows zip extracts them at the root
    /// and the Linux and macOS tarballs put them under `build/bin`. Anything
    /// that joins a fixed path onto the runtime directory works on one of the
    /// three and silently finds nothing on the other two.
    #[test]
    fn a_sibling_tool_is_found_under_the_tarball_layout() {
        let root = scratch("sibling");
        let bin = root.join("build").join("bin");
        fs::create_dir_all(&bin).expect("bin directory");
        for stem in ["llama-cli", "llama-server", "llama-quantize"] {
            fs::write(bin.join(stem), b"binary").expect("binary");
        }

        for stem in ["llama-server", "llama-quantize"] {
            assert_eq!(
                locate_named(&root, stem).expect("a sibling tool"),
                bin.join(stem),
                "{stem} was not found under the tarball layout"
            );
            assert!(
                !root.join(stem).is_file(),
                "the fixture must not also place {stem} at the root, or this                  test would pass for a flat join too"
            );
        }

        let _ = fs::remove_dir_all(&root);
    }

    /// A Windows archive extracted on Linux, or the reverse, is still found.
    #[test]
    fn either_naming_convention_is_recognised() {
        let root = scratch("naming");
        fs::create_dir_all(&root).expect("directory");
        fs::write(root.join("llama-cli.exe"), b"windows").expect("windows binary");
        assert_eq!(
            locate_executable(&root).expect("an executable"),
            root.join("llama-cli.exe")
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_empty_directory_has_no_runtime() {
        let root = scratch("empty");
        fs::create_dir_all(&root).expect("directory");
        assert!(locate_executable(&root).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    /// Something that merely mentions the name is not the runtime.
    #[test]
    fn a_lookalike_filename_is_not_the_runtime() {
        let root = scratch("lookalike");
        fs::create_dir_all(&root).expect("directory");
        for name in ["llama-cli.txt", "llama-clip", "llama-cli-old", "README"] {
            fs::write(root.join(name), b"not the runtime").expect("decoy");
        }
        assert!(locate_executable(&root).is_err());
        let _ = fs::remove_dir_all(&root);
    }
}
