#!/usr/bin/env bash
set -euo pipefail

# Builds the desktop installers -- a .dmg on macOS, a .deb and an .AppImage on
# Linux -- each carrying the window *and* the node it supervises.
#
# The node is passed in rather than built here so that the app and the daemon
# it will start always come from one build, the same rule package-linux.sh
# enforces for the three command line binaries.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <hocmesh-binary> <version> <output-dir> [tauri-cli-version]" >&2
  exit 2
fi

binary_dir=$(cd "$(dirname "$1")" && pwd -P)
binary="$binary_dir/$(basename "$1")"
version=${2#v}
mkdir -p "$3"
output_dir=$(cd "$3" && pwd -P)
tauri_cli_version=${4:-2.11.4}
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
desktop_dir="$repository_root/crates/hocmesh-desktop"

[[ -f "$binary" ]] || { echo "hocmesh binary not found: $binary" >&2; exit 1; }
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "installer version must be numeric major.minor.patch: $version" >&2
  exit 1
}

# The bundler takes its version from tauri.conf.json, so a tree whose config
# disagrees with its VERSION file would ship an installer named for one
# release and reporting another.
configured_version=$(sed -n 's/^[[:space:]]*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  "$desktop_dir/tauri.conf.json" | head -1)
repository_version=$(tr -d '[:space:]' < "$repository_root/VERSION")
[[ "$configured_version" == "$repository_version" ]] || {
  echo "tauri.conf.json says $configured_version but VERSION says $repository_version" >&2
  exit 1
}

# Tauri names a sidecar for the triple it was built for and strips that suffix
# when it bundles, which is what lands the node next to the app as plain
# hocmesh -- the first place supervisor::candidate_paths looks.
host_triple=$(rustc -vV | sed -n 's/^host: //p')
[[ -n "$host_triple" ]] || { echo "could not read the host target triple from rustc" >&2; exit 1; }
mkdir -p "$desktop_dir/binaries"
install -m 0755 "$binary" "$desktop_dir/binaries/hocmesh-$host_triple"

# Both of these are sent to stderr so that the only thing this script writes
# to stdout is the artifact paths a caller wants to capture.
command -v cargo-tauri >/dev/null 2>&1 ||
  cargo install tauri-cli --version "$tauri_cli_version" --locked >&2

(cd "$desktop_dir" && cargo tauri build --config tauri.bundle.json -- --locked) >&2

bundle_dir="$repository_root/target/release/bundle"

# The newest match, so a stale bundle from an earlier version is never the one
# that ships.
newest() {
  find "$bundle_dir/$1" -maxdepth 1 -type f -name "$2" -print0 2>/dev/null |
    xargs -0 --no-run-if-empty ls -t 2>/dev/null | head -1
}

copy_bundle() {
  local subdirectory=$1 pattern=$2 artifact_name=$3 source
  source=$(newest "$subdirectory" "$pattern")
  [[ -n "$source" ]] || { echo "the bundler produced no $pattern in $subdirectory" >&2; exit 1; }
  [[ -s "$source" ]] || { echo "$source is empty" >&2; exit 1; }
  cp -f "$source" "$output_dir/$artifact_name"
  printf '%s\n' "$output_dir/$artifact_name"
}

case "$(uname -s)" in
  Darwin)
    case "$(uname -m)" in
      arm64|aarch64) arch=aarch64 ;;
      *) arch=x86_64 ;;
    esac
    dmg=$(copy_bundle dmg "*.dmg" "hocmesh-desktop-$version-$arch.dmg")

    # An installer that lays down the window without the daemon would install
    # an app that cannot start anything, and it would look perfectly fine from
    # the outside. Mount it and check for both.
    dmg_mount=$(mktemp -d)
    trap 'hdiutil detach "$dmg_mount" >/dev/null 2>&1 || true; rm -rf -- "$dmg_mount"' EXIT
    hdiutil attach -nobrowse -readonly -mountpoint "$dmg_mount" "$dmg" >/dev/null
    for expected in hocmesh hocmesh-desktop; do
      # Assigned rather than piped into grep: grep -q exits on the first match,
      # which hands the producer a SIGPIPE that pipefail then reports as a
      # failure of the very check that just succeeded.
      found=$(find "$dmg_mount" -type f -name "$expected" -print -quit)
      [[ -n "$found" ]] || { echo "$expected is absent from $dmg" >&2; exit 1; }
    done
    hdiutil detach "$dmg_mount" >/dev/null

    artifacts=("$dmg")
    ;;
  Linux)
    case "$(uname -m)" in
      aarch64|arm64) arch=aarch64; debian_arch=arm64 ;;
      *) arch=x86_64; debian_arch=amd64 ;;
    esac
    deb=$(copy_bundle deb "*.deb" "hocmesh-desktop_${version}_${debian_arch}.deb")
    appimage=$(copy_bundle appimage "*.AppImage" "hocmesh-desktop-$version-$arch.AppImage")

    # An installer that lays down the window without the daemon would install
    # an app that cannot start anything, and it would look perfectly fine from
    # the outside. Open both and check.
    dpkg-deb --info "$deb" >/dev/null
    # Listed once and searched twice, rather than piped into grep -q: grep
    # exits on the first match, which hands dpkg-deb a SIGPIPE that pipefail
    # then reports as a failure of the very check that just succeeded.
    deb_contents=$(dpkg-deb --contents "$deb")
    for expected in hocmesh hocmesh-desktop; do
      grep -qE "/$expected\$" <<<"$deb_contents" || {
        echo "$expected is absent from $deb" >&2
        exit 1
      }
    done

    chmod +x "$appimage"
    appimage_stage=$(mktemp -d)
    trap 'rm -rf -- "$appimage_stage"' EXIT
    (cd "$appimage_stage" && "$appimage" --appimage-extract >/dev/null)
    for expected in hocmesh hocmesh-desktop; do
      found=$(find "$appimage_stage/squashfs-root" -type f -name "$expected" -print -quit)
      [[ -n "$found" ]] || { echo "$expected is absent from $appimage" >&2; exit 1; }
    done
    artifacts=("$deb" "$appimage")
    ;;
  *)
    echo "unsupported platform: use scripts/package-desktop.ps1 on Windows" >&2
    exit 1
    ;;
esac

printf '%s\n' "${artifacts[@]}"
