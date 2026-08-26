#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <hocmesh-binary> <version> <output-dir>" >&2
  exit 2
fi

binary_dir=$(cd "$(dirname "$1")" && pwd -P)
binary="$binary_dir/$(basename "$1")"
version=${2#v}
mkdir -p "$3"
output_dir=$(cd "$3" && pwd -P)
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

[[ -f "$binary" ]] || { echo "hocmesh binary not found: $binary" >&2; exit 1; }
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)*$ ]] || {
  echo "invalid package version: $version" >&2
  exit 1
}
command -v pkgbuild >/dev/null || { echo "pkgbuild is required" >&2; exit 1; }

cd "$repository_root"

stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
install -d "$stage/root/usr/local/bin" "$stage/root/usr/local/share/doc/hocmesh"
install -m 0755 "$binary" "$stage/root/usr/local/bin/hocmesh"
install -m 0644 README.md LICENSE "$stage/root/usr/local/share/doc/hocmesh/"

artifact="$output_dir/hocmesh-${version}.pkg"
pkgbuild \
  --root "$stage/root" \
  --identifier org.hocmesh.compute.client \
  --version "$version" \
  --install-location / \
  "$artifact"
pkgutil --check-signature "$artifact" >/dev/null 2>&1 || true
pkgutil --payload-files "$artifact" > "$stage/payload-files.txt"
grep -q 'usr/local/bin/hocmesh$' "$stage/payload-files.txt"
echo "$artifact"
