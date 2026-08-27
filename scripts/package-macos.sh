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
# Taken from the node binary's own directory so all three come from one build.
coordinator="$binary_dir/hocmesh-coordinator"
validator="$binary_dir/hocmesh-validator"
for companion in "$coordinator" "$validator"; do
  [[ -f "$companion" ]] || {
    echo "expected to package $companion alongside $binary; build the whole workspace first" >&2
    exit 1
  }
done
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
install -m 0755 "$coordinator" "$stage/root/usr/local/bin/hocmesh-coordinator"
install -m 0755 "$validator" "$stage/root/usr/local/bin/hocmesh-validator"
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
grep -q 'usr/local/bin/hocmesh-coordinator$' "$stage/payload-files.txt"
grep -q 'usr/local/bin/hocmesh-validator$' "$stage/payload-files.txt"
echo "$artifact"
