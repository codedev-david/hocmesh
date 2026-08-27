#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <hocmesh-binary> <version> <output-dir> [architecture]" >&2
  exit 2
fi

binary_dir=$(cd "$(dirname "$1")" && pwd -P)
binary="$binary_dir/$(basename "$1")"
version=${2#v}
mkdir -p "$3"
output_dir=$(cd "$3" && pwd -P)
architecture=${4:-amd64}
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
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.+~-][0-9A-Za-z.+~-]+)*$ ]] || {
  echo "invalid Debian version: $version" >&2
  exit 1
}
[[ "$architecture" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "invalid architecture" >&2; exit 1; }

cd "$repository_root"

stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
root="$stage/hocmesh_${version}_${architecture}"
install -d "$root/DEBIAN" "$root/usr/bin" "$root/usr/share/doc/hocmesh"
install -m 0755 "$binary" "$root/usr/bin/hocmesh"
install -m 0755 "$coordinator" "$root/usr/bin/hocmesh-coordinator"
install -m 0755 "$validator" "$root/usr/bin/hocmesh-validator"
install -m 0644 README.md LICENSE "$root/usr/share/doc/hocmesh/"

sed \
  -e "s/@VERSION@/$version/g" \
  -e "s/@ARCHITECTURE@/$architecture/g" \
  packaging/linux/control.in > "$root/DEBIAN/control"

artifact="$output_dir/hocmesh_${version}_${architecture}.deb"
dpkg-deb --root-owner-group --build "$root" "$artifact"
dpkg-deb --info "$artifact" >/dev/null
dpkg-deb --contents "$artifact" > "$stage/contents.txt"
grep -q 'usr/bin/hocmesh$' "$stage/contents.txt"
grep -q 'usr/bin/hocmesh-coordinator$' "$stage/contents.txt"
grep -q 'usr/bin/hocmesh-validator$' "$stage/contents.txt"
echo "$artifact"
