#!/usr/bin/env bash
set -euo pipefail

# Builds the headless .rpm. package-linux.sh builds the .deb from the same three
# binaries and the same version; this exists because the machines most likely to
# lend spare capacity to a mesh are servers, and a large share of servers are
# not Debian.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <hocmesh-binary> <version> <output-dir> [architecture]" >&2
  exit 2
fi

binary_dir=$(cd "$(dirname "$1")" && pwd -P)
binary="$binary_dir/$(basename "$1")"
version=${2#v}
mkdir -p "$3"
output_dir=$(cd "$3" && pwd -P)
architecture=${4:-x86_64}
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

command -v rpmbuild >/dev/null 2>&1 || {
  echo "rpmbuild is not installed; install the rpm package" >&2
  exit 1
}

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

# An rpm version may not contain a hyphen -- the hyphen is what separates
# version from release in every rpm filename and query -- so a prerelease tag
# that is legal in Debian has to be spelled with a tilde here.
version=${version//-/\~}
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.+~][0-9A-Za-z.+~]+)*$ ]] || {
  echo "invalid RPM version: $version" >&2
  exit 1
}

# Accepts the Debian spelling too, so the release job can pass one architecture
# to both packaging scripts and neither has to know about the other.
case "$architecture" in
  amd64|x86_64) architecture=x86_64 ;;
  arm64|aarch64) architecture=aarch64 ;;
  *) echo "unsupported architecture: $architecture" >&2; exit 1 ;;
esac

cd "$repository_root"

stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
buildroot="$stage/buildroot"
install -d "$stage/SPECS" "$stage/RPMS" "$buildroot/usr/bin" "$buildroot/usr/share/doc/hocmesh"
install -m 0755 "$binary" "$buildroot/usr/bin/hocmesh"
install -m 0755 "$coordinator" "$buildroot/usr/bin/hocmesh-coordinator"
install -m 0755 "$validator" "$buildroot/usr/bin/hocmesh-validator"
install -m 0644 README.md LICENSE "$buildroot/usr/share/doc/hocmesh/"

sed \
  -e "s/@VERSION@/$version/g" \
  -e "s/@ARCHITECTURE@/$architecture/g" \
  packaging/linux/hocmesh.spec.in > "$stage/SPECS/hocmesh.spec"

rpmbuild \
  --define "_topdir $stage" \
  --define "_rpmdir $stage/RPMS" \
  --define "dist %{nil}" \
  --buildroot "$buildroot" \
  -bb "$stage/SPECS/hocmesh.spec" >&2

built=$(find "$stage/RPMS" -type f -name '*.rpm' -print -quit)
[[ -n "$built" ]] || { echo "rpmbuild produced no package" >&2; exit 1; }

artifact="$output_dir/hocmesh-${version}-1.${architecture}.rpm"
cp -f "$built" "$artifact"

# Opened rather than trusted, exactly as the .deb is: a package that installs a
# node without the coordinator and validator beside it looks perfectly healthy
# from the outside and cannot run a mesh.
contents=$(rpm -qlp "$artifact")
for expected in hocmesh hocmesh-coordinator hocmesh-validator; do
  grep -qE "^/usr/bin/$expected\$" <<<"$contents" || {
    echo "$expected is absent from $artifact" >&2
    exit 1
  }
done
rpm -qp --obsoletes "$artifact" | grep -qE '(^|[[:space:]])hoc-mesh-desktop([[:space:]]|$)' || {
  echo "$artifact does not obsolete hoc-mesh-desktop; the two installs would collide" >&2
  exit 1
}

echo "$artifact"
