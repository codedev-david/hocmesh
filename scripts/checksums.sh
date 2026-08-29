#!/usr/bin/env bash
# Rewrite SHA256SUMS.txt over every tracked file except itself.
#
# .gitattributes checks the tree out with `eol=lf` everywhere, so a hash taken
# here is the hash taken on any other platform. The file lists tracked paths,
# which means it has to be run after staging whatever the release contains and
# before committing -- a checksum list that predates the commit it describes is
# worse than none, because it looks authoritative.
set -euo pipefail
cd "$(dirname "$0")/.."
git ls-files -z \
  | grep -zv '^SHA256SUMS\.txt$' \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  | sed 's/^\([0-9a-f]\{64\}\) \*/\1  .\//' >SHA256SUMS.txt
echo "SHA256SUMS.txt: $(wc -l <SHA256SUMS.txt) files"
