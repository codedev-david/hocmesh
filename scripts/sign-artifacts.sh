#!/usr/bin/env bash
#
# Sign the release artifacts in a directory.
#
# macOS: codesign the app bundles, the .dmg and the .pkg with a Developer ID,
# then notarise if credentials are present. Linux: detach-sign the checksum
# files with GPG, which is what lets somebody verify a .deb or an .AppImage
# they did not download from us.
#
# Signing proves the bytes came from the holder of the key and have not been
# altered since. It does not stop anyone copying an installer, and nothing can.
# See docs/DISTRIBUTION.md.
#
# With no key configured this says so and succeeds, so a fork or a local build
# still works. Set HOCMESH_SIGNING_REQUIRED=1 (CI does for tagged releases) to
# turn a missing key into a failure rather than a silently unsigned release.
set -euo pipefail

directory="${1:?usage: sign-artifacts.sh <directory>}"
required="${HOCMESH_SIGNING_REQUIRED:-0}"

fail_or_warn() {
  if [ "$required" = "1" ]; then
    echo "$1" >&2
    exit 1
  fi
  echo "$1"
}

case "$(uname -s)" in
  Darwin)
    if [ -z "${MACOS_CERT_P12_BASE64:-}" ] || [ -z "${MACOS_SIGN_IDENTITY:-}" ]; then
      fail_or_warn "No macOS signing identity configured; artifacts are UNSIGNED."
      exit 0
    fi

    keychain="$(mktemp -d)/hocmesh-signing.keychain-db"
    keychain_password="$(uuidgen)"
    certificate="$(mktemp).p12"
    # The keychain and the certificate both hold secrets; remove them whatever
    # happens, including on a signing failure.
    trap 'security delete-keychain "$keychain" 2>/dev/null || true; rm -f "$certificate"' EXIT

    printf '%s' "$MACOS_CERT_P12_BASE64" | base64 --decode > "$certificate"
    security create-keychain -p "$keychain_password" "$keychain"
    security set-keychain-settings -lut 21600 "$keychain"
    security unlock-keychain -p "$keychain_password" "$keychain"
    security import "$certificate" -k "$keychain" -P "${MACOS_CERT_PASSWORD:-}" \
      -T /usr/bin/codesign
    security set-key-partition-list -S apple-tool:,apple: -s -k "$keychain_password" "$keychain"
    security list-keychains -d user -s "$keychain" "$(security list-keychains -d user | tr -d ' "')"

    signed=0
    while IFS= read -r artifact; do
      echo "Signing $artifact"
      # --options runtime is the hardened runtime, which notarisation requires.
      codesign --force --deep --timestamp --options runtime \
        --keychain "$keychain" --sign "$MACOS_SIGN_IDENTITY" "$artifact"
      codesign --verify --strict --verbose=2 "$artifact"
      signed=$((signed + 1))
    done < <(find "$directory" -maxdepth 2 \( -name '*.dmg' -o -name '*.pkg' -o -name '*.app' \))

    if [ "$signed" -eq 0 ]; then
      echo "No macOS artifacts found under $directory" >&2
      exit 1
    fi
    echo "Signed $signed macOS artifact(s)."

    if [ -n "${MACOS_NOTARY_APPLE_ID:-}" ] && [ -n "${MACOS_NOTARY_PASSWORD:-}" ] \
       && [ -n "${MACOS_NOTARY_TEAM_ID:-}" ]; then
      while IFS= read -r artifact; do
        echo "Notarising $artifact"
        xcrun notarytool submit "$artifact" --wait \
          --apple-id "$MACOS_NOTARY_APPLE_ID" \
          --password "$MACOS_NOTARY_PASSWORD" \
          --team-id "$MACOS_NOTARY_TEAM_ID"
        # Stapling is what lets Gatekeeper approve the file offline.
        xcrun stapler staple "$artifact"
      done < <(find "$directory" -maxdepth 2 \( -name '*.dmg' -o -name '*.pkg' \))
    else
      fail_or_warn "Signed but NOT notarised; Gatekeeper will still warn on first run."
    fi
    ;;

  *)
    if [ -z "${GPG_PRIVATE_KEY:-}" ]; then
      fail_or_warn "No GPG key configured; checksums are UNSIGNED."
      exit 0
    fi

    home="$(mktemp -d)"
    trap 'rm -rf "$home"' EXIT
    export GNUPGHOME="$home"
    chmod 700 "$home"
    printf '%s' "$GPG_PRIVATE_KEY" | base64 --decode | gpg --batch --import

    signed=0
    while IFS= read -r checksum; do
      echo "Signing $checksum"
      gpg --batch --yes --pinentry-mode loopback \
        ${GPG_PASSPHRASE:+--passphrase "$GPG_PASSPHRASE"} \
        --armor --detach-sign "$checksum"
      gpg --verify "$checksum.asc" "$checksum"
      signed=$((signed + 1))
    done < <(find "$directory" -name '*.sha256')

    if [ "$signed" -eq 0 ]; then
      echo "No .sha256 files found under $directory; run the checksum step first" >&2
      exit 1
    fi
    echo "Signed $signed checksum file(s)."
    ;;
esac
