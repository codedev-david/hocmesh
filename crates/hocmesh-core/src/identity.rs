use anyhow::{Context, Result, bail};
use argon2::Argon2;
use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD};
use chacha20poly1305::{
    XChaCha20Poly1305, XNonce,
    aead::{Aead, KeyInit},
};
use ed25519_dalek::{Signer, SigningKey};
use hocmesh_protocol::{AuthProof, canonical_auth_message, node_id_from_public_key, now_unix};
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// The environment variable that supplies the passphrase, when one is used.
///
/// An environment variable rather than a prompt because a validator has to come
/// back up unattended after a reboot: anything that needs a human present would
/// mean the network loses a seat every time a machine restarts.
pub const IDENTITY_PASSPHRASE_ENV: &str = "HOCMESH_IDENTITY_PASSPHRASE";

/// A signing key sealed with a key derived from a passphrase.
#[derive(Debug, Serialize, Deserialize)]
struct SealedSecret {
    kdf: String,
    salt_b64: String,
    nonce_b64: String,
    ciphertext_b64: String,
}

/// What is on disk. Version 1 holds the key in the clear; version 2 seals it.
///
/// Version 1 is still read, and deliberately. An identity is not a credential
/// that can be reissued: the node id the rest of the network knows a machine by
/// is derived from that key, so refusing to open an older file would evict
/// every node that had one.
#[derive(Debug, Serialize, Deserialize)]
struct StoredIdentity {
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    secret_key_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sealed: Option<SealedSecret>,
}

fn derive_key(passphrase: &str, salt: &[u8]) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), salt, &mut out)
        .map_err(|e| anyhow::anyhow!("deriving the identity key failed: {e}"))?;
    Ok(out)
}

fn seal(key: &[u8; 32], passphrase: &str) -> Result<StoredIdentity> {
    let mut salt = [0u8; 16];
    let mut nonce = [0u8; 24];
    OsRng.fill_bytes(&mut salt);
    OsRng.fill_bytes(&mut nonce);
    let derived = derive_key(passphrase, &salt)?;
    let ciphertext = XChaCha20Poly1305::new(&derived.into())
        .encrypt(XNonce::from_slice(&nonce), key.as_slice())
        .map_err(|_| anyhow::anyhow!("sealing the identity failed"))?;
    Ok(StoredIdentity {
        version: 2,
        secret_key_b64: None,
        sealed: Some(SealedSecret {
            kdf: "argon2id".into(),
            salt_b64: STANDARD_NO_PAD.encode(salt),
            nonce_b64: STANDARD_NO_PAD.encode(nonce),
            ciphertext_b64: STANDARD_NO_PAD.encode(ciphertext),
        }),
    })
}

fn unseal(s: &SealedSecret, passphrase: &str) -> Result<[u8; 32]> {
    if s.kdf != "argon2id" {
        bail!("unsupported identity key derivation {}", s.kdf)
    }
    let salt = STANDARD_NO_PAD.decode(&s.salt_b64)?;
    let nonce = STANDARD_NO_PAD.decode(&s.nonce_b64)?;
    if nonce.len() != 24 {
        bail!("sealed identity carries a malformed nonce")
    }
    let derived = derive_key(passphrase, &salt)?;
    let plain = XChaCha20Poly1305::new(&derived.into())
        .decrypt(
            XNonce::from_slice(&nonce),
            STANDARD_NO_PAD.decode(&s.ciphertext_b64)?.as_slice(),
        )
        .map_err(|_| {
            anyhow::anyhow!("the identity passphrase is wrong, or the file was altered")
        })?;
    key_from_bytes(&plain)
}

fn key_from_bytes(b: &[u8]) -> Result<[u8; 32]> {
    b.try_into()
        .map_err(|_| anyhow::anyhow!("identity secret key must be 32 bytes"))
}

fn write_identity(path: &Path, s: &StoredIdentity) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(s)?)?;
    restrict_permissions(path, s.sealed.is_some())
}

fn open_secret(s: &StoredIdentity, passphrase: Option<&str>) -> Result<[u8; 32]> {
    if let Some(sealed) = &s.sealed {
        if s.version != 2 {
            bail!("unsupported sealed identity file version {}", s.version)
        }
        let Some(p) = passphrase else {
            bail!("this identity is sealed; set {IDENTITY_PASSPHRASE_ENV} to open it")
        };
        return unseal(sealed, p);
    }
    if s.version != 1 {
        bail!("unsupported identity file version {}", s.version)
    }
    let Some(b64) = &s.secret_key_b64 else {
        bail!("identity file carries no key")
    };
    key_from_bytes(&STANDARD_NO_PAD.decode(b64)?)
}

#[derive(Clone)]
pub struct NodeIdentity {
    signing_key: SigningKey,
}
impl NodeIdentity {
    pub fn load_or_create(home: &Path) -> Result<Self> {
        let passphrase = std::env::var(IDENTITY_PASSPHRASE_ENV)
            .ok()
            .filter(|p| !p.is_empty());
        Self::load_or_create_sealed(home, passphrase.as_deref())
    }

    /// Loads or creates an identity, sealed with `passphrase` if one is given.
    ///
    /// The ledger itself is deliberately not encrypted: it is meant to be
    /// replayable by anyone. The signing key is the opposite - it is the one
    /// thing here that genuinely has to stay secret, because a validator's key
    /// is the whole quorum's security, so that is what gets sealed.
    pub fn load_or_create_sealed(home: &Path, passphrase: Option<&str>) -> Result<Self> {
        fs::create_dir_all(home).with_context(|| format!("creating {}", home.display()))?;
        let path = identity_path(home);
        if path.exists() {
            let stored: StoredIdentity = serde_json::from_str(&fs::read_to_string(&path)?)?;
            let key = open_secret(&stored, passphrase)?;
            // Turning the passphrase on seals what is already there. An
            // operator hardening a running node must not have to throw away the
            // node id the rest of the network already knows it by.
            if stored.sealed.is_none()
                && let Some(p) = passphrase
            {
                write_identity(&path, &seal(&key, p)?)?;
            }
            return Ok(Self {
                signing_key: SigningKey::from_bytes(&key),
            });
        }
        let signing_key = SigningKey::generate(&mut OsRng);
        let stored = match passphrase {
            Some(p) => seal(&signing_key.to_bytes(), p)?,
            None => StoredIdentity {
                version: 1,
                secret_key_b64: Some(STANDARD_NO_PAD.encode(signing_key.to_bytes())),
                sealed: None,
            },
        };
        write_identity(&path, &stored)?;
        Ok(Self { signing_key })
    }
    pub fn node_id(&self) -> String {
        node_id_from_public_key(&self.signing_key.verifying_key().to_bytes())
    }
    pub fn public_key_b64(&self) -> String {
        STANDARD_NO_PAD.encode(self.signing_key.verifying_key().to_bytes())
    }
    pub fn sign_bytes_b64(&self, bytes: &[u8]) -> String {
        STANDARD_NO_PAD.encode(self.signing_key.sign(bytes).to_bytes())
    }
    pub fn auth(&self, action: &str, body_hash: &str) -> AuthProof {
        let timestamp = now_unix();
        let node_id = self.node_id();
        let mut nonce = [0u8; 24];
        OsRng.fill_bytes(&mut nonce);
        let nonce_b64 = STANDARD_NO_PAD.encode(nonce);
        let msg = canonical_auth_message(action, &node_id, timestamp, &nonce_b64, body_hash);
        let sig = self.signing_key.sign(msg.as_bytes());
        AuthProof {
            node_id,
            timestamp,
            nonce_b64,
            signature_b64: STANDARD_NO_PAD.encode(sig.to_bytes()),
        }
    }
}
pub fn identity_path(home: &Path) -> PathBuf {
    home.join("identity.json")
}
#[cfg(unix)]
fn restrict_permissions(path: &Path, _sealed: bool) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut p = fs::metadata(path)?.permissions();
    p.set_mode(0o600);
    fs::set_permissions(path, p)?;
    Ok(())
}

/// There is no file-mode equivalent here, so an unsealed key is only as private
/// as the directory that holds it.
///
/// Said out loud rather than passed over in silence: this returned `Ok(())`
/// with no comment, which reads as "protected" to anyone skimming it.
#[cfg(not(unix))]
fn restrict_permissions(path: &Path, sealed: bool) -> Result<()> {
    if !sealed {
        eprintln!(
            "hocmesh: {} holds an unsealed signing key and this platform has no \
             file-mode enforcement. Set {IDENTITY_PASSPHRASE_ENV} to seal it.",
            path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home(name: &str) -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("hocmesh-identity-{name}-{n}"))
    }

    #[test]
    fn a_sealed_identity_reopens_with_its_passphrase() {
        let h = home("roundtrip");
        let a = NodeIdentity::load_or_create_sealed(&h, Some("correct horse")).unwrap();
        let b = NodeIdentity::load_or_create_sealed(&h, Some("correct horse")).unwrap();
        assert_eq!(a.node_id(), b.node_id());
        assert_eq!(a.public_key_b64(), b.public_key_b64());
    }

    #[test]
    fn a_sealed_identity_keeps_no_readable_key_on_disk() {
        let h = home("nokey");
        let id = NodeIdentity::load_or_create_sealed(&h, Some("pw")).unwrap();
        let raw = fs::read_to_string(identity_path(&h)).unwrap();
        assert!(!raw.contains("secret_key_b64"));
        assert!(!raw.contains(&STANDARD_NO_PAD.encode(id.signing_key.to_bytes())));
        assert!(raw.contains("argon2id"));
    }

    #[test]
    fn a_sealed_identity_will_not_open_without_the_right_passphrase() {
        let h = home("wrongpw");
        NodeIdentity::load_or_create_sealed(&h, Some("pw")).unwrap();
        assert!(NodeIdentity::load_or_create_sealed(&h, Some("not pw")).is_err());
        assert!(NodeIdentity::load_or_create_sealed(&h, None).is_err());
    }

    /// Hardening a node that is already in a validator set must not change what
    /// the rest of the network knows it as.
    #[test]
    fn sealing_an_existing_identity_preserves_its_node_id() {
        let h = home("upgrade");
        let before = NodeIdentity::load_or_create_sealed(&h, None).unwrap();
        assert!(
            fs::read_to_string(identity_path(&h))
                .unwrap()
                .contains("secret_key_b64")
        );

        let sealed = NodeIdentity::load_or_create_sealed(&h, Some("pw")).unwrap();
        assert_eq!(before.node_id(), sealed.node_id());
        assert!(
            !fs::read_to_string(identity_path(&h))
                .unwrap()
                .contains("secret_key_b64")
        );

        let reopened = NodeIdentity::load_or_create_sealed(&h, Some("pw")).unwrap();
        assert_eq!(before.node_id(), reopened.node_id());
    }
}
