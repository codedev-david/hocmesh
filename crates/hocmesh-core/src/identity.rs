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

/// Prints the node id and nothing else.
///
/// Not derived, and that is the point: a derived `Debug` would put the signing
/// key into every log line, panic message and test failure that ever formatted
/// an identity, which is the sort of leak nobody writes on purpose.
impl std::fmt::Debug for NodeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeIdentity")
            .field("node_id", &self.node_id())
            .finish_non_exhaustive()
    }
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

    /// Opens the identity already in `home`, or `None` when there is none.
    ///
    /// Deliberately separate from `load_or_create`: a command that merely asks
    /// about an account must not bring one into existence. Otherwise running
    /// `hocmesh identity show` on the machine you are about to restore a backup
    /// onto would mint a key, and the restore would then have to force its way
    /// past the very key that looking had just created.
    pub fn load_existing(home: &Path, passphrase: Option<&str>) -> Result<Option<Self>> {
        let path = identity_path(home);
        if !path.exists() {
            return Ok(None);
        }
        let stored: StoredIdentity = serde_json::from_str(&fs::read_to_string(&path)?)
            .with_context(|| format!("reading {}", path.display()))?;
        Ok(Some(Self {
            signing_key: SigningKey::from_bytes(&open_secret(&stored, passphrase)?),
        }))
    }
    pub fn public_key_b64(&self) -> String {
        STANDARD_NO_PAD.encode(self.signing_key.verifying_key().to_bytes())
    }

    /// Seals this account into a file that can be carried to another machine.
    ///
    /// Refuses an empty passphrase rather than writing an unsealed one: a
    /// backup with no passphrase is just the key, and the whole reason this
    /// exists instead of "copy identity.json" is that the copy people actually
    /// make ends up in cloud sync, a chat message, or a USB stick in a drawer.
    pub fn export_backup(&self, passphrase: &str) -> Result<IdentityBackup> {
        if passphrase.is_empty() {
            bail!("an identity backup must be sealed; set {IDENTITY_EXPORT_PASSPHRASE_ENV}")
        }
        let sealed = seal(&self.signing_key.to_bytes(), passphrase)?
            .sealed
            .expect("seal always produces a sealed secret");
        Ok(IdentityBackup {
            format: IDENTITY_BACKUP_FORMAT.into(),
            version: 1,
            node_id: self.node_id(),
            public_key_b64: self.public_key_b64(),
            created_at: now_unix(),
            sealed,
        })
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

/// The environment variable that supplies the passphrase for a *backup file*.
///
/// Separate from [`IDENTITY_PASSPHRASE_ENV`] so the two decisions stay
/// separate: a node may run with its key unsealed on a machine only its owner
/// can reach and still refuse to let that key travel unencrypted. Falls back to
/// the node passphrase when it is not set, so one passphrase is enough for
/// anyone who wants it to be.
pub const IDENTITY_EXPORT_PASSPHRASE_ENV: &str = "HOCMESH_IDENTITY_EXPORT_PASSPHRASE";

/// What a backup file says it is, so a wrong file is refused rather than parsed.
pub const IDENTITY_BACKUP_FORMAT: &str = "hocmesh-identity-backup";

/// An account, sealed for travel.
///
/// The whole account is this key. There is no server holding a copy, no
/// recovery question and nobody with the authority to reissue it, because
/// nobody ever held it -- the network only ever saw signatures. So the file
/// that moves an account between machines is the one thing standing between an
/// owner and an unreachable balance, and it is deliberately not the raw key: a
/// backup is always sealed, whatever the node it came from was doing.
///
/// The node id and public key ride in the clear on purpose. They are public
/// values with nothing to protect, and having them readable is what lets you
/// see *whose* account a file holds -- and therefore whether it is the one you
/// meant -- before you type a passphrase into it.
#[derive(Debug, Serialize, Deserialize)]
pub struct IdentityBackup {
    pub format: String,
    pub version: u32,
    pub node_id: String,
    pub public_key_b64: String,
    pub created_at: i64,
    sealed: SealedSecret,
}

/// Writes a backup, refusing to silently replace one that is already there.
///
/// An accidental overwrite here is not a lost file, it is a lost account.
pub fn write_backup(path: &Path, backup: &IdentityBackup, force: bool) -> Result<()> {
    if path.exists() && !force {
        bail!(
            "{} already exists; pass --force to replace it",
            path.display()
        )
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    fs::write(path, serde_json::to_string_pretty(backup)?)
        .with_context(|| format!("writing {}", path.display()))?;
    // Sealed, so this is defence in depth rather than the only defence -- but a
    // key file left world-readable is still worth not doing.
    restrict_permissions(path, true)
}

/// Reads a backup file, checking it is one before believing anything in it.
pub fn read_backup(path: &Path) -> Result<IdentityBackup> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let backup: IdentityBackup = serde_json::from_str(&raw)
        .with_context(|| format!("{} is not a hocMESH identity backup", path.display()))?;
    if backup.format != IDENTITY_BACKUP_FORMAT {
        bail!(
            "{} is not a hocMESH identity backup (format {:?})",
            path.display(),
            backup.format
        )
    }
    if backup.version != 1 {
        bail!(
            "{} is a version {} backup and this build understands version 1",
            path.display(),
            backup.version
        )
    }
    Ok(backup)
}

/// Where a replaced identity is moved to, rather than destroyed.
fn displaced_path(path: &Path) -> PathBuf {
    path.with_extension(format!("json.replaced-{}", now_unix()))
}

/// Adopts a backup as this home's identity.
///
/// Three refusals, and each one exists because the failure it prevents cannot
/// be undone afterwards:
///
/// * A backup whose header claims a different account than the key inside it is
///   refused. Otherwise a file could be edited to *look* like the account you
///   meant to restore, and you would find out only when the network did not
///   know you.
/// * An import over a *different* identity needs `force`, because the key it
///   overwrites may be the only copy of an account with a balance on it.
/// * An import over an identity that will not open -- wrong passphrase, damaged
///   file -- also needs `force`, because a key that cannot be read is still a
///   key, and refusing to guess is the only safe reading of that situation.
///
/// Even under `force` the displaced file is renamed, never deleted.
pub fn import_backup(
    home: &Path,
    backup: &IdentityBackup,
    backup_passphrase: &str,
    local_passphrase: Option<&str>,
    force: bool,
) -> Result<NodeIdentity> {
    let key = unseal(&backup.sealed, backup_passphrase)?;
    let signing_key = SigningKey::from_bytes(&key);
    let public = signing_key.verifying_key().to_bytes();
    if node_id_from_public_key(&public) != backup.node_id
        || STANDARD_NO_PAD.encode(public) != backup.public_key_b64
    {
        bail!("this backup names an account its key does not produce; refusing to import it")
    }

    fs::create_dir_all(home).with_context(|| format!("creating {}", home.display()))?;
    let path = identity_path(home);
    if path.exists() {
        let opened = fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|raw| Ok(serde_json::from_str::<StoredIdentity>(&raw)?))
            .and_then(|s| open_secret(&s, local_passphrase))
            .map(|k| {
                node_id_from_public_key(&SigningKey::from_bytes(&k).verifying_key().to_bytes())
            });
        match opened {
            // Restoring the account that is already here: nothing to displace.
            Ok(id) if id == backup.node_id => {}
            Ok(id) if !force => bail!(
                "{} already holds account {id}; pass --force to replace it with {}",
                path.display(),
                backup.node_id
            ),
            Err(_) if !force => bail!(
                "{} holds an identity this command cannot open, so it cannot tell you what \
                 replacing it would cost; pass --force to replace it anyway",
                path.display()
            ),
            _ => {
                let moved = displaced_path(&path);
                fs::rename(&path, &moved).with_context(|| {
                    format!("moving the identity it is replacing to {}", moved.display())
                })?;
            }
        }
    }

    let stored = match local_passphrase {
        Some(p) => seal(&key, p)?,
        None => StoredIdentity {
            version: 1,
            secret_key_b64: Some(STANDARD_NO_PAD.encode(key)),
            sealed: None,
        },
    };
    write_identity(&path, &stored)?;
    Ok(NodeIdentity { signing_key })
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

    fn replaced_files(home: &Path) -> Vec<PathBuf> {
        fs::read_dir(home)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().contains(".replaced-"))
            .collect()
    }

    /// The new-laptop story, end to end. Nothing about the account is tied to
    /// the machine it was made on: the balance follows the key, so moving the
    /// key is the whole of moving the account.
    #[test]
    fn an_exported_account_arrives_intact_on_a_machine_that_never_had_it() {
        let old = home("export-old");
        let new = home("export-new");
        let laptop = NodeIdentity::load_or_create_sealed(&old, None).unwrap();

        let file = new.join("backup.json");
        write_backup(
            &file,
            &laptop.export_backup("a long passphrase").unwrap(),
            false,
        )
        .unwrap();
        let restored = import_backup(
            &new,
            &read_backup(&file).unwrap(),
            "a long passphrase",
            None,
            false,
        )
        .unwrap();

        assert_eq!(laptop.node_id(), restored.node_id());
        assert_eq!(laptop.public_key_b64(), restored.public_key_b64());
        // The signature is the part the network checks, so prove that, not just
        // that two strings match.
        assert_eq!(
            laptop.sign_bytes_b64(b"a shard result"),
            restored.sign_bytes_b64(b"a shard result")
        );
        // And it is really on disk, not just in the returned value.
        assert_eq!(
            NodeIdentity::load_or_create_sealed(&new, None)
                .unwrap()
                .node_id(),
            laptop.node_id()
        );
    }

    /// A backup travels. That is the entire point of it, and the reason it is
    /// sealed even when the node it came from was not.
    #[test]
    fn a_backup_never_carries_the_key_in_the_clear() {
        let h = home("backup-sealed");
        let id = NodeIdentity::load_or_create_sealed(&h, None).unwrap();
        assert!(
            fs::read_to_string(identity_path(&h))
                .unwrap()
                .contains("secret_key_b64"),
            "this test is only meaningful if the node itself was unsealed"
        );

        let file = h.join("backup.json");
        write_backup(&file, &id.export_backup("pw").unwrap(), false).unwrap();
        let raw = fs::read_to_string(&file).unwrap();

        assert!(!raw.contains("secret_key_b64"));
        assert!(!raw.contains(&STANDARD_NO_PAD.encode(id.signing_key.to_bytes())));
        assert!(raw.contains("argon2id"));
        // Readable in the clear on purpose: you can see whose account this is
        // before you type a passphrase into it.
        assert!(raw.contains(&id.node_id()));
    }

    #[test]
    fn an_unsealed_backup_cannot_be_asked_for() {
        let h = home("backup-nopw");
        let id = NodeIdentity::load_or_create_sealed(&h, None).unwrap();
        assert!(id.export_backup("").is_err());
    }

    #[test]
    fn a_backup_will_not_open_with_the_wrong_passphrase() {
        let h = home("backup-wrongpw");
        let new = home("backup-wrongpw-new");
        let id = NodeIdentity::load_or_create_sealed(&h, None).unwrap();
        let b = id.export_backup("right").unwrap();
        assert!(import_backup(&new, &b, "wrong", None, false).is_err());
        assert!(
            !identity_path(&new).exists(),
            "a failed import must leave nothing behind"
        );
    }

    /// The header is readable, so it is also editable. A file that claims to be
    /// somebody's account has to be refused, or the check it exists for -- "is
    /// this the account I meant?" -- would be answering with attacker-supplied
    /// text.
    #[test]
    fn a_backup_whose_header_disagrees_with_its_key_is_refused() {
        let h = home("backup-liar");
        let new = home("backup-liar-new");
        let id = NodeIdentity::load_or_create_sealed(&h, None).unwrap();
        let mut b = id.export_backup("pw").unwrap();
        b.node_id = "hocmesh:node:somebody-else".into();

        let err = import_backup(&new, &b, "pw", None, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not produce"), "{err}");
        assert!(!identity_path(&new).exists());
    }

    /// Losing a key loses an account permanently, so the destructive path is
    /// gated and even then keeps what it displaced.
    #[test]
    fn importing_over_another_account_needs_force_and_never_deletes_the_old_key() {
        let a = home("clobber-a");
        let b = home("clobber-b");
        let mine = NodeIdentity::load_or_create_sealed(&a, None).unwrap();
        let theirs = NodeIdentity::load_or_create_sealed(&b, None).unwrap();
        assert_ne!(mine.node_id(), theirs.node_id());
        let backup = mine.export_backup("pw").unwrap();

        let err = import_backup(&b, &backup, "pw", None, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--force"), "{err}");
        assert_eq!(
            NodeIdentity::load_or_create_sealed(&b, None)
                .unwrap()
                .node_id(),
            theirs.node_id(),
            "a refused import must not have touched the account that was there"
        );

        let forced = import_backup(&b, &backup, "pw", None, true).unwrap();
        assert_eq!(forced.node_id(), mine.node_id());

        let displaced = replaced_files(&b);
        assert_eq!(displaced.len(), 1, "the replaced key should still exist");
        let raw = fs::read_to_string(&displaced[0]).unwrap();
        let stored: StoredIdentity = serde_json::from_str(&raw).unwrap();
        let key = open_secret(&stored, None).unwrap();
        assert_eq!(
            node_id_from_public_key(&SigningKey::from_bytes(&key).verifying_key().to_bytes()),
            theirs.node_id()
        );
    }

    /// Restoring a backup onto the machine it came from is the most likely way
    /// anyone will ever use this -- "did my backup work?" -- and it must not
    /// demand `--force` or leave debris.
    #[test]
    fn restoring_the_account_already_here_asks_for_nothing() {
        let h = home("idempotent");
        let id = NodeIdentity::load_or_create_sealed(&h, None).unwrap();
        let b = id.export_backup("pw").unwrap();

        let again = import_backup(&h, &b, "pw", None, false).unwrap();
        assert_eq!(again.node_id(), id.node_id());
        assert!(replaced_files(&h).is_empty());
    }

    /// An identity that will not open is still an identity. Nothing can say
    /// what overwriting it would cost, so it refuses to guess.
    #[test]
    fn importing_over_an_unopenable_identity_needs_force() {
        let a = home("opaque-a");
        let b = home("opaque-b");
        let mine = NodeIdentity::load_or_create_sealed(&a, None).unwrap();
        NodeIdentity::load_or_create_sealed(&b, Some("their passphrase")).unwrap();
        let backup = mine.export_backup("pw").unwrap();

        let err = import_backup(&b, &backup, "pw", None, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot open"), "{err}");
        assert!(import_backup(&b, &backup, "pw", None, true).is_ok());
        assert_eq!(replaced_files(&b).len(), 1);
    }

    /// How the backup was sealed and how the node stores its key are separate
    /// decisions: you can restore a travelling backup straight into a sealed
    /// node without the key ever touching the disk unencrypted.
    #[test]
    fn a_restored_account_can_be_sealed_locally_with_a_different_passphrase() {
        let old = home("reseal-old");
        let new = home("reseal-new");
        let id = NodeIdentity::load_or_create_sealed(&old, None).unwrap();
        let b = id.export_backup("travel passphrase").unwrap();

        let restored = import_backup(
            &new,
            &b,
            "travel passphrase",
            Some("machine passphrase"),
            false,
        )
        .unwrap();
        assert_eq!(restored.node_id(), id.node_id());

        let raw = fs::read_to_string(identity_path(&new)).unwrap();
        assert!(!raw.contains("secret_key_b64"));
        assert!(NodeIdentity::load_or_create_sealed(&new, None).is_err());
        assert_eq!(
            NodeIdentity::load_or_create_sealed(&new, Some("machine passphrase"))
                .unwrap()
                .node_id(),
            id.node_id()
        );
    }

    #[test]
    fn a_file_that_is_not_a_backup_is_refused_before_anything_believes_it() {
        let h = home("notabackup");
        fs::create_dir_all(&h).unwrap();
        let f = h.join("x.json");

        fs::write(&f, "{\"hello\":1}").unwrap();
        assert!(read_backup(&f).is_err());

        let id = NodeIdentity::load_or_create_sealed(&h, None).unwrap();
        let mut b = id.export_backup("pw").unwrap();
        b.version = 2;
        let f2 = h.join("future.json");
        write_backup(&f2, &b, false).unwrap();
        let err = read_backup(&f2).unwrap_err().to_string();
        assert!(err.contains("version 2"), "{err}");
    }

    #[test]
    fn writing_a_backup_will_not_quietly_replace_one() {
        let h = home("nooverwrite");
        let id = NodeIdentity::load_or_create_sealed(&h, None).unwrap();
        let f = h.join("backup.json");
        write_backup(&f, &id.export_backup("pw").unwrap(), false).unwrap();
        assert!(write_backup(&f, &id.export_backup("pw").unwrap(), false).is_err());
        assert!(write_backup(&f, &id.export_backup("pw").unwrap(), true).is_ok());
    }
}
