use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD};
use ed25519_dalek::{Signer, SigningKey};
use mesh_protocol::{AuthProof, canonical_auth_message, node_id_from_public_key, now_unix};
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize, Deserialize)]
struct StoredIdentity {
    version: u32,
    secret_key_b64: String,
}
#[derive(Clone)]
pub struct NodeIdentity {
    signing_key: SigningKey,
}
impl NodeIdentity {
    pub fn load_or_create(home: &Path) -> Result<Self> {
        fs::create_dir_all(home).with_context(|| format!("creating {}", home.display()))?;
        let path = identity_path(home);
        if path.exists() {
            let raw = fs::read_to_string(&path)?;
            let s: StoredIdentity = serde_json::from_str(&raw)?;
            if s.version != 1 {
                bail!("unsupported identity file version {}", s.version)
            }
            let b = STANDARD_NO_PAD.decode(s.secret_key_b64)?;
            let k: [u8; 32] = b
                .try_into()
                .map_err(|_| anyhow::anyhow!("identity secret key must be 32 bytes"))?;
            return Ok(Self {
                signing_key: SigningKey::from_bytes(&k),
            });
        }
        let signing_key = SigningKey::generate(&mut OsRng);
        let s = StoredIdentity {
            version: 1,
            secret_key_b64: STANDARD_NO_PAD.encode(signing_key.to_bytes()),
        };
        fs::write(&path, serde_json::to_string_pretty(&s)?)?;
        restrict_permissions(&path)?;
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
fn restrict_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut p = fs::metadata(path)?.permissions();
    p.set_mode(0o600);
    fs::set_permissions(path, p)?;
    Ok(())
}
#[cfg(not(unix))]
fn restrict_permissions(_: &Path) -> Result<()> {
    Ok(())
}
