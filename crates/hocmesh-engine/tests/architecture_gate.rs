//! What decides whether a model file can be run.
//!
//! The engine used to answer that with a list of architecture names, and the
//! list's own doc comment admitted the problem: a name is necessary and not
//! sufficient, because the list cannot see inside the file. It also cut the
//! other way. Most published models are the shape this engine implements --
//! RMS norm, SwiGLU, `attn_norm -> q,k,v -> rope -> attention -> out` -- and a
//! new one is published under a new name every few weeks. Refusing those by
//! name refused files this build computes correctly.
//!
//! So the name now answers exactly one question, the only one it is the
//! authority on: which pairs of elements a rotary embedding rotates together,
//! which is not written in a GGUF file at all. Everything else is decided by
//! what the file actually holds.
//!
//! These tests are about that split. The interesting one is the last: an
//! architecture admitted by name is still refused when its tensors say it is a
//! different model. If that ever passes, the override became a bypass.

use hocmesh_engine::fixture::Recipe;
use hocmesh_engine::{ModelConfig, WeightFile};

struct TestDir {
    path: std::path::PathBuf,
}

impl TestDir {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let ordinal = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("hocmesh-arch-{}-{ordinal}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create temp dir");
        TestDir { path }
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn model(recipe: Recipe) -> (TestDir, std::path::PathBuf) {
    let dir = TestDir::new();
    let path = dir.path.join("fixture.gguf");
    recipe.write(&path).expect("write fixture");
    (dir, path)
}

fn header(path: &std::path::Path) -> Vec<u8> {
    WeightFile::open(path).expect("open model").header
}

/// Set the override for the duration of `f`, then put it back.
///
/// Serialised, because an environment variable is process-wide and Rust runs
/// the tests in one binary on threads of a single process. Without the lock the
/// two tests below pass or fail depending on which thread reaches the variable
/// first -- which is worse than no test, and is exactly what happened when this
/// was written without one.
fn with_override<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let key = hocmesh_engine::ASSUME_ARCHITECTURE;
    let previous = std::env::var(key).ok();
    // SAFETY: single-threaded by construction -- see the comment above. The
    // whole crate's use of this variable is reading it in `rope_style_for`.
    unsafe {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
    let out = f();
    unsafe {
        match previous {
            Some(previous) => std::env::set_var(key, previous),
            None => std::env::remove_var(key),
        }
    }
    out
}

#[test]
fn the_name_answers_only_the_question_it_is_the_authority_on() {
    // A name this build knows loads with no ceremony.
    let (_dir, known) = model(Recipe {
        architecture: "qwen3".into(),
        qk_norm: true,
        ..Recipe::default()
    });
    let config = ModelConfig::from_header(&header(&known)).expect("a known architecture loads");
    assert_eq!(config.architecture, "qwen3");

    // A name it does not know is refused -- but the refusal says what is
    // missing and how to supply it, rather than only that the name is absent.
    // A model published last month under a new name is not a different shape.
    let (_dir, unknown) = model(Recipe {
        architecture: "granite".into(),
        ..Recipe::default()
    });
    let refusal = with_override(None, || {
        ModelConfig::from_header(&header(&unknown)).expect_err("an unknown name is refused")
    })
    .to_string();
    assert!(
        refusal.contains("rotary pairing") && refusal.contains("HOCMESH_ASSUME_ARCHITECTURE"),
        "the refusal does not name the one fact it is missing or the way to \
         supply it, so it reads as \"this model is unsupported\" when what it \
         means is \"tell me one thing\": {refusal}"
    );

    // Told which shape to read it as, the same file loads -- and reads as that
    // shape, not as one the operator has renamed.
    let assumed = with_override(Some("llama"), || {
        ModelConfig::from_header(&header(&unknown)).expect("an assumed architecture loads")
    });
    assert_eq!(
        assumed.architecture, "granite",
        "the override rewrote what the file declares; it names the shape to \
         read the file as, and the file's own claim is what the metadata keys \
         are prefixed with"
    );
    let plain = model(Recipe::default());
    let reference = ModelConfig::from_header(&header(&plain.1)).expect("llama loads");
    assert_eq!(
        assumed.rope_style, reference.rope_style,
        "assuming llama did not produce llama's rotary pairing, which is the \
         only thing the override is for"
    );

    // An override naming a shape this build also does not know is refused: it
    // is a shape to read the file as, not a way to say "run it anyway".
    let nonsense = with_override(Some("gemma3"), || {
        ModelConfig::from_header(&header(&unknown)).expect_err("an unknown override is refused")
    })
    .to_string();
    assert!(
        nonsense.contains("not an architecture this build knows either"),
        "unexpected refusal: {nonsense}"
    );
}

#[test]
fn the_override_names_a_shape_and_does_not_excuse_one() {
    // The load-bearing test. `granite` is refused by name; told to read it as
    // llama it gets past that gate -- and must still be refused, because the
    // file carries a tensor this build would not read and running it would
    // silently leave that term out of the forward pass.
    let (_dir, path) = model(Recipe {
        architecture: "granite".into(),
        unknown_tensor: true,
        ..Recipe::default()
    });

    // The header check alone passes: the offending tensor is in the directory,
    // not in the metadata, so this is exactly the case a name-based gate could
    // never have caught and a shape-based one has to.
    let refusal = with_override(Some("llama"), || {
        let config =
            ModelConfig::from_header(&header(&path)).expect("the header is a llama header");
        assert_eq!(config.block_count, Recipe::default().block_count);
        let mut file = WeightFile::open(&path).expect("open model");
        match hocmesh_engine::Stage::load(&mut file, 0..config.block_count) {
            Ok(_) => panic!(
                "a file carrying a term this build does not compute was loaded \
                 because the operator named a shape for it"
            ),
            Err(error) => error,
        }
    })
    .to_string();
    assert!(
        refusal.contains("does not read"),
        "the override let through a file this engine would compute wrongly, \
         which makes it a bypass rather than a declaration: {refusal}"
    );
}
