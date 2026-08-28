//! The property the whole system rests on.
//!
//! If a model cut into stages did not compute the same thing as the same model
//! run whole, then "run a model no single machine can hold" would be a
//! different and worse model, not the same one. Everything else — planning,
//! pricing, settlement, transport — is only worth having if this holds.
//!
//! These tests build a real GGUF file, run it whole, run it in pieces, and
//! compare. They compare bit patterns, not approximations: every stage's
//! arithmetic depends only on the activation it was handed and the weights it
//! holds, so there is no rounding difference to tolerate, and tolerating one
//! would hide a real divergence.

use hocmesh_engine::fixture::Recipe;
use hocmesh_engine::{Activation, Stage, WeightFile};

/// Run a whole model over a prompt and return the logits after each token.
fn run(path: &std::path::Path, cuts: &[u32], tokens: &[u32]) -> Vec<Vec<f32>> {
    let file = WeightFile::open(path).expect("open model");
    let config = hocmesh_engine::ModelConfig::from_header(&file.header).expect("config");
    let mut bounds = vec![0u32];
    bounds.extend_from_slice(cuts);
    bounds.push(config.block_count);

    let mut stages: Vec<Stage> = bounds
        .windows(2)
        .map(|pair| {
            let mut file = WeightFile::open(path).expect("open model");
            Stage::load(&mut file, pair[0]..pair[1]).expect("load stage")
        })
        .collect();

    let mut out = Vec::new();
    for (position, token) in tokens.iter().enumerate() {
        let mut activation = stages[0].embed(&[*token], position as u32).expect("embed");
        for stage in stages.iter_mut() {
            // Round-trip through the wire encoding on every hop, so the test
            // exercises the bytes a real pipeline would actually send.
            let framed = activation.to_bytes();
            let received = Activation::from_bytes(&framed).expect("decode activation");
            activation = stage.forward(&received).expect("forward");
        }
        let logits = stages
            .last()
            .expect("at least one stage")
            .logits(&activation)
            .expect("logits");
        // Cheap, and it guards the thing the whole file rests on: `inf == inf`
        // and two identical NaN bit patterns compare equal, so a model that had
        // saturated would make every comparison below pass without computing
        // anything.
        assert!(
            logits.iter().all(|value| value.is_finite()),
            "the model produced non-finite logits, which would compare equal however wrongly it was split"
        );
        out.push(logits);
    }
    out
}

/// A directory that removes itself, matching what the integration tests use
/// rather than pulling in a crate for six lines.
struct TestDir {
    path: std::path::PathBuf,
}

impl TestDir {
    fn new() -> Self {
        // A counter, not just a timestamp. Windows advances the system clock
        // in ~15 ms steps, so two tests starting in the same tick would get
        // the same name -- and then one of them would delete the other's model
        // out from under it on drop.
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let ordinal = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("hocmesh-engine-{}-{ordinal}", std::process::id()));
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

/// The headline: eight blocks in one process and the same eight blocks split
/// four ways produce identical logits, token for token and bit for bit.
#[test]
fn a_model_split_across_stages_computes_exactly_what_it_computes_whole() {
    let (_dir, path) = model(Recipe {
        block_count: 8,
        ..Recipe::default()
    });
    let prompt = [3u32, 17, 5, 41, 0, 22];

    let whole = run(&path, &[], &prompt);
    for cuts in [
        vec![4u32],                // two stages
        vec![2, 5],                // three, unevenly
        vec![1, 2, 3, 4, 5, 6, 7], // eight, one block each
        vec![7],                   // lopsided: seven blocks then one
    ] {
        let split = run(&path, &cuts, &prompt);
        assert_eq!(
            whole.len(),
            split.len(),
            "cuts {cuts:?} produced a different number of steps"
        );
        for (step, (a, b)) in whole.iter().zip(&split).enumerate() {
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "cuts {cuts:?} diverged at token {step}"
            );
        }
    }
}

/// The same, for every weight format the engine can execute. A codec that
/// decoded slightly differently on a boundary would show up here and nowhere
/// else.
#[test]
fn the_split_holds_for_every_supported_weight_format() {
    for kind in [
        hocmesh_engine::dequant::F32,
        hocmesh_engine::dequant::F16,
        hocmesh_engine::dequant::BF16,
        hocmesh_engine::dequant::Q8_0,
        hocmesh_engine::dequant::Q4_0,
        hocmesh_engine::dequant::Q4_1,
        hocmesh_engine::dequant::Q5_0,
        hocmesh_engine::dequant::Q5_1,
    ] {
        let (_dir, path) = model(Recipe {
            block_count: 4,
            weight_kind: kind,
            ..Recipe::default()
        });
        let prompt = [7u32, 1, 30];
        let whole = run(&path, &[], &prompt);
        let split = run(&path, &[1, 3], &prompt);
        assert_eq!(
            whole
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            split
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{} diverged when split",
            hocmesh_engine::dequant::type_name(kind)
        );
    }
}

/// Grouped-query attention shares each key/value head between several query
/// heads. Getting the mapping wrong still runs, so it is checked explicitly
/// rather than assumed to be covered by the default recipe.
#[test]
fn grouped_query_attention_splits_the_same_way() {
    for (heads, kv_heads) in [(4u32, 4u32), (4, 2), (8, 1)] {
        let (_dir, path) = model(Recipe {
            block_count: 4,
            embedding_length: 32,
            head_count: heads,
            head_count_kv: kv_heads,
            ..Recipe::default()
        });
        let prompt = [2u32, 9, 14, 3];
        assert_eq!(
            run(&path, &[], &prompt),
            run(&path, &[2], &prompt),
            "{heads} heads over {kv_heads} key/value heads diverged when split"
        );
    }
}

/// A stage must be told its position rather than counting its own steps.
/// Feeding the same prompt from a fresh cache twice must give the same answer;
/// feeding it at a different position must not.
#[test]
fn position_comes_from_the_activation_and_not_from_the_cache() {
    let (_dir, path) = model(Recipe::default());
    let mut file = WeightFile::open(&path).expect("open");
    let mut stage = Stage::load(&mut file, 0..4).expect("load");

    let first = stage.embed(&[11], 0).expect("embed");
    let once = stage.forward(&first).expect("forward");
    stage.reset();
    let again = stage.forward(&first).expect("forward");
    assert_eq!(once, again, "a reset stage did not repeat itself");

    stage.reset();
    // Position 0 must be present before position 1 can be: the cache is
    // append-only within a sequence, and a gap is an error rather than a hole.
    let ahead = stage.embed(&[11], 1).expect("embed");
    let error = stage.forward(&ahead).unwrap_err().to_string();
    assert!(error.contains("out of order"), "{error}");
}

/// A model with no `output.weight` ties its head to its embedding table. That
/// is loadable when one stage holds both ends and must be refused, clearly,
/// when it does not.
#[test]
fn a_tied_output_head_is_refused_when_the_ends_are_on_different_stages() {
    let (_dir, path) = model(Recipe {
        block_count: 4,
        separate_output_head: false,
        ..Recipe::default()
    });
    let mut file = WeightFile::open(&path).expect("open");
    assert!(
        Stage::load(&mut file, 0..4).is_ok(),
        "one stage holds both ends"
    );

    let mut file = WeightFile::open(&path).expect("open");
    let error = Stage::load(&mut file, 2..4)
        .err()
        .expect("a split tied head must be refused")
        .to_string();
    assert!(error.contains("ties its output head"), "{error}");
}

/// An empty range is a network hop that computes nothing, and a range past the
/// end is a planning bug. Both are refused at load, before any weight is read.
#[test]
fn an_impossible_layer_range_is_refused_at_load() {
    let (_dir, path) = model(Recipe::default());
    for (range, expected) in [(2..2, "computes nothing"), (3..9, "run past")] {
        let mut file = WeightFile::open(&path).expect("open");
        let error = Stage::load(&mut file, range.clone())
            .err()
            .expect("an impossible range must be refused")
            .to_string();
        assert!(error.contains(expected), "{range:?}: {error}");
    }
}

/// The comparisons above are only worth anything if the model they compare is
/// doing something. A network whose logits were all `inf`, or all the same
/// number, would pass every one of them while computing nothing.
#[test]
fn the_fixture_computes_something_worth_comparing() {
    let (_dir, path) = model(Recipe {
        block_count: 4,
        ..Recipe::default()
    });
    let steps = run(&path, &[], &[3u32, 17, 5, 41]);

    let mut winners = std::collections::BTreeSet::new();
    for logits in &steps {
        assert!(
            logits.iter().all(|value| value.is_finite()),
            "non-finite logits"
        );
        let spread = logits.iter().cloned().fold(f32::MIN, f32::max)
            - logits.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            spread > 1e-3,
            "every token is equally likely (spread {spread}), so argmax proves nothing"
        );
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .expect("non-empty")
            .0;
        winners.insert(best);
    }
    assert!(
        winners.len() > 1,
        "the same token won every position, so the sequence is not being read"
    );
}
