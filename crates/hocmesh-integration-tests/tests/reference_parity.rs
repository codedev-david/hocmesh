//! The forward pass, checked against an implementation nobody here wrote.
//!
//! Every other test in this repository compares hocMESH against hocMESH. The
//! split-versus-whole tests are bit-exact and worth having, but they share an
//! author with the thing they check: if the attention layout, the RoPE pairing
//! or a dequantiser were wrong, a split model would reproduce a whole model's
//! mistake exactly and every assertion would still pass. Those tests establish
//! that splitting changes nothing. What is established here is that the thing
//! being split was right to begin with.
//!
//! The reference is llama.cpp, driven as a server and fed token ids directly so
//! that no tokeniser sits between the two implementations.
//!
//! Two claims are made, and they are deliberately different in kind:
//!
//! * **f32 weights: the generated tokens must be identical.** Both sides do f32
//!   arithmetic over the same numbers, so there is nothing here they can
//!   legitimately disagree about.
//! * **Quantised weights: the *decoding* must be bit-identical**, checked
//!   against llama.cpp's own conversion back to f32. Generated tokens are not
//!   compared for quantised weights, and that is not a weakening of the claim
//!   -- `quantised_generation_is_not_compared_because_the_fixture_is_nearly_tied`
//!   records what makes such a comparison meaningless here.
//!
//! Nothing runs without llama.cpp present. Rather than fail on a machine that
//! has not got it, each test prints why it did nothing -- but CI installs the
//! runtime, so a silent skip there would be a hole rather than a courtesy.

use anyhow::{Context, Result, bail};
use hocmesh_gpu::runtime;
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    env, fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

/// What `stage-run` prints.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct Generated {
    tokens: Vec<u32>,
    steps: usize,
    logits_sha256: String,
    argmax_per_step: Vec<u32>,
}

/// The llama.cpp binaries these tests drive.
struct Reference {
    server: PathBuf,
    quantize: PathBuf,
}

/// Find an installed llama.cpp, or `None` on a machine that has not got one.
///
/// `runtime-install` puts it under the home directory it was given, in a
/// build-tagged folder; `HOCMESH_LLAMA_DIR` overrides that for anyone who
/// compiled their own.
fn reference(workspace: &Path) -> Option<Reference> {
    let mut roots = Vec::new();
    if let Ok(dir) = env::var("HOCMESH_LLAMA_DIR") {
        roots.push(PathBuf::from(dir));
    }
    if let Ok(entries) = fs::read_dir(workspace.join(".hocmesh").join("runtime")) {
        for entry in entries.flatten() {
            roots.push(entry.path());
        }
    }
    roots.into_iter().find_map(|root| {
        // Discovered, not joined. The Windows zip puts the binaries at the
        // root of the extracted tree and the Linux and macOS tarballs put them
        // under `build/bin`, so a flat join finds a runtime on one platform
        // and quietly finds none on the other two -- which read as a skip
        // until `HOCMESH_REQUIRE_REFERENCE` turned skips into failures.
        let server = runtime::locate_named(&root, "llama-server").ok()?;
        let quantize = runtime::locate_named(&root, "llama-quantize").ok()?;
        Some(Reference { server, quantize })
    })
}

/// The prompts, as token ids. There is no tokeniser in this test on purpose:
/// llama.cpp accepts an array of ids as a prompt, so both implementations are
/// handed the same numbers and a tokeniser difference cannot be mistaken for an
/// arithmetic one.
const PROMPTS: [&[u32]; 4] = [&[3], &[3, 17, 5], &[7, 7, 7, 7], &[40, 2, 88, 13, 61, 9]];
const NEW_TOKENS: usize = 12;

/// One process, f32 weights, against llama.cpp.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unsplit_f32_model_generates_exactly_what_llama_cpp_generates() -> Result<()> {
    let workspace = workspace_root()?;
    let Some(reference) = reference(&workspace) else {
        return skip();
    };
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let model = tmp.path.join("model.gguf");
    write_fixture(&node, &home, &model, "f32")?;

    let server = LlamaServer::start(&reference, &model).await?;

    for prompt in PROMPTS {
        let ours: Generated = serde_json::from_str(&run_capture(
            node_command(&node, &home)
                .arg("stage-run")
                .arg("--model")
                .arg(&model)
                .arg("--tokens")
                .arg(join(prompt))
                .arg("--max-new-tokens")
                .arg(NEW_TOKENS.to_string()),
            "stage-run",
        )?)?;
        let theirs = server.generate(prompt, NEW_TOKENS).await?;
        agreed_on_a_full_run(&ours, prompt, &theirs);

        assert_eq!(
            &ours.tokens[prompt.len()..],
            theirs.as_slice(),
            "hocmesh and llama.cpp disagree on what this model generates from \
             {prompt:?}. Both were given the same token ids and the same f32 \
             weights, so one of them has the arithmetic wrong -- the usual \
             suspects being the RoPE pairing, the grouped-query head mapping, \
             where the RMS-norm epsilon is applied, and the order of the two \
             SwiGLU branches."
        );
    }
    Ok(())
}

/// The headline claim, checked against an outside implementation: three
/// processes, none of which hold a whole model, agreeing token for token with
/// llama.cpp running that model whole.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_model_split_across_processes_generates_what_llama_cpp_generates() -> Result<()> {
    const BLOCKS: u32 = 6;
    let workspace = workspace_root()?;
    let Some(reference) = reference(&workspace) else {
        return skip();
    };
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let model = tmp.path.join("model.gguf");
    write_fixture(&node, &home, &model, "f32")?;

    // A small chunk size, or every shard is a whole copy and the test says
    // nothing about splitting.
    run_ok(
        node_command(&node, &home)
            .arg("model-import")
            .arg(&model)
            .arg("--model-id")
            .arg("fixture")
            .arg("--format")
            .arg("gguf")
            .arg("--architecture")
            .arg("llama")
            .arg("--chunk-size")
            .arg("32768"),
        "model-import",
    )?;

    let mut shards = Vec::new();
    for (start, end) in [(0u32, 2u32), (2, 4), (4, BLOCKS)] {
        let path = tmp.path.join(format!("stage-{start}-{end}.gguf"));
        run_ok(
            node_command(&node, &home)
                .arg("model-shard")
                .arg("--model-id")
                .arg("fixture")
                .arg("--blocks")
                .arg(format!("{start}..{end}"))
                .arg("--output")
                .arg(&path),
            "model-shard",
        )?;
        shards.push((start, end, path));
    }

    let ports = [free_port()?, free_port()?, free_port()?];
    let mut stages = Vec::new();
    // Tail first: a stage will not start until whatever it hands off to is
    // already addressable.
    for index in (0..shards.len()).rev() {
        let (start, end, path) = &shards[index];
        let mut command = node_command(&node, &home);
        command
            .arg("stage-serve")
            .arg("--model")
            .arg(path)
            .arg("--blocks")
            .arg(format!("{start}..{end}"))
            .arg("--listen")
            .arg(format!("127.0.0.1:{}", ports[index]));
        if index + 1 < shards.len() {
            command
                .arg("--next")
                .arg(format!("http://127.0.0.1:{}", ports[index + 1]));
        }
        stages.push(ProcessGuard::spawn(&mut command)?);
        wait_for_port(ports[index])?;
    }

    let server = LlamaServer::start(&reference, &model).await?;

    for prompt in PROMPTS {
        let split: Generated = serde_json::from_str(&run_capture(
            node_command(&node, &home)
                .arg("stage-run")
                .arg("--head")
                .arg(format!("http://127.0.0.1:{}", ports[0]))
                .arg("--tokens")
                .arg(join(prompt))
                .arg("--max-new-tokens")
                .arg(NEW_TOKENS.to_string()),
            "stage-run over the chain",
        )?)?;
        let theirs = server.generate(prompt, NEW_TOKENS).await?;
        agreed_on_a_full_run(&split, prompt, &theirs);

        assert_eq!(
            &split.tokens[prompt.len()..],
            theirs.as_slice(),
            "three processes, none of which hold the whole model, disagreed \
             with llama.cpp running it whole, on prompt {prompt:?}"
        );
    }
    Ok(())
}

/// Every quantised format must decode to exactly the numbers llama.cpp decodes.
///
/// llama.cpp quantises the fixture, then converts that result back to f32 with
/// its own decoder. That f32 file is the reference: our decoder is handed the
/// same quantised bytes and must reproduce those values exactly. A difference
/// means a misread block layout -- the fifth bit of `q5_0` and `q5_1` lives in
/// a bit-scattered field of its own, which is the easiest thing in the format
/// to get subtly wrong and the hardest to notice, because wrong weights do not
/// crash, they quietly answer differently.
///
/// Bit-exactness is asserted rather than closeness because decoding is pure
/// integer unpacking and one multiply; it has no rounding freedom, and a
/// tolerance would hide exactly the mistakes this is looking for.
#[test]
fn every_quantised_format_decodes_to_exactly_what_llama_cpp_decodes() -> Result<()> {
    let workspace = workspace_root()?;
    let Some(reference) = reference(&workspace) else {
        return skip();
    };
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let source = tmp.path.join("f32.gguf");
    write_fixture(&node, &home, &source, "f32")?;

    let mut checked = Vec::new();
    for format in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "bf16"] {
        let quantised = tmp.path.join(format!("{format}.gguf"));
        let restored = tmp.path.join(format!("{format}-back.gguf"));

        // A format this build will not write is skipped rather than failed:
        // which formats llama.cpp supports is its business, not ours.
        let written = Command::new(&reference.quantize)
            .arg(&source)
            .arg(&quantised)
            .arg(format)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?
            .success();
        if !written {
            eprintln!("llama.cpp will not write {format} here; skipping it");
            continue;
        }
        run_ok(
            Command::new(&reference.quantize)
                .arg("--allow-requantize")
                .arg(&quantised)
                .arg(&restored)
                .arg("f32"),
            "llama-quantize back to f32",
        )?;

        let compared = compare_tensors(&quantised, &restored)
            .with_context(|| format!("comparing {format} against llama.cpp's own decoding"))?;
        assert!(
            !compared.is_empty(),
            "no tensor in the {format} file was actually stored as {format}, \
             so this comparison checked nothing"
        );
        checked.push((format, kind_names(&compared)));
    }

    // If llama.cpp wrote none of them the loop above proved nothing, and a test
    // that silently proves nothing is worse than one that fails.
    assert!(
        checked.len() >= 4,
        "only {} formats could be checked against llama.cpp: {checked:?}",
        checked.len()
    );
    eprintln!("decoded identically to llama.cpp: {checked:?}");
    Ok(())
}

/// Why generated tokens are compared for f32 weights but not for quantised
/// ones -- written as a test so the reasoning is checked rather than trusted.
///
/// For quantised weights llama.cpp does not decode to f32 and then multiply; it
/// quantises the *activations* as well and takes integer dot products. That is
/// a different arithmetic path, so small differences are expected and are
/// nobody's bug. They only matter here because this fixture's weights are
/// random, which leaves its logits nearly tied -- the top two candidates sit a
/// few hundredths of a nat apart -- so an argmax over them is settled by
/// rounding. Comparing quantised generation would be measuring that noise.
///
/// What is asserted instead is the property that makes the distinction
/// necessary: quantisation changes the logits, so the weights demonstrably
/// reach the arithmetic, and the exactness claim above is not being made about
/// a decoder whose output nothing consumes.
#[test]
fn quantised_generation_is_not_compared_because_the_fixture_is_nearly_tied() -> Result<()> {
    let workspace = workspace_root()?;
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let mut runs = Vec::new();
    for weights in ["f32", "q4_0"] {
        let model = tmp.path.join(format!("{weights}.gguf"));
        write_fixture(&node, &home, &model, weights)?;
        let run: Generated = serde_json::from_str(&run_capture(
            node_command(&node, &home)
                .arg("stage-run")
                .arg("--model")
                .arg(&model)
                .arg("--tokens")
                .arg("3,17,5")
                .arg("--max-new-tokens")
                .arg("4"),
            "stage-run",
        )?)?;
        runs.push(run);
    }

    assert_ne!(
        runs[0].logits_sha256, runs[1].logits_sha256,
        "four-bit weights produced logits bit-identical to f32, which would \
         mean the quantised weights are not reaching the arithmetic at all"
    );
    Ok(())
}

/// Decode every quantised tensor of `quantised` with our decoder and compare it
/// against the same tensor in `restored`, which llama.cpp decoded itself.
///
/// Returns the set of GGML type codes actually decoded, not a count. Asking
/// llama.cpp for `q4_k_m` does not get a file full of Q4_K -- it picks a type
/// per tensor, and on a tensor it will not quantise it picks something else
/// entirely. A count cannot tell the difference between "Q4_K decoded
/// correctly" and "nothing in this file was Q4_K", and the second one is how a
/// decoder test passes while checking nothing.
fn compare_tensors(quantised: &Path, restored: &Path) -> Result<BTreeSet<u32>> {
    use hocmesh_model::gguf;

    let left = fs::read(quantised)?;
    let right = fs::read(restored)?;
    let left_dir = gguf::tensor_directory(&left)?.context("no tensor directory")?;
    let right_dir = gguf::tensor_directory(&right)?.context("no tensor directory")?;

    let mut compared = BTreeSet::new();
    for tensor in &left_dir.tensors {
        // Tensors llama.cpp left as f32 -- the norms, and anything too small to
        // be worth quantising -- say nothing about a decoder.
        if tensor.kind == hocmesh_engine::dequant::F32 {
            continue;
        }
        let Some(reference) = right_dir.tensors.iter().find(|t| t.name == tensor.name) else {
            bail!("{} is missing from llama.cpp's f32 conversion", tensor.name);
        };
        let count = tensor.element_count().context("bad shape")? as usize;
        let len = tensor.data_len().context("unknown type")? as usize;

        let start = (left_dir.data_start + tensor.offset) as usize;
        let mut ours = vec![0f32; count];
        hocmesh_engine::dequant::dequantize(tensor.kind, &left[start..start + len], &mut ours)?;

        let start = (right_dir.data_start + reference.offset) as usize;
        let theirs = right[start..start + count * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));

        for (index, (ours, theirs)) in ours.iter().zip(theirs).enumerate() {
            assert_eq!(
                ours.to_bits(),
                theirs.to_bits(),
                "{} element {index}: we decode {ours}, llama.cpp decodes {theirs}",
                tensor.name
            );
        }
        compared.insert(tensor.kind);
    }
    Ok(compared)
}

/// The same claim for the k-quants, which is where most real model files live.
///
/// Split from the plain formats because it needs a differently shaped fixture.
/// A k-quant super-block is 256 elements wide and llama.cpp will not store a
/// row that is not a multiple of that -- it silently picks another type
/// instead. The narrow fixture the rest of this file uses would therefore
/// produce a file with no k-quant tensor in it at all, and a decoder test that
/// decodes nothing passes.
///
/// That is also why the assertion at the end is about *types* rather than
/// tensors: asking for `q4_k_m` gets a mixture chosen per tensor, so the only
/// way to know Q2_K through Q6_K were each exercised is to record what was
/// actually decoded and require all five.
///
/// Bit-exactness again, for the same reason: unpacking six-bit scales that
/// straddle byte boundaries has no rounding freedom, and a tolerance would
/// hide precisely the off-by-one-field mistakes this exists to catch.
#[test]
fn every_k_quant_decodes_to_exactly_what_llama_cpp_decodes() -> Result<()> {
    let workspace = workspace_root()?;
    let Some(reference) = reference(&workspace) else {
        return skip();
    };
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let source = tmp.path.join("f32.gguf");
    write_wide_fixture(&node, &home, &source)?;

    let mut checked = Vec::new();
    let mut decoded = BTreeSet::new();
    for format in ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k"] {
        let quantised = tmp.path.join(format!("{format}.gguf"));
        let restored = tmp.path.join(format!("{format}-back.gguf"));

        let written = Command::new(&reference.quantize)
            .arg(&source)
            .arg(&quantised)
            .arg(format)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?
            .success();
        if !written {
            eprintln!("llama.cpp will not write {format} here; skipping it");
            continue;
        }
        run_ok(
            Command::new(&reference.quantize)
                .arg("--allow-requantize")
                .arg(&quantised)
                .arg(&restored)
                .arg("f32"),
            "llama-quantize back to f32",
        )?;

        let compared = compare_tensors(&quantised, &restored)
            .with_context(|| format!("comparing {format} against llama.cpp's own decoding"))?;
        assert!(
            !compared.is_empty(),
            "no tensor in the {format} file was stored quantised at all"
        );
        checked.push((format, kind_names(&compared)));
        decoded.extend(compared);
    }

    // The point of the assertion. Anything missing here is a decoder this file
    // did not check, however many tensors it compared.
    let wanted = BTreeSet::from([
        hocmesh_engine::dequant::Q2_K,
        hocmesh_engine::dequant::Q3_K,
        hocmesh_engine::dequant::Q4_K,
        hocmesh_engine::dequant::Q5_K,
        hocmesh_engine::dequant::Q6_K,
    ]);
    let missing = wanted.difference(&decoded).copied().collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "no tensor was stored as {:?}, so those decoders are unchecked; \
         llama.cpp wrote {checked:?}",
        kind_names(&missing.into_iter().collect())
    );
    eprintln!("decoded identically to llama.cpp: {checked:?}");
    Ok(())
}

/// A k-quant file written by llama.cpp loads and generates, end to end.
///
/// The decoder being bit-exact is a claim about `dequantize`. It says nothing
/// about whether a real `q4_k_m` file -- the form most published models
/// actually ship in -- gets as far as the decoder, because a second gate reads
/// the type code at load time and used to refuse these outright. A decoder
/// nothing can reach is not support.
///
/// So this runs the model, and asserts the logits differ from the same
/// fixture in f32: identical logits would mean the quantised weights never
/// reached the arithmetic, which is exactly how a load path that silently
/// substituted something else would look.
#[test]
fn a_k_quant_model_from_llama_cpp_loads_and_generates() -> Result<()> {
    let workspace = workspace_root()?;
    let Some(reference) = reference(&workspace) else {
        return skip();
    };
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let source = tmp.path.join("f32.gguf");
    write_wide_fixture(&node, &home, &source)?;

    let quantised = tmp.path.join("q4_k_m.gguf");
    run_ok(
        Command::new(&reference.quantize)
            .arg(&source)
            .arg(&quantised)
            .arg("q4_k_m"),
        "llama-quantize to q4_k_m",
    )?;

    // A mixture, as llama.cpp chooses it: this file holds Q4_K and Q6_K
    // tensors, so running it exercises two decoders through the load path
    // rather than one.
    let kinds = stored_kinds(&quantised)?;
    assert!(
        kinds.contains(&hocmesh_engine::dequant::Q4_K),
        "llama.cpp wrote no Q4_K tensor, so this runs an ordinary file: \
         {:?}",
        kind_names(&kinds)
    );

    let mut runs = Vec::new();
    for model in [&source, &quantised] {
        runs.push(serde_json::from_str::<Generated>(&run_capture(
            node_command(&node, &home)
                .arg("stage-run")
                .arg("--model")
                .arg(model)
                .arg("--tokens")
                .arg("3,17,5")
                .arg("--max-new-tokens")
                .arg("4"),
            "stage-run",
        )?)?);
    }

    for run in &runs {
        assert_eq!(
            run.tokens.len(),
            7,
            "a three-token prompt plus four new ones is seven: {:?}",
            run.tokens
        );
    }
    assert_ne!(
        runs[0].logits_sha256, runs[1].logits_sha256,
        "a q4_k_m file produced logits bit-identical to f32, which would mean \
         its weights are not reaching the arithmetic at all"
    );
    eprintln!("ran a q4_k_m file holding {:?}", kind_names(&kinds));
    Ok(())
}

/// Which GGML types a file actually stores, ignoring the f32 norms.
fn stored_kinds(model: &Path) -> Result<BTreeSet<u32>> {
    let bytes = fs::read(model)?;
    let directory =
        hocmesh_model::gguf::tensor_directory(&bytes)?.context("no tensor directory")?;
    Ok(directory
        .tensors
        .iter()
        .map(|tensor| tensor.kind)
        .filter(|kind| *kind != hocmesh_engine::dequant::F32)
        .collect())
}

/// GGML type codes as their names, so a failure says `Q4_K` and not `12`.
fn kind_names(kinds: &BTreeSet<u32>) -> Vec<String> {
    kinds
        .iter()
        .map(|kind| hocmesh_engine::dequant::type_name(*kind).to_string())
        .collect()
}

/// A fixture whose every row is a multiple of the 256-element k-quant
/// super-block, so llama.cpp stores k-quants rather than falling back.
fn write_wide_fixture(node: &Path, home: &Path, output: &Path) -> Result<()> {
    run_ok(
        node_command(node, home)
            .arg("model-fixture")
            .arg("--output")
            .arg(output)
            .arg("--weights")
            .arg("f32")
            .arg("--family")
            .arg("llama")
            // 256 wide, 512 deep in the feed-forward, 512 vocabulary entries:
            // every tensor's leading dimension is then 256 or 512.
            .arg("--embedding-length")
            .arg("256")
            .arg("--feed-forward-length")
            .arg("512")
            .arg("--vocab")
            .arg("512"),
        "model-fixture (k-quant width)",
    )
}

/// Both implementations must have generated the whole run before their
/// answers are compared.
///
/// Two empty lists are equal, and so are two lists that both stopped after
/// one token. Either would let this file pass while comparing nothing -- if
/// `return_tokens` were ever dropped from llama.cpp's response, or a sampler
/// setting made it stop early, the assertion that follows would go quietly
/// vacuous rather than fail. Length is checked first so that cannot happen.
fn agreed_on_a_full_run(ours: &Generated, prompt: &[u32], theirs: &[u32]) {
    assert_eq!(
        ours.tokens.len(),
        prompt.len() + NEW_TOKENS,
        "stage-run returned {} tokens in total; a {}-token prompt plus {NEW_TOKENS} new ones is {}",
        ours.tokens.len(),
        prompt.len(),
        prompt.len() + NEW_TOKENS
    );
    assert_eq!(
        theirs.len(),
        NEW_TOKENS,
        "llama.cpp returned {} tokens rather than {NEW_TOKENS}; comparing against that would prove nothing",
        theirs.len()
    );
}

/// Say why nothing was checked, and pass -- unless this machine was told the
/// comparison is not optional, in which case fail.
///
/// A developer without llama.cpp installed should not have a red suite. CI is
/// the other case: there the runtime is installed on purpose, so a skip would
/// mean the one test that checks this engine against an outside implementation
/// had quietly stopped running. `HOCMESH_REQUIRE_REFERENCE=1` turns the
/// courtesy off, in the same way `HOCMESH_SIGNING_REQUIRED` turns the
/// artifact-signing skip into a build failure.
fn skip() -> Result<()> {
    let advice = "Run `cargo run -p hocmesh -- runtime-install` from the workspace root, or point HOCMESH_LLAMA_DIR at a build of it.";
    if env::var("HOCMESH_REQUIRE_REFERENCE").is_ok_and(|value| value != "0") {
        bail!("HOCMESH_REQUIRE_REFERENCE is set but no llama.cpp was found. {advice}");
    }
    eprintln!("skipping: no llama.cpp here. {advice}");
    Ok(())
}

fn join(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// One real checkpoint shape per architecture this engine claims to run.
///
/// The architecture list is a promise, and until this existed the promise was
/// checked against one shape: llama's. Every other name on it was believed on
/// the strength of sharing a loader, which is exactly the reasoning the list's
/// own doc comment warns against -- `qwen2` keeps projection biases and
/// `qwen3` normalises each head's query and key before rotating, and a build
/// that ignores either loads the file, runs, and generates fluent text that is
/// not what the model says. There is no output to inspect that distinguishes
/// that from a correct run, so the only check is another implementation.
///
/// The head width is *not* varied here, and that is a limit worth stating
/// rather than hiding. Four of Qwen3's six sizes declare
/// `attention.key_length` because `embedding_length / head_count` gives the
/// wrong answer for them, so it is an axis that deserves a reference check --
/// but llama.cpp b10657 does not apply that key to these files. Measured:
/// with `qwen3.attention.key_length = 16` present and byte-correct in the
/// header, it asked for `blk.0.attn_k.weight` of `[64, 16]`, which is
/// `(embedding_length / head_count) * head_count_kv`; raising `head_count_kv`
/// from 2 to 4 moved its expectation to `[64, 32]`, so the divided width, not
/// the declared one, is what it used, and the same happens under `llama`. A
/// fixture built on that key therefore never loads there and could only be
/// made to pass by not testing it. The axis stays covered by
/// `split_matches_whole`, which runs a decoupled key and value width through
/// whole and split execution, and by the config unit tests that read the two
/// keys -- neither of which is a second implementation, which is why this
/// says so out loud.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_architecture_generates_what_llama_cpp_generates() -> Result<()> {
    let workspace = workspace_root()?;
    let Some(reference) = reference(&workspace) else {
        return skip();
    };
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    for (family, head_dim) in [("llama", None), ("qwen2", None), ("qwen3", None::<u32>)] {
        let tmp = TestDir::new()?;
        let home = tmp.path.join("home");
        let model = tmp.path.join("model.gguf");
        write_family_fixture(&node, &home, &model, "f32", family, head_dim)?;

        let server = LlamaServer::start(&reference, &model).await?;
        for prompt in PROMPTS {
            let ours: Generated = serde_json::from_str(&run_capture(
                node_command(&node, &home)
                    .arg("stage-run")
                    .arg("--model")
                    .arg(&model)
                    .arg("--tokens")
                    .arg(join(prompt))
                    .arg("--max-new-tokens")
                    .arg(NEW_TOKENS.to_string()),
                "stage-run",
            )?)?;
            let theirs = server.generate(prompt, NEW_TOKENS).await?;
            agreed_on_a_full_run(&ours, prompt, &theirs);

            assert_eq!(
                &ours.tokens[prompt.len()..],
                theirs.as_slice(),
                "hocmesh and llama.cpp disagree on what this {family} model generates \
                 from {prompt:?}. The terms that differ between these families are the \
                 per-head query and key norms, the projection biases, the rotary pairing \
                 and the declared head width."
            );
        }
    }
    Ok(())
}

fn write_fixture(node: &Path, home: &Path, output: &Path, weights: &str) -> Result<()> {
    write_family_fixture(node, home, output, weights, "llama", None)
}

/// Write a fixture shaped like one particular family.
fn write_family_fixture(
    node: &Path,
    home: &Path,
    output: &Path,
    weights: &str,
    family: &str,
    head_dim: Option<u32>,
) -> Result<()> {
    let mut command = node_command(node, home);
    command
        .arg("model-fixture")
        .arg("--output")
        .arg(output)
        .arg("--weights")
        .arg(weights)
        .arg("--family")
        .arg(family);
    if let Some(width) = head_dim {
        command.arg("--head-dim").arg(width.to_string());
    }
    run_ok(&mut command, "model-fixture")
}

/// A llama.cpp server, stopped when this is dropped.
struct LlamaServer {
    _process: ProcessGuard,
    port: u16,
    http: reqwest::Client,
}

impl LlamaServer {
    async fn start(reference: &Reference, model: &Path) -> Result<Self> {
        let port = free_port()?;
        let mut command = Command::new(&reference.server);
        command
            .arg("-m")
            .arg(model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("-c")
            .arg("512")
            // CPU, and a full-precision KV cache. This compares arithmetic; a
            // half-precision cache would introduce a difference that has
            // nothing to do with either implementation being right.
            .arg("-ngl")
            .arg("0")
            .arg("--cache-type-k")
            .arg("f32")
            .arg("--cache-type-v")
            .arg("f32")
            .arg("--no-warmup");
        let process = ProcessGuard::spawn(&mut command)?;

        let http = reqwest::Client::new();
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if Instant::now() > deadline {
                bail!("llama-server never became ready on port {port}");
            }
            if let Ok(response) = http
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
                && response.status().is_success()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        Ok(Self {
            _process: process,
            port,
            http,
        })
    }

    /// Greedy generation from raw token ids.
    async fn generate(&self, prompt: &[u32], predict: usize) -> Result<Vec<u32>> {
        #[derive(Deserialize)]
        struct Completion {
            tokens: Vec<u32>,
        }
        let body = serde_json::json!({
            "prompt": prompt,
            "n_predict": predict,
            "temperature": 0.0,
            "top_k": 1,
            "samplers": ["top_k"],
            "return_tokens": true,
            // Every prompt is evaluated from nothing. Otherwise a cached prefix
            // left by the previous prompt decides part of this answer, and the
            // comparison stops being about this prompt.
            "cache_prompt": false,
        });
        let completion = self
            .http
            .post(format!("http://127.0.0.1:{}/completion", self.port))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<Completion>()
            .await?;
        Ok(completion.tokens)
    }
}

fn node_command(node: &Path, home: &Path) -> Command {
    let mut command = Command::new(node);
    command.arg("--home").arg(home);
    command
}

fn build_node(workspace: &Path) -> Result<()> {
    run_ok(
        Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .current_dir(workspace)
            .arg("build")
            .arg("-p")
            .arg("hocmesh"),
        "cargo build -p hocmesh",
    )
}

fn run_ok(command: &mut Command, label: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("running {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed ({}): {}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn run_capture(command: &mut Command, label: &str) -> Result<String> {
    let output = command
        .output()
        .with_context(|| format!("running {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn workspace_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("locating the workspace root")
}

fn free_port() -> Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn wait_for_port(port: u16) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    bail!("nothing came up on port {port}")
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// A scratch directory, removed when the test ends.
///
/// The name carries a counter as well as the process id: Windows advances its
/// clock in about 15ms steps, so two tests starting in the same tick would
/// otherwise share a directory and delete each other's files.
struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Result<Self> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let path = env::temp_dir().join(format!(
            "hocmesh-parity-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// A child process, killed when this is dropped.
struct ProcessGuard {
    child: Child,
}

impl ProcessGuard {
    fn spawn(command: &mut Command) -> Result<Self> {
        let child = command
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        Ok(Self { child })
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
