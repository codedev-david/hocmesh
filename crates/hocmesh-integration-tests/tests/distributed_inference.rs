//! The claim the whole project is named for, checked against real processes.
//!
//! A model is cut into shards, each shard is written to disk *without* the
//! other shards' bytes, and a separate operating-system process is started for
//! each one. None of them can read a layer it does not hold: the bytes are not
//! there. They are chained over TCP, a prompt is pushed in at the head, and
//! generated tokens come back out.
//!
//! The result is then compared against the same model run whole in one process.
//! Not approximately — the comparison is a SHA-256 over the exact bit patterns
//! of every logit of every step. If a split model computed anything other than
//! precisely what the whole model computes, "run a model no single machine can
//! hold" would be a claim about a different and worse model.
//!
//! Everything here goes through the shipped `hocmesh` binary. A test that
//! called the library directly would prove the library works; this proves the
//! product does.

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::{
    env, fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// What `stage-run` prints, either way it is run.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct Generated {
    tokens: Vec<u32>,
    steps: usize,
    logits_sha256: String,
    argmax_per_step: Vec<u32>,
}

/// What `stage-sessions` prints.
#[derive(Debug, Clone, Deserialize)]
struct SessionReport {
    live: usize,
    peak: usize,
    #[allow(dead_code)]
    capacity: usize,
}

/// What `model-shard` prints.
#[derive(Debug, Clone, Deserialize)]
struct ShardReport {
    blocks: [u32; 2],
    chunks_kept: usize,
    chunks_total: usize,
    bytes_present: u64,
    bytes_total: u64,
}

/// The headline test: three processes, no whole copy anywhere, identical
/// output.
#[test]
fn a_model_split_across_processes_generates_exactly_what_one_process_generates() -> Result<()> {
    const BLOCKS: u32 = 6;
    const PROMPT: &str = "3,17,5";
    const NEW_TOKENS: &str = "6";

    let workspace = workspace_root()?;
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let whole = tmp.path.join("model.gguf");

    run_ok(
        node_command(&node, &home)
            .arg("model-fixture")
            .arg("--output")
            .arg(&whole)
            .arg("--blocks")
            .arg(BLOCKS.to_string()),
        "model-fixture",
    )?;

    // A small chunk size on purpose. The default is four megabytes, which would
    // put this entire model in one chunk and make every shard a whole copy --
    // the test would pass while proving nothing.
    run_ok(
        node_command(&node, &home)
            .arg("model-import")
            .arg(&whole)
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

    let ranges = [(0u32, 2u32), (2, 4), (4, 6)];
    let mut shards = Vec::new();
    for (start, end) in ranges {
        let path = tmp.path.join(format!("stage-{start}-{end}.gguf"));
        let report: ShardReport = serde_json::from_str(&run_capture(
            node_command(&node, &home)
                .arg("model-shard")
                .arg("--model-id")
                .arg("fixture")
                .arg("--blocks")
                .arg(format!("{start}..{end}"))
                .arg("--output")
                .arg(&path),
            "model-shard",
        )?)?;

        assert_eq!(report.blocks, [start, end]);
        assert!(
            report.bytes_present < report.bytes_total,
            "the shard for blocks {start}..{end} holds the entire model \
             ({} of {} bytes), so nothing here is being distributed",
            report.bytes_present,
            report.bytes_total
        );
        assert!(
            report.chunks_kept < report.chunks_total,
            "the shard for blocks {start}..{end} kept every chunk"
        );
        // The file is created at full length so tensors sit where the header
        // says; what makes it a shard is that most of it is a hole.
        assert_eq!(
            fs::metadata(&path)?.len(),
            report.bytes_total,
            "a shard must still be the model's declared length"
        );
        shards.push((start, end, path, report));
    }

    // No shard is even half the model, so no single process could be running it
    // whole no matter what it claimed.
    for (start, end, _, report) in &shards {
        let share = report.bytes_present as f64 / report.bytes_total as f64;
        assert!(
            share < 0.5,
            "blocks {start}..{end} hold {:.0}% of the model",
            share * 100.0
        );
    }

    let ports = [free_port()?, free_port()?, free_port()?];
    let mut servers = Vec::new();
    // Started from the tail backwards: a stage refuses to start without the
    // stage it hands off to being addressable, and starting them in this order
    // is what an operator would do anyway.
    for index in (0..shards.len()).rev() {
        let (start, end, path, _) = &shards[index];
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
        servers.push(ProcessGuard::spawn(&mut command)?);
        wait_for_port(ports[index])?;
    }

    let distributed: Generated = serde_json::from_str(&run_capture(
        node_command(&node, &home)
            .arg("stage-run")
            .arg("--head")
            .arg(format!("http://127.0.0.1:{}", ports[0]))
            .arg("--tokens")
            .arg(PROMPT)
            .arg("--max-new-tokens")
            .arg(NEW_TOKENS),
        "stage-run over the chain",
    )?)?;

    let single: Generated = serde_json::from_str(&run_capture(
        node_command(&node, &home)
            .arg("stage-run")
            .arg("--model")
            .arg(&whole)
            .arg("--tokens")
            .arg(PROMPT)
            .arg("--max-new-tokens")
            .arg(NEW_TOKENS),
        "stage-run in one process",
    )?)?;

    // The model has to be doing something, or every assertion below is true of
    // a chain of processes that all return the same constant.
    assert!(
        single
            .argmax_per_step
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 1,
        "the model picked the same token at every step: {:?}",
        single.argmax_per_step
    );
    assert!(
        single.tokens.len() > 3,
        "nothing was generated beyond the prompt"
    );

    assert_eq!(
        distributed.logits_sha256, single.logits_sha256,
        "three processes holding a third of the model each computed something \
         different from one process holding all of it\n  distributed: {:?}\n  whole:       {:?}",
        distributed.argmax_per_step, single.argmax_per_step
    );
    assert_eq!(distributed, single);

    drop(servers);
    Ok(())
}

/// The safety property underneath the headline one.
///
/// A shard is a file full of holes, and a hole reads back as zeros rather than
/// as an error. Zeros are a perfectly good weight matrix: a stage that read
/// them would answer confidently and wrongly, and nothing downstream could
/// tell. So a stage checks the bytes it holds against the blocks it was asked
/// to run, and refuses to start when they do not cover it.
#[test]
fn a_stage_refuses_to_run_layers_whose_bytes_it_does_not_have() -> Result<()> {
    let workspace = workspace_root()?;
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let whole = tmp.path.join("model.gguf");

    run_ok(
        node_command(&node, &home)
            .arg("model-fixture")
            .arg("--output")
            .arg(&whole)
            .arg("--blocks")
            .arg("6"),
        "model-fixture",
    )?;
    run_ok(
        node_command(&node, &home)
            .arg("model-import")
            .arg(&whole)
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

    let shard = tmp.path.join("stage-0-2.gguf");
    run_ok(
        node_command(&node, &home)
            .arg("model-shard")
            .arg("--model-id")
            .arg("fixture")
            .arg("--blocks")
            .arg("0..2")
            .arg("--output")
            .arg(&shard),
        "model-shard",
    )?;

    // Blocks 4..6 are not in this file. Serving them would mean reading holes.
    let output = node_command(&node, &home)
        .arg("stage-serve")
        .arg("--model")
        .arg(&shard)
        .arg("--blocks")
        .arg("4..6")
        .arg("--listen")
        .arg(format!("127.0.0.1:{}", free_port()?))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;
    assert!(
        !output.status.success(),
        "a stage served layers it does not hold"
    );
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(
        message.contains("is missing bytes"),
        "the refusal should say which bytes are absent, not merely fail: {message}"
    );

    Ok(())
}

/// Several sequences in flight over one chain, each one still exactly itself.
///
/// The serial pipeline had one attention cache per stage and no way to tell one
/// caller from another, so two prompts in flight at once would have written into
/// the same history -- and the failure would not have looked like a failure. Both
/// callers would have got fluent, plausible, wrong tokens back.
///
/// So this runs the prompts twice: once one after another, which is the
/// definition of correct, and once all at the same time. Every byte of every
/// logit has to match, and the results have to come back in the order the
/// prompts were given rather than the order they finished.
///
/// `peak` is what stops this passing for the wrong reason. A chain that handed
/// the sequences one cache between them, or ran them strictly one after
/// another, would produce identical output and prove nothing; peak is the most
/// caches that ever existed side by side, so it says the three histories were
/// really held apart rather than merely taking turns. It is checked on the last
/// stage as well as the first, because a session id that reached the head and
/// not the tail would leave the tail mixing the sequences back together --
/// which is exactly the bug this is here to catch.
#[test]
fn concurrent_sequences_over_one_chain_stay_separate_and_come_back_in_order() -> Result<()> {
    const BLOCKS: u32 = 6;
    const NEW_TOKENS: &str = "5";
    // Different lengths as well as different tokens: prompts that took the same
    // number of steps could interleave perfectly and still hide an off-by-one in
    // whose position is whose.
    const PROMPTS: [&str; 3] = ["3,17,5", "9", "2,7,7,1"];

    let workspace = workspace_root()?;
    build_node(&workspace)?;
    let node = workspace.join("target").join("debug").join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let home = tmp.path.join("home");
    let whole = tmp.path.join("model.gguf");

    run_ok(
        node_command(&node, &home)
            .arg("model-fixture")
            .arg("--output")
            .arg(&whole)
            .arg("--blocks")
            .arg(BLOCKS.to_string()),
        "model-fixture",
    )?;
    run_ok(
        node_command(&node, &home)
            .arg("model-import")
            .arg(&whole)
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

    let ranges = [(0u32, 2u32), (2, 4), (4, 6)];
    let mut shards = Vec::new();
    for (start, end) in ranges {
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
    let mut servers = Vec::new();
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
        servers.push(ProcessGuard::spawn(&mut command)?);
        wait_for_port(ports[index])?;
    }
    let head = format!("http://127.0.0.1:{}", ports[0]);
    let tail = format!("http://127.0.0.1:{}", ports[2]);

    // One at a time. This is the answer the concurrent run has to reproduce.
    let mut alone = Vec::new();
    for prompt in PROMPTS {
        alone.push(serde_json::from_str::<Generated>(&run_capture(
            node_command(&node, &home)
                .arg("stage-run")
                .arg("--head")
                .arg(&head)
                .arg("--tokens")
                .arg(prompt)
                .arg("--max-new-tokens")
                .arg(NEW_TOKENS),
            "stage-run over the chain",
        )?)?);
    }

    // Three prompts that generate the same thing would make every comparison
    // below true of a chain that ignored its input entirely.
    let distinct = alone
        .iter()
        .map(|run| run.logits_sha256.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        distinct.len(),
        PROMPTS.len(),
        "the prompts do not produce different output, so nothing below \
         distinguishes separate sequences from mixed-up ones: {distinct:?}"
    );

    // A caller that runs its prompts one after another never has two sequences
    // alive at once -- which is what makes the peak after the concurrent run
    // mean something.
    let serial: SessionReport = serde_json::from_str(&run_capture(
        node_command(&node, &home)
            .arg("stage-sessions")
            .arg("--stage")
            .arg(&head),
        "stage-sessions after the serial run",
    )?)?;
    assert_eq!(
        serial.peak, 1,
        "sequences overlapped during a run that sent them one at a time"
    );
    assert_eq!(
        serial.live, 0,
        "a finished sequence is still holding a cache: {serial:?}"
    );

    // All three at once.
    let mut command = node_command(&node, &home);
    command.arg("stage-run-many").arg("--head").arg(&head);
    for prompt in PROMPTS {
        command.arg("--tokens").arg(prompt);
    }
    let together: Vec<Generated> = serde_json::from_str(&run_capture(
        command.arg("--max-new-tokens").arg(NEW_TOKENS),
        "stage-run-many over the chain",
    )?)?;

    assert_eq!(
        together.len(),
        alone.len(),
        "a prompt went missing between the two runs"
    );
    for (index, (concurrent, sequential)) in together.iter().zip(&alone).enumerate() {
        assert_eq!(
            concurrent, sequential,
            "prompt {index} ({:?}) generated something different when it shared \
             the chain with the others\n  alone:    {:?}\n  together: {:?}",
            PROMPTS[index], sequential.argmax_per_step, concurrent.argmax_per_step
        );
    }

    // ...and they really did share it, at both ends of the chain.
    for (label, address) in [("head", &head), ("tail", &tail)] {
        let report: SessionReport = serde_json::from_str(&run_capture(
            node_command(&node, &home)
                .arg("stage-sessions")
                .arg("--stage")
                .arg(address),
            "stage-sessions after the concurrent run",
        )?)?;
        assert!(
            report.peak > 1,
            "the {label} stage never held more than one sequence at a time, so \
             the caches under test never coexisted and the comparison above              proves nothing: {report:?}"
        );
        assert_eq!(
            report.live, 0,
            "the {label} stage is still holding caches for finished sequences, \
             which is how a long-running chain reaches its cap: {report:?}"
        );
    }

    drop(servers);
    Ok(())
}

// -- harness -----------------------------------------------------------------

fn node_command(node: &Path, home: &Path) -> Command {
    let mut command = Command::new(node);
    command.arg("--home").arg(home);
    command
}

/// Build only the node binary. `--workspace --bins` would also build the
/// desktop app, and linking a webview would make a headless machine unable to
/// run a test that never opens a window.
fn build_node(workspace: &Path) -> Result<()> {
    run_ok(
        Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .arg("build")
            .arg("--ignore-rust-version")
            .arg("--bins")
            .arg("-p")
            .arg("hocmesh")
            .current_dir(workspace),
        "build the node binary",
    )
}

fn run_ok(command: &mut Command, label: &str) -> Result<()> {
    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("running {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn run_capture(command: &mut Command, label: &str) -> Result<String> {
    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("running {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn workspace_root() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("cannot resolve workspace root"))
}

fn free_port() -> Result<u16> {
    Ok(TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

fn wait_for_port(port: u16) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    bail!("no stage started listening on port {port}")
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Result<Self> {
        // A counter as well as a clock: Windows advances the system time in
        // ~15 ms steps, so two tests starting in the same tick would otherwise
        // share a directory and delete each other's model.
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let ordinal = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!("hocmesh-distributed-{suffix}-{ordinal}"));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct ProcessGuard {
    child: Option<Child>,
}

impl ProcessGuard {
    fn spawn(command: &mut Command) -> Result<Self> {
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("spawning a stage")?;
        Ok(Self { child: Some(child) })
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
