use anyhow::{Context, Result, anyhow, bail};
use mesh_core::{compute::execute_work, identity::NodeIdentity};
use mesh_ledger::types::{LedgerHead, ValidatorSet};
use mesh_protocol::{
    BalanceResponse, ErrorResponse, JobStatusResponse, NodeCapabilities, PollRequest, PollResponse,
    RegisterRequest, ResultRequest, SubmitJobRequest, SubmitJobResponse, WorkAssignment, WorkSpec,
    empty_body_hash, job_id_from_auth, register_body_hash, result_body_hash, submit_body_hash,
};
use reqwest::Client;
use serde_json::json;
use std::{
    env, fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn four_validator_quorum_earn_spend_recover_and_audit() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("mesh-validator"));
    let coordinator_bin = bin_dir.join(exe("mesh-coordinator"));

    let tmp = TestDir::new()?;
    let http = Client::new();

    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let mut members = Vec::new();
    let mut validator_homes = Vec::new();
    let mut validator_dbs = Vec::new();
    for (index, port) in validator_ports.iter().enumerate() {
        let home = tmp.path.join(format!("validator-{index}"));
        let output = Command::new(&validator_bin)
            .arg("id")
            .arg("--home")
            .arg(&home)
            .output()
            .with_context(|| format!("creating validator identity {index}"))?;
        if !output.status.success() {
            bail!(
                "validator id failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let stdout = String::from_utf8(output.stdout)?;
        let validator_id = parse_value(&stdout, "validator_id=")?;
        let public_key_b64 = parse_value(&stdout, "public_key_b64=")?;
        members.push(json!({
            "validator_id": validator_id,
            "url": format!("http://127.0.0.1:{port}"),
            "public_key_b64": public_key_b64,
        }));
        validator_homes.push(home);
        validator_dbs.push(tmp.path.join(format!("validator-{index}.db")));
    }

    let validators_path = tmp.path.join("validators.json");
    fs::write(
        &validators_path,
        serde_json::to_vec_pretty(&json!({
            "threshold": 3,
            "community_issuance_limit_mcu": 1_000_000_000i64,
            "members": members,
        }))?,
    )?;
    let set: ValidatorSet = serde_json::from_slice(&fs::read(&validators_path)?)?;

    let mut validators = Vec::new();
    for index in 0..4 {
        validators.push(ProcessGuard::spawn(
            Command::new(&validator_bin)
                .arg("serve")
                .arg("--home")
                .arg(&validator_homes[index])
                .arg("--db")
                .arg(&validator_dbs[index])
                .arg("--listen")
                .arg(format!("127.0.0.1:{}", validator_ports[index]))
                .arg("--validators")
                .arg(&validators_path),
        )?);
        wait_health(&http, validator_ports[index]).await?;
    }

    let coordinator_db = tmp.path.join("coordinator.db");
    run_ok(
        Command::new(&coordinator_bin)
            .arg("seed")
            .arg("--db")
            .arg(&coordinator_db)
            .arg("--validators")
            .arg(&validators_path)
            .arg("--start")
            .arg("2")
            .arg("--end")
            .arg("2000")
            .arg("--shards")
            .arg("4"),
        "seed community job",
    )?;

    let coordinator_port = free_port()?;
    let _coordinator = ProcessGuard::spawn(
        Command::new(&coordinator_bin)
            .arg("serve")
            .arg("--db")
            .arg(&coordinator_db)
            .arg("--listen")
            .arg(format!("127.0.0.1:{coordinator_port}"))
            .arg("--validators")
            .arg(&validators_path),
    )?;
    wait_health(&http, coordinator_port).await?;
    let coordinator = format!("http://127.0.0.1:{coordinator_port}");

    let node_a = TestNode::new(&tmp.path.join("node-a"))?;
    let node_b = TestNode::new(&tmp.path.join("node-b"))?;
    let node_c = TestNode::new(&tmp.path.join("node-c"))?;
    register(&http, &coordinator, &node_a).await?;
    register(&http, &coordinator, &node_b).await?;
    register(&http, &coordinator, &node_c).await?;

    let zero = balance(&http, &coordinator, &node_a).await?;
    assert_eq!(zero.balance_mcu, 0, "new node must begin at 0 CU");

    let community_assignment = poll_until_assignment(&http, &coordinator, &node_a, None).await?;
    complete_assignment(&http, &coordinator, &node_a, &community_assignment).await?;
    let earned = balance(&http, &coordinator, &node_a).await?;
    assert!(
        earned.balance_mcu > 0,
        "node A should earn CU from community work"
    );

    let paid = submit(
        &http,
        &coordinator,
        &node_a,
        WorkSpec::PrimeCount { start: 2, end: 100 },
        2,
    )
    .await?;
    assert!(paid.reserved_mcu > 0);
    assert!(paid.balance_mcu < earned.balance_mcu);

    let self_poll = poll(&http, &coordinator, &node_a).await?;
    if let Some(assignment) = self_poll.assignment {
        assert_ne!(
            assignment.job_id, paid.job_id,
            "requester must not receive its own paid shard"
        );
    }

    let duplicate_paid_result = complete_next_for_job(&http, &coordinator, &node_b, &paid.job_id)
        .await?
        .context("node B should complete a paid shard")?;
    complete_next_for_job(&http, &coordinator, &node_c, &paid.job_id)
        .await?
        .context("node C should complete a paid shard")?;
    let paid_status = job_status(&http, &coordinator, &paid.job_id).await?;
    assert_eq!(paid_status.status, "completed");
    assert_eq!(
        paid_status.completed_assignments, paid_status.total_assignments,
        "paid job escrow should drain through completed shard rewards"
    );

    assert!(
        post_result_raw(&http, &coordinator, &duplicate_paid_result)
            .await
            .is_err(),
        "duplicate result settlement must be rejected"
    );

    let reused = signed_submit(&node_a, WorkSpec::PrimeCount { start: 2, end: 20 }, 1)?;
    let first_reused_response: SubmitJobResponse =
        post_json(&http, &coordinator, "/v1/jobs/submit", &reused).await?;
    assert_eq!(first_reused_response.job_id, job_id_from_auth(&reused.auth));
    assert!(
        post_json::<_, SubmitJobResponse>(&http, &coordinator, "/v1/jobs/submit", &reused)
            .await
            .is_err(),
        "reused submit nonce/request must be rejected"
    );

    validators[2].kill();
    let outage_job = submit(
        &http,
        &coordinator,
        &node_a,
        WorkSpec::PrimeCount {
            start: 100,
            end: 150,
        },
        1,
    )
    .await?;
    assert!(
        outage_job.ledger_entry_hash.is_some(),
        "3 remaining validators should still certify with 3-of-4 threshold"
    );

    run_ok(
        Command::new(&validator_bin)
            .arg("sync")
            .arg("--db")
            .arg(&validator_dbs[2])
            .arg("--validators")
            .arg(&validators_path),
        "sync restarted validator",
    )?;
    validators[2] = ProcessGuard::spawn(
        Command::new(&validator_bin)
            .arg("serve")
            .arg("--home")
            .arg(&validator_homes[2])
            .arg("--db")
            .arg(&validator_dbs[2])
            .arg("--listen")
            .arg(format!("127.0.0.1:{}", validator_ports[2]))
            .arg("--validators")
            .arg(&validators_path),
    )?;
    wait_health(&http, validator_ports[2]).await?;

    let heads = validator_heads(&http, &set).await?;
    assert!(
        heads
            .windows(2)
            .all(|pair| pair[0].sequence == pair[1].sequence
                && pair[0].entry_hash == pair[1].entry_hash),
        "all validator heads should match after rejoin"
    );
    for db in &validator_dbs {
        run_ok(
            Command::new(&validator_bin)
                .arg("audit")
                .arg("--db")
                .arg(db)
                .arg("--validators")
                .arg(&validators_path),
            "audit validator ledger",
        )?;
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn coordinator_recovers_community_reservation_after_intent_persisted() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("mesh-validator"));
    let coordinator_bin = bin_dir.join(exe("mesh-coordinator"));

    let tmp = TestDir::new()?;
    let http = Client::new();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, _set) =
        create_validator_set(&tmp, &validator_bin, &validator_ports)?;

    let coordinator_db = tmp.path.join("coordinator-recovery.db");
    let failed_seed = Command::new(&coordinator_bin)
        .arg("seed")
        .arg("--db")
        .arg(&coordinator_db)
        .arg("--validators")
        .arg(&validators_path)
        .arg("--start")
        .arg("2")
        .arg("--end")
        .arg("500")
        .arg("--shards")
        .arg("2")
        .output()
        .context("running seed expected to fail before validators are online")?;
    assert!(
        !failed_seed.status.success(),
        "seed should fail after persisting an intent when no validators are reachable"
    );

    let mut validators = Vec::new();
    for index in 0..4 {
        validators.push(
            start_validator(
                &validator_bin,
                &validators_path,
                &validator_homes[index],
                &validator_dbs[index],
                validator_ports[index],
                &http,
            )
            .await?,
        );
    }

    run_ok(
        Command::new(&coordinator_bin)
            .arg("recover")
            .arg("--db")
            .arg(&coordinator_db)
            .arg("--validators")
            .arg(&validators_path),
        "recover pending community reservation",
    )?;
    run_ok(
        Command::new(&coordinator_bin)
            .arg("recover")
            .arg("--db")
            .arg(&coordinator_db)
            .arg("--validators")
            .arg(&validators_path),
        "repeat recovery idempotently",
    )?;

    let coordinator_port = free_port()?;
    let _coordinator = ProcessGuard::spawn(
        Command::new(&coordinator_bin)
            .arg("serve")
            .arg("--db")
            .arg(&coordinator_db)
            .arg("--listen")
            .arg(format!("127.0.0.1:{coordinator_port}"))
            .arg("--validators")
            .arg(&validators_path),
    )?;
    wait_health(&http, coordinator_port).await?;
    let coordinator = format!("http://127.0.0.1:{coordinator_port}");

    let node = TestNode::new(&tmp.path.join("recovery-node"))?;
    register(&http, &coordinator, &node).await?;
    let assignment = poll_until_assignment(&http, &coordinator, &node, None).await?;
    assert!(
        assignment.system_funded,
        "recovered community reservation should release system-funded work"
    );
    complete_assignment(&http, &coordinator, &node, &assignment).await?;
    assert!(balance(&http, &coordinator, &node).await?.balance_mcu > 0);

    drop(validators);
    Ok(())
}

struct TestNode {
    identity: NodeIdentity,
}

impl TestNode {
    fn new(home: &Path) -> Result<Self> {
        Ok(Self {
            identity: NodeIdentity::load_or_create(home)?,
        })
    }
}

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Result<Self> {
        let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!("mesh-integration-{suffix}"));
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
            .context("spawning test process")?;
        Ok(Self { child: Some(child) })
    }

    fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        self.kill();
    }
}

fn create_validator_set(
    tmp: &TestDir,
    validator_bin: &Path,
    validator_ports: &[u16; 4],
) -> Result<(PathBuf, Vec<PathBuf>, Vec<PathBuf>, ValidatorSet)> {
    let mut members = Vec::new();
    let mut validator_homes = Vec::new();
    let mut validator_dbs = Vec::new();
    for (index, port) in validator_ports.iter().enumerate() {
        let home = tmp.path.join(format!("validator-{index}"));
        let output = Command::new(validator_bin)
            .arg("id")
            .arg("--home")
            .arg(&home)
            .output()
            .with_context(|| format!("creating validator identity {index}"))?;
        if !output.status.success() {
            bail!(
                "validator id failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let stdout = String::from_utf8(output.stdout)?;
        let validator_id = parse_value(&stdout, "validator_id=")?;
        let public_key_b64 = parse_value(&stdout, "public_key_b64=")?;
        members.push(json!({
            "validator_id": validator_id,
            "url": format!("http://127.0.0.1:{port}"),
            "public_key_b64": public_key_b64,
        }));
        validator_homes.push(home);
        validator_dbs.push(tmp.path.join(format!("validator-{index}.db")));
    }
    let validators_path = tmp.path.join("validators.json");
    fs::write(
        &validators_path,
        serde_json::to_vec_pretty(&json!({
            "threshold": 3,
            "community_issuance_limit_mcu": 1_000_000_000i64,
            "members": members,
        }))?,
    )?;
    let set: ValidatorSet = serde_json::from_slice(&fs::read(&validators_path)?)?;
    Ok((validators_path, validator_homes, validator_dbs, set))
}

async fn start_validator(
    validator_bin: &Path,
    validators_path: &Path,
    home: &Path,
    db: &Path,
    port: u16,
    http: &Client,
) -> Result<ProcessGuard> {
    let guard = ProcessGuard::spawn(
        Command::new(validator_bin)
            .arg("serve")
            .arg("--home")
            .arg(home)
            .arg("--db")
            .arg(db)
            .arg("--listen")
            .arg(format!("127.0.0.1:{port}"))
            .arg("--validators")
            .arg(validators_path),
    )?;
    wait_health(http, port).await?;
    Ok(guard)
}

async fn register(http: &Client, coordinator: &str, node: &TestNode) -> Result<()> {
    let public_key_b64 = node.identity.public_key_b64();
    let caps = capabilities();
    let body_hash = register_body_hash(&public_key_b64, &caps)?;
    let req = RegisterRequest {
        auth: node.identity.auth("register", &body_hash),
        public_key_b64,
        capabilities: caps,
    };
    let _: serde_json::Value = post_json(http, coordinator, "/v1/nodes/register", &req).await?;
    Ok(())
}

async fn poll(http: &Client, coordinator: &str, node: &TestNode) -> Result<PollResponse> {
    let req = PollRequest {
        auth: node.identity.auth("poll", &empty_body_hash()),
    };
    post_json(http, coordinator, "/v1/work/poll", &req).await
}

async fn poll_until_assignment(
    http: &Client,
    coordinator: &str,
    node: &TestNode,
    job_id: Option<&str>,
) -> Result<WorkAssignment> {
    for _ in 0..100 {
        if let Some(assignment) = poll(http, coordinator, node).await?.assignment
            && job_id.is_none_or(|id| assignment.job_id == id)
        {
            return Ok(assignment);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    bail!("timed out waiting for assignment")
}

async fn complete_next_for_job(
    http: &Client,
    coordinator: &str,
    node: &TestNode,
    job_id: &str,
) -> Result<Option<ResultRequest>> {
    for _ in 0..100 {
        if let Some(assignment) = poll(http, coordinator, node).await?.assignment {
            let req = complete_assignment(http, coordinator, node, &assignment).await?;
            if assignment.job_id == job_id {
                return Ok(Some(req));
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Ok(None)
}

async fn complete_assignment(
    http: &Client,
    coordinator: &str,
    node: &TestNode,
    assignment: &WorkAssignment,
) -> Result<ResultRequest> {
    let result = execute_work(&assignment.work);
    let body_hash = result_body_hash(
        &assignment.assignment_id,
        &assignment.job_id,
        assignment.shard_index,
        &assignment.work,
        assignment.reward_mcu,
        assignment.system_funded,
        &result,
    )?;
    let req = ResultRequest {
        auth: node.identity.auth("result", &body_hash),
        assignment_id: assignment.assignment_id.clone(),
        job_id: assignment.job_id.clone(),
        shard_index: assignment.shard_index,
        work: assignment.work.clone(),
        reward_mcu: assignment.reward_mcu,
        system_funded: assignment.system_funded,
        result,
    };
    post_result_raw(http, coordinator, &req).await?;
    Ok(req)
}

async fn post_result_raw(http: &Client, coordinator: &str, req: &ResultRequest) -> Result<()> {
    let _: serde_json::Value = post_json(http, coordinator, "/v1/work/result", req).await?;
    Ok(())
}

async fn submit(
    http: &Client,
    coordinator: &str,
    node: &TestNode,
    work: WorkSpec,
    shards: u32,
) -> Result<SubmitJobResponse> {
    let req = signed_submit(node, work, shards)?;
    post_json(http, coordinator, "/v1/jobs/submit", &req).await
}

fn signed_submit(node: &TestNode, work: WorkSpec, shards: u32) -> Result<SubmitJobRequest> {
    let body_hash = submit_body_hash(&work, shards)?;
    Ok(SubmitJobRequest {
        auth: node.identity.auth("submit", &body_hash),
        work,
        shards,
    })
}

async fn balance(http: &Client, coordinator: &str, node: &TestNode) -> Result<BalanceResponse> {
    get_json(
        http,
        coordinator,
        &format!("/v1/nodes/{}/balance", node.identity.node_id()),
    )
    .await
}

async fn job_status(http: &Client, coordinator: &str, job_id: &str) -> Result<JobStatusResponse> {
    get_json(http, coordinator, &format!("/v1/jobs/{job_id}")).await
}

async fn validator_heads(http: &Client, set: &ValidatorSet) -> Result<Vec<LedgerHead>> {
    let mut heads = Vec::new();
    for member in &set.members {
        let proof: serde_json::Value = http
            .get(format!("{}/v1/ledger/head", member.url))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        heads.push(serde_json::from_value(proof["head"].clone())?);
    }
    Ok(heads)
}

async fn get_json<T: serde::de::DeserializeOwned>(
    http: &Client,
    base: &str,
    path: &str,
) -> Result<T> {
    decode(http.get(format!("{base}{path}")).send().await?).await
}

async fn post_json<T: serde::Serialize + ?Sized, R: serde::de::DeserializeOwned>(
    http: &Client,
    base: &str,
    path: &str,
    value: &T,
) -> Result<R> {
    decode(
        http.post(format!("{base}{path}"))
            .json(value)
            .send()
            .await?,
    )
    .await
}

async fn decode<T: serde::de::DeserializeOwned>(response: reqwest::Response) -> Result<T> {
    let status = response.status();
    let text = response.text().await?;
    if !status.is_success() {
        if let Ok(error) = serde_json::from_str::<ErrorResponse>(&text) {
            bail!("HTTP {status}: {}", error.error);
        }
        bail!("HTTP {status}: {text}");
    }
    serde_json::from_str(&text).map_err(Into::into)
}

async fn wait_health(http: &Client, port: u16) -> Result<()> {
    let url = format!("http://127.0.0.1:{port}/health");
    for _ in 0..100 {
        if let Ok(response) = http.get(&url).send().await
            && response.status().is_success()
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    bail!("service on port {port} did not become healthy")
}

fn capabilities() -> NodeCapabilities {
    NodeCapabilities {
        protocol_version: mesh_protocol::PROTOCOL_VERSION,
        hostname: "integration-test".into(),
        os: env::consts::OS.into(),
        arch: env::consts::ARCH.into(),
        cpu_brand: "test-cpu".into(),
        logical_cpus: 4,
        total_memory_bytes: 8 * 1024 * 1024 * 1024,
        cpu_benchmark_score: 1_000,
        gpus: Vec::new(),
        model_seed_url: None,
        cached_model_manifests: Vec::new(),
        coordinator_latency_micros: 0,
        model_bandwidth_kbps: 100_000,
        accelerator_load_permille: 0,
        ai_runtime_ready: false,
    }
}

fn build_bins(workspace: &Path) -> Result<()> {
    run_ok(
        Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .arg("build")
            .arg("--ignore-rust-version")
            .arg("--workspace")
            .arg("--bins")
            .current_dir(workspace),
        "build workspace binaries",
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

fn workspace_root() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("cannot resolve workspace root"))
}

fn parse_value(output: &str, prefix: &str) -> Result<String> {
    output
        .lines()
        .find_map(|line| line.strip_prefix(prefix).map(str::trim))
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("missing {prefix} in output: {output}"))
}

fn free_port() -> Result<u16> {
    Ok(TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}
