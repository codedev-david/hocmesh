use anyhow::{Context, Result, anyhow, bail};
use hocmesh_core::{
    compute::{execute_work, split_work, work_cost_mcu},
    identity::NodeIdentity,
};
use hocmesh_ledger::{
    network::LedgerNetwork,
    types::{
        COMMUNITY_ISSUANCE_ACCOUNT, LedgerHead, LedgerTransaction, Posting, TransactionEvidence,
        TransactionKind, ValidatorMember, ValidatorSet, ValidatorSignature, escrow_account,
    },
};
use hocmesh_protocol::{
    BalanceResponse, ErrorResponse, JobStatusResponse, NodeCapabilities, PollRequest, PollResponse,
    RegisterRequest, ResultRequest, SubmitJobRequest, SubmitJobResponse, WorkAssignment, WorkSpec,
    empty_body_hash, job_id_from_auth, now_unix, register_body_hash, result_body_hash,
    submit_body_hash,
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
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));

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
    let node_bin = bin_dir.join(exe("hocmesh"));
    let seed_job = "job_community_seed";
    let seed_sponsors = sponsors_file(
        &tmp.path,
        &node_bin,
        &validator_homes,
        &validators_path,
        seed_job,
        (2, 200000, 4),
    )?;
    run_ok(
        Command::new(&coordinator_bin)
            .arg("seed")
            .arg("--job-id")
            .arg(seed_job)
            .arg("--sponsors")
            .arg(&seed_sponsors)
            .arg("--db")
            .arg(&coordinator_db)
            .arg("--validators")
            .arg(&validators_path)
            .arg("--start")
            .arg("2")
            .arg("--end")
            .arg("200000")
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
    // ---- Inference, under quorum ----
    //
    // This is the leg the whole network exists for: one node lends its GPU,
    // another spends the CU it earned counting primes to use it. Until now
    // AI ran entirely outside the ledger, so contributing a GPU earned
    // nothing and using one cost nothing. Every number below is certified by
    // the validator set and replayed by the audit at the end of this test.
    let gpu_node = TestNode::new(&tmp.path.join("node-gpu"))?;
    register_with(&http, &coordinator, &gpu_node, ai_capabilities()).await?;

    let manifest = hocmesh_model::ModelManifest {
        schema_version: 1,
        model_id: "quorum-tiny".into(),
        revision: "v1".into(),
        format: hocmesh_model::ModelFormat::Gguf,
        architecture: "llama".into(),
        // Small enough that a prime shard can pay for a prompt.
        parameter_count: Some(1_000),
        tensor_dtype: Some("q4".into()),
        total_size_bytes: 4_096,
        chunks: vec![hocmesh_model::ChunkRef {
            index: 0,
            sha256: hocmesh_protocol::hash_bytes(b"weights"),
            size_bytes: 4_096,
        }],
        metadata: Default::default(),
    };
    let register_model = hocmesh_ai::RegisterModelRequest {
        auth: node_a.identity.auth(
            "register_model",
            &hocmesh_ai::register_model_body_hash(&manifest)?,
        ),
        manifest: manifest.clone(),
    };
    let _: hocmesh_ai::RegisterModelResponse = post_json(
        &http,
        &coordinator,
        "/v1/ai/models/register",
        &register_model,
    )
    .await?;

    // The requester prices its own job from the published manifest and signs
    // that price. Nothing downstream is allowed to raise it.
    let prompts = vec!["what is a compute unit".to_string()];
    let billing = hocmesh_ai::bill_for_prompts(
        &manifest.digest()?,
        manifest
            .parameter_count
            .context("manifest must be priceable")?,
        manifest.total_size_bytes,
        &prompts,
        16,
    )?;
    let inference_price = billing.max_cost_mcu;
    let spender_before = balance(&http, &coordinator, &node_a).await?.balance_mcu;
    let gpu_before = balance(&http, &coordinator, &gpu_node).await?.balance_mcu;
    assert!(
        spender_before >= inference_price,
        "CU earned on the CPU has to be able to buy GPU time: have {spender_before}, need {inference_price}"
    );

    let mut submit_ai = hocmesh_ai::SubmitInferenceRequest {
        auth: node_a.identity.auth("unused", &empty_body_hash()),
        model_id: "quorum-tiny".into(),
        revision: "v1".into(),
        prompts: prompts.clone(),
        max_tokens: 16,
        temperature_milli: 0,
        seed: 1,
        requirements: hocmesh_ai::InferenceRequirements {
            required_backends: [hocmesh_gpu::BackendKind::Cuda].into_iter().collect(),
            minimum_memory_bytes: 1,
            needs_fp16: true,
            needs_bf16: false,
            needs_int8: false,
            batch_size: 1,
            pipeline_stages: 1,
            tensor_parallelism: 1,
        },
        layer_count: 2,
        billing: billing.clone(),
    };
    submit_ai.auth = node_a.identity.auth(
        "submit_inference",
        &hocmesh_ai::submit_inference_body_hash(&submit_ai)?,
    );
    let ai_job: hocmesh_ai::SubmitInferenceResponse =
        post_json(&http, &coordinator, "/v1/ai/jobs/submit", &submit_ai).await?;

    // Escrow left the requester for exactly the number it signed - and the
    // ledger, not the coordinator, is what says so.
    let spender_after_submit = balance(&http, &coordinator, &node_a).await?.balance_mcu;
    assert_eq!(
        spender_after_submit,
        spender_before - inference_price,
        "reserving inference must move exactly the priced CU into escrow"
    );

    let ai_poll = hocmesh_ai::PollInferenceRequest {
        auth: gpu_node.identity.auth("poll_inference", &empty_body_hash()),
    };
    let leased: hocmesh_ai::PollInferenceResponse =
        post_json(&http, &coordinator, "/v1/ai/work/poll", &ai_poll).await?;
    let assignment = leased
        .assignment
        .context("the GPU node should be handed the inference batch")?;

    // The provider prices its own batch from the assignment. Because the
    // price is closed form, it is the same number the ledger recomputes.
    let (batch_start, batch_end, reward_mcu) = hocmesh_ai::assignment_claim(&assignment)
        .context("an assignment must carry a priceable batch")?;
    assert_eq!(
        reward_mcu, inference_price,
        "one batch, whole job, whole price"
    );

    let answer = "a unit of machine work";
    let mut report = hocmesh_ai::ReportInferenceRequest {
        auth: gpu_node.identity.auth("unused", &empty_body_hash()),
        assignment_id: assignment.assignment_id.clone(),
        job_id: ai_job.job_id.clone(),
        batch_start,
        batch_end,
        reward_mcu,
        outputs: vec![hocmesh_ai::PromptOutput {
            prompt_index: 0,
            text: answer.into(),
            output_sha256: hocmesh_protocol::hash_bytes(answer.as_bytes()),
            duration_ms: 1,
        }],
    };
    report.auth = gpu_node.identity.auth(
        "report_inference",
        &hocmesh_ai::report_inference_body_hash(&report)?,
    );
    let settled: hocmesh_ai::ReportInferenceResponse =
        post_json(&http, &coordinator, "/v1/ai/work/result", &report).await?;
    assert!(settled.accepted);
    assert_eq!(settled.reward_mcu, inference_price);

    // Delivery earns nothing on its own. The provider's signature says what it
    // computed; only the requester's says the answer was worth paying for.
    assert_eq!(
        balance(&http, &coordinator, &gpu_node).await?.balance_mcu,
        gpu_before,
        "a delivered batch is not a paid batch"
    );
    let outputs_digest = hocmesh_protocol::hash_json(&report.outputs)?;
    let receipt = hocmesh_ai::ReceiptInferenceRequest {
        auth: node_a.identity.auth(
            "receipt_inference",
            &hocmesh_protocol::inference_receipt_body_hash(
                &report.assignment_id,
                &ai_job.job_id,
                report.batch_start,
                report.batch_end,
                inference_price,
                &outputs_digest,
            )?,
        ),
        assignment_id: report.assignment_id.clone(),
    };
    let taken: hocmesh_ai::ReceiptInferenceResponse =
        post_json(&http, &coordinator, "/v1/ai/jobs/receipt", &receipt).await?;
    assert_eq!(taken.outputs, report.outputs);
    assert_eq!(
        balance(&http, &coordinator, &gpu_node).await?.balance_mcu,
        gpu_before,
        "taking delivery moves escrow into holding, not into a provider"
    );
    let accept = hocmesh_ai::SettleInferenceRequest {
        auth: node_a.identity.auth(
            "accept_inference",
            &hocmesh_protocol::inference_verdict_body_hash(
                true,
                &report.assignment_id,
                &ai_job.job_id,
                report.batch_start,
                report.batch_end,
                inference_price,
                &outputs_digest,
            )?,
        ),
        assignment_id: report.assignment_id.clone(),
        accepted: true,
        reason: String::new(),
    };
    let paid: hocmesh_ai::SettleInferenceResponse =
        post_json(&http, &coordinator, "/v1/ai/jobs/settle", &accept).await?;
    assert_eq!(paid.paid_mcu, inference_price);

    // CPU work paid for GPU work, across the ledger, with nothing minted:
    // this is the trade the network is for, and it is now enforceable.
    let gpu_after = balance(&http, &coordinator, &gpu_node).await?.balance_mcu;
    assert_eq!(gpu_after, gpu_before + inference_price);
    assert_eq!(
        balance(&http, &coordinator, &node_a).await?.balance_mcu,
        spender_before - inference_price
    );
    assert_eq!(
        (spender_after_submit - spender_before) + (gpu_after - gpu_before),
        0,
        "an inference job must neither mint nor burn CU"
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

    // A checkpoint has to be reachable the way an operator would reach it:
    // mirror the ledger, ask the quorum what state it holds, keep that
    // answer, throw away the history it vouches for, and audit anyway.
    let node_bin = bin_dir.join(exe("hocmesh"));
    let mirror = tmp.path.join("mirror.db");
    let mirror = mirror.to_string_lossy().to_string();
    for (cmd, label) in [
        ("ledger-sync", "mirror the quorum ledger"),
        ("ledger-checkpoint", "record a quorum checkpoint"),
        ("ledger-prune", "prune below the checkpoint"),
        ("ledger-audit", "audit from the checkpoint"),
    ] {
        run_ok(
            Command::new(&node_bin)
                .arg(cmd)
                .arg("--db")
                .arg(&mirror)
                .arg("--validators")
                .arg(&validators_path),
            label,
        )?;
    }

    // And the pruning has to be real: replaying from genesis must now be
    // impossible rather than quietly succeeding on a shortened history.
    assert!(
        run_ok(
            Command::new(&node_bin)
                .arg("ledger-audit")
                .arg("--full")
                .arg("--db")
                .arg(&mirror)
                .arg("--validators")
                .arg(&validators_path),
            "genesis audit of a pruned mirror",
        )
        .is_err(),
        "a pruned ledger must refuse to claim it audited from genesis"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn coordinator_recovers_community_reservation_after_intent_persisted() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));

    let tmp = TestDir::new()?;
    let http = Client::new();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, _set) =
        create_validator_set(&tmp, &validator_bin, &validator_ports)?;

    let coordinator_db = tmp.path.join("coordinator-recovery.db");
    let node_bin = bin_dir.join(exe("hocmesh"));
    let seed_job = "job_recovery_seed";
    let seed_sponsors = sponsors_file(
        &tmp.path,
        &node_bin,
        &validator_homes,
        &validators_path,
        seed_job,
        (2, 500, 2),
    )?;
    let failed_seed = Command::new(&coordinator_bin)
        .arg("seed")
        .arg("--job-id")
        .arg(seed_job)
        .arg("--sponsors")
        .arg(&seed_sponsors)
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
        let path = env::temp_dir().join(format!("hocmesh-integration-{suffix}"));
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

/// A node that can actually answer an inference batch.
///
/// The scheduler refuses to place AI work on hardware that never claimed to
/// have any, so proving the inference economy needs a node that did.
fn ai_capabilities() -> NodeCapabilities {
    let mut caps = capabilities();
    caps.ai_runtime_ready = true;
    caps.gpus = vec![hocmesh_protocol::GpuCapability {
        stable_id: "gpu-integration".into(),
        vendor: "nvidia".into(),
        name: "test".into(),
        backend: "cuda".into(),
        memory_mb: Some(1024),
        driver_version: None,
        compute_version: Some("8.0".into()),
        supports_fp16: true,
        supports_bf16: true,
        supports_int8: true,
        benchmark_bytes_per_second: Some(1),
        benchmark_p95_micros: Some(1),
    }];
    caps.shared_gpu_percent = 100;
    caps
}

async fn register_with(
    http: &Client,
    coordinator: &str,
    node: &TestNode,
    caps: NodeCapabilities,
) -> Result<()> {
    let public_key_b64 = node.identity.public_key_b64();
    let body_hash = register_body_hash(&public_key_b64, &caps)?;
    let req = RegisterRequest {
        auth: node.identity.auth("register", &body_hash),
        public_key_b64,
        capabilities: caps,
    };
    let _: serde_json::Value = post_json(http, coordinator, "/v1/nodes/register", &req).await?;
    Ok(())
}

fn capabilities() -> NodeCapabilities {
    NodeCapabilities {
        protocol_version: hocmesh_protocol::PROTOCOL_VERSION,
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
        shared_logical_cpus: 2,
        shared_memory_bytes: 4 * 1024 * 1024 * 1024,
        shared_gpu_percent: 0,
        network_coordinate: None,
        probe_endpoint: None,
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

/// Concurrent settlements have to share entries, or the ledger is capped at
/// one consensus round per CU movement no matter how much hardware is behind
/// it. This is the load-bearing claim of the batching work, so it is measured
/// against four real validators rather than argued about.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_settlements_share_ledger_entries() -> Result<()> {
    const SETTLEMENTS: usize = 16;

    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));

    let tmp = TestDir::new()?;
    let http = Client::new();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, set) =
        create_validator_set(&tmp, &validator_bin, &validator_ports)?;
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

    let net = LedgerNetwork::new(set)?;
    let before = net.head_quorum().await?.sequence;
    let work = WorkSpec::PrimeCount { start: 2, end: 200 };
    let cost: i64 = split_work(&work, 1).iter().map(work_cost_mcu).sum();
    let mut handles = Vec::new();
    // Sponsorships are collected up front, on purpose. They are an operator
    // action taken before a mint is submitted, and shelling out to a validator
    // machine mid-loop would serialise the submissions this test exists to
    // send at once - measuring the harness instead of the batching.
    let node_bin = bin_dir.join(exe("hocmesh"));
    let mut sponsorships = Vec::new();
    for index in 0..SETTLEMENTS {
        let job_id = format!("job_batching_{index}");
        sponsorships.push(community_vouches(
            &node_bin,
            &validator_homes,
            &validators_path,
            &job_id,
            (2, 200, 1),
        )?);
    }
    for (index, sponsors) in sponsorships.into_iter().enumerate() {
        let job_id = format!("job_batching_{index}");
        let tx = LedgerTransaction {
            transaction_id: format!("community_reserve_{job_id}"),
            kind: TransactionKind::CommunityReserve,
            postings: vec![
                Posting {
                    account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                    delta_mcu: -cost,
                },
                Posting {
                    account_id: escrow_account(&job_id),
                    delta_mcu: cost,
                },
            ],
            evidence: TransactionEvidence::CommunityReserve {
                job_id,
                work: work.clone(),
                shards: 1,
                sponsors,
            },
            created_at: now_unix(),
        };
        let net = net.clone();
        handles.push(tokio::spawn(async move { net.transact(tx).await }));
    }
    let mut entries = std::collections::HashSet::new();
    for handle in handles {
        entries.insert(handle.await??.entry.entry_hash);
    }
    let after = net.head_quorum().await?.sequence;
    let rounds = after - before;
    println!("{SETTLEMENTS} concurrent settlements took {rounds} consensus rounds");
    assert_eq!(
        rounds as usize,
        entries.len(),
        "every settled entry must show up in the chain exactly once"
    );
    assert!(
        rounds < SETTLEMENTS as u64,
        "{SETTLEMENTS} concurrent settlements took {rounds} rounds; batching is not happening"
    );

    // And the batching must not have loosened any rule: every validator has to
    // be able to replay what it just agreed to.
    for db in &validator_dbs {
        run_ok(
            Command::new(&validator_bin)
                .arg("audit")
                .arg("--db")
                .arg(db)
                .arg("--validators")
                .arg(&validators_path),
            "audit validator ledger after batched settlements",
        )?;
    }
    drop(validators);
    Ok(())
}

/// A fifth validator is admitted by vouch, catches up, and then holds a seat
/// the bootstrap file has never heard of.
///
/// This is the whole membership path through the real binaries. Three sitting
/// validators sponsor a joiner by name, the change is certified into the chain,
/// the joiner replays from the *genesis* file and arrives holding a set that
/// file does not describe, and a client still addressing the old seats follows
/// the change forward instead of stalling. The last settlement runs with one of
/// the original validators killed, so a quorum of four out of five can only
/// form if the joiner's signature counts for as much as anybody else's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vouched_validator_joins_and_the_chain_carries_the_change() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = Client::new();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, set) =
        create_validator_set(&tmp, &validator_bin, &validator_ports)?;
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

    // Something in the chain before the set moves, so the joiner has history to
    // replay that predates its own admission.
    let net = LedgerNetwork::new(set.clone())?;
    settle_community(
        &net,
        "before",
        &node_bin,
        &validator_homes,
        &validators_path,
    )
    .await?;
    let before = net.head_quorum().await?.sequence;

    // A fifth identity, created but in nobody's set.
    let joiner_port = free_port()?;
    let joiner_home = tmp.path.join("validator-4");
    let joiner_db = tmp.path.join("validator-4.db");
    let joiner = validator_member(&validator_bin, &joiner_home, joiner_port)?;
    let member_path = tmp.path.join("joiner.json");
    fs::write(&member_path, serde_json::to_vec_pretty(&joiner)?)?;

    // Sponsors, one at a time and each on its own machine's key. Three is the
    // sitting set's own threshold: admission is never cheaper than agreement.
    let mut vouches = Vec::new();
    for home in validator_homes.iter().take(3) {
        vouches.push(vouch(&node_bin, home, &validators_path, &member_path, 4)?);
    }
    let vouches_path = tmp.path.join("vouches.json");
    fs::write(&vouches_path, serde_json::to_vec_pretty(&vouches)?)?;

    let next_path = tmp.path.join("validators.next.json");
    run_ok(
        Command::new(&node_bin)
            .arg("--home")
            .arg(&validator_homes[0])
            .arg("membership-commit")
            .arg("--validators")
            .arg(&validators_path)
            .arg("--action")
            .arg("join")
            .arg("--member")
            .arg(&member_path)
            .arg("--threshold")
            .arg("4")
            .arg("--vouches")
            .arg(&vouches_path)
            .arg("--out")
            .arg(&next_path),
        "membership-commit",
    )?;
    let next: ValidatorSet = serde_json::from_slice(&fs::read(&next_path)?)?;
    assert_eq!(next.members.len(), 5);
    assert_eq!(next.threshold, 4);

    // The joiner bootstraps from the file that predates it. Entries up to its
    // own admission are checked against the four seats that certified them, and
    // everything after against the five the chain then hands over - which is
    // the only reason a file with no mention of this validator is enough.
    run_ok(
        Command::new(&validator_bin)
            .arg("sync")
            .arg("--db")
            .arg(&joiner_db)
            .arg("--validators")
            .arg(&validators_path),
        "joiner sync",
    )?;
    validators.push(
        start_validator(
            &validator_bin,
            &validators_path,
            &joiner_home,
            &joiner_db,
            joiner_port,
            &http,
        )
        .await?,
    );

    // A client still holding the four-member file settles again. Quorum is four
    // of five now, which that file cannot reach, so this only passes if the
    // client followed the change forward on its own.
    settle_community(&net, "after", &node_bin, &validator_homes, &validators_path).await?;
    assert_eq!(net.set().members.len(), 5);
    assert!(net.head_quorum().await?.sequence > before);

    // Take an original validator away. Four seats are left, the threshold is
    // four, and one of them is the joiner: nothing settles from here unless the
    // admission was real.
    validators[0].kill();
    settle_community(
        &net,
        "without-a-founder",
        &node_bin,
        &validator_homes,
        &validators_path,
    )
    .await?;

    let head = net.head_quorum().await?;
    let joiner_head = validator_head(&http, &joiner.url).await?;
    assert_eq!(joiner_head.entry_hash, head.entry_hash);
    assert_eq!(joiner_head.sequence, head.sequence);

    for mut v in validators {
        v.kill();
    }
    Ok(())
}

/// Read a validator's identity out of its home and describe it as a member.
fn validator_member(validator_bin: &Path, home: &Path, port: u16) -> Result<ValidatorMember> {
    let output = Command::new(validator_bin)
        .arg("id")
        .arg("--home")
        .arg(home)
        .output()
        .context("creating joiner identity")?;
    if !output.status.success() {
        bail!(
            "validator id failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8(output.stdout)?;
    Ok(ValidatorMember {
        validator_id: parse_value(&stdout, "validator_id=")?,
        url: format!("http://127.0.0.1:{port}"),
        public_key_b64: parse_value(&stdout, "public_key_b64=")?,
    })
}

/// One sitting validator's signed sponsorship, produced the way an operator
/// produces it: locally, from the machine that holds the key.
fn vouch(
    node_bin: &Path,
    home: &Path,
    validators_path: &Path,
    member_path: &Path,
    threshold: usize,
) -> Result<ValidatorSignature> {
    let output = Command::new(node_bin)
        .arg("--home")
        .arg(home)
        .arg("membership-vouch")
        .arg("--validators")
        .arg(validators_path)
        .arg("--action")
        .arg("join")
        .arg("--member")
        .arg(member_path)
        .arg("--threshold")
        .arg(threshold.to_string())
        .output()
        .context("running membership-vouch")?;
    if !output.status.success() {
        bail!(
            "membership-vouch failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8(output.stdout)?;
    let line = stdout
        .lines()
        .next_back()
        .ok_or_else(|| anyhow!("membership-vouch printed nothing"))?;
    Ok(serde_json::from_str(line)?)
}

/// Write the sponsorships a community mint needs into a file the coordinator
/// can carry. The coordinator holds no key that can mint, so this is the only
/// way a seeded job gets authorized.
fn sponsors_file(
    dir: &Path,
    node_bin: &Path,
    homes: &[PathBuf],
    validators_path: &Path,
    job_id: &str,
    work: (u64, u64, u32),
) -> Result<PathBuf> {
    let vouches = community_vouches(node_bin, homes, validators_path, job_id, work)?;
    let path = dir.join(format!("sponsors-{job_id}.json"));
    fs::write(&path, serde_json::to_vec_pretty(&vouches)?)?;
    Ok(path)
}

/// Collect enough real sponsorships to mint a community job.
///
/// Shelled out to the node binary on purpose: the sponsorship has to come off
/// the key that actually sits in a validator's home, so a test that signed
/// in-process would be proving something the operator flow never does.
fn community_vouches(
    node_bin: &Path,
    homes: &[PathBuf],
    validators_path: &Path,
    job_id: &str,
    work: (u64, u64, u32),
) -> Result<Vec<ValidatorSignature>> {
    let (start, end, shards) = work;
    // Every home handed in signs. The set's threshold can move between
    // collecting sponsorships and submitting the mint - that is what happens
    // the moment a validator joins - and a spare signature costs nothing while
    // a missing one costs the whole settlement.
    let mut out = Vec::new();
    for home in homes {
        let output = Command::new(node_bin)
            .arg("--home")
            .arg(home)
            .arg("community-vouch")
            .arg("--validators")
            .arg(validators_path)
            .arg("--job-id")
            .arg(job_id)
            .arg("--start")
            .arg(start.to_string())
            .arg("--end")
            .arg(end.to_string())
            .arg("--shards")
            .arg(shards.to_string())
            .output()
            .context("running community-vouch")?;
        if !output.status.success() {
            bail!(
                "community-vouch failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let stdout = String::from_utf8(output.stdout)?;
        let line = stdout
            .lines()
            .next_back()
            .ok_or_else(|| anyhow!("community-vouch printed nothing"))?;
        out.push(serde_json::from_str(line)?);
    }
    Ok(out)
}

/// Mint a small community job and settle its reservation through the quorum.
async fn settle_community(
    net: &LedgerNetwork,
    label: &str,
    node_bin: &Path,
    homes: &[PathBuf],
    validators_path: &Path,
) -> Result<()> {
    let work = WorkSpec::PrimeCount { start: 2, end: 200 };
    let shards = 1;
    let cost: i64 = split_work(&work, shards).iter().map(work_cost_mcu).sum();
    let job_id = format!("job_membership_{label}");
    let sponsors = community_vouches(node_bin, homes, validators_path, &job_id, (2, 200, shards))?;
    net.transact(LedgerTransaction {
        transaction_id: format!("community_reserve_{job_id}"),
        kind: TransactionKind::CommunityReserve,
        postings: vec![
            Posting {
                account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                delta_mcu: -cost,
            },
            Posting {
                account_id: escrow_account(&job_id),
                delta_mcu: cost,
            },
        ],
        evidence: TransactionEvidence::CommunityReserve {
            job_id,
            work,
            shards,
            sponsors,
        },
        created_at: now_unix(),
    })
    .await?;
    Ok(())
}

/// One validator's own view of the head, unsigned-checked - the caller is
/// asserting agreement, not trusting the answer.
async fn validator_head(http: &Client, url: &str) -> Result<LedgerHead> {
    let proof: serde_json::Value = http
        .get(format!("{}/v1/ledger/head", url.trim_end_matches('/')))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(serde_json::from_value(proof["head"].clone())?)
}

/// Two proposers, one height, and no way back.
///
/// A validator will not sign two different entries at the same sequence, which
/// is what keeps the chain from forking. But a lock with nothing above it is
/// also a lock with no way out: two clients leading a round at the same height
/// split the set between two entries, neither reaches threshold, nothing is
/// applied anywhere - and every seat is now holding a hash that will never
/// carry a certificate. The height has to be recoverable, or the ledger stops.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_split_proposal_does_not_wedge_the_height() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = Client::new();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, set) =
        create_validator_set(&tmp, &validator_bin, &validator_ports)?;
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

    // Two mints that are each perfectly valid at the same height. Only one of
    // them can be sequence 1, and nothing decides which.

    let left = community_mint(
        &node_bin,
        &validator_homes,
        &validators_path,
        "job_split_left",
    )?;
    let right = community_mint(
        &node_bin,
        &validator_homes,
        &validators_path,
        "job_split_right",
    )?;

    // Hand each half of the set a different first entry. Both halves accept -
    // each batch is valid - and neither half is big enough to certify.
    for (index, tx, proposer) in [
        (0usize, &left, "left"),
        (1, &left, "left"),
        (2, &right, "right"),
        (3, &right, "right"),
    ] {
        let vote: serde_json::Value = post_json(
            &http,
            &set.members[index].url,
            "/v1/ledger/propose",
            &json!({
                "transactions": [tx],
                "sequence": 1,
                "ballot": { "number": 1, "proposer": proposer },
            }),
        )
        .await?;
        assert_eq!(
            vote.get("accepted").and_then(serde_json::Value::as_bool),
            Some(true),
            "validator {index} refused a valid first entry: {vote}"
        );
    }

    // Nobody certified anything, so the chain is still at genesis and every
    // seat is holding a vote for an entry that will never exist. A client that
    // arrives now must still be able to get a transaction settled.
    let net = LedgerNetwork::new(set.clone())?;
    assert_eq!(net.head_quorum().await?.sequence, 0);

    let settled = net
        .transact(community_mint(
            &node_bin,
            &validator_homes,
            &validators_path,
            "job_split_after",
        )?)
        .await
        .context("a height split between two proposers stayed wedged")?;
    // Height 1 was not thrown away: the arriving client adopted the entry the
    // set had already half-signed and finished it, then settled its own batch
    // at 2. Nothing was lost and nothing was decided twice.
    assert_eq!(settled.entry.sequence, 2);

    for mut validator in validators {
        validator.kill();
    }
    Ok(())
}

/// Two clients that know nothing about each other, reaching for the same
/// height at the same moment.
///
/// Neither is a leader and neither defers to the other; they simply keep
/// climbing until each has a height of its own. Both settlements have to land,
/// and they have to land at different heights.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_independent_proposers_both_settle() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = Client::new();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, set) =
        create_validator_set(&tmp, &validator_bin, &validator_ports)?;
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

    // Separate clients, so separate ballot lines - exactly the situation a
    // single coordinator process serialises away and a real deployment does
    // not.
    let first = LedgerNetwork::new(set.clone())?;
    let second = LedgerNetwork::new(set.clone())?;

    let a = community_mint(&node_bin, &validator_homes, &validators_path, "job_race_a")?;
    let b = community_mint(&node_bin, &validator_homes, &validators_path, "job_race_b")?;
    let (ra, rb) = tokio::join!(first.transact(a), second.transact(b));
    let (ra, rb) = (ra?, rb?);
    assert_ne!(
        ra.entry.sequence, rb.entry.sequence,
        "two proposers settled at the same height"
    );
    assert_eq!(first.head_quorum().await?.sequence, 2);

    for mut validator in validators {
        validator.kill();
    }
    Ok(())
}

/// A sponsored community mint, built the way the coordinator builds one.
fn community_mint(
    node_bin: &Path,
    homes: &[PathBuf],
    validators_path: &Path,
    job_id: &str,
) -> Result<LedgerTransaction> {
    let work = WorkSpec::PrimeCount { start: 2, end: 100 };
    let shards = 1u32;
    let cost: i64 = split_work(&work, shards).iter().map(work_cost_mcu).sum();
    let sponsors = community_vouches(node_bin, homes, validators_path, job_id, (2, 100, shards))?;
    Ok(LedgerTransaction {
        transaction_id: format!("community_reserve_{job_id}"),
        kind: TransactionKind::CommunityReserve,
        postings: vec![
            Posting {
                account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                delta_mcu: -cost,
            },
            Posting {
                account_id: escrow_account(job_id),
                delta_mcu: cost,
            },
        ],
        evidence: TransactionEvidence::CommunityReserve {
            job_id: job_id.to_string(),
            work: work.clone(),
            shards,
            sponsors,
        },
        created_at: now_unix(),
    })
}
