use anyhow::{Context, Result, anyhow, bail};
use hocmesh_core::{
    compute::{execute_work, split_work, work_cost_mcu},
    identity::NodeIdentity,
};
use hocmesh_ledger::{
    network::LedgerNetwork,
    types::{
        AccountHistory, COMMUNITY_ISSUANCE_ACCOUNT, LedgerHead, LedgerTransaction, Posting,
        ProviderRewardEvidence, QuorumCertificate, TransactionEvidence, TransactionKind,
        ValidatorMember, ValidatorSet, ValidatorSignature, escrow_account,
    },
    validate::{build_entry, membership_hash},
};
use hocmesh_protocol::{
    AuthProof, BalanceResponse, ErrorResponse, JobStatusResponse, NodeCapabilities, PollRequest,
    PollResponse, ReconciliationResponse, RegisterRequest, ResultRequest, SubmitJobRequest,
    SubmitJobResponse, WorkAssignment, WorkResult, WorkSpec, canonical_auth_message,
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
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
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
    let http = test_client();

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

    // A second workload type, run through the same pipeline as the primes
    // above. The point is not that Collatz is interesting: it is that adding
    // a workload costs a spec, a result, and an audit rule, and nothing in
    // scheduling, settlement or the ledger has to know it exists.
    let collatz = submit(
        &http,
        &coordinator,
        &node_a,
        WorkSpec::CollatzPeak {
            start: 1,
            end: 1_000,
        },
        2,
    )
    .await?;
    assert!(collatz.reserved_mcu > 0, "a collatz job must cost CU");
    complete_next_for_job(&http, &coordinator, &node_b, &collatz.job_id)
        .await?
        .context("node B should complete a collatz shard")?;
    complete_next_for_job(&http, &coordinator, &node_c, &collatz.job_id)
        .await?
        .context("node C should complete a collatz shard")?;
    let collatz_status = job_status(&http, &coordinator, &collatz.job_id).await?;
    assert_eq!(collatz_status.status, "completed");
    // 871 is the longest trajectory starting below 1000, at 178 steps, and
    // the rollup has to survive being reassembled from two separate shards
    // that each only saw half the range.
    let peak = collatz_status
        .collatz_peak
        .context("a completed collatz job must report a peak")?;
    assert_eq!(peak.steps, 178);
    assert_eq!(peak.seed, 871);

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

    // And the checkpoint has to be portable, because the reason to have one is
    // that somebody else can start from it. Write it out, hand the file to an
    // empty database, and that database has to end up somewhere it could never
    // have reached on its own: this mirror no longer holds the history below
    // the checkpoint, so nothing here could have replayed it.
    let snapshot = tmp.path.join("ledger-snapshot.json");
    run_ok(
        Command::new(&node_bin)
            .arg("ledger-snapshot")
            .arg("--db")
            .arg(&mirror)
            .arg("--validators")
            .arg(&validators_path)
            .arg("--out")
            .arg(&snapshot),
        "write a snapshot of the checkpointed ledger",
    )?;

    let newcomer = tmp.path.join("newcomer.db");
    let newcomer = newcomer.to_string_lossy().to_string();
    run_ok(
        Command::new(&node_bin)
            .arg("ledger-restore")
            .arg("--db")
            .arg(&newcomer)
            .arg("--validators")
            .arg(&validators_path)
            .arg("--snapshot")
            .arg(&snapshot),
        "start an empty ledger from the snapshot",
    )?;

    for (cmd, label) in [
        ("ledger-sync", "catch the newcomer up from the checkpoint"),
        ("ledger-audit", "audit the newcomer from its checkpoint"),
    ] {
        run_ok(
            Command::new(&node_bin)
                .arg(cmd)
                .arg("--db")
                .arg(&newcomer)
                .arg("--validators")
                .arg(&validators_path),
            label,
        )?;
    }

    // A restore is for a database with nothing in it. Pointed at one that is
    // already carrying a ledger it has to refuse, or the same command that
    // bootstraps a newcomer would also be the one that rewrites an operator's
    // history out from under them.
    assert!(
        run_ok(
            Command::new(&node_bin)
                .arg("ledger-restore")
                .arg("--db")
                .arg(&newcomer)
                .arg("--validators")
                .arg(&validators_path)
                .arg("--snapshot")
                .arg(&snapshot),
            "restore over a populated ledger",
        )
        .is_err(),
        "a ledger that already had a chain was overwritten by a snapshot"
    );

    // Balances say where an account stands; an operator reconciling a bill
    // needs to see how it got there. The index has to agree with the chain it
    // is an index over -- both the validator serving it and the mirror the node
    // reads locally -- or it is just a second, softer set of books.
    let earner = node_a.identity.node_id();
    let served: AccountHistory = get_json(
        &http,
        &format!("http://127.0.0.1:{}", validator_ports[0]),
        &format!("/v1/ledger/history/{earner}?limit=100"),
    )
    .await?;
    assert!(
        !served.entries.is_empty(),
        "an account that earned and spent has to have postings behind it"
    );
    let net: i64 = served.entries.iter().map(|e| e.delta_mcu).sum();
    assert_eq!(
        net,
        balance(&http, &coordinator, &node_a).await?.balance_mcu,
        "the postings have to add up to the balance the quorum reports"
    );
    assert!(
        served.entries.iter().any(|e| e.delta_mcu > 0)
            && served.entries.iter().any(|e| e.delta_mcu < 0),
        "this node both earned and spent, so both directions have to show"
    );

    // The same history through the node CLI, once out of the local mirror and
    // once off the network, has to be the same history.
    let from_mirror = run_capture(
        Command::new(&node_bin)
            .arg("ledger-history")
            .arg("--db")
            .arg(&mirror)
            .arg("--account")
            .arg(&earner),
        "read account history out of the local mirror",
    )?;
    let from_network = run_capture(
        Command::new(&node_bin)
            .arg("ledger-history")
            .arg("--validators")
            .arg(&validators_path)
            .arg("--account")
            .arg(&earner),
        "read account history off the validator network",
    )?;
    assert_eq!(
        from_mirror, from_network,
        "a mirror and the network it mirrors must not disagree about history"
    );
    for entry in &served.entries {
        assert!(
            from_mirror.contains(&entry.transaction_id),
            "the CLI dropped a posting the validator served: {}",
            entry.transaction_id
        );
    }

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
    let http = test_client();
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

    // A structurally broken intent: its claim key does not derive from its own
    // transaction, so it can never settle under that key. Queued ahead of the
    // real one on purpose -- before this was fault-isolated, one row like this
    // stopped every intent behind it on every pass, forever.
    let poisoned = "claim_poisoned_for_reconciliation";
    {
        let conn = rusqlite::Connection::open(&coordinator_db)?;
        conn.execute(
            "INSERT INTO ledger_intents(claim_key,intent_kind,object_id,transaction_json,status,created_at,updated_at) \
             VALUES(?1,?2,?3,?4,'pending',0,0)",
            rusqlite::params![
                poisoned,
                "community_reserve",
                "job_that_never_existed",
                serde_json::to_string(&community_mint(&node_bin, &validator_homes, &validators_path, "job_that_never_existed")?)?,
            ],
        )?;
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

    // The pass must have finished the healthy intent and parked the broken one,
    // rather than dying on the broken one and never reaching the healthy one.
    {
        let conn = rusqlite::Connection::open(&coordinator_db)?;
        let (status, last_error): (String, Option<String>) = conn.query_row(
            "SELECT status,last_error FROM ledger_intents WHERE claim_key=?1",
            rusqlite::params![poisoned],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        assert_eq!(
            status, "unrecoverable",
            "an intent whose claim key cannot derive from its own transaction has to stop being retried"
        );
        assert!(
            last_error.is_some_and(|e| e.contains("claim mismatch")),
            "a parked intent has to say why it was parked"
        );
        let healthy: String = conn.query_row(
            "SELECT status FROM ledger_intents WHERE object_id=?1",
            rusqlite::params![seed_job],
            |r| r.get(0),
        )?;
        assert_eq!(
            healthy, "certified",
            "the intent queued behind the broken one still had to settle"
        );
    }

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

    // The same picture over HTTP: the parked intent is visible to an operator,
    // and there is no endpoint to push it through -- that would be the
    // coordinator ruling on CU, which it is never allowed to do.
    let view: ReconciliationResponse =
        get_json(&http, &coordinator, "/v1/ledger/reconciliation").await?;
    let parked = view
        .unsettled
        .iter()
        .find(|i| i.claim_key == poisoned)
        .context("the parked intent has to show up in the reconciliation view")?;
    assert_eq!(parked.status, "unrecoverable");
    assert!(
        !view.unsettled.iter().any(|i| i.object_id == seed_job),
        "a settled intent is not unfinished business"
    );
    let printed = run_capture(
        Command::new(&node_bin)
            .arg("--coordinator")
            .arg(&coordinator)
            .arg("reconciliation"),
        "operator view of stuck intents",
    )?;
    assert!(printed.contains(poisoned), "{printed}");

    drop(validators);
    Ok(())
}

/// A coordinator is a cache, so losing one must not lose a job.
///
/// Every fact a scheduler needs is already on the chain: a reservation names
/// the job, its spec and its shard count, and a reward names the shard it
/// settled. What this proves is the part that matters for CU -- the shard
/// that was already paid is never handed out a second time, and the job
/// finishes on a database that was empty when the old coordinator died.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replacement_coordinator_rebuilds_from_the_chain_and_finishes_the_job() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, _set) =
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

    let original_db = tmp.path.join("coordinator-original.db");
    let seed_job = "job_rebuild_seed";
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
            .arg(&original_db)
            .arg("--validators")
            .arg(&validators_path)
            .arg("--start")
            .arg("2")
            .arg("--end")
            .arg("200000")
            .arg("--shards")
            .arg("4"),
        "seed community work so the requester can pay",
    )?;

    let first_port = free_port()?;
    let first = ProcessGuard::spawn(
        Command::new(&coordinator_bin)
            .arg("serve")
            .arg("--db")
            .arg(&original_db)
            .arg("--listen")
            .arg(format!("127.0.0.1:{first_port}"))
            .arg("--validators")
            .arg(&validators_path),
    )?;
    wait_health(&http, first_port).await?;
    let first_url = format!("http://127.0.0.1:{first_port}");

    let requester = TestNode::new(&tmp.path.join("rebuild-requester"))?;
    let worker = TestNode::new(&tmp.path.join("rebuild-worker"))?;
    register(&http, &first_url, &requester).await?;
    register(&http, &first_url, &worker).await?;

    let community = poll_until_assignment(&http, &first_url, &requester, Some(seed_job)).await?;
    complete_assignment(&http, &first_url, &requester, &community).await?;
    let funded = balance(&http, &first_url, &requester).await?;
    assert!(
        funded.balance_mcu > 0,
        "the requester has to earn before it can spend"
    );

    let work = WorkSpec::PrimeCount {
        start: 2,
        end: 20_000,
    };
    let paid = submit(&http, &first_url, &requester, work.clone(), 3).await?;
    assert!(paid.reserved_mcu > 0);

    let settled = complete_next_for_job(&http, &first_url, &worker, &paid.job_id)
        .await?
        .context("one shard should finish before the coordinator dies")?;
    let paid_once = balance(&http, &first_url, &worker).await?.balance_mcu;
    assert!(
        paid_once > 0,
        "the finished shard should have been rewarded"
    );

    // The coordinator dies with the job half done, and its database dies with
    // it. Nothing below reads that file again.
    drop(first);

    let rebuilt_db = tmp.path.join("coordinator-rebuilt.db");
    run_ok(
        Command::new(&coordinator_bin)
            .arg("rebuild")
            .arg("--db")
            .arg(&rebuilt_db)
            .arg("--validators")
            .arg(&validators_path),
        "rebuild scheduling state from the chain",
    )?;
    run_ok(
        Command::new(&coordinator_bin)
            .arg("rebuild")
            .arg("--db")
            .arg(&rebuilt_db)
            .arg("--validators")
            .arg(&validators_path),
        "repeat the rebuild idempotently",
    )?;

    let second_port = free_port()?;
    let _second = ProcessGuard::spawn(
        Command::new(&coordinator_bin)
            .arg("serve")
            .arg("--db")
            .arg(&rebuilt_db)
            .arg("--listen")
            .arg(format!("127.0.0.1:{second_port}"))
            .arg("--validators")
            .arg(&validators_path),
    )?;
    wait_health(&http, second_port).await?;
    let second_url = format!("http://127.0.0.1:{second_port}");

    // A rebuilt node row is a placeholder the scheduler ignores, so workers
    // have to come back and say what they can actually do.
    register(&http, &second_url, &requester).await?;
    register(&http, &second_url, &worker).await?;

    let recovered = job_status(&http, &second_url, &paid.job_id).await?;
    assert_eq!(
        recovered.requester_node_id.as_deref(),
        Some(requester.identity.node_id().as_str())
    );
    assert!(!recovered.system_funded, "a paid job must not become free");
    assert_eq!(recovered.reserved_mcu, paid.reserved_mcu);
    assert_eq!(recovered.total_assignments, 3);
    assert_eq!(
        recovered.completed_assignments, 1,
        "the replacement must already know one shard is settled"
    );

    // Re-delivering a shard the chain already paid for is refused, and refused
    // on a database that never saw the original delivery: the rebuild carried
    // the settled status forward, so the second claim has nothing to earn.
    let replay_hash = result_body_hash(
        &settled.assignment_id,
        &settled.job_id,
        settled.shard_index,
        &settled.work,
        settled.reward_mcu,
        settled.system_funded,
        &settled.result,
    )?;
    let replay = ResultRequest {
        auth: worker.identity.auth("result", &replay_hash),
        ..settled.clone()
    };
    let refused = post_result_raw(&http, &second_url, &replay).await;
    assert!(
        refused.is_err(),
        "the replacement must refuse a shard the chain already settled"
    );
    assert_eq!(
        balance(&http, &second_url, &worker).await?.balance_mcu,
        paid_once,
        "a rebuild must not let the same shard be paid twice"
    );

    let mut finished = vec![settled.shard_index];
    for _ in 0..2 {
        let next = complete_next_for_job(&http, &second_url, &worker, &paid.job_id)
            .await?
            .context("the replacement should hand out the shards nobody finished")?;
        assert_ne!(
            next.assignment_id, settled.assignment_id,
            "a shard the chain already paid for must never be offered again"
        );
        finished.push(next.shard_index);
    }
    finished.sort_unstable();
    assert_eq!(
        finished,
        vec![0, 1, 2],
        "the rebuilt schedule should cover the job exactly once"
    );

    let done = job_status(&http, &second_url, &paid.job_id).await?;
    assert_eq!(done.status, "completed");
    assert_eq!(done.completed_assignments, 3);
    let expected = match execute_work(&work) {
        WorkResult::PrimeCount { count, .. } => count,
        other => bail!("prime work returned {other:?}"),
    };
    assert_eq!(
        done.prime_count_total,
        Some(expected),
        "a job finished across two coordinators must still be the right answer"
    );

    // The chain, not the coordinator, is where payment lives: the worker was
    // paid once for the shard it finished before the crash and twice more
    // after, and no rebuild added a fourth.
    let ledger_paid = balance(&http, &second_url, &worker).await?.balance_mcu;
    assert!(
        ledger_paid > paid_once,
        "the shards finished after the rebuild should have been paid"
    );

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
    let urls: Vec<String> = validator_ports
        .iter()
        .map(|port| format!("http://127.0.0.1:{port}"))
        .collect();
    create_validator_set_at(tmp, validator_bin, &urls)
}

/// Build a set whose advertised URLs are not the ports the validators listen
/// on.
///
/// The advertised URL is part of the signed membership, so a proxy cannot be
/// slipped in front of a live set after the fact: every node has to agree on
/// the same addresses or the membership hashes stop matching. Pointing the
/// whole set at relay ports from the start is how a test gets to break the
/// wire underneath a network that is already running.
fn create_validator_set_at(
    tmp: &TestDir,
    validator_bin: &Path,
    urls: &[String],
) -> Result<(PathBuf, Vec<PathBuf>, Vec<PathBuf>, ValidatorSet)> {
    let mut members = Vec::new();
    let mut validator_homes = Vec::new();
    let mut validator_dbs = Vec::new();
    for (index, url) in urls.iter().enumerate() {
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
            "url": url,
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
        memory_bandwidth_bytes_per_second: None,
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

/// Build the three binaries these tests actually launch.
///
/// Named rather than `--workspace --bins`, which would also build the desktop
/// app and so make a headless machine unable to run a test that never opens a
/// window.
fn build_bins(workspace: &Path) -> Result<()> {
    run_ok(
        Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .arg("build")
            .arg("--ignore-rust-version")
            .arg("--bins")
            .arg("-p")
            .arg("hocmesh")
            .arg("-p")
            .arg("hocmesh-coordinator")
            .arg("-p")
            .arg("hocmesh-validator")
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

/// Runs a node command and hands back its stdout.
///
/// `run_ok` throws stdout away, which is fine for commands whose only product
/// is a side effect; a command whose product *is* the output needs this.
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
    let http = test_client();
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
    let http = test_client();
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
    let http = test_client();
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
    let http = test_client();
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
    // Both proposals carry a certificate, but a validator can still be
    // finishing the write that makes the newer one visible to a reader. The
    // claim being tested is that two racing proposers converge, not that they
    // converge before anyone has finished writing anything down.
    let mut converged = false;
    for _ in 0..100 {
        if matches!(first.head_quorum().await, Ok(head) if head.sequence == 2) {
            converged = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(converged, "the quorum never agreed on the second entry");

    for mut validator in validators {
        validator.kill();
    }
    Ok(())
}

/// A client that loses sight of its own settlement has to be able to learn it
/// happened.
///
/// This is the ordinary end of a round nobody observed: the transaction is
/// applied, the certificate never gets back, the client climbs, and the
/// validators turn it away with "claim already settled" -- its own success,
/// refusing it. Told "rejected" about work the ledger did, a caller runs a job
/// that is already paid for and reports a reservation that exists as missing.
/// So the second attempt has to come back with the certificate of the entry
/// that already carries the transaction, at the height it landed at, and no
/// second entry may appear.
///
/// The refusal still has to mean something, though. A transaction that is not
/// the settled one, under a claim key that is spent, is a coordinator paying
/// one assignment twice under different numbers -- and that is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_settled_transaction_resolves_to_its_entry_and_an_impostor_does_not() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
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

    let client = LedgerNetwork::new(set.clone())?;
    let tx = community_mint(&node_bin, &validator_homes, &validators_path, "job_idem")?;
    let landed = client.transact(tx.clone()).await?;

    // The same transaction again. The validators have it and will refuse it.
    let again = client
        .transact(tx.clone())
        .await
        .context("a client was refused its own settled transaction")?;
    assert_eq!(
        (again.entry.sequence, &again.entry.entry_hash),
        (landed.entry.sequence, &landed.entry.entry_hash),
        "the resubmission has to resolve to the entry that carried it"
    );

    // A client that never saw the round reaches the same answer, because the
    // answer comes from the quorum and not from anything remembered locally.
    let stranger = LedgerNetwork::new(set.clone())?;
    let third = stranger
        .transact(tx.clone())
        .await
        .context("a client that never saw the round could not resolve the claim")?;
    assert_eq!(third.entry.sequence, landed.entry.sequence);

    // Same claim key, different transaction: refused, and still refused after
    // the resolution path has had its look at it.
    let mut impostor = tx.clone();
    impostor.transaction_id = format!("{}_again", tx.transaction_id);
    let refused = client.transact(impostor).await;
    assert!(
        refused.is_err(),
        "a different transaction settled under a claim that was already spent: {refused:?}"
    );

    // Nothing was applied twice.
    assert_eq!(
        client.head_quorum().await?.sequence,
        landed.entry.sequence,
        "resubmitting a settled transaction added an entry"
    );

    for mut validator in validators {
        validator.kill();
    }
    Ok(())
}

/// A quorum that has not agreed on a head *yet* is not a quorum that refused.
///
/// Two proposers reaching for one height leave the validators briefly split
/// across it, and so does any hiccup that puts the set below threshold for a
/// moment. A client that reads either as final abandons a transaction the
/// chain was about to accept -- the failure mode is silent, and it strands
/// work nobody rejected.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_waits_out_a_head_the_quorum_has_not_agreed_on() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let mut links = Vec::new();
    for port in validator_ports {
        links.push(FaultLink::to(port).await?);
    }
    let urls: Vec<String> = links.iter().map(|l| l.url()).collect();
    let (validators_path, validator_homes, validator_dbs, set) =
        create_validator_set_at(&tmp, &validator_bin, &urls)?;
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

    // Settle one entry the ordinary way, so the head the client later has to
    // wait for is a real one rather than genesis.
    let client = LedgerNetwork::new(set.clone())?;
    let warmup = community_mint(&node_bin, &validator_homes, &validators_path, "job_wait_a")?;
    client.transact(warmup).await?;
    let contested = community_mint(&node_bin, &validator_homes, &validators_path, "job_wait_b")?;

    // Drop the set one seat below threshold. Nothing has refused anything --
    // there is simply no head three of four validators will agree on.
    links[0].cut();
    links[1].cut();
    assert!(
        client.head_quorum().await.is_err(),
        "the split has to be real before the client is asked to survive it"
    );

    // Heal while the client is mid-round. With the split treated as a refusal
    // this transaction is already lost by now; treated as a deferral it lands.
    let healer = tokio::spawn({
        let (a, b) = (links[0].cut.clone(), links[1].cut.clone());
        async move {
            tokio::time::sleep(Duration::from_millis(120)).await;
            a.store(false, Ordering::SeqCst);
            b.store(false, Ordering::SeqCst);
        }
    });
    let settled = client.transact(contested).await;
    healer.await?;
    let cert = settled.context("a client gave up on a head the quorum was about to agree on")?;
    assert_eq!(
        cert.entry.sequence, 2,
        "the deferred transaction has to land on the entry after the warmup"
    );
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

/// Settlement must survive the two things a real network always has: distance,
/// and a peer that goes away.
///
/// Every coordinator-to-validator hop is put behind 45 ms of one-way latency,
/// which is a plausible cross-continent link and about 90 ms per round trip.
/// Then one of the four validators is cut off from the coordinator entirely.
/// Three remain, the threshold is three, so work must still settle.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn settlement_survives_wan_latency_and_a_minority_partition() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let mut links = Vec::new();
    for port in validator_ports {
        links.push(FaultLink::to(port).await?);
    }
    let urls: Vec<String> = links.iter().map(|l| l.url()).collect();
    let (validators_path, validator_homes, validator_dbs, set) =
        create_validator_set_at(&tmp, &validator_bin, &urls)?;

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

    for link in &links {
        link.set_one_way_latency(45);
    }

    let coordinator_db = tmp.path.join("coordinator-wan.db");
    let seed_job = "job_wan_seed";
    let seed_sponsors = sponsors_file(
        &tmp.path,
        &node_bin,
        &validator_homes,
        &validators_path,
        seed_job,
        (2, 50000, 4),
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
            .arg("50000")
            .arg("--shards")
            .arg("4"),
        "seed community job across a delayed link",
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

    let node = TestNode::new(&tmp.path.join("wan-node"))?;
    register(&http, &coordinator, &node).await?;

    // A full round of settlement across four delayed links.
    let first = poll_until_assignment(&http, &coordinator, &node, Some(seed_job)).await?;
    complete_assignment(&http, &coordinator, &node, &first).await?;
    let after_first = balance(&http, &coordinator, &node).await?.balance_mcu;
    assert!(after_first > 0, "a delayed link must not stop settlement");

    // One validator is now unreachable from the coordinator. Three of four
    // remain and the threshold is three, so the next shard must still settle.
    links[0].cut();
    let second = poll_until_assignment(&http, &coordinator, &node, Some(seed_job)).await?;
    complete_assignment(&http, &coordinator, &node, &second).await?;
    let after_second = balance(&http, &coordinator, &node).await?.balance_mcu;
    assert!(
        after_second > after_first,
        "losing a minority of the quorum must not stop settlement"
    );

    // The link comes back. The three validators that stayed reachable have
    // been settling all along, so they agree; the isolated one is strictly
    // behind, because a replica that missed commits does not invent them.
    links[0].heal();
    let mut connected = None;
    for _ in 0..100 {
        // A request across a link that was just cut and healed can hang on a
        // socket the partition left behind, and one such request must not eat
        // the whole retry budget. Bound it and ask again.
        let fetch = tokio::time::timeout(Duration::from_secs(5), validator_heads(&http, &set));
        let Ok(Ok(heads)) = fetch.await else {
            tokio::time::sleep(Duration::from_millis(100)).await;
            continue;
        };
        if heads[1].sequence > 0 && heads[1..].iter().all(|h| h.sequence == heads[1].sequence) {
            connected = Some(heads[1].sequence);
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let connected = connected.context("the reachable validators never agreed on a head")?;
    let behind = validator_heads(&http, &set).await?[0].sequence;
    assert!(
        behind < connected,
        "a validator cut off from the quorum must fall behind, not keep pace"
    );

    // Catching it up is the documented repair: stop it, replay from its
    // peers, start it again. A healed link does not do that on its own.
    drop(validators.remove(0));
    run_ok(
        Command::new(&validator_bin)
            .arg("sync")
            .arg("--db")
            .arg(&validator_dbs[0])
            .arg("--validators")
            .arg(&validators_path),
        "replay the isolated validator from its peers",
    )?;
    validators.push(
        start_validator(
            &validator_bin,
            &validators_path,
            &validator_homes[0],
            &validator_dbs[0],
            validator_ports[0],
            &http,
        )
        .await?,
    );
    let healed = validator_heads(&http, &set).await?;
    assert!(
        healed.iter().all(|h| h.sequence == connected),
        "every validator should hold the same head once the laggard has replayed"
    );

    drop(validators);
    Ok(())
}

/// Losing a majority of the quorum must stop settlement, not fake it.
///
/// The worker takes an assignment while the network is whole, then the
/// coordinator is cut off from two of four validators - one short of the
/// three it needs. Delivery must fail. When the link comes back the pending
/// settlement must complete, and it must be worth exactly one shard - not
/// two, and not none.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_majority_partition_stops_settlement_and_recovery_pays_once() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let mut links = Vec::new();
    for port in validator_ports {
        links.push(FaultLink::to(port).await?);
    }
    let urls: Vec<String> = links.iter().map(|l| l.url()).collect();
    let (validators_path, validator_homes, validator_dbs, _set) =
        create_validator_set_at(&tmp, &validator_bin, &urls)?;

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

    for link in &links {
        link.set_one_way_latency(5);
    }

    let coordinator_db = tmp.path.join("coordinator-partition.db");
    let seed_job = "job_partition_seed";
    let seed_sponsors = sponsors_file(
        &tmp.path,
        &node_bin,
        &validator_homes,
        &validators_path,
        seed_job,
        (2, 50000, 4),
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
            .arg("50000")
            .arg("--shards")
            .arg("4"),
        "seed community job before partitioning the quorum",
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

    let node = TestNode::new(&tmp.path.join("partition-node"))?;
    register(&http, &coordinator, &node).await?;

    let first = poll_until_assignment(&http, &coordinator, &node, Some(seed_job)).await?;
    complete_assignment(&http, &coordinator, &node, &first).await?;
    let earned = balance(&http, &coordinator, &node).await?.balance_mcu;
    assert!(earned > 0);

    // Take the next shard while the network is whole, then remove the
    // coordinator's majority before delivering it.
    let stranded = poll_until_assignment(&http, &coordinator, &node, Some(seed_job)).await?;
    links[0].cut();
    links[1].cut();

    let refused = complete_assignment(&http, &coordinator, &node, &stranded).await;
    assert!(
        refused.is_err(),
        "settlement must fail outright when the quorum is out of reach"
    );

    links[0].heal();
    links[1].heal();

    // Recovery asks the validators whether the persisted transaction was ever
    // certified, and finishes it either way. Running it twice must not pay
    // twice: the reward claim key is the shard, and the ledger owns it.
    for label in [
        "reconcile the stranded settlement",
        "repeat it idempotently",
    ] {
        run_ok(
            Command::new(&coordinator_bin)
                .arg("recover")
                .arg("--db")
                .arg(&coordinator_db)
                .arg("--validators")
                .arg(&validators_path),
            label,
        )?;
    }

    let settled = balance(&http, &coordinator, &node).await?.balance_mcu;
    assert_eq!(
        settled,
        earned + stranded.reward_mcu,
        "the stranded shard must be worth exactly one reward once the link heals"
    );

    drop(validators);
    Ok(())
}

/// Machines in different places do not agree on the time.
///
/// Signatures are bound to a timestamp, so a node whose clock has drifted far
/// enough is indistinguishable from someone replaying yesterday's request.
/// The live API therefore holds a bounded skew window. This checks both edges
/// of it against a running coordinator: a few minutes of drift is tolerated,
/// and drift past the window is refused even though the signature is real.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drifting_clock_is_tolerated_up_to_the_window_and_refused_past_it() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, _set) =
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

    let coordinator_db = tmp.path.join("coordinator-skew.db");
    let seed_job = "job_skew_seed";
    let seed_sponsors = sponsors_file(
        &tmp.path,
        &node_bin,
        &validator_homes,
        &validators_path,
        seed_job,
        (2, 50000, 4),
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
            .arg("50000")
            .arg("--shards")
            .arg("4"),
        "seed community job for the skew test",
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

    let node = TestNode::new(&tmp.path.join("skew-node"))?;
    register(&http, &coordinator, &node).await?;

    // Signed by the right key, for the right action, with the right body -
    // and refused anyway, because the clock behind it is hours out.
    let far_past = now_unix() - hocmesh_protocol::AUTH_MAX_CLOCK_SKEW_SECS - 3_600;
    let stale = PollRequest {
        auth: skewed_auth(&node, "poll", &empty_body_hash(), far_past),
    };
    let rejected: Result<PollResponse> =
        post_json(&http, &coordinator, "/v1/work/poll", &stale).await;
    assert!(
        rejected.is_err(),
        "a clock hours out of step must not be able to claim work"
    );

    // A machine a couple of minutes fast is ordinary, not hostile, and it
    // still has to be able to work.
    let mut claimed = None;
    for _ in 0..100 {
        let drifted = now_unix() + hocmesh_protocol::AUTH_MAX_CLOCK_SKEW_SECS / 2;
        let ahead = PollRequest {
            auth: skewed_auth(&node, "poll", &empty_body_hash(), drifted),
        };
        let response: PollResponse =
            post_json(&http, &coordinator, "/v1/work/poll", &ahead).await?;
        if let Some(assignment) = response.assignment {
            claimed = Some(assignment);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let claimed = claimed.context("a clock inside the window still has to get work")?;
    assert_eq!(claimed.job_id, seed_job);

    drop(validators);
    Ok(())
}

/// `NodeIdentity::auth` always stamps the current time. A node with a wrong
/// clock stamps a wrong time and signs it just as honestly, which is what this
/// reproduces: a real signature over a timestamp the caller chooses.
fn skewed_auth(node: &TestNode, action: &str, body_hash: &str, timestamp: i64) -> AuthProof {
    let node_id = node.identity.node_id();
    static SKEW_NONCE: AtomicU64 = AtomicU64::new(0);
    let unique = SKEW_NONCE.fetch_add(1, Ordering::Relaxed);
    let nonce_b64 = format!("skew-nonce-{timestamp}-{unique:016}");
    let msg = canonical_auth_message(action, &node_id, timestamp, &nonce_b64, body_hash);
    let signature_b64 = node.identity.sign_bytes_b64(msg.as_bytes());
    AuthProof {
        node_id,
        timestamp,
        nonce_b64,
        signature_b64,
    }
}

/// A TCP relay that can be told to behave like a wide-area link.
///
/// Loopback never produces the conditions that break distributed systems: no
/// propagation delay, no peer that simply stops answering. The relay sits in
/// front of a validator and forwards bytes both ways, delaying each burst by
/// a configurable one-way latency and cutting the link on demand - including
/// mid-request, which is what a real partition does.
struct FaultLink {
    port: u16,
    cut: Arc<AtomicBool>,
    one_way_ms: Arc<AtomicU64>,
    accept: tokio::task::JoinHandle<()>,
}

impl FaultLink {
    async fn to(target: u16) -> Result<Self> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
        let port = listener.local_addr()?.port();
        let cut = Arc::new(AtomicBool::new(false));
        let one_way_ms = Arc::new(AtomicU64::new(0));
        let accept = tokio::spawn(accept_loop(
            listener,
            target,
            cut.clone(),
            one_way_ms.clone(),
        ));
        Ok(Self {
            port,
            cut,
            one_way_ms,
            accept,
        })
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Delay every burst of bytes in each direction, so a round trip costs
    /// roughly twice this.
    fn set_one_way_latency(&self, ms: u64) {
        self.one_way_ms.store(ms, Ordering::SeqCst);
    }

    fn cut(&self) {
        self.cut.store(true, Ordering::SeqCst);
    }

    fn heal(&self) {
        self.cut.store(false, Ordering::SeqCst);
    }
}

impl Drop for FaultLink {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

async fn accept_loop(
    listener: tokio::net::TcpListener,
    target: u16,
    cut: Arc<AtomicBool>,
    one_way_ms: Arc<AtomicU64>,
) {
    loop {
        // A transient accept error must not retire the link for good: a test
        // whose network quietly stopped existing reports a partition that was
        // never asked for.
        let Ok((inbound, _)) = listener.accept().await else {
            continue;
        };
        if cut.load(Ordering::SeqCst) {
            continue;
        }
        tokio::spawn(relay(inbound, target, cut.clone(), one_way_ms.clone()));
    }
}

async fn relay(
    mut inbound: tokio::net::TcpStream,
    target: u16,
    cut: Arc<AtomicBool>,
    one_way_ms: Arc<AtomicU64>,
) {
    let Ok(mut outbound) = tokio::net::TcpStream::connect(("127.0.0.1", target)).await else {
        return;
    };
    let (mut from_client, mut to_client) = inbound.split();
    let (mut from_server, mut to_server) = outbound.split();
    let up = pump(&mut from_client, &mut to_server, &cut, &one_way_ms);
    let down = pump(&mut from_server, &mut to_client, &cut, &one_way_ms);
    // Whichever direction ends first ends the connection. try_join! waited for
    // both, so when the validator closed an idle keep-alive socket the relay
    // swallowed the close and left the client half open: the next request sent
    // on that pooled socket went nowhere until an outer timeout noticed.
    tokio::select! {
        _ = up => {}
        _ = down => {}
    }
}

/// Copy one direction, honouring latency and noticing a cut between reads.
///
/// The short read timeout is what makes a partition bite mid-request: a
/// connection already open when the link goes down is torn down rather than
/// left to hang until some outer timeout notices.
async fn pump<R, W>(
    reader: &mut R,
    writer: &mut W,
    cut: &AtomicBool,
    one_way_ms: &AtomicU64,
) -> std::io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = vec![0u8; 16 * 1024];
    // Whether the link has been silent long enough that the next bytes start a
    // new message rather than continue one already crossing it.
    let mut quiet = true;
    loop {
        if cut.load(Ordering::SeqCst) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionAborted,
                "link cut",
            ));
        }
        let read = tokio::time::timeout(Duration::from_millis(25), reader.read(&mut buf)).await;
        let n = match read {
            Ok(Ok(0)) => return Ok(()),
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                quiet = true;
                continue;
            }
        };
        let delay = one_way_ms.load(Ordering::SeqCst);
        // Propagation is paid once per message, not once per TCP segment: a
        // continuation arrives behind the bytes that preceded it and has
        // already crossed the wire. Charging every burst made the simulated
        // delay depend on how the kernel happened to chunk the stream, which
        // is a property of the machine rather than of the network under test.
        if delay > 0 && quiet {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        quiet = false;
        writer.write_all(&buf[..n]).await?;
        writer.flush().await?;
    }
}

/// No request in these tests should ever take a minute. A test that hangs on a
/// socket reports nothing; a test that fails on a timeout reports exactly what
/// stopped answering.
fn test_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .expect("an HTTP client with a timeout")
}

/// The reason one validator gives for refusing a transaction.
///
/// `LedgerNetwork::transact` reports a round as "only N valid votes", which is
/// the right thing for a caller to see and useless for a test that wants to
/// know which rule fired. Asking a single seat directly gets the rule.
async fn refusal_reason(
    http: &Client,
    net: &LedgerNetwork,
    url: &str,
    tx: &LedgerTransaction,
) -> Result<String> {
    let head = net.head_quorum().await?;
    let vote: serde_json::Value = post_json(
        http,
        url,
        "/v1/ledger/propose",
        &json!({
            "transactions": [tx],
            "sequence": head.sequence + 1,
            "ballot": { "number": 1, "proposer": "test-liar" },
        }),
    )
    .await?;
    if vote.get("accepted").and_then(serde_json::Value::as_bool) == Some(true) {
        bail!("the validator accepted a transaction the test expected it to refuse: {vote}");
    }
    Ok(vote
        .get("error")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string())
}

/// A validator that signs two different histories at the same height, while
/// the network it belongs to cannot see it doing so.
///
/// This is the failure the threshold exists for. The advertised URL of every
/// seat is part of the signed membership, so a fault link in front of one seat
/// removes it from the network's view - but the seat is still listening on its
/// real port, and an attacker that knows the port can talk to it in private.
/// That is a validator alone with someone who wants a fork.
///
/// Nothing stops it signing. Refusing to sign is not what makes a quorum safe;
/// arithmetic is. Three of four is a quorum, so any two quorums share two
/// seats, so any two quorums share at least one seat that is not the traitor -
/// and that seat has already committed. The equivocating signature is real,
/// verifiable, and worth nothing on its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_equivocating_seat_cannot_fork_a_partitioned_quorum() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let mut links = Vec::new();
    for port in validator_ports {
        links.push(FaultLink::to(port).await?);
    }
    let urls: Vec<String> = links.iter().map(|l| l.url()).collect();
    let (validators_path, validator_homes, validator_dbs, set) =
        create_validator_set_at(&tmp, &validator_bin, &urls)?;

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

    // The seat's real address, which the membership does not mention. Reaching
    // it this way is the whole point: the network believes seat 3 is gone.
    let traitor_direct = format!("http://127.0.0.1:{}", validator_ports[3]);

    let net = LedgerNetwork::new(set.clone())?;
    let genesis = net.head_quorum().await?;
    assert_eq!(genesis.sequence, 0);

    let honest = community_mint(
        &node_bin,
        &validator_homes,
        &validators_path,
        "job_equivocation_honest",
    )?;
    let fork = community_mint(
        &node_bin,
        &validator_homes,
        &validators_path,
        "job_equivocation_fork",
    )?;

    // Cut the seat out of the network, then get it alone.
    links[3].cut();

    let fork_vote: serde_json::Value = post_json(
        &http,
        &traitor_direct,
        "/v1/ledger/propose",
        &json!({
            "transactions": [fork.clone()],
            "sequence": 1,
            "ballot": { "number": 9, "proposer": "byzantine" },
        }),
    )
    .await?;
    assert_eq!(
        fork_vote
            .get("accepted")
            .and_then(serde_json::Value::as_bool),
        Some(true),
        "the partitioned seat should have had no reason to refuse: {fork_vote}"
    );
    let fork_signature = fork_vote
        .get("signature_b64")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("no signature in the fork vote: {fork_vote}"))?
        .to_string();
    let fork_entry_hash = fork_vote
        .get("entry_hash")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    // Meanwhile the three seats that remain reachable settle the real history.
    let settled = net
        .transact(honest.clone())
        .await
        .context("three of four seats must still be a quorum")?;
    assert_eq!(settled.entry.sequence, 1);
    let honest_hash = settled.entry.entry_hash.clone();
    assert_ne!(honest_hash, fork_entry_hash);

    // And now the equivocation proper: the same seat, at the same height,
    // signing the history it just contradicted. It accepts, because a lone
    // validator has no way to know the rest of the set has moved.
    let second_vote: serde_json::Value = post_json(
        &http,
        &traitor_direct,
        "/v1/ledger/propose",
        &json!({
            "transactions": [honest],
            "sequence": 1,
            "ballot": { "number": 10, "proposer": "byzantine" },
        }),
    )
    .await?;
    assert_eq!(
        second_vote
            .get("accepted")
            .and_then(serde_json::Value::as_bool),
        Some(true),
        "the second half of the equivocation was refused, so the test proves nothing: {second_vote}"
    );
    assert_ne!(
        second_vote
            .get("entry_hash")
            .and_then(serde_json::Value::as_str),
        Some(fork_entry_hash.as_str()),
        "two votes at one height that agree are not an equivocation"
    );

    links[3].heal();

    // The attacker now needs two more signatures for its fork, and there is
    // nowhere left to get them: every other seat has committed height 1, so
    // height 1 is no longer a proposal any of them will entertain.
    for index in 0..3 {
        let refusal: serde_json::Value = post_json(
            &http,
            &set.members[index].url,
            "/v1/ledger/propose",
            &json!({
                "transactions": [fork.clone()],
                "sequence": 1,
                "ballot": { "number": 99, "proposer": "byzantine" },
            }),
        )
        .await?;
        assert_eq!(
            refusal.get("accepted").and_then(serde_json::Value::as_bool),
            Some(false),
            "seat {index} was willing to re-vote a height it had already committed: {refusal}"
        );
    }

    // So the best certificate the fork will ever have carries one signature.
    // It is a real signature over a real entry, and it is two short of a
    // quorum: post it and the honest seats do the arithmetic themselves.
    let forged = QuorumCertificate {
        entry: build_entry(1, genesis.entry_hash.clone(), vec![fork])?,
        membership_hash: membership_hash(&set)?,
        signatures: vec![ValidatorSignature {
            validator_id: set.members[3].validator_id.clone(),
            signature_b64: fork_signature,
        }],
    };
    for index in 0..4 {
        let rejected = http
            .post(format!("{}/v1/ledger/commit", set.members[index].url))
            .json(&forged)
            .send()
            .await?;
        assert!(
            !rejected.status().is_success(),
            "seat {index} committed a fork carrying one signature out of a threshold of {}",
            set.threshold
        );
    }

    // Nothing moved. The chain still has exactly one history at height 1, the
    // fork's job was never funded, and the set is still able to settle.
    let head = net.head_quorum().await?;
    assert_eq!(head.sequence, 1);
    assert_eq!(head.entry_hash, honest_hash);
    assert_eq!(
        net.balance_quorum(&escrow_account("job_equivocation_fork"))
            .await?
            .balance_mcu,
        0,
        "a fork that never certified must never have funded anything"
    );
    let after = net
        .transact(community_mint(
            &node_bin,
            &validator_homes,
            &validators_path,
            "job_equivocation_after",
        )?)
        .await
        .context("an equivocating seat must not be able to wedge the set")?;
    assert_eq!(after.entry.sequence, 2);
    assert_eq!(after.entry.previous_hash, honest_hash);

    drop(validators);
    Ok(())
}

/// A coordinator that lies about what it scheduled.
///
/// The coordinator is the one component with no key of consequence. It hands
/// out work, watches results come back, and proposes settlements - but a
/// settlement is only ever a proposal, and every claim inside one is
/// re-derived by the validators from evidence the coordinator did not author.
/// This walks through the lies a corrupted coordinator would actually want to
/// tell.
///
/// The work here is real and the provider's signature over it is real: the
/// test polls a genuine assignment, computes the answer, and signs it, then
/// never gives it to the coordinator. What it builds instead is the exact
/// transaction the coordinator would have built, and then spoils one field at
/// a time. Two of the lies are told with the provider's help, because the
/// interesting question is not whether a coordinator can forge a signature -
/// it cannot - but whether a coordinator and a provider working together can
/// take more than the work was worth.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lying_coordinator_cannot_take_more_than_the_work_was_worth() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
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

    let coordinator_db = tmp.path.join("coordinator-liar.db");
    let job = "job_lying_coordinator";
    let sponsors = sponsors_file(
        &tmp.path,
        &node_bin,
        &validator_homes,
        &validators_path,
        job,
        (2, 50000, 4),
    )?;
    run_ok(
        Command::new(&coordinator_bin)
            .arg("seed")
            .arg("--job-id")
            .arg(job)
            .arg("--sponsors")
            .arg(&sponsors)
            .arg("--db")
            .arg(&coordinator_db)
            .arg("--validators")
            .arg(&validators_path)
            .arg("--start")
            .arg("2")
            .arg("--end")
            .arg("50000")
            .arg("--shards")
            .arg("4"),
        "seed the job the coordinator will lie about",
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

    let worker = TestNode::new(&tmp.path.join("liar-worker"))?;
    let accomplice = TestNode::new(&tmp.path.join("liar-accomplice"))?;
    register(&http, &coordinator, &worker).await?;

    // A real assignment, really computed, really signed - and deliberately
    // never handed back, so the ledger has not yet seen this shard.
    let assignment = poll_until_assignment(&http, &coordinator, &worker, Some(job)).await?;
    let result = execute_work(&assignment.work);
    let reward = work_cost_mcu(&assignment.work);
    assert_eq!(reward, assignment.reward_mcu);

    // Exactly what the coordinator builds on a good day. Every lie below is
    // this transaction with one field moved.
    let lie =
        |result: WorkResult, reward_mcu: i64, assignment_id: &str, payee: &str, auth: AuthProof| {
            LedgerTransaction {
                transaction_id: format!("reward_{}", assignment.assignment_id),
                kind: TransactionKind::ProviderReward,
                postings: vec![
                    Posting {
                        account_id: escrow_account(job),
                        delta_mcu: -reward_mcu,
                    },
                    Posting {
                        account_id: payee.to_string(),
                        delta_mcu: reward_mcu,
                    },
                ],
                evidence: TransactionEvidence::ProviderReward(ProviderRewardEvidence {
                    job_id: job.to_string(),
                    assignment_id: assignment_id.to_string(),
                    shard_index: assignment.shard_index,
                    reward_mcu,
                    provider_public_key_b64: worker.identity.public_key_b64(),
                    provider_auth: auth,
                    work: assignment.work.clone(),
                    result,
                    system_funded: assignment.system_funded,
                    // A coordinator colluding with a provider would pick the
                    // gentlest audit it could find. It is recorded and ignored.
                    provisional_audit_nonce: 0,
                }),
                created_at: now_unix(),
            }
        };
    // The provider signs whatever it is handed, which is what makes the
    // collusion cases possible: every field the coordinator would want to move
    // is inside the body hash, so moving one alone only breaks the signature.
    // A test that stopped there would be testing ed25519, not this system.
    let sign_result =
        |assignment_id: &str, result: &WorkResult, reward_mcu: i64| -> Result<AuthProof> {
            let bh = result_body_hash(
                assignment_id,
                job,
                assignment.shard_index,
                &assignment.work,
                reward_mcu,
                assignment.system_funded,
                result,
            )?;
            Ok(worker.identity.auth("result", &bh))
        };
    let truthful_auth = sign_result(&assignment.assignment_id, &result, reward)?;

    let net = LedgerNetwork::new(set.clone())?;
    let escrow_before = net.balance_quorum(&escrow_account(job)).await?.balance_mcu;
    assert!(escrow_before >= reward);

    // Lie one: the shard was done, but pay somebody else for it. The provider
    // signed a result, not a destination, so the destination is not the
    // coordinator's to choose - it is read back off the signature.
    let stolen = lie(
        result.clone(),
        reward,
        &assignment.assignment_id,
        &accomplice.identity.node_id(),
        truthful_auth.clone(),
    );
    let err = refusal_reason(&http, &net, &set.members[0].url, &stolen).await?;
    assert!(
        err.contains("reward postings do not match verified work"),
        "a coordinator redirected a reward to a node that did nothing: {err}"
    );

    // Lie two: the same shard, honestly done, billed at twice the price - and
    // this time the provider is in on it and signs the inflated figure, so the
    // signature verifies. The price is not a claim either party gets to make:
    // it falls out of the work spec, and the validators recompute it.
    //
    // The overcharge is deliberately small enough to fit inside the escrow the
    // job already holds. Asking for ten times the price would be refused too,
    // but for want of funds, and that would prove only that this job was
    // underfunded rather than that the price is not negotiable.
    let inflated = reward * 2;
    assert!(
        inflated < escrow_before,
        "the overcharge has to be affordable or the balance rule answers first"
    );
    let padded = lie(
        result.clone(),
        inflated,
        &assignment.assignment_id,
        &worker.identity.node_id(),
        sign_result(&assignment.assignment_id, &result, inflated)?,
    );
    let err = refusal_reason(&http, &net, &set.members[0].url, &padded).await?;
    assert!(
        err.contains("declared reward does not equal deterministic work cost"),
        "coordinator and provider together overcharged for a shard: {err}"
    );

    // Lie three: an assignment that was never scheduled. The id binds a job to
    // a shard index by construction, so a coordinator inventing schedule
    // history has to invent an id that cannot be derived.
    let made_up = "assignment-the-coordinator-made-up";
    let invented = lie(
        result.clone(),
        reward,
        made_up,
        &worker.identity.node_id(),
        sign_result(made_up, &result, reward)?,
    );
    let err = refusal_reason(&http, &net, &set.members[0].url, &invented).await?;
    assert!(
        err.contains("assignment id is not deterministic"),
        "a coordinator paid out an assignment it invented: {err}"
    );

    // Lie four: the answer is wrong, the provider signs it anyway, and the
    // coordinator reports a flattering audit. The recorded nonce is advisory;
    // the validators draw their own challenge from the chain position, which
    // neither of them could see when they chose the lie.
    let wrong = match &result {
        WorkResult::PrimeCount {
            count,
            bucket_counts,
            duration_ms,
        } => WorkResult::PrimeCount {
            count: count + 1,
            bucket_counts: bucket_counts.clone(),
            duration_ms: *duration_ms,
        },
        WorkResult::CollatzPeak {
            peak_steps,
            peak_seed,
            bucket_peaks,
            bucket_seeds,
            duration_ms,
        } => WorkResult::CollatzPeak {
            peak_steps: peak_steps + 1,
            peak_seed: *peak_seed,
            bucket_peaks: bucket_peaks.clone(),
            bucket_seeds: bucket_seeds.clone(),
            duration_ms: *duration_ms,
        },
        WorkResult::MatrixMultiply { rows, duration_ms } => {
            let mut rows = rows.clone();
            if let Some(first) = rows.first_mut() {
                *first = first.wrapping_add(1);
            }
            WorkResult::MatrixMultiply {
                rows,
                duration_ms: *duration_ms,
            }
        }
    };
    assert_ne!(wrong, result, "the test needs a genuinely different answer");
    let fabricated = lie(
        wrong.clone(),
        reward,
        &assignment.assignment_id,
        &worker.identity.node_id(),
        sign_result(&assignment.assignment_id, &wrong, reward)?,
    );
    let err = refusal_reason(&http, &net, &set.members[0].url, &fabricated).await?;
    assert!(
        err.contains("provider result does not verify"),
        "a signed wrong answer was paid for: {err}"
    );

    // None of it moved anything.
    assert_eq!(
        net.balance_quorum(&escrow_account(job)).await?.balance_mcu,
        escrow_before,
        "a refused settlement still touched the escrow"
    );
    assert_eq!(
        net.balance_quorum(&worker.identity.node_id())
            .await?
            .balance_mcu,
        0
    );

    // The truth settles, once. Told twice it is no longer true the second
    // time: the shard is the claim, and the claim is the ledger's, not the
    // coordinator's.
    let truthful = lie(
        result,
        reward,
        &assignment.assignment_id,
        &worker.identity.node_id(),
        truthful_auth,
    );
    net.transact(truthful.clone())
        .await
        .context("an honest reward was refused")?;
    let err = refusal_reason(&http, &net, &set.members[0].url, &truthful).await?;
    assert!(
        err.contains("claim already settled"),
        "a coordinator paid one shard twice: {err}"
    );
    assert_eq!(
        net.balance_quorum(&worker.identity.node_id())
            .await?
            .balance_mcu,
        reward,
        "the provider must hold exactly one shard's worth of CU"
    );
    assert_eq!(
        net.balance_quorum(&accomplice.identity.node_id())
            .await?
            .balance_mcu,
        0,
        "the node that did nothing must hold nothing"
    );

    drop(validators);
    Ok(())
}

/// Two coordinators over one job store, and one of them dies.
///
/// The thing under test is that nothing has to be elected, transferred, or
/// agreed for the survivor to pick up the dead peer's work. Ownership is a
/// pure function of the job id and the set of coordinators currently
/// answering, so the moment `b` stops answering `a`'s probes, `a`'s answer to
/// "who owns this job" changes on its own -- and the shards `a` was refusing
/// to hand out become shards it offers.
///
/// The split is checked against a live `b` first, from both sides, because a
/// failover test that never establishes the pre-failure state proves only that
/// one coordinator will serve everything.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_federated_coordinator_takes_over_the_jobs_of_a_peer_that_stops_answering() -> Result<()>
{
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));
    let node_bin = bin_dir.join(exe("hocmesh"));

    let tmp = TestDir::new()?;
    let http = test_client();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, _set) =
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

    // One database, two coordinators. This is what federation is for: the job
    // store is shared, so without an owner rule both would hand out the same
    // shard to different workers and pay for it twice.
    let db = tmp.path.join("federated.db");
    let port_a = free_port()?;
    let port_b = free_port()?;
    let url_a = format!("http://127.0.0.1:{port_a}");
    let url_b = format!("http://127.0.0.1:{port_b}");

    let config_a = federation_config(&tmp.path, "a", "eu", &url_a, &[("b", "us", &url_b)])?;
    let config_b = federation_config(&tmp.path, "b", "us", &url_b, &[("a", "eu", &url_a)])?;

    let coordinator_a = start_coordinator(
        &coordinator_bin,
        &db,
        &validators_path,
        Some(&config_a),
        port_a,
        &http,
    )
    .await?;
    let mut coordinator_b = start_coordinator(
        &coordinator_bin,
        &db,
        &validators_path,
        Some(&config_b),
        port_b,
        &http,
    )
    .await?;

    // Each has to actually see the other before any claim about the split
    // means anything. Peers start down on purpose -- a coordinator that has
    // not yet reached a peer must not assume it is there.
    wait_for_live(&http, &url_a, &["a", "b"]).await?;
    wait_for_live(&http, &url_b, &["a", "b"]).await?;

    // Find one job id each coordinator owns. Ownership is a hash, so this is a
    // lookup rather than a guess, and it is asked of both coordinators to show
    // they agree without having talked about it.
    let mut owned_by_a = None;
    let mut owned_by_b = None;
    for index in 0..64 {
        let job_id = format!("fed-job-{index}");
        let from_a = job_owner(&http, &url_a, &job_id).await?;
        let from_b = job_owner(&http, &url_b, &job_id).await?;
        assert_eq!(
            from_a, from_b,
            "both coordinators must name the same owner for {job_id} without an election"
        );
        match from_a.as_str() {
            "a" if owned_by_a.is_none() => owned_by_a = Some(job_id),
            "b" if owned_by_b.is_none() => owned_by_b = Some(job_id),
            _ => {}
        }
        if owned_by_a.is_some() && owned_by_b.is_some() {
            break;
        }
    }
    let job_a = owned_by_a.context("no job id hashed to coordinator a in 64 tries")?;
    let job_b = owned_by_b.context("no job id hashed to coordinator b in 64 tries")?;

    // Community-funded work, so a worker can be paid without a requester
    // having earned first. Both jobs are seeded into the one shared database.
    for job_id in [&job_a, &job_b] {
        let sponsors = sponsors_file(
            &tmp.path,
            &node_bin,
            &validator_homes,
            &validators_path,
            job_id,
            (2, 200000, 4),
        )?;
        run_ok(
            Command::new(&coordinator_bin)
                .arg("seed")
                .arg("--job-id")
                .arg(job_id)
                .arg("--sponsors")
                .arg(&sponsors)
                .arg("--db")
                .arg(&db)
                .arg("--validators")
                .arg(&validators_path)
                .arg("--start")
                .arg("2")
                .arg("--end")
                .arg("200000")
                .arg("--shards")
                .arg("4"),
            "seed federated community work",
        )?;
    }

    let worker = TestNode::new(&tmp.path.join("federation-worker"))?;
    register(&http, &url_a, &worker).await?;

    // While `b` is alive, `a` will not hand out `b`'s shards no matter how
    // many times it is asked -- even though both jobs are sitting in the
    // database it is reading.
    for _ in 0..4 {
        let offered = poll(&http, &url_a, &worker).await?;
        let assignment = offered
            .assignment
            .context("coordinator a should still be serving its own job")?;
        assert_eq!(
            assignment.job_id, job_a,
            "coordinator a must not hand out a job that hashes to b"
        );
    }
    // And the same holds in the other direction.
    let from_b = poll(&http, &url_b, &worker).await?;
    let assignment_b = from_b
        .assignment
        .context("coordinator b should be serving its own job")?;
    assert_eq!(
        assignment_b.job_id, job_b,
        "coordinator b must not hand out a job that hashes to a"
    );

    // The peer dies. Nothing tells `a` about it; `a` has to notice.
    coordinator_b.kill();
    wait_for_live(&http, &url_a, &["a"]).await?;
    let status = federation_status(&http, &url_a).await?;
    let peer_b = status
        .peers
        .iter()
        .find(|p| p.coordinator_id == "b")
        .context("coordinator a should still list b")?;
    assert!(!peer_b.up, "b should be marked down after the misses");
    assert!(
        peer_b.consecutive_misses >= 2,
        "b should be down because probes failed, not for some other reason"
    );

    // Ownership moved on its own: the same job id now answers `a`.
    assert_eq!(
        job_owner(&http, &url_a, &job_b).await?,
        "a",
        "the surviving coordinator should own the dead peer's jobs"
    );

    // And the takeover is real work, not a status flag: `a` now offers the
    // job it was refusing, and the shard settles and pays.
    let before = balance(&http, &url_a, &worker).await?.balance_mcu;
    let taken_over = poll_until_assignment(&http, &url_a, &worker, Some(&job_b))
        .await
        .context("coordinator a should offer the dead peer's shards")?;
    assert_eq!(taken_over.job_id, job_b);
    complete_assignment(&http, &url_a, &worker, &taken_over).await?;
    let after = balance(&http, &url_a, &worker).await?.balance_mcu;
    assert!(
        after > before,
        "the taken-over shard should have been settled and paid ({before} -> {after})"
    );

    drop(coordinator_a);
    for mut validator in validators {
        validator.kill();
    }
    Ok(())
}

/// The topology view describes the machines the scheduler is choosing from.
///
/// It is read-only and moves no CU, so the only thing worth asserting is that
/// it tells the truth: every registered machine appears, a machine that never
/// placed itself is reported as unplaced rather than as nearby, and a cluster
/// request that cannot be satisfied says so instead of returning a short set.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_topology_view_reports_who_is_available_and_how_far_apart_they_are() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let validator_bin = bin_dir.join(exe("hocmesh-validator"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));

    let tmp = TestDir::new()?;
    let http = test_client();
    let validator_ports = [free_port()?, free_port()?, free_port()?, free_port()?];
    let (validators_path, validator_homes, validator_dbs, _set) =
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

    let db = tmp.path.join("topology.db");
    let port = free_port()?;
    let url = format!("http://127.0.0.1:{port}");
    let coordinator =
        start_coordinator(&coordinator_bin, &db, &validators_path, None, port, &http).await?;

    let first = TestNode::new(&tmp.path.join("topology-a"))?;
    let second = TestNode::new(&tmp.path.join("topology-b"))?;
    register(&http, &url, &first).await?;
    register(&http, &url, &second).await?;

    let view: serde_json::Value = get_json(&http, &url, "/v1/topology").await?;
    assert_eq!(view["online"].as_u64(), Some(2));
    let nodes = view["nodes"].as_array().context("nodes array")?;
    assert_eq!(nodes.len(), 2);
    for node in nodes {
        assert_eq!(
            node["located"].as_bool(),
            Some(false),
            "a node that has not placed itself must not be reported as located"
        );
        assert!(node["shared_memory_bytes"].as_u64().unwrap_or(0) > 0);
    }

    let pair: serde_json::Value = get_json(&http, &url, "/v1/topology?cluster=2").await?;
    let cluster = pair["cluster"].as_object().context("a pair should form")?;
    assert_eq!(
        cluster["node_ids"].as_array().map(Vec::len),
        Some(2),
        "a cluster of two should hold exactly two machines"
    );
    assert_eq!(
        cluster["worst_edge_micros"].as_u64(),
        pair["unknown_edge_micros"].as_u64(),
        "two unplaced machines are at the placeholder distance, not at zero"
    );

    let impossible: serde_json::Value = get_json(&http, &url, "/v1/topology?cluster=5").await?;
    assert!(impossible["cluster"].is_null());
    assert!(
        impossible["cluster_unavailable"].is_string(),
        "an unsatisfiable request should say why rather than return a short set"
    );

    drop(coordinator);
    for mut validator in validators {
        validator.kill();
    }
    Ok(())
}

/// Write a `--federation` file.
fn federation_config(
    dir: &Path,
    coordinator_id: &str,
    region: &str,
    advertise: &str,
    peers: &[(&str, &str, &str)],
) -> Result<PathBuf> {
    let path = dir.join(format!("federation-{coordinator_id}.json"));
    let peers: Vec<serde_json::Value> = peers
        .iter()
        .map(|(id, region, url)| json!({ "coordinator_id": id, "region": region, "url": url }))
        .collect();
    fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "coordinator_id": coordinator_id,
            "region": region,
            "advertise": advertise,
            "peers": peers,
            // Short enough that the test does not sit through a production
            // probe cycle, long enough that a loaded machine is not called
            // dead for being slow.
            "probe_interval_secs": 1,
            "misses_before_down": 2,
        }))?,
    )?;
    Ok(path)
}

async fn start_coordinator(
    coordinator_bin: &Path,
    db: &Path,
    validators_path: &Path,
    federation: Option<&Path>,
    port: u16,
    http: &Client,
) -> Result<ProcessGuard> {
    let mut command = Command::new(coordinator_bin);
    command
        .arg("serve")
        .arg("--db")
        .arg(db)
        .arg("--listen")
        .arg(format!("127.0.0.1:{port}"))
        .arg("--validators")
        .arg(validators_path);
    if let Some(federation) = federation {
        command.arg("--federation").arg(federation);
    }
    let guard = ProcessGuard::spawn(&mut command)?;
    wait_health(http, port).await?;
    Ok(guard)
}

#[derive(serde::Deserialize)]
struct TestPeerHealth {
    coordinator_id: String,
    up: bool,
    consecutive_misses: u32,
}

#[derive(serde::Deserialize)]
struct TestFederationStatus {
    live: Vec<String>,
    peers: Vec<TestPeerHealth>,
}

async fn federation_status(http: &Client, url: &str) -> Result<TestFederationStatus> {
    let report: serde_json::Value = get_json(http, url, "/v1/federation/status").await?;
    let status = report
        .get("status")
        .cloned()
        .filter(|value| !value.is_null())
        .context("coordinator reported no federation")?;
    Ok(serde_json::from_value(status)?)
}

/// Wait until a coordinator's live set is exactly `expected`.
///
/// Bounded rather than unbounded: the probe interval is a second, so a set
/// that has not converged in twenty is a failure and not slowness.
async fn wait_for_live(http: &Client, url: &str, expected: &[&str]) -> Result<()> {
    let expected: Vec<String> = expected.iter().map(|id| (*id).to_string()).collect();
    let mut last = Vec::new();
    for _ in 0..200 {
        if let Ok(status) = federation_status(http, url).await {
            if status.live == expected {
                return Ok(());
            }
            last = status.live;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(anyhow!(
        "{url} never settled on live set {expected:?}, last saw {last:?}"
    ))
}

async fn job_owner(http: &Client, url: &str, job_id: &str) -> Result<String> {
    let report: serde_json::Value =
        get_json(http, url, &format!("/v1/federation/jobs/{job_id}")).await?;
    report
        .get("owner")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .context("federation job report carried no owner")
}
