use crate::client::MeshClient;
use anyhow::Result;
use mesh_core::compute::execute_work;
use mesh_protocol::NodeCapabilities;
use std::{sync::Arc, time::Duration};
use tokio::task::JoinSet;

pub async fn run(
    client: MeshClient,
    capabilities: NodeCapabilities,
    workers: usize,
    poll_ms: u64,
) -> Result<()> {
    let registered = client.register(&capabilities).await?;
    println!("MESH node {} registered", registered.node_id);
    println!(
        "Available balance: {:.3} CU",
        registered.balance_mcu as f64 / 1000.0
    );
    println!("Contribution workers: {}", workers);

    let capabilities = Arc::new(capabilities);
    let heartbeat_client = client.clone();
    let heartbeat_caps = capabilities.clone();
    let heartbeat = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            if let Err(error) = heartbeat_client.heartbeat(&heartbeat_caps).await {
                tracing::warn!(%error, "heartbeat failed");
            }
        }
    });

    let mut workers_set = JoinSet::new();
    for worker_id in 0..workers.max(1) {
        let worker_client = client.clone();
        workers_set
            .spawn(async move { worker_loop(worker_client, worker_id, poll_ms.max(100)).await });
    }

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("Shutdown requested; stopping MESH node.");
        }
        result = workers_set.join_next() => {
            match result {
                Some(Ok(Ok(()))) => tracing::warn!("worker exited unexpectedly"),
                Some(Ok(Err(error))) => tracing::error!(%error, "worker failed"),
                Some(Err(error)) => tracing::error!(%error, "worker task panicked"),
                None => tracing::warn!("all workers exited"),
            }
        }
    }

    heartbeat.abort();
    workers_set.abort_all();
    Ok(())
}

async fn worker_loop(client: MeshClient, worker_id: usize, poll_ms: u64) -> Result<()> {
    let idle_delay = Duration::from_millis(poll_ms);
    loop {
        let poll = match client.poll().await {
            Ok(poll) => poll,
            Err(error) => {
                tracing::warn!(worker_id = worker_id, error = %error, "poll failed");
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };

        let Some(assignment) = poll.assignment else {
            tokio::time::sleep(idle_delay).await;
            continue;
        };

        tracing::info!(
            worker_id = worker_id,
            assignment_id = %assignment.assignment_id,
            job_id = %assignment.job_id,
            shard = assignment.shard_index,
            reward_mcu = assignment.reward_mcu,
            "executing contribution assignment"
        );

        let work = assignment.work.clone();
        let result = tokio::task::spawn_blocking(move || execute_work(&work)).await?;

        match client.report_result(&assignment, &result).await {
            Ok(settlement) => {
                tracing::info!(
                    worker_id = worker_id,
                    assignment_id = %assignment.assignment_id,
                    earned_cu = settlement.reward_mcu as f64 / 1000.0,
                    balance_cu = settlement.balance_mcu as f64 / 1000.0,
                    "verified contribution settled"
                );
            }
            Err(error) => {
                tracing::warn!(worker_id = worker_id, assignment_id = %assignment.assignment_id, error = %error, "result submission failed");
                // The coordinator lease will expire and requeue the work if the result was not accepted.
            }
        }
    }
}
