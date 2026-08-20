use anyhow::{Context, Result, bail};
use mesh_core::identity::NodeIdentity;
use mesh_protocol::{
    BalanceResponse, ErrorResponse, HeartbeatRequest, JobStatusResponse, NetworkStatsResponse,
    NodeCapabilities, NodeStatusResponse, PollRequest, PollResponse, RegisterRequest,
    RegisterResponse, ResultRequest, ResultResponse, SubmitJobRequest, SubmitJobResponse,
    WorkAssignment, WorkResult, WorkSpec, empty_body_hash, heartbeat_body_hash, register_body_hash,
    result_body_hash, submit_body_hash,
};
use reqwest::{Client, Response};
use serde::de::DeserializeOwned;

#[derive(Clone)]
pub struct MeshClient {
    http: Client,
    coordinator: String,
    identity: NodeIdentity,
}

impl MeshClient {
    pub fn new(coordinator: impl Into<String>, identity: NodeIdentity) -> Self {
        Self {
            http: Client::new(),
            coordinator: coordinator.into().trim_end_matches('/').to_string(),
            identity,
        }
    }

    pub fn node_id(&self) -> String {
        self.identity.node_id()
    }

    pub async fn register(&self, capabilities: &NodeCapabilities) -> Result<RegisterResponse> {
        let public_key_b64 = self.identity.public_key_b64();
        let body_hash = register_body_hash(&public_key_b64, capabilities)?;
        let req = RegisterRequest {
            auth: self.identity.auth("register", &body_hash),
            public_key_b64,
            capabilities: capabilities.clone(),
        };
        self.post("/v1/nodes/register", &req).await
    }

    pub async fn heartbeat(&self, capabilities: &NodeCapabilities) -> Result<()> {
        let body_hash = heartbeat_body_hash(capabilities)?;
        let req = HeartbeatRequest {
            auth: self.identity.auth("heartbeat", &body_hash),
            capabilities: capabilities.clone(),
        };
        let _: serde_json::Value = self.post("/v1/nodes/heartbeat", &req).await?;
        Ok(())
    }

    pub async fn poll(&self) -> Result<PollResponse> {
        let req = PollRequest {
            auth: self.identity.auth("poll", &empty_body_hash()),
        };
        self.post("/v1/work/poll", &req).await
    }

    pub async fn report_result(
        &self,
        assignment: &WorkAssignment,
        result: &WorkResult,
    ) -> Result<ResultResponse> {
        let body_hash = result_body_hash(
            &assignment.assignment_id,
            &assignment.job_id,
            assignment.shard_index,
            &assignment.work,
            assignment.reward_mcu,
            assignment.system_funded,
            result,
        )?;
        let req = ResultRequest {
            auth: self.identity.auth("result", &body_hash),
            assignment_id: assignment.assignment_id.clone(),
            job_id: assignment.job_id.clone(),
            shard_index: assignment.shard_index,
            work: assignment.work.clone(),
            reward_mcu: assignment.reward_mcu,
            system_funded: assignment.system_funded,
            result: result.clone(),
        };
        self.post("/v1/work/result", &req).await
    }

    pub async fn submit(&self, work: WorkSpec, shards: u32) -> Result<SubmitJobResponse> {
        let body_hash = submit_body_hash(&work, shards)?;
        let req = SubmitJobRequest {
            auth: self.identity.auth("submit", &body_hash),
            work,
            shards,
        };
        self.post("/v1/jobs/submit", &req).await
    }

    pub async fn balance(&self) -> Result<BalanceResponse> {
        self.get(&format!("/v1/nodes/{}/balance", self.node_id()))
            .await
    }

    pub async fn node_status(&self) -> Result<NodeStatusResponse> {
        self.get(&format!("/v1/nodes/{}", self.node_id())).await
    }

    pub async fn job_status(&self, job_id: &str) -> Result<JobStatusResponse> {
        self.get(&format!("/v1/jobs/{job_id}")).await
    }

    pub async fn network_stats(&self) -> Result<NetworkStatsResponse> {
        self.get("/v1/network/stats").await
    }

    async fn post<TReq: serde::Serialize + ?Sized, TResp: DeserializeOwned>(
        &self,
        path: &str,
        request: &TReq,
    ) -> Result<TResp> {
        let response = self
            .http
            .post(format!("{}{}", self.coordinator, path))
            .json(request)
            .send()
            .await
            .with_context(|| format!("calling coordinator {}", self.coordinator))?;
        decode(response).await
    }

    async fn get<TResp: DeserializeOwned>(&self, path: &str) -> Result<TResp> {
        let response = self
            .http
            .get(format!("{}{}", self.coordinator, path))
            .send()
            .await
            .with_context(|| format!("calling coordinator {}", self.coordinator))?;
        decode(response).await
    }
}

async fn decode<T: DeserializeOwned>(response: Response) -> Result<T> {
    let status = response.status();
    let text = response
        .text()
        .await
        .context("reading coordinator response")?;
    if !status.is_success() {
        if let Ok(error) = serde_json::from_str::<ErrorResponse>(&text) {
            bail!("coordinator returned {}: {}", status, error.error);
        }
        bail!("coordinator returned {}: {}", status, text);
    }
    serde_json::from_str(&text).with_context(|| format!("decoding coordinator response: {text}"))
}
