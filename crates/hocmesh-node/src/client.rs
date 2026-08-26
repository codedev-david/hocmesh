use anyhow::{Context, Result, bail};
use hocmesh_ai::{
    FailInferenceRequest, FailInferenceResponse, InferenceJobStatus, PlanRequest, PlanResponse,
    PollInferenceRequest, PollInferenceResponse, RefundInferenceRequest, RefundInferenceResponse,
    RegisterModelRequest, RegisterModelResponse, ReportInferenceRequest, ReportInferenceResponse,
    SubmitInferenceRequest, SubmitInferenceResponse, fail_inference_body_hash, plan_body_hash,
    register_model_body_hash, report_inference_body_hash, submit_inference_body_hash,
};
use hocmesh_core::identity::NodeIdentity;
use hocmesh_model::ModelManifest;
use hocmesh_protocol::{
    BalanceResponse, ErrorResponse, HeartbeatRequest, JobStatusResponse, NetworkStatsResponse,
    NodeCapabilities, NodeStatusResponse, PeerSampleResponse, PollRequest, PollResponse,
    RefundRequest, RefundResponse, RegisterRequest, RegisterResponse, ResultRequest,
    ResultResponse, SubmitJobRequest, SubmitJobResponse, WorkAssignment, WorkResult, WorkSpec,
    empty_body_hash, heartbeat_body_hash, refund_body_hash, register_body_hash, result_body_hash,
    submit_body_hash,
};
use reqwest::{Client, Response};
use serde::de::DeserializeOwned;

#[derive(Clone)]
pub struct HocMeshClient {
    http: Client,
    coordinator: String,
    identity: NodeIdentity,
}

impl HocMeshClient {
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

    /// Takes back the escrow on every shard of a job that the mesh let lapse.
    /// The coordinator says which shards those are and what they were for;
    /// this signs each one, and the ledger checks that story against the
    /// reservation it certified before any CU moves.
    pub async fn reclaim(&self, job_id: &str) -> Result<Vec<RefundResponse>> {
        let job = self.job_status(job_id).await?;
        let mut reclaimed = Vec::new();
        for shard in job.refundable {
            let body_hash = refund_body_hash(
                &shard.assignment_id,
                job_id,
                shard.shard_index,
                &shard.work,
                shard.refund_mcu,
                job.system_funded,
            )?;
            // Community work was never anyone-in-particular's to pay for, so
            // there is nobody to sign for its return either.
            let auth = (!job.system_funded).then(|| self.identity.auth("refund", &body_hash));
            let req = RefundRequest {
                assignment_id: shard.assignment_id,
                auth,
            };
            reclaimed.push(self.post("/v1/work/refund", &req).await?);
        }
        Ok(reclaimed)
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

    pub async fn register_model(&self, manifest: &ModelManifest) -> Result<RegisterModelResponse> {
        let body_hash = register_model_body_hash(manifest)?;
        let request = RegisterModelRequest {
            auth: self.identity.auth("register_model", &body_hash),
            manifest: manifest.clone(),
        };
        self.post("/v1/ai/models/register", &request).await
    }

    pub async fn plan_ai(&self, mut request: PlanRequest) -> Result<PlanResponse> {
        let body_hash = plan_body_hash(&request)?;
        request.auth = self.identity.auth("plan_ai", &body_hash);
        self.post("/v1/ai/plan", &request).await
    }

    pub async fn submit_inference(
        &self,
        mut request: SubmitInferenceRequest,
    ) -> Result<SubmitInferenceResponse> {
        let body_hash = submit_inference_body_hash(&request)?;
        request.auth = self.identity.auth("submit_inference", &body_hash);
        self.post("/v1/ai/jobs/submit", &request).await
    }

    pub async fn poll_inference(&self) -> Result<PollInferenceResponse> {
        let request = PollInferenceRequest {
            auth: self.identity.auth("poll_inference", &empty_body_hash()),
        };
        self.post("/v1/ai/work/poll", &request).await
    }

    /// Fetch a published manifest so a requester can price its own job.
    ///
    /// The requester needs the parameter count and the digest to write a bill,
    /// and it has to get them from the published manifest rather than from
    /// whoever is about to be paid.
    pub async fn get_model(&self, model_id: &str, revision: &str) -> Result<ModelManifest> {
        self.get(&format!("/v1/ai/models/{model_id}/{revision}"))
            .await
    }
    pub async fn report_inference(
        &self,
        assignment_id: String,
        job_id: String,
        batch_start: u32,
        batch_end: u32,
        reward_mcu: i64,
        outputs: Vec<hocmesh_ai::PromptOutput>,
    ) -> Result<ReportInferenceResponse> {
        let mut request = ReportInferenceRequest {
            auth: self.identity.auth("unused", &empty_body_hash()),
            assignment_id,
            job_id,
            batch_start,
            batch_end,
            reward_mcu,
            outputs,
        };
        let body_hash = report_inference_body_hash(&request)?;
        request.auth = self.identity.auth("report_inference", &body_hash);
        self.post("/v1/ai/work/result", &request).await
    }

    pub async fn fail_inference(
        &self,
        assignment_id: String,
        reason: String,
    ) -> Result<FailInferenceResponse> {
        let mut request = FailInferenceRequest {
            auth: self.identity.auth("unused", &empty_body_hash()),
            assignment_id,
            reason,
        };
        let body_hash = fail_inference_body_hash(&request)?;
        request.auth = self.identity.auth("fail_inference", &body_hash);
        self.post("/v1/ai/work/fail", &request).await
    }

    /// Ask for probe targets to measure ourselves against.
    ///
    /// Unauthenticated on purpose: the reply is a directory of endpoints and
    /// positions that every node already publishes, and requiring a signature
    /// would imply the coordinator vouches for the answer. It does not - it
    /// never times a round trip.
    pub async fn peers(&self) -> Result<PeerSampleResponse> {
        self.get("/v1/network/peers").await
    }

    pub async fn inference_status(&self, job_id: &str) -> Result<InferenceJobStatus> {
        self.get(&format!("/v1/ai/jobs/{job_id}")).await
    }

    /// Takes back the escrow on every batch of an inference job the mesh let
    /// lapse.
    ///
    /// The coordinator says which batches lapsed and what they were priced at,
    /// but it is not believed: the requester signs each claim itself, and the
    /// ledger re-derives the amount from the billing it certified at reserve
    /// time before any CU moves.
    pub async fn reclaim_inference(&self, job_id: &str) -> Result<Vec<RefundInferenceResponse>> {
        let job = self.inference_status(job_id).await?;
        let mut reclaimed = Vec::new();
        for batch in job.refundable {
            let body_hash = hocmesh_protocol::inference_refund_body_hash(
                &batch.assignment_id,
                job_id,
                batch.batch_start,
                batch.batch_end,
                batch.refund_mcu,
            )?;
            let req = RefundInferenceRequest {
                auth: self.identity.auth("refund_inference", &body_hash),
                job_id: job_id.to_string(),
                assignment_id: batch.assignment_id,
                batch_start: batch.batch_start,
                batch_end: batch.batch_end,
                refund_mcu: batch.refund_mcu,
            };
            reclaimed.push(self.post("/v1/ai/jobs/refund", &req).await?);
        }
        Ok(reclaimed)
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

#[cfg(test)]
mod tests {
    use super::*;
    use hocmesh_core::identity::NodeIdentity;
    use hocmesh_protocol::PeerSample;

    /// A client pointed at `router`, with a throwaway identity.
    async fn client_for(router: axum::Router, tag: &str) -> (HocMeshClient, std::path::PathBuf) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let home =
            std::env::temp_dir().join(format!("hocmesh-client-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let identity = NodeIdentity::load_or_create(&home).unwrap();
        (
            HocMeshClient::new(format!("http://{address}"), identity),
            home,
        )
    }

    /// The peer sample is the whole bootstrap directory a node gets, so it has
    /// to arrive intact - an endpoint it cannot dial is a peer it cannot use.
    #[tokio::test]
    async fn a_peer_sample_arrives_with_the_endpoints_it_was_sent_with() {
        let router = axum::Router::new().route(
            "/v1/network/peers",
            axum::routing::get(|| async {
                axum::Json(PeerSampleResponse {
                    peers: vec![PeerSample {
                        node_id: "peer-a".to_string(),
                        probe_endpoint: "http://10.0.0.7:8646".to_string(),
                        coordinate: None,
                    }],
                })
            }),
        );
        let (client, home) = client_for(router, "peers").await;

        let sample = client.peers().await.unwrap();
        assert_eq!(sample.peers.len(), 1);
        assert_eq!(sample.peers[0].node_id, "peer-a");
        assert_eq!(sample.peers[0].probe_endpoint, "http://10.0.0.7:8646");
        std::fs::remove_dir_all(&home).unwrap();
    }

    /// When the coordinator explains itself, the operator should read the
    /// coordinator's words, not a generic transport failure.
    #[tokio::test]
    async fn a_coordinator_error_reaches_the_operator_as_written() {
        let router = axum::Router::new().route(
            "/v1/network/peers",
            axum::routing::get(|| async {
                (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    axum::Json(ErrorResponse {
                        error: "directory unavailable".to_string(),
                    }),
                )
            }),
        );
        let (client, home) = client_for(router, "error").await;

        let error = client.peers().await.unwrap_err().to_string();
        assert!(error.contains("503"), "{error}");
        assert!(error.contains("directory unavailable"), "{error}");
        std::fs::remove_dir_all(&home).unwrap();
    }

    /// A reply that cannot be read has to say so and show what arrived - a
    /// proxy's error page is the usual culprit and looks nothing like a bug
    /// in the coordinator.
    #[tokio::test]
    async fn an_unreadable_reply_shows_what_actually_arrived() {
        let router = axum::Router::new().route(
            "/v1/network/peers",
            axum::routing::get(|| async { "<html>gateway timeout</html>" }),
        );
        let (client, home) = client_for(router, "garbage").await;

        let error = client.peers().await.unwrap_err().to_string();
        assert!(error.contains("gateway timeout"), "{error}");
        std::fs::remove_dir_all(&home).unwrap();
    }
}
