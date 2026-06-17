use partial_stateless::PartialExecutionWitnessSidecar;
use std::{env, net::SocketAddr};
use tokio::sync::{broadcast, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{debug, info, warn};

pub(crate) mod proto {
    tonic::include_proto!("partialstateless");
}

use proto::{
    partial_execution_witness_sidecars_server::{
        PartialExecutionWitnessSidecars, PartialExecutionWitnessSidecarsServer,
    },
    SidecarFrame, SubscribeSidecarsRequest,
};

const GRPC_ADDR_ENV: &str = "CACHE_VOPS_GRPC_ADDR";
const DEFAULT_GRPC_ADDR: &str = "127.0.0.1:10000";

#[derive(Clone, Debug)]
pub(crate) struct SidecarPublisher {
    tx: broadcast::Sender<SidecarFrame>,
}

#[derive(Clone, Debug)]
struct SidecarService {
    tx: broadcast::Sender<SidecarFrame>,
}

impl SidecarPublisher {
    pub(crate) fn new() -> Self {
        let (tx, _) = broadcast::channel(32);
        Self { tx }
    }

    pub(crate) fn sender(&self) -> broadcast::Sender<SidecarFrame> {
        self.tx.clone()
    }

    pub(crate) fn publish(&self, sidecar: &PartialExecutionWitnessSidecar) -> eyre::Result<()> {
        let json = sidecar.to_json_value();
        let sidecar_json = serde_json::to_vec(&json)?;
        let counts = sidecar.partial_execution_witness.missing_targets.counts();
        let target_count =
            counts.accounts + counts.storage_slots + counts.code_hashes + counts.headers;
        let payload_bytes =
            sidecar.partial_execution_witness.payload_total_bytes().unwrap_or_default();

        let frame = SidecarFrame {
            event: sidecar.envelope.event.to_string(),
            block_number: sidecar.envelope.block_number,
            block_hash: sidecar.envelope.block_hash.clone(),
            parent_hash: sidecar.envelope.parent_hash.clone(),
            target_source: sidecar
                .partial_execution_witness
                .missing_targets
                .source
                .as_str()
                .to_string(),
            payload_kind: sidecar.partial_execution_witness.payload.kind().to_string(),
            target_count: target_count as u64,
            payload_bytes: payload_bytes as u64,
            sidecar_json,
        };

        match self.tx.send(frame) {
            Ok(receiver_count) => {
                debug!(receiver_count, "published partial execution witness sidecar frame");
            }
            Err(err) => {
                debug!(error = %err, "dropped sidecar frame because no gRPC subscribers are active");
            }
        }

        Ok(())
    }
}

impl SidecarService {
    const fn new(tx: broadcast::Sender<SidecarFrame>) -> Self {
        Self { tx }
    }
}

#[tonic::async_trait]
impl PartialExecutionWitnessSidecars for SidecarService {
    type SubscribeStream = ReceiverStream<Result<SidecarFrame, Status>>;

    async fn subscribe(
        &self,
        _request: Request<SubscribeSidecarsRequest>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let (out_tx, out_rx) = mpsc::channel(8);
        let mut rx = self.tx.subscribe();

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(frame) => {
                        if out_tx.send(Ok(frame)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!(skipped, "partial-stateless sidecar gRPC subscriber lagged");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(out_rx)))
    }
}

pub(crate) fn grpc_addr_from_env() -> eyre::Result<SocketAddr> {
    let raw = env::var(GRPC_ADDR_ENV).unwrap_or_else(|_| DEFAULT_GRPC_ADDR.to_string());
    raw.parse().map_err(|err| eyre::eyre!("invalid {GRPC_ADDR_ENV}={raw}: {err}"))
}

pub(crate) async fn serve_sidecars(
    addr: SocketAddr,
    tx: broadcast::Sender<SidecarFrame>,
) -> eyre::Result<()> {
    info!(%addr, "starting partial-stateless sidecar gRPC stream");
    Server::builder()
        .add_service(
            PartialExecutionWitnessSidecarsServer::new(SidecarService::new(tx))
                .max_encoding_message_size(usize::MAX)
                .max_decoding_message_size(usize::MAX),
        )
        .serve(addr)
        .await?;
    Ok(())
}
