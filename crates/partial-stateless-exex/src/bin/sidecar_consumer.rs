use std::{env, time::Duration};

mod proto {
    tonic::include_proto!("partialstateless");
}

use proto::{
    partial_execution_witness_sidecars_client::PartialExecutionWitnessSidecarsClient,
    SubscribeSidecarsRequest,
};

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:10000";

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let endpoint = env::args()
        .nth(1)
        .or_else(|| env::var("CACHE_VOPS_GRPC_ENDPOINT").ok())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    let max_frames = env::var("CACHE_VOPS_CONSUMER_MAX")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(3);

    let mut client = PartialExecutionWitnessSidecarsClient::connect(endpoint.clone())
        .await?
        .max_encoding_message_size(usize::MAX)
        .max_decoding_message_size(usize::MAX);
    let mut stream = client.subscribe(SubscribeSidecarsRequest {}).await?.into_inner();

    eprintln!(
        "connected to partial-stateless sidecar stream at {endpoint}; waiting for {max_frames} frame(s)"
    );

    let mut received = 0usize;
    while let Some(frame) =
        tokio::time::timeout(Duration::from_secs(180), stream.message()).await??
    {
        received += 1;
        println!(
            "sidecar frame #{received}: block={} hash={} event={} source={} payload={} targets={} payload_bytes={} json_bytes={}",
            frame.block_number,
            frame.block_hash,
            frame.event,
            frame.target_source,
            frame.payload_kind,
            frame.target_count,
            frame.payload_bytes,
            frame.sidecar_json.len(),
        );

        if received >= max_frames {
            break;
        }
    }

    Ok(())
}
