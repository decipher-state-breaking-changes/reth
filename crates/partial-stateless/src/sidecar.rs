use crate::{CacheDescriptor, TargetStats, WitnessTargets};
use serde_json::json;

/// Block envelope for a partial execution witness sidecar.
#[derive(Debug, Clone)]
pub struct SidecarBlockContext {
    /// Notification event that produced this sidecar.
    pub event: &'static str,
    /// Block number.
    pub block_number: u64,
    /// Block hash.
    pub block_hash: String,
    /// Parent block hash.
    pub parent_hash: String,
    /// Parent state root.
    pub parent_state_root: Option<String>,
    /// Post-execution state root.
    pub post_state_root: String,
    /// Transaction count in the block.
    pub tx_count: usize,
}

/// RLP-encoded trie node carried by a sidecar payload.
#[derive(Debug, Clone)]
pub struct EncodedProofNode {
    /// Hex encoded trie path in nibble space.
    pub path: String,
    /// Hex encoded RLP node bytes.
    pub rlp: String,
    /// Raw RLP node byte length.
    pub byte_len: usize,
}

/// Storage multiproof for one account.
#[derive(Debug, Clone)]
pub struct EncodedStorageMultiproof {
    /// Hashed account address.
    pub hashed_address: String,
    /// Storage trie root from the account leaf.
    pub storage_root: String,
    /// RLP-encoded storage trie proof nodes.
    pub nodes: Vec<EncodedProofNode>,
}

/// Bytecode preimage carried outside the trie proof.
#[derive(Debug, Clone)]
pub struct EncodedCodePreimage {
    /// Code hash referenced by an account leaf.
    pub code_hash: String,
    /// Hex encoded bytecode.
    pub bytecode: String,
    /// Raw bytecode length.
    pub byte_len: usize,
}

/// Stats for a state multiproof payload.
#[derive(Debug, Clone, Copy, Default)]
pub struct StateMultiproofPayloadStats {
    /// Account trie proof node count.
    pub account_node_count: usize,
    /// Account trie proof node bytes.
    pub account_node_bytes: usize,
    /// Number of storage multiproofs.
    pub storage_multiproof_count: usize,
    /// Storage trie proof node count.
    pub storage_node_count: usize,
    /// Storage trie proof node bytes.
    pub storage_node_bytes: usize,
    /// Included bytecode preimage count.
    pub code_preimage_count: usize,
    /// Included bytecode preimage bytes.
    pub code_preimage_bytes: usize,
    /// Requested code hashes that were not available from the parent state provider.
    pub unavailable_code_preimage_count: usize,
    /// Multiproof generation latency in milliseconds.
    pub generation_latency_ms: u128,
}

/// State multiproof payload encoded in JSON-safe hex strings.
#[derive(Debug, Clone)]
pub struct StateMultiproofPayload {
    /// Account trie proof nodes.
    pub account_nodes: Vec<EncodedProofNode>,
    /// Storage trie multiproofs grouped by hashed account.
    pub storage_multiproofs: Vec<EncodedStorageMultiproof>,
    /// Bytecode preimages included with the sidecar.
    pub code_preimages: Vec<EncodedCodePreimage>,
    /// Code hashes that were requested but unavailable.
    pub unavailable_code_hashes: Vec<String>,
    /// Payload statistics.
    pub stats: StateMultiproofPayloadStats,
}

/// Witness payload for the current producer.
#[derive(Debug, Clone)]
pub enum WitnessPayload {
    /// No proof bytes are included yet.
    NoneSkeletonOnly,
    /// State multiproof payload for cache-missing account/storage targets.
    StateMultiproofV1(StateMultiproofPayload),
}

impl WitnessPayload {
    /// Stable payload kind.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::NoneSkeletonOnly => "none_skeleton_only",
            Self::StateMultiproofV1(_) => "state_multiproof_v1",
        }
    }

    /// Convert proof payload metadata to JSON.
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::NoneSkeletonOnly => json!({
                "kind": self.kind(),
            }),
            Self::StateMultiproofV1(payload) => json!({
                "kind": self.kind(),
                "account_nodes": payload.account_nodes.iter().map(EncodedProofNode::to_json_value).collect::<Vec<_>>(),
                "storage_multiproofs": payload.storage_multiproofs.iter().map(EncodedStorageMultiproof::to_json_value).collect::<Vec<_>>(),
                "code_preimages": payload.code_preimages.iter().map(EncodedCodePreimage::to_json_value).collect::<Vec<_>>(),
                "unavailable_code_hashes": payload.unavailable_code_hashes,
                "stats": payload.stats.to_json_value(),
            }),
        }
    }
}

impl EncodedProofNode {
    /// Convert encoded proof node to JSON.
    pub fn to_json_value(&self) -> serde_json::Value {
        json!({
            "path": self.path,
            "rlp": self.rlp,
            "byte_len": self.byte_len,
        })
    }
}

impl EncodedStorageMultiproof {
    /// Convert encoded storage multiproof to JSON.
    pub fn to_json_value(&self) -> serde_json::Value {
        json!({
            "hashed_address": self.hashed_address,
            "storage_root": self.storage_root,
            "nodes": self.nodes.iter().map(EncodedProofNode::to_json_value).collect::<Vec<_>>(),
        })
    }
}

impl EncodedCodePreimage {
    /// Convert encoded bytecode preimage to JSON.
    pub fn to_json_value(&self) -> serde_json::Value {
        json!({
            "code_hash": self.code_hash,
            "bytecode": self.bytecode,
            "byte_len": self.byte_len,
        })
    }
}

impl StateMultiproofPayloadStats {
    /// Total raw payload bytes represented by this payload.
    pub const fn total_bytes(self) -> usize {
        self.account_node_bytes + self.storage_node_bytes + self.code_preimage_bytes
    }

    /// Convert stats to JSON.
    pub fn to_json_value(self) -> serde_json::Value {
        json!({
            "account_node_count": self.account_node_count,
            "account_node_bytes": self.account_node_bytes,
            "storage_multiproof_count": self.storage_multiproof_count,
            "storage_node_count": self.storage_node_count,
            "storage_node_bytes": self.storage_node_bytes,
            "code_preimage_count": self.code_preimage_count,
            "code_preimage_bytes": self.code_preimage_bytes,
            "unavailable_code_preimage_count": self.unavailable_code_preimage_count,
            "generation_latency_ms": self.generation_latency_ms,
            "total_bytes": self.total_bytes(),
        })
    }
}

/// Cache-filtered execution witness payload.
#[derive(Debug, Clone)]
pub struct PartialExecutionWitness {
    /// Targets missing from the sidecar's cache descriptor.
    pub missing_targets: WitnessTargets,
    /// Target count stats.
    pub stats: TargetStats,
    /// Witness payload.
    pub payload: WitnessPayload,
}

impl PartialExecutionWitness {
    /// Convert partial execution witness to the current JSON artifact representation.
    pub fn to_json_value(&self) -> serde_json::Value {
        json!({
            "target_source": {
                "kind": self.missing_targets.source.as_str(),
                "completeness": self.missing_targets.source.completeness(),
            },
            "missing_targets": self.missing_targets.to_json_value(),
            "payload": self.payload.to_json_value(),
            "stats": {
                "targets": self.stats.to_json_value(),
            },
        })
    }
}

/// Minimal partial execution witness sidecar contract.
#[derive(Debug, Clone)]
pub struct PartialExecutionWitnessSidecar {
    /// Block envelope.
    pub envelope: SidecarBlockContext,
    /// Cache descriptor assumed by the sidecar.
    pub cache_descriptor: CacheDescriptor,
    /// Cache-filtered execution witness payload.
    pub partial_execution_witness: PartialExecutionWitness,
}

impl PartialExecutionWitnessSidecar {
    /// Convert sidecar to the current JSON artifact representation.
    pub fn to_json_value(&self) -> serde_json::Value {
        json!({
            "schema_version": 2,
            "sidecar_kind": "partial_execution_witness_sidecar",
            "envelope": {
                "event": self.envelope.event,
                "block_number": self.envelope.block_number,
                "block_hash": self.envelope.block_hash,
                "parent_hash": self.envelope.parent_hash,
                "parent_state_root": self.envelope.parent_state_root,
                "post_state_root": self.envelope.post_state_root,
                "tx_count": self.envelope.tx_count,
            },
            "cache_descriptor": self.cache_descriptor.to_json_value(),
            "partial_execution_witness": self.partial_execution_witness.to_json_value(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheDescriptor, TargetCounts, TargetSourceKind};

    #[test]
    fn sidecar_json_keeps_payload_minimal() {
        let sidecar = PartialExecutionWitnessSidecar {
            envelope: SidecarBlockContext {
                event: "commit",
                block_number: 1,
                block_hash: "0xblock".to_string(),
                parent_hash: "0xparent".to_string(),
                parent_state_root: Some("0xroot".to_string()),
                post_state_root: "0xpost".to_string(),
                tx_count: 0,
            },
            cache_descriptor: CacheDescriptor::no_cache(0),
            partial_execution_witness: PartialExecutionWitness {
                missing_targets: WitnessTargets::empty(TargetSourceKind::BundleChangedState),
                stats: TargetStats {
                    full: TargetCounts::default(),
                    cache_hits: TargetCounts::default(),
                    cache_misses: TargetCounts::default(),
                },
                payload: WitnessPayload::NoneSkeletonOnly,
            },
        };

        let value = sidecar.to_json_value();
        assert_eq!(value["schema_version"], 2);
        assert_eq!(value["sidecar_kind"], "partial_execution_witness_sidecar");
        assert_eq!(value["partial_execution_witness"]["payload"]["kind"], "none_skeleton_only");
        assert!(value.get("partial_execution_witness").is_some());
        assert!(value.get("bundle_summary").is_none());
        assert!(value.get("witness_targets").is_none());
        assert!(value.get("block_number").is_none());
        assert!(value.get("missing_targets").is_none());
        assert!(value.get("proof_payload").is_none());
        assert!(value.get("stats").is_none());
    }
}
