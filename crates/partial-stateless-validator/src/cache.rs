//! Hot trie-node cache with pluggable eviction policies.
//!
//! In partial statelessness, a validator keeps a bounded cache of trie nodes;
//! the builder only ships (in the witness) what isn't cached. A node cache here
//! stores `keccak(node) => node_bytes` so it can both **serve** reads during
//! state reconstruction and **measure** how many of a block's nodes were already
//! resident (hits) versus had to come from the witness (misses).
//!
//! Add a new policy by implementing [`NodeCache`] and registering it in
//! [`build`]. Capacity is measured in number of nodes, matching the team's Go
//! `psl-cache-bench` harness so results are directly comparable.

use crate::{trie::NodeSource, witness::IndexedWitness};
use alloy_primitives::{Bytes, B256};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// A bounded cache of trie nodes (`keccak(node) => node`).
pub trait NodeCache {
    /// Policy name (for reporting).
    fn name(&self) -> &str;
    /// Whether the node is currently resident (a hit if read now).
    fn contains(&self, key: &B256) -> bool;
    /// Borrow a resident node's bytes, if present.
    fn get(&self, key: &B256) -> Option<&Bytes>;
    /// Admit a node, evicting per policy so the cache never exceeds capacity.
    fn insert(&mut self, key: B256, bytes: Bytes);
    /// Number of resident nodes.
    fn len(&self) -> usize;
    /// Whether the cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Maximum number of nodes the cache may hold.
    fn capacity(&self) -> usize;
}

/// Build a cache for the named policy and capacity (in nodes).
pub fn build(policy: &str, capacity: usize) -> eyre::Result<Box<dyn NodeCache>> {
    match policy {
        "lru" => Ok(Box::new(LruCache::new(capacity))),
        "fifo" => Ok(Box::new(FifoCache::new(capacity))),
        // Register new policies here (e.g. "sieve", "s3-fifo") — implement
        // NodeCache, add an arm, and append the name to AVAILABLE_POLICIES.
        other => Err(eyre::eyre!(
            "unknown cache policy {other:?}; available: {:?}",
            AVAILABLE_POLICIES
        )),
    }
}

/// Names of all registered policies (for `--policy all` / `--help` / errors).
pub const AVAILABLE_POLICIES: &[&str] = &["lru", "fifo"];

/// Least-recently-used cache (baseline, deterministic given access order).
///
/// Recency is updated on [`NodeCache::insert`] (admission/refresh); reads via
/// [`NodeCache::get`] do not change ordering, so per-block hit counting can
/// snapshot residency before the block's nodes are admitted.
pub struct LruCache {
    capacity: usize,
    /// `key => (bytes, last_tick)`.
    store: HashMap<B256, (Bytes, u64)>,
    /// `tick => key`, ordered so the smallest tick is the LRU victim.
    order: BTreeMap<u64, B256>,
    tick: u64,
}

impl LruCache {
    /// Create an LRU cache holding at most `capacity` nodes.
    pub fn new(capacity: usize) -> Self {
        Self { capacity, store: HashMap::new(), order: BTreeMap::new(), tick: 0 }
    }
}

impl NodeCache for LruCache {
    fn name(&self) -> &str {
        "lru"
    }

    fn contains(&self, key: &B256) -> bool {
        self.store.contains_key(key)
    }

    fn get(&self, key: &B256) -> Option<&Bytes> {
        self.store.get(key).map(|(bytes, _)| bytes)
    }

    fn insert(&mut self, key: B256, bytes: Bytes) {
        if self.capacity == 0 {
            return;
        }
        self.tick += 1;
        let now = self.tick;

        // Refresh an existing entry's recency.
        if let Some(entry) = self.store.get(&key) {
            let old_tick = entry.1;
            self.order.remove(&old_tick);
            self.order.insert(now, key);
            self.store.get_mut(&key).expect("present").1 = now;
            return;
        }

        // Evict LRU victims until there is room for the new node.
        while self.store.len() >= self.capacity {
            let Some((&victim_tick, _)) = self.order.iter().next() else { break };
            let victim_key = self.order.remove(&victim_tick).expect("present");
            self.store.remove(&victim_key);
        }

        self.store.insert(key, (bytes, now));
        self.order.insert(now, key);
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

/// First-in-first-out cache: evict in admission order, never reorder on a hit.
/// A useful contrast to LRU and a template for new policies (mirrors the team's
/// Go `fifo.go`).
pub struct FifoCache {
    capacity: usize,
    store: HashMap<B256, Bytes>,
    order: VecDeque<B256>,
}

impl FifoCache {
    /// Create a FIFO cache holding at most `capacity` nodes.
    pub fn new(capacity: usize) -> Self {
        Self { capacity, store: HashMap::new(), order: VecDeque::new() }
    }
}

impl NodeCache for FifoCache {
    fn name(&self) -> &str {
        "fifo"
    }

    fn contains(&self, key: &B256) -> bool {
        self.store.contains_key(key)
    }

    fn get(&self, key: &B256) -> Option<&Bytes> {
        self.store.get(key)
    }

    fn insert(&mut self, key: B256, bytes: Bytes) {
        if self.capacity == 0 || self.store.contains_key(&key) {
            return; // already resident: no reorder (this is what makes it FIFO)
        }
        while self.store.len() >= self.capacity {
            let Some(victim) = self.order.pop_front() else { break };
            self.store.remove(&victim);
        }
        self.store.insert(key, bytes);
        self.order.push_back(key);
    }

    fn len(&self) -> usize {
        self.store.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

/// A [`NodeSource`] that serves nodes from the cache first, then the witness —
/// exactly the `{cache ∪ witness}` set a partial-stateless validator holds.
pub struct CacheWitnessSource<'a> {
    pub cache: &'a dyn NodeCache,
    pub witness: &'a IndexedWitness,
}

impl NodeSource for CacheWitnessSource<'_> {
    fn get(&self, hash: &B256) -> Option<&[u8]> {
        if let Some(bytes) = self.cache.get(hash) {
            return Some(bytes.as_ref());
        }
        self.witness.nodes.get(hash).map(|bytes| bytes.as_ref())
    }
}
