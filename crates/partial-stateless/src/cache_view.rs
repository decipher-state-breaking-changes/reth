use crate::CacheDescriptor;

/// Read-only view of a deterministic cache profile.
pub trait CacheView {
    /// Descriptor for the cache policy and anchor block.
    fn descriptor(&self) -> &CacheDescriptor;

    /// Returns true when the cache contains the account pre-state.
    fn contains_account(&self, address: &str) -> bool;

    /// Returns true when the cache contains the storage pre-state.
    fn contains_storage(&self, address: &str, slot: &str) -> bool;

    /// Returns true when the cache contains the bytecode preimage.
    fn contains_code(&self, code_hash: &str) -> bool;

    /// Returns true when the cache contains the ancestor header.
    fn contains_header(&self, number: u64) -> bool;
}

/// Debug cache view where every target is missing.
#[derive(Debug, Clone)]
pub struct NoCacheView {
    descriptor: CacheDescriptor,
}

/// Debug cache view where every target is present.
#[derive(Debug, Clone)]
pub struct AllHitDebugView {
    descriptor: CacheDescriptor,
}

impl NoCacheView {
    /// Create a no-cache view anchored at `cache_block`.
    pub const fn new(cache_block: u64) -> Self {
        Self { descriptor: CacheDescriptor::no_cache(cache_block) }
    }
}

impl AllHitDebugView {
    /// Create an all-hit debug view anchored at `cache_block`.
    pub const fn new(cache_block: u64) -> Self {
        Self { descriptor: CacheDescriptor::all_hit_debug(cache_block) }
    }
}

impl CacheView for NoCacheView {
    fn descriptor(&self) -> &CacheDescriptor {
        &self.descriptor
    }

    fn contains_account(&self, _address: &str) -> bool {
        false
    }

    fn contains_storage(&self, _address: &str, _slot: &str) -> bool {
        false
    }

    fn contains_code(&self, _code_hash: &str) -> bool {
        false
    }

    fn contains_header(&self, _number: u64) -> bool {
        false
    }
}

impl CacheView for AllHitDebugView {
    fn descriptor(&self) -> &CacheDescriptor {
        &self.descriptor
    }

    fn contains_account(&self, _address: &str) -> bool {
        true
    }

    fn contains_storage(&self, _address: &str, _slot: &str) -> bool {
        true
    }

    fn contains_code(&self, _code_hash: &str) -> bool {
        true
    }

    fn contains_header(&self, _number: u64) -> bool {
        true
    }
}
