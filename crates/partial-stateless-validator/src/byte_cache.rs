//! Abstract byte-aware cache **experiment harness** + pluggable cache policies.
//!
//! The harness ([`run`]) is policy-agnostic: it drives the validator loop —
//! *simulate a block against the current cache, then update the cache with what
//! the block touched* — and measures two things per the protocol cost model:
//!
//! - **hit rate** — fraction of accesses served from the resident cache;
//! - **sidecar bytes** — witness bytes the builder must ship for the misses.
//!
//! Any cache mechanism is testable by implementing the [`Cache`] trait. A cache
//! sees one [`Item`] at a time via [`Cache::access`] (returning the sidecar bytes
//! it costs: `0` = hit) and an end-of-block hook ([`Cache::end_block`]) to refresh
//! itself from the post-executed state. Everything is a pure function of the
//! block-ordered access sequence — the determinism the protocol requires.
//!
//! Building blocks provided here:
//! - [`ByteLru`] — client-style byte-bounded LRU (the baseline).
//! - [`ByteS3Fifo`] — byte-bounded, scan-resistant, frequency-aware S3-FIFO.
//! - [`TopCache`] — a **composable decorator** that pins the trie top
//!   (`depth <= N`, re-produced free each block) in front of any inner cache.
//! - [`V2Cache`] — our configurable v2 design = `TopCache` over [`ByteS3Fifo`]
//!   with budget split (see [`V2Config`]).

use alloy_primitives::B256;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

/// Kind of cached item: a trie node or a contract bytecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    /// A state/storage-trie node (`keccak(rlp_node)`).
    Node,
    /// A contract bytecode (`codeHash`).
    Code,
}

impl ItemKind {
    /// Discriminant used in cache keys, so a node hash and a code hash that
    /// happen to share bytes never collide.
    fn tag(self) -> u8 {
        match self {
            Self::Node => 0,
            Self::Code => 1,
        }
    }
}

/// One state access in a block's execution: an item the builder must ship if it
/// is not already resident in the validator's cache.
#[derive(Debug, Clone)]
pub struct Item {
    /// Node or code.
    pub kind: ItemKind,
    /// Content identity: `keccak(node)` for nodes, `codeHash` for code.
    pub id: B256,
    /// Size in bytes (its witness cost on a miss).
    pub size: u64,
    /// Trie depth (root = 0); meaningful for nodes, `0` for code.
    pub depth: u8,
}

/// Cache key namespaced by kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    tag: u8,
    id: B256,
}

fn key_of(it: &Item) -> Key {
    Key { tag: it.kind.tag(), id: it.id }
}

// ----------------------------------------------------------------------------
// Abstract experiment harness
// ----------------------------------------------------------------------------

/// A pluggable cache mechanism under test.
///
/// `access` accounts for one read during block execution and returns the bytes
/// the **sidecar (builder witness)** must serve: `0` on a hit, `item.size` on a
/// miss. Implementations update their own residency/recency/frequency here.
/// `end_block` is an optional hook, called once the block has executed, handed
/// every item the block touched (the post-executed state the validator now
/// holds) so the cache can refresh itself (e.g. re-pin the trie top).
pub trait Cache {
    /// Policy name (for reporting); may be composed (e.g. `top3+s3fifo`).
    fn name(&self) -> String;
    /// Account for one access; return sidecar bytes (`0` = hit).
    fn access(&mut self, item: &Item) -> u64;
    /// Refresh from the block's post-executed state. Default: no-op.
    fn end_block(&mut self, _block: &[Item]) {}
}

/// What an experiment measures over a whole trace.
#[derive(Debug, Default, Clone, Copy)]
pub struct Metrics {
    /// Total accesses seen.
    pub accesses: u64,
    /// Accesses served from cache (sidecar cost 0).
    pub hits: u64,
    /// Sum of all accessed item sizes (hits + misses) — the uncached witness.
    pub bytes_accessed: u64,
    /// Witness bytes the sidecar shipped (misses only) — the real protocol cost.
    pub bytes_served: u64,
    /// Of `bytes_served`: trie-node bytes.
    pub node_served: u64,
    /// Of `bytes_served`: bytecode bytes.
    pub code_served: u64,
}

impl Metrics {
    /// Accesses that missed.
    pub fn misses(&self) -> u64 {
        self.accesses - self.hits
    }
    /// Fraction of accesses served from cache.
    pub fn hit_rate(&self) -> f64 {
        if self.accesses > 0 {
            self.hits as f64 / self.accesses as f64
        } else {
            0.0
        }
    }
}

/// Replay a block-ordered trace through `cache`, measuring hit rate and sidecar
/// bytes. This is the whole experiment: it knows nothing about depth, budgets,
/// or top-pinning — those are the cache's concern.
pub fn run(cache: &mut dyn Cache, trace: &[Vec<Item>]) -> Metrics {
    let mut m = Metrics::default();
    for block in trace {
        // 1. Simulate the block on top of the current cache.
        for it in block {
            let served = cache.access(it);
            m.accesses += 1;
            m.bytes_accessed += it.size;
            if served == 0 {
                m.hits += 1;
            } else {
                m.bytes_served += served;
                match it.kind {
                    ItemKind::Node => m.node_served += served,
                    ItemKind::Code => m.code_served += served,
                }
            }
        }
        // 2. Update the cache with the post-executed state.
        cache.end_block(block);
    }
    m
}

/// Uncached-witness ceiling for context: the best any cache can do is serve
/// every *repeat* access, so the minimum sidecar bytes is the sum of each
/// distinct item's size (its compulsory first miss), and the max hit rate is
/// `1 - distinct/accesses`.
#[derive(Debug, Default, Clone, Copy)]
pub struct TraceFloor {
    /// Distinct items (compulsory misses).
    pub distinct: u64,
    /// Total accesses.
    pub accesses: u64,
    /// Sum of distinct item sizes (minimum possible sidecar bytes).
    pub compulsory_bytes: u64,
}

impl TraceFloor {
    /// Best hit rate any cache could reach on this trace.
    pub fn max_hit_rate(&self) -> f64 {
        if self.accesses > 0 {
            (self.accesses - self.distinct) as f64 / self.accesses as f64
        } else {
            0.0
        }
    }
}

/// Compute the compulsory floor of a trace (cache-independent).
pub fn trace_floor(trace: &[Vec<Item>]) -> TraceFloor {
    let mut seen: HashMap<Key, u64> = HashMap::new();
    let mut f = TraceFloor::default();
    for block in trace {
        for it in block {
            f.accesses += 1;
            if seen.insert(key_of(it), it.size).is_none() {
                f.distinct += 1;
                f.compulsory_bytes += it.size;
            }
        }
    }
    f
}

// ----------------------------------------------------------------------------
// Composable building block: pin the trie top in front of any inner cache
// ----------------------------------------------------------------------------

/// Decorator that makes the trie top (`depth <= top_n`) free: the validator
/// re-executes each block and re-produces the top, so those reads never enter
/// the witness. Everything else (deep nodes + code) flows to the inner cache.
/// Compose it with any [`Cache`] to give that policy the top-pinning benefit.
pub struct TopCache<C> {
    top_n: u8,
    inner: C,
}

impl<C: Cache> TopCache<C> {
    /// Pin `depth <= top_n` in front of `inner`.
    pub fn new(top_n: u8, inner: C) -> Self {
        Self { top_n, inner }
    }
}

impl<C: Cache> Cache for TopCache<C> {
    fn name(&self) -> String {
        format!("top{}+{}", self.top_n, self.inner.name())
    }

    fn access(&mut self, it: &Item) -> u64 {
        // Top region: re-produced each block by the validator -> never shipped.
        if it.kind == ItemKind::Node && it.depth <= self.top_n {
            return 0;
        }
        self.inner.access(it)
    }

    fn end_block(&mut self, block: &[Item]) {
        self.inner.end_block(block);
    }
}

// ----------------------------------------------------------------------------
// Configurable v2 design
// ----------------------------------------------------------------------------

/// Average resident bytes of a near-top trie node (branch/extension ~0.5 KB),
/// used to charge the memory cost of pinning the top-N partial trie.
pub const DEFAULT_TOP_NODE_SIZE: u64 = 480;

/// Resident bytes needed to hold the full top-`n` partial trie so every
/// `depth <= n` read hits: `Σ_{d=0..=n} 16^d × node_size` (grows 16×/level).
pub fn top_resident(top_n: u8, node_size: u64) -> u64 {
    let (mut positions, mut level) = (0u128, 1u128);
    for _ in 0..=top_n {
        positions += level;
        level = level.saturating_mul(16);
    }
    (positions.saturating_mul(node_size as u128)).min(u64::MAX as u128) as u64
}

/// Largest `N` whose pinned top fits within `total_budget` (leaving room for the
/// deep+code cache). A convenience for `--top-n auto`.
pub fn auto_top_n(total_budget: u64, node_size: u64) -> u8 {
    let mut best = 0u8;
    for n in 0..=10u8 {
        if top_resident(n, node_size) < total_budget {
            best = n;
        } else {
            break;
        }
    }
    best
}

/// Configuration of the v2 cache. Everything tunable lives here.
#[derive(Debug, Clone, Copy)]
pub struct V2Config {
    /// Total memory budget in bytes (top residency + deep/code cache).
    pub total_budget: u64,
    /// Trie depth pinned as free (re-produced each block).
    pub top_n: u8,
    /// Assumed resident size of a top node, for the residency charge.
    pub top_node_size: u64,
}

impl V2Config {
    /// A config with the default top-node size.
    pub fn new(total_budget: u64, top_n: u8) -> Self {
        Self { total_budget, top_n, top_node_size: DEFAULT_TOP_NODE_SIZE }
    }
    /// Memory reserved to pin the top-`top_n` partial trie.
    pub fn top_resident(&self) -> u64 {
        top_resident(self.top_n, self.top_node_size)
    }
    /// Budget left for the deep-node + code cache after pinning the top.
    pub fn cache_budget(&self) -> u64 {
        self.total_budget.saturating_sub(self.top_resident())
    }
}

/// The v2 best-practice cache: pin the trie top (free, re-produced each block)
/// and serve everything below it — deep nodes **and** code, unified — from a
/// byte-bounded scan-resistant S3-FIFO sized to the leftover budget. Configure
/// via [`V2Config`].
pub struct V2Cache {
    cfg: V2Config,
    inner: TopCache<ByteS3Fifo>,
}

/// Code frequency head start in the v2 deep cache. `>= 1` makes code promote to
/// the protected queue instead of being evicted by one-shot node churn; 2 is the
/// robust setting (3 is identical, "straight to main" hurts at tight budgets).
pub const V2_CODE_BOOST: i8 = 2;

impl V2Cache {
    /// Build the v2 cache from its config. The deep cache is a byte-bounded
    /// S3-FIFO with **code protection** (a frequency head start for bytecode —
    /// the biggest byte saver), the one tuning that beat plain S3-FIFO at every
    /// budget. Use [`V2Cache::plain`] for the un-protected baseline.
    pub fn new(cfg: V2Config) -> Self {
        let inner =
            TopCache::new(cfg.top_n, ByteS3Fifo::new(cfg.cache_budget()).with_code_boost(V2_CODE_BOOST, false));
        Self { cfg, inner }
    }

    /// The v2 design without code protection (plain S3-FIFO deep cache) — the
    /// baseline the code-protected default improves on.
    pub fn plain(cfg: V2Config) -> Self {
        let inner = TopCache::new(cfg.top_n, ByteS3Fifo::new(cfg.cache_budget()));
        Self { cfg, inner }
    }

    /// Build the v2 cache with a trie-depth admission prior on the deep cache
    /// (warm bands get an admission boost).
    pub fn with_warm(cfg: V2Config, warm: &WarmConfig) -> Self {
        let inner = TopCache::new(
            cfg.top_n,
            ByteS3Fifo::with_warm(cfg.cache_budget(), warm.bands.clone(), warm.to_main, warm.init_freq),
        );
        Self { cfg, inner }
    }

    /// Build the v2 cache around a pre-tuned deep S3-FIFO (it must be sized to
    /// `cfg.cache_budget()`). Used for the ghost-reinforcement / code-boost
    /// variants.
    pub fn from_s3(cfg: V2Config, s3: ByteS3Fifo) -> Self {
        Self { cfg, inner: TopCache::new(cfg.top_n, s3) }
    }
}

/// Trie-depth admission prior for the v2 deep cache. Nodes whose depth is in any
/// `bands` range (inclusive) are boosted on admission. The defaults target the
/// observed reuse hotspot (hot-contract storage, depths ~10–13).
#[derive(Debug, Clone)]
pub struct WarmConfig {
    /// Inclusive depth ranges treated as reuse-heavy.
    pub bands: Vec<(u8, u8)>,
    /// Admit warm nodes straight to `main` (skip probation).
    pub to_main: bool,
    /// Initial frequency for warm nodes.
    pub init_freq: i8,
}

impl Default for WarmConfig {
    fn default() -> Self {
        // Hot-contract storage tries (depths ~10–13) carry most cross-block reuse;
        // a freq head start (kept in probation) promotes them without flooding main.
        Self { bands: vec![(10, 13)], to_main: false, init_freq: 1 }
    }
}

impl Cache for V2Cache {
    fn name(&self) -> String {
        format!(
            "v2(budget={}MB,topN={},cache={}MB)",
            self.cfg.total_budget / 1_000_000,
            self.cfg.top_n,
            self.cfg.cache_budget() / 1_000_000,
        )
    }

    fn access(&mut self, it: &Item) -> u64 {
        self.inner.access(it)
    }

    fn end_block(&mut self, block: &[Item]) {
        self.inner.end_block(block);
    }
}

// ----------------------------------------------------------------------------
// Baseline policy: byte-bounded LRU
// ----------------------------------------------------------------------------

/// Byte-bounded **LRU** — the client-style baseline (geth/nethermind/reth caches
/// are LRU/FIFO-like). Recency via a monotonic tick + ordered map, so it is
/// deterministic given the access order and needs no external crate.
pub struct ByteLru {
    budget: u64,
    resident: u64,
    tick: u64,
    /// `key => (size, last_tick)`.
    store: HashMap<Key, (u64, u64)>,
    /// `tick => key`, smallest tick = LRU victim.
    order: BTreeMap<u64, Key>,
}

impl ByteLru {
    /// Create a byte-bounded LRU holding at most `budget` bytes.
    pub fn new(budget: u64) -> Self {
        Self { budget, resident: 0, tick: 0, store: HashMap::new(), order: BTreeMap::new() }
    }
}

impl Cache for ByteLru {
    fn name(&self) -> String {
        "lru".to_string()
    }

    fn access(&mut self, it: &Item) -> u64 {
        let k = key_of(it);
        if let Some(entry) = self.store.get(&k) {
            // Hit: refresh recency.
            let old = entry.1;
            self.order.remove(&old);
            self.tick += 1;
            self.order.insert(self.tick, k);
            self.store.get_mut(&k).expect("present").1 = self.tick;
            return 0;
        }
        if it.size > self.budget {
            return it.size; // can never fit; don't pollute
        }
        self.tick += 1;
        self.store.insert(k, (it.size, self.tick));
        self.order.insert(self.tick, k);
        self.resident += it.size;
        while self.resident > self.budget {
            let Some((&t, _)) = self.order.iter().next() else { break };
            let vk = self.order.remove(&t).expect("present");
            if let Some((s, _)) = self.store.remove(&vk) {
                self.resident -= s;
            }
        }
        it.size
    }
}

// ----------------------------------------------------------------------------
// Baseline policy: byte-bounded FIFO (geth fastcache-style)
// ----------------------------------------------------------------------------

/// Byte-bounded **FIFO** — evict in insertion order, never reorder on a hit.
/// Models go-ethereum's `fastcache` clean-trie cache (a sharded ring buffer).
pub struct ByteFifo {
    budget: u64,
    resident: u64,
    store: HashMap<Key, u64>,
    order: VecDeque<Key>,
}

impl ByteFifo {
    /// Create a byte-bounded FIFO holding at most `budget` bytes.
    pub fn new(budget: u64) -> Self {
        Self { budget, resident: 0, store: HashMap::new(), order: VecDeque::new() }
    }
}

impl Cache for ByteFifo {
    fn name(&self) -> String {
        "fifo".to_string()
    }

    fn access(&mut self, it: &Item) -> u64 {
        let k = key_of(it);
        if self.store.contains_key(&k) {
            return 0; // hit; no reorder (this is what makes it FIFO)
        }
        if it.size > self.budget {
            return it.size;
        }
        self.store.insert(k, it.size);
        self.order.push_back(k);
        self.resident += it.size;
        while self.resident > self.budget {
            let Some(victim) = self.order.pop_front() else { break };
            if let Some(s) = self.store.remove(&victim) {
                self.resident -= s;
            }
        }
        it.size
    }
}

// ----------------------------------------------------------------------------
// Baseline policy: byte-bounded CLOCK (nethermind ClockCache-style)
// ----------------------------------------------------------------------------

/// Byte-bounded **CLOCK** (second-chance): a ring with a reference bit set on
/// hit; eviction skips (and clears) referenced entries, evicting the first
/// unreferenced one. Models nethermind's `ClockCache`. All ops O(1) amortized.
pub struct ByteClock {
    budget: u64,
    resident: u64,
    store: HashMap<Key, (u64, bool)>, // size, referenced
    ring: VecDeque<Key>,
}

impl ByteClock {
    /// Create a byte-bounded CLOCK cache holding at most `budget` bytes.
    pub fn new(budget: u64) -> Self {
        Self { budget, resident: 0, store: HashMap::new(), ring: VecDeque::new() }
    }
}

impl Cache for ByteClock {
    fn name(&self) -> String {
        "clock".to_string()
    }

    fn access(&mut self, it: &Item) -> u64 {
        let k = key_of(it);
        if let Some(e) = self.store.get_mut(&k) {
            e.1 = true; // hit: set reference bit
            return 0;
        }
        if it.size > self.budget {
            return it.size;
        }
        self.store.insert(k, (it.size, false));
        self.ring.push_back(k);
        self.resident += it.size;
        while self.resident > self.budget {
            let Some(front) = self.ring.pop_front() else { break };
            match self.store.get_mut(&front) {
                Some(e) if e.1 => {
                    e.1 = false; // second chance: clear bit, requeue
                    self.ring.push_back(front);
                }
                Some(e) => {
                    let s = e.0;
                    self.store.remove(&front);
                    self.resident -= s;
                }
                None => {}
            }
        }
        it.size
    }
}

// ----------------------------------------------------------------------------
// Policy: byte-bounded S3-FIFO
// ----------------------------------------------------------------------------

/// Byte-bounded **S3-FIFO**. Scan-resistant (a `small` probation FIFO filters the
/// one-shot tail) and frequency-aware (a `main` FIFO with a 2-bit counter gives
/// reused items second chances), with a `ghost` queue that fast-tracks
/// recently-evicted keys back into `main`. All ops are O(1). Code is
/// high-frequency, so this unified cache retains hot bytecode automatically — no
/// separate code cache needed. Faithful port of Go `byteS3`.
pub struct ByteS3Fifo {
    budget: u64,
    resident: u64,
    small_resident: u64,
    s_target: u64,
    small: VecDeque<Key>,
    main: VecDeque<Key>,
    items: HashMap<Key, Entry>,
    ghost: HashSet<Key>,
    ghost_list: VecDeque<(Key, u64)>,
    ghost_bytes: u64,
    /// Optional trie-depth admission prior (off when empty). Nodes whose depth
    /// falls in a structurally reuse-heavy band (e.g. hot-contract storage,
    /// depths 10–13) get an admission boost so they survive the one-shot churn
    /// instead of starting cold in probation. Deterministic and
    /// builder-reproducible (depth is recomputable).
    warm_bands: Vec<(u8, u8)>,
    /// If true, warm nodes skip probation (admitted straight to `main`).
    warm_to_main: bool,
    /// Initial frequency for warm nodes (a head start toward the protected queue).
    warm_init_freq: i8,
    /// Frequency a ghost-hit (proven reuse) is re-admitted to `main` with. The
    /// stock S3-FIFO uses 0 (one eviction cycle of grace); >0 gives proven-reused
    /// items — code included — a real second chance without a depth heuristic.
    ghost_readmit_freq: i8,
    /// Initial frequency head start for **code** items (bytecode is reused ~20×;
    /// keeping it resident is the single biggest byte saver).
    code_init_freq: i8,
    /// Admit code straight to `main` (skip probation) — strongest code protection,
    /// the unified-cache equivalent of clients' dedicated code caches.
    code_to_main: bool,
}

/// A resident entry: its size and 2-bit frequency counter (0..=3).
struct Entry {
    size: u64,
    freq: i8,
}

impl ByteS3Fifo {
    /// Create a byte-bounded S3-FIFO holding at most `budget` bytes.
    pub fn new(budget: u64) -> Self {
        Self {
            budget,
            resident: 0,
            small_resident: 0,
            s_target: budget / 10,
            small: VecDeque::new(),
            main: VecDeque::new(),
            items: HashMap::new(),
            ghost: HashSet::new(),
            ghost_list: VecDeque::new(),
            ghost_bytes: 0,
            warm_bands: Vec::new(),
            warm_to_main: false,
            warm_init_freq: 0,
            ghost_readmit_freq: 0,
            code_init_freq: 0,
            code_to_main: false,
        }
    }

    /// Re-admit ghost hits (proven reuse) to `main` with frequency `f` (a real
    /// second chance). Helps proven-reused items — code included — uniformly.
    pub fn with_ghost_readmit(mut self, f: i8) -> Self {
        self.ghost_readmit_freq = f;
        self
    }

    /// Give freshly-admitted **code** an initial frequency head start `f`, so the
    /// single biggest byte saver (bytecode, ~20× reuse) resists eviction by the
    /// one-shot trie-node churn. If `to_main`, code also skips probation.
    pub fn with_code_boost(mut self, f: i8, to_main: bool) -> Self {
        self.code_init_freq = f;
        self.code_to_main = to_main;
        self
    }

    /// Create an S3-FIFO with a trie-depth admission prior: nodes whose depth is
    /// in any `warm_bands` range (inclusive) are boosted on first admission —
    /// either straight into `main` (`to_main`) or kept in probation with an
    /// initial frequency `init_freq`.
    pub fn with_warm(budget: u64, warm_bands: Vec<(u8, u8)>, to_main: bool, init_freq: i8) -> Self {
        let mut c = Self::new(budget);
        c.warm_bands = warm_bands;
        c.warm_to_main = to_main;
        c.warm_init_freq = init_freq;
        c
    }

    /// Whether `depth` falls in a warm (reuse-heavy) band.
    fn is_warm(&self, depth: u8) -> bool {
        self.warm_bands.iter().any(|&(lo, hi)| depth >= lo && depth <= hi)
    }

    fn evict(&mut self) {
        if self.main.is_empty() || self.small_resident > self.s_target {
            self.evict_small();
        } else {
            self.evict_main();
        }
    }

    fn evict_small(&mut self) {
        let Some(k) = self.small.pop_back() else {
            self.evict_main();
            return;
        };
        let e = self.items.get_mut(&k).expect("small entry present");
        self.small_resident -= e.size;
        if e.freq > 0 {
            // Promote to main (total resident unchanged).
            e.freq = 0;
            self.main.push_front(k);
        } else {
            let size = e.size;
            self.items.remove(&k);
            self.resident -= size;
            self.add_ghost(k, size);
        }
    }

    fn evict_main(&mut self) {
        let Some(k) = self.main.pop_back() else { return };
        let e = self.items.get_mut(&k).expect("main entry present");
        if e.freq > 0 {
            e.freq -= 1; // second chance
            self.main.push_front(k);
        } else {
            let size = e.size;
            self.items.remove(&k);
            self.resident -= size;
        }
    }

    fn add_ghost(&mut self, k: Key, size: u64) {
        self.ghost.insert(k);
        self.ghost_list.push_front((k, size));
        self.ghost_bytes += size;
        while self.ghost_bytes > self.budget {
            let Some((old, s)) = self.ghost_list.pop_back() else { break };
            self.ghost.remove(&old);
            self.ghost_bytes -= s;
        }
    }
}

impl Cache for ByteS3Fifo {
    fn name(&self) -> String {
        "s3fifo".to_string()
    }

    fn access(&mut self, it: &Item) -> u64 {
        let k = key_of(it);
        if let Some(e) = self.items.get_mut(&k) {
            if e.freq < 3 {
                e.freq += 1;
            }
            return 0;
        }
        if it.size > self.budget {
            return it.size; // can never fit; don't pollute
        }
        if self.ghost.remove(&k) {
            // Recently evicted then re-requested = proven reuse. Admit to main
            // with a configurable second chance (stock S3-FIFO uses 0).
            self.items.insert(k, Entry { size: it.size, freq: self.ghost_readmit_freq });
            self.main.push_front(k);
        } else if it.kind == ItemKind::Code && self.code_to_main {
            // Code is the biggest byte saver -> protect it like a dedicated code
            // cache by admitting straight to main (skip probation).
            self.items.insert(k, Entry { size: it.size, freq: self.code_init_freq });
            self.main.push_front(k);
        } else if it.kind == ItemKind::Node && self.warm_to_main && self.is_warm(it.depth) {
            // Reuse-heavy structural band: skip probation.
            self.items.insert(k, Entry { size: it.size, freq: self.warm_init_freq });
            self.main.push_front(k);
        } else {
            // One-shot tail starts cold in probation. Code gets a head start (it
            // is the biggest byte saver); warm depth bands get one too.
            let freq = match it.kind {
                ItemKind::Code => self.code_init_freq,
                ItemKind::Node if self.is_warm(it.depth) => self.warm_init_freq,
                _ => 0,
            };
            self.items.insert(k, Entry { size: it.size, freq });
            self.small.push_front(k);
            self.small_resident += it.size;
        }
        self.resident += it.size;
        while self.resident > self.budget {
            self.evict();
        }
        it.size
    }
}

// ----------------------------------------------------------------------------
// Dedicated frequency code cache + split cache (Rule 3 alternative)
// ----------------------------------------------------------------------------

/// Byte-bounded **LFU** (least-frequently-used, LRU tie-break): keeps the most
/// frequently accessed items, evicting the lowest-count (then least-recent) when
/// over budget. Used as a dedicated *code* cache — "keep the top contracts by
/// access frequency". Deterministic (counts/ticks are a pure function of the
/// access order). O(log n) per access.
pub struct ByteLfu {
    budget: u64,
    resident: u64,
    tick: u64,
    store: HashMap<Key, LfuEntry>,
    /// `(count, tick) => key`, smallest = lowest-frequency, least-recent victim.
    order: BTreeMap<(u64, u64), Key>,
}

struct LfuEntry {
    size: u64,
    count: u64,
    tick: u64,
}

impl ByteLfu {
    /// Create a byte-bounded LFU holding at most `budget` bytes.
    pub fn new(budget: u64) -> Self {
        Self { budget, resident: 0, tick: 0, store: HashMap::new(), order: BTreeMap::new() }
    }
}

impl Cache for ByteLfu {
    fn name(&self) -> String {
        "lfu".to_string()
    }

    fn access(&mut self, it: &Item) -> u64 {
        let k = key_of(it);
        self.tick += 1;
        if let Some(e) = self.store.get(&k) {
            let (old_count, old_tick) = (e.count, e.tick);
            self.order.remove(&(old_count, old_tick));
            let e = self.store.get_mut(&k).expect("present");
            e.count += 1;
            e.tick = self.tick;
            self.order.insert((e.count, e.tick), k);
            return 0;
        }
        if it.size > self.budget {
            return it.size;
        }
        self.store.insert(k, LfuEntry { size: it.size, count: 1, tick: self.tick });
        self.order.insert((1, self.tick), k);
        self.resident += it.size;
        while self.resident > self.budget {
            let Some((&victim_key, &vk)) = self.order.iter().next() else { break };
            self.order.remove(&victim_key);
            if let Some(e) = self.store.remove(&vk) {
                self.resident -= e.size;
            }
        }
        it.size
    }
}

/// v2 with a **dedicated code cache** (Rule 3 alternative): code goes to its own
/// byte-bounded LFU (`code_budget`), deep nodes to a plain S3-FIFO with the rest.
/// Top region is free as in [`V2Cache`]. Tests whether reserving budget for the
/// hottest contracts beats the unified code-protected cache.
pub struct V2Split {
    cfg: V2Config,
    code_budget: u64,
    top_n: u8,
    code: ByteLfu,
    deep: ByteS3Fifo,
}

impl V2Split {
    /// Build with `code_budget` bytes reserved for the code LFU; deep nodes get
    /// `cache_budget - code_budget`.
    pub fn new(cfg: V2Config, code_budget: u64) -> Self {
        let code_budget = code_budget.min(cfg.cache_budget());
        let deep_budget = cfg.cache_budget().saturating_sub(code_budget);
        Self {
            cfg,
            code_budget,
            top_n: cfg.top_n,
            code: ByteLfu::new(code_budget),
            deep: ByteS3Fifo::new(deep_budget),
        }
    }
}

impl Cache for V2Split {
    fn name(&self) -> String {
        format!("v2split(code={}MB,deep={}MB)", self.code_budget / 1_000_000, (self.cfg.cache_budget() - self.code_budget) / 1_000_000)
    }

    fn access(&mut self, it: &Item) -> u64 {
        match it.kind {
            ItemKind::Node if it.depth <= self.top_n => 0, // top region: free
            ItemKind::Code => self.code.access(it),
            ItemKind::Node => self.deep.access(it),
        }
    }
}

// ----------------------------------------------------------------------------
// Factory
// ----------------------------------------------------------------------------

/// Names of all registered policies (for `--policies` / errors).
pub const POLICIES: &[&str] =
    &["lru", "fifo", "clock", "s3fifo", "lru+top", "v2plain", "v2", "v2depth", "v2split"];

/// Build a [`Cache`] by name, given the total byte budget and top-N config.
///
/// - `lru` / `s3fifo` — the raw policy over the whole budget (no top pinning).
/// - `lru+top` — LRU under the same top-pinning + budget split as v2 (isolates
///   the *policy* contribution from the *top-pinning* contribution).
/// - `v2plain` — the v2 design with a plain S3-FIFO deep cache (the baseline).
/// - `v2` — v2 with **code protection** ([`V2Cache::new`]), the tuned default.
/// - `v2depth` — v2 with the trie-depth admission prior ([`WarmConfig`]); kept as
///   a documented negative result (it steals `main` space from code and loses).
/// - `v2split` — v2 with a dedicated `code_budget`-byte LFU code cache ([`V2Split`]).
pub fn build(
    policy: &str,
    total_budget: u64,
    top_n: u8,
    top_node_size: u64,
    code_budget: u64,
) -> eyre::Result<Box<dyn Cache>> {
    let cache_budget = total_budget.saturating_sub(top_resident(top_n, top_node_size));
    let cfg = V2Config { total_budget, top_n, top_node_size };
    Ok(match policy {
        "lru" => Box::new(ByteLru::new(total_budget)),
        "fifo" => Box::new(ByteFifo::new(total_budget)),
        "clock" => Box::new(ByteClock::new(total_budget)),
        "s3fifo" => Box::new(ByteS3Fifo::new(total_budget)),
        "lru+top" => Box::new(TopCache::new(top_n, ByteLru::new(cache_budget))),
        "v2plain" => Box::new(V2Cache::plain(cfg)),
        "v2" => Box::new(V2Cache::new(cfg)),
        "v2depth" => Box::new(V2Cache::with_warm(cfg, &WarmConfig::default())),
        "v2split" => Box::new(V2Split::new(cfg, code_budget)),
        other => eyre::bail!("unknown policy {other:?}; available: {POLICIES:?}"),
    })
}
