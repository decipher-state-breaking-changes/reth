//! Replaying one corpus through one cache policy, and pricing the result under every tree.
//!
//! The comparison rests on one fact about the system being studied: the cache policy is defined
//! over addresses and slots, not over tree positions. Which keys a block misses is therefore the
//! same number whichever tree the state is committed in, and it can be replayed exactly — this
//! module drives the production `NetworkStateCache` and the production eviction policy rather than
//! reimplementing either, so the miss set it prices is the miss set the measured MPT arms were
//! built from. The report checks that equality against the recorded run instead of assuming it.
//!
//! What changes between arms is only what a miss *costs*. That keeps every difference this study
//! reports attributable to the commitment scheme, which is the one thing it is trying to measure.

use crate::{
    coverage::CodeCoverage,
    keys::{
        mpt_account_key, mpt_storage_key, Eip6800Keys, Eip7864Keys, HeaderLayout, StemId,
        TreeEmbedding, TreeKey, TreeRegion,
    },
    mpt::hexary_path_nodes,
    population::BackgroundPopulation,
    witness::{
        binary_witness, verkle_witness, RegionPopulations, RetainedStems, StemOccupancy,
        StemTargets, WitnessCost,
    },
};
use alloy_primitives::{Address, B256};
use partial_stateless::{
    accessed_state::BlockAccessedState,
    network_cache::{MissResult, NetworkStateCache},
    policy::LastNBlocksPolicy,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
};

/// Bytes of the aggregated polynomial opening a Verkle proof carries once per block.
///
/// Eight rounds of two group elements, plus the final scalar and the aggregated commitment. It does
/// not grow with the number of keys opened, which is the property that makes the scheme worth its
/// elliptic curve; carrying it as a named constant keeps that visible in the byte breakdown.
pub const VERKLE_IPA_PROOF_BYTES: u64 = 8 * 2 * 32 + 32 + 32;

/// Deepest binary path the study will follow before declaring a key alone.
const MAX_BINARY_DEPTH: u32 = 64;

/// One cache policy under test.
#[derive(Debug, Clone)]
pub struct ArmSpec {
    /// Name as the measured run reports it.
    pub name: String,
    /// Blocks an account stays warm for; `None` is the no-cache arm.
    pub account_window: Option<u64>,
    /// Blocks a storage slot or bytecode stays warm for.
    pub storage_window: Option<u64>,
}

impl ArmSpec {
    /// The no-cache baseline every other arm is quoted against.
    pub fn weak() -> Self {
        Self { name: "weak".into(), account_window: None, storage_window: None }
    }

    /// A Last-N-blocks arm.
    pub fn windows(account: u64, storage: u64) -> Self {
        Self {
            name: format!("{account}/{storage}"),
            account_window: Some(account),
            storage_window: Some(storage),
        }
    }
}

/// Modelled state sizes and layout choices the whole run is fixed to.
#[derive(Debug, Clone)]
pub struct StudyParams {
    /// Accounts in the modelled state, which is also the header-region stem count.
    pub account_population: u64,
    /// Storage slots in the modelled state.
    pub storage_population: u64,
    /// Stems the whole unified tree holds.
    pub total_stem_population: u64,
    /// Stems the out-of-header code region holds.
    pub code_stem_population: u64,
    /// Which EIP-7864 header layout to place binary-tree keys under.
    pub header_layout: HeaderLayout,
    /// Fraction of a contract's code chunks a call is assumed to run.
    ///
    /// Only consulted when [`Self::measured_coverage`] is absent.
    pub code_coverage: f64,
    /// Which chunks each bytecode was measured to run, when a measurement exists.
    ///
    /// A measurement replaces the assumption per bytecode rather than in aggregate: contracts
    /// differ enormously in how much of themselves a call touches, and one global fraction
    /// applied to all of them would reproduce the mean while getting every stem's occupancy
    /// wrong.
    pub measured_coverage: Option<Arc<CodeCoverage>>,
    /// How much of an opened stem is assumed occupied beyond what the corpus names.
    ///
    /// `None` derives it from the measured slot count and the modelled stem count, which are not
    /// independent: a state of a given size spread over fewer stems fills each of them more. Set
    /// it explicitly only to quote a bound.
    pub stem_occupancy: Option<StemOccupancy>,
    /// Storage slots a modelled per-account storage trie holds, for the MPT calibration only.
    pub mpt_storage_trie_population: u64,
}

impl Default for StudyParams {
    fn default() -> Self {
        Self {
            account_population: 412_044_818,
            storage_population: 1_636_513_307,
            total_stem_population: 2_000_000_000,
            code_stem_population: 3_000_000,
            header_layout: HeaderLayout::TABLE,
            code_coverage: 1.0,
            measured_coverage: None,
            stem_occupancy: None,
            mpt_storage_trie_population: 4_096,
        }
    }
}

impl StudyParams {
    /// Stems in the out-of-header storage region: whatever the total is not accounted for by the
    /// header and code regions.
    pub const fn storage_stem_population(&self) -> u64 {
        self.total_stem_population
            .saturating_sub(self.account_population)
            .saturating_sub(self.code_stem_population)
    }

    /// The occupancy this run prices stems at.
    ///
    /// Derived rather than assumed when it is not set. The measured slot count and the modelled
    /// stem count together fix the mean occupancy of a storage stem, so sweeping the two
    /// independently would price states that do not exist: 1.6 billion slots over 1.6 billion stems
    /// is one slot each, and over 85 million stems is nineteen. One observed suffix is subtracted,
    /// because a stem is only opened when something in it was touched.
    pub fn effective_occupancy(&self) -> StemOccupancy {
        if let Some(explicit) = self.stem_occupancy {
            return explicit
        }
        let stems = self.storage_stem_population();
        let mean = if stems == 0 { 0.0 } else { self.storage_population as f64 / stems as f64 };
        StemOccupancy {
            outside_header: (mean.round() as u32).saturating_sub(1).min(255),
            // A header stem's occupancy is mostly known — basic data, code hash, and the code
            // chunks that follow from the code length — and the corpus puts only 2.7%
            // of accessed slots in one, which over 412 million accounts is well under
            // one slot each.
            in_header: 0,
        }
    }
}

/// What one block cost under one arm.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BlockResult {
    /// Canonical block number.
    pub block_number: u64,
    /// Arm name.
    pub arm: String,
    /// Accounts the block touched.
    pub accessed_accounts: usize,
    /// Storage slots the block touched.
    pub accessed_storage: usize,
    /// Bytecodes the block touched.
    pub accessed_codes: usize,
    /// Accounts the cache did not hold.
    pub missed_accounts: usize,
    /// Storage slots the cache did not hold.
    pub missed_storage: usize,
    /// Bytecodes the cache did not hold.
    pub missed_codes: usize,
    /// Raw bytes of the bytecode the block missed, as the MPT ships it.
    pub missed_code_bytes: u64,
    /// Accounts whose code leaves the tree witness had to open.
    ///
    /// Higher than `missed_codes` whenever one missed bytecode is deployed at several accounts the
    /// block touched: the MPT ships such code once, the unified trees carry it at each account's
    /// own leaves.
    pub code_bearing_accounts: usize,
    /// Raw bytecode bytes that could not be placed in the tree for want of an owner.
    pub unowned_code_bytes: u64,
    /// EIP-7864 binary-tree witness for the whole block, state and code in one proof.
    pub binary: WitnessCost,
    /// EIP-6800 Verkle witness for the whole block, state and code in one proof.
    pub verkle: WitnessCost,
    /// EIP-7864 witness for the block's account and storage state, with contract code left out.
    ///
    /// A separate proof, not a subtraction: dropping the code targets changes which nodes are
    /// shared and which stems are opened at all, so the state-only cost is not a component of
    /// the whole one.
    ///
    /// This is the comparison the MPT can be held to on equal terms. Under the MPT, code is not in
    /// the trie: the account leaf holds a hash and the bytecode travels beside the proof as a
    /// blob, so its size is a property of the contract rather than of the commitment. Both
    /// readings are reported because they answer different questions --- this one asks what
    /// the commitment scheme costs, and the whole-witness one asks what a validator receives.
    pub binary_state_only: WitnessCost,
    /// EIP-6800 witness for the block's account and storage state, with contract code left out.
    pub verkle_state_only: WitnessCost,
    /// Trie nodes a hexary path model predicts for the same miss set.
    pub mpt_model_nodes: u64,
    /// Distinct stems the miss set opened in the binary tree.
    pub binary_stems_opened: u64,
    /// Distinct stems the miss set opened in the Verkle tree.
    pub verkle_stems_opened: u64,
    /// Stems the receiver's binary-tree cache retained when the block arrived.
    pub binary_retained_stems: u64,
}

impl BlockResult {
    /// Total binary-tree witness bytes.
    pub const fn binary_total_bytes(&self) -> u64 {
        self.binary.total_bytes()
    }

    /// Total Verkle witness bytes.
    pub const fn verkle_total_bytes(&self) -> u64 {
        self.verkle.total_bytes()
    }

    /// Binary-tree witness bytes for state alone.
    pub const fn binary_state_bytes(&self) -> u64 {
        self.binary_state_only.total_bytes()
    }

    /// Verkle witness bytes for state alone.
    pub const fn verkle_state_bytes(&self) -> u64 {
        self.verkle_state_only.total_bytes()
    }
}

/// One arm's live state as the corpus is replayed through it.
///
/// `Debug` is written out rather than derived because the production cache does not implement it,
/// and a study that printed a whole warm cache would not be printing anything a reader could use.
pub struct Arm {
    spec: ArmSpec,
    cache: NetworkStateCache,
    binary: TreeState<Eip7864Keys>,
    verkle: TreeState<Eip6800Keys>,
    mpt_accounts_retained: RetainedStems,
    mpt_storage_retained: HashMap<Address, RetainedStems>,
    /// Code length by hash, so a cached account's code chunks can be placed without the code.
    code_len: HashMap<B256, usize>,
    /// Every account seen holding a given bytecode.
    ///
    /// A set, not one address. The MPT addresses code by hash, so a contract deployed at many
    /// addresses is one blob; the unified trees put code at an *account's* leaves, where the same
    /// bytecode is as many chunk ranges as it has deployments. Keeping every owner is what lets
    /// the tree arms be charged for that and the difference be reported rather than assumed
    /// away.
    code_owners: HashMap<B256, BTreeSet<Address>>,
    /// Bytecodes the cache held after the last block.
    retained_codes: HashSet<B256>,
    /// Whether this arm holds nothing between blocks.
    ///
    /// The no-cache baseline is a validator that held nothing when the block arrived, which is not
    /// the same object as a policy with a zero-length window: `LastNBlocks` retention is inclusive
    /// of its cutoff, so a zero window still keeps everything the current block touched, and the
    /// baseline every other arm is quoted against would quietly be a one-block cache.
    cold_every_block: bool,
}

/// One tree's retained frontier under one embedding.
struct TreeState<K> {
    keys: K,
    retained: RetainedStems,
}

impl std::fmt::Debug for Arm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arm")
            .field("name", &self.spec.name)
            .field("binary_retained_stems", &self.binary.retained.len())
            .field("verkle_retained_stems", &self.verkle.retained.len())
            .field("known_code_hashes", &self.code_len.len())
            .finish_non_exhaustive()
    }
}

impl Arm {
    /// Builds an arm from its policy specification.
    pub fn new(spec: ArmSpec, layout: HeaderLayout) -> Self {
        let cold_every_block = spec.account_window.is_none();
        let cache = Self::fresh_cache(&spec);
        Self {
            spec,
            cache,
            binary: TreeState { keys: Eip7864Keys::new(layout), retained: RetainedStems::new() },
            verkle: TreeState { keys: Eip6800Keys, retained: RetainedStems::new() },
            mpt_accounts_retained: RetainedStems::new(),
            mpt_storage_retained: HashMap::new(),
            code_len: HashMap::new(),
            code_owners: HashMap::new(),
            retained_codes: HashSet::new(),
            cold_every_block,
        }
    }

    /// A cold cache under this arm's windows.
    ///
    /// The no-cache arm still gets a cache object, so that it answers `compute_miss` through the
    /// same production code every other arm uses and a bug cannot hide in the baseline alone. Its
    /// windows are never reached: it is rebuilt before they could apply.
    fn fresh_cache(spec: &ArmSpec) -> NetworkStateCache {
        NetworkStateCache::new(
            Box::new(LastNBlocksPolicy::new(spec.account_window.unwrap_or(1))),
            Box::new(LastNBlocksPolicy::new(spec.storage_window.unwrap_or(1))),
        )
    }

    /// The arm's name.
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    /// The miss set this arm's cache produces for `accessed`.
    pub fn miss(&self, accessed: &BlockAccessedState) -> MissResult {
        self.cache.compute_miss(accessed)
    }

    /// How many distinct stems the receiver retains in the binary tree.
    pub fn retained_stems(&self) -> usize {
        self.binary.retained.len()
    }

    /// Applies a block to the arm's cache and brings its retained tree sets along with it.
    pub fn advance(&mut self, block_number: u64, accessed: &BlockAccessedState) {
        for (hash, code) in &accessed.codes {
            self.code_len.insert(*hash, code.len());
        }
        for (address, account) in &accessed.accounts {
            if let Some(hash) = account.code_hash {
                self.code_owners.entry(hash).or_default().insert(*address);
            }
        }
        if self.cold_every_block {
            // Rebuilt rather than cleared, so "cold" is the object a first-ever run would have
            // rather than one carrying whatever the last block left behind.
            self.cache = Self::fresh_cache(&self.spec);
            self.binary.retained = RetainedStems::new();
            self.verkle.retained = RetainedStems::new();
            self.mpt_accounts_retained = RetainedStems::new();
            self.mpt_storage_retained.clear();
            self.retained_codes.clear();
            return
        }
        self.cache.on_block_executed(block_number, accessed);
        self.reconcile_codes();
        let Some(delta) = self.cache.last_block_membership_delta() else { return };

        for address in &delta.accounts_added {
            self.retain_account(*address);
        }
        for address in &delta.accounts_removed {
            self.release_account(*address);
        }
        for (address, slot) in &delta.storage_added {
            self.binary
                .retained
                .add(self.binary.keys.storage_slot(*address, (*slot).into()).stem_id());
            self.verkle
                .retained
                .add(self.verkle.keys.storage_slot(*address, (*slot).into()).stem_id());
            self.mpt_storage_retained
                .entry(*address)
                .or_default()
                .add(mpt_stem(mpt_storage_key(*slot)));
        }
        for (address, slot) in &delta.storage_removed {
            self.binary
                .retained
                .remove(self.binary.keys.storage_slot(*address, (*slot).into()).stem_id());
            self.verkle
                .retained
                .remove(self.verkle.keys.storage_slot(*address, (*slot).into()).stem_id());
            if let Some(retained) = self.mpt_storage_retained.get_mut(address) {
                retained.remove(mpt_stem(mpt_storage_key(*slot)));
                if retained.is_empty() {
                    self.mpt_storage_retained.remove(address);
                }
            }
        }
    }

    /// Brings the retained stem sets in line with the bytecodes the cache now holds.
    ///
    /// A diff over the cache's own code map rather than an incremental hook, because the production
    /// membership delta carries accounts and storage only. Under the unified trees a cached
    /// bytecode is cached *state* — its chunks are leaves — so leaving it out would credit the
    /// tree arms with less cache than they have. The cache is keyed by hash and the trees are
    /// keyed by account, so a cached bytecode retains stems at every account known to hold it.
    fn reconcile_codes(&mut self) {
        let now: HashSet<B256> = self.cache.codes().keys().copied().collect();
        let entering: Vec<B256> = now.difference(&self.retained_codes).copied().collect();
        let leaving: Vec<B256> = self.retained_codes.difference(&now).copied().collect();
        for hash in entering {
            for (binary, verkle) in self.code_stems(&hash) {
                self.binary.retained.add(binary);
                self.verkle.retained.add(verkle);
            }
        }
        for hash in leaving {
            for (binary, verkle) in self.code_stems(&hash) {
                self.binary.retained.remove(binary);
                self.verkle.retained.remove(verkle);
            }
        }
        self.retained_codes = now;
    }

    /// The stems outside an account's header that a bytecode occupies, under both embeddings.
    ///
    /// Header-resident chunks are deliberately absent: they sit in the account's header stem, whose
    /// retention follows the account, and counting them here would double it.
    fn code_stems(&self, code_hash: &B256) -> Vec<(StemId, StemId)> {
        let (Some(owners), Some(len)) =
            (self.code_owners.get(code_hash), self.code_len.get(code_hash))
        else {
            return Vec::new()
        };
        let chunks = Eip7864Keys::chunk_count(*len);
        let mut stems = Vec::new();
        for address in owners {
            let mut binary_seen = BTreeSet::new();
            for chunk in self.binary.keys.code_chunks_in_header()..chunks {
                binary_seen.insert(self.binary.keys.code_chunk(*address, chunk).stem_id());
            }
            let mut verkle_seen = BTreeSet::new();
            for chunk in self.verkle.keys.code_chunks_in_header()..chunks {
                verkle_seen.insert(self.verkle.keys.code_chunk(*address, chunk).stem_id());
            }
            let mut binary_iter = binary_seen.into_iter();
            let mut verkle_iter = verkle_seen.into_iter();
            loop {
                match (binary_iter.next(), verkle_iter.next()) {
                    (Some(b), Some(v)) => stems.push((b, v)),
                    (Some(b), None) => stems.push((b, b)),
                    (None, Some(v)) => stems.push((v, v)),
                    (None, None) => break,
                }
            }
        }
        stems
    }

    /// An account entering the cache brings its header stem — basic data and code hash both.
    fn retain_account(&mut self, address: Address) {
        for _ in 0..2 {
            self.binary.retained.add(self.binary.keys.header_stem(address));
            self.verkle.retained.add(self.verkle.keys.header_stem(address));
        }
        self.mpt_accounts_retained.add(mpt_stem(mpt_account_key(address)));
    }

    fn release_account(&mut self, address: Address) {
        for _ in 0..2 {
            self.binary.retained.remove(self.binary.keys.header_stem(address));
            self.verkle.retained.remove(self.verkle.keys.header_stem(address));
        }
        self.mpt_accounts_retained.remove(mpt_stem(mpt_account_key(address)));
    }
}

/// Prices one block under one arm without advancing it.
pub fn price_block(
    arm: &Arm,
    accessed: &BlockAccessedState,
    params: &StudyParams,
    populations: &Populations,
    block_number: u64,
) -> BlockResult {
    let miss = arm.miss(accessed);

    let binary_plan = plan_targets(&arm.binary.keys, arm, accessed, &miss, params);
    let verkle_plan = plan_targets(&arm.verkle.keys, arm, accessed, &miss, params);

    let binary_wire = |region| arm.binary.keys.stem_wire_bytes(region);
    let verkle_wire = |region| arm.verkle.keys.stem_wire_bytes(region);
    let binary = binary_witness(
        &binary_plan.targets,
        &arm.binary.retained,
        &populations.binary,
        &binary_wire,
        MAX_BINARY_DEPTH,
    );
    let verkle = verkle_witness(
        &verkle_plan.targets,
        &arm.verkle.retained,
        &populations.verkle,
        &verkle_wire,
        VERKLE_IPA_PROOF_BYTES,
    );
    let binary_state_only = binary_witness(
        &binary_plan.state_targets,
        &arm.binary.retained,
        &populations.binary,
        &binary_wire,
        MAX_BINARY_DEPTH,
    );
    let verkle_state_only = verkle_witness(
        &verkle_plan.state_targets,
        &arm.verkle.retained,
        &populations.verkle,
        &verkle_wire,
        VERKLE_IPA_PROOF_BYTES,
    );

    let account_keys: BTreeSet<[u8; 32]> =
        miss.missed_accounts.iter().map(|address| mpt_account_key(*address)).collect();
    let mut mpt_nodes =
        hexary_path_nodes(&account_keys, &arm.mpt_accounts_retained, &populations.mpt_accounts);
    let mut by_account: BTreeMap<Address, BTreeSet<[u8; 32]>> = BTreeMap::new();
    for (address, slot) in &miss.missed_storage {
        by_account.entry(*address).or_default().insert(mpt_storage_key(*slot));
    }
    let empty_retained = RetainedStems::new();
    for (address, keys) in &by_account {
        let retained = arm.mpt_storage_retained.get(address).unwrap_or(&empty_retained);
        mpt_nodes += hexary_path_nodes(keys, retained, &populations.mpt_storage);
    }

    BlockResult {
        block_number,
        arm: arm.spec.name.clone(),
        accessed_accounts: accessed.accounts.len(),
        accessed_storage: accessed.storage.len(),
        accessed_codes: accessed.codes.len(),
        missed_accounts: miss.missed_accounts.len(),
        missed_storage: miss.missed_storage.len(),
        missed_codes: miss.missed_codes.len(),
        missed_code_bytes: binary_plan.missed_code_bytes,
        code_bearing_accounts: binary_plan.code_bearing_accounts,
        unowned_code_bytes: binary_plan.unowned_code_bytes,
        binary,
        verkle,
        binary_state_only,
        verkle_state_only,
        mpt_model_nodes: mpt_nodes,
        binary_stems_opened: binary_plan.targets.len() as u64,
        verkle_stems_opened: verkle_plan.targets.len() as u64,
        binary_retained_stems: arm.binary.retained.len() as u64,
    }
}

/// The whole of one block's witness demand under one embedding.
struct TargetPlan {
    targets: BTreeMap<StemId, StemTargets>,
    /// The same, with contract code left out.
    state_targets: BTreeMap<StemId, StemTargets>,
    missed_code_bytes: u64,
    unowned_code_bytes: u64,
    code_bearing_accounts: usize,
}

/// Builds the single target map a block's witness is computed from.
///
/// State and code go in one map on purpose. An account's basic data and its first code chunks share
/// its header stem, so pricing them as two proofs and adding the totals would charge that stem's
/// outer path and identifier twice, and would let the two halves disagree about which of its
/// suffixes are occupied.
fn plan_targets<K: TreeEmbedding>(
    keys: &K,
    arm: &Arm,
    accessed: &BlockAccessedState,
    miss: &MissResult,
    params: &StudyParams,
) -> TargetPlan {
    let mut targets: BTreeMap<StemId, StemTargets> = BTreeMap::new();
    let mut missed_code_bytes = 0u64;
    let mut unowned_code_bytes = 0u64;

    for address in &miss.missed_accounts {
        insert_target(&mut targets, keys.basic_data(*address));
        // An account with no code has no code-hash leaf to open: both proposals pack `code_size`
        // into the basic-data leaf, and `AccountAccess` normalises the empty code hash to `None`,
        // so a plain EOA costs one leaf here rather than two.
        if accessed.accounts.get(address).is_some_and(|account| account.code_hash.is_some()) {
            insert_target(&mut targets, keys.code_hash(*address));
        }
    }
    for (address, slot) in &miss.missed_storage {
        insert_target(&mut targets, keys.storage_slot(*address, (*slot).into()));
    }

    // The state-only view is taken here, before code is added: it is the same account and storage
    // demand priced as its own proof.
    let mut state_targets = targets.clone();

    // Code, at every account the block touched that holds a missed bytecode. The cache is keyed by
    // hash, so one entry covers every deployment; the trees are keyed by account, so each
    // deployment is its own chunk range and its own paths.
    let missed: HashSet<B256> = miss.missed_codes.iter().copied().collect();
    let mut code_bearing_accounts = 0usize;
    let mut owners: BTreeMap<B256, BTreeSet<Address>> = BTreeMap::new();
    for (address, account) in &accessed.accounts {
        if account.code_hash.is_some_and(|hash| missed.contains(&hash)) {
            owners.entry(account.code_hash.expect("checked")).or_default().insert(*address);
        }
    }
    for hash in &miss.missed_codes {
        let Some(code) = accessed.codes.get(hash) else { continue };
        missed_code_bytes += code.len() as u64;
        let Some(addresses) = owners.get(hash) else {
            // The corpus names no account holding this bytecode in this block, and code cannot be
            // placed in a tree without one. Reported rather than absorbed: unpriced bytes would
            // surface as a smaller witness, the one direction an error here must never be free to
            // take.
            unowned_code_bytes += code.len() as u64;
            continue
        };
        let chunks = K::chunk_count(code.len());
        for address in addresses {
            code_bearing_accounts += 1;
            for chunk in covered_chunks(chunks, params, hash) {
                let key = keys.code_chunk(*address, chunk);
                let entry = targets.entry(key.stem_id()).or_default();
                entry.targets.insert(key.suffix);
                entry.occupied.insert(key.suffix);
            }
            // Every chunk of the contract exists whether or not this call runs it, so the unrun
            // ones still occupy their suffixes and still cost sibling hashes.
            for chunk in 0..chunks {
                let key = keys.code_chunk(*address, chunk);
                targets.entry(key.stem_id()).or_default().occupied.insert(key.suffix);
            }
        }
    }

    annotate_occupancy(keys, arm, accessed, &mut targets, params);
    annotate_occupancy(keys, arm, accessed, &mut state_targets, params);

    TargetPlan {
        targets,
        state_targets,
        missed_code_bytes,
        unowned_code_bytes,
        code_bearing_accounts,
    }
}

/// Fills in what is known and what is modelled about each opened stem's occupancy.
fn annotate_occupancy<K: TreeEmbedding>(
    keys: &K,
    arm: &Arm,
    accessed: &BlockAccessedState,
    targets: &mut BTreeMap<StemId, StemTargets>,
    params: &StudyParams,
) {
    // Header stems: basic data always exists, the code hash exists when the account has code, and
    // the header-resident code chunks follow from the code length. Modelling these as absent would
    // understate the tree by making sibling subtrees look empty.
    let touched: HashSet<Address> = accessed.accounts.keys().copied().collect();
    for address in &touched {
        let stem = keys.header_stem(*address);
        let Some(entry) = targets.get_mut(&stem) else { continue };
        entry.occupied.insert(0);
        let account = accessed.accounts.get(address);
        if account.is_some_and(|a| a.code_hash.is_some()) {
            entry.occupied.insert(1);
        }
        if let Some(len) = account.and_then(|a| a.code_hash).and_then(|h| arm.code_len.get(&h)) {
            let chunks = K::chunk_count(*len).min(keys.code_chunks_in_header());
            for chunk in 0..chunks {
                let suffix = keys.code_offset() + chunk;
                if suffix < 256 {
                    entry.occupied.insert(suffix as u8);
                }
            }
        }
    }

    // Suffixes the receiver already holds inside a targeted stem: a cached slot of an account whose
    // basic data is cold still reveals part of that account's stem subtree.
    for (address, slot) in accessed.storage.keys() {
        if arm.cache.contains_storage(address, slot) {
            let key = keys.storage_slot(*address, (*slot).into());
            if let Some(entry) = targets.get_mut(&key.stem_id()) {
                entry.held.insert(key.suffix);
                entry.occupied.insert(key.suffix);
            }
        }
    }
    for (address, account) in &accessed.accounts {
        if arm.cache.contains_account(address) {
            let stem = keys.header_stem(*address);
            if let Some(entry) = targets.get_mut(&stem) {
                entry.held.insert(0);
                if account.code_hash.is_some() {
                    entry.held.insert(1);
                }
            }
        }
    }

    let occupancy = params.effective_occupancy();
    for (stem, entry) in targets.iter_mut() {
        entry.modelled_extra = if stem.region == TreeRegion::Header {
            occupancy.in_header
        } else {
            occupancy.outside_header
        };
    }
}

/// The background populations one run is fixed to.
#[derive(Debug)]
pub struct Populations {
    /// The binary tree's three regions.
    pub binary: RegionPopulations,
    /// The Verkle tree's single region.
    pub verkle: RegionPopulations,
    /// Keys in the MPT's account trie.
    pub mpt_accounts: BackgroundPopulation,
    /// Keys in a modelled per-account storage trie.
    pub mpt_storage: BackgroundPopulation,
}

impl Populations {
    /// Builds the populations named by `params`.
    pub fn new(params: &StudyParams) -> Self {
        Self {
            binary: RegionPopulations::new(
                BackgroundPopulation::new(params.account_population, "binary-header"),
                BackgroundPopulation::new(params.code_stem_population, "binary-code"),
                BackgroundPopulation::new(params.storage_stem_population(), "binary-storage"),
            ),
            verkle: RegionPopulations::unified(BackgroundPopulation::new(
                params.total_stem_population,
                "verkle-unified",
            )),
            mpt_accounts: BackgroundPopulation::new(params.account_population, "mpt-accounts"),
            mpt_storage: BackgroundPopulation::new(
                params.mpt_storage_trie_population,
                "mpt-storage",
            ),
        }
    }
}

fn insert_target(targets: &mut BTreeMap<StemId, StemTargets>, key: TreeKey) {
    let entry = targets.entry(key.stem_id()).or_default();
    entry.targets.insert(key.suffix);
    entry.occupied.insert(key.suffix);
}

/// A hashed MPT key, wrapped so it can share the retained-stem machinery.
const fn mpt_stem(key: [u8; 32]) -> StemId {
    StemId { region: TreeRegion::Unified, stem: key }
}

/// The chunks a call runs, measured where a measurement exists and modelled where it does not.
///
/// A measured entry is used verbatim. A bytecode the measurement never saw *entered* is not zero
/// coverage --- it can be read by `EXTCODECOPY` or `EXTCODEHASH` without executing --- and
/// returning nothing for it would drop code the witness has to carry, so the modelled fraction
/// stands in.
///
/// Modelled coverage is spread pseudo-randomly rather than taken as a leading run. Real execution
/// runs basic blocks, which are contiguous, so a contiguous model would understate how many stems a
/// partially-run contract opens and a scattered one overstates it. The scattered choice is the
/// pessimistic one for the tree arms, which is the right direction for a value they are credited
/// with rather than measured on.
fn covered_chunks(chunks: u32, params: &StudyParams, code_hash: &B256) -> Vec<u32> {
    if let Some(ran) =
        params.measured_coverage.as_ref().and_then(|measured| measured.chunks_of(code_hash))
    {
        return ran.iter().copied().filter(|chunk| *chunk < chunks).collect()
    }
    let coverage = params.code_coverage;
    if coverage >= 1.0 {
        return (0..chunks).collect()
    }
    let wanted = ((f64::from(chunks) * coverage).ceil() as u32).min(chunks);
    let mut picked = Vec::with_capacity(wanted as usize);
    for chunk in 0..chunks {
        let mut hasher = blake3::Hasher::new();
        hasher.update(code_hash.as_slice());
        hasher.update(&chunk.to_le_bytes());
        let draw =
            u32::from_le_bytes(hasher.finalize().as_bytes()[..4].try_into().expect("4 bytes"));
        if u64::from(draw) * u64::from(chunks) < u64::from(u32::MAX) * u64::from(wanted) {
            picked.push(chunk);
        }
    }
    picked
}

/// How the corpus's own storage accesses cluster into stems.
///
/// The unified tree's stem count is the one population figure the node cannot supply. Its database
/// holds hashed state only — `PlainStorageState` is empty — and both proposals group storage by
/// *plain* slot number, so the grouping cannot be recovered from what the node stores. What the
/// corpus does carry is plain `(address, slot)` pairs, and the ratio of distinct slots to distinct
/// stems in them bounds the grouping from the clustered side: blocks touch the sequentially
/// laid-out parts of contracts more often than the hash-scattered parts, so this ratio is an upper
/// bound on how much the whole state clusters, and therefore a lower bound on the stem count.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StemCensus {
    /// Distinct `(address, slot)` pairs seen.
    pub distinct_slots: usize,
    /// Distinct stems those slots occupy, header stems included once each.
    pub distinct_storage_stems: usize,
    /// Distinct slots that fell inside their account's header stem.
    pub header_resident_slots: usize,
    /// Distinct header stems those slots fell into.
    pub header_stems_with_storage: usize,
    /// Distinct accounts seen.
    pub distinct_accounts: usize,
    /// Distinct bytecodes seen.
    pub distinct_codes: usize,
    /// Out-of-header code stems those bytecodes occupy.
    pub code_stems: usize,
    /// Accounts seen holding a bytecode, counted once per (bytecode, account) pair.
    pub code_deployments: usize,
}

impl StemCensus {
    /// Slots per storage stem, as observed.
    pub fn slots_per_stem(&self) -> f64 {
        if self.distinct_storage_stems == 0 {
            return 0.0
        }
        self.distinct_slots as f64 / self.distinct_storage_stems as f64
    }

    /// Accounts per distinct bytecode, as observed.
    pub fn deployments_per_code(&self) -> f64 {
        if self.distinct_codes == 0 {
            return 0.0
        }
        self.code_deployments as f64 / self.distinct_codes as f64
    }

    /// Extrapolates a whole-state stem count from measured account and slot totals.
    pub fn extrapolate(&self, accounts: u64, slots: u64) -> u64 {
        let per_stem = self.slots_per_stem().max(1.0);
        let header_share = if self.distinct_slots == 0 {
            0.0
        } else {
            self.header_resident_slots as f64 / self.distinct_slots as f64
        };
        let out_of_header = (slots as f64) * (1.0 - header_share);
        accounts + (out_of_header / per_stem) as u64
    }
}

/// Accumulates a [`StemCensus`] over a corpus.
#[derive(Debug)]
pub struct CensusBuilder {
    keys: Eip7864Keys,
    slots: HashSet<(Address, B256)>,
    storage_stems: HashSet<[u8; 32]>,
    header_stems_with_storage: HashSet<[u8; 32]>,
    header_resident: usize,
    accounts: HashSet<Address>,
    codes: HashSet<B256>,
    code_stems: HashSet<[u8; 32]>,
    deployments: HashSet<(B256, Address)>,
}

impl CensusBuilder {
    /// A census under `layout`.
    pub fn new(layout: HeaderLayout) -> Self {
        Self {
            keys: Eip7864Keys::new(layout),
            slots: HashSet::new(),
            storage_stems: HashSet::new(),
            header_stems_with_storage: HashSet::new(),
            header_resident: 0,
            accounts: HashSet::new(),
            codes: HashSet::new(),
            code_stems: HashSet::new(),
            deployments: HashSet::new(),
        }
    }

    /// Folds one block's accesses in.
    pub fn observe(&mut self, accessed: &BlockAccessedState) {
        for address in accessed.accounts.keys() {
            self.accounts.insert(*address);
        }
        for (address, slot) in accessed.storage.keys() {
            if !self.slots.insert((*address, *slot)) {
                continue
            }
            let key = self.keys.storage_slot(*address, (*slot).into());
            if key.region == TreeRegion::Header {
                // One account's leading slots are one header stem, so this counts the stem and not
                // the slots that landed in it.
                self.header_resident += 1;
                self.header_stems_with_storage.insert(key.stem);
            } else {
                self.storage_stems.insert(key.stem);
            }
        }
        for (address, account) in &accessed.accounts {
            if let Some(hash) = account.code_hash {
                self.deployments.insert((hash, *address));
            }
        }
        for (hash, code) in &accessed.codes {
            if !self.codes.insert(*hash) {
                continue
            }
            let Some(address) = accessed
                .accounts
                .iter()
                .filter(|(_, account)| account.code_hash == Some(*hash))
                .map(|(address, _)| *address)
                .min()
            else {
                continue
            };
            let chunks = Eip7864Keys::chunk_count(code.len());
            for chunk in self.keys.code_chunks_in_header()..chunks {
                self.code_stems.insert(self.keys.code_chunk(address, chunk).stem);
            }
        }
    }

    /// The census so far.
    pub fn finish(&self) -> StemCensus {
        StemCensus {
            distinct_slots: self.slots.len(),
            distinct_storage_stems: self.storage_stems.len() + self.header_stems_with_storage.len(),
            header_resident_slots: self.header_resident,
            header_stems_with_storage: self.header_stems_with_storage.len(),
            distinct_accounts: self.accounts.len(),
            distinct_codes: self.codes.len(),
            code_stems: self.code_stems.len(),
            code_deployments: self.deployments.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    use partial_stateless::policy::AccountData;

    fn contract(code_hash: B256) -> AccountData {
        AccountData { nonce: 0, balance: U256::ZERO, code_hash: Some(code_hash) }
    }

    #[test]
    fn the_no_cache_arm_holds_nothing_between_blocks() {
        let mut arm = Arm::new(ArmSpec::weak(), HeaderLayout::TABLE);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(Address::repeat_byte(7), contract(B256::repeat_byte(3)));
        accessed.storage.insert((Address::repeat_byte(7), B256::repeat_byte(1)), U256::from(9));

        arm.advance(100, &accessed);
        assert_eq!(arm.retained_stems(), 0, "a zero-length window still retains; a reset must not");

        let miss = arm.miss(&accessed);
        assert_eq!(miss.missed_accounts.len(), 1);
        assert_eq!(miss.missed_storage.len(), 1);
    }

    #[test]
    fn a_cached_arm_keeps_what_it_saw() {
        let mut arm = Arm::new(ArmSpec::windows(90, 60), HeaderLayout::TABLE);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(Address::repeat_byte(7), contract(B256::repeat_byte(3)));

        arm.advance(100, &accessed);
        assert!(arm.retained_stems() > 0);
        assert!(arm.miss(&accessed).missed_accounts.is_empty());
    }

    #[test]
    fn the_state_only_proof_leaves_code_out_and_is_priced_on_its_own() {
        let code_hash = B256::repeat_byte(0x5a);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(Address::repeat_byte(0x11), contract(code_hash));
        accessed.storage.insert((Address::repeat_byte(0x11), B256::repeat_byte(2)), U256::from(7));
        accessed.codes.insert(code_hash, vec![0u8; 31 * 600].into());

        let arm = Arm::new(ArmSpec::weak(), HeaderLayout::TABLE);
        let params = StudyParams::default();
        let miss = arm.miss(&accessed);
        let plan = plan_targets(&arm.binary.keys, &arm, &accessed, &miss, &params);

        assert!(
            plan.state_targets.len() < plan.targets.len(),
            "600 chunks must open stems the state-only view does not"
        );
        let header = arm.binary.keys.header_stem(Address::repeat_byte(0x11));
        let state_header = plan.state_targets.get(&header).expect("the account is a target");
        assert!(
            state_header.targets.iter().all(|suffix| *suffix < 4),
            "the state-only view opens basic data and code hash, not code chunks"
        );
        assert!(
            plan.targets.get(&header).expect("merged").targets.len() > state_header.targets.len(),
            "the merged view opens the header-resident code chunks as well"
        );
    }

    #[test]
    fn one_bytecode_at_two_accounts_opens_two_chunk_ranges() {
        // The MPT ships such code once, keyed by hash. The unified trees keep it at each account's
        // own leaves, and a study that priced one copy would credit them with a saving the tree
        // does not offer.
        let code_hash = B256::repeat_byte(0x9f);
        let mut accessed = BlockAccessedState::default();
        accessed.accounts.insert(Address::repeat_byte(0x10), contract(code_hash));
        accessed.accounts.insert(Address::repeat_byte(0x20), contract(code_hash));
        accessed.codes.insert(code_hash, vec![0u8; 31 * 40].into());

        let arm = Arm::new(ArmSpec::weak(), HeaderLayout::TABLE);
        let params = StudyParams::default();
        let miss = arm.miss(&accessed);
        assert_eq!(miss.missed_codes.len(), 1, "the cache dedups code by hash");

        let plan = plan_targets(&arm.binary.keys, &arm, &accessed, &miss, &params);
        assert_eq!(plan.code_bearing_accounts, 2, "each deployment is its own chunk range");
    }

    #[test]
    fn occupancy_follows_the_stem_count_it_is_quoted_with() {
        // The same measured state over fewer stems fills each of them more; sweeping the two
        // independently would price a state that does not exist.
        let sparse = StudyParams { total_stem_population: 2_000_000_000, ..Default::default() };
        let dense = StudyParams { total_stem_population: 500_000_000, ..Default::default() };
        assert_eq!(sparse.effective_occupancy().outside_header, 0);
        assert!(
            dense.effective_occupancy().outside_header > 10,
            "1.6e9 slots over 85M stems is about nineteen each, not one"
        );
        let pinned =
            StudyParams { stem_occupancy: Some(StemOccupancy::full()), ..Default::default() };
        assert_eq!(pinned.effective_occupancy(), StemOccupancy::full());
    }

    #[test]
    fn the_census_counts_header_stems_not_header_slots() {
        let mut census = CensusBuilder::new(HeaderLayout::TABLE);
        let mut accessed = BlockAccessedState::default();
        let address = Address::repeat_byte(0x31);
        // Three of one account's leading slots: three slots, but one header stem.
        for slot in 0..3u8 {
            accessed.storage.insert((address, B256::left_padding_from(&[slot])), U256::ZERO);
        }
        census.observe(&accessed);
        let out = census.finish();
        assert_eq!(out.distinct_slots, 3);
        assert_eq!(out.header_stems_with_storage, 1);
        assert_eq!(out.distinct_storage_stems, 1, "one account's leading slots are one stem");
    }
}
