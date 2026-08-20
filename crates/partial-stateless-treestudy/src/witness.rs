//! What a cache-complement witness costs, in the one accounting all backends share.
//!
//! Every scheme in this study answers the same question: the receiver already holds a bounded set
//! of state keys and the tree nodes on the paths to them; a block touches some keys it does not
//! hold; what must be sent so the receiver can authenticate those keys against the parent root and
//! recompute the next one?
//!
//! The answer decomposes the same way in all three schemes — path nodes, leaf identification,
//! values — and differs only in what a path node *is* and how many of them a level costs. Keeping
//! that decomposition in one place is what makes the arms comparable: a byte counted for one scheme
//! is counted under the same rule for the others, and the parts that genuinely differ are the parts
//! the study is measuring.
//!
//! Three exclusions apply identically everywhere, and they are the whole of the cache's effect:
//!
//! - **Held.** A node whose parent lies on a path to a retained key is already in the receiver's
//!   trie, complete with its children's hashes. This is the frontier graft the production witness
//!   v3 performs, stated so it applies to any tree.
//! - **Derived.** A node the receiver computes anyway, because it lies on another target's path in
//!   this same witness — or because this witness already carries it. Both are one rule: a node is
//!   paid for once.
//! - **Empty.** A subtree with no state under it hashes to a known constant, so it costs a
//!   structural bit rather than a hash.
//!
//! A witness covers a whole block, so it is computed from a single target set. Pricing state and
//! code separately and adding the totals would charge an account's header stem twice — its basic
//! data and its first code chunks share it — and would let the two halves disagree about which of
//! that stem's suffixes are occupied.

use crate::{
    keys::{Prefix, StemId, TreeRegion},
    population::{BackgroundPopulation, Descent},
};
use std::collections::{BTreeMap, BTreeSet};

/// Bytes of one hash on the wire.
pub const HASH_BYTES: u64 = 32;
/// Bytes of one leaf value.
pub const VALUE_BYTES: u64 = 32;
/// Levels in the binary subtree that commits a stem's 256 values.
const SUBTREE_DEPTH: u32 = 8;

/// A set of stems the receiver's trie retains paths to.
///
/// Refcounted because several state keys map to one stem — an account's basic data, its code hash,
/// and its header-resident storage and code all share it — so a stem stops being retained only when
/// the last key under it leaves the cache. Ordered per region so that "is anything retained below
/// this prefix?" is a range lookup rather than a scan; that question is asked once per level per
/// target, and it is the only thing the receiver's frontier is a function of.
#[derive(Debug, Default, Clone)]
pub struct RetainedStems {
    regions: BTreeMap<TreeRegion, BTreeSet<[u8; 32]>>,
    refs: BTreeMap<StemId, u32>,
}

impl RetainedStems {
    /// An empty retained set — the no-cache arm.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one more key under `stem`.
    pub fn add(&mut self, stem: StemId) {
        let count = self.refs.entry(stem).or_insert(0);
        *count += 1;
        if *count == 1 {
            self.regions.entry(stem.region).or_default().insert(stem.stem);
        }
    }

    /// Records one fewer key under `stem`.
    pub fn remove(&mut self, stem: StemId) {
        if let Some(count) = self.refs.get_mut(&stem) {
            *count -= 1;
            if *count == 0 {
                self.refs.remove(&stem);
                if let Some(set) = self.regions.get_mut(&stem.region) {
                    set.remove(&stem.stem);
                }
            }
        }
    }

    /// How many distinct stems are retained.
    pub fn len(&self) -> usize {
        self.regions.values().map(BTreeSet::len).sum()
    }

    /// Whether nothing is retained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this exact stem is retained.
    pub fn contains(&self, stem: &StemId) -> bool {
        self.regions.get(&stem.region).is_some_and(|set| set.contains(&stem.stem))
    }

    /// Whether any retained stem lies under `prefix`, within `region`.
    pub fn any_under(&self, region: TreeRegion, prefix: Prefix) -> bool {
        self.regions.get(&region).is_some_and(|set| any_under(set, prefix))
    }

    /// Whether any retained stem other than `stem` shares `prefix`.
    fn shares_with_other(&self, region: TreeRegion, prefix: Prefix, stem: &[u8; 32]) -> bool {
        let Some(set) = self.regions.get(&region) else { return false };
        let (lo, hi) = prefix_range(prefix);
        set.range(lo..=hi).any(|other| other != stem)
    }
}

/// Whether any key in `set` lies under `prefix`.
fn any_under(set: &BTreeSet<[u8; 32]>, prefix: Prefix) -> bool {
    let (lo, hi) = prefix_range(prefix);
    set.range(lo..=hi).next().is_some()
}

/// The inclusive key range a prefix covers.
pub fn prefix_range(prefix: Prefix) -> ([u8; 32], [u8; 32]) {
    let lo = prefix.bytes();
    let mut hi = lo;
    let len = prefix.len();
    let whole = (len / 8) as usize;
    let spare = len % 8;
    if spare == 0 {
        for byte in hi.iter_mut().skip(whole) {
            *byte = 0xff;
        }
    } else {
        hi[whole] |= 0xffu8 >> spare;
        for byte in hi.iter_mut().skip(whole + 1) {
            *byte = 0xff;
        }
    }
    (lo, hi)
}

/// What one witness costs, split so a result can say where the bytes went.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct WitnessCost {
    /// Hashes or commitments carried for tree structure.
    pub path_nodes: u64,
    /// Bytes those nodes occupy.
    pub path_bytes: u64,
    /// Distinct leaf groups identified on the wire.
    pub stems: u64,
    /// Bytes spent identifying them.
    pub stem_bytes: u64,
    /// Leaf values carried.
    pub values: u64,
    /// Bytes those values occupy.
    pub value_bytes: u64,
    /// Bytes of fixed proof material that does not scale with the target count.
    pub fixed_bytes: u64,
    /// Bytes of structural bitmaps: which siblings are empty, which stems are present.
    pub structure_bytes: u64,
    /// Nodes the receiver's retained frontier already covered.
    pub nodes_held_by_cache: u64,
    /// Nodes skipped because this witness already carries or computes them.
    pub nodes_derived: u64,
    /// Nodes skipped because their subtree is empty.
    pub nodes_empty: u64,
}

impl WitnessCost {
    /// Total bytes on the wire.
    pub const fn total_bytes(&self) -> u64 {
        self.path_bytes +
            self.stem_bytes +
            self.value_bytes +
            self.fixed_bytes +
            self.structure_bytes
    }

    /// Folds another cost into this one.
    pub const fn add(&mut self, other: Self) {
        self.path_nodes += other.path_nodes;
        self.path_bytes += other.path_bytes;
        self.stems += other.stems;
        self.stem_bytes += other.stem_bytes;
        self.values += other.values;
        self.value_bytes += other.value_bytes;
        self.fixed_bytes += other.fixed_bytes;
        self.structure_bytes += other.structure_bytes;
        self.nodes_held_by_cache += other.nodes_held_by_cache;
        self.nodes_derived += other.nodes_derived;
        self.nodes_empty += other.nodes_empty;
    }
}

/// One stem a witness must open, and which of its values it must carry.
#[derive(Debug, Clone, Default)]
pub struct StemTargets {
    /// Suffixes whose values the witness carries.
    pub targets: BTreeSet<u8>,
    /// Suffixes the receiver already holds from its cache.
    pub held: BTreeSet<u8>,
    /// Suffixes known to be occupied by state, whether targeted or not.
    pub occupied: BTreeSet<u8>,
    /// Occupied suffixes the corpus cannot name, modelled rather than observed.
    ///
    /// A stem holds 256 values and the corpus sees only the ones this run's blocks touched.
    /// Storage that exists in the same stem but went untouched still contributes sibling
    /// hashes, and treating it as absent understates the witness. See [`StemOccupancy`].
    pub modelled_extra: u32,
}

/// How much of a stem is assumed occupied beyond what the corpus names.
///
/// The corpus records the keys its blocks touched, not the state around them. Inside an opened
/// stem, an untouched-but-existing sibling still costs a hash, so the observed occupancy is a
/// lower bound and the run has to say which bound it is quoting. Zero reproduces the optimistic
/// bound; the measured mean is the central case; 255 is the pessimistic bound where every stem is
/// full.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StemOccupancy {
    /// Extra occupied suffixes assumed in a stem outside any account header.
    pub outside_header: u32,
    /// Extra occupied suffixes assumed in an account's header stem.
    ///
    /// Separate because a header stem's occupancy is mostly *known*: basic data and code hash
    /// always exist and the code chunks follow from the code length. What is unknown is only
    /// the account's first few storage slots.
    pub in_header: u32,
}

impl StemOccupancy {
    /// The optimistic bound: nothing occupied beyond what the corpus names.
    pub const fn zero() -> Self {
        Self { outside_header: 0, in_header: 0 }
    }

    /// The pessimistic bound: every suffix of every opened stem occupied.
    pub const fn full() -> Self {
        Self { outside_header: 255, in_header: 255 }
    }
}

/// The binary-Merkle witness of EIP-7864.
///
/// Costs one hash per sibling along every target's path, which is what arity-2 trades for its short
/// branches: the sibling count per level is `k - 1 = 1`, against 15 for the MPT's hexary branching,
/// while the number of levels grows as `log2` rather than `log16`.
pub fn binary_witness(
    targets: &BTreeMap<StemId, StemTargets>,
    retained: &RetainedStems,
    populations: &RegionPopulations,
    stem_wire_bytes: &dyn Fn(TreeRegion) -> u64,
    max_depth: u32,
) -> WitnessCost {
    let mut cost = WitnessCost::default();
    let target_stems = by_region(targets);
    // A node is paid for once per witness. Two targets sharing a prefix walk the same nodes above
    // the point they diverge, and without this the shared upper path is charged to each of them —
    // which inflates whichever arm has the most targets, and the no-cache arm has the most.
    let mut emitted: BTreeSet<(TreeRegion, Prefix)> = BTreeSet::new();

    for (stem, stem_targets) in targets {
        let region = stem.region;
        let population = populations.get(region);
        let empty = BTreeSet::new();
        let peers = target_stems.get(&region).unwrap_or(&empty);
        let descent = population.descend(&stem.stem, max_depth);
        let depth = stem_node_depth(stem, &descent, peers, retained, max_depth);

        for level in 1..=depth {
            let node = Prefix::of(&stem.stem, level);
            let sibling = node.sibling();
            let parent = node.truncated(level - 1);

            if retained.any_under(region, parent) {
                cost.nodes_held_by_cache += 1;
                continue
            }
            if any_under(peers, sibling) {
                cost.nodes_derived += 1;
                continue
            }
            let sibling_populated = descent.sibling_count(level) > 0 ||
                retained.any_under(region, sibling) ||
                any_under(peers, sibling);
            if !sibling_populated {
                cost.nodes_empty += 1;
                continue
            }
            if !emitted.insert((region, sibling)) {
                cost.nodes_derived += 1;
                continue
            }
            cost.path_nodes += 1;
            cost.path_bytes += HASH_BYTES;
        }
        // One bit per level says whether that level's sibling was empty or carried.
        cost.structure_bytes += u64::from(depth).div_ceil(8);

        if !retained.contains(stem) {
            cost.stems += 1;
            cost.stem_bytes += stem_wire_bytes(region);
        }
        cost.add(stem_subtree_witness(stem, stem_targets));
    }
    cost
}

/// The 256-ary vector-commitment witness of EIP-6800.
///
/// A vector commitment opens a child without its siblings, so a level costs one commitment rather
/// than `k - 1` hashes, and 256-ary branching makes the tree about four levels deep at mainnet
/// scale. The polynomial openings then aggregate into one proof whose size does not grow with the
/// number of keys opened, which is the property the scheme exists for.
pub fn verkle_witness(
    targets: &BTreeMap<StemId, StemTargets>,
    retained: &RetainedStems,
    populations: &RegionPopulations,
    stem_wire_bytes: &dyn Fn(TreeRegion) -> u64,
    ipa_proof_bytes: u64,
) -> WitnessCost {
    /// One level of a 256-ary tree consumes this many bits of the stem.
    const LEVEL_BITS: u32 = 8;
    /// Deepest a stem can sit before its position is exhausted.
    const MAX_LEVELS: u32 = 31;

    let mut cost = WitnessCost::default();
    if targets.is_empty() {
        return cost
    }
    let target_stems = by_region(targets);
    let mut emitted: BTreeSet<(TreeRegion, Prefix)> = BTreeSet::new();

    for (stem, stem_targets) in targets {
        let region = stem.region;
        let population = populations.get(region);
        let empty = BTreeSet::new();
        let peers = target_stems.get(&region).unwrap_or(&empty);
        let descent = population.descend(&stem.stem, MAX_LEVELS * LEVEL_BITS);
        // A 256-ary node splits when two keys share its whole byte, so the branch point is the
        // binary branch depth rounded up to the next byte.
        let levels = descent.alone_at.div_ceil(LEVEL_BITS).clamp(1, MAX_LEVELS);
        let levels = deepen_for_real_stems(stem, levels, peers, retained, MAX_LEVELS);

        for level in 1..=levels {
            let node = Prefix::of(&stem.stem, level * LEVEL_BITS);
            if retained.any_under(region, node) {
                cost.nodes_held_by_cache += 1;
                continue
            }
            if !emitted.insert((region, node)) {
                cost.nodes_derived += 1;
                continue
            }
            cost.path_nodes += 1;
            cost.path_bytes += HASH_BYTES;
        }

        if !retained.contains(stem) {
            cost.stems += 1;
            cost.stem_bytes += stem_wire_bytes(region);
            // A stem's values are committed in two halves, so a stem opened at all costs the half
            // or halves its targets fall in.
            let lower = stem_targets.targets.iter().any(|s| *s < 128);
            let upper = stem_targets.targets.iter().any(|s| *s >= 128);
            let halves = u64::from(lower) + u64::from(upper);
            cost.path_nodes += halves;
            cost.path_bytes += halves * HASH_BYTES;
        }
        cost.values += stem_targets.targets.len() as u64;
        cost.value_bytes += stem_targets.targets.len() as u64 * VALUE_BYTES;
        // Per-stem presence and depth markers, as `depth_extension_present` carries them.
        cost.structure_bytes += 1;
    }
    cost.fixed_bytes += ipa_proof_bytes;
    cost
}

/// The background populations of a scheme's regions.
///
/// EIP-7864 separates headers, code, and storage by a storage-type prefix, so a header stem
/// branches among the roughly 412 million other header stems rather than among every stem in the
/// state. A single-population model would put every path several levels too deep.
#[derive(Debug)]
pub struct RegionPopulations {
    by_region: [BackgroundPopulation; 3],
}

impl RegionPopulations {
    /// Populations for headers, out-of-header code, and out-of-header storage.
    pub const fn new(
        header: BackgroundPopulation,
        code: BackgroundPopulation,
        storage: BackgroundPopulation,
    ) -> Self {
        Self { by_region: [header, code, storage] }
    }

    /// One population shared by every region, for a scheme with a single tree.
    pub fn unified(all: BackgroundPopulation) -> Self {
        Self { by_region: [all.clone(), all.clone(), all] }
    }

    /// The population governing `region`.
    pub const fn get(&self, region: TreeRegion) -> &BackgroundPopulation {
        &self.by_region[region.index()]
    }
}

fn by_region(targets: &BTreeMap<StemId, StemTargets>) -> BTreeMap<TreeRegion, BTreeSet<[u8; 32]>> {
    let mut out: BTreeMap<TreeRegion, BTreeSet<[u8; 32]>> = BTreeMap::new();
    for stem in targets.keys() {
        out.entry(stem.region).or_default().insert(stem.stem);
    }
    out
}

/// Depth at which a stem's own node can sit: one level below the deepest key that shares its path.
fn stem_node_depth(
    stem: &StemId,
    descent: &Descent,
    peers: &BTreeSet<[u8; 32]>,
    retained: &RetainedStems,
    max_depth: u32,
) -> u32 {
    let mut depth = descent.alone_at;
    // A retained or co-targeted stem sharing the path pushes the branch point deeper than the
    // background alone would, and both are exact rather than modelled.
    while depth < max_depth {
        let prefix = Prefix::of(&stem.stem, depth);
        let (lo, hi) = prefix_range(prefix);
        let shared = peers.range(lo..=hi).any(|other| other != &stem.stem) ||
            retained.shares_with_other(stem.region, prefix, &stem.stem);
        if !shared {
            break
        }
        depth += 1;
    }
    depth.max(1)
}

/// Pushes a 256-ary branch point deeper when real stems — targeted or retained — share the path.
fn deepen_for_real_stems(
    stem: &StemId,
    from: u32,
    peers: &BTreeSet<[u8; 32]>,
    retained: &RetainedStems,
    max_levels: u32,
) -> u32 {
    let mut levels = from;
    while levels < max_levels {
        let prefix = Prefix::of(&stem.stem, levels * 8);
        let (lo, hi) = prefix_range(prefix);
        let shared = peers.range(lo..=hi).any(|other| other != &stem.stem) ||
            retained.shares_with_other(stem.region, prefix, &stem.stem);
        if !shared {
            break
        }
        levels += 1;
    }
    levels
}

/// The depth-8 subtree that commits a stem's 256 values.
///
/// Small enough to walk explicitly over a 256-bit occupancy mask, which is worth doing: much of a
/// stem's occupancy is exactly known — an account's basic data and code hash are always present and
/// its code chunks follow from its code length — and only the remainder is modelled.
fn stem_subtree_witness(stem: &StemId, stem_targets: &StemTargets) -> WitnessCost {
    let mut cost = WitnessCost::default();
    if stem_targets.targets.is_empty() {
        return cost
    }

    let targets = SuffixSet::from_iter(stem_targets.targets.iter().copied());
    let held = SuffixSet::from_iter(stem_targets.held.iter().copied());
    let mut occupied = SuffixSet::from_iter(stem_targets.occupied.iter().copied());
    occupied.union_with(&targets);
    occupied.union_with(&held);
    occupied.add_modelled(stem, stem_targets.modelled_extra);

    let mut emitted: BTreeSet<(u32, u8)> = BTreeSet::new();
    for suffix in &stem_targets.targets {
        for level in 1..=SUBTREE_DEPTH {
            let parent_level = level - 1;
            // Held: a cached suffix under this node's parent means the receiver's stem subtree is
            // revealed that far, so the sibling hash is already in hand.
            if held.any_under(suffix_prefix(*suffix, parent_level), parent_level) {
                cost.nodes_held_by_cache += 1;
                continue
            }
            let sibling =
                suffix_prefix_with(*suffix, parent_level, !suffix_bit(*suffix, level - 1));
            if targets.any_under(sibling, level) {
                cost.nodes_derived += 1;
                continue
            }
            if !occupied.any_under(sibling, level) {
                cost.nodes_empty += 1;
                continue
            }
            if !emitted.insert((level, sibling)) {
                cost.nodes_derived += 1;
                continue
            }
            cost.path_nodes += 1;
            cost.path_bytes += HASH_BYTES;
        }
        cost.values += 1;
        cost.value_bytes += VALUE_BYTES;
    }
    cost.structure_bytes +=
        u64::from(SUBTREE_DEPTH).div_ceil(8) * stem_targets.targets.len() as u64;
    cost
}

/// A stem's 256 suffixes as a bitmask.
///
/// The subtree walk asks "is any occupied suffix under this prefix?" eight times per target, and
/// the occupancy of a contract's header stem runs to hundreds of code chunks. As a mask the
/// question is two shifts and an and; as a set scan it was the run's hot loop.
#[derive(Debug, Clone, Copy, Default)]
struct SuffixSet {
    words: [u64; 4],
}

impl SuffixSet {
    fn from_iter(items: impl Iterator<Item = u8>) -> Self {
        let mut set = Self::default();
        for item in items {
            set.insert(item);
        }
        set
    }

    const fn insert(&mut self, suffix: u8) {
        self.words[(suffix >> 6) as usize] |= 1u64 << (suffix & 63);
    }

    fn union_with(&mut self, other: &Self) {
        for (word, extra) in self.words.iter_mut().zip(other.words) {
            *word |= extra;
        }
    }

    /// Occupies `extra` further suffixes, chosen deterministically from the stem's identity.
    ///
    /// Pseudo-random rather than contiguous: real storage in one stem is either an array, which is
    /// contiguous, or hash-scattered, which is not, and scattering is the choice that costs the
    /// tree arms more sibling hashes. Where the corpus cannot say, the study takes the
    /// expensive reading.
    fn add_modelled(&mut self, stem: &StemId, extra: u32) {
        if extra == 0 {
            return
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[stem.region.storage_type()]);
        hasher.update(&stem.stem);
        let seed = hasher.finalize();
        let bytes = seed.as_bytes();
        for step in 0..extra.min(255) {
            self.insert(bytes[(step as usize) % 32].wrapping_add(step as u8));
        }
    }

    /// Whether any member starts with the `level`-bit prefix `prefix`.
    fn any_under(&self, prefix: u8, level: u32) -> bool {
        if level == 0 {
            return self.words.iter().any(|word| *word != 0)
        }
        let span = 1u32 << (8 - level);
        let low = u32::from(prefix) << (8 - level);
        let high = low + span - 1;
        for suffix in low..=high {
            if self.words[(suffix >> 6) as usize] >> (suffix & 63) & 1 == 1 {
                return true
            }
        }
        false
    }
}

/// Bit `index` of a suffix, most significant first.
const fn suffix_bit(suffix: u8, index: u32) -> bool {
    suffix >> (7 - index) & 1 == 1
}

/// The first `level` bits of `suffix`, in the low bits of the returned byte.
const fn suffix_prefix(suffix: u8, level: u32) -> u8 {
    if level == 0 {
        0
    } else {
        suffix >> (8 - level)
    }
}

/// `suffix`'s first `level` bits with one more bit appended.
const fn suffix_prefix_with(suffix: u8, level: u32, bit: bool) -> u8 {
    (suffix_prefix(suffix, level) << 1) | (bit as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::TreeRegion;

    fn stem(byte: u8) -> StemId {
        let mut out = [0u8; 32];
        out[0] = byte;
        StemId { region: TreeRegion::Header, stem: out }
    }

    fn one_target(suffix: u8) -> StemTargets {
        let mut targets = StemTargets::default();
        targets.targets.insert(suffix);
        targets.occupied.insert(suffix);
        targets
    }

    fn pops(size: u64, label: &str) -> RegionPopulations {
        RegionPopulations::unified(BackgroundPopulation::new(size, label))
    }

    const WIRE: &dyn Fn(TreeRegion) -> u64 = &|_| 33;

    #[test]
    fn retained_stems_drop_only_when_the_last_key_leaves() {
        let mut retained = RetainedStems::new();
        retained.add(stem(1));
        retained.add(stem(1));
        retained.remove(stem(1));
        assert!(retained.contains(&stem(1)), "a stem with a key left must stay retained");
        retained.remove(stem(1));
        assert!(!retained.contains(&stem(1)));
        assert!(retained.is_empty());
    }

    #[test]
    fn one_region_does_not_see_anothers_stems() {
        let mut retained = RetainedStems::new();
        let mut header = stem(0x42);
        retained.add(header);
        header.region = TreeRegion::Storage;
        assert!(!retained.contains(&header), "regions are separate trees under EIP-7864");
    }

    #[test]
    fn prefix_ranges_cover_exactly_their_subtree() {
        let p = Prefix::of(&stem(0b1010_0000).stem, 3);
        let (lo, hi) = prefix_range(p);
        assert_eq!(lo[0], 0b1010_0000);
        assert_eq!(hi[0], 0b1011_1111);
        assert_eq!(hi[31], 0xff);
    }

    #[test]
    fn a_cache_that_holds_the_target_stem_pays_no_path_nodes() {
        let population = pops(1 << 30, "held");
        let mut targets = BTreeMap::new();
        targets.insert(stem(0x42), one_target(0));

        let cold = binary_witness(&targets, &RetainedStems::new(), &population, WIRE, 40);
        let mut retained = RetainedStems::new();
        retained.add(stem(0x42));
        let warm = binary_witness(&targets, &retained, &population, WIRE, 40);

        assert!(cold.path_bytes > 0);
        assert_eq!(warm.path_bytes, 0, "every level sat below a retained frontier");
        assert!(warm.nodes_held_by_cache > 0);
        assert_eq!(warm.stem_bytes, 0, "a retained stem needs no identifier");
    }

    #[test]
    fn a_lone_branch_is_as_long_as_the_eip_says_it_is() {
        // EIP-7864 states a branch of `32 * (k - 1) * log(N) / log(k)` bytes, which for arity 2 is
        // one hash per level over log2(N) levels. This is that claim, checked against the
        // accounting rather than against a second derivation of it: one target alone in a
        // population of N.
        for exponent in [20u32, 27, 30, 34] {
            let population = pops(1u64 << exponent, "branch-length");
            let mut carried = 0u64;
            let mut samples = 0u64;
            for seed in 0..24u8 {
                let mut targets = BTreeMap::new();
                let mut key = [0u8; 32];
                key[..32].copy_from_slice(blake3::hash(&[seed]).as_bytes());
                targets.insert(StemId { region: TreeRegion::Header, stem: key }, one_target(0));
                carried += binary_witness(&targets, &RetainedStems::new(), &population, WIRE, 80)
                    .path_nodes;
                samples += 1;
            }
            let mean = carried as f64 / samples as f64;
            let expected = f64::from(exponent);
            assert!(
                (mean - expected).abs() < 3.0,
                "a lone branch in a 2^{exponent} state carried {mean:.1} hashes, not about                  {expected:.0}"
            );
        }
    }

    #[test]
    fn a_bigger_state_makes_paths_longer() {
        let mut targets = BTreeMap::new();
        targets.insert(stem(0x99), one_target(7));
        let small = binary_witness(&targets, &RetainedStems::new(), &pops(1 << 20, "s"), WIRE, 60);
        let big = binary_witness(&targets, &RetainedStems::new(), &pops(1 << 34, "b"), WIRE, 60);
        assert!(
            big.path_nodes > small.path_nodes,
            "a 16k-fold larger state must deepen the path: {} vs {}",
            big.path_nodes,
            small.path_nodes
        );
    }

    #[test]
    fn a_shared_outer_path_is_charged_once_however_many_targets_walk_it() {
        // Two stems agreeing on their first byte share eight levels of outer path. Charging each of
        // them separately for those levels was the defect this asserts against.
        let population = pops(1 << 34, "shared-outer");
        let mut a = [0u8; 32];
        a[0] = 0xab;
        a[1] = 0x01;
        let mut b = a;
        b[1] = 0x02;

        let mut one = BTreeMap::new();
        one.insert(StemId { region: TreeRegion::Header, stem: a }, one_target(0));
        let single = binary_witness(&one, &RetainedStems::new(), &population, WIRE, 60);

        let mut two = one.clone();
        two.insert(StemId { region: TreeRegion::Header, stem: b }, one_target(0));
        let pair = binary_witness(&two, &RetainedStems::new(), &population, WIRE, 60);

        let shared = single.path_nodes * 2 - pair.path_nodes;
        assert!(
            shared >= 8,
            "the shared byte must be deduplicated: {} vs {} nodes",
            pair.path_nodes,
            single.path_nodes * 2
        );
        assert!(pair.nodes_derived > 0);
    }

    #[test]
    fn a_stem_opened_for_two_reasons_is_paid_for_once() {
        // An account's basic data and its first code chunk share the header stem. Pricing them in
        // one target map must not charge that stem's outer path or identifier twice.
        let population = pops(1 << 30, "merged");
        let mut split_a = BTreeMap::new();
        split_a.insert(stem(0x55), one_target(0));
        let mut split_b = BTreeMap::new();
        split_b.insert(stem(0x55), one_target(4));
        let separate = {
            let mut cost = binary_witness(&split_a, &RetainedStems::new(), &population, WIRE, 40);
            cost.add(binary_witness(&split_b, &RetainedStems::new(), &population, WIRE, 40));
            cost
        };

        let mut merged_targets = StemTargets::default();
        merged_targets.targets.extend([0u8, 4]);
        merged_targets.occupied.extend([0u8, 4]);
        let mut merged = BTreeMap::new();
        merged.insert(stem(0x55), merged_targets);
        let together = binary_witness(&merged, &RetainedStems::new(), &population, WIRE, 40);

        assert!(together.total_bytes() < separate.total_bytes());
        assert_eq!(together.stems, 1);
        assert_eq!(separate.stems, 2, "the split accounting identified the same stem twice");
    }

    #[test]
    fn modelled_occupancy_adds_sibling_hashes_a_sparse_stem_would_not_pay() {
        let population = pops(1 << 30, "occupancy");
        let mut sparse = BTreeMap::new();
        sparse.insert(stem(0x66), one_target(0));
        let bare = binary_witness(&sparse, &RetainedStems::new(), &population, WIRE, 40);

        let mut dense_targets = one_target(0);
        dense_targets.modelled_extra = 64;
        let mut dense = BTreeMap::new();
        dense.insert(stem(0x66), dense_targets);
        let filled = binary_witness(&dense, &RetainedStems::new(), &population, WIRE, 40);

        assert!(
            filled.path_bytes > bare.path_bytes,
            "unobserved occupancy must cost sibling hashes: {} vs {}",
            filled.path_bytes,
            bare.path_bytes
        );
    }

    #[test]
    fn verkle_is_shallower_than_binary_for_the_same_targets() {
        let population = pops(1 << 31, "arity");
        let mut targets = BTreeMap::new();
        for i in 0..16u8 {
            targets.insert(stem(i), one_target(0));
        }
        let binary = binary_witness(&targets, &RetainedStems::new(), &population, WIRE, 60);
        let verkle = verkle_witness(&targets, &RetainedStems::new(), &population, WIRE, 576);
        assert!(
            verkle.path_nodes < binary.path_nodes,
            "256-ary paths must carry fewer nodes: {} vs {}",
            verkle.path_nodes,
            binary.path_nodes
        );
    }

    #[test]
    fn empty_siblings_cost_no_hash() {
        let population = pops(0, "empty");
        let mut targets = BTreeMap::new();
        targets.insert(stem(0x77), one_target(3));
        let cost = binary_witness(&targets, &RetainedStems::new(), &population, WIRE, 40);
        assert_eq!(cost.path_bytes, 0, "nothing else is in the state to hash against");
    }

    #[test]
    fn suffix_masks_answer_the_same_question_a_scan_would() {
        let set = SuffixSet::from_iter([0u8, 129, 255].into_iter());
        assert!(set.any_under(0, 1), "0 is in the lower half");
        assert!(set.any_under(1, 1), "129 and 255 are in the upper half");
        assert!(set.any_under(0, 8));
        assert!(!set.any_under(1, 8), "suffix 1 is not a member");
        assert!(set.any_under(0, 0));
        assert!(!SuffixSet::default().any_under(0, 0));
    }
}
