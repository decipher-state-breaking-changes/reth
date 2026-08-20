//! Tree positions, one embedding per proposal, and the prefix arithmetic they are walked with.
//!
//! EIP-7864 and EIP-6800 are often described as the same tree with a different commitment. They are
//! not, and the difference lands squarely on witness size. EIP-7864 prefixes every key with a
//! storage type, so headers, code, and storage occupy three separate regions whose depths follow
//! their own populations, and its keys are variable-length byte strings — 33 bytes for a header
//! stem, 65 for a storage stem. EIP-6800 has one tree, a fixed 31-byte stem, and a header layout
//! fixed at 64 storage slots and 128 code chunks. Pricing one scheme's proof against the other's
//! embedding would attribute to the commitment a difference that comes from the key layout, so each
//! proposal gets its own [`TreeEmbedding`] here and the study never shares one between them.
//!
//! What both embeddings reproduce exactly is the *sharing structure*: which keys land in one stem,
//! and how many stems a contract's code or storage occupies. That is what a multiproof's size is a
//! function of. The position hash itself is BLAKE3 in both — the reference hash for EIP-7864, and a
//! stand-in for EIP-6800's Pedersen commitment, whose output is uniform over the same space, so
//! stem positions are distributionally identical and no size depends on the substitution.

use alloy_primitives::{keccak256, Address, B256, U256};

/// Bit-addressed path from the root of a binary tree.
///
/// Backed by a fixed 32-byte buffer because every position in this study is a hash: 256 bits is the
/// whole key space, and a path is never longer than the key that names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Prefix {
    bits: [u8; 32],
    len: u32,
}

impl Prefix {
    /// The empty path — the root.
    pub const fn root() -> Self {
        Self { bits: [0u8; 32], len: 0 }
    }

    /// The first `len` bits of `key`, most significant bit of byte zero first.
    pub fn of(key: &[u8; 32], len: u32) -> Self {
        debug_assert!(len <= 256);
        let mut bits = [0u8; 32];
        let whole = (len / 8) as usize;
        bits[..whole].copy_from_slice(&key[..whole]);
        let spare = len % 8;
        if spare != 0 {
            let mask = 0xffu8 << (8 - spare);
            bits[whole] = key[whole] & mask;
        }
        Self { bits, len }
    }

    /// The first `len` bits of `bits`, taken big-endian. Test and diagnostic helper.
    pub fn from_bits(bits: u64, len: u32) -> Self {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&bits.to_be_bytes());
        Self::of(&key, len)
    }

    /// Path length in bits.
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether this is the root.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The masked path bytes.
    pub const fn bytes(&self) -> [u8; 32] {
        self.bits
    }

    /// Bit at `index`, counting from the root.
    pub const fn bit(&self, index: u32) -> bool {
        let byte = self.bits[(index / 8) as usize];
        byte >> (7 - index % 8) & 1 == 1
    }

    /// This path cut back to `len` bits.
    pub fn truncated(&self, len: u32) -> Self {
        debug_assert!(len <= self.len);
        Self::of(&self.bits, len)
    }

    /// This path with one more bit.
    pub const fn pushed(&self, bit: bool) -> Self {
        let mut next = *self;
        next.len += 1;
        if bit {
            let index = self.len;
            next.bits[(index / 8) as usize] |= 1 << (7 - index % 8);
        }
        next
    }

    /// This path's sibling: the same path with its last bit flipped.
    pub fn sibling(&self) -> Self {
        debug_assert!(self.len > 0);
        let index = self.len - 1;
        let mut next = *self;
        next.bits[(index / 8) as usize] ^= 1 << (7 - index % 8);
        next
    }

    /// Whether `self` is a prefix of `other`.
    pub fn is_prefix_of(&self, other: &Self) -> bool {
        if self.len > other.len {
            return false
        }
        other.truncated(self.len) == *self
    }
}

/// Which of a scheme's key regions a stem lives in.
///
/// EIP-7864 gives headers, code, and storage separate storage-type prefixes, which puts them in
/// separate regions of the tree; a header stem branches among the other header stems and not among
/// the far more numerous storage stems, so its paths are shorter than a single-population model
/// would make them. EIP-6800 has one region and the enum collapses to [`Self::Unified`] there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TreeRegion {
    /// Account headers: basic data, code hash, and whatever storage and code share the stem.
    Header,
    /// Code chunks past the header stem.
    Code,
    /// Storage slots past the header stem.
    Storage,
    /// The single region of a scheme that does not separate them.
    Unified,
}

impl TreeRegion {
    /// Index into a per-region array.
    pub const fn index(self) -> usize {
        match self {
            Self::Header | Self::Unified => 0,
            Self::Code => 1,
            Self::Storage => 2,
        }
    }

    /// The storage-type byte EIP-7864 prefixes keys in this region with.
    pub const fn storage_type(self) -> u8 {
        match self {
            Self::Header | Self::Unified => 0,
            Self::Code => 1,
            Self::Storage => 255,
        }
    }
}

/// One position in a unified tree: which region, which stem, and which of its 256 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TreeKey {
    /// Region the stem lives in.
    pub region: TreeRegion,
    /// The stem, as the bit string a path walk descends.
    pub stem: [u8; 32],
    /// Which of the stem's 256 values.
    pub suffix: u8,
}

impl TreeKey {
    /// The stem alone, with no suffix.
    pub const fn stem_id(&self) -> StemId {
        StemId { region: self.region, stem: self.stem }
    }
}

/// A stem's identity, which is what a witness opens and a cache retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StemId {
    /// Region the stem lives in.
    pub region: TreeRegion,
    /// The stem's bit string.
    pub stem: [u8; 32],
}

/// Values per stem, in both proposals.
pub const STEM_SUBTREE_WIDTH: u32 = 256;
/// Bytes of contract code carried by one code chunk; the 32nd byte is the PUSHDATA marker.
pub const CODE_CHUNK_BYTES: usize = 31;

/// How a proposal places account state in its tree.
pub trait TreeEmbedding {
    /// Name as the report writes it.
    fn name(&self) -> &'static str;

    /// The stem an account's header occupies.
    fn header_stem(&self, address: Address) -> StemId;

    /// Where an account's packed basic data sits.
    fn basic_data(&self, address: Address) -> TreeKey;

    /// Where an account's code hash sits.
    fn code_hash(&self, address: Address) -> TreeKey;

    /// Where a storage slot sits.
    fn storage_slot(&self, address: Address, slot: U256) -> TreeKey;

    /// Where a code chunk sits.
    fn code_chunk(&self, address: Address, chunk_id: u32) -> TreeKey;

    /// How many of an account's leading code chunks share its header stem.
    fn code_chunks_in_header(&self) -> u32;

    /// Suffix at which header-resident code chunks begin.
    fn code_offset(&self) -> u32;

    /// How many of an account's leading storage slots share its header stem.
    fn storage_chunks_in_header(&self) -> u32;

    /// Suffix at which header-resident storage slots begin.
    fn header_storage_offset(&self) -> u32;

    /// Bytes a stem identifier occupies on the wire, in this region.
    fn stem_wire_bytes(&self, region: TreeRegion) -> u64;

    /// How many chunks a contract of `code_len` bytes occupies.
    fn chunk_count(code_len: usize) -> u32
    where
        Self: Sized,
    {
        code_len.div_ceil(CODE_CHUNK_BYTES) as u32
    }
}

/// The header-stem split of EIP-7864, which the EIP states two ways.
///
/// Its constant table reads 4 storage slots and 16 code chunks; its prose describes 64 and 128; and
/// the invariant printed beneath the table (`STEM_SUBTREE_WIDTH > CODE_OFFSET >
/// HEADER_STORAGE_OFFSET`) holds only for the second reading. The split decides how often an
/// account access and a storage access open one stem instead of two, so it is a parameter here and
/// the study runs both rather than presenting either as the specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderLayout {
    /// Suffix at which header-resident storage slots begin.
    pub header_storage_offset: u32,
    /// Suffix at which header-resident code chunks begin.
    pub code_offset: u32,
    /// How many of an account's leading code chunks live in its header stem.
    pub code_chunks_in_header: u32,
    /// How many of an account's leading storage slots live in its header stem.
    pub storage_chunks_in_header: u32,
}

impl HeaderLayout {
    /// The constant table as printed in EIP-7864.
    pub const TABLE: Self = Self {
        header_storage_offset: 20,
        code_offset: 4,
        code_chunks_in_header: 16,
        storage_chunks_in_header: 4,
    };

    /// The layout EIP-7864's prose describes, which is also EIP-6800's.
    pub const PROSE: Self = Self {
        header_storage_offset: 64,
        code_offset: 128,
        code_chunks_in_header: 128,
        storage_chunks_in_header: 64,
    };

    /// Selects a layout by name.
    pub fn by_name(name: &str) -> eyre::Result<Self> {
        match name {
            "table" => Ok(Self::TABLE),
            "prose" => Ok(Self::PROSE),
            other => {
                Err(eyre::eyre!("unknown header layout `{other}` (expected `table` or `prose`)"))
            }
        }
    }
}

/// EIP-7864's unified binary tree.
#[derive(Debug, Clone, Copy)]
pub struct Eip7864Keys {
    layout: HeaderLayout,
}

impl Eip7864Keys {
    /// Keys under `layout`.
    pub const fn new(layout: HeaderLayout) -> Self {
        Self { layout }
    }

    /// The layout in force.
    pub const fn layout(&self) -> HeaderLayout {
        self.layout
    }
}

impl TreeEmbedding for Eip7864Keys {
    fn name(&self) -> &'static str {
        "binary"
    }

    fn header_stem(&self, address: Address) -> StemId {
        StemId {
            region: TreeRegion::Header,
            stem: position(TreeRegion::Header, &[address32(address).as_slice()]),
        }
    }

    fn basic_data(&self, address: Address) -> TreeKey {
        let stem = self.header_stem(address);
        TreeKey { region: stem.region, stem: stem.stem, suffix: 0 }
    }

    fn code_hash(&self, address: Address) -> TreeKey {
        let stem = self.header_stem(address);
        TreeKey { region: stem.region, stem: stem.stem, suffix: 1 }
    }

    fn storage_slot(&self, address: Address, slot: U256) -> TreeKey {
        let header_slots = U256::from(self.layout.storage_chunks_in_header);
        if slot < header_slots {
            let suffix = self.layout.header_storage_offset + slot.to::<u32>();
            debug_assert!(suffix < STEM_SUBTREE_WIDTH);
            let stem = self.header_stem(address);
            return TreeKey { region: stem.region, stem: stem.stem, suffix: suffix as u8 }
        }
        let offset = slot - header_slots;
        let high = offset / U256::from(STEM_SUBTREE_WIDTH);
        let low = (offset % U256::from(STEM_SUBTREE_WIDTH)).to::<u32>();
        let address32 = address32(address);
        // `hash(address) + hash(address || high)`, per the EIP: binding the group index through a
        // second hash of the address is what stops storage keys ground to collide in one contract
        // from being replayed against another.
        let stem = position(
            TreeRegion::Storage,
            &[
                blake3::hash(address32.as_slice()).as_bytes().as_slice(),
                blake3::hash(&[address32.as_slice(), &high.to_be_bytes::<32>()].concat())
                    .as_bytes()
                    .as_slice(),
            ],
        );
        TreeKey { region: TreeRegion::Storage, stem, suffix: low as u8 }
    }

    fn code_chunk(&self, address: Address, chunk_id: u32) -> TreeKey {
        if chunk_id < self.layout.code_chunks_in_header {
            let suffix = self.layout.code_offset + chunk_id;
            debug_assert!(suffix < STEM_SUBTREE_WIDTH);
            let stem = self.header_stem(address);
            return TreeKey { region: stem.region, stem: stem.stem, suffix: suffix as u8 }
        }
        let offset = chunk_id - self.layout.code_chunks_in_header;
        let high = offset / STEM_SUBTREE_WIDTH;
        let low = offset % STEM_SUBTREE_WIDTH;
        let address32 = address32(address);
        let stem =
            position(TreeRegion::Code, &[address32.as_slice(), high.to_be_bytes().as_slice()]);
        TreeKey { region: TreeRegion::Code, stem, suffix: low as u8 }
    }

    fn code_chunks_in_header(&self) -> u32 {
        self.layout.code_chunks_in_header
    }

    fn code_offset(&self) -> u32 {
        self.layout.code_offset
    }

    fn storage_chunks_in_header(&self) -> u32 {
        self.layout.storage_chunks_in_header
    }

    fn header_storage_offset(&self) -> u32 {
        self.layout.header_storage_offset
    }

    /// A storage-type byte, then the region's position bytes.
    ///
    /// The storage region's position is two hashes, so its stems are twice the width of a header or
    /// code stem — a difference the wire pays for and a single constant would hide.
    fn stem_wire_bytes(&self, region: TreeRegion) -> u64 {
        match region {
            TreeRegion::Storage => 1 + 64,
            _ => 1 + 32,
        }
    }
}

/// EIP-6800's Verkle tree.
///
/// One region, a 31-byte stem, and a header layout fixed by the EIP rather than chosen: 64 storage
/// slots and 128 code chunks share an account's header stem.
#[derive(Debug, Clone, Copy, Default)]
pub struct Eip6800Keys;

impl Eip6800Keys {
    /// Storage slots sharing the header stem.
    pub const HEADER_STORAGE_OFFSET: u32 = 64;
    /// Suffix at which header-resident code chunks begin.
    pub const CODE_OFFSET: u32 = 128;
    /// Slots per stem past the header.
    pub const NODE_WIDTH: u32 = 256;

    /// `pedersen_hash(address32, tree_index)[:31]`, with the position hash standing in for
    /// Pedersen.
    fn tree_position(address: Address, tree_index: U256) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"eip6800");
        hasher.update(address32(address).as_slice());
        hasher.update(&tree_index.to_be_bytes::<32>());
        let mut out = *hasher.finalize().as_bytes();
        // The stem is 31 bytes; zeroing the last keeps the path walk from reading bits that are not
        // part of the stem's identity.
        out[31] = 0;
        out
    }

    fn key_at(address: Address, position: U256) -> TreeKey {
        let width = U256::from(Self::NODE_WIDTH);
        let tree_index = position / width;
        let suffix = (position % width).to::<u32>() as u8;
        TreeKey {
            region: TreeRegion::Unified,
            stem: Self::tree_position(address, tree_index),
            suffix,
        }
    }
}

impl TreeEmbedding for Eip6800Keys {
    fn name(&self) -> &'static str {
        "verkle"
    }

    fn header_stem(&self, address: Address) -> StemId {
        StemId { region: TreeRegion::Unified, stem: Self::tree_position(address, U256::ZERO) }
    }

    fn basic_data(&self, address: Address) -> TreeKey {
        Self::key_at(address, U256::ZERO)
    }

    fn code_hash(&self, address: Address) -> TreeKey {
        Self::key_at(address, U256::from(1))
    }

    fn storage_slot(&self, address: Address, slot: U256) -> TreeKey {
        let header_slots = U256::from(Self::CODE_OFFSET - Self::HEADER_STORAGE_OFFSET);
        let position = if slot < header_slots {
            U256::from(Self::HEADER_STORAGE_OFFSET) + slot
        } else {
            // MAIN_STORAGE_OFFSET is 256**31, so the whole of main storage sits above every header
            // position and no slot can collide with one.
            (U256::from(1) << 248) + slot
        };
        Self::key_at(address, position)
    }

    fn code_chunk(&self, address: Address, chunk_id: u32) -> TreeKey {
        Self::key_at(address, U256::from(Self::CODE_OFFSET) + U256::from(chunk_id))
    }

    fn code_chunks_in_header(&self) -> u32 {
        Self::NODE_WIDTH - Self::CODE_OFFSET
    }

    fn code_offset(&self) -> u32 {
        Self::CODE_OFFSET
    }

    fn storage_chunks_in_header(&self) -> u32 {
        Self::CODE_OFFSET - Self::HEADER_STORAGE_OFFSET
    }

    fn header_storage_offset(&self) -> u32 {
        Self::HEADER_STORAGE_OFFSET
    }

    fn stem_wire_bytes(&self, _region: TreeRegion) -> u64 {
        31
    }
}

/// Hashes a stem's pre-image down to the bit string a path walk descends.
fn position(region: TreeRegion, parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[region.storage_type()]);
    for part in parts {
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

/// Both proposals pass addresses as `Address32`: twelve zero bytes then the address.
fn address32(address: Address) -> B256 {
    let mut out = [0u8; 32];
    out[12..].copy_from_slice(address.as_slice());
    B256::from(out)
}

/// The MPT's hashed account key.
pub fn mpt_account_key(address: Address) -> [u8; 32] {
    keccak256(address.as_slice()).0
}

/// The MPT's hashed storage key.
pub fn mpt_storage_key(slot: B256) -> [u8; 32] {
    keccak256(slot.as_slice()).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_walks_and_flips_the_bits_it_says_it_does() {
        let key = [0b1010_1010u8; 32];
        let p = Prefix::of(&key, 3);
        assert_eq!(p.len(), 3);
        assert!(p.bit(0) && !p.bit(1) && p.bit(2));
        assert_eq!(p.pushed(false).len(), 4);
        assert_eq!(p.sibling().sibling(), p);
        assert!(p.truncated(1).is_prefix_of(&p));
        assert!(!p.is_prefix_of(&p.sibling()));
    }

    #[test]
    fn truncation_masks_the_tail_so_equality_is_prefix_equality() {
        let a = Prefix::of(&[0xff; 32], 4);
        let b = Prefix::of(&[0xf0; 32], 4);
        assert_eq!(a, b, "bits past the length must not affect identity");
    }

    #[test]
    fn each_proposal_places_the_same_account_somewhere_different() {
        let address = Address::repeat_byte(0x11);
        let binary = Eip7864Keys::new(HeaderLayout::TABLE).basic_data(address);
        let verkle = Eip6800Keys.basic_data(address);
        assert_ne!(
            binary.stem, verkle.stem,
            "one embedding shared between the schemes would attribute a layout difference to the \
             commitment"
        );
    }

    #[test]
    fn header_data_and_early_storage_share_one_stem_in_both_schemes() {
        let address = Address::repeat_byte(0x11);
        for keys in [&Eip7864Keys::new(HeaderLayout::TABLE) as &dyn TreeEmbedding, &Eip6800Keys] {
            let basic = keys.basic_data(address);
            let slot0 = keys.storage_slot(address, U256::ZERO);
            assert_eq!(basic.stem, slot0.stem, "{}", keys.name());
            assert_ne!(basic.suffix, slot0.suffix, "{}", keys.name());
        }
    }

    #[test]
    fn verkle_keeps_the_layout_its_own_eip_fixes() {
        let keys = Eip6800Keys;
        let address = Address::repeat_byte(0x22);
        assert_eq!(keys.storage_chunks_in_header(), 64);
        assert_eq!(keys.code_chunks_in_header(), 128);
        // Slot 63 is the last one inside the header stem; 64 is the first outside it.
        assert_eq!(keys.storage_slot(address, U256::from(63)).stem, keys.header_stem(address).stem);
        assert_ne!(keys.storage_slot(address, U256::from(64)).stem, keys.header_stem(address).stem);
    }

    #[test]
    fn storage_leaves_the_header_stem_once_it_runs_out_of_room() {
        let keys = Eip7864Keys::new(HeaderLayout::TABLE);
        let address = Address::repeat_byte(0x22);
        assert_eq!(keys.storage_slot(address, U256::from(3)).stem, keys.header_stem(address).stem);
        let outside = keys.storage_slot(address, U256::from(4));
        assert_ne!(outside.stem, keys.header_stem(address).stem);
        assert_eq!(outside.region, TreeRegion::Storage);
    }

    #[test]
    fn adjacent_storage_slots_group_into_one_stem() {
        let keys = Eip7864Keys::new(HeaderLayout::TABLE);
        let address = Address::repeat_byte(0x33);
        let base = 4u64;
        let first = keys.storage_slot(address, U256::from(base));
        let last = keys.storage_slot(address, U256::from(base + 255));
        let over = keys.storage_slot(address, U256::from(base + 256));
        assert_eq!(first.stem, last.stem);
        assert_eq!((first.suffix, last.suffix), (0, 255));
        assert_ne!(first.stem, over.stem);
    }

    #[test]
    fn one_slot_number_in_two_contracts_lands_in_two_stems() {
        let slot = U256::from(1_000_000u64);
        for keys in [&Eip7864Keys::new(HeaderLayout::TABLE) as &dyn TreeEmbedding, &Eip6800Keys] {
            let a = keys.storage_slot(Address::repeat_byte(0xaa), slot);
            let b = keys.storage_slot(Address::repeat_byte(0xbb), slot);
            assert_ne!(a.stem, b.stem, "{}", keys.name());
        }
    }

    #[test]
    fn code_chunks_split_at_each_schemes_own_header_boundary() {
        for keys in [&Eip7864Keys::new(HeaderLayout::TABLE) as &dyn TreeEmbedding, &Eip6800Keys] {
            let address = Address::repeat_byte(0x44);
            let boundary = keys.code_chunks_in_header();
            assert_eq!(keys.code_chunk(address, boundary - 1).stem, keys.header_stem(address).stem);
            assert_ne!(keys.code_chunk(address, boundary).stem, keys.header_stem(address).stem);
        }
    }

    #[test]
    fn a_storage_stem_costs_more_wire_than_a_header_stem_under_7864_only() {
        let binary = Eip7864Keys::new(HeaderLayout::TABLE);
        assert_eq!(binary.stem_wire_bytes(TreeRegion::Header), 33);
        assert_eq!(binary.stem_wire_bytes(TreeRegion::Storage), 65);
        assert_eq!(Eip6800Keys.stem_wire_bytes(TreeRegion::Unified), 31);
    }

    #[test]
    fn chunk_count_rounds_up_to_cover_the_tail() {
        assert_eq!(Eip7864Keys::chunk_count(0), 0);
        assert_eq!(Eip7864Keys::chunk_count(1), 1);
        assert_eq!(Eip7864Keys::chunk_count(31), 1);
        assert_eq!(Eip7864Keys::chunk_count(32), 2);
    }
}
