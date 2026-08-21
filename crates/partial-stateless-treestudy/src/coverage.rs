//! Which contract code a block actually runs.
//!
//! Under the MPT this quantity does not exist as a measurement, because it cannot change anything:
//! code is shipped as one blob keyed by its hash, and a witness carries every byte of a contract
//! whether the block ran one instruction of it or all of them. Nothing in the recorded corpus
//! captures it, because nothing in the MPT-era design has a use for it.
//!
//! EIP-7864 and EIP-6800 both put code in the tree as 31-byte chunks and carry only the chunks a
//! block executes. Coverage therefore becomes a first-class term in their witness size — and the
//! term is not neutral between arms. It is a lever only the successors have, it decides how
//! code-dominated their witnesses are, and code is the component a cache removes best. Assuming
//! full coverage is conservative in bytes and optimistic in ratio, which is exactly the kind of
//! assumption a study should replace with a number.
//!
//! The number is obtainable offline and without a state database. A block's recorded access set is
//! its complete read set — every account, slot, and bytecode it touched, with values — so the block
//! can be re-executed against the access set itself, with an inspector recording which chunk each
//! executed program counter falls in.

use alloy_primitives::{Address, Bytes, B256, U256};
use partial_stateless::accessed_state::BlockAccessedState;
use reth_primitives_traits::Account;
use revm::{
    bytecode::Bytecode,
    context::ContextTr,
    database::DBErrorMarker,
    interpreter::{
        interpreter_types::{InputsTr, Jumps},
        Interpreter, InterpreterTypes,
    },
    state::AccountInfo,
    Database, Inspector,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
};

/// Bytes of contract code one chunk carries.
pub const CODE_CHUNK_BYTES: usize = 31;

/// Which chunks of each bytecode a corpus was observed to execute.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CodeCoverage {
    /// Executed chunk indices per bytecode, ascending.
    ///
    /// Keyed by code hash rather than by account: the same bytecode at two addresses runs the same
    /// instructions, and a chunk index means the same thing at both.
    pub chunks: BTreeMap<B256, Vec<u32>>,
    /// Total chunks each bytecode has, so a coverage fraction can be recomputed from the file.
    pub total_chunks: BTreeMap<B256, u32>,
    /// Executing addresses the access set could not resolve to a bytecode.
    ///
    /// Reported rather than dropped: unresolved code is code whose coverage is unknown, and a
    /// coverage figure computed as though it were absent would be quietly too confident.
    #[serde(default)]
    pub unresolved: u64,
}

impl CodeCoverage {
    /// The chunks of `code_hash` this corpus ran, or `None` when it never entered that bytecode.
    ///
    /// Absence is not zero coverage. A bytecode can be *read* without being run — `EXTCODECOPY`,
    /// `EXTCODEHASH`, a plain balance query against a contract — and a caller that treated a
    /// missing entry as "no chunks" would silently drop code the witness has to carry.
    pub fn chunks_of(&self, code_hash: &B256) -> Option<&[u32]> {
        self.chunks.get(code_hash).map(Vec::as_slice)
    }

    /// Fraction of all chunks the corpus ran, over every bytecode it entered.
    pub fn overall_fraction(&self) -> f64 {
        let ran: u64 = self.chunks.values().map(|c| c.len() as u64).sum();
        let total: u64 = self
            .chunks
            .keys()
            .filter_map(|h| self.total_chunks.get(h))
            .map(|c| u64::from(*c))
            .sum();
        if total == 0 {
            0.0
        } else {
            ran as f64 / total as f64
        }
    }

    /// How many bytecodes were entered, and how many were only ever read.
    pub fn entered_and_read_only(&self) -> (usize, usize) {
        (self.chunks.len(), self.total_chunks.len().saturating_sub(self.chunks.len()))
    }

    /// Folds another block's observations in.
    pub fn merge(&mut self, other: &Self) {
        for (hash, chunks) in &other.chunks {
            let entry = self.chunks.entry(*hash).or_default();
            entry.extend_from_slice(chunks);
            entry.sort_unstable();
            entry.dedup();
        }
        for (hash, total) in &other.total_chunks {
            self.total_chunks.insert(*hash, *total);
        }
        self.unresolved += other.unresolved;
    }
}

/// Records the chunk each executed program counter falls in.
///
/// Keyed by the address whose code is running, not by the target of the call: under `DELEGATECALL`
/// those differ, and it is the *code owner's* leaves a witness has to open. That is also the key
/// the unified trees use, so recording it this way is the faithful choice rather than a workaround
/// for what the interpreter exposes. Initcode has no bytecode address and is skipped, because code
/// that was never deployed is in no tree.
#[derive(Debug, Default)]
pub struct ChunkInspector {
    seen: HashMap<Address, BTreeSet<u32>>,
}

impl ChunkInspector {
    /// A fresh recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// What it saw, as a coverage record over the bytecodes in `accessed`.
    ///
    /// Addresses are resolved to code hashes here rather than during execution, because the access
    /// set is what says which bytecode an address held and the interpreter does not. An address the
    /// recording does not name is counted in `unresolved` instead of being dropped.
    pub fn finish(self, accessed: &BlockAccessedState) -> CodeCoverage {
        let mut coverage = CodeCoverage::default();
        for (hash, code) in &accessed.codes {
            coverage.total_chunks.insert(*hash, chunk_count(code.len()));
        }
        for (address, chunks) in self.seen {
            let Some(hash) = accessed.accounts.get(&address).and_then(|a| a.code_hash) else {
                coverage.unresolved += 1;
                continue
            };
            // Code created in this block may not be in the access set's code map; take its extent
            // from the chunks actually reached.
            coverage
                .total_chunks
                .entry(hash)
                .or_insert_with(|| chunks.iter().copied().max().map_or(0, |top| top + 1));
            let entry = coverage.chunks.entry(hash).or_default();
            entry.extend(chunks);
            entry.sort_unstable();
            entry.dedup();
        }
        coverage
    }
}

impl<CTX, INTR> Inspector<CTX, INTR> for ChunkInspector
where
    CTX: ContextTr,
    INTR: InterpreterTypes,
{
    fn step(&mut self, interp: &mut Interpreter<INTR>, _context: &mut CTX) {
        let Some(address) = interp.input.bytecode_address().copied() else { return };
        let chunk = (interp.bytecode.pc() / CODE_CHUNK_BYTES) as u32;
        self.seen.entry(address).or_default().insert(chunk);
    }
}

/// How many chunks a contract of `code_len` bytes occupies.
pub const fn chunk_count(code_len: usize) -> u32 {
    code_len.div_ceil(CODE_CHUNK_BYTES) as u32
}

/// A database backed by one block's recorded parent-state witness.
///
/// The witness, not the access set. Both name the same keys, but the access set records what
/// execution *left behind* --- it is read off revm's cache after the block is merged --- so a
/// replay against it starts every account one nonce too high and the first transaction is refused.
/// The transition witness is proved against the parent root and is the only pre-state the corpus
/// has.
///
/// Anything the witness cannot answer is a gap in the recording rather than a missing feature, and
/// is surfaced as an error rather than as a default value: a zero returned for an unproven read
/// would change execution, and so would change the coverage this exists to measure.
#[derive(Debug)]
pub struct WitnessDatabase {
    accounts: HashMap<Address, Option<Account>>,
    storage: HashMap<(Address, B256), U256>,
    codes: HashMap<B256, Bytes>,
    block_hashes: BTreeMap<u64, B256>,
}

impl WitnessDatabase {
    /// Wraps materialised parent state, answering `BLOCKHASH` from `block_hashes`.
    pub const fn new(
        accounts: HashMap<Address, Option<Account>>,
        storage: HashMap<(Address, B256), U256>,
        codes: HashMap<B256, Bytes>,
        block_hashes: BTreeMap<u64, B256>,
    ) -> Self {
        Self { accounts, storage, codes, block_hashes }
    }
}

/// What the recorded witness could not answer.
#[derive(Debug)]
pub enum WitnessDatabaseError {
    /// A bytecode the block ran was not in the recording.
    MissingCode(B256),
    /// A `BLOCKHASH` outside the recorded ancestor range.
    MissingBlockHash(u64),
}

impl fmt::Display for WitnessDatabaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCode(hash) => {
                write!(f, "bytecode {hash} is not in the recorded witness")
            }
            Self::MissingBlockHash(number) => {
                write!(f, "block hash for {number} is not in the recorded ancestors")
            }
        }
    }
}

impl std::error::Error for WitnessDatabaseError {}
impl DBErrorMarker for WitnessDatabaseError {}

impl Database for WitnessDatabase {
    type Error = WitnessDatabaseError;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        // An address the witness proves absent and one it does not name are both empty to the
        // block: the witness covers every key the block reads.
        let Some(Some(account)) = self.accounts.get(&address) else { return Ok(None) };
        let code_hash = account.bytecode_hash.unwrap_or(revm::primitives::KECCAK_EMPTY);
        let code = self.codes.get(&code_hash).map(|bytes| Bytecode::new_raw(bytes.clone()));
        Ok(Some(AccountInfo {
            balance: account.balance,
            nonce: account.nonce,
            code_hash,
            account_id: None,
            code,
        }))
    }

    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        if code_hash == revm::primitives::KECCAK_EMPTY {
            return Ok(Bytecode::default())
        }
        self.codes
            .get(&code_hash)
            .map(|bytes| Bytecode::new_raw(bytes.clone()))
            .ok_or(WitnessDatabaseError::MissingCode(code_hash))
    }

    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        Ok(self.storage.get(&(address, B256::from(index))).copied().unwrap_or_default())
    }

    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        self.block_hashes
            .get(&number)
            .copied()
            .ok_or(WitnessDatabaseError::MissingBlockHash(number))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::Bytes;
    use partial_stateless::policy::AccountData;

    #[test]
    fn a_program_counter_lands_in_the_chunk_that_holds_it() {
        let chunk_of = |pc: usize| pc / CODE_CHUNK_BYTES;
        assert_eq!([0, 30, 31, 61, 62].map(chunk_of), [0, 0, 1, 1, 2]);
    }

    #[test]
    fn chunk_count_rounds_up_to_cover_the_tail() {
        assert_eq!(chunk_count(0), 0);
        assert_eq!(chunk_count(1), 1);
        assert_eq!(chunk_count(31), 1);
        assert_eq!(chunk_count(32), 2);
    }

    #[test]
    fn code_running_under_delegatecall_is_credited_to_the_code_owner() {
        // The inspector keys on the bytecode address, which under `DELEGATECALL` is the delegate
        // and not the caller. That is the account whose leaves a witness opens.
        let owner = Address::repeat_byte(0xd1);
        let hash = B256::repeat_byte(0xc0);
        let mut accessed = BlockAccessedState::default();
        accessed
            .accounts
            .insert(owner, AccountData { nonce: 0, balance: U256::ZERO, code_hash: Some(hash) });
        accessed.codes.insert(hash, Bytes::from(vec![0u8; 93]));

        let mut inspector = ChunkInspector::new();
        inspector.seen.entry(owner).or_default().extend([0u32, 2]);
        let coverage = inspector.finish(&accessed);
        assert_eq!(coverage.chunks_of(&hash), Some(&[0u32, 2][..]));
        assert_eq!(coverage.total_chunks.get(&hash), Some(&3));
        assert_eq!(coverage.unresolved, 0);
    }

    #[test]
    fn an_address_the_recording_cannot_name_is_counted_not_dropped() {
        let mut inspector = ChunkInspector::new();
        inspector.seen.entry(Address::repeat_byte(0xee)).or_default().insert(0);
        let coverage = inspector.finish(&BlockAccessedState::default());
        assert_eq!(coverage.unresolved, 1);
        assert!(coverage.chunks.is_empty());
    }

    #[test]
    fn a_bytecode_read_but_never_entered_is_not_zero_coverage() {
        let hash = B256::repeat_byte(7);
        let mut accessed = BlockAccessedState::default();
        accessed.codes.insert(hash, Bytes::from(vec![0u8; 200]));
        let coverage = ChunkInspector::new().finish(&accessed);
        assert_eq!(coverage.total_chunks.get(&hash), Some(&7));
        assert!(
            coverage.chunks_of(&hash).is_none(),
            "an unentered bytecode must be distinguishable from one that ran nothing"
        );
        assert_eq!(coverage.entered_and_read_only(), (0, 1));
    }

    #[test]
    fn merging_two_blocks_unions_their_chunks() {
        let hash = B256::repeat_byte(9);
        let mut first = CodeCoverage::default();
        first.chunks.insert(hash, vec![0, 2]);
        first.total_chunks.insert(hash, 4);
        let mut second = CodeCoverage::default();
        second.chunks.insert(hash, vec![2, 3]);
        second.total_chunks.insert(hash, 4);
        first.merge(&second);
        assert_eq!(first.chunks_of(&hash), Some(&[0u32, 2, 3][..]));
        assert!((first.overall_fraction() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn the_witness_answers_what_the_block_reads_and_refuses_what_it_cannot_prove() {
        let address = Address::repeat_byte(1);
        let hash = B256::repeat_byte(2);
        let mut accounts = HashMap::new();
        accounts.insert(
            address,
            Some(Account { nonce: 3, balance: U256::from(11), bytecode_hash: Some(hash) }),
        );
        accounts.insert(Address::repeat_byte(8), None);
        let mut storage = HashMap::new();
        storage.insert((address, B256::from(U256::from(5))), U256::from(42));
        let mut codes = HashMap::new();
        codes.insert(hash, Bytes::from(vec![0x60u8, 0x00]));

        let mut db = WitnessDatabase::new(accounts, storage, codes, BTreeMap::new());
        let info = db.basic(address).expect("proved").expect("present");
        assert_eq!((info.nonce, info.balance), (3, U256::from(11)));
        assert_eq!(db.storage(address, U256::from(5)).expect("proved"), U256::from(42));
        assert_eq!(db.storage(address, U256::from(6)).expect("unread"), U256::ZERO);
        assert!(db.basic(Address::repeat_byte(8)).expect("proved absent").is_none());
        assert!(db.basic(Address::repeat_byte(9)).expect("unnamed").is_none());
        assert!(matches!(
            db.code_by_hash(B256::repeat_byte(3)),
            Err(WitnessDatabaseError::MissingCode(_))
        ));
        assert!(matches!(db.block_hash(1), Err(WitnessDatabaseError::MissingBlockHash(1))));
    }
}
