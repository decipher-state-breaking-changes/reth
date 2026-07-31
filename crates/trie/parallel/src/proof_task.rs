//! Parallel proof computation using worker pools with dedicated database transactions.
//!
//!
//! # Architecture
//!
//! - **Worker Pools**: Pre-spawned workers with dedicated database transactions
//!   - Storage pool: Handles storage proofs
//!   - Account pool: Handles account multiproofs
//! - **Direct Channel Access**: `ProofWorkerHandle` provides type-safe queue methods with direct
//!   access to worker channels, eliminating routing overhead
//! - **Automatic Shutdown**: Workers terminate gracefully when all handles are dropped
//!
//! # Message Flow
//!
//! 1. The `SparseTrieCacheTask` prepares a storage or account job and hands it to
//!    `ProofWorkerHandle`. The job carries a `ProofResultContext` so the worker knows how to send
//!    the result back.
//! 2. A worker receives the job, runs the proof, and sends a `ProofResultMessage` through the
//!    provided `ProofResultSender`.
//! 3. The `SparseTrieCacheTask` receives the message and proceeds with its state-root logic.
//!
//! Each job gets its own direct channel so results go straight back to the `SparseTrieCacheTask`.
//! That keeps ordering decisions in one place and lets workers run independently.
//!
//! ```text
//! SparseTrieCacheTask -> ProofWorkerHandle -> Storage/Account Worker
//!        ^                       |
//!        |                       v
//! ProofResultMessage <-- ProofResultSender
//! ```

use crate::{
    root::ParallelStateRootError,
    value_encoder::{AsyncAccountValueEncoder, ValueEncoderStats},
};
use alloy_primitives::{
    map::{B256Map, B256Set},
    B256, U256,
};
use crossbeam_channel::{unbounded, Receiver as CrossbeamReceiver, Sender as CrossbeamSender};
use reth_execution_errors::StateProofError;
use reth_primitives_traits::{dashmap::DashMap, FastInstant as Instant};
use reth_provider::{DatabaseProviderROFactory, ProviderError, ProviderResult};
use reth_storage_errors::db::DatabaseError;
use reth_tasks::{LazyHandle, Runtime, WorkerPool};
use reth_trie::{
    hashed_cursor::{HashedCursorFactory, HashedStorageCursor, InstrumentedHashedCursor},
    proof_v2,
    trie_cursor::{InstrumentedTrieCursor, TrieCursorFactory, TrieStorageCursor},
    DecodedMultiProofV2, HashedPostState, MultiProofTargetsV2, ProofTrieNodeV2, ProofV2Target,
};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tracing::{debug, debug_span, error, instrument, trace};

#[cfg(feature = "metrics")]
use crate::proof_task_metrics::{
    ProofTaskCursorMetrics, ProofTaskCursorMetricsCache, ProofTaskTrieMetrics,
};

/// Engine and one-shot proof handles use separate coordinators and physical worker pools. Within
/// each family, paired queue admission keeps both coordinator lane orders aligned.
const ENGINE_STORAGE_COORDINATOR: &str = "storage-workers";
const ENGINE_ACCOUNT_COORDINATOR: &str = "account-workers";
const ONE_SHOT_STORAGE_COORDINATOR: &str = "oneshot-strg";
const ONE_SHOT_ACCOUNT_COORDINATOR: &str = "oneshot-acct";

#[derive(Clone, Copy, Debug)]
enum ProofWorkerPoolKind {
    Engine,
    OneShot,
}

impl ProofWorkerPoolKind {
    const fn coordinators(self) -> (&'static str, &'static str) {
        match self {
            Self::Engine => (ENGINE_STORAGE_COORDINATOR, ENGINE_ACCOUNT_COORDINATOR),
            Self::OneShot => (ONE_SHOT_STORAGE_COORDINATOR, ONE_SHOT_ACCOUNT_COORDINATOR),
        }
    }

    fn storage_pool(self, runtime: &Runtime) -> &WorkerPool {
        match self {
            Self::Engine => runtime.proof_storage_worker_pool(),
            Self::OneShot => runtime.one_shot_proof_storage_worker_pool(),
        }
    }

    fn account_pool(self, runtime: &Runtime) -> &WorkerPool {
        match self {
            Self::Engine => runtime.proof_account_worker_pool(),
            Self::OneShot => runtime.one_shot_proof_account_worker_pool(),
        }
    }
}

/// Type alias for the V2 account proof calculator with instrumented cursors.
type V2AccountProofCalculator<'a, Provider> = proof_v2::ProofCalculator<
    InstrumentedTrieCursor<'a, <Provider as TrieCursorFactory>::AccountTrieCursor<'a>>,
    InstrumentedHashedCursor<'a, <Provider as HashedCursorFactory>::AccountCursor<'a>>,
    AsyncAccountValueEncoder<
        InstrumentedTrieCursor<'a, <Provider as TrieCursorFactory>::StorageTrieCursor<'a>>,
        InstrumentedHashedCursor<'a, <Provider as HashedCursorFactory>::StorageCursor<'a>>,
    >,
>;

/// Type alias for the V2 storage proof calculator with instrumented cursors.
type V2StorageProofCalculator<'a, Provider> = proof_v2::StorageProofCalculator<
    InstrumentedTrieCursor<'a, <Provider as TrieCursorFactory>::StorageTrieCursor<'a>>,
    InstrumentedHashedCursor<'a, <Provider as HashedCursorFactory>::StorageCursor<'a>>,
>;

/// Tracks worker availability counts.
///
/// It uses cacheline-aligned flags to avoid core-to-core chatter.
#[derive(Debug)]
struct AvailabilitySheet {
    /// One flag per worker, each on its own cacheline. Workers store `true` when idle,
    /// `false` when busy. Only the owning worker writes; the dispatcher only reads.
    flags: Vec<crossbeam_utils::CachePadded<AtomicBool>>,
}

impl AvailabilitySheet {
    /// Creates a new sheet with `count` workers, all initially marked as busy.
    fn new(count: usize) -> Self {
        let flags =
            (0..count).map(|_| crossbeam_utils::CachePadded::new(AtomicBool::new(false))).collect();
        Self { flags }
    }

    /// Returns `true` if more than one worker is currently idle.
    ///
    /// Note, that this is somewhat racy since a flag that was just saying `idle` and we counted it
    /// as such might turn into `busy` right away.
    fn has_multiple_idle(&self) -> bool {
        let mut idle = 0u32;
        for flag in &self.flags {
            if flag.load(Ordering::Relaxed) {
                idle += 1;
                if idle > 1 {
                    return true;
                }
            }
        }
        false
    }

    /// Marks the given worker as idle.
    fn mark_idle(&self, worker_id: usize) {
        self.flags[worker_id].store(true, Ordering::Relaxed);
    }

    /// Marks the given worker as busy.
    fn mark_busy(&self, worker_id: usize) {
        self.flags[worker_id].store(false, Ordering::Relaxed);
    }
}

/// A handle that provides type-safe access to proof worker pools.
///
/// The handle stores direct senders to both storage and account worker pools,
/// eliminating the need for a routing thread. All handles share reference-counted
/// channels, and workers shut down gracefully when all handles are dropped.
#[derive(Debug, Clone)]
pub struct ProofWorkerHandle {
    /// Direct sender to storage worker pool
    storage_work_tx: CrossbeamSender<StorageWorkerJob>,
    /// Direct sender to account worker pool
    account_work_tx: CrossbeamSender<AccountWorkerJob>,
    /// Per-worker availability flags for storage workers. Used to determine whether to chunk
    /// multiproofs.
    storage_availability: Arc<AvailabilitySheet>,
    /// Per-worker availability flags for account workers. Used to determine whether to chunk
    /// multiproofs.
    account_availability: Arc<AvailabilitySheet>,
    /// Total number of storage workers spawned
    storage_worker_count: usize,
    /// Total number of account workers spawned
    account_worker_count: usize,
    /// Completion signal for the storage worker pool.
    storage_worker_shutdown: LazyHandle<()>,
    /// Completion signal for the account worker pool.
    account_worker_shutdown: LazyHandle<()>,
}

impl ProofWorkerHandle {
    /// Spawns storage and account worker pools with dedicated database transactions.
    ///
    /// Returns a handle for submitting proof tasks to the worker pools.
    /// Workers run until the last handle is dropped.
    ///
    /// # Parameters
    /// - `runtime`: The centralized runtime used to spawn blocking worker tasks
    /// - `task_ctx`: Shared context with database view and prefix sets
    /// - `halve_workers`: Whether to halve the worker pool size (for small blocks)
    #[instrument(
        name = "ProofWorkerHandle::new",
        level = "debug",
        target = "trie::proof_task",
        skip_all
    )]
    pub fn new<Factory>(
        runtime: &Runtime,
        task_ctx: ProofTaskCtx<Factory>,
        halve_workers: bool,
    ) -> Self
    where
        Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let divisor = if halve_workers { 2 } else { 1 };
        // Keep at least one worker in each pool. A one-thread runtime with `halve_workers`
        // enabled would otherwise spawn no receivers and every dispatched proof would fail.
        let storage_worker_count =
            (runtime.proof_storage_worker_pool().current_num_threads() / divisor).max(1);
        let account_worker_count =
            (runtime.proof_account_worker_pool().current_num_threads() / divisor).max(1);

        Self::new_with_worker_counts(
            runtime,
            task_ctx,
            storage_worker_count,
            account_worker_count,
            ProofWorkerPoolKind::Engine,
        )
    }

    /// Spawns proof workers with explicit counts on the selected paired coordinator lanes and
    /// pools.
    fn new_with_worker_counts<Factory>(
        runtime: &Runtime,
        task_ctx: ProofTaskCtx<Factory>,
        storage_worker_count: usize,
        account_worker_count: usize,
        worker_pool_kind: ProofWorkerPoolKind,
    ) -> Self
    where
        Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        assert!(account_worker_count > 0, "at least one account proof worker is required");
        let (storage_work_tx, storage_work_rx) = unbounded::<StorageWorkerJob>();
        let (account_work_tx, account_work_rx) = unbounded::<AccountWorkerJob>();

        let cached_storage_roots = Arc::<DashMap<_, _>>::default();

        let storage_availability = Arc::new(AvailabilitySheet::new(storage_worker_count));
        let account_availability = Arc::new(AvailabilitySheet::new(account_worker_count));

        debug!(
            target: "trie::proof_task",
            storage_worker_count,
            account_worker_count,
            ?worker_pool_kind,
            "Spawning proof worker pools"
        );

        // Each coordinator blocks until all of its worker loops exit (channel close), so run it on
        // a persistent named thread. Submit the cooperating coordinators atomically to preserve
        // their queue order when multiple proof handles are constructed concurrently.
        let storage_rt = runtime.clone();
        let storage_task_ctx = task_ctx.clone();
        let storage_avail = storage_availability.clone();
        let storage_roots = cached_storage_roots.clone();
        let storage_parent_span = tracing::Span::current();
        let storage_pool_kind = worker_pool_kind;
        let storage_workers = move || {
            let worker_id = AtomicUsize::new(0);
            let workers_remaining = AtomicUsize::new(storage_worker_count);
            storage_pool_kind.storage_pool(&storage_rt).broadcast(
                storage_worker_count,
                |_| {
                    let worker_id = worker_id.fetch_add(1, Ordering::Relaxed);
                    let span = debug_span!(target: "trie::proof_task", parent: storage_parent_span.clone(), "storage_worker", ?worker_id);
                    let _guard = span.enter();

                    #[cfg(feature = "metrics")]
                    let metrics = ProofTaskTrieMetrics::default();
                    #[cfg(feature = "metrics")]
                    let cursor_metrics = ProofTaskCursorMetrics::new();

                    let worker = StorageProofWorker::new(
                        storage_task_ctx.clone(),
                        storage_work_rx.clone(),
                        worker_id,
                        storage_avail.clone(),
                        storage_roots.clone(),
                        #[cfg(feature = "metrics")]
                        metrics,
                        #[cfg(feature = "metrics")]
                        cursor_metrics,
                    );
                    let result = worker.run();
                    if let Err(error) = &result {
                        error!(
                            target: "trie::proof_task",
                            worker_id,
                            ?error,
                            "Storage worker failed"
                        );
                    }
                    if workers_remaining.fetch_sub(1, Ordering::AcqRel) == 1 &&
                        let Err(error) = result
                    {
                        fail_pending_storage_jobs(&storage_work_rx, error);
                    }
                },
            );
        };

        let account_rt = runtime.clone();
        let account_tx = storage_work_tx.clone();
        let account_avail = account_availability.clone();
        let account_parent_span = tracing::Span::current();
        let account_pool_kind = worker_pool_kind;
        let account_workers = move || {
            let worker_id = AtomicUsize::new(0);
            let workers_remaining = AtomicUsize::new(account_worker_count);
            account_pool_kind.account_pool(&account_rt).broadcast(
                account_worker_count,
                |_| {
                    let worker_id = worker_id.fetch_add(1, Ordering::Relaxed);
                    let span = debug_span!(target: "trie::proof_task", parent: account_parent_span.clone(), "account_worker", ?worker_id);
                    let _guard = span.enter();

                    #[cfg(feature = "metrics")]
                    let metrics = ProofTaskTrieMetrics::default();
                    #[cfg(feature = "metrics")]
                    let cursor_metrics = ProofTaskCursorMetrics::new();

                    let worker = AccountProofWorker::new(
                        task_ctx.clone(),
                        account_work_rx.clone(),
                        worker_id,
                        account_tx.clone(),
                        account_avail.clone(),
                        cached_storage_roots.clone(),
                        #[cfg(feature = "metrics")]
                        metrics,
                        #[cfg(feature = "metrics")]
                        cursor_metrics,
                    );
                    let result = worker.run();
                    if let Err(error) = &result {
                        error!(
                            target: "trie::proof_task",
                            worker_id,
                            ?error,
                            "Account worker failed"
                        );
                    }
                    if workers_remaining.fetch_sub(1, Ordering::AcqRel) == 1 &&
                        let Err(error) = result
                    {
                        fail_pending_account_jobs(&account_work_rx, error);
                    }
                },
            );
        };

        let (storage_coordinator, account_coordinator) = worker_pool_kind.coordinators();
        let (storage_worker_shutdown, account_worker_shutdown) = runtime.spawn_blocking_named_pair(
            storage_coordinator,
            storage_workers,
            account_coordinator,
            account_workers,
        );

        Self {
            storage_work_tx,
            account_work_tx,
            storage_availability,
            account_availability,
            storage_worker_count,
            account_worker_count,
            storage_worker_shutdown,
            account_worker_shutdown,
        }
    }

    /// Closes this handle's worker channels and waits for both pools to release their providers.
    ///
    /// This is used only by the one-shot wrapper, which owns the sole handle.
    fn shutdown_and_wait(self) {
        let Self {
            storage_work_tx,
            account_work_tx,
            storage_worker_shutdown,
            account_worker_shutdown,
            ..
        } = self;
        drop(account_work_tx);
        account_worker_shutdown.get();
        drop(storage_work_tx);
        storage_worker_shutdown.get();
    }

    /// Returns `true` if more than one storage worker is currently idle.
    pub fn has_multiple_idle_storage_workers(&self) -> bool {
        self.storage_availability.has_multiple_idle()
    }

    /// Returns `true` if more than one account worker is currently idle.
    pub fn has_multiple_idle_account_workers(&self) -> bool {
        self.account_availability.has_multiple_idle()
    }

    /// Returns the number of pending storage tasks in the queue.
    pub fn pending_storage_tasks(&self) -> usize {
        self.storage_work_tx.len()
    }

    /// Returns the number of pending account tasks in the queue.
    pub fn pending_account_tasks(&self) -> usize {
        self.account_work_tx.len()
    }

    /// Returns the total number of storage workers in the pool.
    pub const fn total_storage_workers(&self) -> usize {
        self.storage_worker_count
    }

    /// Returns the total number of account workers in the pool.
    pub const fn total_account_workers(&self) -> usize {
        self.account_worker_count
    }

    /// Dispatch a storage proof computation to storage worker pool
    ///
    /// The result will be sent via the `proof_result_sender` channel.
    pub fn dispatch_storage_proof(
        &self,
        input: StorageProofInput,
        proof_result_sender: CrossbeamSender<StorageProofResultMessage>,
    ) -> Result<(), ProviderError> {
        let hashed_address = input.hashed_address;
        self.storage_work_tx
            .send(StorageWorkerJob::StorageProof { input, proof_result_sender })
            .map_err(|err| {
                let StorageWorkerJob::StorageProof { proof_result_sender, .. } = err.0;
                let _ = proof_result_sender.send(StorageProofResultMessage {
                    hashed_address,
                    result: Err(
                        DatabaseError::Other("storage workers unavailable".to_string()).into()
                    ),
                });

                ProviderError::other(std::io::Error::other("storage workers unavailable"))
            })
    }

    /// Dispatch an account multiproof computation
    ///
    /// The result will be sent via the `result_sender` channel included in the input.
    pub fn dispatch_account_multiproof(
        &self,
        input: AccountMultiproofInput,
    ) -> Result<(), ProviderError> {
        self.account_work_tx
            .send(AccountWorkerJob::AccountMultiproof { input: Box::new(input) })
            .map_err(|err| {
                let error =
                    ProviderError::other(std::io::Error::other("account workers unavailable"));

                let AccountWorkerJob::AccountMultiproof { input } = err.0;
                let ProofResultContext { sender: result_tx, state, start_time: start } =
                    input.into_proof_result_sender();

                let _ = result_tx.send(ProofResultMessage {
                    result: Err(ParallelStateRootError::Provider(error.clone())),
                    elapsed: start.elapsed(),
                    state,
                });

                error
            })
    }
}

/// Fails work that could not be consumed because every storage worker failed to initialize.
fn fail_pending_storage_jobs(work_rx: &CrossbeamReceiver<StorageWorkerJob>, error: ProviderError) {
    let message = format!("all storage proof workers failed to initialize: {error}");
    for job in work_rx.try_iter() {
        let StorageWorkerJob::StorageProof { input, proof_result_sender } = job;
        let _ = proof_result_sender.send(StorageProofResultMessage {
            hashed_address: input.hashed_address,
            result: Err(DatabaseError::Other(message.clone()).into()),
        });
    }
}

/// Fails work that could not be consumed because every account worker failed to initialize.
fn fail_pending_account_jobs(work_rx: &CrossbeamReceiver<AccountWorkerJob>, error: ProviderError) {
    for job in work_rx.try_iter() {
        let AccountWorkerJob::AccountMultiproof { input } = job;
        let ProofResultContext { sender: result_tx, state, start_time: start } =
            input.into_proof_result_sender();
        let _ = result_tx.send(ProofResultMessage {
            result: Err(ParallelStateRootError::Provider(error.clone())),
            elapsed: start.elapsed(),
            state,
        });
    }
}

/// Effective worker counts used by a one-shot parallel multiproof.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParallelMultiproofWorkerStats {
    /// Number of storage proof workers started for the target set.
    pub storage_workers: usize,
    /// Number of account proof workers started for the target set.
    pub account_workers: usize,
}

/// Computes one V2 multiproof using isolated one-shot account and storage proof worker pools.
///
/// This submits one account job, which dispatches the targeted storage tries across the storage
/// workers and overlaps their computation with the account-trie walk. The isolated worker handle is
/// scoped to this call so every worker's read transaction is released after the proof completes.
pub fn parallel_multiproof_v2<Factory>(
    runtime: &Runtime,
    factory: Factory,
    targets: MultiProofTargetsV2,
    halve_workers: bool,
) -> Result<DecodedMultiProofV2, ParallelStateRootError>
where
    Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>
        + Clone
        + Send
        + Sync
        + 'static,
{
    parallel_multiproof_v2_with_stats(runtime, factory, targets, halve_workers)
        .map(|(proof, _)| proof)
}

/// Computes one V2 multiproof and returns the effective one-shot worker counts.
///
/// Exactly one account worker is used because this wrapper submits one account job. Storage workers
/// are limited to the number of distinct targeted storage tries and the configured half/full pool
/// cap. One-shot work uses isolated lazy storage/account pools, and concurrent one-shot handles use
/// atomically paired coordinator lanes so they cannot acquire those pools in opposite orders.
pub fn parallel_multiproof_v2_with_stats<Factory>(
    runtime: &Runtime,
    factory: Factory,
    targets: MultiProofTargetsV2,
    halve_workers: bool,
) -> Result<(DecodedMultiProofV2, ParallelMultiproofWorkerStats), ParallelStateRootError>
where
    Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>
        + Clone
        + Send
        + Sync
        + 'static,
{
    if targets.is_empty() {
        return Ok((DecodedMultiProofV2::default(), ParallelMultiproofWorkerStats::default()))
    }

    let divisor = if halve_workers { 2 } else { 1 };
    let requested_storage_cap =
        (runtime.proof_storage_worker_pool().configured_num_threads() / divisor).max(1);
    let isolated_storage_cap =
        runtime.one_shot_proof_storage_worker_pool().configured_num_threads();
    let storage_worker_cap = requested_storage_cap.min(isolated_storage_cap);
    let worker_stats = ParallelMultiproofWorkerStats {
        storage_workers: targets.storage_targets.len().max(1).min(storage_worker_cap),
        account_workers: 1,
    };

    let (result_tx, result_rx) = unbounded();
    let proof_handle = ProofWorkerHandle::new_with_worker_counts(
        runtime,
        ProofTaskCtx::new(factory),
        worker_stats.storage_workers,
        worker_stats.account_workers,
        ProofWorkerPoolKind::OneShot,
    );
    let proof_result_sender =
        ProofResultContext::new(result_tx, HashedPostState::default(), Instant::now());
    if let Err(error) = proof_handle
        .dispatch_account_multiproof(AccountMultiproofInput { targets, proof_result_sender })
    {
        proof_handle.shutdown_and_wait();
        return Err(error.into())
    }

    let message = match result_rx.recv() {
        Ok(message) => message,
        Err(_) => {
            proof_handle.shutdown_and_wait();
            return Err(ParallelStateRootError::Other(
                "parallel multiproof result channel closed".to_string(),
            ))
        }
    };
    proof_handle.shutdown_and_wait();
    message.result.map(|proof| (proof, worker_stats))
}

/// Data used for initializing cursor factories that is shared across all proof worker instances.
#[derive(Clone, Debug)]
pub struct ProofTaskCtx<Factory> {
    /// The factory for creating state providers.
    factory: Factory,
    /// Maximum random jitter to apply before each proof computation (trie-debug only).
    #[cfg(feature = "trie-debug")]
    proof_jitter: Option<Duration>,
}

impl<Factory> ProofTaskCtx<Factory> {
    /// Creates a new [`ProofTaskCtx`] with the given factory.
    pub const fn new(factory: Factory) -> Self {
        Self {
            factory,
            #[cfg(feature = "trie-debug")]
            proof_jitter: None,
        }
    }

    /// Sets the maximum proof jitter duration (trie-debug only).
    #[cfg(feature = "trie-debug")]
    pub const fn with_proof_jitter(mut self, jitter: Option<Duration>) -> Self {
        self.proof_jitter = jitter;
        self
    }
}

/// This contains all information shared between account proof worker instances.
#[derive(Debug)]
pub struct ProofTaskTx<Provider> {
    /// The provider that implements `TrieCursorFactory` and `HashedCursorFactory`.
    provider: Provider,

    /// Identifier for the worker within the worker pool, used only for tracing.
    id: usize,
}

impl<Provider> ProofTaskTx<Provider> {
    /// Initializes a [`ProofTaskTx`] with the given provider and ID.
    const fn new(provider: Provider, id: usize) -> Self {
        Self { provider, id }
    }
}

impl<Provider> ProofTaskTx<Provider>
where
    Provider: TrieCursorFactory + HashedCursorFactory,
{
    fn compute_v2_storage_proof<TC, HC>(
        &self,
        input: StorageProofInput,
        calculator: &mut proof_v2::StorageProofCalculator<TC, HC>,
    ) -> Result<StorageProofResult, StateProofError>
    where
        TC: TrieStorageCursor,
        HC: HashedStorageCursor<Value = U256>,
    {
        let StorageProofInput { hashed_address, mut targets } = input;

        let span = debug_span!(
            target: "trie::proof_task",
            "V2 Storage proof calculation",
            n = %targets.len(),
        );
        let _span_guard = span.enter();

        let proof_start = Instant::now();

        // If targets is empty it means the caller only wants the root node.
        let proof = if targets.is_empty() {
            let root_node = calculator.storage_root_node(hashed_address)?;
            vec![root_node]
        } else {
            calculator.storage_proof(hashed_address, &mut targets)?
        };

        let root = calculator.compute_root_hash(&proof)?;

        trace!(
            target: "trie::proof_task",
            hashed_address = ?hashed_address,
            proof_time_us = proof_start.elapsed().as_micros(),
            ?root,
            worker_id = self.id,
            "Completed V2 storage proof calculation"
        );

        Ok(StorageProofResult { proof, root })
    }
}

/// Channel used by worker threads to deliver `ProofResultMessage` items back to
/// `SparseTrieCacheTask`.
///
/// Workers use this sender to deliver proof results directly to `SparseTrieCacheTask`.
pub type ProofResultSender = CrossbeamSender<ProofResultMessage>;

/// Message containing a completed proof result with metadata for direct delivery to
/// `SparseTrieCacheTask`.
///
/// This type enables workers to send proof results directly to the `SparseTrieCacheTask` event
/// loop.
#[derive(Debug)]
pub struct ProofResultMessage {
    /// The proof calculation result
    pub result: Result<DecodedMultiProofV2, ParallelStateRootError>,
    /// Time taken for the entire proof calculation (from dispatch to completion)
    pub elapsed: Duration,
    /// Original state update that triggered this proof
    pub state: HashedPostState,
}

/// Context for sending proof calculation results back to `SparseTrieCacheTask`.
///
/// This struct contains all context needed to send and track proof calculation results.
/// Workers use this to deliver completed proofs back to the main event loop.
#[derive(Debug, Clone)]
pub struct ProofResultContext {
    /// Channel sender for result delivery
    pub sender: ProofResultSender,
    /// Original state update that triggered this proof
    pub state: HashedPostState,
    /// Calculation start time for measuring elapsed duration
    pub start_time: Instant,
}

impl ProofResultContext {
    /// Creates a new proof result context.
    pub const fn new(
        sender: ProofResultSender,
        state: HashedPostState,
        start_time: Instant,
    ) -> Self {
        Self { sender, state, start_time }
    }
}

/// The results of a storage proof calculation.
#[derive(Debug)]
pub(crate) struct StorageProofResult {
    /// The calculated V2 proof nodes
    pub proof: Vec<ProofTrieNodeV2>,
    /// The storage root calculated by the V2 proof
    pub root: Option<B256>,
}

impl StorageProofResult {
    /// Returns the calculated root of the trie, if one can be calculated from the proof.
    const fn root(&self) -> Option<B256> {
        self.root
    }
}

/// Message containing a completed storage proof result with metadata.
#[derive(Debug)]
pub struct StorageProofResultMessage {
    /// The hashed address this storage proof belongs to
    #[allow(dead_code)]
    pub(crate) hashed_address: B256,
    /// The storage proof calculation result
    pub(crate) result: Result<StorageProofResult, StateProofError>,
}

/// Internal message for storage workers.
#[derive(Debug)]
pub(crate) enum StorageWorkerJob {
    /// Storage proof computation request
    StorageProof {
        /// Storage proof input parameters
        input: StorageProofInput,
        /// Context for sending the proof result.
        proof_result_sender: CrossbeamSender<StorageProofResultMessage>,
    },
}

/// Worker for storage trie operations.
///
/// Each worker maintains a dedicated database transaction and processes
/// storage proof requests.
struct StorageProofWorker<Factory> {
    /// Shared task context with database factory and prefix sets
    task_ctx: ProofTaskCtx<Factory>,
    /// Channel for receiving work
    work_rx: CrossbeamReceiver<StorageWorkerJob>,
    /// Unique identifier for this worker (used for tracing)
    worker_id: usize,
    /// Per-worker availability flags
    availability: Arc<AvailabilitySheet>,
    /// Cached storage roots
    cached_storage_roots: Arc<DashMap<B256, B256>>,
    /// Metrics collector for this worker
    #[cfg(feature = "metrics")]
    metrics: ProofTaskTrieMetrics,
    /// Cursor metrics for this worker
    #[cfg(feature = "metrics")]
    cursor_metrics: ProofTaskCursorMetrics,
}

impl<Factory> StorageProofWorker<Factory>
where
    Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>,
{
    /// Creates a new storage proof worker.
    const fn new(
        task_ctx: ProofTaskCtx<Factory>,
        work_rx: CrossbeamReceiver<StorageWorkerJob>,
        worker_id: usize,
        availability: Arc<AvailabilitySheet>,
        cached_storage_roots: Arc<DashMap<B256, B256>>,
        #[cfg(feature = "metrics")] metrics: ProofTaskTrieMetrics,
        #[cfg(feature = "metrics")] cursor_metrics: ProofTaskCursorMetrics,
    ) -> Self {
        Self {
            task_ctx,
            work_rx,
            worker_id,
            availability,
            cached_storage_roots,
            #[cfg(feature = "metrics")]
            metrics,
            #[cfg(feature = "metrics")]
            cursor_metrics,
        }
    }

    /// Runs the worker loop, processing jobs until the channel closes.
    ///
    /// # Lifecycle
    ///
    /// 1. Initializes database provider and transaction
    /// 2. Advertises availability
    /// 3. Processes jobs in a loop:
    ///    - Receives job from channel
    ///    - Marks worker as busy
    ///    - Processes the job
    ///    - Marks worker as available
    /// 4. Shuts down when channel closes
    ///
    /// # Panic Safety
    ///
    /// If this function panics, the worker thread terminates but other workers
    /// continue operating and the system degrades gracefully.
    fn run(mut self) -> ProviderResult<()> {
        // Create provider from factory
        let provider = self.task_ctx.factory.database_provider_ro()?;
        let proof_tx = ProofTaskTx::new(provider, self.worker_id);

        trace!(
            target: "trie::proof_task",
            worker_id = self.worker_id,
            "Storage worker started"
        );

        let mut storage_proofs_processed = 0u64;
        let mut cursor_metrics_cache = ProofTaskCursorMetricsCache::default();
        let trie_cursor = proof_tx.provider.storage_trie_cursor(B256::ZERO)?;
        let hashed_cursor = proof_tx.provider.hashed_storage_cursor(B256::ZERO)?;
        let instrumented_trie_cursor =
            InstrumentedTrieCursor::new(trie_cursor, &mut cursor_metrics_cache.storage_trie_cursor);
        let instrumented_hashed_cursor = InstrumentedHashedCursor::new(
            hashed_cursor,
            &mut cursor_metrics_cache.storage_hashed_cursor,
        );
        let mut v2_calculator = proof_v2::StorageProofCalculator::new_storage(
            instrumented_trie_cursor,
            instrumented_hashed_cursor,
        );

        // Initially mark this worker as available.
        self.availability.mark_idle(self.worker_id);

        let mut total_idle_time = Duration::ZERO;
        let mut idle_start = Instant::now();

        while let Ok(job) = self.work_rx.recv() {
            total_idle_time += idle_start.elapsed();

            // Mark worker as busy.
            self.availability.mark_busy(self.worker_id);

            #[cfg(feature = "trie-debug")]
            if let Some(max_jitter) = self.task_ctx.proof_jitter {
                let jitter =
                    Duration::from_nanos(rand::random_range(0..=max_jitter.as_nanos() as u64));
                trace!(
                    target: "trie::proof_task",
                    worker_id = self.worker_id,
                    jitter_us = jitter.as_micros(),
                    "Storage worker applying proof jitter"
                );
                std::thread::sleep(jitter);
            }

            match job {
                StorageWorkerJob::StorageProof { input, proof_result_sender } => {
                    self.process_storage_proof(
                        &proof_tx,
                        &mut v2_calculator,
                        input,
                        proof_result_sender,
                        &mut storage_proofs_processed,
                    );
                }
            }

            // Mark worker as available again.
            self.availability.mark_idle(self.worker_id);

            idle_start = Instant::now();
        }

        // Drop calculator to release mutable borrows on cursor_metrics_cache.
        drop(v2_calculator);

        trace!(
            target: "trie::proof_task",
            worker_id = self.worker_id,
            storage_proofs_processed,
            total_idle_time_us = total_idle_time.as_micros(),
            "Storage worker shutting down"
        );

        #[cfg(feature = "metrics")]
        {
            self.metrics.record_storage_worker_idle_time(total_idle_time);
            self.cursor_metrics.record(&mut cursor_metrics_cache);
        }

        Ok(())
    }

    /// Processes a storage proof request.
    fn process_storage_proof<Provider, TC, HC>(
        &self,
        proof_tx: &ProofTaskTx<Provider>,
        v2_calculator: &mut proof_v2::StorageProofCalculator<TC, HC>,
        input: StorageProofInput,
        proof_result_sender: CrossbeamSender<StorageProofResultMessage>,
        storage_proofs_processed: &mut u64,
    ) where
        Provider: TrieCursorFactory + HashedCursorFactory,
        TC: TrieStorageCursor,
        HC: HashedStorageCursor<Value = U256>,
    {
        let hashed_address = input.hashed_address;
        let proof_start = Instant::now();

        trace!(
            target: "trie::proof_task",
            worker_id = self.worker_id,
            hashed_address = ?hashed_address,
            targets_len = input.targets.len(),
            "Processing V2 storage proof"
        );

        let result = proof_tx.compute_v2_storage_proof(input, v2_calculator);

        let proof_elapsed = proof_start.elapsed();
        *storage_proofs_processed += 1;

        let root = result.as_ref().ok().and_then(|result| result.root());

        if proof_result_sender.send(StorageProofResultMessage { hashed_address, result }).is_err() {
            trace!(
                target: "trie::proof_task",
                worker_id = self.worker_id,
                hashed_address = ?hashed_address,
                storage_proofs_processed,
                "Proof result receiver dropped, discarding result"
            );
        }

        if let Some(root) = root {
            self.cached_storage_roots.insert(hashed_address, root);
        }

        trace!(
            target: "trie::proof_task",
            worker_id = self.worker_id,
            hashed_address = ?hashed_address,
            proof_time_us = proof_elapsed.as_micros(),
            total_processed = storage_proofs_processed,
            ?root,
            "Storage proof completed"
        );
    }
}

/// Worker for account trie operations.
///
/// Each worker maintains a dedicated database transaction and processes
/// account multiproof requests.
struct AccountProofWorker<Factory> {
    /// Shared task context with database factory and prefix sets
    task_ctx: ProofTaskCtx<Factory>,
    /// Channel for receiving work
    work_rx: CrossbeamReceiver<AccountWorkerJob>,
    /// Unique identifier for this worker (used for tracing)
    worker_id: usize,
    /// Channel for dispatching storage proof work (for pre-dispatched target proofs)
    storage_work_tx: CrossbeamSender<StorageWorkerJob>,
    /// Per-worker availability flags
    availability: Arc<AvailabilitySheet>,
    /// Cached storage roots
    cached_storage_roots: Arc<DashMap<B256, B256>>,
    /// Metrics collector for this worker
    #[cfg(feature = "metrics")]
    metrics: ProofTaskTrieMetrics,
    /// Cursor metrics for this worker
    #[cfg(feature = "metrics")]
    cursor_metrics: ProofTaskCursorMetrics,
}

impl<Factory> AccountProofWorker<Factory>
where
    Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>,
{
    /// Creates a new account proof worker.
    #[expect(clippy::too_many_arguments)]
    const fn new(
        task_ctx: ProofTaskCtx<Factory>,
        work_rx: CrossbeamReceiver<AccountWorkerJob>,
        worker_id: usize,
        storage_work_tx: CrossbeamSender<StorageWorkerJob>,
        availability: Arc<AvailabilitySheet>,
        cached_storage_roots: Arc<DashMap<B256, B256>>,
        #[cfg(feature = "metrics")] metrics: ProofTaskTrieMetrics,
        #[cfg(feature = "metrics")] cursor_metrics: ProofTaskCursorMetrics,
    ) -> Self {
        Self {
            task_ctx,
            work_rx,
            worker_id,
            storage_work_tx,
            availability,
            cached_storage_roots,
            #[cfg(feature = "metrics")]
            metrics,
            #[cfg(feature = "metrics")]
            cursor_metrics,
        }
    }

    /// Runs the worker loop, processing jobs until the channel closes.
    ///
    /// # Lifecycle
    ///
    /// 1. Initializes database provider and transaction
    /// 2. Advertises availability
    /// 3. Processes jobs in a loop:
    ///    - Receives job from channel
    ///    - Marks worker as busy
    ///    - Processes the job
    ///    - Marks worker as available
    /// 4. Shuts down when channel closes
    ///
    /// # Panic Safety
    ///
    /// If this function panics, the worker thread terminates but other workers
    /// continue operating and the system degrades gracefully.
    fn run(mut self) -> ProviderResult<()> {
        let provider = self.task_ctx.factory.database_provider_ro()?;

        trace!(
            target: "trie::proof_task",
            worker_id=self.worker_id,
            "Account worker started"
        );

        let mut account_proofs_processed = 0u64;
        let mut cursor_metrics_cache = ProofTaskCursorMetricsCache::default();

        // Create both account and storage calculators for V2 proofs.
        // The storage calculator is wrapped in Rc<RefCell<...>> for sharing with value encoders.
        let account_trie_cursor = provider.account_trie_cursor()?;
        let account_hashed_cursor = provider.hashed_account_cursor()?;

        let storage_trie_cursor = provider.storage_trie_cursor(B256::ZERO)?;
        let storage_hashed_cursor = provider.hashed_storage_cursor(B256::ZERO)?;

        let instrumented_account_trie_cursor = InstrumentedTrieCursor::new(
            account_trie_cursor,
            &mut cursor_metrics_cache.account_trie_cursor,
        );
        let instrumented_account_hashed_cursor = InstrumentedHashedCursor::new(
            account_hashed_cursor,
            &mut cursor_metrics_cache.account_hashed_cursor,
        );
        let instrumented_storage_trie_cursor = InstrumentedTrieCursor::new(
            storage_trie_cursor,
            &mut cursor_metrics_cache.storage_trie_cursor,
        );
        let instrumented_storage_hashed_cursor = InstrumentedHashedCursor::new(
            storage_hashed_cursor,
            &mut cursor_metrics_cache.storage_hashed_cursor,
        );

        let mut v2_account_calculator =
            proof_v2::ProofCalculator::<
                _,
                _,
                AsyncAccountValueEncoder<
                    InstrumentedTrieCursor<
                        '_,
                        <Factory::Provider as TrieCursorFactory>::StorageTrieCursor<'_>,
                    >,
                    InstrumentedHashedCursor<
                        '_,
                        <Factory::Provider as HashedCursorFactory>::StorageCursor<'_>,
                    >,
                >,
            >::new(instrumented_account_trie_cursor, instrumented_account_hashed_cursor);
        let v2_storage_calculator =
            Rc::new(RefCell::new(proof_v2::StorageProofCalculator::new_storage(
                instrumented_storage_trie_cursor,
                instrumented_storage_hashed_cursor,
            )));

        // Count this worker as available only after successful initialization.
        self.availability.mark_idle(self.worker_id);

        let mut total_idle_time = Duration::ZERO;
        let mut idle_start = Instant::now();
        let mut value_encoder_stats_cache = ValueEncoderStats::default();

        while let Ok(job) = self.work_rx.recv() {
            total_idle_time += idle_start.elapsed();

            // Mark worker as busy.
            self.availability.mark_busy(self.worker_id);

            #[cfg(feature = "trie-debug")]
            if let Some(max_jitter) = self.task_ctx.proof_jitter {
                let jitter =
                    Duration::from_nanos(rand::random_range(0..=max_jitter.as_nanos() as u64));
                trace!(
                    target: "trie::proof_task",
                    worker_id = self.worker_id,
                    jitter_us = jitter.as_micros(),
                    "Account worker applying proof jitter"
                );
                std::thread::sleep(jitter);
            }

            match job {
                AccountWorkerJob::AccountMultiproof { input } => {
                    let value_encoder_stats = self.process_account_multiproof::<Factory::Provider>(
                        &mut v2_account_calculator,
                        v2_storage_calculator.clone(),
                        *input,
                        &mut account_proofs_processed,
                    );
                    total_idle_time += value_encoder_stats.storage_wait_time;
                    value_encoder_stats_cache.extend(&value_encoder_stats);
                }
            }

            // Mark worker as available again.
            self.availability.mark_idle(self.worker_id);

            idle_start = Instant::now();
        }

        // Drop calculators to release mutable borrows on cursor_metrics_cache.
        drop(v2_account_calculator);
        drop(v2_storage_calculator);

        trace!(
            target: "trie::proof_task",
            worker_id=self.worker_id,
            account_proofs_processed,
            total_idle_time_us = total_idle_time.as_micros(),
            "Account worker shutting down"
        );

        #[cfg(feature = "metrics")]
        {
            self.metrics.record_account_worker_idle_time(total_idle_time);
            self.cursor_metrics.record(&mut cursor_metrics_cache);
            self.metrics.record_value_encoder_stats(&value_encoder_stats_cache);
        }

        Ok(())
    }

    fn compute_v2_account_multiproof<'a, Provider>(
        &self,
        v2_account_calculator: &mut V2AccountProofCalculator<'a, Provider>,
        v2_storage_calculator: Rc<RefCell<V2StorageProofCalculator<'a, Provider>>>,
        targets: MultiProofTargetsV2,
    ) -> Result<(DecodedMultiProofV2, ValueEncoderStats), ParallelStateRootError>
    where
        Provider: TrieCursorFactory + HashedCursorFactory + 'a,
    {
        let MultiProofTargetsV2 { mut account_targets, storage_targets } = targets;

        let span = debug_span!(
            target: "trie::proof_task",
            "Account V2 multiproof calculation",
            account_targets = account_targets.len(),
            storage_targets = storage_targets.values().map(|t| t.len()).sum::<usize>(),
        );
        let _span_guard = span.enter();

        trace!(target: "trie::proof_task", "Processing V2 account multiproof");

        let storage_proof_receivers =
            dispatch_v2_storage_proofs(&self.storage_work_tx, &account_targets, storage_targets)?;

        let mut value_encoder = AsyncAccountValueEncoder::new(
            storage_proof_receivers,
            self.cached_storage_roots.clone(),
            v2_storage_calculator,
        );

        let account_proofs =
            v2_account_calculator.proof(&mut value_encoder, &mut account_targets)?;

        let (storage_proofs, value_encoder_stats) = value_encoder.finalize()?;

        let proof = DecodedMultiProofV2 { account_proofs, storage_proofs };

        Ok((proof, value_encoder_stats))
    }

    /// Processes an account multiproof request.
    ///
    /// Returns stats from the value encoder used during proof computation.
    fn process_account_multiproof<'a, Provider>(
        &self,
        v2_account_calculator: &mut V2AccountProofCalculator<'a, Provider>,
        v2_storage_calculator: Rc<RefCell<V2StorageProofCalculator<'a, Provider>>>,
        input: AccountMultiproofInput,
        account_proofs_processed: &mut u64,
    ) -> ValueEncoderStats
    where
        Provider: TrieCursorFactory + HashedCursorFactory + 'a,
    {
        let proof_start = Instant::now();

        let AccountMultiproofInput { targets, proof_result_sender } = input;
        let (result, value_encoder_stats) = match self.compute_v2_account_multiproof::<Provider>(
            v2_account_calculator,
            v2_storage_calculator,
            targets,
        ) {
            Ok((proof, stats)) => (Ok(proof), stats),
            Err(e) => (Err(e), ValueEncoderStats::default()),
        };

        let ProofResultContext { sender: result_tx, state, start_time: start } =
            proof_result_sender;

        let proof_elapsed = proof_start.elapsed();
        let total_elapsed = start.elapsed();
        *account_proofs_processed += 1;

        // Send result to SparseTrieCacheTask
        if result_tx.send(ProofResultMessage { result, elapsed: total_elapsed, state }).is_err() {
            trace!(
                target: "trie::proof_task",
                worker_id=self.worker_id,
                account_proofs_processed,
                "Account multiproof receiver dropped, discarding result"
            );
        }

        trace!(
            target: "trie::proof_task",
            proof_time_us = proof_elapsed.as_micros(),
            total_elapsed_us = total_elapsed.as_micros(),
            total_processed = account_proofs_processed,
            "Account multiproof completed"
        );

        value_encoder_stats
    }
}

/// Queues V2 storage proofs for all accounts in the targets and returns receivers.
///
/// This function queues all storage proof tasks to the worker pool but returns immediately
/// with receivers, allowing the account trie walk to proceed in parallel with storage proof
/// computation. This enables interleaved parallelism for better performance.
///
/// Propagates errors up if queuing fails. Receivers must be consumed by the caller.
fn dispatch_v2_storage_proofs(
    storage_work_tx: &CrossbeamSender<StorageWorkerJob>,
    account_targets: &[ProofV2Target],
    mut storage_targets: B256Map<Vec<ProofV2Target>>,
) -> Result<B256Map<CrossbeamReceiver<StorageProofResultMessage>>, ParallelStateRootError> {
    let mut storage_proof_receivers =
        B256Map::with_capacity_and_hasher(account_targets.len(), Default::default());

    // Collect hashed addresses from account targets that need their storage roots computed
    let account_target_addresses: B256Set = account_targets.iter().map(|t| t.key()).collect();

    // For storage targets with associated account proofs, ensure the first target has
    // min_len(0) so the root node is returned for storage root computation
    for (hashed_address, targets) in &mut storage_targets {
        if account_target_addresses.contains(hashed_address) &&
            let Some(first) = targets.first_mut()
        {
            *first = first.with_min_len(0);
        }
    }

    // Sort storage targets by address for optimal dispatch order.
    // Since trie walk processes accounts in lexicographical order, dispatching in the same order
    // reduces head-of-line blocking when consuming results.
    let mut sorted_storage_targets: Vec<_> = storage_targets.into_iter().collect();
    sorted_storage_targets.sort_unstable_by_key(|(addr, _)| *addr);

    // Dispatch all proofs for targeted storage slots
    for (hashed_address, targets) in sorted_storage_targets {
        // Create channel for receiving StorageProofResultMessage
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let input = StorageProofInput::new(hashed_address, targets);

        storage_work_tx
            .send(StorageWorkerJob::StorageProof { input, proof_result_sender: result_tx })
            .map_err(|_| {
                ParallelStateRootError::Other(format!(
                    "Failed to queue storage proof for {hashed_address:?}: storage worker pool unavailable",
                ))
            })?;

        storage_proof_receivers.insert(hashed_address, result_rx);
    }

    Ok(storage_proof_receivers)
}

/// Input parameters for storage proof computation.
#[derive(Debug)]
pub struct StorageProofInput {
    /// The hashed address for which the proof is calculated.
    pub hashed_address: B256,
    /// The set of proof targets
    pub targets: Vec<ProofV2Target>,
}

impl StorageProofInput {
    /// Creates a new [`StorageProofInput`] with the given hashed address and target slots.
    pub const fn new(hashed_address: B256, targets: Vec<ProofV2Target>) -> Self {
        Self { hashed_address, targets }
    }
}

/// Input parameters for account multiproof computation.
#[derive(Debug)]
pub struct AccountMultiproofInput {
    /// The targets for which to compute the multiproof.
    pub targets: MultiProofTargetsV2,
    /// Context for sending the proof result.
    pub proof_result_sender: ProofResultContext,
}

impl AccountMultiproofInput {
    /// Returns the [`ProofResultContext`] for this input, consuming the input.
    fn into_proof_result_sender(self) -> ProofResultContext {
        self.proof_result_sender
    }
}

/// Internal message for account workers.
#[derive(Debug)]
enum AccountWorkerJob {
    /// Account multiproof computation request
    AccountMultiproof {
        /// Account multiproof input parameters
        input: Box<AccountMultiproofInput>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{keccak256, Address, U256};
    use reth_chainspec::{ChainSpec, EthChainSpec};
    use reth_ethereum_primitives::{Block, BlockBody};
    use reth_primitives_traits::{Account, RecoveredBlock, SealedBlock, StorageEntry};
    use reth_provider::{
        test_utils::create_test_provider_factory_with_chain_spec, BlockWriter, ExecutionOutcome,
        HashingWriter,
    };
    use reth_tasks::{RayonConfig, RuntimeBuilder, RuntimeConfig};
    use reth_trie::TrieInput;
    use std::sync::Arc;

    fn test_ctx<Factory>(factory: Factory) -> ProofTaskCtx<Factory> {
        ProofTaskCtx::new(factory)
    }

    #[derive(Clone)]
    struct FailingFactory<Factory>(Factory);

    impl<Factory> DatabaseProviderROFactory for FailingFactory<Factory>
    where
        Factory: DatabaseProviderROFactory,
    {
        type Provider = Factory::Provider;

        fn database_provider_ro(&self) -> ProviderResult<Self::Provider> {
            // Keep the wrapped factory in the type so its concrete provider type remains available
            // without ever opening a database transaction.
            let _ = &self.0;
            Err(ProviderError::other(std::io::Error::other(
                "injected provider initialization failure",
            )))
        }
    }

    /// Ensures `ProofWorkerHandle::new` spawns workers correctly.
    #[test]
    fn spawn_proof_workers_creates_handle() {
        let chain_spec = Arc::new(ChainSpec::default());
        let anchor_hash = chain_spec.genesis_hash();
        let provider_factory = create_test_provider_factory_with_chain_spec(chain_spec);
        let changeset_cache = reth_trie_db::ChangesetCache::new();
        let factory = reth_provider::providers::OverlayStateProviderFactory::new(
            provider_factory,
            reth_provider::providers::OverlayBuilder::<reth_ethereum_primitives::EthPrimitives>::new(
                anchor_hash,
                changeset_cache,
            ),
        );
        let ctx = test_ctx(factory);

        let rayon = RayonConfig::default()
            .with_proof_storage_worker_threads(1)
            .with_proof_account_worker_threads(1);
        let runtime =
            RuntimeBuilder::new(RuntimeConfig::default().with_rayon(rayon)).build().unwrap();
        let proof_handle = ProofWorkerHandle::new(&runtime, ctx, true);

        // Verify handle can be cloned
        let _cloned_handle = proof_handle.clone();
        assert_eq!(proof_handle.total_storage_workers(), 1);
        assert_eq!(proof_handle.total_account_workers(), 1);

        // Workers shut down automatically when handle is dropped
        drop(proof_handle);
    }

    #[test]
    fn one_shot_parallel_multiproof_matches_serial() {
        let chain_spec = Arc::new(ChainSpec::default());
        let anchor_hash = chain_spec.genesis_hash();
        let provider_factory =
            create_test_provider_factory_with_chain_spec(Arc::clone(&chain_spec));
        let address_a = Address::repeat_byte(0x11);
        let address_b = Address::repeat_byte(0x12);
        let slot_a = B256::repeat_byte(0x21);
        let slot_b = B256::repeat_byte(0x22);

        {
            let provider_rw = provider_factory.provider_rw().unwrap();
            let genesis_block = RecoveredBlock::new_sealed(
                SealedBlock::<Block>::seal_parts(
                    chain_spec.genesis_header().clone(),
                    BlockBody::default(),
                ),
                vec![],
            );
            provider_rw
                .append_blocks_with_state(
                    vec![genesis_block],
                    &ExecutionOutcome::default(),
                    Default::default(),
                )
                .unwrap();
            provider_rw
                .insert_account_for_hashing([
                    (
                        address_a,
                        Some(Account { nonce: 1, balance: U256::from(2), bytecode_hash: None }),
                    ),
                    (
                        address_b,
                        Some(Account { nonce: 2, balance: U256::from(3), bytecode_hash: None }),
                    ),
                ])
                .unwrap();
            provider_rw
                .insert_storage_for_hashing([
                    (address_a, [StorageEntry { key: slot_a, value: U256::from(4) }]),
                    (address_b, [StorageEntry { key: slot_b, value: U256::from(5) }]),
                ])
                .unwrap();
            provider_rw.commit().unwrap();
        }

        let targets = move || MultiProofTargetsV2 {
            account_targets: vec![
                ProofV2Target::new(keccak256(address_a)),
                ProofV2Target::new(keccak256(address_b)),
            ],
            storage_targets: B256Map::from_iter([
                (keccak256(address_a), vec![ProofV2Target::new(keccak256(slot_a))]),
                (keccak256(address_b), vec![ProofV2Target::new(keccak256(slot_b))]),
            ]),
        };
        let serial = provider_factory
            .latest()
            .unwrap()
            .multiproof_v2(TrieInput::default(), targets())
            .unwrap();
        let overlay_factory = reth_provider::providers::OverlayStateProviderFactory::new(
            provider_factory,
            reth_provider::providers::OverlayBuilder::<
                reth_ethereum_primitives::EthPrimitives,
            >::new(anchor_hash, reth_trie_db::ChangesetCache::new()),
        );

        let rayon = RayonConfig::default()
            .with_proof_storage_worker_threads(4)
            .with_proof_account_worker_threads(2);
        let runtime =
            RuntimeBuilder::new(RuntimeConfig::default().with_rayon(rayon)).build().unwrap();
        let (parallel, worker_stats) =
            parallel_multiproof_v2_with_stats(&runtime, overlay_factory.clone(), targets(), false)
                .unwrap();
        assert_eq!(parallel, serial);
        assert_eq!(worker_stats.storage_workers, 2);
        assert_eq!(worker_stats.account_workers, 1);

        // A normal Engine proof handle may keep every Engine proof worker idle while it waits for
        // state updates. The isolated one-shot pools must still let this proof finish promptly.
        let engine_handle =
            ProofWorkerHandle::new(&runtime, ProofTaskCtx::new(overlay_factory.clone()), false);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !engine_handle
            .storage_availability
            .flags
            .iter()
            .all(|flag| flag.load(Ordering::Relaxed)) ||
            !engine_handle
                .account_availability
                .flags
                .iter()
                .all(|flag| flag.load(Ordering::Relaxed))
        {
            assert!(std::time::Instant::now() < deadline, "Engine proof workers did not start");
            std::thread::yield_now();
        }

        let (isolated_tx, isolated_rx) = std::sync::mpsc::channel();
        let isolated_runtime = runtime.clone();
        let isolated_factory = overlay_factory.clone();
        let isolated_targets = targets();
        let isolated = std::thread::spawn(move || {
            let result = parallel_multiproof_v2(
                &isolated_runtime,
                isolated_factory,
                isolated_targets,
                false,
            );
            let _ = isolated_tx.send(result);
        });
        let isolated_result = isolated_rx.recv_timeout(Duration::from_secs(5));
        engine_handle.shutdown_and_wait();
        isolated.join().unwrap();
        assert_eq!(
            isolated_result
                .expect("one-shot proof should not queue behind an idle Engine proof handle")
                .unwrap(),
            serial
        );

        // Concurrent proof handles are serialized in the same order on both coordinator lanes.
        // Both calls must complete rather than each holding one pool while waiting for the other.
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();
        let first_runtime = runtime.clone();
        let first_factory = overlay_factory.clone();
        let first_targets = targets();
        let first_tx = completion_tx.clone();
        let first = std::thread::spawn(move || {
            let result =
                parallel_multiproof_v2(&first_runtime, first_factory, first_targets, false);
            let _ = first_tx.send(result);
        });
        let second_runtime = runtime;
        let second_factory = overlay_factory;
        let second_targets = targets();
        let second = std::thread::spawn(move || {
            let result =
                parallel_multiproof_v2(&second_runtime, second_factory, second_targets, false);
            let _ = completion_tx.send(result);
        });

        let first_result = completion_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("first concurrent one-shot proof should complete")
            .unwrap();
        let second_result = completion_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("second concurrent one-shot proof should complete")
            .unwrap();
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(first_result, serial);
        assert_eq!(second_result, serial);
    }

    #[test]
    fn one_shot_provider_initialization_failure_returns_promptly() {
        let chain_spec = Arc::new(ChainSpec::default());
        let anchor_hash = chain_spec.genesis_hash();
        let provider_factory =
            create_test_provider_factory_with_chain_spec(Arc::clone(&chain_spec));
        let overlay_factory = reth_provider::providers::OverlayStateProviderFactory::new(
            provider_factory,
            reth_provider::providers::OverlayBuilder::<
                reth_ethereum_primitives::EthPrimitives,
            >::new(anchor_hash, reth_trie_db::ChangesetCache::new()),
        );
        let failing_factory = FailingFactory(overlay_factory);
        let targets = MultiProofTargetsV2 {
            account_targets: vec![ProofV2Target::new(keccak256(Address::repeat_byte(0x44)))],
            storage_targets: B256Map::default(),
        };
        let rayon = RayonConfig::default()
            .with_proof_storage_worker_threads(2)
            .with_proof_account_worker_threads(2);
        let runtime =
            RuntimeBuilder::new(RuntimeConfig::default().with_rayon(rayon)).build().unwrap();

        let (completion_tx, completion_rx) = std::sync::mpsc::channel();
        let proof = std::thread::spawn(move || {
            let result = parallel_multiproof_v2(&runtime, failing_factory, targets, false);
            let _ = completion_tx.send(result);
        });
        let result = completion_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("provider initialization failure must not hang the one-shot proof");
        proof.join().unwrap();
        assert!(matches!(result, Err(ParallelStateRootError::Provider(_))));
    }
}
