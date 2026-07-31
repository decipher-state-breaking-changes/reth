//! A map of named single-thread worker pools.
//!
//! Each worker is a dedicated OS thread that processes closures sent to it via a channel.
//! This is a substitute for `spawn_blocking` that reuses the same OS thread for the same
//! named task, like a 1-thread thread pool keyed by name.

use dashmap::DashMap;
use std::{panic::AssertUnwindSafe, sync::Mutex, thread};
use tokio::sync::{mpsc, oneshot};

type BoxedTask = Box<dyn FnOnce() + Send + 'static>;

/// A single-thread worker that processes closures sequentially on a dedicated OS thread.
struct WorkerThread {
    /// Sender to submit work to this worker's thread.
    tx: mpsc::UnboundedSender<BoxedTask>,
    /// The OS thread handle. Taken during shutdown to join.
    handle: Option<thread::JoinHandle<()>>,
}

impl WorkerThread {
    /// Spawns a new worker thread with the given name.
    fn new(name: &'static str) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<BoxedTask>();
        let handle = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Some(task) = rx.blocking_recv() {
                    let _ = std::panic::catch_unwind(AssertUnwindSafe(task));
                }
            })
            .unwrap_or_else(|e| panic!("failed to spawn worker thread {name:?}: {e}"));

        Self { tx, handle: Some(handle) }
    }
}

/// A map of named single-thread workers.
///
/// Each unique name gets a dedicated OS thread that is reused for all tasks submitted under
/// that name. Workers are created lazily on first use.
pub(crate) struct WorkerMap {
    workers: DashMap<&'static str, WorkerThread>,
    /// Serializes queue insertion across named workers.
    ///
    /// Individual tasks only hold this while cloning a sender and enqueueing. Paired submissions
    /// use the same lock to ensure no other task can be inserted between the two queue positions.
    enqueue_lock: Mutex<()>,
}

impl Default for WorkerMap {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerMap {
    /// Creates a new empty `WorkerMap`.
    pub(crate) fn new() -> Self {
        Self { workers: DashMap::new(), enqueue_lock: Mutex::new(()) }
    }

    /// Returns a sender for the named worker, creating the worker if needed.
    fn sender(&self, name: &'static str) -> mpsc::UnboundedSender<BoxedTask> {
        self.workers.entry(name).or_insert_with(|| WorkerThread::new(name)).tx.clone()
    }

    /// Spawns a closure on the dedicated worker thread for the given name.
    ///
    /// If no worker thread exists for this name yet, one is created with the given name as
    /// the OS thread name. The closure executes on the worker's OS thread and the returned
    /// future resolves with the result.
    pub(crate) fn spawn_on<F, R>(&self, name: &'static str, f: F) -> oneshot::Receiver<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (result_tx, result_rx) = oneshot::channel();

        let task: BoxedTask = Box::new(move || {
            let _ = result_tx.send(f());
        });

        let _enqueue_guard = self.enqueue_lock.lock().expect("worker enqueue lock poisoned");
        let _ = self.sender(name).send(task);

        result_rx
    }

    /// Atomically queues two closures on their respective named worker threads.
    ///
    /// Both tasks are inserted while holding the same enqueue lock used by [`Self::spawn_on`].
    /// This keeps their relative position aligned across independent FIFO worker queues, which is
    /// required when the tasks cooperate and remain alive until both have started. The names must
    /// be distinct so both tasks can start concurrently.
    pub(crate) fn spawn_on_pair<F1, R1, F2, R2>(
        &self,
        first_name: &'static str,
        first: F1,
        second_name: &'static str,
        second: F2,
    ) -> (oneshot::Receiver<R1>, oneshot::Receiver<R2>)
    where
        F1: FnOnce() -> R1 + Send + 'static,
        R1: Send + 'static,
        F2: FnOnce() -> R2 + Send + 'static,
        R2: Send + 'static,
    {
        let (first_result_tx, first_result_rx) = oneshot::channel();
        assert_ne!(first_name, second_name, "paired worker names must be distinct");
        let (second_result_tx, second_result_rx) = oneshot::channel();

        let first_task: BoxedTask = Box::new(move || {
            let _ = first_result_tx.send(first());
        });
        let second_task: BoxedTask = Box::new(move || {
            let _ = second_result_tx.send(second());
        });

        let _enqueue_guard = self.enqueue_lock.lock().expect("worker enqueue lock poisoned");
        let first_tx = self.sender(first_name);
        let second_tx = self.sender(second_name);
        let _ = first_tx.send(first_task);
        let _ = second_tx.send(second_task);

        (first_result_rx, second_result_rx)
    }
}

impl Drop for WorkerMap {
    fn drop(&mut self) {
        for (_, mut w) in std::mem::take(&mut self.workers) {
            // Drop sender so the thread's recv loop exits, then join.
            drop(w.tx);
            if let Some(handle) = w.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

impl std::fmt::Debug for WorkerMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerMap").field("num_workers", &self.workers.len()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn worker_map_basic() {
        let map = WorkerMap::new();

        let result = map.spawn_on("test", || 42).await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn worker_map_same_thread() {
        let map = WorkerMap::new();

        let id1 = map.spawn_on("test", || thread::current().id()).await.unwrap();
        let id2 = map.spawn_on("test", || thread::current().id()).await.unwrap();
        assert_eq!(id1, id2, "same name should run on the same thread");
    }

    #[tokio::test]
    async fn worker_map_different_names_different_threads() {
        let map = WorkerMap::new();

        let id1 = map.spawn_on("worker-a", || thread::current().id()).await.unwrap();
        let id2 = map.spawn_on("worker-b", || thread::current().id()).await.unwrap();
        assert_ne!(id1, id2, "different names should run on different threads");
    }

    #[tokio::test]
    async fn worker_map_sequential_execution() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let map = WorkerMap::new();
        let counter = Arc::new(AtomicUsize::new(0));

        let mut receivers = Vec::new();
        for i in 0..10 {
            let c = counter.clone();
            let rx = map.spawn_on("sequential", move || {
                let val = c.fetch_add(1, Ordering::SeqCst);
                assert_eq!(val, i, "tasks should execute in order");
                val
            });
            receivers.push(rx);
        }

        for (i, rx) in receivers.into_iter().enumerate() {
            let val = rx.await.unwrap();
            assert_eq!(val, i);
        }
    }

    #[tokio::test]
    async fn worker_map_thread_name() {
        let map = WorkerMap::new();

        let name = map
            .spawn_on("custom-worker", || thread::current().name().unwrap().to_string())
            .await
            .unwrap();
        assert_eq!(name, "custom-worker");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn worker_map_pair_submission_keeps_lane_order_aligned() {
        use std::{
            sync::{mpsc as std_mpsc, Arc, Barrier, Condvar},
            time::Duration,
        };

        fn blocking_task(
            pair: usize,
            lane: &'static str,
            started_tx: std_mpsc::Sender<(usize, &'static str)>,
            release: Arc<(Mutex<bool>, Condvar)>,
        ) -> impl FnOnce() + Send + 'static {
            move || {
                started_tx.send((pair, lane)).unwrap();
                let (lock, cvar) = &*release;
                drop(cvar.wait_while(lock.lock().unwrap(), |released| !*released).unwrap());
            }
        }

        let map = Arc::new(WorkerMap::new());
        let start = Arc::new(Barrier::new(3));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let (started_tx, started_rx) = std_mpsc::channel();

        let first = {
            let map = Arc::clone(&map);
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let started_tx = started_tx.clone();
            thread::spawn(move || {
                start.wait();
                map.spawn_on_pair(
                    "pair-lane-a",
                    blocking_task(1, "a", started_tx.clone(), Arc::clone(&release)),
                    "pair-lane-b",
                    blocking_task(1, "b", started_tx, release),
                )
            })
        };
        let second = {
            let map = Arc::clone(&map);
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let started_tx = started_tx.clone();
            thread::spawn(move || {
                start.wait();
                map.spawn_on_pair(
                    "pair-lane-b",
                    blocking_task(2, "b", started_tx.clone(), Arc::clone(&release)),
                    "pair-lane-a",
                    blocking_task(2, "a", started_tx, release),
                )
            })
        };

        start.wait();
        let first_handles = first.join().unwrap();
        let second_handles = second.join().unwrap();

        let first_started = started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let second_started = started_rx.recv_timeout(Duration::from_secs(2)).unwrap();

        // Release every task before asserting so a regression cannot leave worker threads blocked
        // during test cleanup.
        {
            let (lock, cvar) = &*release;
            *lock.lock().unwrap() = true;
            cvar.notify_all();
        }

        first_handles.0.await.unwrap();
        first_handles.1.await.unwrap();
        second_handles.0.await.unwrap();
        second_handles.1.await.unwrap();

        assert_eq!(
            first_started.0, second_started.0,
            "the first task on each lane must belong to the same paired submission"
        );
        assert_ne!(first_started.1, second_started.1);
    }
}
