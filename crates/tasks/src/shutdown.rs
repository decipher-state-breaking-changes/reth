//! Helper for shutdown signals

use futures_util::{
    future::{FusedFuture, Shared},
    FutureExt,
};
use std::{
    future::Future,
    pin::Pin,
    sync::{atomic::AtomicUsize, Arc},
    task::{ready, Context, Poll},
};
use tokio::sync::oneshot;

/// A Future that resolves when the shutdown event has been fired.
#[derive(Debug)]
pub struct GracefulShutdown {
    shutdown: Shutdown,
    guard: Option<GracefulShutdownGuard>,
}

impl GracefulShutdown {
    pub(crate) const fn new(shutdown: Shutdown, guard: GracefulShutdownGuard) -> Self {
        Self { shutdown, guard: Some(guard) }
    }

    /// Returns a new shutdown future that ignores the returned [`GracefulShutdownGuard`].
    ///
    /// This just maps the return value of the future to `()`, it does not drop the guard.
    pub fn ignore_guard(self) -> impl Future<Output = ()> + Send + Sync + Unpin + 'static {
        self.map(drop)
    }
}

impl Future for GracefulShutdown {
    type Output = GracefulShutdownGuard;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        ready!(self.shutdown.poll_unpin(cx));
        Poll::Ready(self.get_mut().guard.take().expect("Future polled after completion"))
    }
}

impl Clone for GracefulShutdown {
    fn clone(&self) -> Self {
        Self {
            shutdown: self.shutdown.clone(),
            guard: self.guard.as_ref().map(|g| GracefulShutdownGuard::new(Arc::clone(&g.0))),
        }
    }
}

/// A guard that fires once dropped to signal the [`TaskManager`](crate::TaskManager) that the
/// [`GracefulShutdown`] has completed.
#[derive(Debug)]
#[must_use = "if unused the task will not be gracefully shutdown"]
pub struct GracefulShutdownGuard(Arc<AtomicUsize>);

impl GracefulShutdownGuard {
    pub(crate) fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for GracefulShutdownGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// A Future that resolves when the shutdown event has been fired.
#[derive(Debug, Clone)]
pub struct Shutdown(Shared<oneshot::Receiver<()>>);

/// Why the task shutdown future completed.
///
/// A requested shutdown sends a value through the channel. If the task manager itself goes away
/// without requesting shutdown, the sender is dropped instead. Callers that merely need to stop
/// can continue awaiting [`Shutdown`]; callers that must not turn a task-manager failure into a
/// clean shutdown can inspect this cause through [`Shutdown::cause`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownCause {
    /// The runtime or task manager explicitly requested an orderly shutdown.
    Requested,
    /// The shutdown sender disappeared without making that request.
    SenderDropped,
}

impl Shutdown {
    /// Waits for shutdown while preserving whether it was requested or merely lost its sender.
    pub async fn cause(self) -> ShutdownCause {
        match self.0.await {
            Ok(()) => ShutdownCause::Requested,
            Err(_) => ShutdownCause::SenderDropped,
        }
    }
}

impl Future for Shutdown {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let pin = self.get_mut();
        if pin.0.is_terminated() || pin.0.poll_unpin(cx).is_ready() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Shutdown signal that fires either manually or on drop by closing the channel
#[derive(Debug)]
pub struct Signal(oneshot::Sender<()>);

impl Signal {
    /// Fire the signal manually.
    pub fn fire(self) {
        let _ = self.0.send(());
    }
}

/// Create a channel pair that's used to propagate shutdown event
pub fn signal() -> (Signal, Shutdown) {
    let (sender, receiver) = oneshot::channel();
    (Signal(sender), Shutdown(receiver.shared()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::future::join_all;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_shutdown() {
        let (_signal, _shutdown) = signal();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_drop_signal() {
        let (signal, shutdown) = signal();

        tokio::task::spawn(async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            drop(signal)
        });

        shutdown.await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shutdown_cause_distinguishes_a_request_from_a_lost_sender() {
        let (requested_signal, shutdown) = signal();
        requested_signal.fire();
        assert_eq!(shutdown.cause().await, ShutdownCause::Requested);

        let (dropped_signal, shutdown) = signal();
        drop(dropped_signal);
        assert_eq!(shutdown.cause().await, ShutdownCause::SenderDropped);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multi_shutdowns() {
        let (signal, shutdown) = signal();

        let mut tasks = Vec::with_capacity(100);
        for _ in 0..100 {
            let shutdown = shutdown.clone();
            let task = tokio::task::spawn(async move {
                shutdown.await;
            });
            tasks.push(task);
        }

        drop(signal);

        join_all(tasks).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_drop_signal_from_thread() {
        let (signal, shutdown) = signal();

        let _thread = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(500));
            drop(signal)
        });

        shutdown.await;
    }
}
