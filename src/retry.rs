//! Generic retry utility with exponential backoff.
//!
//! Provides a simple, reusable mechanism for retrying fallible operations
//! with configurable limits and backoff. Used for XWM reconnection and
//! any other subsystem that should survive transient failures.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tracing::{error, info, warn};

/// Configuration for retry behavior.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Maximum number of consecutive retry attempts before giving up.
    pub max_retries: u32,
    /// Initial delay between retries. Doubles after each attempt.
    pub initial_backoff: Duration,
    /// Upper bound on backoff delay (prevents unbounded growth).
    pub max_backoff: Duration,
}

impl RetryPolicy {
    /// Sensible defaults: 5 retries, 100ms → 1.6s exponential backoff.
    pub const DEFAULT: Self = Self {
        max_retries: 5,
        initial_backoff: Duration::from_millis(100),
        max_backoff: Duration::from_secs(5),
    };

    /// Compute the backoff duration for a given attempt (0-indexed).
    #[inline(always)]
    fn backoff_for(&self, attempt: u32) -> Duration {
        let backoff = self.initial_backoff.saturating_mul(1 << attempt.min(16));
        backoff.min(self.max_backoff)
    }
}

/// Run `f` repeatedly until it succeeds or the retry policy is exhausted.
///
/// - On `Ok(())`: returns immediately (clean exit).
/// - On `Err`: logs a warning, sleeps with exponential backoff, and retries.
/// - Checks `shutdown` between retries; if set, stops early.
///
/// Returns `Ok(())` if `f` ever returned `Ok`, or the last `Err` if all
/// retries were exhausted.
pub fn retry_with_backoff<F>(
    label: &str,
    policy: &RetryPolicy,
    shutdown: &AtomicBool,
    mut f: F,
) -> anyhow::Result<()>
where
    F: FnMut() -> anyhow::Result<()>,
{
    let mut attempt = 0u32;

    loop {
        match f() {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempt += 1;
                if attempt > policy.max_retries {
                    error!(
                        error = ?e,
                        attempts = attempt,
                        "{label}: failed after {attempt} attempts, giving up",
                    );
                    return Err(e);
                }

                let backoff = policy.backoff_for(attempt - 1);
                warn!(
                    error = ?e,
                    attempt,
                    max = policy.max_retries,
                    backoff_ms = backoff.as_millis() as u64,
                    "{label}: failed, retrying",
                );

                std::thread::sleep(backoff);

                if !shutdown.load(Ordering::Relaxed) {
                    info!("{label}: compositor shutting down, not retrying");
                    return Err(e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn succeeds_immediately() {
        let shutdown = AtomicBool::new(true);
        let policy = RetryPolicy::DEFAULT;
        let result = retry_with_backoff("test", &policy, &shutdown, || Ok(()));
        assert!(result.is_ok());
    }

    #[test]
    fn retries_then_succeeds() {
        let shutdown = AtomicBool::new(true);
        let policy = RetryPolicy {
            max_retries: 3,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(10),
        };
        let mut calls = 0u32;
        let result = retry_with_backoff("test", &policy, &shutdown, || {
            calls += 1;
            if calls < 3 {
                anyhow::bail!("not yet");
            }
            Ok(())
        });
        assert!(result.is_ok());
        assert_eq!(calls, 3);
    }

    #[test]
    fn exhausts_retries() {
        let shutdown = AtomicBool::new(true);
        let policy = RetryPolicy {
            max_retries: 2,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(10),
        };
        let result = retry_with_backoff("test", &policy, &shutdown, || {
            anyhow::bail!("always fails");
        });
        assert!(result.is_err());
    }

    #[test]
    fn backoff_capped_at_max() {
        let policy = RetryPolicy {
            max_retries: 10,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(1),
        };
        // Attempt 20 would be 100ms * 2^20 = ~104s, but capped at 1s.
        assert_eq!(policy.backoff_for(20), Duration::from_secs(1));
    }
}
