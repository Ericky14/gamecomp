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
