use super::*;

#[test]
fn pacer_60hz_interval() {
    let pacer = FramePacer::new(60);
    // 60Hz ≈ 16.67ms
    assert!(pacer.refresh_interval_ns() > 16_000_000);
    assert!(pacer.refresh_interval_ns() < 17_000_000);
}

#[test]
fn rolling_peak_decays_toward_steady_state() {
    let mut pacer = FramePacer::new(60);
    // Feed constant draw time well below the initial 3ms peak.
    for _ in 0..200 {
        pacer.update_draw_time(2_000_000); // 2ms
    }
    // Rolling peak should decay close to 2ms (but not exact due to
    // integer arithmetic and slow 98%/2% decay).
    let rolling = pacer.rolling_draw_time_ns();
    assert!(
        rolling > 1_900_000 && rolling < 2_200_000,
        "rolling peak was {rolling}"
    );
}

#[test]
fn rolling_peak_spikes_instantly() {
    let mut pacer = FramePacer::new(60);
    // Settle at 2ms.
    for _ in 0..200 {
        pacer.update_draw_time(2_000_000);
    }
    let before = pacer.rolling_draw_time_ns();
    // Sudden spike to 10ms — well above rolling + red_zone/2.
    pacer.update_draw_time(10_000_000);
    let after = pacer.rolling_draw_time_ns();
    // The peak should jump to exactly the spike value.
    assert_eq!(after, 10_000_000, "should spike instantly");
    assert!(after > before * 3, "spike should be dramatic");
}

#[test]
fn compositing_floor_raises_offset() {
    let mut pacer = FramePacer::new(60);
    // Settle rolling peak to 1ms (below compositing floor).
    for _ in 0..500 {
        pacer.update_draw_time(1_000_000);
    }

    let now = 100_000_000u64;
    pacer.mark_vblank(now);

    // Without compositing: offset uses raw rolling peak (~1ms).
    pacer.set_compositing(false);
    let wake_no_comp = pacer.next_wakeup_ns(now);

    // With compositing: offset uses floor of 2.4ms.
    pacer.set_compositing(true);
    let wake_comp = pacer.next_wakeup_ns(now);

    // Compositing should wake up earlier (lower timestamp = more headroom).
    assert!(
        wake_comp < wake_no_comp,
        "compositing floor should produce earlier wakeup: comp={wake_comp}, no_comp={wake_no_comp}"
    );
}

#[test]
fn vrr_mode_minimal_offset() {
    let mut pacer = FramePacer::new(60);
    pacer.set_vrr(true);
    let now = 1_000_000_000u64;
    let wakeup = pacer.next_wakeup_ns(now);
    // VRR mode should wake up almost immediately.
    assert!(wakeup - now < 1_000_000, "VRR offset too large");
}

#[test]
fn vblank_anchoring() {
    let mut pacer = FramePacer::new(60);
    // Anchor at T=100ms.
    pacer.mark_vblank(100_000_000);
    // Check wakeup at T=115ms (1.67ms before next VBlank at ~116.67ms).
    let wakeup = pacer.next_wakeup_ns(110_000_000);
    // Should wake before the VBlank at T≈116.67ms.
    assert!(
        wakeup < 116_666_667,
        "wakeup time {} is after VBlank",
        wakeup
    );
    assert!(wakeup > 110_000_000, "wakeup {} is in the past", wakeup);
}

// ─── FpsLimiter tests ───────────────────────────────────────────

#[test]
fn limiter_disabled_always_releases() {
    let limiter = FpsLimiter::new(0, 60);
    assert!(limiter.should_release(0));
    assert!(limiter.should_release(1_000_000_000));
}

#[test]
fn limiter_first_frame_always_releases() {
    let limiter = FpsLimiter::new(30, 60);
    assert!(limiter.should_release(0));
}

#[test]
fn limiter_30fps_blocks_before_interval() {
    let mut limiter = FpsLimiter::new(30, 60);
    // At 30 FPS, interval = 33_333_333 ns (~33.33ms).
    let t0 = 100_000_000u64; // 100ms
    limiter.mark_released(t0);

    // 10ms later — too soon.
    assert!(!limiter.should_release(t0 + 10_000_000));
    // 20ms later — still too soon.
    assert!(!limiter.should_release(t0 + 20_000_000));
    // 33ms later — still slightly too early (interval is 33.33ms, fudge is 0.2ms).
    assert!(!limiter.should_release(t0 + 33_000_000));
    // 33.2ms later — within fudge of 33.33ms deadline.
    assert!(limiter.should_release(t0 + 33_200_000));
    // 34ms later — past deadline.
    assert!(limiter.should_release(t0 + 34_000_000));
}

#[test]
fn limiter_60fps_interval() {
    let limiter = FpsLimiter::new(60, 60);
    // 60 FPS = ~16.67ms interval.
    assert!(limiter.next_release_ns().is_none()); // No reference yet.

    let mut limiter = FpsLimiter::new(60, 60);
    limiter.mark_released(1); // Use nonzero to avoid "first frame" path.
    let next = limiter.next_release_ns().unwrap();
    // next = 1 + 16_666_666
    let interval = next - 1;
    assert!(
        interval > 16_000_000 && interval < 17_000_000,
        "interval was {interval}"
    );
}

#[test]
fn limiter_time_until_release() {
    let mut limiter = FpsLimiter::new(60, 60);
    limiter.mark_released(100_000_000);
    let wait = limiter.time_until_release(100_000_000);
    // Should wait ~16.67ms.
    assert!(wait.as_micros() > 16_000 && wait.as_micros() < 17_000);
}

#[test]
fn limiter_runtime_update() {
    let mut limiter = FpsLimiter::new(30, 60);
    limiter.set_target_fps(60);
    assert_eq!(limiter.target_fps(), 60);
    limiter.mark_released(1); // Nonzero to avoid "first frame" path.
    let next = limiter.next_release_ns().unwrap();
    let interval = next - 1;
    assert!(interval > 16_000_000 && interval < 17_000_000);
}

#[test]
fn limiter_pending_tracking() {
    let mut limiter = FpsLimiter::new(30, 60);
    assert!(!limiter.has_pending());
    limiter.add_pending();
    limiter.add_pending();
    assert!(limiter.has_pending());
    limiter.clear_pending();
    assert!(!limiter.has_pending());
}
