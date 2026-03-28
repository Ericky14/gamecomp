//! Adaptive frame pacing and VBlank scheduling.
//!
//! The frame pacer determines *when* to wake up the compositor to begin
//! rendering, so that composition completes just before the display's
//! VBlank deadline. This minimizes both latency and missed frames.
//!
//! Two scheduling modes:
//!
//! - **Fixed refresh**: Uses a rolling peak draw time with asymmetric decay
//!   plus a configurable "red zone" safety margin. The peak spikes instantly
//!   on load increases and decays slowly (98% old / 2% new per frame),
//!   providing worst-case protection with minimal overshoot. Wakes up
//!   `offset` nanoseconds before the predicted VBlank, where
//!   `offset = rolling_peak_draw_time + red_zone`.
//!
//! - **VRR (variable refresh rate)**: Minimal offset — present as soon as
//!   composition finishes. The display syncs to the compositor, not vice versa.
//!
//! The pacer uses `timerfd` for sub-millisecond wakeup precision. When a page
//! flip completes, the DRM thread signals the pacer to re-anchor its VBlank
//! prediction.

use std::time::{Duration, Instant};

use tracing::{debug, trace};

/// Default red zone (safety margin) in nanoseconds.
/// Provides a buffer for scheduling jitter and draw time variance.
const DEFAULT_RED_ZONE_NS: u64 = 1_650_000; // 1.65ms

/// Default initial draw time estimate in nanoseconds.
const DEFAULT_DRAW_TIME_NS: u64 = 3_000_000; // 3ms

/// Minimum draw time floor when compositing (vs direct scanout).
/// Prevents GPU clock feedback loops when the compositor is blending.
const MIN_COMPOSITING_DRAW_TIME_NS: u64 = 2_400_000; // 2.4ms

/// Minimum VBlank time accounting for untrackable `drm_commit` latency.
const MIN_VBLANK_TIME_NS: u64 = 350_000; // 0.35ms

/// Minimum composition time in nanoseconds (for VRR flushing).
const VRR_FLUSHING_TIME_NS: u64 = 300_000; // 0.3ms

/// Rate of decay for the rolling peak draw time, as parts per thousand.
/// 980 = 98% old + 2% new — slow decay preserves worst-case protection.
const RATE_OF_DECAY_PER_MILLE: u64 = 980;

/// Maximum value for the decay denominator (100% = 1000).
const RATE_OF_DECAY_MAX: u64 = 1000;

/// Adaptive frame pacer.
///
/// Tracks draw time history and VBlank timing to schedule compositor
/// wakeups at the optimal moment before each VBlank deadline.
///
/// Uses a rolling-peak-with-asymmetric-decay algorithm:
/// - **Spike-up**: If the latest draw time exceeds the rolling peak by
///   more than `red_zone / 2`, the peak jumps instantly to the new value.
///   This prevents missed VBlanks on sudden load increases.
/// - **Decay**: Otherwise the peak decays slowly toward the current draw
///   time at 98%/2% per frame, preserving worst-case headroom.
pub struct FramePacer {
    /// Rolling peak draw time (nanoseconds) — the primary scheduling input.
    /// Asymmetrically tracks worst-case draw time: instant spike-up, slow decay.
    rolling_max_draw_time_ns: u64,
    /// Last raw draw time reported (nanoseconds).
    last_draw_time_ns: u64,
    /// Red zone safety margin (nanoseconds).
    red_zone_ns: u64,
    /// Refresh interval (nanoseconds), e.g., ~16.67ms for 60Hz.
    refresh_interval_ns: u64,
    /// Last known VBlank timestamp (CLOCK_MONOTONIC nanoseconds).
    last_vblank_ns: u64,
    /// Whether VRR is currently active.
    vrr_active: bool,
    /// Whether this frame involves composition (vs direct scanout).
    was_compositing: bool,
    /// Frame counter.
    frame_count: u64,
    /// Monotonic clock reference for frame timing.
    epoch: Instant,
}

impl FramePacer {
    /// Create a new frame pacer for the given refresh rate.
    pub fn new(refresh_hz: u32) -> Self {
        let refresh_interval_ns = if refresh_hz > 0 {
            1_000_000_000u64 / refresh_hz as u64
        } else {
            16_666_667 // Default to ~60Hz
        };

        info_span_stub(refresh_hz, refresh_interval_ns);

        Self {
            rolling_max_draw_time_ns: DEFAULT_DRAW_TIME_NS,
            last_draw_time_ns: DEFAULT_DRAW_TIME_NS,
            red_zone_ns: DEFAULT_RED_ZONE_NS,
            refresh_interval_ns,
            last_vblank_ns: 0,
            vrr_active: false,
            was_compositing: false,
            frame_count: 0,
            epoch: Instant::now(),
        }
    }

    /// Calculate the next wakeup time relative to `now_ns` (CLOCK_MONOTONIC).
    ///
    /// Returns the absolute timestamp (in nanoseconds) when the compositor
    /// should wake up to begin the next frame.
    pub fn next_wakeup_ns(&self, now_ns: u64) -> u64 {
        if self.vrr_active {
            // VRR mode: wake up with minimal delay.
            let offset = if self.was_compositing {
                VRR_FLUSHING_TIME_NS
            } else {
                0
            };
            return now_ns + offset;
        }

        // Fixed refresh mode: predict next VBlank and wake up early enough.
        //
        // Apply compositing floor and clamp to half the refresh interval.
        let mut draw_time = self.rolling_max_draw_time_ns;
        if self.was_compositing {
            draw_time = draw_time.max(MIN_COMPOSITING_DRAW_TIME_NS);
        }
        // Clamp: if offset exceeds half the VBlank, something is very wrong.
        draw_time = draw_time.min(self.refresh_interval_ns.saturating_sub(self.red_zone_ns));

        let offset = draw_time + self.red_zone_ns;

        if self.last_vblank_ns == 0 {
            // No VBlank reference yet — wake up after estimated interval.
            return now_ns + self.refresh_interval_ns.saturating_sub(offset);
        }

        // Calculate the next VBlank after `now_ns`.
        let elapsed = now_ns.saturating_sub(self.last_vblank_ns);
        let intervals = elapsed / self.refresh_interval_ns + 1;
        let next_vblank = self.last_vblank_ns + intervals * self.refresh_interval_ns;

        // Wake up `offset` nanoseconds before VBlank.
        next_vblank.saturating_sub(offset)
    }

    /// Calculate how long to sleep from now until the next wakeup.
    pub fn time_until_wakeup(&self, now_ns: u64) -> Duration {
        let target = self.next_wakeup_ns(now_ns);
        if target <= now_ns {
            Duration::ZERO
        } else {
            Duration::from_nanos(target - now_ns)
        }
    }

    /// Record a VBlank event from the DRM page flip handler.
    ///
    /// Re-anchors the VBlank prediction to the actual display timing.
    pub fn mark_vblank(&mut self, vblank_ns: u64) {
        self.last_vblank_ns = vblank_ns;
        self.frame_count += 1;
        trace!(vblank_ns, frame = self.frame_count, "VBlank");
    }

    /// Update the rolling peak draw time after a frame completes.
    ///
    /// Uses the asymmetric rolling-peak algorithm:
    /// - If `draw_time_ns` exceeds the rolling peak by more than `red_zone / 2`,
    ///   the peak jumps instantly to the new value (sawtooth spike-up).
    /// - Otherwise the peak decays slowly: `98% * old + 2% * new`.
    ///
    /// `draw_time_ns` is the time from wakeup to page-flip completion.
    pub fn update_draw_time(&mut self, draw_time_ns: u64) {
        self.last_draw_time_ns = draw_time_ns;

        let draw = draw_time_ns;
        let half_red = self.red_zone_ns / 2;

        let new_rolling = if draw.saturating_sub(half_red) > self.rolling_max_draw_time_ns {
            // Spike: draw time blew past rolling peak by more than half the
            // red zone — jump instantly to avoid missed VBlanks.
            debug!(
                draw_time_ns,
                rolling = self.rolling_max_draw_time_ns,
                "draw time spike, jumped rolling peak"
            );
            draw
        } else {
            // Slow decay toward current draw time.
            // new = (decay% * old + (100% - decay%) * current) / 100%
            (RATE_OF_DECAY_PER_MILLE * self.rolling_max_draw_time_ns
                + (RATE_OF_DECAY_MAX - RATE_OF_DECAY_PER_MILLE) * draw)
                / RATE_OF_DECAY_MAX
        };

        self.rolling_max_draw_time_ns = new_rolling;

        trace!(
            draw_time_ns,
            rolling = self.rolling_max_draw_time_ns,
            "draw time updated"
        );
    }

    /// Set whether VRR is active.
    pub fn set_vrr(&mut self, active: bool) {
        self.vrr_active = active;
    }

    /// Set whether the last frame involved compositing (vs direct scanout).
    pub fn set_compositing(&mut self, compositing: bool) {
        self.was_compositing = compositing;
    }

    /// Set the refresh rate. Updates the interval accordingly.
    pub fn set_refresh_rate(&mut self, refresh_hz: u32) {
        if refresh_hz > 0 {
            self.refresh_interval_ns = 1_000_000_000u64 / refresh_hz as u64;
        }
    }

    /// Set the red zone safety margin.
    pub fn set_red_zone(&mut self, ns: u64) {
        self.red_zone_ns = ns;
    }

    /// Get the current rolling peak draw time in nanoseconds.
    #[inline(always)]
    pub fn rolling_draw_time_ns(&self) -> u64 {
        self.rolling_max_draw_time_ns
    }

    /// Get the current refresh interval in nanoseconds.
    #[inline(always)]
    pub fn refresh_interval_ns(&self) -> u64 {
        self.refresh_interval_ns
    }

    /// Get the frame count.
    #[inline(always)]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

fn info_span_stub(refresh_hz: u32, interval_ns: u64) {
    tracing::info!(
        refresh_hz,
        interval_ms = interval_ns as f64 / 1_000_000.0,
        "frame pacer initialized"
    );
}

// ─── FPS Limiter ────────────────────────────────────────────────────

/// Fudge factor for VRR-style timestamp comparison (0.2 ms).
///
/// Avoids skipping a frame callback when 'now' is very close to the
/// scheduled release time.
const FPS_LIMIT_FUDGE_NS: u64 = 200_000;

/// Frame-callback throttle for FPS limiting.
///
/// Instead of sending `wl_surface.frame` callbacks immediately on every
/// commit, the compositor withholds them until the next allowed frame
/// boundary. This prevents the client from rendering faster than the
/// target FPS regardless of the client's own render loop timing.
///
/// Two modes:
///
/// - **Disabled** (`target_fps == 0`): Callbacks fire immediately after each
///   commit, capped only by the display's refresh rate (via the host's frame
///   callback in wayland mode, or VBlank in DRM mode).
///
/// - **Fixed limit** (`target_fps > 0`): Callbacks are gated to at most
///   `target_fps` per second. Uses timestamp-based scheduling: the next
///   callback is released only after `interval_ns` has elapsed since the
///   last one.
pub struct FpsLimiter {
    /// Target FPS (0 = disabled / uncapped).
    target_fps: u32,
    /// Minimum interval between frame callbacks in nanoseconds.
    /// Computed as `1_000_000_000 / target_fps`.
    interval_ns: u64,
    /// Timestamp (CLOCK_MONOTONIC ns) of the last fired frame callback.
    last_callback_ns: u64,
    /// Number of frame callbacks withheld (pending release).
    pending_count: u32,
    /// Whether VRR is active. When VRR is off and we know the display
    /// refresh rate, we use VBlank-divisor mode instead of raw timestamps.
    vrr_active: bool,
    /// Display refresh rate in Hz (for VBlank-divisor calculation).
    display_refresh_hz: u32,
    /// VBlank counter (incremented on each VBlank/frame-callback).
    vblank_idx: u64,
}

impl FpsLimiter {
    /// Create a new FPS limiter.
    ///
    /// `target_fps`: 0 = disabled (uncapped), >0 = limit to this rate.
    /// `display_refresh_hz`: the display's native refresh rate.
    pub fn new(target_fps: u32, display_refresh_hz: u32) -> Self {
        let interval_ns = if target_fps > 0 {
            1_000_000_000u64 / target_fps as u64
        } else {
            0
        };

        debug!(
            target_fps,
            display_refresh_hz,
            interval_ms = interval_ns as f64 / 1_000_000.0,
            "FPS limiter initialized"
        );

        Self {
            target_fps,
            interval_ns,
            last_callback_ns: 0,
            pending_count: 0,
            vrr_active: false,
            display_refresh_hz,
            vblank_idx: 0,
        }
    }

    /// Check whether a frame callback should be released right now.
    ///
    /// `now_ns` is the current CLOCK_MONOTONIC timestamp in nanoseconds.
    ///
    /// Returns `true` if the callback should fire, `false` if it should
    /// be withheld until later.
    #[inline(always)]
    pub fn should_release(&self, now_ns: u64) -> bool {
        // Disabled — always release immediately.
        if self.target_fps == 0 {
            return true;
        }

        // First frame — always release.
        if self.last_callback_ns == 0 {
            return true;
        }

        let deadline = self.last_callback_ns + self.interval_ns;

        // Release if we're at or past the deadline (with fudge tolerance).
        now_ns + FPS_LIMIT_FUDGE_NS >= deadline
    }

    /// Record that a frame callback was released at `now_ns`.
    #[inline(always)]
    pub fn mark_released(&mut self, now_ns: u64) {
        self.last_callback_ns = now_ns;
        self.vblank_idx += 1;
    }

    /// Compute when the next callback should be released (absolute ns).
    ///
    /// Returns `None` if the limiter is disabled or no reference time
    /// is available yet.
    pub fn next_release_ns(&self) -> Option<u64> {
        if self.target_fps == 0 || self.last_callback_ns == 0 {
            return None;
        }
        Some(self.last_callback_ns + self.interval_ns)
    }

    /// Compute how long to sleep until the next release point.
    pub fn time_until_release(&self, now_ns: u64) -> Duration {
        match self.next_release_ns() {
            Some(t) if t > now_ns => Duration::from_nanos(t - now_ns),
            _ => Duration::ZERO,
        }
    }

    /// Update the target FPS at runtime (e.g., from an X11 atom).
    pub fn set_target_fps(&mut self, fps: u32) {
        self.target_fps = fps;
        self.interval_ns = if fps > 0 {
            1_000_000_000u64 / fps as u64
        } else {
            0
        };
        debug!(
            fps,
            interval_ms = self.interval_ns as f64 / 1_000_000.0,
            "FPS limit updated"
        );
    }

    /// Update VRR state.
    pub fn set_vrr(&mut self, active: bool) {
        self.vrr_active = active;
    }

    /// Update the display refresh rate.
    pub fn set_display_refresh(&mut self, hz: u32) {
        self.display_refresh_hz = hz;
    }

    /// Get the current target FPS (0 = uncapped).
    #[inline(always)]
    pub fn target_fps(&self) -> u32 {
        self.target_fps
    }

    /// Register a new pending callback (client requested a frame).
    #[inline(always)]
    pub fn add_pending(&mut self) {
        self.pending_count += 1;
    }

    /// Clear the pending count (after firing).
    #[inline(always)]
    pub fn clear_pending(&mut self) {
        self.pending_count = 0;
    }

    /// Whether there are pending callbacks waiting to be released.
    #[inline(always)]
    pub fn has_pending(&self) -> bool {
        self.pending_count > 0
    }
}

#[cfg(test)]
#[path = "frame_pacer_tests.rs"]
mod tests;
