//! Performance statistics and stats pipe output.
//!
//! Tracks frame timing metrics and optionally writes them to a named pipe
//! that external tools (e.g., MangoHUD) can read from.
//!
//! All counters are updated on the main thread. The stats pipe is written
//! to asynchronously to avoid blocking the compositor.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use tracing::debug;

/// Per-frame timing statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Frame sequence number.
    pub seq: u64,
    /// Time from wakeup to GPU submission (nanoseconds).
    pub draw_time_ns: u64,
    /// Time from commit to page flip completion (nanoseconds).
    pub flip_time_ns: u64,
    /// Whether this frame used direct scanout.
    pub direct_scanout: bool,
    /// Whether this frame used VRR.
    pub vrr_active: bool,
    /// Number of composition layers.
    pub layer_count: u32,
    /// Current FPS (frames per second).
    pub fps: f64,
}

/// Stats tracker and pipe writer.
pub struct StatsTracker {
    /// Recent frame stats for FPS calculation.
    recent_frame_times: [u64; 60],
    /// Index into recent_frame_times ring buffer.
    ring_index: usize,
    /// Total frame count.
    total_frames: u64,
    /// Start time for FPS calculation.
    fps_start: Instant,
    /// Frames since last FPS calculation.
    fps_frame_count: u64,
    /// Current FPS.
    current_fps: f64,
    /// Stats pipe path (if enabled).
    pipe_path: Option<PathBuf>,
    /// Stats pipe file handle.
    pipe: Option<std::fs::File>,
}

impl StatsTracker {
    /// Create a new stats tracker.
    pub fn new(pipe_path: Option<PathBuf>) -> Self {
        Self {
            recent_frame_times: [0u64; 60],
            ring_index: 0,
            total_frames: 0,
            fps_start: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            pipe_path,
            pipe: None,
        }
    }

    /// Record a completed frame.
    pub fn record_frame(&mut self, stats: &FrameStats) {
        // Update ring buffer.
        self.recent_frame_times[self.ring_index] = stats.draw_time_ns;
        self.ring_index = (self.ring_index + 1) % self.recent_frame_times.len();
        self.total_frames += 1;

        // FPS calculation (once per second).
        self.fps_frame_count += 1;
        let elapsed = self.fps_start.elapsed();
        if elapsed.as_secs() >= 1 {
            self.current_fps = self.fps_frame_count as f64 / elapsed.as_secs_f64();
            self.fps_frame_count = 0;
            self.fps_start = Instant::now();

            debug!(
                fps = format!("{:.1}", self.current_fps),
                total_frames = self.total_frames,
                "fps update"
            );
        }

        // Write to stats pipe if enabled.
        self.write_pipe(stats);
    }

    /// Get the current FPS.
    #[inline(always)]
    pub fn fps(&self) -> f64 {
        self.current_fps
    }

    /// Get the total frame count.
    #[inline(always)]
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Average draw time in nanoseconds (over recent frames).
    pub fn avg_draw_time_ns(&self) -> u64 {
        let (sum, count) = self
            .recent_frame_times
            .iter()
            .filter(|&&t| t > 0)
            .fold((0u64, 0u64), |(s, c), &t| (s + t, c + 1));
        if count == 0 {
            return 0;
        }
        sum / count
    }

    /// Write stats to the named pipe (non-blocking).
    fn write_pipe(&mut self, stats: &FrameStats) {
        let Some(ref path) = self.pipe_path else {
            return;
        };

        // Lazily open the pipe.
        if self.pipe.is_none() {
            match std::fs::OpenOptions::new().write(true).open(path) {
                Ok(f) => self.pipe = Some(f),
                Err(_) => return, // Pipe not ready yet.
            }
        }

        if let Some(ref mut pipe) = self.pipe {
            let line = format!(
                "{} {} {} {} {} {:.1}\n",
                stats.seq,
                stats.draw_time_ns,
                stats.flip_time_ns,
                if stats.direct_scanout { 1 } else { 0 },
                stats.layer_count,
                stats.fps,
            );
            if pipe.write_all(line.as_bytes()).is_err() {
                // Pipe broken — reader disconnected.
                self.pipe = None;
            }
        }
    }
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod tests;
