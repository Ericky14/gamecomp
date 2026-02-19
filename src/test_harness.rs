//! Test harness and utilities for gamecomp.
//!
//! Provides reusable testing infrastructure for all modules:
//! - [`TestCompositor`] — a fully wired compositor instance using the headless
//!   backend, suitable for integration tests.
//! - [`MockBackend`] — a programmable mock backend for unit testing composition
//!   logic without real hardware.
//! - Frame assertion helpers for validating scene descriptions and timing.
//!
//! Design: All test utilities avoid real hardware access. They use
//! [`HeadlessBackend`] or [`MockBackend`] so tests can run in CI without
//! GPU, display, or input devices.

use std::collections::VecDeque;
use std::os::unix::io::RawFd;
use std::time::{Duration, Instant};

use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};

use crate::backend::{Backend, BackendCaps, ConnectorInfo, DmaBuf, FlipResult, Framebuffer};
use crate::compositor::scene::{
    BlendMode, ColorSpace, CompositionMode, FilterMode, FrameInfo, Layer, MAX_LAYERS, Rect,
};
use crate::config::Config;
use crate::frame_pacer::FramePacer;
use crate::input::InputHandler;
use crate::stats::StatsTracker;

// ── MockBackend ──────────────────────────────────────────────────────────

/// A programmable mock backend for testing.
///
/// Records all calls made to it and returns configurable responses.
/// Useful for testing composition logic, frame pacing, and event handling
/// without touching real DRM hardware.
pub struct MockBackend {
    /// Capabilities to advertise.
    pub caps: BackendCaps,
    /// Scanout formats to advertise.
    pub formats: Vec<DrmFormat>,
    /// Whether `try_direct_scanout` should succeed.
    pub direct_scanout_succeeds: bool,
    /// Whether `present` should succeed.
    pub present_succeeds: bool,
    /// Record of presented framebuffers.
    pub presented_frames: Vec<PresentedFrame>,
    /// Record of direct scanout attempts.
    pub scanout_attempts: Vec<bool>,
    /// Record of imported DMA-BUFs.
    pub imported_dmabufs: Vec<DmaBufRecord>,
    /// Frame counter.
    pub frame_count: u64,
    /// Fake connector info.
    connector_info: Vec<ConnectorInfo>,
    /// VRR state.
    pub vrr_enabled: bool,
    /// Simulated flip latency.
    pub flip_latency: Duration,
    /// Whether init has been called.
    pub initialized: bool,
}

/// Record of a presented frame.
#[derive(Debug, Clone)]
pub struct PresentedFrame {
    pub format: DrmFourcc,
    pub size: (u32, u32),
    pub timestamp: Instant,
    pub frame_number: u64,
}

/// Record of an imported DMA-BUF.
#[derive(Debug, Clone)]
pub struct DmaBufRecord {
    pub width: u32,
    pub height: u32,
    pub format: DrmFourcc,
}

impl MockBackend {
    /// Create a mock backend with sensible defaults.
    pub fn new(_width: u32, _height: u32) -> Self {
        Self {
            caps: BackendCaps {
                vrr: true,
                hdr: false,
                tearing: false,
                explicit_sync: true,
                modifiers: true,
            },
            formats: vec![
                DrmFormat {
                    code: DrmFourcc::Argb8888,
                    modifier: DrmModifier::Linear,
                },
                DrmFormat {
                    code: DrmFourcc::Xrgb8888,
                    modifier: DrmModifier::Linear,
                },
            ],
            direct_scanout_succeeds: true,
            present_succeeds: true,
            presented_frames: Vec::new(),
            scanout_attempts: Vec::new(),
            imported_dmabufs: Vec::new(),
            frame_count: 0,
            connector_info: Vec::new(),
            vrr_enabled: false,
            flip_latency: Duration::ZERO,
            initialized: false,
        }
    }

    /// Reset all recorded state (for reuse between tests).
    pub fn reset(&mut self) {
        self.presented_frames.clear();
        self.scanout_attempts.clear();
        self.imported_dmabufs.clear();
        self.frame_count = 0;
    }

    /// Number of frames presented.
    pub fn present_count(&self) -> usize {
        self.presented_frames.len()
    }

    /// Number of direct scanout attempts.
    pub fn scanout_attempt_count(&self) -> usize {
        self.scanout_attempts.len()
    }

    /// Number of successful direct scanout attempts.
    pub fn scanout_success_count(&self) -> usize {
        self.scanout_attempts.iter().filter(|&&s| s).count()
    }
}

impl Backend for MockBackend {
    fn init(&mut self) -> anyhow::Result<()> {
        self.initialized = true;
        // SAFETY: Mock handles never used for real DRM operations.
        self.connector_info.push(ConnectorInfo {
            handle: unsafe { std::mem::transmute::<u32, drm::control::connector::Handle>(1u32) },
            crtc: unsafe { std::mem::transmute::<u32, drm::control::crtc::Handle>(1u32) },
            name: "MOCK-1".to_string(),
            mode: unsafe { std::mem::zeroed() },
            physical_size_mm: (300, 200),
            vrr_enabled: false,
        });
        Ok(())
    }

    fn connectors(&self) -> &[ConnectorInfo] {
        &self.connector_info
    }

    fn capabilities(&self) -> BackendCaps {
        self.caps
    }

    fn scanout_formats(&self) -> &[DrmFormat] {
        &self.formats
    }

    fn import_dmabuf(&mut self, dmabuf: &DmaBuf) -> anyhow::Result<Framebuffer> {
        self.imported_dmabufs.push(DmaBufRecord {
            width: dmabuf.width,
            height: dmabuf.height,
            format: dmabuf.format,
        });
        Ok(Framebuffer {
            // SAFETY: Mock handle is never used for real DRM operations.
            handle: unsafe {
                std::mem::transmute::<u32, drm::control::framebuffer::Handle>(
                    (self.frame_count as u32).max(1),
                )
            },
            format: dmabuf.format,
            modifier: dmabuf.modifier,
            size: (dmabuf.width, dmabuf.height),
        })
    }

    fn try_direct_scanout(&mut self, _fb: &Framebuffer) -> anyhow::Result<bool> {
        let result = self.direct_scanout_succeeds;
        self.scanout_attempts.push(result);
        Ok(result)
    }

    fn present(&mut self, fb: &Framebuffer) -> anyhow::Result<FlipResult> {
        if !self.present_succeeds {
            return Ok(FlipResult::Failed(anyhow::anyhow!("mock present failure")));
        }
        self.frame_count += 1;
        self.presented_frames.push(PresentedFrame {
            format: fb.format,
            size: fb.size,
            timestamp: Instant::now(),
            frame_number: self.frame_count,
        });
        Ok(FlipResult::Queued)
    }

    fn drm_fd(&self) -> Option<RawFd> {
        None
    }

    fn handle_page_flip(&mut self) -> anyhow::Result<Option<u64>> {
        Ok(None)
    }

    fn set_vrr(&mut self, enabled: bool) -> anyhow::Result<()> {
        self.vrr_enabled = enabled;
        Ok(())
    }
}

// ── TestCompositor ───────────────────────────────────────────────────────

/// A fully wired compositor instance for integration testing.
///
/// Uses [`MockBackend`] internally. Provides high-level methods for
/// simulating compositor lifecycle, frame submission, and assertion.
pub struct TestCompositor {
    /// Mock display backend.
    pub backend: MockBackend,
    /// Frame pacer (60Hz default).
    pub pacer: FramePacer,
    /// Stats tracker (no pipe).
    pub stats: StatsTracker,
    /// Input handler.
    pub input: InputHandler,
    /// Configuration.
    pub config: Config,
    /// Submitted frame history.
    pub frame_history: VecDeque<FrameInfo>,
    /// Maximum frame history size.
    pub max_history: usize,
}

impl TestCompositor {
    /// Create a new test compositor with default 1920x1080 @ 60Hz configuration.
    pub fn new() -> Self {
        Self::with_resolution(1920, 1080, 60)
    }

    /// Create a test compositor with the given resolution and refresh rate.
    pub fn with_resolution(width: u32, height: u32, refresh_hz: u32) -> Self {
        let mut backend = MockBackend::new(width, height);
        backend.init().expect("mock backend init failed");

        let pacer = FramePacer::new(refresh_hz);
        let stats = StatsTracker::new(None);
        let input = InputHandler::new().expect("input handler init failed");
        let config = Config::default();

        Self {
            backend,
            pacer,
            stats,
            input,
            config,
            frame_history: VecDeque::new(),
            max_history: 120,
        }
    }

    /// Submit a frame for composition.
    pub fn submit_frame(&mut self, frame: FrameInfo) -> anyhow::Result<FlipResult> {
        // Record in history.
        if self.frame_history.len() >= self.max_history {
            self.frame_history.pop_front();
        }
        self.frame_history.push_back(frame);

        // Determine if direct scanout is possible.
        if frame.is_direct_scanout_candidate() {
            let fb = Framebuffer {
                // SAFETY: Mock handle is never used for real DRM operations.
                handle: unsafe {
                    std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32)
                },
                format: DrmFourcc::Argb8888,
                modifier: DrmModifier::Linear,
                size: (frame.output_width, frame.output_height),
            };
            if self.backend.try_direct_scanout(&fb)? {
                return self.backend.present(&fb);
            }
        }

        // Fall back to composition + present.
        let fb = Framebuffer {
            // SAFETY: Mock handle is never used for real DRM operations.
            handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
            format: DrmFourcc::Argb8888,
            modifier: DrmModifier::Linear,
            size: (frame.output_width, frame.output_height),
        };
        self.backend.present(&fb)
    }

    /// Simulate N frames with the given generator function.
    pub fn run_frames<F>(&mut self, count: usize, mut frame_fn: F) -> Vec<FlipResult>
    where
        F: FnMut(u64) -> FrameInfo,
    {
        let mut results = Vec::with_capacity(count);
        for i in 0..count {
            let frame = frame_fn(i as u64);
            match self.submit_frame(frame) {
                Ok(result) => results.push(result),
                Err(e) => {
                    panic!("frame {} failed: {}", i, e);
                }
            }
        }
        results
    }

    /// Assert that N frames were presented to the backend.
    pub fn assert_frame_count(&self, expected: usize) {
        assert_eq!(
            self.backend.present_count(),
            expected,
            "expected {} frames presented, got {}",
            expected,
            self.backend.present_count()
        );
    }

    /// Assert that all frames used direct scanout.
    pub fn assert_all_direct_scanout(&self) {
        assert!(
            self.backend.scanout_attempts.iter().all(|&s| s),
            "not all frames used direct scanout: {:?}",
            self.backend.scanout_attempts
        );
    }

    /// Assert that no frames used direct scanout.
    pub fn assert_no_direct_scanout(&self) {
        assert!(
            self.backend.scanout_attempts.iter().all(|&s| !s),
            "some frames used direct scanout unexpectedly: {:?}",
            self.backend.scanout_attempts
        );
    }

    /// Get the last submitted frame.
    pub fn last_frame(&self) -> Option<&FrameInfo> {
        self.frame_history.back()
    }
}

// ── Frame assertion helpers ──────────────────────────────────────────────

/// Assert that a FrameInfo has the expected number of active layers.
pub fn assert_layer_count(frame: &FrameInfo, expected: u32) {
    assert_eq!(
        frame.layer_count, expected,
        "expected {} layers, got {}",
        expected, frame.layer_count
    );
}

/// Assert that a specific layer is active and has the expected properties.
pub fn assert_layer_active(frame: &FrameInfo, index: usize) {
    assert!(
        index < MAX_LAYERS,
        "layer index {} out of bounds (max {})",
        index,
        MAX_LAYERS
    );
    assert!(
        frame.layers[index].active,
        "layer {} should be active",
        index
    );
}

/// Assert that a specific layer is inactive.
pub fn assert_layer_inactive(frame: &FrameInfo, index: usize) {
    assert!(index < MAX_LAYERS, "layer index {} out of bounds", index);
    assert!(
        !frame.layers[index].active,
        "layer {} should be inactive",
        index
    );
}

/// Assert that a frame's composition mode matches expected.
pub fn assert_composition_mode(frame: &FrameInfo, expected: CompositionMode) {
    assert_eq!(
        frame.mode, expected,
        "expected composition mode {:?}, got {:?}",
        expected, frame.mode
    );
}

/// Assert that a layer covers the full output.
pub fn assert_layer_fullscreen(frame: &FrameInfo, index: usize) {
    let layer = &frame.layers[index];
    assert_eq!(
        (layer.dst.width, layer.dst.height),
        (frame.output_width, frame.output_height),
        "layer {} does not cover full output: {:?} vs {}x{}",
        index,
        layer.dst,
        frame.output_width,
        frame.output_height
    );
}

/// Builder for creating test FrameInfo with fluent API.
pub struct FrameBuilder {
    frame: FrameInfo,
}

impl FrameBuilder {
    /// Start building a frame with the given output dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let frame = FrameInfo {
            output_width: width,
            output_height: height,
            ..Default::default()
        };
        Self { frame }
    }

    /// Add a fullscreen opaque layer (typical game frame).
    ///
    /// Places the layer at [`LAYER_APP`] (index 1), matching the compositor's
    /// layer slot convention.
    pub fn with_fullscreen_app(mut self, texture_index: u32) -> Self {
        use crate::compositor::scene::LAYER_APP;
        let w = self.frame.output_width;
        let h = self.frame.output_height;
        self.frame.layers[LAYER_APP] = Layer {
            texture_index,
            active: true,
            src: Rect {
                x: 0,
                y: 0,
                width: w,
                height: h,
            },
            dst: Rect {
                x: 0,
                y: 0,
                width: w,
                height: h,
            },
            opacity: 1.0,
            filter: FilterMode::Nearest,
            color_space: ColorSpace::Srgb,
            blend: BlendMode::Opaque,
            format: DrmFourcc::Argb8888 as u32,
        };
        // layer_count tracks the highest occupied slot + 1.
        if self.frame.layer_count <= LAYER_APP as u32 {
            self.frame.layer_count = (LAYER_APP + 1) as u32;
        }
        self
    }

    /// Add an overlay layer at the given position.
    ///
    /// Places the layer at [`LAYER_OVERLAY`] (index 2).
    pub fn with_overlay(
        mut self,
        texture_index: u32,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        opacity: f32,
    ) -> Self {
        use crate::compositor::scene::LAYER_OVERLAY;
        self.frame.layers[LAYER_OVERLAY] = Layer {
            texture_index,
            active: true,
            src: Rect {
                x: 0,
                y: 0,
                width: w,
                height: h,
            },
            dst: Rect {
                x,
                y,
                width: w,
                height: h,
            },
            opacity,
            filter: FilterMode::Linear,
            color_space: ColorSpace::Srgb,
            blend: BlendMode::AlphaPreMultiplied,
            format: DrmFourcc::Argb8888 as u32,
        };
        if self.frame.layer_count <= LAYER_OVERLAY as u32 {
            self.frame.layer_count = (LAYER_OVERLAY + 1) as u32;
        }
        self
    }

    /// Add a cursor layer.
    ///
    /// Places the layer at [`LAYER_CURSOR`] (index 3).
    pub fn with_cursor(mut self, texture_index: u32, x: i32, y: i32) -> Self {
        use crate::compositor::scene::LAYER_CURSOR;
        self.frame.layers[LAYER_CURSOR] = Layer {
            texture_index,
            active: true,
            src: Rect {
                x: 0,
                y: 0,
                width: 24,
                height: 24,
            },
            dst: Rect {
                x,
                y,
                width: 24,
                height: 24,
            },
            opacity: 1.0,
            filter: FilterMode::Nearest,
            color_space: ColorSpace::Srgb,
            blend: BlendMode::AlphaPreMultiplied,
            format: DrmFourcc::Argb8888 as u32,
        };
        if self.frame.layer_count <= LAYER_CURSOR as u32 {
            self.frame.layer_count = (LAYER_CURSOR + 1) as u32;
        }
        self
    }

    /// Set the composition mode.
    pub fn with_mode(mut self, mode: CompositionMode) -> Self {
        self.frame.mode = mode;
        self
    }

    /// Set the frame sequence number.
    pub fn with_seq(mut self, seq: u64) -> Self {
        self.frame.seq = seq;
        self
    }

    /// Set VRR state.
    pub fn with_vrr(mut self, active: bool) -> Self {
        self.frame.vrr_active = active;
        self
    }

    /// Build the frame. Automatically sets the composition mode if not set.
    pub fn build(mut self) -> FrameInfo {
        if self.frame.mode == CompositionMode::Skip {
            // Auto-detect mode based on layers.
            if self.frame.layer_count == 0 {
                self.frame.mode = CompositionMode::Skip;
            } else if self.frame.is_direct_scanout_candidate() {
                self.frame.mode = CompositionMode::DirectScanout;
            } else {
                self.frame.mode = CompositionMode::Composite;
            }
        }
        self.frame
    }
}

// ── Timing test helpers ──────────────────────────────────────────────────

/// Measure execution time of a closure in nanoseconds.
pub fn measure_ns<F: FnOnce() -> R, R>(f: F) -> (R, u64) {
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_nanos() as u64;
    (result, elapsed)
}

/// Assert that a closure executes within the given time budget.
pub fn assert_within_budget<F: FnOnce()>(budget: Duration, name: &str, f: F) {
    let start = Instant::now();
    f();
    let elapsed = start.elapsed();
    assert!(
        elapsed <= budget,
        "{} took {:?}, budget was {:?}",
        name,
        elapsed,
        budget
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_backend_init() {
        let mut backend = MockBackend::new(1920, 1080);
        assert!(!backend.initialized);
        backend.init().unwrap();
        assert!(backend.initialized);
        assert_eq!(backend.connectors().len(), 1);
        assert_eq!(backend.connectors()[0].name, "MOCK-1");
    }

    #[test]
    fn mock_backend_present_records() {
        let mut backend = MockBackend::new(1920, 1080);
        backend.init().unwrap();

        let fb = Framebuffer {
            // SAFETY: Mock handle is never used for real DRM operations.
            handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
            format: DrmFourcc::Argb8888,
            modifier: DrmModifier::Linear,
            size: (1920, 1080),
        };
        backend.present(&fb).unwrap();
        backend.present(&fb).unwrap();

        assert_eq!(backend.present_count(), 2);
        assert_eq!(backend.frame_count, 2);
    }

    #[test]
    fn mock_backend_scanout_tracking() {
        let mut backend = MockBackend::new(1920, 1080);
        backend.init().unwrap();
        backend.direct_scanout_succeeds = true;

        let fb = Framebuffer {
            // SAFETY: Mock handle is never used for real DRM operations.
            handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
            format: DrmFourcc::Argb8888,
            modifier: DrmModifier::Linear,
            size: (1920, 1080),
        };

        assert!(backend.try_direct_scanout(&fb).unwrap());
        backend.direct_scanout_succeeds = false;
        assert!(!backend.try_direct_scanout(&fb).unwrap());

        assert_eq!(backend.scanout_attempt_count(), 2);
        assert_eq!(backend.scanout_success_count(), 1);
    }

    #[test]
    fn mock_backend_reset() {
        let mut backend = MockBackend::new(1920, 1080);
        backend.init().unwrap();
        let fb = Framebuffer {
            // SAFETY: Mock handle is never used for real DRM operations.
            handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
            format: DrmFourcc::Argb8888,
            modifier: DrmModifier::Linear,
            size: (1920, 1080),
        };
        backend.present(&fb).unwrap();
        assert_eq!(backend.present_count(), 1);

        backend.reset();
        assert_eq!(backend.present_count(), 0);
        assert_eq!(backend.frame_count, 0);
    }

    #[test]
    fn test_compositor_direct_scanout() {
        let mut comp = TestCompositor::new();
        let frame = FrameBuilder::new(1920, 1080).with_fullscreen_app(0).build();

        assert!(frame.is_direct_scanout_candidate());
        comp.submit_frame(frame).unwrap();
        comp.assert_frame_count(1);
    }

    #[test]
    fn test_compositor_composite_with_overlay() {
        let mut comp = TestCompositor::new();
        let frame = FrameBuilder::new(1920, 1080)
            .with_fullscreen_app(0)
            .with_overlay(1, 100, 100, 300, 200, 0.8)
            .build();

        assert!(!frame.is_direct_scanout_candidate());
        assert_eq!(frame.mode, CompositionMode::Composite);
        comp.submit_frame(frame).unwrap();
        comp.assert_frame_count(1);
    }

    #[test]
    fn test_compositor_run_frames() {
        let mut comp = TestCompositor::new();
        let results = comp.run_frames(60, |seq| {
            FrameBuilder::new(1920, 1080)
                .with_fullscreen_app(0)
                .with_seq(seq)
                .build()
        });
        assert_eq!(results.len(), 60);
        comp.assert_frame_count(60);
    }

    #[test]
    fn frame_builder_auto_detects_mode() {
        // Single fullscreen = DirectScanout.
        let frame = FrameBuilder::new(1920, 1080).with_fullscreen_app(0).build();
        assert_eq!(frame.mode, CompositionMode::DirectScanout);

        // With overlay = Composite.
        let frame = FrameBuilder::new(1920, 1080)
            .with_fullscreen_app(0)
            .with_cursor(1, 500, 500)
            .build();
        assert_eq!(frame.mode, CompositionMode::Composite);

        // Empty = Skip.
        let frame = FrameBuilder::new(1920, 1080).build();
        assert_eq!(frame.mode, CompositionMode::Skip);
    }

    #[test]
    fn frame_builder_layer_assertions() {
        let frame = FrameBuilder::new(1920, 1080)
            .with_fullscreen_app(0)
            .with_overlay(1, 0, 0, 200, 100, 0.5)
            .build();

        // App at LAYER_APP (1) + overlay at LAYER_OVERLAY (2) → layer_count = 3.
        assert_layer_count(&frame, 3);
        assert_layer_active(&frame, 1); // LAYER_APP
        assert_layer_active(&frame, 2); // LAYER_OVERLAY
        assert_layer_fullscreen(&frame, 1); // App covers full output
    }

    #[test]
    fn timing_helper_works() {
        let (result, ns) = measure_ns(|| 42);
        assert_eq!(result, 42);
        assert!(ns < 1_000_000_000); // Less than 1 second.
    }

    #[test]
    fn vrf_mode_configuration() {
        let mut backend = MockBackend::new(1920, 1080);
        backend.init().unwrap();
        assert!(!backend.vrr_enabled);
        backend.set_vrr(true).unwrap();
        assert!(backend.vrr_enabled);
    }
}
