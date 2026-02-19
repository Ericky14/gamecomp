//! Backend abstraction for display output.
//!
//! Defines the [`Backend`] trait that all display backends must implement.
//! Each backend owns its display resources exclusively — the render thread
//! is the sole consumer. Communication with the main thread happens via channels.
//!
//! Three backends are provided:
//! - [`DrmBackend`](drm::DrmBackend) — direct KMS/DRM output for production use.
//! - [`HeadlessBackend`](headless::HeadlessBackend) — offscreen rendering for CI and robotics.
//! - [`WaylandBackend`](wayland::WaylandBackend) — runs inside another Wayland compositor.

pub mod drm;
pub mod gpu;
pub mod headless;
pub mod wayland;

use std::os::unix::io::RawFd;

use ::drm::control::{Mode, connector, crtc, framebuffer};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};

/// Typed errors for backend operations.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// No connected output available for presentation.
    #[error("no connected output")]
    NoOutput,
    /// Primary plane not found for the output.
    #[error("no primary plane")]
    NoPrimaryPlane,
    /// Memory mapping a shared buffer failed.
    #[error("mmap failed")]
    MmapFailed,
}

/// Capabilities advertised by a backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendCaps {
    /// Whether the display supports variable refresh rate.
    pub vrr: bool,
    /// Whether the display supports HDR output.
    pub hdr: bool,
    /// Whether tearing presentation is supported.
    pub tearing: bool,
    /// Whether explicit sync (DRM syncobj) is available.
    pub explicit_sync: bool,
    /// Whether DMA-BUF modifiers are used for buffer allocation.
    pub modifiers: bool,
}

/// Information about a connected display output.
#[derive(Debug, Clone)]
pub struct ConnectorInfo {
    /// Connector handle.
    pub handle: connector::Handle,
    /// CRTC handle assigned to this connector.
    pub crtc: crtc::Handle,
    /// Display name (e.g., "eDP-1", "HDMI-A-1").
    pub name: String,
    /// Current display mode.
    pub mode: Mode,
    /// Physical size in millimeters (width, height).
    pub physical_size_mm: (u32, u32),
    /// Whether VRR is currently enabled on this connector.
    pub vrr_enabled: bool,
}

/// A scanout buffer that can be presented to a display plane.
#[derive(Debug, Clone, Copy)]
pub struct Framebuffer {
    /// DRM framebuffer handle.
    pub handle: framebuffer::Handle,
    /// DRM format of the buffer.
    pub format: DrmFourcc,
    /// DRM modifier of the buffer.
    pub modifier: DrmModifier,
    /// Buffer dimensions (width, height).
    pub size: (u32, u32),
}

/// DMA-BUF descriptor for importing client buffers.
#[derive(Debug, Clone)]
pub struct DmaBuf {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// DRM pixel format.
    pub format: DrmFourcc,
    /// DRM modifier.
    pub modifier: DrmModifier,
    /// Per-plane file descriptors, offsets, and strides.
    pub planes: Vec<DmaBufPlane>,
}

/// A single plane of a DMA-BUF.
#[derive(Debug, Clone, Copy)]
pub struct DmaBufPlane {
    /// File descriptor for this plane.
    pub fd: RawFd,
    /// Byte offset into the fd.
    pub offset: u32,
    /// Stride (bytes per row) for this plane.
    pub stride: u32,
}

/// Result of attempting a page flip.
#[derive(Debug)]
pub enum FlipResult {
    /// Flip was successfully queued. The backend will signal completion.
    Queued,
    /// Direct scanout was used — the client buffer is on a hardware plane.
    DirectScanout,
    /// The flip could not be performed (e.g., no compatible plane).
    Failed(anyhow::Error),
}

/// The display backend trait.
///
/// Backends own all display hardware state and run on the render thread.
/// They receive [`FrameInfo`] from the main thread and produce page flips.
pub trait Backend: Send {
    /// Initialize the backend and discover connected outputs.
    fn init(&mut self) -> anyhow::Result<()>;

    /// Return information about connected outputs.
    fn connectors(&self) -> &[ConnectorInfo];

    /// Return the backend's capabilities.
    fn capabilities(&self) -> BackendCaps;

    /// Return the preferred scanout formats for the primary plane.
    fn scanout_formats(&self) -> &[DrmFormat];

    /// Import a DMA-BUF as a framebuffer for scanout.
    fn import_dmabuf(&mut self, dmabuf: &DmaBuf) -> anyhow::Result<Framebuffer>;

    /// Attempt direct scanout of a client framebuffer.
    ///
    /// Returns `true` if the buffer was successfully assigned to a hardware plane
    /// and an atomic commit (TEST_ONLY) succeeded.
    fn try_direct_scanout(&mut self, fb: &Framebuffer) -> anyhow::Result<bool>;

    /// Present a framebuffer to the display via atomic commit.
    ///
    /// This queues a page flip. The caller should wait for the flip event
    /// on the DRM fd before submitting the next frame.
    fn present(&mut self, fb: &Framebuffer) -> anyhow::Result<FlipResult>;

    /// Return the DRM device fd for polling page flip events.
    fn drm_fd(&self) -> Option<RawFd>;

    /// Handle a page flip event. Called when the DRM fd is readable.
    fn handle_page_flip(&mut self) -> anyhow::Result<()>;

    /// Set VRR (variable refresh rate) enabled/disabled.
    fn set_vrr(&mut self, enabled: bool) -> anyhow::Result<()>;
}
