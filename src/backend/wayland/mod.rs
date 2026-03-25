//! Wayland backend for running inside another compositor.
//!
//! When gamecomp runs as a windowed application inside an existing Wayland
//! or X11 session, it uses this backend. The wayland backend creates a
//! `wl_surface` + `xdg_toplevel` on the host compositor and presents
//! composed frames via `wl_shm` buffers (Vulkan swapchain is planned).
//!
//! This mode is essential for:
//! - **Development**: Test the compositor without dedicating a physical display.
//! - **CI**: Run integration tests in a headless Wayland compositor (e.g., weston).
//! - **Embedding**: Run inside another compositor as a window.
//!
//! The backend connects as a Wayland client to the host compositor, creates an
//! `xdg_toplevel` surface, and presents composed frames.
//!
//! Thread model: The backend runs on the render thread. A separate event
//! dispatch thread (`gamecomp-wayland`) handles host compositor events and
//! sends them to the render thread via channel.

mod event_loop;
mod host_state;

use std::os::unix::io::{OwnedFd, RawFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::Context;
use drm::control::{connector, crtc, framebuffer};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use tracing::{info, warn};

use super::{Backend, BackendCaps, ConnectorInfo, DmaBuf, FlipResult, Framebuffer};
use crate::wayland::protocols::CommittedBuffer;
use event_loop::{HostLoopParams, wayland_event_loop};

/// Cursor update from a client, forwarded to the host compositor.
pub enum CursorUpdate {
    /// Client set a cursor image via `wl_pointer.set_cursor`.
    Image {
        /// ARGB8888 pixel data.
        pixels: Vec<u8>,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Hotspot X offset.
        hotspot_x: i32,
        /// Hotspot Y offset.
        hotspot_y: i32,
    },
    /// Client requested cursor hide (surface = None in set_cursor).
    Hide,
}

/// Events from the host compositor.
#[derive(Debug)]
pub enum WaylandEvent {
    /// The window was resized by the host compositor.
    /// `width`/`height` are host logical coords; `physical_width`/`physical_height`
    /// are the pixel dimensions after applying fractional scale.
    Resized {
        width: u32,
        height: u32,
        physical_width: u32,
        physical_height: u32,
    },
    /// The window should close.
    CloseRequested,
    /// Pointer motion within the window.
    PointerMotion { x: f64, y: f64 },
    /// Pointer button press/release.
    PointerButton { button: u32, pressed: bool },
    /// Scroll event.
    Scroll { dx: f64, dy: f64 },
    /// Keyboard key event.
    Key { key: u32, pressed: bool },
    /// Keyboard modifier state update from host.
    Modifiers {
        mods_depressed: u32,
        mods_latched: u32,
        mods_locked: u32,
        group: u32,
    },
    /// XKB keymap from the host compositor (format, fd, size).
    Keymap { format: u32, fd: OwnedFd, size: u32 },
    /// Window gained focus.
    FocusIn,
    /// Window lost focus.
    FocusOut,
    /// Frame callback from host compositor (safe to present).
    FrameCallback,
}

/// Configuration for the wayland backend.
pub struct WaylandConfig {
    /// Initial window width.
    pub width: u32,
    /// Initial window height.
    pub height: u32,
    /// Window title.
    pub title: String,
    /// Whether to start in fullscreen mode.
    pub fullscreen: bool,
    /// Whether to use Vulkan swapchain for presentation.
    pub use_vulkan: bool,
    /// Host compositor's WAYLAND_DISPLAY (saved before we overwrite it).
    pub host_wayland_display: Option<String>,
    /// Receiver for committed frames from the Wayland server.
    pub committed_frame_rx: Option<std::sync::mpsc::Receiver<CommittedBuffer>>,
    /// Receiver for cursor image updates from the Wayland server.
    pub cursor_rx: Option<std::sync::mpsc::Receiver<CursorUpdate>>,
    /// Shared atomic for detected host display refresh rate (millihertz).
    /// Written by the event thread, read by the main thread.
    pub detected_refresh_mhz: Arc<AtomicU32>,
    /// Shared host DMA-BUF format→modifier map. Written by the event thread
    /// after the initial roundtrip, read by the client-facing dmabuf module
    /// to advertise formats that allow zero-copy forwarding.
    pub host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
}

impl std::fmt::Debug for WaylandConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaylandConfig")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("title", &self.title)
            .field("fullscreen", &self.fullscreen)
            .field("use_vulkan", &self.use_vulkan)
            .field("host_wayland_display", &self.host_wayland_display)
            .finish()
    }
}

impl Clone for WaylandConfig {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            title: self.title.clone(),
            fullscreen: self.fullscreen,
            use_vulkan: self.use_vulkan,
            host_wayland_display: self.host_wayland_display.clone(),
            committed_frame_rx: None, // Receiver is not cloneable.
            cursor_rx: None,
            detected_refresh_mhz: self.detected_refresh_mhz.clone(),
            host_dmabuf_formats: Arc::new(
                parking_lot::Mutex::new(std::collections::HashMap::new()),
            ),
        }
    }
}

impl Default for WaylandConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            title: "gamecomp".to_string(),
            fullscreen: false,
            use_vulkan: true,
            host_wayland_display: None,
            committed_frame_rx: None,
            cursor_rx: None,
            detected_refresh_mhz: Arc::new(AtomicU32::new(0)),
            host_dmabuf_formats: Arc::new(
                parking_lot::Mutex::new(std::collections::HashMap::new()),
            ),
        }
    }
}

/// Nested Wayland backend.
///
/// Presents composited frames as a window inside another Wayland compositor.
/// Uses a Vulkan swapchain for efficient presentation.
pub struct WaylandBackend {
    /// Configuration.
    config: WaylandConfig,
    /// Current window dimensions.
    width: u32,
    height: u32,
    /// Fake connector info for the wayland "display".
    connector_info: Vec<ConnectorInfo>,
    /// Supported formats (common Vulkan swapchain formats).
    scanout_formats: Vec<DrmFormat>,
    /// Backend capabilities (limited compared to DRM).
    caps: BackendCaps,
    /// Frame counter.
    frame_count: u64,
    /// Whether a frame callback is pending (throttle presentation).
    frame_callback_pending: bool,
    /// Whether the window is focused.
    focused: bool,
    /// Channel receiver for events from the host event dispatch thread.
    event_rx: Option<std::sync::mpsc::Receiver<WaylandEvent>>,
    /// Shared flag to signal the event thread to stop.
    running: Arc<AtomicBool>,
    /// Event dispatch thread handle.
    event_thread: Option<std::thread::JoinHandle<()>>,
    /// Shared atomic for detected host display refresh rate (millihertz).
    /// 0 = not yet detected. Written by the event thread.
    detected_refresh_mhz: Arc<AtomicU32>,
}

impl WaylandBackend {
    /// Create a new wayland backend with the given configuration.
    pub fn new(config: WaylandConfig) -> Self {
        let width = config.width;
        let height = config.height;
        let detected_refresh_mhz = config.detected_refresh_mhz.clone();
        Self {
            config,
            width,
            height,
            connector_info: Vec::new(),
            scanout_formats: Vec::new(),
            caps: BackendCaps::default(),
            frame_count: 0,
            frame_callback_pending: false,
            focused: false,
            event_rx: None,
            running: Arc::new(AtomicBool::new(true)),
            event_thread: None,
            detected_refresh_mhz,
        }
    }

    /// Default wayland backend (1280x720 window).
    pub fn new_default() -> Self {
        Self::new(WaylandConfig::default())
    }

    /// Drain pending events from the host compositor.
    ///
    /// Returns events that the main thread should process (input forwarding, etc.).
    pub fn drain_events(&mut self) -> Vec<WaylandEvent> {
        let Some(ref rx) = self.event_rx else {
            return Vec::new();
        };

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match &event {
                WaylandEvent::Resized {
                    width,
                    height,
                    physical_width,
                    physical_height,
                } => {
                    self.width = *width;
                    self.height = *height;
                    // Update connector info.
                    if let Some(ci) = self.connector_info.first_mut() {
                        ci.name = format!("WAYLAND-1 ({}x{})", width, height);
                    }
                    info!(
                        width,
                        height, physical_width, physical_height, "wayland window resized"
                    );
                }
                WaylandEvent::CloseRequested => {
                    info!("wayland window close requested");
                }
                WaylandEvent::FrameCallback => {
                    self.frame_callback_pending = false;
                }
                WaylandEvent::FocusIn => {
                    self.focused = true;
                }
                WaylandEvent::FocusOut => {
                    self.focused = false;
                }
                _ => {}
            }
            events.push(event);
        }
        events
    }

    /// Whether the window currently has focus.
    #[inline(always)]
    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// Whether the backend's event loop is still running.
    ///
    /// Returns `false` after the host window is closed or the event
    /// thread exits. The render thread should check this and trigger
    /// global shutdown when it returns `false`.
    #[inline(always)]
    pub fn is_alive(&self) -> bool {
        // Ordering: Acquire to see the Release store from the event thread.
        self.running.load(Ordering::Acquire)
    }

    /// Current window size.
    #[inline(always)]
    pub fn window_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Detected host display refresh rate in Hz, or 0 if not yet detected.
    ///
    /// The event thread writes this when it receives `wl_output.mode` from
    /// the host compositor. The main thread polls it to update the FPS limiter.
    #[inline(always)]
    pub fn detected_refresh_hz(&self) -> u32 {
        // Ordering: Relaxed — main thread polls this periodically.
        let mhz = self.detected_refresh_mhz.load(Ordering::Relaxed);
        if mhz > 0 { (mhz + 500) / 1000 } else { 0 }
    }

    /// Get the shared refresh rate atomic (for passing to the main thread).
    pub fn detected_refresh_mhz(&self) -> &Arc<AtomicU32> {
        &self.detected_refresh_mhz
    }
}

impl Backend for WaylandBackend {
    fn init(&mut self) -> anyhow::Result<()> {
        info!(
            width = self.width,
            height = self.height,
            title = %self.config.title,
            "initializing wayland backend"
        );

        // Create fake connector info for the wayland window.
        // SAFETY: These handles are never used for DRM operations.
        self.connector_info.push(ConnectorInfo {
            handle: unsafe { std::mem::transmute::<u32, connector::Handle>(1u32) },
            crtc: unsafe { std::mem::transmute::<u32, crtc::Handle>(1u32) },
            name: format!("WAYLAND-1 ({}x{})", self.width, self.height),
            mode: unsafe { std::mem::zeroed() },
            physical_size_mm: (0, 0),
            vrr_enabled: false,
        });

        // Common Vulkan swapchain formats.
        self.scanout_formats = vec![
            DrmFormat {
                code: DrmFourcc::Argb8888,
                modifier: DrmModifier::Linear,
            },
            DrmFormat {
                code: DrmFourcc::Xrgb8888,
                modifier: DrmModifier::Linear,
            },
            DrmFormat {
                code: DrmFourcc::Abgr8888,
                modifier: DrmModifier::Linear,
            },
            DrmFormat {
                code: DrmFourcc::Xbgr8888,
                modifier: DrmModifier::Linear,
            },
        ];

        // Nested mode has limited capabilities.
        self.caps = BackendCaps {
            vrr: false,
            hdr: false, // Host compositor may support it, but we don't expose it yet.
            tearing: false,
            explicit_sync: false,
            modifiers: false,
        };

        // Set up event channel.
        let (tx, rx) = std::sync::mpsc::channel();
        self.event_rx = Some(rx);

        // Spawn host event dispatch thread.
        let running = self.running.clone();
        let detected_refresh_mhz = self.detected_refresh_mhz.clone();
        let host_dmabuf_formats = self.config.host_dmabuf_formats.clone();
        let width = self.width;
        let height = self.height;
        let title = self.config.title.clone();
        let host_display = self.config.host_wayland_display.clone();
        let committed_rx = self.config.committed_frame_rx.take();
        let cursor_rx = self.config.cursor_rx.take();
        self.event_thread = Some(
            std::thread::Builder::new()
                .name("gamecomp-wayland".to_string())
                .spawn(move || {
                    wayland_event_loop(HostLoopParams {
                        running,
                        tx,
                        width,
                        height,
                        title,
                        host_display,
                        committed_rx,
                        cursor_rx,
                        detected_refresh_mhz,
                        host_dmabuf_formats,
                    });
                })
                .context("failed to spawn wayland event thread")?,
        );

        info!("wayland backend initialized");
        Ok(())
    }

    fn connectors(&self) -> &[ConnectorInfo] {
        &self.connector_info
    }

    fn capabilities(&self) -> BackendCaps {
        self.caps
    }

    fn scanout_formats(&self) -> &[DrmFormat] {
        &self.scanout_formats
    }

    fn import_dmabuf(&mut self, dmabuf: &DmaBuf) -> anyhow::Result<Framebuffer> {
        // In wayland mode, we track the dmabuf metadata.
        // Actual presentation goes through the Vulkan swapchain.
        Ok(Framebuffer {
            handle: unsafe {
                std::mem::transmute::<u32, framebuffer::Handle>((self.frame_count as u32).max(1))
            },
            format: dmabuf.format,
            modifier: dmabuf.modifier,
            size: (dmabuf.width, dmabuf.height),
        })
    }

    fn try_direct_scanout(&mut self, _fb: &Framebuffer) -> anyhow::Result<bool> {
        // Direct scanout is not possible in wayland mode —
        // we must always composite to the Vulkan swapchain.
        Ok(false)
    }

    fn present(&mut self, _fb: &Framebuffer) -> anyhow::Result<FlipResult> {
        // TODO: Present to Vulkan swapchain.
        // For now, simulate a successful flip.
        self.frame_count += 1;
        self.frame_callback_pending = true;
        Ok(FlipResult::Queued)
    }

    fn drm_fd(&self) -> Option<RawFd> {
        // No DRM fd in wayland mode.
        None
    }

    fn handle_page_flip(&mut self) -> anyhow::Result<Option<u64>> {
        // In wayland mode, "page flip" completion comes from the host's frame callback.
        Ok(None)
    }

    fn set_vrr(&mut self, _enabled: bool) -> anyhow::Result<()> {
        // VRR not available in wayland mode.
        warn!("VRR not supported in wayland mode");
        Ok(())
    }
}

impl Drop for WaylandBackend {
    fn drop(&mut self) {
        // Signal event thread to stop.
        // Ordering: Release to ensure the write is visible to the event thread.
        self.running.store(false, Ordering::Release);
        if let Some(thread) = self.event_thread.take() {
            let _ = thread.join();
        }
        info!("wayland backend shut down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wayland_backend_creates_with_defaults() {
        let backend = WaylandBackend::new_default();
        assert_eq!(backend.width, 1280);
        assert_eq!(backend.height, 720);
        assert!(backend.connectors().is_empty()); // Not yet initialized.
    }

    #[test]
    fn wayland_backend_custom_config() {
        let config = WaylandConfig {
            width: 1920,
            height: 1080,
            title: "test".to_string(),
            fullscreen: true,
            use_vulkan: true,
            host_wayland_display: None,
            committed_frame_rx: None,
            cursor_rx: None,
            detected_refresh_mhz: Arc::new(AtomicU32::new(0)),
            host_dmabuf_formats: Arc::new(
                parking_lot::Mutex::new(std::collections::HashMap::new()),
            ),
        };
        let backend = WaylandBackend::new(config);
        assert_eq!(backend.width, 1920);
        assert_eq!(backend.height, 1080);
    }

    #[test]
    fn wayland_backend_no_direct_scanout() {
        let mut backend = WaylandBackend::new_default();
        let _ = backend.init();
        let fb = Framebuffer {
            handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
            format: DrmFourcc::Argb8888,
            modifier: DrmModifier::Linear,
            size: (1280, 720),
        };
        // Nested mode never supports direct scanout.
        assert!(!backend.try_direct_scanout(&fb).unwrap());
    }

    #[test]
    fn wayland_backend_caps_are_limited() {
        let mut backend = WaylandBackend::new_default();
        let _ = backend.init();
        let caps = backend.capabilities();
        assert!(!caps.vrr);
        assert!(!caps.hdr);
        assert!(!caps.tearing);
        assert!(!caps.explicit_sync);
    }

    #[test]
    fn wayland_backend_present_increments_frame() {
        let mut backend = WaylandBackend::new_default();
        let _ = backend.init();
        let fb = Framebuffer {
            handle: unsafe { std::mem::transmute::<u32, drm::control::framebuffer::Handle>(1u32) },
            format: DrmFourcc::Argb8888,
            modifier: DrmModifier::Linear,
            size: (1280, 720),
        };
        let result = backend.present(&fb);
        assert!(matches!(result.unwrap(), FlipResult::Queued));
        assert_eq!(backend.frame_count, 1);
    }
}
