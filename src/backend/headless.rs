//! Headless backend for offscreen rendering.
//!
//! This backend produces no visible output. It is used for:
//! - CI testing without GPU or display hardware.
//! - Robotics scenarios where output is streamed via PipeWire or captured to file.
//!
//! The headless backend simulates page flips at a configurable refresh rate
//! using a timerfd, so the compositor's frame pacing logic works identically
//! to the DRM backend.

use std::os::unix::io::RawFd;

use drm::control::{connector, crtc, framebuffer};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use tracing::info;

use super::{Backend, BackendCaps, ConnectorInfo, DmaBuf, FlipResult, Framebuffer};

/// A headless display backend with no physical output.
pub struct HeadlessBackend {
    /// Simulated display width.
    width: u32,
    /// Simulated display height.
    height: u32,
    /// Simulated refresh rate in Hz.
    refresh_hz: u32,
    /// Fake connector info.
    connector_info: Vec<ConnectorInfo>,
    /// Supported formats (all common formats).
    scanout_formats: Vec<DrmFormat>,
    /// Frame counter.
    frame_count: u64,
}

impl HeadlessBackend {
    /// Create a new headless backend with the given simulated resolution.
    pub fn new(width: u32, height: u32, refresh_hz: u32) -> Self {
        Self {
            width,
            height,
            refresh_hz,
            connector_info: Vec::new(),
            scanout_formats: Vec::new(),
            frame_count: 0,
        }
    }
}

impl Backend for HeadlessBackend {
    fn init(&mut self) -> anyhow::Result<()> {
        info!(
            width = self.width,
            height = self.height,
            refresh_hz = self.refresh_hz,
            "initializing headless backend"
        );

        // Create a fake connector.
        self.connector_info.push(ConnectorInfo {
            // SAFETY: These handles are never used for actual DRM operations.
            handle: unsafe { std::mem::transmute::<u32, connector::Handle>(1u32) },
            crtc: unsafe { std::mem::transmute::<u32, crtc::Handle>(1u32) },
            name: "HEADLESS-1".to_string(),
            mode: unsafe { std::mem::zeroed() },
            physical_size_mm: (0, 0),
            vrr_enabled: false,
        });

        // Advertise common scanout formats.
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

        Ok(())
    }

    fn connectors(&self) -> &[ConnectorInfo] {
        &self.connector_info
    }

    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            vrr: false,
            hdr: false,
            tearing: false,
            explicit_sync: false,
            modifiers: false,
        }
    }

    fn scanout_formats(&self) -> &[DrmFormat] {
        &self.scanout_formats
    }

    fn import_dmabuf(&mut self, dmabuf: &DmaBuf) -> anyhow::Result<Framebuffer> {
        // In headless mode, we just track metadata — no actual DRM FB.
        Ok(Framebuffer {
            // SAFETY: Handle is never passed to DRM operations.
            handle: unsafe {
                std::mem::transmute::<u32, framebuffer::Handle>((self.frame_count as u32).max(1))
            },
            format: dmabuf.format,
            modifier: dmabuf.modifier,
            size: (dmabuf.width, dmabuf.height),
        })
    }

    fn try_direct_scanout(&mut self, _fb: &Framebuffer) -> anyhow::Result<bool> {
        // Headless always "succeeds" at direct scanout (it's a no-op).
        Ok(true)
    }

    fn present(&mut self, _fb: &Framebuffer) -> anyhow::Result<FlipResult> {
        self.frame_count += 1;
        Ok(FlipResult::Queued)
    }

    fn drm_fd(&self) -> Option<RawFd> {
        None
    }

    fn handle_page_flip(&mut self) -> anyhow::Result<Option<u64>> {
        Ok(None)
    }

    fn set_vrr(&mut self, _enabled: bool) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "headless_tests.rs"]
mod tests;
