//! Render pipeline abstraction.
//!
//! Defines the [`Renderer`] trait that decouples the composition pipeline from
//! specific GPU APIs (Vulkan, V3D, V4, software). Each renderer owns its GPU
//! state exclusively on the render thread.
//!
//! The pipeline is split into stages, each represented by a trait:
//! - [`Renderer`] — core composition (blit, clear, present)
//! - [`PostProcessor`] — optional post-processing (FSR, NIS, CAS, tonemapping)
//! - [`ColorPipeline`] — color management (SDR→HDR, gamut mapping, LUT application)
//!
//! Hardware-specific renderers (e.g., Pi5 V3D/V4) implement these traits
//! and are selected at runtime based on detected hardware.

pub mod color;
pub mod post_process;

use std::os::unix::io::BorrowedFd;

use anyhow::Result;

/// Pixel format + modifier pair for buffer negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferFormat {
    /// DRM fourcc format code.
    pub fourcc: u32,
    /// DRM modifier (e.g., linear, tiled, compressed).
    pub modifier: u64,
}

/// Parameters for importing a DMA-BUF into the renderer.
#[derive(Debug)]
pub struct DmaBufImport<'a> {
    /// File descriptor for the DMA-BUF plane.
    pub fd: BorrowedFd<'a>,
    /// Buffer width in pixels.
    pub width: u32,
    /// Buffer height in pixels.
    pub height: u32,
    /// DRM fourcc format code.
    pub format: u32,
    /// DRM modifier.
    pub modifier: u64,
    /// Byte offset into the fd.
    pub offset: u32,
    /// Stride (bytes per row).
    pub stride: u32,
}

/// An exported buffer ready for presentation to a display backend or host compositor.
#[derive(Debug)]
pub struct ExportedFrame {
    /// DMA-BUF planes.
    pub planes: Vec<ExportedPlane>,
    /// Buffer width in pixels.
    pub width: u32,
    /// Buffer height in pixels.
    pub height: u32,
    /// DRM fourcc format.
    pub format: u32,
    /// DRM modifier.
    pub modifier: u64,
}

/// A single plane of an exported DMA-BUF.
#[derive(Debug)]
pub struct ExportedPlane {
    /// File descriptor for this plane.
    pub fd: std::os::unix::io::OwnedFd,
    /// Byte offset into the fd.
    pub offset: u32,
    /// Stride (bytes per row).
    pub stride: u32,
}

/// Capabilities reported by a renderer.
#[derive(Debug, Clone, Default)]
pub struct RendererCaps {
    /// Human-readable renderer name (e.g., "Vulkan (NVIDIA RTX 4080)").
    pub name: &'static str,
    /// Supported input formats for importing client buffers.
    pub import_formats: Vec<BufferFormat>,
    /// Supported output formats for scanout/export.
    pub export_formats: Vec<BufferFormat>,
    /// Whether the renderer supports compute-based composition.
    pub compute_composite: bool,
    /// Whether the renderer supports explicit sync (timeline semaphores / syncobj).
    pub explicit_sync: bool,
    /// Maximum texture dimensions.
    pub max_texture_size: u32,
}

/// The core renderer trait.
///
/// Renderers own all GPU state and run exclusively on the render thread.
/// They import client DMA-BUFs, perform composition (blit + scale),
/// and export the result as a new DMA-BUF for the display backend.
///
/// Implementations:
/// - `VulkanRenderer` — production path using `ash` (desktop GPUs, NVIDIA, AMD)
/// - `V3dRenderer` — Broadcom V3D/V4 for Raspberry Pi 5 (future)
/// - `SoftwareRenderer` — CPU fallback for headless/CI (future)
pub trait Renderer: Send {
    /// Query renderer capabilities.
    fn caps(&self) -> &RendererCaps;

    /// Import a client DMA-BUF for use as a source texture.
    ///
    /// The returned handle is opaque and renderer-specific. It remains valid
    /// until `release_import()` is called or the renderer is dropped.
    fn import_dmabuf(&mut self, import: &DmaBufImport<'_>) -> Result<ImportHandle>;

    /// Release a previously imported buffer.
    fn release_import(&mut self, handle: ImportHandle);

    /// Blit a source buffer to the output with contain-style scaling.
    ///
    /// Clears the output to `clear_color`, then blits the source centered
    /// with aspect ratio preserved (letterbox/pillarbox). Returns the
    /// exported output buffer.
    fn blit_contain(
        &mut self,
        source: ImportHandle,
        src_width: u32,
        src_height: u32,
        clear_color: [f32; 4],
    ) -> Result<ExportedFrame>;

    /// Resize the output dimensions. Called when the window/display is resized.
    fn resize_output(&mut self, width: u32, height: u32) -> Result<()>;

    /// Number of output image slots (for buffer rotation).
    fn output_count(&self) -> usize;

    /// Current output image index (incremented after each blit).
    fn output_index(&self) -> usize;

    /// Wait for all pending GPU work to complete.
    fn wait_idle(&mut self) -> Result<()>;
}

/// Opaque handle to an imported client buffer.
///
/// The handle is valid only for the renderer that created it.
/// It is lightweight (fits in a register) and `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImportHandle(pub u64);
