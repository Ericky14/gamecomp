//! Vulkan DMA-BUF blit pipeline for wayland backend.
//!
//! Imports client DMA-BUFs into Vulkan VkImages, blits them onto pre-allocated
//! output images with explicit DRM format modifiers, and exports the result as
//! DMA-BUF fds for the host compositor.
//!
//! Used as a fallback when the host compositor does not support the client's
//! format/modifier pair, or when explicitly forced via `GAMECOMP_FORCE_BLIT=1`.
//!
//! ## Extensions used
//!
//! - `VK_KHR_external_memory_fd` — export/import memory as fd
//! - `VK_EXT_external_memory_dma_buf` — DMA-BUF as external memory type
//! - `VK_EXT_image_drm_format_modifier` — create images with explicit DRM modifiers
//! - `VK_KHR_image_format_list` — required by drm_format_modifier

use std::collections::HashMap;
use std::os::unix::io::OwnedFd;

use anyhow::{Context, bail};
use ash::vk;
use drm_fourcc::DrmFourcc;
use tracing::{debug, info, trace, warn};
#[path = "vulkan_blitter_blit.rs"]
mod blit;
#[path = "vulkan_blitter_device.rs"]
mod device;
#[path = "vulkan_blitter_images.rs"]
mod images;
#[path = "vulkan_blitter_modifiers.rs"]
mod modifiers;

#[cfg(test)]
#[path = "vulkan_blitter_tests.rs"]
mod tests;

// ─── Module-level constants ─────────────────────────────────────────

/// DRM_FORMAT_MOD_INVALID — sentinel meaning "no explicit modifier".
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// DRM_FORMAT_MOD_LINEAR — linear (scanline-order) tiling.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// Vulkan usage flags for output images (compute destination + scanout export).
const OUTPUT_IMAGE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::STORAGE.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw(),
);

/// Vulkan usage flags for imported client images (compute source).
const CLIENT_IMAGE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::TRANSFER_SRC.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
);

/// Required format features for output image modifiers.
///
/// Modifiers must support all four feature flags to be usable for
/// compute shader composition + export.
const OUTPUT_MODIFIER_FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::from_raw(
    vk::FormatFeatureFlags::SAMPLED_IMAGE.as_raw()
        | vk::FormatFeatureFlags::STORAGE_IMAGE.as_raw()
        | vk::FormatFeatureFlags::TRANSFER_DST.as_raw()
        | vk::FormatFeatureFlags::TRANSFER_SRC.as_raw(),
);

/// Required format features for importable output modifiers (GBM path).
const IMPORTABLE_MODIFIER_FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::from_raw(
    vk::FormatFeatureFlags::TRANSFER_DST.as_raw() | vk::FormatFeatureFlags::TRANSFER_SRC.as_raw(),
);

/// Required format features for client import modifiers.
const IMPORT_MODIFIER_FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::from_raw(
    vk::FormatFeatureFlags::TRANSFER_SRC.as_raw() | vk::FormatFeatureFlags::SAMPLED_IMAGE.as_raw(),
);

/// Vulkan format used for XRGB8888 / ARGB8888 DMA-BUFs.
const VK_FORMAT_XRGB: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// Standard single-layer color subresource range (mip 0, layer 0).
///
/// Used throughout barrier, image view, and copy operations.
const COLOR_SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// DMA-BUF plane descriptor for an exported output image.
#[derive(Debug)]
pub struct ExportedPlane {
    pub fd: OwnedFd,
    pub offset: u32,
    pub stride: u32,
}

/// Exported output buffer ready to send to the host compositor.
#[derive(Debug)]
pub struct ExportedBuffer {
    pub planes: Vec<ExportedPlane>,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub modifier: u64,
    /// Index of the output image that was blitted to (0..output_count).
    /// Used by the caller to cache DRM framebuffers per output image.
    pub buffer_index: usize,
}

/// A pre-allocated output image backed by exportable Vulkan memory.
struct OutputImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// DRM format modifier selected for this image.
    modifier: u64,
    /// Per-plane DMA-BUF export info.
    planes: Vec<OutputPlaneInfo>,
    width: u32,
    height: u32,
    format: u32,
    /// Whether this image has been blitted to at least once.
    /// Controls the `oldLayout` in the acquire barrier:
    /// - `false` → `UNDEFINED` (fresh image, no metadata to preserve)
    /// - `true`  → `GENERAL`  (preserves block-linear tiling metadata)
    blitted: bool,
    /// Whether this blitter owns the plane fds (self-allocated via Vulkan).
    /// GBM-imported outputs have fds owned by `GbmOutputBuffer` — Drop must
    /// not close them.
    owns_fds: bool,
}

struct OutputPlaneInfo {
    fd: i32,
    offset: u32,
    stride: u32,
}

/// Compute contain-style scaling: fit content inside output while preserving
/// aspect ratio. Returns (offset_x, offset_y, dst_width, dst_height).
#[inline(always)]
fn compute_contain_rect(src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> (i32, i32, i32, i32) {
    // Compute scale factor for contain (fit) behavior.
    let scale_x = out_w as f64 / src_w as f64;
    let scale_y = out_h as f64 / src_h as f64;
    let scale = scale_x.min(scale_y);

    // Scaled destination dimensions.
    let dst_w = (src_w as f64 * scale).round() as i32;
    let dst_h = (src_h as f64 * scale).round() as i32;

    // Center the content within the output.
    let offset_x = (out_w as i32 - dst_w) / 2;
    let offset_y = (out_h as i32 - dst_h) / 2;

    (offset_x, offset_y, dst_w, dst_h)
}

/// Push constants for the blit compute shader (blit.comp).
///
/// Must match the GLSL `PushConstants` layout exactly:
/// - `dstSize`:       output image dimensions
/// - `contentOffset`: top-left of the content region (letterbox/pillarbox)
/// - `contentSize`:   scaled content region dimensions
#[repr(C)]
#[derive(Clone, Copy)]
struct BlitPushConstants {
    dst_w: u32,
    dst_h: u32,
    content_offset_x: i32,
    content_offset_y: i32,
    content_w: u32,
    content_h: u32,
}

/// Vulkan-based DMA-BUF compositor.
///
/// Imports client DMA-BUFs, composites them onto output images via a
/// compute shader with contain-rect scaling, and exports the result
/// for DRM scanout. Uses a single compute dispatch per frame instead
/// of separate vkCmdClearColorImage + vkCmdBlitImage commands.
pub struct VulkanBlitter {
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,

    /// Pre-allocated output images (triple-buffered).
    output_images: Vec<OutputImage>,
    /// Current output image index (round-robin).
    output_index: usize,

    /// Cache of imported VkImages keyed by DMA-BUF (dev, ino).
    import_cache: HashMap<(u64, u64), ImportedImage>,

    /// Extension function pointers.
    ext_memory_fd: ash::khr::external_memory_fd::Device,
    ext_drm_modifier: ash::ext::image_drm_format_modifier::Device,

    /// All valid DRM modifiers for B8G8R8A8_UNORM (single-plane, non-INVALID).
    /// Used when importing client DMA-BUFs — the driver picks the matching one.
    import_modifiers: Vec<u64>,

    // --- Compute pipeline for shader-based composition ---
    /// Linear sampler for bilinear-filtered client texture reads.
    sampler: vk::Sampler,
    /// Descriptor set layout: binding 0 = combined_image_sampler,
    /// binding 1 = storage_image.
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Pipeline layout with push constants for BlitPushConstants.
    pipeline_layout: vk::PipelineLayout,
    /// Compute pipeline for blit.comp shader.
    compute_pipeline: vk::Pipeline,
    /// Descriptor pool for per-frame descriptor set updates.
    descriptor_pool: vk::DescriptorPool,
    /// Single descriptor set — updated every frame before dispatch.
    descriptor_set: vk::DescriptorSet,
    /// Shader module for blit.comp (destroyed on drop).
    shader_module: vk::ShaderModule,
    /// Per-output image views (STORAGE, for compute shader writes).
    output_image_views: Vec<vk::ImageView>,
}

struct ImportedImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// Image view for compute shader sampling (SAMPLED usage).
    image_view: vk::ImageView,
    width: u32,
    height: u32,
}

impl VulkanBlitter {
    /// Create a new Vulkan blitter with scanout modifiers from the DRM plane.
    ///
    /// `scanout_modifiers` should be the output of
    /// `DrmBackend::query_primary_plane_modifiers()` — the intersection of
    /// what the display plane can scan out and what Vulkan can export is
    /// computed internally. If empty, queries Vulkan for the best modifier.
    pub fn new_with_scanout_modifiers(
        width: u32,
        height: u32,
        scanout_modifiers: &[u64],
    ) -> anyhow::Result<Self> {
        Self::create_with_outputs(width, height, scanout_modifiers)
    }

    /// Create a new Vulkan blitter with a single forced modifier (for tests).
    pub fn new_with_modifier(
        width: u32,
        height: u32,
        forced_modifier: Option<u64>,
    ) -> anyhow::Result<Self> {
        match forced_modifier {
            Some(m) => Self::create_with_outputs(width, height, &[m]),
            None => Self::create_with_outputs(width, height, &[]),
        }
    }

    /// Create a new Vulkan blitter for the given output dimensions.
    pub fn new(width: u32, height: u32) -> anyhow::Result<Self> {
        Self::create_with_outputs(width, height, &[])
    }

    /// Internal constructor that sets up the Vulkan device AND allocates
    /// output images (self-allocated, Vulkan-exported). Used by `new()`,
    /// `new_with_modifier()`, and tests that don't need GBM.
    fn create_with_outputs(
        width: u32,
        height: u32,
        scanout_modifiers: &[u64],
    ) -> anyhow::Result<Self> {
        let mut blitter = Self::create()?;

        // Allocate triple-buffered output images.
        blitter.allocate_output_images(width, height, 3, scanout_modifiers)?;

        info!(
            width,
            height,
            num_outputs = blitter.output_images.len(),
            modifier = format!("0x{:016x}", blitter.output_images[0].modifier),
            "blitter: initialized with self-allocated outputs"
        );

        Ok(blitter)
    }

    /// Create a blitter for the GBM-backed output path (DRM backend).
    ///
    /// Sets up the Vulkan device but does NOT allocate output images.
    /// The caller must:
    /// 1. Call `compute_importable_modifiers()` to get Vulkan-compatible modifiers.
    /// 2. Allocate GBM buffers with those modifiers.
    /// 3. Call `import_output_images()` with the GBM DMA-BUF descriptors.
    pub fn new_for_import() -> anyhow::Result<Self> {
        Self::create()
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        // SAFETY: physical_device is valid; returns memory properties struct.
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        (0..mem_props.memory_type_count).find(|&i| {
            (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
        })
    }
}

impl Drop for VulkanBlitter {
    fn drop(&mut self) {
        // SAFETY: Waits for all GPU work to complete before destroying resources.
        let _ = unsafe { self.device.device_wait_idle() };

        // Destroy imported images and their views.
        for (_, img) in self.import_cache.drain() {
            // SAFETY: ImageView, Image, and Memory are valid; GPU is idle.
            unsafe {
                self.device.destroy_image_view(img.image_view, None);
                self.device.destroy_image(img.image, None);
                self.device.free_memory(img.memory, None);
            }
        }

        // Destroy output image views.
        for &view in &self.output_image_views {
            // SAFETY: ImageView is valid; GPU is idle.
            unsafe {
                self.device.destroy_image_view(view, None);
            }
        }

        // Destroy output images. Only close fds we own (self-allocated).
        // GBM-imported outputs have fds owned by GbmOutputBuffer — closing
        // them here would cause a double-close.
        for out in &self.output_images {
            // SAFETY: Images and memory are valid; GPU is idle.
            unsafe {
                if out.owns_fds {
                    for plane in &out.planes {
                        libc::close(plane.fd);
                    }
                }
                self.device.destroy_image(out.image, None);
                self.device.free_memory(out.memory, None);
            }
        }

        // Destroy compute pipeline resources in reverse creation order.
        // SAFETY: All resources are valid handles; GPU is idle (waited above).
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.compute_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device.destroy_sampler(self.sampler, None);
        }

        // SAFETY: All dependent resources (images, memory, pipeline, etc.)
        // have been destroyed above. Device and instance are destroyed last,
        // in reverse creation order.
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Convert a DRM fourcc format to a VkFormat.
fn drm_format_to_vk(drm_format: u32) -> anyhow::Result<vk::Format> {
    match drm_format {
        x if x == DrmFourcc::Xrgb8888 as u32 || x == DrmFourcc::Argb8888 as u32 => {
            Ok(vk::Format::B8G8R8A8_UNORM)
        }
        x if x == DrmFourcc::Xbgr8888 as u32 || x == DrmFourcc::Abgr8888 as u32 => {
            Ok(vk::Format::R8G8B8A8_UNORM)
        }
        x if x == DrmFourcc::Rgb565 as u32 => Ok(vk::Format::R5G6B5_UNORM_PACK16),
        other => bail!("unsupported DRM format 0x{:08x} for Vulkan import", other),
    }
}
