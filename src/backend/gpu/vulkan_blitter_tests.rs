//! Tests for the Vulkan DMA-BUF blit pipeline.
//!
//! Contains unit tests for pure functions (`compute_contain_rect`, `drm_format_to_vk`),
//! GPU integration tests that exercise the full blit pipeline on real hardware,
//! and test-only helper methods on `VulkanBlitter`.

use super::*;

// ─── Test-only helpers on VulkanBlitter ─────────────────────────────

impl VulkanBlitter {
    /// Create a test DMA-BUF filled with a solid B8G8R8A8 color.
    ///
    /// Returns an `OwnedFd` for the LINEAR DMA-BUF. The caller must keep
    /// it alive until `blit()` completes (it dups internally).
    fn create_test_source_dmabuf(
        &mut self,
        width: u32,
        height: u32,
        bgra: [u8; 4],
    ) -> anyhow::Result<(OwnedFd, u32, u32)> {
        use std::os::unix::io::FromRawFd;

        let buf_size = (width as u64) * (height as u64) * 4;

        // --- HOST_VISIBLE staging buffer with pixel data ---
        let buf_info = vk::BufferCreateInfo::default()
            .size(buf_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        // SAFETY: Device is valid; buffer create info is fully initialized.
        let staging_buf = unsafe { self.device.create_buffer(&buf_info, None) }
            .context("test: create staging buffer")?;
        // SAFETY: Device and buffer are valid.
        let buf_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buf) };
        let buf_mem_type = self
            .find_memory_type(
                buf_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("test: no host-visible memory")?;
        let buf_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(buf_reqs.size)
            .memory_type_index(buf_mem_type);
        // SAFETY: Valid allocation info with verified memory type.
        let staging_mem = unsafe { self.device.allocate_memory(&buf_alloc, None) }
            .context("test: alloc staging")?;
        // SAFETY: Buffer and memory are valid; offset 0.
        unsafe { self.device.bind_buffer_memory(staging_buf, staging_mem, 0) }
            .context("test: bind staging")?;

        // SAFETY: Memory is HOST_VISIBLE + HOST_COHERENT; map covers buf_size bytes.
        unsafe {
            let ptr = self
                .device
                .map_memory(staging_mem, 0, buf_size, vk::MemoryMapFlags::empty())
                .context("test: map staging")?;
            let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, buf_size as usize);
            for pixel in slice.chunks_exact_mut(4) {
                pixel.copy_from_slice(&bgra);
            }
            self.device.unmap_memory(staging_mem);
        }

        // --- Source image: DRM_FORMAT_MODIFIER_EXT, LINEAR, DMA-BUF export ---
        let modifiers = [0u64]; // LINEAR
        let mut drm_mod =
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&modifiers);
        let mut ext_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let vk_formats = [vk::Format::B8G8R8A8_UNORM];
        let mut fmt_list = vk::ImageFormatListCreateInfo::default().view_formats(&vk_formats);
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut drm_mod)
            .push_next(&mut ext_info)
            .push_next(&mut fmt_list);

        // SAFETY: Valid image create info with DRM modifier chain.
        let image = unsafe { self.device.create_image(&img_info, None) }
            .context("test: create source image")?;
        // SAFETY: Device and image are valid.
        let img_reqs = unsafe { self.device.get_image_memory_requirements(image) };

        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let img_mem_type = self
            .find_memory_type(
                img_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .context("test: no device-local memory")?;
        let img_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(img_reqs.size)
            .memory_type_index(img_mem_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        // SAFETY: Valid alloc with export + dedicated chains.
        let img_mem = unsafe { self.device.allocate_memory(&img_alloc, None) }
            .context("test: alloc source image")?;
        // SAFETY: Image and memory are valid; offset 0 for dedicated alloc.
        unsafe { self.device.bind_image_memory(image, img_mem, 0) }
            .context("test: bind source image")?;

        // --- GPU copy: staging buffer → image ---
        // SAFETY: Fence is signaled (SIGNALED flag at creation or prior wait).
        unsafe {
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: All Vulkan handles valid. Image transitions and copy are
        // properly sequenced via pipeline barriers. Fence synchronizes completion.
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device
                .begin_command_buffer(self.command_buffer, &begin)?;

            // UNDEFINED → TRANSFER_DST (new image, content don't-care).
            let barrier = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,   // tightly packed
                buffer_image_height: 0, // tightly packed
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            };
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Release to FOREIGN_EXT so blit()'s acquire barrier works.
            let release = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[release],
            );

            self.device.end_command_buffer(self.command_buffer)?;
        }

        let submit =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
        // SAFETY: Queue, command buffer, and fence are valid.
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        // Export DMA-BUF fd.
        let fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(img_mem)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        // SAFETY: Memory was allocated with DMA_BUF_EXT export.
        let raw_fd = unsafe { self.ext_memory_fd.get_memory_fd(&fd_info) }
            .context("test: export source fd")?;

        // Query the actual stride from the driver. Some drivers add
        // stride alignment padding to LINEAR images beyond width*bpp.
        // SAFETY: Image and device are valid.
        let subresource = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        let layout = unsafe { self.device.get_image_subresource_layout(image, subresource) };
        let actual_stride = layout.row_pitch as u32;
        let actual_offset = layout.offset as u32;

        // Clean up Vulkan objects. The DMA-BUF fd keeps the kernel buffer alive.
        // SAFETY: GPU work completed (fence waited). Staging is transient.
        // The DMA-BUF fd holds a kernel ref to the underlying buffer.
        unsafe {
            self.device.destroy_buffer(staging_buf, None);
            self.device.free_memory(staging_mem, None);
            self.device.destroy_image(image, None);
            self.device.free_memory(img_mem, None);
        }

        // SAFETY: raw_fd is a valid DMA-BUF fd from Vulkan export.
        Ok((
            unsafe { OwnedFd::from_raw_fd(raw_fd) },
            actual_stride,
            actual_offset,
        ))
    }

    /// Read back an output image as tightly-packed B8G8R8A8 pixel data.
    ///
    /// Copies the output image to a staging buffer on the GPU, then maps
    /// it to CPU. Returns `width × height × 4` bytes.
    fn readback_output_pixels(&mut self, index: usize) -> anyhow::Result<Vec<u8>> {
        let width = self.output_images[index].width;
        let height = self.output_images[index].height;
        let image = self.output_images[index].image;
        let buf_size = (width as u64) * (height as u64) * 4;

        // --- HOST_VISIBLE staging buffer ---
        let buf_info = vk::BufferCreateInfo::default()
            .size(buf_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        // SAFETY: Device is valid.
        let staging_buf = unsafe { self.device.create_buffer(&buf_info, None) }
            .context("readback: create staging")?;
        // SAFETY: Device and buffer are valid.
        let buf_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buf) };
        let buf_mem_type = self
            .find_memory_type(
                buf_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("readback: no host-visible memory")?;
        let buf_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(buf_reqs.size)
            .memory_type_index(buf_mem_type);
        // SAFETY: Valid allocation info.
        let staging_mem = unsafe { self.device.allocate_memory(&buf_alloc, None) }
            .context("readback: alloc staging")?;
        // SAFETY: Buffer and memory are valid; offset 0.
        unsafe { self.device.bind_buffer_memory(staging_buf, staging_mem, 0) }
            .context("readback: bind staging")?;

        // --- GPU copy: output image → staging buffer ---
        // SAFETY: Fence is signaled from blit() completion.
        unsafe {
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: All handles valid. Barriers properly sequence the copy.
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device
                .begin_command_buffer(self.command_buffer, &begin)?;

            // Acquire from FOREIGN_EXT → TRANSFER_SRC_OPTIMAL.
            // The output image was released to FOREIGN_EXT by blit()'s final barrier.
            let acquire = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(self.queue_family_index)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[acquire],
            );

            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,   // tightly packed
                buffer_image_height: 0, // tightly packed
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            };
            self.device.cmd_copy_image_to_buffer(
                self.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buf,
                &[region],
            );

            // Release back to FOREIGN_EXT so the blitter can reuse the image.
            let release = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[release],
            );

            self.device.end_command_buffer(self.command_buffer)?;
        }

        let submit =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
        // SAFETY: Queue, command buffer, and fence are valid.
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        // Map and read.
        let mut pixels = vec![0u8; buf_size as usize];
        // SAFETY: Staging memory is HOST_VISIBLE + HOST_COHERENT. GPU work
        // completed (fence waited). Copy covers buf_size bytes.
        unsafe {
            let ptr = self
                .device
                .map_memory(staging_mem, 0, buf_size, vk::MemoryMapFlags::empty())
                .context("readback: map staging")?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, pixels.as_mut_ptr(), buf_size as usize);
            self.device.unmap_memory(staging_mem);
        }

        // Clean up.
        // SAFETY: GPU work completed (fence waited); staging objects are transient.
        unsafe {
            self.device.destroy_buffer(staging_buf, None);
            self.device.free_memory(staging_mem, None);
        }

        Ok(pixels)
    }
}

// ─── compute_contain_rect tests ─────────────────────────────────────

#[test]
fn contain_rect_same_size() {
    // Same source and output → fills entire output, no offset.
    let (x, y, w, h) = compute_contain_rect(1920, 1080, 1920, 1080);
    assert_eq!((x, y, w, h), (0, 0, 1920, 1080));
}

#[test]
fn contain_rect_square_into_wide() {
    // 500x500 → 3840x2160: limited by height (scale = 2160/500 = 4.32).
    // dst_h = 2160, dst_w = 500 * 4.32 = 2160, pillarboxed.
    let (x, y, w, h) = compute_contain_rect(500, 500, 3840, 2160);
    assert_eq!(h, 2160, "height should fill output");
    assert_eq!(w, 2160, "width should equal height (square)");
    assert_eq!(y, 0, "no vertical offset");
    assert_eq!(x, (3840 - 2160) / 2, "centered horizontally");
}

#[test]
fn contain_rect_wide_into_square() {
    // 1920x1080 → 1000x1000: limited by width (scale = 1000/1920 ≈ 0.5208).
    // dst_w = 1000, dst_h ≈ 563, letterboxed.
    let (x, y, w, h) = compute_contain_rect(1920, 1080, 1000, 1000);
    assert_eq!(w, 1000, "width should fill output");
    assert_eq!(x, 0, "no horizontal offset");
    assert!(h < 1000, "height should be less than output (letterbox)");
    assert!(y > 0, "should have vertical offset (centered)");
    // Verify centering: offset = (1000 - h) / 2
    assert_eq!(y, (1000 - h) / 2);
}

#[test]
fn contain_rect_tall_into_wide() {
    // 1080x1920 → 3840x2160: limited by height (scale = 2160/1920 = 1.125).
    // dst_h = 2160, dst_w = 1080 * 1.125 = 1215, pillarboxed wide.
    let (x, y, w, h) = compute_contain_rect(1080, 1920, 3840, 2160);
    assert_eq!(h, 2160, "height should fill output");
    assert_eq!(y, 0, "no vertical offset");
    assert!(w < 3840, "width should not fill output (pillarbox)");
    assert_eq!(x, (3840 - w) / 2, "centered horizontally");
}

#[test]
fn contain_rect_exact_double() {
    // 960x540 → 1920x1080: exact 2× scale, no bars.
    let (x, y, w, h) = compute_contain_rect(960, 540, 1920, 1080);
    assert_eq!((x, y, w, h), (0, 0, 1920, 1080));
}

#[test]
fn contain_rect_downscale() {
    // 3840x2160 → 1920x1080: exact 0.5× scale, no bars.
    let (x, y, w, h) = compute_contain_rect(3840, 2160, 1920, 1080);
    assert_eq!((x, y, w, h), (0, 0, 1920, 1080));
}

// ─── drm_format_to_vk tests ────────────────────────────────────────

#[test]
fn format_xrgb8888() {
    let vk_fmt = drm_format_to_vk(DrmFourcc::Xrgb8888 as u32).unwrap();
    assert_eq!(vk_fmt, vk::Format::B8G8R8A8_UNORM);
}

#[test]
fn format_argb8888() {
    let vk_fmt = drm_format_to_vk(DrmFourcc::Argb8888 as u32).unwrap();
    assert_eq!(vk_fmt, vk::Format::B8G8R8A8_UNORM);
}

#[test]
fn format_xbgr8888() {
    let vk_fmt = drm_format_to_vk(DrmFourcc::Xbgr8888 as u32).unwrap();
    assert_eq!(vk_fmt, vk::Format::R8G8B8A8_UNORM);
}

#[test]
fn format_abgr8888() {
    let vk_fmt = drm_format_to_vk(DrmFourcc::Abgr8888 as u32).unwrap();
    assert_eq!(vk_fmt, vk::Format::R8G8B8A8_UNORM);
}

#[test]
fn format_rgb565() {
    let vk_fmt = drm_format_to_vk(DrmFourcc::Rgb565 as u32).unwrap();
    assert_eq!(vk_fmt, vk::Format::R5G6B5_UNORM_PACK16);
}

#[test]
fn format_unsupported() {
    // NV12 is not in our mapping — should error.
    let result = drm_format_to_vk(DrmFourcc::Nv12 as u32);
    assert!(result.is_err());
}

// ─── barrier constant tests ────────────────────────────────────────

#[test]
fn queue_family_foreign_ext_value() {
    // VK_QUEUE_FAMILY_FOREIGN_EXT = ~2 = 0xFFFFFFFD
    assert_eq!(vk::QUEUE_FAMILY_FOREIGN_EXT, !2u32);
}

// ─── GPU integration tests ─────────────────────────────────────────
// These tests require a Vulkan-capable GPU. They create a real VulkanBlitter,
// fill a source DMA-BUF with known pixel data, blit it, and read back the
// output to verify correctness. Skipped gracefully if no GPU is available.

/// Helper: get the output index that was just written by the last blit().
fn last_output_index(blitter: &VulkanBlitter) -> usize {
    if blitter.output_index() == 0 {
        blitter.output_count() - 1
    } else {
        blitter.output_index() - 1
    }
}

/// Diagnostic test: verifies DMA-BUF export roundtrip at 256×256.
///
/// Uses auto-detected modifier (the same path as DRM presentation).
/// After blit, re-imports the exported DMA-BUF fd as a new VkImage
/// and reads it back to verify pixel data survives the
/// export → DMA-BUF fd → reimport cycle. This catches tiling/modifier
/// mismatches that produce static noise on screen.
#[test]
fn gpu_dmabuf_export_reimport_roundtrip() {
    // 128×128 solid magenta → 256×256 output with auto-detected modifier.
    let mut blitter = match VulkanBlitter::new(256, 256) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping gpu_dmabuf_export_reimport_roundtrip: {e}");
            return;
        }
    };

    // Log which modifier was selected for output images.
    let out_modifier = blitter.output_images[0].modifier;
    eprintln!(
        "output image modifier: 0x{:016x} ({})",
        out_modifier,
        if out_modifier == 0 {
            "LINEAR"
        } else {
            "tiled/vendor-specific"
        }
    );

    // B8G8R8A8_UNORM: magenta = [B=255, G=0, R=255, A=255].
    let magenta = [255u8, 0, 255, 255];
    let (src_fd, src_stride, src_offset) = blitter
        .create_test_source_dmabuf(128, 128, magenta)
        .expect("create test source");

    let result = blitter
        .blit(
            std::os::unix::io::AsFd::as_fd(&src_fd),
            128,
            128,
            DrmFourcc::Xrgb8888 as u32,
            0, // LINEAR source
            src_offset,
            src_stride,
        )
        .expect("blit");

    assert_eq!(result.width, 256);
    assert_eq!(result.height, 256);
    eprintln!(
        "exported buffer: modifier=0x{:016x} planes={}",
        result.modifier,
        result.planes.len()
    );

    // --- Step 1: Read back via internal Vulkan path (our gold reference) ---
    let idx = last_output_index(&blitter);
    let internal_pixels = blitter
        .readback_output_pixels(idx)
        .expect("internal readback");
    assert_eq!(internal_pixels.len(), 256 * 256 * 4);

    // 128×128 → 256×256 same aspect ratio → fills entirely, no bars.
    // Center pixel should be magenta.
    let center = (128 * 256 + 128) * 4;
    assert_eq!(
        internal_pixels[center], 255,
        "internal readback: center B=255"
    );
    assert_eq!(
        internal_pixels[center + 1],
        0,
        "internal readback: center G=0"
    );
    assert_eq!(
        internal_pixels[center + 2],
        255,
        "internal readback: center R=255"
    );

    // Corner pixel should also be magenta (no letterboxing).
    assert_eq!(internal_pixels[0], 255, "internal readback: corner B=255");
    assert_eq!(internal_pixels[1], 0, "internal readback: corner G=0");
    assert_eq!(internal_pixels[2], 255, "internal readback: corner R=255");

    // --- Step 2: Re-import the exported DMA-BUF and read back ---
    // This tests the exact same path that DRM scanout would use.
    let export_fd = &result.planes[0].fd;
    let (reimported_image, reimported_memory, reimported_view) = blitter
        .import_client_dmabuf(
            std::os::unix::io::AsFd::as_fd(export_fd),
            result.width,
            result.height,
            result.format,
            result.modifier,
            result.planes[0].offset,
            result.planes[0].stride,
        )
        .expect("reimport exported DMA-BUF");

    // Query the modifier the driver selected for the reimport.
    let mut reimport_drm_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
    unsafe {
        blitter
            .ext_drm_modifier
            .get_image_drm_format_modifier_properties(reimported_image, &mut reimport_drm_props)
            .expect("query reimported modifier");
    }
    eprintln!(
        "reimported modifier: 0x{:016x} (original export: 0x{:016x})",
        reimport_drm_props.drm_format_modifier, result.modifier
    );
    // Some drivers may report a slightly different modifier on reimport
    // (e.g. 0x...6015 → 0x...6014) which is expected behavior — the
    // driver auto-detects the actual tiling from the DMA-BUF. The pixel
    // data correctness check below is what actually matters.
    if reimport_drm_props.drm_format_modifier != result.modifier {
        eprintln!(
            "WARNING: modifier mismatch on reimport (0x{:016x} != 0x{:016x}). \
             This is common with some drivers and usually benign.",
            reimport_drm_props.drm_format_modifier, result.modifier
        );
    }

    // Read reimported image via the shared readback helper.
    let reimport_pixels = readback_image_direct(&blitter, reimported_image, 256, 256);

    // Clean up reimport resources.
    unsafe {
        blitter.device.destroy_image_view(reimported_view, None);
        blitter.device.destroy_image(reimported_image, None);
        blitter.device.free_memory(reimported_memory, None);
    }

    // --- Step 3: Verify reimported pixels match internal readback ---
    // Check center pixel from reimported data.
    let reimport_center = (128 * 256 + 128) * 4;
    eprintln!(
        "reimport center pixel: B={} G={} R={} A={}",
        reimport_pixels[reimport_center],
        reimport_pixels[reimport_center + 1],
        reimport_pixels[reimport_center + 2],
        reimport_pixels[reimport_center + 3],
    );

    assert_eq!(
        reimport_pixels[reimport_center], 255,
        "reimport: center B=255"
    );
    assert_eq!(
        reimport_pixels[reimport_center + 1],
        0,
        "reimport: center G=0"
    );
    assert_eq!(
        reimport_pixels[reimport_center + 2],
        255,
        "reimport: center R=255"
    );

    // Check corner pixel.
    assert_eq!(reimport_pixels[0], 255, "reimport: corner B=255");
    assert_eq!(reimport_pixels[1], 0, "reimport: corner G=0");
    assert_eq!(reimport_pixels[2], 255, "reimport: corner R=255");

    // Verify all pixels match between internal readback and reimport.
    // If there's a tiling/modifier mismatch, this will show exactly
    // how many pixels differ.
    let mut mismatches = 0u64;
    for i in 0..internal_pixels.len() {
        if internal_pixels[i] != reimport_pixels[i] {
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        // Sample some mismatch locations for debugging.
        let mut samples = 0;
        for pixel_idx in 0..(256 * 256) {
            let off = pixel_idx * 4;
            let matches = internal_pixels[off] == reimport_pixels[off]
                && internal_pixels[off + 1] == reimport_pixels[off + 1]
                && internal_pixels[off + 2] == reimport_pixels[off + 2];
            if matches {
                continue;
            }
            let row = pixel_idx / 256;
            let col = pixel_idx % 256;
            eprintln!(
                "MISMATCH at ({},{}): internal=[{},{},{},{}] reimport=[{},{},{},{}]",
                col,
                row,
                internal_pixels[off],
                internal_pixels[off + 1],
                internal_pixels[off + 2],
                internal_pixels[off + 3],
                reimport_pixels[off],
                reimport_pixels[off + 1],
                reimport_pixels[off + 2],
                reimport_pixels[off + 3],
            );
            samples += 1;
            if samples >= 10 {
                break;
            }
        }
    }
    assert_eq!(
        mismatches,
        0,
        "DMA-BUF reimport has {mismatches} byte mismatches out of {} — \
         likely a tiling/modifier mismatch in export",
        internal_pixels.len()
    );
}

#[test]
fn gpu_blit_solid_color() {
    // 32×32 solid red → 64×64 output (same aspect ratio, fills entirely).
    let mut blitter = match VulkanBlitter::new_with_modifier(64, 64, Some(0)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping gpu_blit_solid_color: {e}");
            return;
        }
    };

    // B8G8R8A8_UNORM: solid red = [B=0, G=0, R=255, A=255].
    let red = [0u8, 0, 255, 255];
    let (src_fd, src_stride, src_offset) = blitter
        .create_test_source_dmabuf(32, 32, red)
        .expect("create test source");

    let result = blitter
        .blit(
            std::os::unix::io::AsFd::as_fd(&src_fd),
            32,
            32,
            DrmFourcc::Xrgb8888 as u32,
            0, // LINEAR
            src_offset,
            src_stride,
        )
        .expect("blit");

    assert_eq!(result.width, 64);
    assert_eq!(result.height, 64);

    // compute_contain_rect(32,32,64,64) = (0,0,64,64): entire output filled.
    let idx = last_output_index(&blitter);
    let pixels = blitter.readback_output_pixels(idx).expect("readback");
    assert_eq!(pixels.len(), 64 * 64 * 4);

    // Verify center pixel is red.
    let center = (32 * 64 + 32) * 4;
    assert_eq!(pixels[center], 0, "center B=0");
    assert_eq!(pixels[center + 1], 0, "center G=0");
    assert_eq!(pixels[center + 2], 255, "center R=255");

    // Verify corner pixel is also red (no letterbox for square→square).
    assert_eq!(pixels[0], 0, "corner B=0");
    assert_eq!(pixels[1], 0, "corner G=0");
    assert_eq!(pixels[2], 255, "corner R=255");
}

#[test]
fn gpu_blit_letterbox() {
    // 64×32 wide source → 64×64 square output.
    // contain_rect(64,32,64,64) = (0, 16, 64, 32):
    //   rows 0..16  = black (top letterbox)
    //   rows 16..48 = green (content)
    //   rows 48..64 = black (bottom letterbox)
    let mut blitter = match VulkanBlitter::new_with_modifier(64, 64, Some(0)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping gpu_blit_letterbox: {e}");
            return;
        }
    };

    // B8G8R8A8_UNORM: solid green = [B=0, G=255, R=0, A=255].
    let green = [0u8, 255, 0, 255];
    let (src_fd, src_stride, src_offset) = blitter
        .create_test_source_dmabuf(64, 32, green)
        .expect("create test source");

    blitter
        .blit(
            std::os::unix::io::AsFd::as_fd(&src_fd),
            64,
            32,
            DrmFourcc::Xrgb8888 as u32,
            0,
            src_offset,
            src_stride,
        )
        .expect("blit");

    let idx = last_output_index(&blitter);
    let pixels = blitter.readback_output_pixels(idx).expect("readback");

    let w = 64usize;

    // Top letterbox row 0, center column: should be black (from clear).
    let top = 32 * 4;
    assert_eq!(pixels[top], 0, "top bar B=0");
    assert_eq!(pixels[top + 1], 0, "top bar G=0");
    assert_eq!(pixels[top + 2], 0, "top bar R=0");

    // Content area row 32, center column: should be green.
    let mid = (32 * w + 32) * 4;
    assert_eq!(pixels[mid], 0, "content B=0");
    assert_eq!(pixels[mid + 1], 255, "content G=255");
    assert_eq!(pixels[mid + 2], 0, "content R=0");

    // Bottom letterbox row 63: should be black.
    let bot = (63 * w + 32) * 4;
    assert_eq!(pixels[bot], 0, "bottom bar B=0");
    assert_eq!(pixels[bot + 1], 0, "bottom bar G=0");
    assert_eq!(pixels[bot + 2], 0, "bottom bar R=0");
}

#[test]
fn gpu_blit_pillarbox() {
    // 32×64 tall source → 64×64 square output.
    // contain_rect(32,64,64,64) = (16, 0, 32, 64):
    //   cols 0..16  = black (left pillarbox)
    //   cols 16..48 = blue (content)
    //   cols 48..64 = black (right pillarbox)
    let mut blitter = match VulkanBlitter::new_with_modifier(64, 64, Some(0)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping gpu_blit_pillarbox: {e}");
            return;
        }
    };

    // B8G8R8A8_UNORM: solid blue = [B=255, G=0, R=0, A=255].
    let blue = [255u8, 0, 0, 255];
    let (src_fd, src_stride, src_offset) = blitter
        .create_test_source_dmabuf(32, 64, blue)
        .expect("create test source");

    blitter
        .blit(
            std::os::unix::io::AsFd::as_fd(&src_fd),
            32,
            64,
            DrmFourcc::Xrgb8888 as u32,
            0,
            src_offset,
            src_stride,
        )
        .expect("blit");

    let idx = last_output_index(&blitter);
    let pixels = blitter.readback_output_pixels(idx).expect("readback");

    let w = 64usize;

    // Left pillarbox: row 32, col 0 → black.
    let left = 32 * w * 4;
    assert_eq!(pixels[left], 0, "left bar B=0");
    assert_eq!(pixels[left + 1], 0, "left bar G=0");
    assert_eq!(pixels[left + 2], 0, "left bar R=0");

    // Content area: row 32, col 32 → blue.
    let mid = (32 * w + 32) * 4;
    assert_eq!(pixels[mid], 255, "content B=255");
    assert_eq!(pixels[mid + 1], 0, "content G=0");
    assert_eq!(pixels[mid + 2], 0, "content R=0");

    // Right pillarbox: row 32, col 63 → black.
    let right = (32 * w + 63) * 4;
    assert_eq!(pixels[right], 0, "right bar B=0");
    assert_eq!(pixels[right + 1], 0, "right bar G=0");
    assert_eq!(pixels[right + 2], 0, "right bar R=0");
}

/// Full-resolution diagnostic test with pixel dump.
///
/// Blits a 500×500 solid cyan source onto a 3840×2160 output
/// (the exact scenario of vkcube on a 4K display). Verifies every
/// pixel: pillarbox bars must be black, content region must be cyan.
///
/// If `GAMECOMP_DUMP_DIR` is set, writes PPM image files for visual
/// inspection:
///   - `output_internal.ppm` — Vulkan readback of the output image
///   - `output_reimport.ppm` — DMA-BUF export → reimport readback
///
/// Uses auto-detected modifier (vendor-specific tiling) to test
/// the exact code path used in production.
#[test]
fn gpu_blit_4k_with_dump() {
    let out_w: u32 = 3840;
    let out_h: u32 = 2160;
    let src_w: u32 = 500;
    let src_h: u32 = 500;

    let mut blitter = match VulkanBlitter::new(out_w, out_h) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping gpu_blit_4k_with_dump: {e}");
            return;
        }
    };

    let out_modifier = blitter.output_images[0].modifier;
    eprintln!(
        "4K test: output modifier 0x{:016x} ({})",
        out_modifier,
        if out_modifier == 0 {
            "LINEAR"
        } else {
            "tiled/vendor-specific"
        }
    );

    // Solid cyan: B8G8R8A8_UNORM → [B=255, G=255, R=0, A=255].
    let cyan = [255u8, 255, 0, 255];
    let (src_fd, src_stride, src_offset) = blitter
        .create_test_source_dmabuf(src_w, src_h, cyan)
        .expect("create 500×500 cyan source");
    eprintln!(
        "4K test: source stride={src_stride} offset={src_offset} (expected stride={})",
        src_w * 4
    );

    let result = blitter
        .blit(
            std::os::unix::io::AsFd::as_fd(&src_fd),
            src_w,
            src_h,
            DrmFourcc::Xrgb8888 as u32,
            0, // LINEAR source
            src_offset,
            src_stride,
        )
        .expect("blit 500×500 → 3840×2160");

    assert_eq!(result.width, out_w);
    assert_eq!(result.height, out_h);
    eprintln!(
        "4K test: exported modifier 0x{:016x}, stride {}, planes {}",
        result.modifier,
        result.planes[0].stride,
        result.planes.len()
    );

    // Read back output image via Vulkan.
    let idx = last_output_index(&blitter);
    let pixels = blitter
        .readback_output_pixels(idx)
        .expect("readback 3840×2160");
    assert_eq!(pixels.len(), (out_w as usize) * (out_h as usize) * 4);

    // Compute expected layout.
    // contain_rect(500, 500, 3840, 2160):
    //   scale = min(3840/500, 2160/500) = min(7.68, 4.32) = 4.32
    //   dst_w = round(500 * 4.32) = 2160
    //   dst_h = round(500 * 4.32) = 2160
    //   offset_x = (3840 - 2160) / 2 = 840
    //   offset_y = 0
    let (exp_x, exp_y, exp_w, exp_h) = compute_contain_rect(src_w, src_h, out_w, out_h);
    eprintln!(
        "4K test: expected content rect: x={} y={} w={} h={}",
        exp_x, exp_y, exp_w, exp_h
    );

    // Verify pixels in three regions.
    let w = out_w as usize;
    let mut black_ok = 0u64;
    let mut black_bad = 0u64;
    let mut cyan_ok = 0u64;
    let mut cyan_bad = 0u64;
    let mut first_bad_black: Option<(usize, usize, [u8; 4])> = None;
    let mut first_bad_cyan: Option<(usize, usize, [u8; 4])> = None;

    // Allow tolerance for bilinear interpolation at content edges
    // and block-linear compression quantization. At 4.32× upscale,
    // the LINEAR filter blends over several pixels at each boundary.
    // Additionally, tiled compression can introduce ±6 rounding per
    // channel even in inner pixels.
    const TOLERANCE: u8 = 10;
    let edge_margin = ((out_w.max(out_h) / src_w.min(src_h)) as usize) * 2 + 8;

    for row in 0..(out_h as usize) {
        for col in 0..(out_w as usize) {
            let off = (row * w + col) * 4;
            let px = [
                pixels[off],
                pixels[off + 1],
                pixels[off + 2],
                pixels[off + 3],
            ];

            let in_content = col >= exp_x as usize
                && col < (exp_x + exp_w) as usize
                && row >= exp_y as usize
                && row < (exp_y + exp_h) as usize;

            // Letterbox/pillarbox region — should be black.
            if !in_content {
                if px[0] <= TOLERANCE && px[1] <= TOLERANCE && px[2] <= TOLERANCE {
                    black_ok += 1;
                } else {
                    black_bad += 1;
                    first_bad_black.get_or_insert((col, row, px));
                }
                continue;
            }

            // Edge pixels (bilinear blending zone) — just count.
            let inner = col >= (exp_x as usize + edge_margin)
                && col < ((exp_x + exp_w) as usize).saturating_sub(edge_margin)
                && row >= (exp_y as usize + edge_margin)
                && row < ((exp_y + exp_h) as usize).saturating_sub(edge_margin);
            if !inner {
                cyan_ok += 1;
                continue;
            }

            // Inner pixels must be exact cyan [255,255,0,X].
            if px[0] >= 255 - TOLERANCE && px[1] >= 255 - TOLERANCE && px[2] <= TOLERANCE {
                cyan_ok += 1;
            } else {
                cyan_bad += 1;
                first_bad_cyan.get_or_insert((col, row, px));
            }
        }
    }

    eprintln!(
        "4K test results: black_ok={black_ok} black_bad={black_bad} \
         cyan_ok={cyan_ok} cyan_bad={cyan_bad}"
    );
    if let Some((x, y, px)) = first_bad_black {
        eprintln!(
            "  first bad black at ({x},{y}): [{},{},{},{}]",
            px[0], px[1], px[2], px[3]
        );
    }
    if let Some((x, y, px)) = first_bad_cyan {
        eprintln!(
            "  first bad cyan at ({x},{y}): [{},{},{},{}]",
            px[0], px[1], px[2], px[3]
        );
    }

    // Dump PPM files if requested.
    if let Ok(dir) = std::env::var("GAMECOMP_DUMP_DIR") {
        let path = format!("{dir}/output_internal.ppm");
        write_ppm(&path, out_w as usize, out_h as usize, &pixels);
        eprintln!("4K test: wrote {path}");

        // Also do a reimport readback and dump that.
        let export_fd = &result.planes[0].fd;
        let (ri_img, ri_mem, ri_view) = blitter
            .import_client_dmabuf(
                std::os::unix::io::AsFd::as_fd(export_fd),
                result.width,
                result.height,
                result.format,
                result.modifier,
                result.planes[0].offset,
                result.planes[0].stride,
            )
            .expect("reimport for dump");

        // Read back reimported pixels via same infrastructure.
        // We need to temporarily swap the imported image into an
        // output slot. Instead, do a direct GPU copy.
        let ri_pixels = readback_image_direct(&blitter, ri_img, result.width, result.height);

        // SAFETY: reimported image/memory no longer needed after readback.
        unsafe {
            blitter.device.destroy_image_view(ri_view, None);
            blitter.device.destroy_image(ri_img, None);
            blitter.device.free_memory(ri_mem, None);
        }

        let path2 = format!("{dir}/output_reimport.ppm");
        write_ppm(&path2, out_w as usize, out_h as usize, &ri_pixels);
        eprintln!("4K test: wrote {path2}");

        // Count reimport mismatches.
        let mut ri_mismatches = 0u64;
        for i in 0..pixels.len() {
            if pixels[i] != ri_pixels[i] {
                ri_mismatches += 1;
            }
        }
        eprintln!("4K test: internal vs reimport byte mismatches: {ri_mismatches}");
    }

    assert_eq!(
        black_bad, 0,
        "letterbox/pillarbox regions have {black_bad} non-black pixels"
    );
    assert_eq!(cyan_bad, 0, "content region has {cyan_bad} non-cyan pixels");
}

/// Write BGRA pixel data as a PPM (RGB) image file.
fn write_ppm(path: &str, width: usize, height: usize, bgra: &[u8]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("create PPM");
    write!(f, "P6\n{width} {height}\n255\n").expect("PPM header");
    for pixel in bgra.chunks_exact(4) {
        // BGRA → RGB
        f.write_all(&[pixel[2], pixel[1], pixel[0]])
            .expect("PPM pixel");
    }
}

/// Direct GPU readback of an arbitrary VkImage (not necessarily an
/// output image). Used for reimport diagnostics.
fn readback_image_direct(
    blitter: &VulkanBlitter,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let buf_size = (width as u64) * (height as u64) * 4;
    let buf_info = vk::BufferCreateInfo::default()
        .size(buf_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST);
    let staging_buf =
        unsafe { blitter.device.create_buffer(&buf_info, None) }.expect("staging buf");
    let buf_reqs = unsafe { blitter.device.get_buffer_memory_requirements(staging_buf) };
    let buf_mem_type = blitter
        .find_memory_type(
            buf_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("host-visible memory");
    let buf_alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(buf_reqs.size)
        .memory_type_index(buf_mem_type);
    let staging_mem =
        unsafe { blitter.device.allocate_memory(&buf_alloc, None) }.expect("staging alloc");
    unsafe {
        blitter
            .device
            .bind_buffer_memory(staging_buf, staging_mem, 0)
    }
    .expect("bind staging");

    unsafe {
        blitter
            .device
            .wait_for_fences(&[blitter.fence], true, u64::MAX)
            .unwrap();
        blitter.device.reset_fences(&[blitter.fence]).unwrap();

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        blitter
            .device
            .reset_command_buffer(blitter.command_buffer, vk::CommandBufferResetFlags::empty())
            .unwrap();
        blitter
            .device
            .begin_command_buffer(blitter.command_buffer, &begin)
            .unwrap();

        let acquire = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
            .dst_queue_family_index(blitter.queue_family_index)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        blitter.device.cmd_pipeline_barrier(
            blitter.command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[acquire],
        );

        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        blitter.device.cmd_copy_image_to_buffer(
            blitter.command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buf,
            &[region],
        );

        blitter
            .device
            .end_command_buffer(blitter.command_buffer)
            .unwrap();

        let submit = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&blitter.command_buffer));
        blitter
            .device
            .queue_submit(blitter.queue, &[submit], blitter.fence)
            .unwrap();
        blitter
            .device
            .wait_for_fences(&[blitter.fence], true, u64::MAX)
            .unwrap();
    }

    let mut pixels = vec![0u8; buf_size as usize];
    unsafe {
        let ptr = blitter
            .device
            .map_memory(staging_mem, 0, buf_size, vk::MemoryMapFlags::empty())
            .expect("map staging");
        std::ptr::copy_nonoverlapping(ptr as *const u8, pixels.as_mut_ptr(), buf_size as usize);
        blitter.device.unmap_memory(staging_mem);
    }

    unsafe {
        blitter.device.destroy_buffer(staging_buf, None);
        blitter.device.free_memory(staging_mem, None);
    }

    pixels
}

/// Multi-frame blit cycle test.
///
/// Simulates the real present loop: 9 frames of solid red blitted
/// through the triple-buffered output images. Verifies:
/// - `buffer_index` cycles through 0, 1, 2, 0, 1, 2, ...
/// - Each frame's output pixels are correct (red content, black bars)
/// - The same `buffer_index` reused on later frames still produces
///   correct output (catches stale-buffer issues)
#[test]
fn gpu_blit_multi_frame_cycle() {
    let mut blitter = match VulkanBlitter::new_with_modifier(64, 64, Some(0)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping gpu_blit_multi_frame_cycle: {e}");
            return;
        }
    };

    let output_count = blitter.output_count();
    assert!(output_count >= 2, "need at least 2 output images");

    // B8G8R8A8_UNORM: solid red = [B=0, G=0, R=255, A=255].
    let red = [0u8, 0, 255, 255];
    let (src_fd, src_stride, src_offset) = blitter
        .create_test_source_dmabuf(32, 32, red)
        .expect("create test source");

    // Blit 9 frames (3 full cycles through triple-buffer).
    let num_frames = output_count * 3;
    for frame in 0..num_frames {
        let result = blitter
            .blit(
                std::os::unix::io::AsFd::as_fd(&src_fd),
                32,
                32,
                DrmFourcc::Xrgb8888 as u32,
                0, // LINEAR
                src_offset,
                src_stride,
            )
            .unwrap_or_else(|_| panic!("blit frame {frame}"));

        let expected_index = frame % output_count;
        assert_eq!(
            result.buffer_index, expected_index,
            "frame {frame}: expected buffer_index={expected_index}, got {}",
            result.buffer_index
        );

        // Verify output pixels are correct.
        let idx = last_output_index(&blitter);
        let pixels = blitter
            .readback_output_pixels(idx)
            .unwrap_or_else(|_| panic!("readback frame {frame}"));

        // Center pixel should be red.
        let center = (32 * 64 + 32) * 4;
        assert_eq!(
            pixels[center + 2],
            255,
            "frame {frame} idx {idx}: center R should be 255, got {}",
            pixels[center + 2]
        );
    }
}
