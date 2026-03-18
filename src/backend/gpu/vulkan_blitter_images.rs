//! Output and client image allocation, import, and management.
//!
//! Handles three image paths:
//! - Self-allocated outputs (Vulkan-exported DMA-BUFs for Wayland backend)
//! - GBM-imported outputs (DRM backend — GBM owns memory, Vulkan imports)
//! - Client DMA-BUF imports (per-frame input from Wayland clients)

use super::*;

/// Create a TYPE_2D image view with the standard color subresource range.
///
/// Deduplicates the identical image view creation pattern used for output
/// images (STORAGE) and client images (SAMPLED).
fn create_image_view_2d(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
) -> anyhow::Result<vk::ImageView> {
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(COLOR_SUBRESOURCE_RANGE);
    // SAFETY: Device and image are valid; view info matches image format.
    unsafe { device.create_image_view(&view_info, None) }.context("failed to create image view")
}

impl VulkanBlitter {
    /// Import externally-allocated DMA-BUFs as output images.
    ///
    /// Used with GBM-allocated buffers: GBM owns the memory allocation,
    /// Vulkan imports for rendering via `vkCmdBlitImage` / `vkCmdClearColorImage`,
    /// and DRM presents from native GEM handles (no PRIME).
    ///
    /// Each DMA-BUF fd is dup'd before import since Vulkan takes ownership
    /// of the imported fd.
    pub fn import_output_images(
        &mut self,
        dmabufs: &[crate::backend::DmaBuf],
    ) -> anyhow::Result<()> {
        let drm_fourcc_xrgb8888: u32 = DrmFourcc::Xrgb8888 as u32;

        for (i, dmabuf) in dmabufs.iter().enumerate() {
            let plane = &dmabuf.planes[0];
            let modifier: u64 = dmabuf.modifier.into();

            info!(
                i,
                width = dmabuf.width,
                height = dmabuf.height,
                modifier = format!("0x{:016x}", modifier),
                stride = plane.stride,
                offset = plane.offset,
                fd = plane.fd,
                "blitter: importing GBM output image"
            );

            // Dup the fd — Vulkan's VkImportMemoryFdInfoKHR takes ownership.
            // SAFETY: plane.fd is a valid raw fd from GBM export.
            let borrowed_fd = unsafe { std::os::unix::io::BorrowedFd::borrow_raw(plane.fd) };
            let import_fd = rustix::io::dup(borrowed_fd).context("failed to dup GBM DMA-BUF fd")?;

            // Use the Explicit form since we know the exact modifier from GBM.
            let plane_layout = vk::SubresourceLayout {
                offset: plane.offset as u64,
                size: 0, // filled by driver
                row_pitch: plane.stride as u64,
                array_pitch: 0,
                depth_pitch: 0,
            };
            let mut drm_modifier_explicit =
                vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                    .drm_format_modifier(modifier)
                    .plane_layouts(std::slice::from_ref(&plane_layout));

            let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

            // MUTABLE_FORMAT + format list for display engine compat.
            let format_list = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
            let mut format_list_info =
                vk::ImageFormatListCreateInfo::default().view_formats(&format_list);

            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(VK_FORMAT_XRGB)
                .extent(vk::Extent3D {
                    width: dmabuf.width,
                    height: dmabuf.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(OUTPUT_IMAGE_USAGE)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                .push_next(&mut format_list_info)
                .push_next(&mut drm_modifier_explicit)
                .push_next(&mut external_info);

            // SAFETY: Device is valid; image_info is fully initialized with
            // explicit modifier, external memory, and format list chains.
            let image = unsafe { self.device.create_image(&image_info, None) }
                .with_context(|| format!("failed to create imported output image {i}"))?;

            // SAFETY: Device and image are valid; returns memory requirements.
            let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };

            let mem_type_index = self
                .find_memory_type(
                    mem_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .context("no suitable memory type for imported output image")?;

            // Import the DMA-BUF fd. Vulkan takes ownership of the fd.
            use std::os::unix::io::IntoRawFd;
            let mut import_fd_info = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(import_fd.into_raw_fd());

            // Dedicated allocation — required by some drivers for external memory.
            let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut import_fd_info)
                .push_next(&mut dedicated_info);

            // SAFETY: Device is valid; alloc_info chains import fd and dedicated
            // allocation for the target image. The fd is valid and owned.
            let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
                .with_context(|| format!("failed to import output image memory {i}"))?;

            // SAFETY: Device, image, and memory are valid; offset 0 for dedicated.
            unsafe {
                self.device
                    .bind_image_memory(image, memory, 0)
                    .with_context(|| format!("failed to bind imported output image memory {i}"))?;
            }

            // Verify the modifier matches what we requested.
            let mut drm_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
            // SAFETY: Device, image, and ext_drm_modifier function pointers are
            // valid. Image was created with DRM_FORMAT_MODIFIER_EXT tiling.
            unsafe {
                self.ext_drm_modifier
                    .get_image_drm_format_modifier_properties(image, &mut drm_props)
                    .context("failed to query imported output image modifier")?;
            }

            info!(
                i,
                requested = format!("0x{:016x}", modifier),
                confirmed = format!("0x{:016x}", drm_props.drm_format_modifier),
                "blitter: imported output image modifier confirmed"
            );

            self.output_images.push(OutputImage {
                image,
                memory,
                modifier,
                planes: vec![OutputPlaneInfo {
                    fd: plane.fd, // Keep the original fd (owned by GbmOutputBuffer)
                    offset: plane.offset,
                    stride: plane.stride,
                }],
                width: dmabuf.width,
                height: dmabuf.height,
                format: drm_fourcc_xrgb8888,
                blitted: false,
                owns_fds: false,
            });

            let image_view = create_image_view_2d(&self.device, image, VK_FORMAT_XRGB)
                .with_context(|| format!("failed to create output image view {i}"))?;
            self.output_image_views.push(image_view);
        }

        info!(
            count = self.output_images.len(),
            "blitter: GBM output images imported"
        );
        Ok(())
    }

    /// Allocate output images with explicit DRM format modifiers.
    ///
    /// `scanout_modifiers` is the list of modifiers the DRM primary plane
    /// supports for XRGB8888. We intersect this with Vulkan's exportable
    /// modifiers and pass the intersection to
    /// `VkImageDrmFormatModifierListCreateInfoEXT`, letting the Vulkan driver
    /// pick the optimal tiling.
    pub(super) fn allocate_output_images(
        &mut self,
        width: u32,
        height: u32,
        count: usize,
        scanout_modifiers: &[u64],
    ) -> anyhow::Result<()> {
        // Compute the intersection of scanout modifiers with Vulkan's
        // exportable modifiers. If the caller provided scanout modifiers,
        // we use only those that Vulkan also reports as exportable with
        // TRANSFER_DST + TRANSFER_SRC. If empty, we auto-detect.
        let output_modifiers = self.compute_output_modifiers(scanout_modifiers, width, height)?;

        info!(
            count = output_modifiers.len(),
            "blitter: output modifiers for image allocation"
        );
        for m in &output_modifiers {
            info!(
                modifier = format!("0x{:016x}", m),
                "blitter: output modifier"
            );
        }

        let drm_fourcc_xrgb8888: u32 = DrmFourcc::Xrgb8888 as u32;

        for i in 0..count {
            let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

            // Always use DRM_FORMAT_MODIFIER_EXT tiling with the modifier list.
            // The Vulkan driver selects the best modifier from the list.
            let mut drm_modifier_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&output_modifiers);

            // Chain VkImageFormatListCreateInfo with both the linear and
            // sRGB format variants, and set MUTABLE_FORMAT_BIT. This constrains
            // the driver's modifier selection and ensures compatibility with
            // the display engine on various drivers.
            let format_list = [vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB];
            let mut format_list_info =
                vk::ImageFormatListCreateInfo::default().view_formats(&format_list);

            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(VK_FORMAT_XRGB)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(OUTPUT_IMAGE_USAGE)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                .push_next(&mut format_list_info)
                .push_next(&mut drm_modifier_info)
                .push_next(&mut external_info);

            // SAFETY: Device is valid; image_info is fully initialized with valid
            // format, extent, and chained DRM modifier / external memory structs.
            let image = unsafe { self.device.create_image(&image_info, None) }
                .with_context(|| format!("failed to create output image {i}"))?;

            // SAFETY: Device and image are valid; returns memory requirements.
            let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };

            // Allocate with DMA-BUF export.
            let mut export_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

            let mem_type_index = self
                .find_memory_type(
                    mem_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .context("no suitable memory type for output image")?;

            // Dedicated allocation for exported DMA-BUF images — required
            // by some drivers for correct external memory export/import.
            let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_info)
                .push_next(&mut dedicated_info);

            // SAFETY: Device is valid; alloc_info references valid memory type,
            // export info for DMA-BUF handle type, and dedicated allocation
            // binding the memory to the target image.
            let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
                .with_context(|| format!("failed to allocate output image memory {i}"))?;

            // SAFETY: Device, image, and memory are valid; offset 0 is correct
            // for a dedicated allocation.
            unsafe {
                self.device
                    .bind_image_memory(image, memory, 0)
                    .with_context(|| format!("failed to bind output image memory {i}"))?;
            }

            // Query the modifier the Vulkan driver actually selected.
            let mut drm_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
            // SAFETY: Device, image, and ext_drm_modifier function pointers are
            // valid. Image was created with DRM_FORMAT_MODIFIER_EXT tiling.
            unsafe {
                self.ext_drm_modifier
                    .get_image_drm_format_modifier_properties(image, &mut drm_props)
                    .context("failed to query image DRM modifier properties")?;
            }
            let confirmed_modifier = drm_props.drm_format_modifier;
            debug!(
                i,
                modifier = format!("0x{:016x}", confirmed_modifier),
                "blitter: output image modifier confirmed"
            );

            // Query subresource layout using MEMORY_PLANE_0_EXT (required for
            // DRM_FORMAT_MODIFIER_EXT tiling).
            let subresource = vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                mip_level: 0,
                array_layer: 0,
            };
            // SAFETY: Device and image are valid; subresource specifies plane 0.
            let layout = unsafe { self.device.get_image_subresource_layout(image, subresource) };

            // Export DMA-BUF fd.
            let fd_info = vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            // SAFETY: Device and memory are valid; DMA_BUF_EXT handle type
            // was specified at allocation time via ExportMemoryAllocateInfo.
            let fd = unsafe { self.ext_memory_fd.get_memory_fd(&fd_info) }
                .with_context(|| format!("failed to export output image fd {i}"))?;

            info!(
                i,
                fd,
                offset = layout.offset,
                stride = layout.row_pitch,
                modifier = format!("0x{:016x}", confirmed_modifier),
                "blitter: output image allocated"
            );

            self.output_images.push(OutputImage {
                image,
                memory,
                modifier: confirmed_modifier,
                planes: vec![OutputPlaneInfo {
                    fd,
                    offset: layout.offset as u32,
                    stride: layout.row_pitch as u32,
                }],
                width,
                height,
                format: drm_fourcc_xrgb8888,
                blitted: false,
                owns_fds: true,
            });

            let image_view = create_image_view_2d(&self.device, image, VK_FORMAT_XRGB)
                .with_context(|| format!("failed to create output image view {i}"))?;
            self.output_image_views.push(image_view);
        }

        Ok(())
    }

    /// Import a client DMA-BUF as a VkImage with ImageView for compute
    /// shader sampling.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn import_client_dmabuf(
        &mut self,
        fd: std::os::unix::io::BorrowedFd<'_>,
        width: u32,
        height: u32,
        format: u32,
        modifier: u64,
        offset: u32,
        stride: u32,
    ) -> anyhow::Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let vk_format = drm_format_to_vk(format)?;

        // Import strategy:
        //
        // Known modifier (LINEAR, vendor-tiled, etc.): Use Explicit form
        //   with VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT — tells the driver
        //   the exact memory layout via SubresourceLayout.
        //
        // DRM_FORMAT_MOD_INVALID (legacy XWayland clients): Use
        //   VK_IMAGE_TILING_OPTIMAL — the driver internally knows the
        //   memory layout because it allocated the buffer (or received
        //   it via same-driver DMA-BUF sharing). Using the List form with
        //   DRM_FORMAT_MODIFIER_EXT is unreliable: the driver picks from
        //   our list but may select the wrong block-linear variant,
        //   producing garbled output (static noise, scrambled tiles).
        let plane_layout = vk::SubresourceLayout {
            offset: offset as u64,
            size: 0, // filled by driver
            row_pitch: stride as u64,
            array_pitch: 0,
            depth_pitch: 0,
        };
        let mut drm_modifier_explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(modifier)
            .plane_layouts(std::slice::from_ref(&plane_layout));

        let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let tiling = if modifier != DRM_FORMAT_MOD_INVALID {
            vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT
        } else {
            vk::ImageTiling::OPTIMAL
        };

        let mut image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(CLIENT_IMAGE_USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);

        if modifier != DRM_FORMAT_MOD_INVALID {
            image_info = image_info.push_next(&mut drm_modifier_explicit);
        }
        // DRM_FORMAT_MOD_INVALID: no modifier chain needed — OPTIMAL tiling
        // with DMA_BUF_EXT import lets the driver determine the layout.

        // SAFETY: Device is valid; image_info is fully initialized with valid
        // format, extent, and chained DRM modifier list / external memory structs.
        let image = unsafe { self.device.create_image(&image_info, None) }
            .context("failed to create imported image")?;

        // SAFETY: Device and image are valid.
        let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };

        // Import the DMA-BUF fd as memory.
        let dup_fd = rustix::io::dup(fd).context("failed to dup client dmabuf fd")?;

        let mut import_fd_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(std::os::unix::io::AsRawFd::as_raw_fd(&dup_fd));

        // Prevent Rust from closing the fd — Vulkan takes ownership.
        std::mem::forget(dup_fd);

        let mem_type_index = self
            .find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
            .context("no suitable memory type for imported dmabuf")?;

        // A dedicated allocation is required for importing external DMA-BUF
        // memory on some drivers. Without it the driver doesn't associate
        // the fd with the VkImage, leading to garbage pixel reads.
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index)
            .push_next(&mut import_fd_info)
            .push_next(&mut dedicated_info);

        // SAFETY: Device is valid; alloc_info references valid memory type,
        // import_fd_info provides a valid DMA-BUF fd (Vulkan takes ownership),
        // and dedicated_info binds the allocation to the target image.
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
            .context("failed to allocate imported dmabuf memory")?;

        // SAFETY: Device, image, and memory are valid; offset 0 for the
        // imported DMA-BUF memory.
        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .context("failed to bind imported dmabuf memory")?;
        }

        let image_view = create_image_view_2d(&self.device, image, vk_format)
            .context("failed to create client import image view")?;

        Ok((image, memory, image_view))
    }
}
