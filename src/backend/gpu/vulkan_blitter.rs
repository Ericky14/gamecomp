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
use tracing::{debug, info};

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

/// Vulkan-based DMA-BUF blitter.
///
/// Imports client DMA-BUFs, blits them to output images with explicit
/// modifiers, and exports the result for the host compositor.
pub struct VulkanBlitter {
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    #[allow(dead_code)]
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
}

struct ImportedImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
}

impl VulkanBlitter {
    /// Create a new Vulkan blitter for the given output dimensions.
    pub fn new(width: u32, height: u32) -> anyhow::Result<Self> {
        // Load Vulkan.
        // SAFETY: Loads the Vulkan loader shared library. No preconditions.
        let entry = unsafe { ash::Entry::load() }.context("failed to load Vulkan")?;

        // Create instance.
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"gamecomp-blitter")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: instance_create_info is valid and fully initialized on the stack.
        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .context("failed to create Vulkan instance")?;

        // Select physical device (prefer discrete GPU).
        // SAFETY: Instance was created successfully above.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("failed to enumerate Vulkan devices")?;
        if physical_devices.is_empty() {
            bail!("no Vulkan-capable GPU found");
        }

        let physical_device = physical_devices
            .iter()
            .copied()
            .find(|&pd| {
                // SAFETY: pd is a valid physical device from enumerate_physical_devices.
                let props = unsafe { instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or(physical_devices[0]);

        // SAFETY: physical_device is a valid handle from enumerate_physical_devices.
        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        // SAFETY: device_name is a null-terminated C string within the VkPhysicalDeviceProperties struct.
        let device_name = unsafe { std::ffi::CStr::from_ptr(device_props.device_name.as_ptr()) };
        info!(device = ?device_name, "blitter: selected Vulkan device");

        // Find a graphics queue (need blit/transfer support).
        // SAFETY: physical_device is a valid handle.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_families
            .iter()
            .position(|props| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .context("no graphics queue family found")? as u32;

        // Create device with required extensions.
        let device_extensions = [
            ash::khr::external_memory_fd::NAME.as_ptr(),
            ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ash::ext::image_drm_format_modifier::NAME.as_ptr(),
            ash::khr::image_format_list::NAME.as_ptr(),
            // VK_KHR_external_memory is a device extension promoted to 1.1
            // but we need it for the fd extension chain.
        ];

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extensions);

        // SAFETY: physical_device is valid; device_create_info references valid extensions
        // and queue family index verified above.
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .context("failed to create Vulkan device")?;

        // SAFETY: Device created successfully; queue_family_index validated; queue index 0 requested.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Extension function pointers.
        let ext_memory_fd = ash::khr::external_memory_fd::Device::new(&instance, &device);
        let ext_drm_modifier = ash::ext::image_drm_format_modifier::Device::new(&instance, &device);

        // Command pool + buffer.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: Device is valid; pool_info references a valid queue family index.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .context("failed to create command pool")?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: Device and command_pool are valid; alloc_info requests 1 primary buffer.
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .context("failed to allocate command buffer")?;
        let command_buffer = command_buffers[0];

        // Fence.
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: Device is valid; fence_info is fully initialized on the stack.
        let fence =
            unsafe { device.create_fence(&fence_info, None) }.context("failed to create fence")?;

        let mut blitter = Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            command_buffer,
            fence,
            output_images: Vec::new(),
            output_index: 0,
            import_cache: HashMap::new(),
            ext_memory_fd,
            ext_drm_modifier,
            import_modifiers: Vec::new(),
        };

        // Query all valid modifiers for import (before allocating outputs).
        blitter.import_modifiers = blitter.query_all_valid_modifiers()?;

        // Allocate triple-buffered output images.
        blitter.allocate_output_images(width, height, 3)?;

        info!(
            width,
            height,
            num_outputs = blitter.output_images.len(),
            modifier = format!("0x{:016x}", blitter.output_images[0].modifier),
            "blitter: initialized"
        );

        Ok(blitter)
    }

    /// Query all valid DRM modifiers for B8G8R8A8_UNORM that support TRANSFER_SRC.
    ///
    /// These are used when importing client DMA-BUFs — we provide the full list
    /// to `ImageDrmFormatModifierListCreateInfoEXT` and let the driver determine
    /// which modifier matches the imported buffer's actual tiling layout.
    fn query_all_valid_modifiers(&self) -> anyhow::Result<Vec<u64>> {
        let vk_format = vk::Format::B8G8R8A8_UNORM;

        let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut format_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);
        // SAFETY: physical_device is valid; output struct is on the stack.
        // First call queries count only (no storage provided).
        unsafe {
            self.instance.get_physical_device_format_properties2(
                self.physical_device,
                vk_format,
                &mut format_props2,
            );
        }

        let count = modifier_list.drm_format_modifier_count as usize;
        let mut modifier_props = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
        let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut modifier_props);
        let mut format_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);
        // SAFETY: physical_device is valid; modifier_props has `count` elements
        // matching the count returned by the first query.
        unsafe {
            self.instance.get_physical_device_format_properties2(
                self.physical_device,
                vk_format,
                &mut format_props2,
            );
        }

        const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
        let required = vk::FormatFeatureFlags::TRANSFER_SRC;

        let modifiers: Vec<u64> = modifier_props
            .iter()
            .filter(|mp| {
                mp.drm_format_modifier != DRM_FORMAT_MOD_INVALID
                    && mp.drm_format_modifier_plane_count == 1
                    && mp.drm_format_modifier_tiling_features.contains(required)
            })
            .map(|mp| mp.drm_format_modifier)
            .collect();

        info!(
            count = modifiers.len(),
            "blitter: import modifiers (TRANSFER_SRC, single-plane, non-INVALID)"
        );
        for m in &modifiers {
            debug!(
                modifier = format!("0x{:016x}", m),
                "blitter: import modifier"
            );
        }

        if modifiers.is_empty() {
            bail!("no valid import modifiers found");
        }

        Ok(modifiers)
    }

    /// Query the best explicit modifier for XRGB8888 output images.
    fn query_output_modifier(&self) -> anyhow::Result<u64> {
        // Query format modifiers for VK_FORMAT_B8G8R8A8_UNORM (= DRM XRGB8888 / ARGB8888).
        let vk_format = vk::Format::B8G8R8A8_UNORM;

        let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut format_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);

        // SAFETY: physical_device is valid; first call queries count only.
        unsafe {
            self.instance.get_physical_device_format_properties2(
                self.physical_device,
                vk_format,
                &mut format_props2,
            );
        }

        let count = modifier_list.drm_format_modifier_count as usize;
        if count == 0 {
            bail!("no DRM format modifiers available for B8G8R8A8_UNORM");
        }

        // Now query with storage.
        let mut modifier_props = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
        let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut modifier_props);
        let mut format_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);

        // SAFETY: physical_device is valid; modifier_props has `count` elements.
        unsafe {
            self.instance.get_physical_device_format_properties2(
                self.physical_device,
                vk_format,
                &mut format_props2,
            );
        }

        info!(count, "blitter: available DRM modifiers for B8G8R8A8_UNORM");

        // Pick the best modifier that supports:
        // - TRANSFER_DST (we blit into it)
        // - Single plane (simpler)
        // - NOT DRM_FORMAT_MOD_INVALID (need a concrete modifier for export)
        // - NOT LINEAR (poor performance)
        // Prefer vendor-specific tiled modifiers.
        const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
        const DRM_FORMAT_MOD_LINEAR: u64 = 0;

        let required_features = vk::FormatFeatureFlags::TRANSFER_DST;

        let mut best: Option<(u64, u32)> = None; // (modifier, plane_count)
        for mp in &modifier_props {
            let modifier = mp.drm_format_modifier;
            let planes = mp.drm_format_modifier_plane_count;
            let features = mp.drm_format_modifier_tiling_features;

            debug!(
                modifier = format!("0x{:016x}", modifier),
                planes,
                features = format!("0x{:x}", features.as_raw()),
                "blitter: modifier candidate"
            );

            if modifier == DRM_FORMAT_MOD_INVALID {
                continue;
            }
            if planes != 1 {
                continue; // Stick to single-plane for simplicity.
            }
            if !features.contains(required_features) {
                continue;
            }

            // Prefer non-LINEAR (tiled) over LINEAR.
            match best {
                None => best = Some((modifier, planes)),
                Some((existing, _)) => {
                    if existing == DRM_FORMAT_MOD_LINEAR && modifier != DRM_FORMAT_MOD_LINEAR {
                        best = Some((modifier, planes));
                    }
                }
            }
        }

        let (modifier, _planes) = best.context(
            "no suitable DRM modifier found (need TRANSFER_DST, single-plane, non-INVALID)",
        )?;

        info!(
            modifier = format!("0x{:016x}", modifier),
            "blitter: selected output modifier"
        );

        Ok(modifier)
    }

    /// Allocate output images with explicit DRM format modifiers.
    fn allocate_output_images(
        &mut self,
        width: u32,
        height: u32,
        count: usize,
    ) -> anyhow::Result<()> {
        let modifier = self.query_output_modifier()?;
        let drm_fourcc_xrgb8888: u32 = DrmFourcc::Xrgb8888 as u32;
        let vk_format = vk::Format::B8G8R8A8_UNORM;

        for i in 0..count {
            let modifiers = [modifier];

            // Create image with explicit DRM modifier.
            let mut drm_modifier_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                .drm_format_modifiers(&modifiers);

            let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

            let vk_formats = [vk_format];
            let mut format_list_info =
                vk::ImageFormatListCreateInfo::default().view_formats(&vk_formats);

            let image_info = vk::ImageCreateInfo::default()
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
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut drm_modifier_info)
                .push_next(&mut external_info)
                .push_next(&mut format_list_info);

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

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut export_info);

            // SAFETY: Device is valid; alloc_info references valid memory type
            // and export info for DMA-BUF handle type.
            let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
                .with_context(|| format!("failed to allocate output image memory {i}"))?;

            // SAFETY: Device, image, and memory are valid; offset 0 is correct
            // for a dedicated allocation.
            unsafe {
                self.device
                    .bind_image_memory(image, memory, 0)
                    .with_context(|| format!("failed to bind output image memory {i}"))?;
            }

            // Query actual modifier layout (offset, stride).
            let mut drm_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
            // SAFETY: Device, image, and ext_drm_modifier function pointers are
            // valid. Image was created with DRM_FORMAT_MODIFIER_EXT tiling.
            unsafe {
                self.ext_drm_modifier
                    .get_image_drm_format_modifier_properties(image, &mut drm_props)
                    .context("failed to query image DRM modifier properties")?;
            }

            debug!(
                i,
                modifier = format!("0x{:016x}", drm_props.drm_format_modifier),
                "blitter: output image modifier confirmed"
            );

            // Query subresource layout for plane 0.
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
                modifier = format!("0x{:016x}", modifier),
                "blitter: output image allocated"
            );

            self.output_images.push(OutputImage {
                image,
                memory,
                modifier,
                planes: vec![OutputPlaneInfo {
                    fd,
                    offset: layout.offset as u32,
                    stride: layout.row_pitch as u32,
                }],
                width,
                height,
                format: drm_fourcc_xrgb8888,
            });
        }

        Ok(())
    }

    /// Import a client DMA-BUF as a VkImage for sampling/transfer source.
    #[allow(clippy::too_many_arguments)]
    fn import_client_dmabuf(
        &mut self,
        fd: std::os::unix::io::BorrowedFd<'_>,
        width: u32,
        height: u32,
        format: u32,
        _modifier: u64,
        _offset: u32,
        _stride: u32,
    ) -> anyhow::Result<(vk::Image, vk::DeviceMemory)> {
        // Use the List form — provide all known modifiers and let the
        // driver inspect the DMA-BUF fd to determine the actual tiling layout.
        // The driver knows its own allocations and will pick the right modifier.
        let vk_format = drm_format_to_vk(format)?;

        let mut drm_modifier_list = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
            .drm_format_modifiers(&self.import_modifiers);

        let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_info = vk::ImageCreateInfo::default()
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
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut drm_modifier_list)
            .push_next(&mut external_info);

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

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index)
            .push_next(&mut import_fd_info);

        // SAFETY: Device is valid; alloc_info references valid memory type
        // and import_fd_info provides a valid DMA-BUF fd. Vulkan takes fd ownership.
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }
            .context("failed to allocate imported dmabuf memory")?;

        // SAFETY: Device, image, and memory are valid; offset 0 for the
        // imported DMA-BUF memory.
        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .context("failed to bind imported dmabuf memory")?;
        }

        Ok((image, memory))
    }

    /// Current output image index (advances after each blit).
    #[inline(always)]
    pub fn output_index(&self) -> usize {
        self.output_index
    }

    /// Number of output images (triple-buffer count).
    #[inline(always)]
    pub fn output_count(&self) -> usize {
        self.output_images.len()
    }

    /// Blit a client DMA-BUF to the next output image and return the exported buffer.
    ///
    /// This is the main entry point called per frame from the wayland backend.
    #[allow(clippy::too_many_arguments)]
    pub fn blit(
        &mut self,
        client_fd: std::os::unix::io::BorrowedFd<'_>,
        width: u32,
        height: u32,
        format: u32,
        modifier: u64,
        offset: u32,
        stride: u32,
    ) -> anyhow::Result<ExportedBuffer> {
        // Wait for previous blit to complete.
        // SAFETY: Device and fence are valid. Fence was created with SIGNALED flag
        // and is reset after each successful wait. u64::MAX timeout = wait forever.
        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .context("blit: wait fence")?;
            self.device
                .reset_fences(&[self.fence])
                .context("blit: reset fence")?;
        }

        // Import client DMA-BUF.
        let (src_image, src_memory) =
            self.import_client_dmabuf(client_fd, width, height, format, modifier, offset, stride)?;

        // Get next output image.
        let out_idx = self.output_index;
        self.output_index = (self.output_index + 1) % self.output_images.len();
        let out = &self.output_images[out_idx];

        // Compute contain-style scaling: fit content inside output,
        // preserving aspect ratio with letterboxing/pillarboxing.
        let (dst_x, dst_y, dst_w, dst_h) =
            compute_contain_rect(width, height, out.width, out.height);

        // Record blit command.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: All Vulkan handles (device, command_buffer, images) are valid.
        // src_image was just imported; out.image is a pre-allocated output image.
        // All barriers, blit regions, and submit info are fully initialized on
        // the stack. The fence is unsignaled (reset above) and will be signaled
        // on completion. We wait synchronously before returning.
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .context("blit: reset cmd buf")?;
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .context("blit: begin cmd buf")?;

            // Transition source image: UNDEFINED → TRANSFER_SRC_OPTIMAL
            let src_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(src_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            // Transition dest image: UNDEFINED → TRANSFER_DST_OPTIMAL
            let dst_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(out.image)
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
                &[src_barrier, dst_barrier],
            );

            // Clear the entire output image to black first.
            // This provides the letterbox/pillarbox bars for contain-style scaling.
            let clear_color = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            };
            let clear_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            self.device.cmd_clear_color_image(
                self.command_buffer,
                out.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[clear_range],
            );

            // Blit (handles format conversion if needed).
            let src_subresource = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let dst_subresource = src_subresource;

            // Use contain-style destination region: centered content with
            // aspect ratio preserved. The clear above provides black bars.
            let region = vk::ImageBlit {
                src_subresource,
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: width as i32,
                        y: height as i32,
                        z: 1,
                    },
                ],
                dst_subresource,
                dst_offsets: [
                    vk::Offset3D {
                        x: dst_x,
                        y: dst_y,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: dst_x + dst_w,
                        y: dst_y + dst_h,
                        z: 1,
                    },
                ],
            };

            // Use LINEAR filter for upscale quality when blitting
            // from client resolution to physical pixel resolution.
            self.device.cmd_blit_image(
                self.command_buffer,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                out.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::LINEAR,
            );

            // Transition dest to GENERAL so the host can sample it.
            let final_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(out.image)
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
                &[final_barrier],
            );

            self.device
                .end_command_buffer(self.command_buffer)
                .context("blit: end cmd buf")?;
        }

        // Submit.
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));

        // SAFETY: Device is valid. Submit info references a valid command buffer
        // and fence. The fence will be signaled when the GPU completes the work.
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], self.fence)
                .context("blit: submit")?;

            // Wait synchronously — the wayland event loop is already threaded,
            // and we need the blit done before forwarding the DMA-BUF.
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .context("blit: wait for blit completion")?;
        }

        // Clean up the imported source image (transient — recreated per frame).
        // TODO(perf): cache imported images by (dev, ino) like we cache wl_buffers
        // SAFETY: src_image and src_memory were created in import_client_dmabuf
        // above and are no longer in use (GPU work completed via fence wait).
        unsafe {
            self.device.destroy_image(src_image, None);
            self.device.free_memory(src_memory, None);
        }

        // Export: dup the output image's DMA-BUF fd (we keep the original).
        let out = &self.output_images[out_idx];
        let mut exported_planes = Vec::with_capacity(out.planes.len());
        for plane in &out.planes {
            // SAFETY: plane.fd is a valid raw fd from Vulkan export
            let borrowed = unsafe { std::os::unix::io::BorrowedFd::borrow_raw(plane.fd) };
            let dup_fd = rustix::io::dup(borrowed).context("failed to dup output plane fd")?;
            exported_planes.push(ExportedPlane {
                fd: dup_fd,
                offset: plane.offset,
                stride: plane.stride,
            });
        }

        Ok(ExportedBuffer {
            planes: exported_planes,
            width: out.width,
            height: out.height,
            format: out.format,
            modifier: out.modifier,
        })
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

        // Destroy imported images.
        for (_, img) in self.import_cache.drain() {
            // SAFETY: Image and memory are valid; GPU is idle (waited above).
            unsafe {
                self.device.destroy_image(img.image, None);
                self.device.free_memory(img.memory, None);
            }
        }

        // Destroy output images and close their fds.
        for out in &self.output_images {
            // SAFETY: Images and memory are valid; GPU is idle. Raw fds were
            // obtained from Vulkan's get_memory_fd and are owned by us.
            unsafe {
                // Close the raw fds we got from Vulkan export.
                for plane in &out.planes {
                    libc::close(plane.fd);
                }
                self.device.destroy_image(out.image, None);
                self.device.free_memory(out.memory, None);
            }
        }

        // SAFETY: All dependent resources (images, memory, command buffers)
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
