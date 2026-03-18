//! Vulkan device and compute pipeline initialization.
//!
//! Creates the Vulkan instance, selects a physical device, sets up the
//! graphics queue, command pool, fence, and builds the compute pipeline
//! (shader module, descriptor set layout, pipeline layout, descriptor pool).

use super::*;

impl VulkanBlitter {
    /// Internal constructor — sets up Vulkan device infrastructure only.
    ///
    /// Does NOT allocate output images. Callers use either:
    /// - `create_with_outputs()` to self-allocate (tests, Wayland backend)
    /// - `new_for_import()` + `import_output_images()` (DRM/GBM path)
    pub(super) fn create() -> anyhow::Result<Self> {
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
            ash::ext::queue_family_foreign::NAME.as_ptr(),
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

        // --- Compute pipeline for shader-based composition ---

        // Load blit.comp SPIR-V (compiled by build.rs at build time).
        // include_bytes returns &[u8] which may not be 4-byte aligned,
        // so we convert to u32 words via byte-level reads.
        let spv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/blit.spv"));
        assert!(
            spv_bytes.len().is_multiple_of(4),
            "SPIR-V not 4-byte aligned"
        );
        let spv_words: Vec<u32> = spv_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
        // SAFETY: Device is valid; shader code is valid SPIR-V from build.rs.
        let shader_module = unsafe { device.create_shader_module(&shader_info, None) }
            .context("failed to create blit shader module")?;

        // Linear sampler for bilinear-filtered client texture sampling.
        // CLAMP_TO_EDGE prevents bleeding at texture borders.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: Device is valid; sampler_info is fully initialized.
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }
            .context("failed to create blit sampler")?;

        // Descriptor set layout: binding 0 = combined image sampler (src),
        // binding 1 = storage image (dst).
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Device is valid; layout info references valid bindings.
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&ds_layout_info, None) }
                .context("failed to create descriptor set layout")?;

        // Pipeline layout: one descriptor set + push constants.
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<BlitPushConstants>() as u32);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        // SAFETY: Device is valid; layout info references valid descriptor set layout.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .context("failed to create pipeline layout")?;

        // Compute pipeline.
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);
        // SAFETY: Device is valid; pipeline info references valid shader module
        // and pipeline layout. Pipeline cache is null (no caching).
        let compute_pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[compute_pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| e)
        .context("failed to create blit compute pipeline")?[0];

        // Descriptor pool: 1 combined_image_sampler + 1 storage_image.
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];
        let dp_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: Device is valid; pool info is fully initialized.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&dp_info, None) }
            .context("failed to create descriptor pool")?;

        // Allocate the single descriptor set.
        let ds_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        // SAFETY: Device and pool are valid; set layout matches pool sizes.
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&ds_alloc_info) }
            .context("failed to allocate descriptor set")?[0];

        info!("blitter: compute pipeline created (blit.comp)");

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
            sampler,
            descriptor_set_layout,
            pipeline_layout,
            compute_pipeline,
            descriptor_pool,
            descriptor_set,
            shader_module,
            output_image_views: Vec::new(),
        };

        // Query all valid modifiers for import (before allocating outputs).
        blitter.import_modifiers = blitter.query_all_valid_modifiers()?;

        Ok(blitter)
    }
}
