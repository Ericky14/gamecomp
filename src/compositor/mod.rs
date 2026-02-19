//! Vulkan compute compositor.
//!
//! Owns the Vulkan device, compute pipeline, and texture pool. All Vulkan
//! operations happen exclusively on the render thread. The compositor receives
//! [`FrameInfo`] scene descriptions from the main thread and either performs
//! direct scanout or dispatches a compute shader for multi-layer composition.
//!
//! Key design choices:
//! - **Compute-only queue**: Uses a dedicated compute queue when available,
//!   avoiding graphics pipeline overhead entirely.
//! - **Push constants**: Global composition parameters are passed via push
//!   constants (no descriptor set updates per frame). Layer parameters use
//!   a persistent storage buffer updated once.
//! - **Pre-allocated resources**: Texture pool, descriptor sets, and command
//!   buffers are allocated at init time. Zero allocation in the hot path.
//! - **Timeline semaphores**: Explicit sync with DRM via `VK_KHR_timeline_semaphore`.

pub mod scene;
pub mod texture;

use std::ffi::CStr;

use anyhow::{Context, bail};
use ash::vk;
use tracing::info;

use self::scene::{CompositionMode, FrameInfo};
use self::texture::TexturePool;

/// Vulkan compute compositor.
///
/// Owns all Vulkan state and runs exclusively on the render thread.
pub struct VulkanCompositor {
    /// Vulkan entry point.
    entry: ash::Entry,
    /// Vulkan instance.
    instance: ash::Instance,
    /// Physical device handle.
    physical_device: vk::PhysicalDevice,
    /// Logical device.
    device: ash::Device,
    /// Compute queue.
    compute_queue: vk::Queue,
    /// Compute queue family index.
    queue_family_index: u32,
    /// Command pool for compute commands.
    command_pool: vk::CommandPool,
    /// Pre-allocated command buffer (reused each frame).
    command_buffer: vk::CommandBuffer,
    /// Compute pipeline for composition.
    composite_pipeline: vk::Pipeline,
    /// Pipeline layout.
    pipeline_layout: vk::PipelineLayout,
    /// Descriptor set layout for texture array.
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Descriptor pool.
    descriptor_pool: vk::DescriptorPool,
    /// Texture pool (pre-allocated slots).
    textures: TexturePool,
    /// Fence for synchronizing command buffer execution.
    compute_fence: vk::Fence,
    /// Whether the device supports timeline semaphores.
    has_timeline_semaphores: bool,
    /// Whether we found a compute-only queue.
    has_compute_only_queue: bool,
}

impl VulkanCompositor {
    /// Create and initialize the Vulkan compositor.
    pub fn new() -> anyhow::Result<Self> {
        // Load Vulkan.
        let entry = unsafe { ash::Entry::load() }.context("failed to load Vulkan loader")?;

        // Create instance.
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"gamecomp")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"gamecomp")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .context("failed to create Vulkan instance")?;

        // Select physical device.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .context("failed to enumerate Vulkan devices")?;

        if physical_devices.is_empty() {
            bail!("no Vulkan-capable GPU found");
        }

        // Prefer discrete GPU, fall back to integrated.
        let physical_device = physical_devices
            .iter()
            .copied()
            .find(|&pd| {
                let props = unsafe { instance.get_physical_device_properties(pd) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or(physical_devices[0]);

        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(device_props.device_name.as_ptr()) };
        info!(
            device = ?device_name,
            device_type = ?device_props.device_type,
            "selected Vulkan device"
        );

        // Find compute queue family — prefer compute-only.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let (queue_family_index, has_compute_only) = queue_families
            .iter()
            .enumerate()
            .filter(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .min_by_key(|(_, props)| {
                // Prefer compute-only queue (fewer flags = more specialized).
                props.queue_flags.as_raw().count_ones()
            })
            .map(|(i, props)| {
                let is_compute_only = !props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                (i as u32, is_compute_only)
            })
            .context("no compute queue family found")?;

        info!(
            queue_family = queue_family_index,
            compute_only = has_compute_only,
            "selected compute queue"
        );

        // Create logical device.
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let mut timeline_semaphore_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);

        let mut vulkan_13_features =
            vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .push_next(&mut timeline_semaphore_features)
            .push_next(&mut vulkan_13_features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .context("failed to create Vulkan device")?;

        let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Create command pool.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .context("failed to create command pool")?;

        // Allocate command buffer.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .context("failed to allocate command buffer")?;
        let command_buffer = command_buffers[0];

        // Create compute fence.
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let compute_fence =
            unsafe { device.create_fence(&fence_info, None) }.context("failed to create fence")?;

        info!("Vulkan compositor initialized");

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            compute_queue,
            queue_family_index,
            command_pool,
            command_buffer,
            composite_pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            textures: TexturePool::new(),
            compute_fence,
            has_timeline_semaphores: true,
            has_compute_only_queue: has_compute_only,
        })
    }

    /// Composite the given frame. Returns when the GPU work is submitted.
    ///
    /// For `DirectScanout` mode, this is a no-op — the caller handles it.
    /// For `Composite` mode, dispatches the compute shader.
    pub fn composite(&mut self, frame: &FrameInfo) -> anyhow::Result<()> {
        match frame.mode {
            CompositionMode::DirectScanout | CompositionMode::Skip => {
                // Nothing to do — direct scanout or skip.
                return Ok(());
            }
            CompositionMode::Composite | CompositionMode::Upscale => {
                // Fall through to compute composition.
            }
        }

        // Wait for previous frame's compute work to finish.
        unsafe {
            self.device
                .wait_for_fences(&[self.compute_fence], true, u64::MAX)
                .context("failed waiting for compute fence")?;
            self.device
                .reset_fences(&[self.compute_fence])
                .context("failed to reset fence")?;
        }

        // Record compute commands.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .context("failed to begin command buffer")?;

            // TODO: Bind pipeline, descriptor sets, push constants, dispatch.
            // This will be implemented when shaders are compiled.

            self.device
                .end_command_buffer(self.command_buffer)
                .context("failed to end command buffer")?;
        }

        // Submit.
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));

        unsafe {
            self.device
                .queue_submit(self.compute_queue, &[submit_info], self.compute_fence)
                .context("failed to submit compute work")?;
        }

        Ok(())
    }

    /// Access the texture pool.
    #[inline(always)]
    pub fn textures(&self) -> &TexturePool {
        &self.textures
    }

    /// Access the texture pool mutably.
    #[inline(always)]
    pub fn textures_mut(&mut self) -> &mut TexturePool {
        &mut self.textures
    }

    /// Wait for all GPU work to complete.
    pub fn wait_idle(&self) -> anyhow::Result<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .context("failed to wait for device idle")
        }
    }
}

impl Drop for VulkanCompositor {
    fn drop(&mut self) {
        // Wait for GPU to finish before destroying resources.
        let _ = self.wait_idle();

        unsafe {
            self.device.destroy_fence(self.compute_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            if self.composite_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.composite_pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
