//! Per-frame blit pipeline and output accessors.
//!
//! Contains the main `blit()` entry point that composites a client DMA-BUF
//! onto the next output image via compute shader dispatch, plus DMA-BUF
//! implicit sync polling and output image accessors.

use super::*;

impl VulkanBlitter {
    /// Check if implicit sync fences on a DMA-BUF fd are ready.
    ///
    /// Some Vulkan drivers do NOT participate in DMA-BUF implicit sync —
    /// the `QUEUE_FAMILY_FOREIGN_EXT` acquire barrier only does cache
    /// management, not fence waiting. We must explicitly poll the DMA-BUF
    /// fd to wait for the client's GPU rendering to complete before
    /// compositing.
    ///
    /// Returns `true` if the buffer is ready (GPU rendering complete),
    /// `false` if `timeout_ms` elapsed without the fence signaling.
    ///
    /// We use non-blocking poll(0) from the event loop for async behavior.
    pub fn poll_dmabuf_ready(fd: std::os::unix::io::BorrowedFd<'_>, timeout_ms: i32) -> bool {
        use std::os::unix::io::AsRawFd;

        let mut pollfd = libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };

        // SAFETY: pollfd is a valid stack-allocated struct, fd is a valid
        // DMA-BUF file descriptor. poll() is signal-safe and will return
        // EINTR if interrupted, which we retry.
        loop {
            let ret = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };
            if ret < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue; // Retry on EINTR
                }
                warn!(?err, "poll on DMA-BUF fd failed (treating as ready)");
                return true; // Non-fatal — best-effort sync
            }
            return ret > 0;
        }
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

    /// Return the DMA-BUF descriptor for an output image.
    ///
    /// Uses the **original** fd from `vkGetMemoryFdKHR` (not a dup).
    /// DRM FBs are created at image allocation time, before any
    /// rendering occurs. The caller should
    /// use this to pre-create DRM framebuffers during initialization.
    ///
    /// # Panics
    /// Panics if `index >= output_count()`.
    pub fn output_dmabuf(&self, index: usize) -> crate::backend::DmaBuf {
        let out = &self.output_images[index];
        crate::backend::DmaBuf {
            width: out.width,
            height: out.height,
            format: drm_fourcc::DrmFourcc::try_from(out.format)
                .expect("invalid output format fourcc"),
            modifier: drm_fourcc::DrmModifier::from(out.modifier),
            planes: out
                .planes
                .iter()
                .map(|p| crate::backend::DmaBufPlane {
                    fd: p.fd,
                    offset: p.offset,
                    stride: p.stride,
                })
                .collect(),
        }
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

        // NOTE: DMA-BUF implicit sync is handled by the caller via
        // `poll_dmabuf_ready()` before calling blit(). The caller uses
        // non-blocking poll(0) for async behavior — if the client GPU
        // isn't done yet, the buffer is deferred to the next event loop
        // iteration instead of blocking here.

        // Cache-aware client DMA-BUF import: reuse VkImage + VkDeviceMemory
        // for the same underlying buffer (identified by device + inode).
        // The QUEUE_FAMILY_FOREIGN_EXT acquire barrier refreshes content
        // each frame without re-importing.
        let stat = rustix::fs::fstat(client_fd).context("blit: fstat client DMA-BUF")?;
        let cache_key = (stat.st_dev, stat.st_ino);

        // Evict on size mismatch (client resized the surface).
        if let Some(cached) = self.import_cache.get(&cache_key)
            && (cached.width != width || cached.height != height)
        {
            let old = self.import_cache.remove(&cache_key).unwrap();
            // SAFETY: GPU work from previous frame completed (fence waited above).
            unsafe {
                self.device.destroy_image_view(old.image_view, None);
                self.device.destroy_image(old.image, None);
                self.device.free_memory(old.memory, None);
            }
            debug!(
                old_w = old.width,
                old_h = old.height,
                new_w = width,
                new_h = height,
                "blit: evicted cached import (size changed)"
            );
        }

        let (src_image, src_image_view) = if let Some(cached) = self.import_cache.get(&cache_key) {
            (cached.image, cached.image_view)
        } else {
            let (image, memory, image_view) = self
                .import_client_dmabuf(client_fd, width, height, format, modifier, offset, stride)?;
            self.import_cache.insert(
                cache_key,
                ImportedImage {
                    image,
                    memory,
                    image_view,
                    width,
                    height,
                },
            );
            trace!(
                src_w = width,
                src_h = height,
                dev = stat.st_dev,
                ino = stat.st_ino,
                "blit: imported and cached client DMA-BUF"
            );
            (image, image_view)
        };

        #[cfg(test)]
        eprintln!(
            "blit: client_modifier=0x{:016x} tiling={}",
            modifier,
            if modifier != DRM_FORMAT_MOD_INVALID {
                "MODIFIER_EXT"
            } else {
                "OPTIMAL"
            }
        );

        // Get next output image.
        let out_idx = self.output_index;
        self.output_index = (self.output_index + 1) % self.output_images.len();
        let out = &self.output_images[out_idx];

        // Compute contain-style scaling: fit content inside output,
        // preserving aspect ratio with letterboxing/pillarboxing.
        let (dst_x, dst_y, dst_w, dst_h) =
            compute_contain_rect(width, height, out.width, out.height);

        debug!(
            out_idx,
            src_w = width,
            src_h = height,
            out_w = out.width,
            out_h = out.height,
            dst_x,
            dst_y,
            dst_w,
            dst_h,
            out_stride = out.planes[0].stride,
            out_offset = out.planes[0].offset,
            out_modifier = format!("0x{:016x}", out.modifier),
            "blit: contain rect computed"
        );

        // Record compute shader composition command.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: All Vulkan handles (device, command_buffer, images, pipeline,
        // descriptors) are valid. src_image was just imported or cache-hit;
        // out.image is a pre-allocated output image. All barriers, descriptor
        // writes, and push constants are fully initialized on the stack.
        // The fence is unsignaled (reset above) and will be signaled on
        // completion. We wait synchronously before returning.
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .context("blit: reset cmd buf")?;
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .context("blit: begin cmd buf")?;

            // Acquire source image from the foreign (client) producer.
            // Use GENERAL layout — never SHADER_READ_ONLY_OPTIMAL. Some
            // drivers transitioning to specialised layouts may trigger
            // decompression that doesn't work correctly with OPTIMAL-tiled
            // DMA-BUF imports.
            //
            // Per VK_EXT_image_drm_format_modifier: transitioning from
            // UNDEFINED with srcQueueFamilyIndex = QUEUE_FAMILY_FOREIGN_EXT
            // preserves the imported pixel data.
            let src_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(self.queue_family_index)
                .image(src_image)
                .subresource_range(COLOR_SUBRESOURCE_RANGE);

            // Acquire dest image from the foreign consumer (DRM scanout).
            //
            // For the FIRST blit to a fresh image we use old_layout=UNDEFINED
            // because there is no previous content or metadata to preserve.
            //
            // For SUBSEQUENT blits we use old_layout=GENERAL — the layout
            // the image was left in by the previous release barrier. This
            // is critical: block-linear tiled images carry internal
            // compression metadata that UNDEFINED may discard without
            // properly reinitializing. Use GENERAL for known images,
            // UNDEFINED only for discarded/new ones.
            let dst_old_layout = if out.blitted {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            };
            let dst_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(dst_old_layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(self.queue_family_index)
                .image(out.image)
                .subresource_range(COLOR_SUBRESOURCE_RANGE);

            // Use ALL_COMMANDS_BIT for both stages. This is more
            // conservative than COMPUTE_SHADER→TOP_OF_PIPE
            // and ensures all GPU caches (including block-linear
            // metadata caches) are properly flushed/invalidated.
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[src_barrier, dst_barrier],
            );

            // --- Update descriptor set with current src + dst ---
            let src_desc_image = vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(src_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let dst_desc_image = vk::DescriptorImageInfo::default()
                .image_view(self.output_image_views[out_idx])
                .image_layout(vk::ImageLayout::GENERAL);

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&src_desc_image)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&dst_desc_image)),
            ];
            self.device.update_descriptor_sets(&descriptor_writes, &[]);

            // --- Bind compute pipeline + descriptors + push constants ---
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            let push = BlitPushConstants {
                dst_w: out.width,
                dst_h: out.height,
                content_offset_x: dst_x,
                content_offset_y: dst_y,
                content_w: dst_w as u32,
                content_h: dst_h as u32,
            };
            // SAFETY: push is a #[repr(C)] struct matching the shader layout.
            let push_bytes: &[u8] = std::slice::from_raw_parts(
                std::ptr::from_ref(&push).cast::<u8>(),
                std::mem::size_of::<BlitPushConstants>(),
            );
            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );

            // --- Dispatch compute shader ---
            // 16×16 workgroup → ceil(width/16) × ceil(height/16) groups.
            let group_x = out.width.div_ceil(16);
            let group_y = out.height.div_ceil(16);
            self.device
                .cmd_dispatch(self.command_buffer, group_x, group_y, 1);

            // Release output image to the foreign consumer (DRM scanout).
            // Ownership transfer to QUEUE_FAMILY_FOREIGN_EXT makes the
            // composition result visible to the display controller via
            // DMA-BUF.
            //
            // dstAccessMask is 0 for the release half — the acquire is
            // performed by the external consumer (DRM).
            let dst_release = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(out.image)
                .subresource_range(COLOR_SUBRESOURCE_RANGE);

            // Release source image back to the foreign (client) producer.
            // Without this, the GPU may not flush caches from the read,
            // which can cause stale data on subsequent imports.
            let src_release = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .image(src_image)
                .subresource_range(COLOR_SUBRESOURCE_RANGE);

            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[dst_release, src_release],
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

            // Wait for blit completion — the fence wait at the top of the
            // next blit() call would also suffice, but we need the result
            // ready for immediate presentation to DRM/Wayland.
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .context("blit: wait for blit completion")?;
        }

        // Mark output image as blitted so subsequent acquires use GENERAL
        // layout (preserving block-linear tiling metadata).
        self.output_images[out_idx].blitted = true;

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
            buffer_index: out_idx,
        })
    }
}
