//! Render thread — owns Vulkan compositor and display backend exclusively.
//!
//! The render thread receives committed buffers from the main thread via
//! an `mpsc` channel, performs GPU composition or direct scanout, and
//! signals flip completion back. It runs entirely on a dedicated thread
//! so that GPU work never blocks Wayland dispatch.
//!
//! Two backend paths are supported:
//! - **Wayland (nested)** — forwards buffers to the host compositor via
//!   `wp_viewporter`, monitoring resize and input events.
//! - **DRM (direct)** — drives the display directly via atomic modesetting
//!   with a calloop event loop for page flip handling.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use anyhow::Context;
use tracing::{debug, error, info, trace, warn};

use crate::RUNNING;
use crate::backend;
use crate::backend::Backend;
use crate::config::Config;
use crate::wayland;

/// Render thread entry point.
///
/// Owns the Vulkan compositor and DRM backend exclusively.
/// Receives FrameInfo from the main thread, composites or performs direct
/// scanout, and signals flip completion back.
#[allow(clippy::too_many_arguments)]
pub fn render_thread_main(
    config: &Config,
    host_wayland_display: Option<String>,
    committed_frames: std::sync::mpsc::Receiver<wayland::protocols::CommittedBuffer>,
    cursor_updates: std::sync::mpsc::Receiver<backend::wayland::CursorUpdate>,
    detected_refresh_mhz: Arc<AtomicU32>,
    host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
    drm_device: Option<(std::path::PathBuf, std::os::unix::io::OwnedFd)>,
    vblank_tx: std::sync::mpsc::Sender<u64>,
    session_active: Option<Arc<AtomicBool>>,
    host_physical_width: Arc<AtomicU32>,
    host_physical_height: Arc<AtomicU32>,
    host_input_tx: std::sync::mpsc::Sender<backend::wayland::WaylandEvent>,
) {
    info!("render thread started");

    // Initialize Vulkan compositor.
    match crate::compositor::VulkanCompositor::new() {
        Ok(_c) => {
            info!("Vulkan compositor ready");
        }
        Err(e) => {
            warn!(
                ?e,
                "Vulkan compositor init failed, will use direct scanout only"
            );
        }
    };

    // Initialize backend based on config.
    // The backend must stay alive for the entire render loop — dropping it
    // signals the event thread to shut down.
    let mut _backend: Option<backend::wayland::WaylandBackend> = None;
    let mut _drm_backend: Option<backend::drm::DrmBackend> = None;
    let mut committed_frames = Some(committed_frames);
    let mut cursor_updates = Some(cursor_updates);

    match config.backend {
        crate::config::BackendKind::Wayland => {
            let mut wayland_config = config.to_wayland_config();
            wayland_config.host_wayland_display = host_wayland_display;
            wayland_config.committed_frame_rx = committed_frames.take();
            wayland_config.cursor_rx = cursor_updates.take();
            wayland_config.detected_refresh_mhz = detected_refresh_mhz;
            wayland_config.host_dmabuf_formats = host_dmabuf_formats;
            let mut backend = backend::wayland::WaylandBackend::new(wayland_config);
            if let Err(e) = backend.init() {
                error!(?e, "failed to initialize wayland backend");
                return;
            }
            info!(
                width = backend.window_size().0,
                height = backend.window_size().1,
                "wayland backend initialized"
            );
            _backend = Some(backend);
        }
        crate::config::BackendKind::Drm => {
            if let Some((path, fd)) = drm_device {
                let mut drm = backend::drm::DrmBackend::new(path.clone(), fd);
                if let Err(e) = drm.init() {
                    error!(?e, path = %path.display(), "failed to initialize DRM backend");
                    return;
                }
                let connectors = drm.connectors();
                if let Some(conn) = connectors.first() {
                    let mode = conn.mode;
                    info!(
                        connector = %conn.name,
                        mode_w = mode.size().0,
                        mode_h = mode.size().1,
                        vrr = drm.capabilities().vrr,
                        "DRM backend initialized"
                    );
                }
                _drm_backend = Some(drm);
            } else {
                error!("DRM backend selected but no device fd received");
                return;
            }
        }
        _ => {
            // TODO: Initialize headless backend.
        }
    }

    // --- DRM event loop ---
    // For the DRM backend, set up a calloop event loop on the render thread
    // that polls the DRM fd for page flip events.
    if let Some(ref mut drm) = _drm_backend {
        let rx = committed_frames
            .take()
            .expect("committed_frames receiver missing for DRM path");
        if let Err(e) = run_drm_event_loop(drm, rx, &vblank_tx, session_active.clone()) {
            error!(?e, "DRM event loop exited with error");
        }
        info!("render thread exited");
        return;
    }

    // Wayland / headless: monitor the backend until shutdown.
    while RUNNING.load(Ordering::Relaxed) {
        if let Some(ref mut backend) = _backend {
            if !backend.is_alive() {
                info!("wayland backend closed, initiating shutdown");
                RUNNING.store(false, Ordering::Relaxed);
                break;
            }
            // Drain events and publish host window size changes to the main
            // thread so it can update the Wayland output resolution.
            // Forward input events so they reach Wayland clients.
            for event in backend.drain_events() {
                match event {
                    backend::wayland::WaylandEvent::Resized {
                        width: _,
                        height: _,
                        physical_width,
                        physical_height,
                    } => {
                        host_physical_width.store(physical_width, Ordering::Release);
                        host_physical_height.store(physical_height, Ordering::Release);
                    }
                    backend::wayland::WaylandEvent::FrameCallback => {
                        // Handled by the backend internally.
                    }
                    backend::wayland::WaylandEvent::CloseRequested => {
                        // Handled by is_alive() check above.
                    }
                    other => {
                        // Forward all input events (Key, Modifiers, Keymap,
                        // PointerMotion, PointerButton, Scroll, FocusIn,
                        // FocusOut) to the main thread.
                        let _ = host_input_tx.send(other);
                    }
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    info!("render thread exited");
}

/// Run the DRM event loop on the render thread.
///
/// Registers the DRM fd as a calloop event source so page flip completions
/// are handled promptly. Drains the committed frame channel each iteration,
/// imports the latest DMA-BUF, and presents it via atomic commit.
///
/// The loop runs until [`RUNNING`] is cleared.
fn run_drm_event_loop(
    drm: &mut backend::drm::DrmBackend,
    committed_frames: std::sync::mpsc::Receiver<wayland::protocols::CommittedBuffer>,
    vblank_tx: &std::sync::mpsc::Sender<u64>,
    session_active: Option<Arc<AtomicBool>>,
) -> anyhow::Result<()> {
    let drm_raw_fd = drm.drm_fd().context("DRM backend has no fd")?;

    // Get display resolution for blitter initialization.
    let (display_w, display_h) = drm
        .connectors()
        .first()
        .map(|c| (c.mode.size().0 as u32, c.mode.size().1 as u32))
        .context("no connected display")?;

    // Create Vulkan blitter for GPU composition. The GBM-backed path
    // allocates output buffers via GBM (native GEM handles), imports
    // them into Vulkan for rendering, and creates DRM FBs directly
    // from GBM — bypassing PRIME_FD_TO_HANDLE which causes tiling
    // metadata corruption on NVIDIA.
    let scanout_modifiers = drm.query_primary_plane_modifiers(drm_fourcc::DrmFourcc::Xrgb8888);
    let mut blitter = backend::gpu::vulkan_blitter::VulkanBlitter::new_for_import()
        .context("failed to create Vulkan blitter for DRM composition")?;

    // Compute the intersection of DRM plane modifiers and Vulkan importable
    // modifiers, then allocate GBM buffers with those modifiers.
    let importable_modifiers = blitter
        .compute_importable_modifiers(&scanout_modifiers, display_w, display_h)
        .context("failed to compute importable modifiers")?;

    let gbm_outputs = drm
        .allocate_gbm_output_buffers(3, display_w, display_h, &importable_modifiers)
        .context("failed to allocate GBM output buffers")?;

    // Import GBM DMA-BUFs into Vulkan as output images.
    let gbm_dmabufs: Vec<backend::DmaBuf> = gbm_outputs.iter().map(|o| o.dmabuf.clone()).collect();
    blitter
        .import_output_images(&gbm_dmabufs)
        .context("failed to import GBM output images into Vulkan")?;

    info!(
        display_w,
        display_h,
        output_count = gbm_outputs.len(),
        "Vulkan blitter ready with GBM-backed output buffers"
    );

    let mut event_loop =
        calloop::EventLoop::<bool>::try_new().context("failed to create DRM event loop")?;
    let handle = event_loop.handle();

    // SAFETY: The DRM fd is owned by DrmBackend which outlives this event
    // loop. The fd remains valid until DrmBackend is dropped after this
    // function returns.
    let wrapper = unsafe { calloop::generic::FdWrapper::new(drm_raw_fd) };
    let source =
        calloop::generic::Generic::new(wrapper, calloop::Interest::READ, calloop::Mode::Level);

    handle
        .insert_source(source, |_readiness, _fd, flip_ready| {
            *flip_ready = true;
            Ok(calloop::PostAction::Continue)
        })
        .context("failed to register DRM fd with event loop")?;

    info!("DRM event loop started");

    let mut flip_pending = false;
    let mut flip_ready = false;
    let mut frame_count: u64 = 0;
    /// Maximum consecutive present failures before giving up.
    const MAX_CONSECUTIVE_FAILURES: u32 = 10;
    let mut consecutive_failures: u32 = 0;

    // Pending buffer awaiting async DMA-BUF sync (client GPU not done yet).
    let mut pending_buffer: Option<wayland::protocols::CommittedBuffer> = None;

    // FB cache is pre-populated from GBM allocation — no PRIME import.
    // Each GBM buffer already has a DRM framebuffer created from its
    // native GEM handle at allocation time.
    let mut output_fb_cache: Vec<Option<backend::Framebuffer>> =
        gbm_outputs.iter().map(|o| Some(o.fb)).collect();

    // Keep GBM output buffers alive — they own the DMA-BUF fds that
    // back the Vulkan output images and DRM framebuffers.
    let _gbm_outputs = gbm_outputs;

    // Track the previous direct-scanout FB so we can destroy it after
    // the page flip completes. Unlike the blit path (where output FBs
    // are cached forever), the direct-scanout path creates a new FB
    // every frame from different client DMA-BUFs.
    let mut prev_scanout_fb: Option<drm::control::framebuffer::Handle> = None;

    let mut was_active = true;

    while RUNNING.load(Ordering::Relaxed) {
        // --- Session pause: stop presenting while VT-switched away ---
        // When the session is disabled (VT switch), the kernel revokes
        // DRM master, so all atomic commits would fail with EACCES.
        // Sleep instead of burning CPU on doomed commits.
        if let Some(ref active) = session_active {
            let is_active = active.load(Ordering::Acquire);
            if !is_active {
                if was_active {
                    info!("session inactive, pausing DRM presents");
                    was_active = false;
                }
                // Drain stale frames so we don't replay old content on resume.
                while committed_frames.try_recv().is_ok() {}
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            } else if !was_active {
                info!("session re-enabled, forcing modeset");
                drm.force_modeset();
                flip_pending = false;
                was_active = true;
            }
        }

        // Block up to 16ms for DRM events.
        event_loop
            .dispatch(std::time::Duration::from_millis(16), &mut flip_ready)
            .context("DRM event loop dispatch failed")?;

        // Process page flip completion.
        if flip_ready {
            if let Some(vblank_ns) = drm.handle_page_flip()? {
                // Send vblank timestamp to the main thread for frame pacing.
                let _ = vblank_tx.send(vblank_ns);
            }
            // The previous scanout FB is safe to destroy now that the
            // display has flipped to the new one.
            if let Some(old_fb) = prev_scanout_fb.take() {
                drm.destroy_framebuffer(old_fb);
            }
            flip_pending = false;
            flip_ready = false;
        }

        // Don't submit a new frame while a flip is pending — the display
        // hardware can only queue one flip at a time.
        if flip_pending {
            continue;
        }

        // Drain the channel, keeping only the latest buffer (drop stale frames).
        let mut latest: Option<wayland::protocols::CommittedBuffer> = None;
        while let Ok(buf) = committed_frames.try_recv() {
            latest = Some(buf);
        }

        // --- Async DMA-BUF implicit sync ---
        //
        // Instead of blocking inside blit() waiting for the client's GPU
        // to finish, we do a non-blocking poll(0) here. If the buffer
        // isn't ready yet, we defer it to the next event loop iteration
        // (calloop re-dispatches every 16ms).
        //
        // Fast path (>99% of frames): client GPU finished before commit
        //   → poll(0) returns instantly → blit + present on this iteration.
        //
        // Slow path (rare): client GPU still rendering at commit time
        //   → store as pending_buffer → retry on next iteration.
        if latest.is_some() {
            pending_buffer = latest;
        }

        if let Some(ref buffer) = pending_buffer {
            let ready = match buffer {
                wayland::protocols::CommittedBuffer::DmaBuf { planes, .. } => {
                    use std::os::unix::io::AsFd;
                    backend::gpu::vulkan_blitter::VulkanBlitter::poll_dmabuf_ready(
                        planes[0].fd.as_fd(),
                        0, // Non-blocking
                    )
                }
                // SHM buffers are CPU-side — always ready.
                wayland::protocols::CommittedBuffer::Shm { .. } => true,
            };

            if !ready {
                trace!("DMA-BUF not ready, deferring to next iteration");
                continue;
            }
        }

        if let Some(buffer) = pending_buffer.take() {
            match present_committed_buffer(
                drm,
                buffer,
                &mut blitter,
                display_w,
                display_h,
                &mut output_fb_cache,
                &mut prev_scanout_fb,
            ) {
                Ok(is_async) => {
                    flip_pending = is_async;
                    frame_count += 1;
                    consecutive_failures = 0;
                    if frame_count % 300 == 1 {
                        debug!(frame_count, "DRM frame presented");
                    }
                }
                Err(e) => {
                    consecutive_failures += 1;
                    warn!(
                        ?e,
                        consecutive_failures,
                        max = MAX_CONSECUTIVE_FAILURES,
                        "DRM present failed"
                    );
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                        error!(
                            consecutive_failures,
                            "too many consecutive DRM present failures, exiting"
                        );
                        RUNNING.store(false, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }
    }

    info!(frame_count, "DRM event loop exiting");
    Ok(())
}

/// Convert a [`CommittedBuffer`] to a backend [`DmaBuf`], import it, and
/// present it to the display.
///
/// Flow:
/// 1. If the client buffer matches the display resolution, try direct
///    scanout (zero-copy — no GPU work).
/// 2. Otherwise, blit the client buffer to a display-resolution output
///    image via [`VulkanBlitter`], then import and present that.
///
/// The blitter solves the NVIDIA primary-plane limitation: the DRM plane
/// cannot scale, so the framebuffer MUST match the CRTC mode dimensions.
/// Returns `true` if a page flip is pending (async), `false` if the
/// commit was synchronous (first modeset frame).
fn present_committed_buffer(
    drm: &mut backend::drm::DrmBackend,
    buffer: wayland::protocols::CommittedBuffer,
    blitter: &mut backend::gpu::vulkan_blitter::VulkanBlitter,
    display_w: u32,
    display_h: u32,
    output_fb_cache: &mut [Option<backend::Framebuffer>],
    prev_scanout_fb: &mut Option<drm::control::framebuffer::Handle>,
) -> anyhow::Result<bool> {
    use std::os::unix::io::{AsFd, AsRawFd};

    match buffer {
        wayland::protocols::CommittedBuffer::DmaBuf {
            planes,
            width,
            height,
            format,
            modifier,
        } => {
            // --- Direct scanout fast path ---
            // If the client buffer already matches the display resolution,
            // skip GPU composition entirely (zero-copy).
            if width == display_w && height == display_h {
                let fourcc = drm_fourcc::DrmFourcc::try_from(format)
                    .map_err(|_| anyhow::anyhow!("unknown DRM format 0x{format:08x}"))?;
                let dmabuf = backend::DmaBuf {
                    width,
                    height,
                    format: fourcc,
                    modifier: drm_fourcc::DrmModifier::from(modifier),
                    planes: planes
                        .iter()
                        .map(|p| backend::DmaBufPlane {
                            fd: p.fd.as_raw_fd(),
                            offset: p.offset,
                            stride: p.stride,
                        })
                        .collect(),
                };
                let fb = drm.import_dmabuf(&dmabuf)?;
                if drm.try_direct_scanout(&fb)? {
                    // Track the previous scanout FB for deferred destruction.
                    // The display controller may still be scanning it out
                    // until our page flip completes.
                    *prev_scanout_fb = Some(fb.handle);
                    match drm.present(&fb)? {
                        backend::FlipResult::Queued => return Ok(true),
                        backend::FlipResult::DirectScanout => return Ok(false),
                        backend::FlipResult::Failed(e) => {
                            return Err(e.context("direct scanout flip failed"));
                        }
                    }
                }
                // TEST_ONLY failed (format/modifier mismatch) — fall through to blit.
            }

            // --- GPU composition path ---
            // Blit the client buffer to a display-resolution output image.
            let first_plane = &planes[0];
            let exported = blitter
                .blit(
                    first_plane.fd.as_fd(),
                    width,
                    height,
                    format,
                    modifier,
                    first_plane.offset,
                    first_plane.stride,
                )
                .context("Vulkan blit failed")?;

            // Use the pre-created DRM framebuffer for this output image.
            // FBs were created at startup,
            // so we just look up by buffer index.
            let out_fb = output_fb_cache[exported.buffer_index]
                .as_ref()
                .expect("output FB must be pre-created at startup");

            debug!(
                buffer_index = exported.buffer_index,
                fb = ?out_fb.handle,
                "present: using DRM framebuffer"
            );

            match drm.present(out_fb)? {
                backend::FlipResult::Queued => Ok(true),
                backend::FlipResult::DirectScanout => Ok(false),
                backend::FlipResult::Failed(e) => Err(e.context("blitted frame flip failed")),
            }
        }
        wayland::protocols::CommittedBuffer::Shm { .. } => {
            // TODO: SHM buffers require GPU-side blit (Vulkan compositor).
            trace!("skipping SHM buffer on DRM path (not yet supported)");
            Ok(false)
        }
    }
}
