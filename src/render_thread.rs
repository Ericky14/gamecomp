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

use std::os::unix::io::FromRawFd;
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
    let mut _drm_lease_backend: Option<backend::drm_lease::DrmLeaseBackend> = None;
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
        crate::config::BackendKind::DrmLease => {
            if let Some(raw_fd) = config.drm_lease_fd {
                // SAFETY: The lease fd was inherited from the parent compositor
                // and is guaranteed valid by the spawning process. We take
                // ownership via OwnedFd — the fd will be closed on drop.
                let lease_fd = unsafe { std::os::unix::io::OwnedFd::from_raw_fd(raw_fd) };
                let host_display = config
                    .host_wayland_display
                    .clone()
                    .or(host_wayland_display.clone());
                let mut lease = backend::drm_lease::DrmLeaseBackend::new(lease_fd, host_display);
                if let Err(e) = lease.init() {
                    error!(?e, "failed to initialize DRM lease backend");
                    return;
                }
                let connectors = lease.connectors();
                if let Some(conn) = connectors.first() {
                    let mode = conn.mode;
                    info!(
                        connector = %conn.name,
                        mode_w = mode.size().0,
                        mode_h = mode.size().1,
                        vrr = lease.capabilities().vrr,
                        "DRM lease backend initialized"
                    );
                }
                _drm_lease_backend = Some(lease);
            } else {
                error!("DRM lease backend selected but no lease fd provided");
                return;
            }
        }
        _ => {
            // TODO: Initialize headless backend.
        }
    }

    // --- DRM lease event loop ---
    // Similar to the DRM backend, but uses a lease fd instead of a
    // directly-opened device. Input comes from the Wayland input thread.
    if let Some(ref mut lease) = _drm_lease_backend {
        let rx = committed_frames
            .take()
            .expect("committed_frames receiver missing for DRM lease path");
        if let Err(e) = run_drm_lease_event_loop(
            lease,
            rx,
            &vblank_tx,
            &host_input_tx,
            &host_physical_width,
            &host_physical_height,
        ) {
            error!(?e, "DRM lease event loop exited with error");
        }
        // Ensure the main thread shuts down when the render thread exits
        // (e.g. lease revoked, DRM errors, or normal exit).
        RUNNING.store(false, Ordering::Relaxed);
        info!("render thread exited (DRM lease)");
        return;
    }

    // --- DRM event loop ---
    // For the DRM backend, set up a calloop event loop on the render thread
    // that polls the DRM fd for page flip events.
    if let Some(ref mut drm) = _drm_backend {
        let rx = committed_frames
            .take()
            .expect("committed_frames receiver missing for DRM path");
        if let Err(e) = run_drm_event_loop(
            drm,
            rx,
            &vblank_tx,
            session_active.clone(),
            &host_physical_width,
            &host_physical_height,
        ) {
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
    host_physical_width: &AtomicU32,
    host_physical_height: &AtomicU32,
) -> anyhow::Result<()> {
    let drm_raw_fd = drm.drm_fd().context("DRM backend has no fd")?;

    // Get display resolution for blitter initialization.
    let (display_w, display_h) = drm
        .connectors()
        .first()
        .map(|c| (c.mode.size().0 as u32, c.mode.size().1 as u32))
        .context("no connected display")?;

    // Publish the actual display resolution so the main thread can update
    // the Wayland output advertised to clients. Release ordering pairs
    // with Acquire in the main thread's propagate_host_resolution().
    host_physical_width.store(display_w, Ordering::Release);
    host_physical_height.store(display_h, Ordering::Release);

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

        // NOTE: No poll_dmabuf_ready() check here. The main thread already
        // called sync_dma_buf_fence() in the commit handler before sending
        // the buffer, so the GPU writes are guaranteed complete. With NVIDIA
        // explicit sync, DMA-BUF fds have no implicit fence and poll(POLLIN)
        // would block presentation indefinitely.

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

/// Run the DRM event loop for a lease backend.
///
/// Identical to [`run_drm_event_loop`] but operates on a [`DrmLeaseBackend`]
/// and forwards input events from the Wayland input thread to the main thread.
fn run_drm_lease_event_loop(
    lease: &mut backend::drm_lease::DrmLeaseBackend,
    committed_frames: std::sync::mpsc::Receiver<wayland::protocols::CommittedBuffer>,
    vblank_tx: &std::sync::mpsc::Sender<u64>,
    host_input_tx: &std::sync::mpsc::Sender<backend::wayland::WaylandEvent>,
    host_physical_width: &AtomicU32,
    host_physical_height: &AtomicU32,
) -> anyhow::Result<()> {
    let drm_raw_fd = lease.drm_fd().context("DRM lease backend has no fd")?;

    let (display_w, display_h) = lease
        .connectors()
        .first()
        .map(|c| (c.mode.size().0 as u32, c.mode.size().1 as u32))
        .context("no connected display on lease")?;

    // Publish the actual display resolution so the main thread can update
    // the Wayland output advertised to clients (e.g. Grid).
    host_physical_width.store(display_w, Ordering::Release);
    host_physical_height.store(display_h, Ordering::Release);

    let scanout_modifiers = lease.query_primary_plane_modifiers(drm_fourcc::DrmFourcc::Xrgb8888);
    let mut blitter = backend::gpu::vulkan_blitter::VulkanBlitter::new_for_import()
        .context("failed to create Vulkan blitter for DRM lease")?;

    let importable_modifiers = blitter
        .compute_importable_modifiers(&scanout_modifiers, display_w, display_h)
        .context("failed to compute importable modifiers for lease")?;

    let gbm_outputs = lease
        .allocate_gbm_output_buffers(3, display_w, display_h, &importable_modifiers)
        .context("failed to allocate GBM output buffers on lease")?;

    let gbm_dmabufs: Vec<backend::DmaBuf> = gbm_outputs.iter().map(|o| o.dmabuf.clone()).collect();
    blitter
        .import_output_images(&gbm_dmabufs)
        .context("failed to import GBM output images into Vulkan for lease")?;

    info!(
        display_w,
        display_h,
        output_count = gbm_outputs.len(),
        "Vulkan blitter ready (DRM lease)"
    );

    let mut event_loop =
        calloop::EventLoop::<bool>::try_new().context("failed to create DRM lease event loop")?;
    let handle = event_loop.handle();

    // SAFETY: The DRM lease fd is owned by DrmLeaseBackend which outlives
    // this event loop.
    let wrapper = unsafe { calloop::generic::FdWrapper::new(drm_raw_fd) };
    let source =
        calloop::generic::Generic::new(wrapper, calloop::Interest::READ, calloop::Mode::Level);

    let lease_revoked = Arc::new(AtomicBool::new(false));
    let lease_revoked_cb = Arc::clone(&lease_revoked);

    handle
        .insert_source(source, move |readiness, _fd, flip_ready| {
            // POLLERR on the DRM lease fd means the lease was revoked
            // by the host compositor.
            if readiness.error {
                info!("DRM lease fd signalled error — lease revoked");
                lease_revoked_cb.store(true, Ordering::Relaxed);
                return Ok(calloop::PostAction::Remove);
            }
            *flip_ready = true;
            Ok(calloop::PostAction::Continue)
        })
        .context("failed to register DRM lease fd with event loop")?;

    info!("DRM lease event loop started");

    // Save a "black" output framebuffer for idle/placeholder presentation.
    // GBM buffers are zero-initialized, so this shows as a black screen.
    let idle_fb = gbm_outputs.first().map(|o| o.fb);

    let mut flip_pending = false;
    let mut flip_ready = false;
    let mut frame_count: u64 = 0;
    const MAX_CONSECUTIVE_FAILURES: u32 = 10;
    let mut consecutive_failures: u32 = 0;
    let mut pending_buffer: Option<wayland::protocols::CommittedBuffer> = None;
    let mut modeset_done = false;

    // Fade-in: 300ms ease-in cubic, matching cosmic-comp's fade-out curve.
    let fade_start = std::time::Instant::now();
    const FADE_DURATION: std::time::Duration = std::time::Duration::from_millis(300);

    // Fade-out state: set when EXIT_FADE_OUT is detected.
    let mut fade_out_start: Option<std::time::Instant> = None;

    let mut output_fb_cache: Vec<Option<backend::Framebuffer>> =
        gbm_outputs.iter().map(|o| Some(o.fb)).collect();
    let _gbm_outputs = gbm_outputs;
    let mut prev_scanout_fb: Option<drm::control::framebuffer::Handle> = None;

    while RUNNING.load(Ordering::Relaxed) {
        // Detect lease revocation (POLLERR/POLLHUP on DRM fd).
        if lease_revoked.load(Ordering::Relaxed) {
            info!("DRM lease revoked by host compositor, exiting");
            RUNNING.store(false, Ordering::Relaxed);
            break;
        }

        // Check for fade-out exit request from main thread.
        if fade_out_start.is_none() && crate::EXIT_FADE_OUT.load(Ordering::Relaxed) {
            info!("fade-out exit requested, starting fade to black");
            fade_out_start = Some(std::time::Instant::now());
        }

        // If fade-out is complete, exit.
        if let Some(start) = fade_out_start
            && start.elapsed() >= FADE_DURATION
        {
            info!("fade-out complete, exiting");
            RUNNING.store(false, Ordering::Relaxed);
            break;
        }

        // Forward input events from the Wayland input thread to the main thread.
        for event in lease.drain_input() {
            let _ = host_input_tx.send(event);
        }

        event_loop
            .dispatch(std::time::Duration::from_millis(16), &mut flip_ready)
            .context("DRM lease event loop dispatch failed")?;

        if flip_ready {
            if let Some(vblank_ns) = lease.handle_page_flip()? {
                let _ = vblank_tx.send(vblank_ns);
            }
            if let Some(old_fb) = prev_scanout_fb.take() {
                lease.destroy_framebuffer(old_fb);
            }
            flip_pending = false;
            flip_ready = false;
        }

        if flip_pending {
            trace!("DIAG render: lease flip_pending, skipping");
            continue;
        }

        // --- Initial modeset: present a black frame to claim the display ---
        // This must happen before any client content. The first present()
        // triggers ALLOW_MODESET which sets the display mode and lights up
        // the leased output.
        if !modeset_done {
            if let Some(fb) = idle_fb {
                match lease.present(&fb) {
                    Ok(result) => {
                        info!(
                            ?result,
                            "initial modeset present succeeded, display claimed"
                        );
                        modeset_done = true;
                        crate::DISPLAY_READY.store(true, Ordering::Release);
                        flip_pending = matches!(result, backend::FlipResult::Queued);
                        frame_count += 1;
                    }
                    Err(e) => {
                        warn!(?e, "initial modeset present failed");
                        modeset_done = true; // Don't retry forever
                        crate::DISPLAY_READY.store(true, Ordering::Release);
                    }
                }
            } else {
                modeset_done = true;
                crate::DISPLAY_READY.store(true, Ordering::Release);
            }
            continue;
        }

        let mut latest: Option<wayland::protocols::CommittedBuffer> = None;
        while let Ok(buf) = committed_frames.try_recv() {
            latest = Some(buf);
        }

        if latest.is_some() {
            pending_buffer = latest;
        }

        // NOTE: No poll_dmabuf_ready() check here. The main thread already
        // called sync_dma_buf_fence() in the commit handler before sending
        // the buffer, so the GPU writes are guaranteed complete by the time
        // we receive it. With NVIDIA explicit sync, the DMA-BUF fd has no
        // implicit fence, so poll(POLLIN, 0) would return 0 (not ready)
        // indefinitely, blocking presentation forever.

        if pending_buffer.is_some() {
            info!(
                flip_pending,
                flip_ready,
                modeset_done,
                frame_count,
                "DIAG render: have pending buffer, about to present"
            );
        }

        if let Some(buffer) = pending_buffer.take() {
            // Compute combined fade alpha.
            // Fade-in: ease-in cubic t^3 (0→1) on startup.
            // Fade-out: ease-out cubic 1-(1-t)^3 inverted (1→0) on exit.
            let fade_alpha = if let Some(start) = fade_out_start {
                // Fade-out: content goes from visible to black.
                let elapsed = start.elapsed();
                if elapsed >= FADE_DURATION {
                    0.0_f32
                } else {
                    let t = elapsed.as_secs_f32() / FADE_DURATION.as_secs_f32();
                    // Ease-out: fast start, gentle finish into black.
                    (1.0 - t).powi(3)
                }
            } else {
                // Fade-in on startup.
                let elapsed = fade_start.elapsed();
                if elapsed >= FADE_DURATION {
                    1.0_f32
                } else {
                    let t = elapsed.as_secs_f32() / FADE_DURATION.as_secs_f32();
                    t * t * t
                }
            };

            match present_committed_buffer_lease(
                lease,
                buffer,
                &mut blitter,
                display_w,
                display_h,
                &mut output_fb_cache,
                &mut prev_scanout_fb,
                fade_alpha,
                frame_count,
            ) {
                Ok(is_async) => {
                    flip_pending = is_async;
                    frame_count += 1;
                    consecutive_failures = 0;
                    info!(frame_count, is_async, "DIAG render: frame presented OK");
                }
                Err(e) => {
                    consecutive_failures += 1;
                    info!(
                        ?e,
                        consecutive_failures, "DIAG render: frame present FAILED"
                    );
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                        error!("too many consecutive DRM lease failures, exiting");
                        RUNNING.store(false, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }
    }

    info!(frame_count, "DRM lease event loop exiting");
    Ok(())
}

/// Present a committed buffer via the DRM lease backend.
///
/// Same logic as [`present_committed_buffer`] but using [`DrmLeaseBackend`].
#[allow(clippy::too_many_arguments)]
fn present_committed_buffer_lease(
    lease: &mut backend::drm_lease::DrmLeaseBackend,
    buffer: wayland::protocols::CommittedBuffer,
    blitter: &mut backend::gpu::vulkan_blitter::VulkanBlitter,
    display_w: u32,
    display_h: u32,
    output_fb_cache: &mut [Option<backend::Framebuffer>],
    prev_scanout_fb: &mut Option<drm::control::framebuffer::Handle>,
    alpha: f32,
    frame_count: u64,
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
            // Direct scanout is disabled for now — the client may reuse
            // the DMA-BUF while the CRTC is still scanning it out,
            // causing tearing / static noise. Always copy through the
            // Vulkan blitter which renders into its own output buffers.
            if false && alpha >= 1.0 && width == display_w && height == display_h {
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
                // Direct scanout is best-effort — fall through to the Vulkan
                // blitter if the client buffer format/modifier is incompatible
                // with the DRM plane (e.g. EINVAL from addFB2).
                if let Ok(fb) = lease.import_dmabuf(&dmabuf) {
                    if lease.try_direct_scanout(&fb).unwrap_or(false) {
                        info!("DIAG present: direct scanout path");
                        *prev_scanout_fb = Some(fb.handle);
                        match lease.present(&fb)? {
                            backend::FlipResult::Queued => return Ok(true),
                            backend::FlipResult::DirectScanout => return Ok(false),
                            backend::FlipResult::Failed(e) => {
                                return Err(e.context("lease: direct scanout flip failed"));
                            }
                        }
                    } else {
                        info!("DIAG present: try_direct_scanout returned false, using blitter");
                    }
                } else {
                    info!(
                        width,
                        height,
                        ?fourcc,
                        "DIAG present: import failed, falling back to blitter"
                    );
                }
            }

            let first_plane = &planes[0];
            info!(
                width,
                height,
                format,
                modifier,
                alpha,
                stride = first_plane.stride,
                offset = first_plane.offset,
                "DIAG present: lease blitter path"
            );
            let exported = blitter
                .blit(
                    first_plane.fd.as_fd(),
                    width,
                    height,
                    format,
                    modifier,
                    first_plane.offset,
                    first_plane.stride,
                    alpha,
                )
                .context("Vulkan blit failed (lease)")?;

            info!(
                buffer_index = exported.buffer_index,
                "DIAG present: blit done, presenting output FB"
            );

            let out_fb = output_fb_cache[exported.buffer_index]
                .as_ref()
                .expect("output FB must be pre-created at startup");

            match lease.present(out_fb)? {
                backend::FlipResult::Queued => Ok(true),
                backend::FlipResult::DirectScanout => Ok(false),
                backend::FlipResult::Failed(e) => {
                    Err(e.context("lease: blitted frame flip failed"))
                }
            }
        }
        wayland::protocols::CommittedBuffer::Shm { .. } => {
            trace!("skipping SHM buffer on DRM lease path");
            Ok(false)
        }
    }
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
                // Direct scanout is best-effort — fall through to blit if
                // the client buffer format/modifier is incompatible.
                if let Ok(fb) = drm.import_dmabuf(&dmabuf) {
                    if drm.try_direct_scanout(&fb).unwrap_or(false) {
                        *prev_scanout_fb = Some(fb.handle);
                        match drm.present(&fb)? {
                            backend::FlipResult::Queued => return Ok(true),
                            backend::FlipResult::DirectScanout => return Ok(false),
                            backend::FlipResult::Failed(e) => {
                                return Err(e.context("direct scanout flip failed"));
                            }
                        }
                    }
                } else {
                    debug!(
                        width,
                        height,
                        ?fourcc,
                        "direct scanout import failed, falling back to blitter"
                    );
                }
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
                    1.0,
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
