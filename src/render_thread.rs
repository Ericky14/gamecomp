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
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

use anyhow::Context;
use tracing::{debug, error, info, trace, warn};

use crate::RUNNING;
use crate::backend;
use crate::backend::Backend;
use crate::config::Config;
use crate::wayland;

/// State shared with the calloop event loop callbacks.
///
/// Two events feed into this:
/// - `flip_ready` — DRM page flip completed, the CRTC accepted the frame.
/// - `acquire_ready` — explicit sync acquire fence signaled, the client
///   GPU has finished rendering and the buffer is safe to scanout.
struct RenderLoopState {
    flip_ready: bool,
    acquire_ready: bool,
}

/// Render thread entry point.
///
/// Owns the Vulkan compositor and DRM backend exclusively.
/// Receives FrameInfo from the main thread, composites or performs direct
/// scanout, and signals flip completion back.
#[allow(clippy::too_many_arguments)]
pub fn render_thread_main(
    config: &Config,
    host_wayland_display: Option<String>,
    committed_frames: std::sync::mpsc::Receiver<wayland::protocols::CommittedFrame>,
    cursor_updates: std::sync::mpsc::Receiver<backend::wayland::CursorUpdate>,
    detected_refresh_mhz: Arc<AtomicU32>,
    host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
    drm_device: Option<(std::path::PathBuf, std::os::unix::io::OwnedFd)>,
    vblank_tx: std::sync::mpsc::Sender<u64>,
    session_active: Option<Arc<AtomicBool>>,
    host_physical_width: Arc<AtomicU32>,
    host_physical_height: Arc<AtomicU32>,
    host_input_tx: std::sync::mpsc::Sender<backend::wayland::WaylandEvent>,
    cursor_x: Arc<AtomicI32>,
    cursor_y: Arc<AtomicI32>,
    presents_in_flight: Arc<AtomicU32>,
    syncobj_device: Option<wayland::protocols::SyncobjDevice>,
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
                    let (mode_w, mode_h) = (mode.size().0 as u32, mode.size().1 as u32);
                    // Publish display resolution so the main thread can
                    // scale the hardware cursor position from client coords
                    // to display coords.
                    host_physical_width.store(mode_w, Ordering::Release);
                    host_physical_height.store(mode_h, Ordering::Release);
                    // Publish refresh rate so the main thread updates the
                    // FPS limiter to match the actual display mode.
                    let vrefresh_mhz = mode.vrefresh() * 1000;
                    detected_refresh_mhz.store(vrefresh_mhz, Ordering::Release);
                }

                // Populate DMA-BUF format/modifier advertisement from real
                // GPU capabilities. Without this, only LINEAR and INVALID
                // are advertised, which forces XWayland into slow copy mode
                // (4 strip copies per frame) because NVIDIA's DRI3 buffers
                // use vendor-specific tiling modifiers that aren't in the
                // fallback list.
                {
                    use drm_fourcc::DrmFourcc;
                    let common_formats = [
                        DrmFourcc::Argb8888,
                        DrmFourcc::Xrgb8888,
                        DrmFourcc::Abgr8888,
                        DrmFourcc::Xbgr8888,
                    ];
                    let mut formats_map = host_dmabuf_formats.lock();
                    for fmt in &common_formats {
                        let mut modifiers = drm.query_primary_plane_modifiers(*fmt);
                        // Always include INVALID so legacy clients without
                        // explicit modifier negotiation can still allocate.
                        let invalid: u64 = 0x00ff_ffff_ffff_ffff;
                        if !modifiers.contains(&invalid) {
                            modifiers.push(invalid);
                        }
                        if !modifiers.is_empty() {
                            formats_map.insert(*fmt as u32, modifiers);
                        }
                    }
                    info!(
                        num_formats = formats_map.len(),
                        total_pairs = formats_map.values().map(|v| v.len()).sum::<usize>(),
                        "DRM: populated DMA-BUF format/modifier advertisement from GPU"
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
        let cursor_rx = cursor_updates
            .take()
            .expect("cursor_updates receiver missing for DRM path");
        if let Err(e) = run_drm_event_loop(
            config,
            drm,
            rx,
            cursor_rx,
            &vblank_tx,
            session_active.clone(),
            cursor_x.clone(),
            cursor_y.clone(),
            &presents_in_flight,
            syncobj_device.clone(),
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
#[allow(clippy::too_many_arguments)]
fn run_drm_event_loop(
    config: &Config,
    drm: &mut backend::drm::DrmBackend,
    committed_frames: std::sync::mpsc::Receiver<wayland::protocols::CommittedFrame>,
    cursor_updates: std::sync::mpsc::Receiver<backend::wayland::CursorUpdate>,
    vblank_tx: &std::sync::mpsc::Sender<u64>,
    session_active: Option<Arc<AtomicBool>>,
    cursor_x: Arc<AtomicI32>,
    cursor_y: Arc<AtomicI32>,
    presents_in_flight: &Arc<AtomicU32>,
    syncobj_device: Option<wayland::protocols::SyncobjDevice>,
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
        calloop::EventLoop::<RenderLoopState>::try_new().context("failed to create DRM event loop")?;
    let handle = event_loop.handle();

    // SAFETY: The DRM fd is owned by DrmBackend which outlives this event
    // loop. The fd remains valid until DrmBackend is dropped after this
    // function returns.
    let wrapper = unsafe { calloop::generic::FdWrapper::new(drm_raw_fd) };
    let source =
        calloop::generic::Generic::new(wrapper, calloop::Interest::READ, calloop::Mode::Level);

    handle
        .insert_source(source, |_readiness, _fd, state| {
            state.flip_ready = true;
            Ok(calloop::PostAction::Continue)
        })
        .context("failed to register DRM fd with event loop")?;

    // Enable VRR (Variable Refresh Rate / FreeSync / Adaptive Sync).
    // With VRR, the display scans out the new frame immediately when
    // the page flip is submitted — no waiting for the next fixed VBlank.
    // This eliminates up to 6.9ms of display latency per frame.
    if config.vrr {
        match drm.set_vrr(true) {
            Ok(()) => info!("VRR enabled on display"),
            Err(e) => warn!(?e, "failed to enable VRR"),
        }
    }

    info!("DRM event loop started");

    let mut flip_pending = false;
    let mut loop_state = RenderLoopState {
        flip_ready: false,
        acquire_ready: false,
    };
    let mut frame_count: u64 = 0;
    /// Maximum consecutive present failures before giving up.
    const MAX_CONSECUTIVE_FAILURES: u32 = 10;
    let mut consecutive_failures: u32 = 0;

    // ── Diagnostic counters for frame pacing analysis ──
    let mut last_pageflip_ns: u64 = 0;
    let mut last_present_submit_ns: u64 = 0;
    let mut diag_frame_count: u64 = 0;
    let mut diag_sum_recv_to_present_us: u64 = 0;
    let mut diag_sum_present_to_flip_us: u64 = 0;
    let mut diag_sum_flip_interval_us: u64 = 0;
    let mut diag_last_report_ns: u64 = 0;

    // Explicit sync diagnostic: count times the acquire fence wasn't ready
    // at the first check (client GPU still rendering at commit time).
    let mut acquire_not_ready_count: u64 = 0;

    // Pending frame awaiting async DMA-BUF sync (client GPU not done yet).
    let mut pending_frame: Option<wayland::protocols::CommittedFrame> = None;

    // Calloop registration token for the acquire fence eventfd.
    // When a pending frame's acquire fence isn't immediately ready, we
    // register an eventfd with calloop so the render thread sleeps instead
    // of busy-polling. The token is used to remove the source after the
    // fence signals.
    let mut acquire_token: Option<calloop::RegistrationToken> = None;

    // Timestamp when the current acquire eventfd was registered. Used to
    // detect fences that never signal (e.g., NVIDIA GPU context went cold)
    // and abandon the frame after a timeout.
    let mut acquire_registered_at: Option<std::time::Instant> = None;

    // Render loop iteration counter for periodic diagnostics.
    let mut render_loop_iters: u64 = 0;

    // FB cache is pre-populated from GBM allocation — no PRIME import.
    // Each GBM buffer already has a DRM framebuffer created from its
    // native GEM handle at allocation time.
    let mut output_fb_cache: Vec<Option<backend::Framebuffer>> =
        gbm_outputs.iter().map(|o| Some(o.fb)).collect();

    // Keep GBM output buffers alive — they own the DMA-BUF fds that
    // back the Vulkan output images and DRM framebuffers.
    let _gbm_outputs = gbm_outputs;

    // Track direct-scanout FBs using a two-stage pipeline.
    // Unlike the blit path (where output FBs are cached forever), the
    // direct-scanout path creates a new FB every frame from different
    // client DMA-BUFs. We need two slots because the CRTC continues
    // scanning the active FB until the pending one's pageflip completes:
    //
    //   present(fb_N) → pending = fb_N
    //   pageflip fires → destroy active (fb_N-2), active = pending (fb_N-1)
    //
    // Without this, destroying the FB on its own pageflip makes the CRTC
    // scan a freed framebuffer → NVIDIA shows black between frames.
    let mut active_scanout_fb: Option<backend::Framebuffer> = None;
    let mut pending_scanout_fb: Option<backend::Framebuffer> = None;

    // Last successfully presented framebuffer (blit or direct scanout).
    // Used on VT resume to re-present the last visible frame instead
    // of clearing to black — avoids a permanent black screen when the
    // focused client is idle and won't commit a fresh frame.
    let mut last_presented_fb: Option<backend::Framebuffer> = None;

    let mut was_active = true;

    // VRR idle keepalive: track last present time so we can re-flip
    // the last framebuffer when the client is idle. Without this, the
    // VRR display enters a low-refresh state during idle and the first
    // resume flip takes ~180ms to complete, causing a visible stutter.
    // Gamescope avoids this by always repainting in paint_all() even
    // when idle — we emulate that by re-presenting the last GBM output
    // buffer at a low cadence (~10fps) to keep the display warm.
    let mut last_present_time = std::time::Instant::now();
    /// Interval after which we re-present the last frame to keep VRR
    /// from lowering the display refresh rate.
    const VRR_KEEPALIVE_MS: u64 = 100;

    // Track last cursor position to avoid redundant atomic commits.
    let mut last_cursor_x: i32 = i32::MIN;
    let mut last_cursor_y: i32 = i32::MIN;
    let mut cursor_visible = false;

    // Ensure cursor plane starts hidden — DRM may retain state from
    // a previous session or compositor.
    let _ = drm.hide_cursor();

    // Track last-seen client buffer dimensions for cursor scaling.
    // When the client renders at a lower resolution than the display,
    // the cursor image must be scaled up to match the upscaled frame.
    let mut last_client_w: u32 = display_w;
    let mut last_client_h: u32 = display_h;



    // Startup grace period: keep the black modeset frame on screen for a
    // fixed duration after the first client frame arrives. XWayland's
    // initial surface commits often contain uninitialized VRAM or partial
    // renders — the acquire fence signals immediately because no real GPU
    // work was submitted yet. Rather than guessing a frame count, we use
    // a time-based grace period and signal release points for every
    // dropped frame so the client keeps rendering. Once the grace period
    // expires the next frame is presented — by then the client has had
    // enough render cycles to produce real content.
    const STARTUP_GRACE_MS: u64 = 300;
    let mut startup_grace_start: Option<std::time::Instant> = None;
    let mut startup_done = false;

    // Cache of (format, modifier) combinations that failed direct scanout.
    // --- Present a black frame for the initial modeset ---
    // Clear the first GBM output buffer to opaque black and present it
    // synchronously. This completes the modeset with a clean black screen
    // instead of showing uninitialized GPU memory or a partial first
    // frame from XWayland before the client app is ready.
    blitter.clear_output(0).context("failed to clear output buffer for initial modeset")?;
    let black_fb = output_fb_cache[0]
        .as_ref()
        .expect("output FB must be pre-created at startup");
    match drm.present(black_fb)? {
        backend::FlipResult::DirectScanout => {
            info!("initial modeset: presented black frame (synchronous)");
        }
        other => {
            warn!(?other, "initial modeset: unexpected flip result for black frame");
        }
    }

    // VT pause saved frame: when entering VT pause, blit the latest
    // client frame to a GBM output buffer so we can release the client's
    // buffer immediately. The client keeps all its buffers during pause,
    // continues GPU work, and the GPU channel stays warm. On resume we
    // present this saved GBM buffer for an instant modeset.
    let mut vt_pause_saved_fb: Option<usize> = None;

    // Track the last successfully blitted output buffer index. Used as
    // a fallback for VT pause when no pending_frame is available (common
    // at high frame rates where frames are presented immediately). This
    // avoids showing a black screen on VT resume.
    let mut last_blit_output_idx: Option<usize> = None;

    while RUNNING.load(Ordering::Relaxed) {
        // --- Session pause: stop presenting while VT-switched away ---
        // When the session is disabled (VT switch), the kernel revokes
        // DRM master, so all atomic commits would fail with EACCES.
        // Sleep instead of burning CPU on doomed commits.
        //
        // IMPORTANT: if a pageflip is in-flight when the VT switch
        // fires, we must drain it before entering the sleep loop.
        // The kernel may still deliver the PAGE_FLIP_EVENT even after
        // revoking DRM master — waiting for it ensures the release
        // pipeline completes cleanly (active/pending release points
        // get signaled via the normal pageflip handler), the client
        // keeps its buffers, and the GPU channel stays warm.
        if let Some(ref active) = session_active {
            let is_active = active.load(Ordering::Acquire);
            if !is_active {
                if was_active {
                    let now_us = crate::wayland::monotonic_ns() / 1000;
                    eprintln!("[GC_VT] session_pause: t={now_us}us");
                    info!(
                        flip_pending,
                        "session inactive, pausing DRM presents"
                    );
                    was_active = false;
                    // Clean up any pending acquire eventfd — the fence
                    // state needs a fresh check when we resume or get a
                    // newer frame.
                    if let Some(token) = acquire_token.take() {
                        event_loop.handle().remove(token);
                    }
                    loop_state.acquire_ready = false;
                    acquire_registered_at = None;

                    // Wait for the in-flight pageflip to complete so
                    // presents_in_flight decrements cleanly.
                    if flip_pending {
                        info!("waiting for in-flight pageflip before pausing");
                        let drain_deadline = std::time::Instant::now()
                            + std::time::Duration::from_millis(50);
                        while flip_pending && std::time::Instant::now() < drain_deadline {
                            event_loop
                                .dispatch(
                                    std::time::Duration::from_millis(16),
                                    &mut loop_state,
                                )
                                .context("DRM event loop dispatch failed (VT drain)")?;
                            if loop_state.flip_ready {
                                if let Some(vblank_ns) = drm.handle_page_flip()? {
                                    let _ = vblank_tx.send(vblank_ns);
                                    presents_in_flight.fetch_sub(1, Ordering::Release);
                                }
                                if let Some(old_fb) = active_scanout_fb.take() {
                                    drm.destroy_framebuffer(old_fb.handle);
                                }
                                active_scanout_fb = pending_scanout_fb.take();
                                flip_pending = false;
                                loop_state.flip_ready = false;
                                info!("in-flight pageflip drained before VT pause");
                            }
                        }
                    }

                    // If the flip still didn't complete (kernel ate it),
                    // force-clean the state so we don't deadlock on resume.
                    if flip_pending {
                        warn!("pageflip drain timed out, force-cleaning state");
                        presents_in_flight.store(0, Ordering::Release);
                        flip_pending = false;
                    }

                    // Blit the pending frame (if any) to a GBM output
                    // buffer and release the client's buffer immediately.
                    // This gives the client ALL its buffers back during
                    // the VT pause so it keeps submitting GPU work and
                    // the GPU channel stays warm. On resume we present
                    // the saved GBM copy for an instant modeset.
                    if let Some(frame) = pending_frame.take()
                        && let wayland::protocols::CommittedBuffer::DmaBuf {
                            ref planes,
                            width,
                            height,
                            format,
                            modifier,
                            release_point,
                            ..
                        } = frame.app
                    {
                        use std::os::unix::io::AsFd;
                        let p = &planes[0];
                        match blitter.blit(
                            p.fd.as_fd(),
                            width,
                            height,
                            format,
                            modifier,
                            p.offset,
                            p.stride,
                        ) {
                            Ok(exported) => {
                                vt_pause_saved_fb = Some(exported.buffer_index);
                                info!(
                                    buf_idx = exported.buffer_index,
                                    "blitted pending frame to GBM for VT pause"
                                );
                            }
                            Err(e) => {
                                warn!(?e, "failed to blit pending frame for VT pause");
                            }
                        }
                        // Signal the release point so the client
                        // gets this buffer back immediately.
                        if let Some(rp) = release_point
                            && let Some(ref device) = syncobj_device
                        {
                            let _ = device.timeline_signal(rp.handle(), rp.point);
                        }
                    }
                    // Fallback: if no pending frame was available (common
                    // at high frame rates where frames are presented
                    // immediately), reuse the last blitted output buffer.
                    // This avoids a black screen on VT resume.
                    if vt_pause_saved_fb.is_none()
                        && let Some(idx) = last_blit_output_idx {
                            vt_pause_saved_fb = Some(idx);
                            info!(
                                buf_idx = idx,
                                "reusing last blitted output for VT pause (no pending frame)"
                            );
                        }
                }
                // Drain ALL incoming frames and signal their release points.
                // We already saved a copy to a GBM buffer, so every
                // client buffer can be returned immediately — keeping
                // the client's GPU pipeline active during VT pause.
                while let Ok(frame) = committed_frames.try_recv() {
                    if let wayland::protocols::CommittedBuffer::DmaBuf { release_point: Some(rp), .. } = &frame.app
                        && let Some(ref device) = syncobj_device
                    {
                        let _ = device.timeline_signal(rp.handle(), rp.point);
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            } else if !was_active {
                let vt_resume_us = crate::wayland::monotonic_ns() / 1000;
                let has_saved_fb = vt_pause_saved_fb.is_some();
                eprintln!("[GC_VT] session_resume: t={vt_resume_us}us saved_fb={has_saved_fb}");
                info!(
                    has_saved_fb,
                    "session re-enabled, forcing modeset"
                );
                drm.force_modeset();
                flip_pending = false;
                was_active = true;

                // Clean up the stale scanout pipeline from before VT switch.
                // Pending FB was never scanned — safe to destroy outright.
                // Active FB may be the same as last_presented_fb (direct
                // scanout case) — skip destroy to avoid use-after-free.
                if let Some(old) = pending_scanout_fb.take() {
                    drm.destroy_framebuffer(old.handle);
                }
                if let Some(old) = active_scanout_fb.take()
                    && !last_presented_fb.is_some_and(|fb| fb.handle == old.handle)
                {
                    drm.destroy_framebuffer(old.handle);
                }

                // Present the saved GBM buffer (blitted at pause entry)
                // for an instant modeset. This is our own buffer — the
                // client's buffers were already released during pause.
                // Fall back to black if we have no saved frame.
                let resume_idx = vt_pause_saved_fb.take().unwrap_or(0);
                if !has_saved_fb {
                    if let Err(e) = blitter.clear_output(resume_idx) {
                        warn!(?e, "failed to clear output for VT resume");
                    }
                    info!("presenting black frame for VT resume modeset (no saved frame)");
                }
                let resume_fb = *output_fb_cache[resume_idx]
                    .as_ref()
                    .expect("output FB must be pre-created at startup");
                match drm.present(&resume_fb) {
                    Ok(backend::FlipResult::Queued) => {
                        flip_pending = true;
                    }
                    Ok(_) => {
                        let now_ns = crate::wayland::monotonic_ns();
                        let _ = vblank_tx.send(now_ns);
                    }
                    Err(e) => {
                        warn!(?e, "failed to present after VT switch");
                    }
                }
            }
        }

        // When a flip is pending, block up to 16ms for the DRM page flip
        // event. When no flip is pending, we're ready to present immediately
        // so use a short timeout to avoid sleeping through frame delivery
        // (the committed_frames channel isn't a calloop source).
        let dispatch_timeout = if flip_pending {
            std::time::Duration::from_millis(16)
        } else {
            std::time::Duration::from_millis(1)
        };
        event_loop
            .dispatch(dispatch_timeout, &mut loop_state)
            .context("DRM event loop dispatch failed")?;

        render_loop_iters += 1;
        // Periodic render-loop diagnostic: print every ~1000 iterations
        // (~1 second at 1ms dispatch timeout). This confirms the loop is
        // alive and shows fence-wait state.
        if render_loop_iters.is_multiple_of(1000) {
            let has_pending = pending_frame.is_some();
            let has_acq = acquire_token.is_some();
            let acq_elapsed = acquire_registered_at
                .map(|r| r.elapsed().as_millis() as u64)
                .unwrap_or(0);
            eprintln!(
                "[GC_DIAG] iter={render_loop_iters} pending={has_pending} acq_token={has_acq} \
                 acq_elapsed_ms={acq_elapsed} flip_pending={flip_pending} acq_ready={}",
                loop_state.acquire_ready,
            );
        }

        // Process page flip completion.
        if loop_state.flip_ready {
            if let Some(vblank_ns) = drm.handle_page_flip()? {
                // Diagnostic: measure present-to-flip (display latency)
                let present_to_flip_us = if last_present_submit_ns > 0 {
                    (vblank_ns.saturating_sub(last_present_submit_ns)) / 1000
                } else {
                    0
                };
                let flip_interval_us = if last_pageflip_ns > 0 {
                    (vblank_ns.saturating_sub(last_pageflip_ns)) / 1000
                } else {
                    0
                };
                last_pageflip_ns = vblank_ns;
                diag_sum_present_to_flip_us += present_to_flip_us;
                diag_sum_flip_interval_us += flip_interval_us;

                // Send vblank timestamp to the main thread for frame pacing.
                let _ = vblank_tx.send(vblank_ns);
                // Signal that one in-flight present has completed.
                presents_in_flight.fetch_sub(1, Ordering::Release);
            }
            // Rotate the GBM output FB pipeline: the pending FB is now
            // actively being scanned by the CRTC, so the old active FB
            // (from two flips ago) is safe to destroy.
            if let Some(old_fb) = active_scanout_fb.take() {
                drm.destroy_framebuffer(old_fb.handle);
            }
            active_scanout_fb = pending_scanout_fb.take();
            // NOTE: Release points are signaled immediately after blit
            // (not on pageflip). No release point rotation needed here.
            flip_pending = false;
            loop_state.flip_ready = false;
        }

        // --- Cursor plane updates (independent of primary plane flip) ---
        // Drain cursor image updates from the Wayland protocol layer.
        while let Ok(update) = cursor_updates.try_recv() {
            match update {
                backend::wayland::CursorUpdate::Image {
                    pixels,
                    width,
                    height,
                    hotspot_x,
                    hotspot_y,
                } => {
                    // Scale cursor image when the client renders at a
                    // lower resolution than the display. Use the minimum
                    // of X/Y scale factors to match the blitter's
                    // aspect-preserving "contain" scaling.
                    let scale_x = if last_client_w > 0 {
                        display_w as f64 / last_client_w as f64
                    } else {
                        1.0
                    };
                    let scale_y = if last_client_h > 0 {
                        display_h as f64 / last_client_h as f64
                    } else {
                        1.0
                    };
                    let scale = scale_x.min(scale_y).max(1.0);
                    if let Err(e) =
                        drm.update_cursor_image(&pixels, width, height, hotspot_x, hotspot_y, scale)
                    {
                        warn!(?e, "failed to update cursor image");
                    } else {
                        cursor_visible = true;
                        // Force position update after image change.
                        last_cursor_x = i32::MIN;
                    }
                }
                backend::wayland::CursorUpdate::Hide => {
                    if let Err(e) = drm.hide_cursor() {
                        warn!(?e, "failed to hide cursor");
                    }
                    cursor_visible = false;
                }
            }
        }

        // Update cursor plane position when the pointer has moved.
        if cursor_visible {
            let cx = cursor_x.load(Ordering::Relaxed);
            let cy = cursor_y.load(Ordering::Relaxed);
            if cx != last_cursor_x || cy != last_cursor_y {
                if let Err(e) = drm.update_cursor_position(cx, cy) {
                    warn!(?e, "failed to update cursor position");
                }
                last_cursor_x = cx;
                last_cursor_y = cy;
            }
        }

        // Don't submit a new frame while a flip is pending — the display
        // hardware can only queue one flip at a time.
        if flip_pending {
            continue;
        }

        // Drain the channel, keeping only the latest frame (drop stale frames).
        // Signal release points of coalesced frames so the client can
        // reclaim its buffers. Without this, XWayland's buffer pool gets
        // exhausted and rendering freezes.
        let mut latest: Option<wayland::protocols::CommittedFrame> = None;
        while let Ok(frame) = committed_frames.try_recv() {
            if let Some(mut prev) = latest.replace(frame)
                && let Some(ref device) = syncobj_device
            {
                prev.signal_release_points(device);
            }
        }

        // --- DMA-BUF readiness check ---
        //
        // Ensure the client GPU has finished rendering before presenting.
        //
        // Explicit sync path (acquire_point set): register a DRM syncobj
        //   eventfd with calloop. The eventfd fires when the GPU signals
        //   the acquire fence (rendering complete). This avoids busy-polling
        //   and lets the render thread sleep efficiently — critical after
        //   VT switch when NVIDIA's GPU may take seconds to restore.
        //
        // Implicit sync path (no acquire_point): poll the DMA-BUF fd for
        //   the implicit fence. Non-blocking poll(0) returns true when the
        //   client GPU has finished.
        //
        // If not ready, the event loop dispatch will wake us when the
        // eventfd fires or a new frame arrives.
        if latest.is_some() {
            // New frame replaces the pending one. Cancel any outstanding
            // acquire eventfd from the previous pending frame and signal
            // its release point so the client gets the buffer back.
            if let Some(mut old) = pending_frame.take()
                && let Some(ref device) = syncobj_device
            {
                old.signal_release_points(device);
            }
            if let Some(token) = acquire_token.take() {
                event_loop.handle().remove(token);
                loop_state.acquire_ready = false;
                acquire_registered_at = None;
            }
            pending_frame = latest;
        }

        if let Some(ref frame) = pending_frame {
            let ready = match &frame.app {
                wayland::protocols::CommittedBuffer::DmaBuf {
                    planes,
                    acquire_point,
                    ..
                } => {
                    if let Some(ap) = acquire_point {
                        if loop_state.acquire_ready {
                            // Eventfd already fired — fence is signaled.
                            let now_us = crate::wayland::monotonic_ns() / 1000;
                            eprintln!("[GC_FENCE] acquire_ready: t={now_us}us");
                            true
                        } else if let Some(ref device) = syncobj_device {
                            // Gamescope-style fast path: query the timeline's
                            // signaled point directly (drmSyncobjQuery).
                            // This is a cheap, non-blocking ioctl that
                            // returns the current signaled point. If it's
                            // >= our target, the GPU has finished and we can
                            // present immediately — no eventfd needed.
                            //
                            // We run this check on every loop iteration,
                            // matching gamescope's approach: the eventfd
                            // serves as the primary wakeup mechanism (so we
                            // don't busy-poll in a tight loop), but the
                            // query check catches any case where the eventfd
                            // notification was delayed or lost.
                            let is_ready = device.is_acquire_ready(ap.handle(), ap.point);
                            if is_ready {
                                // Fence is signaled. Clean up eventfd if one
                                // was registered.
                                if let Some(token) = acquire_token.take() {
                                    let elapsed_ms = acquire_registered_at
                                        .map_or(0, |t| t.elapsed().as_millis() as u64);
                                    if !loop_state.acquire_ready {
                                        debug!(
                                            elapsed_ms,
                                            "acquire fence signaled (query), eventfd did not fire"
                                        );
                                    }
                                    event_loop.handle().remove(token);
                                }
                                acquire_registered_at = None;
                                true
                            } else if acquire_token.is_none() {
                                // Not yet signaled and no eventfd registered.
                                // Register one so calloop wakes us when the
                                // kernel signals the fence.
                                acquire_not_ready_count += 1;
                                match device.acquire_eventfd(ap.handle(), ap.point) {
                                    Ok(efd) => {
                                        let src = calloop::generic::Generic::new(
                                            efd,
                                            calloop::Interest::READ,
                                            calloop::Mode::OneShot,
                                        );
                                        match event_loop.handle().insert_source(src, |_readiness, _fd, state| {
                                            state.acquire_ready = true;
                                            Ok(calloop::PostAction::Remove)
                                        }) {
                                            Ok(token) => {
                                                let reg_us = crate::wayland::monotonic_ns() / 1000;
                                                eprintln!("[GC_FENCE] eventfd_registered: t={reg_us}us handle={:?} point={}", ap.handle(), ap.point);
                                                acquire_token = Some(token);
                                                acquire_registered_at = Some(std::time::Instant::now());
                                                debug!("registered acquire eventfd for syncobj fence");
                                            }
                                            Err(e) => {
                                                warn!(?e, "failed to register acquire eventfd, presenting anyway");
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        warn!(?e, "failed to create acquire eventfd, presenting anyway");
                                    }
                                }
                                // If eventfd setup failed, present anyway
                                // rather than waiting for a signal that
                                // will never come.
                                acquire_token.is_none()
                            } else {
                                // Eventfd registered, fence not yet signaled
                                // per query. Wait for calloop to wake us.
                                false
                            }
                        } else {
                            // No syncobj device — shouldn't happen, but
                            // fall through to present anyway.
                            true
                        }
                    } else if syncobj_device.is_some() {
                        // Explicit sync device exists but this frame has no
                        // acquire point. The client hasn't set up syncobj
                        // timelines yet (e.g., XWayland's initial commit
                        // before the app configures explicit sync). NVIDIA
                        // does not set implicit DMA-BUF fences when explicit
                        // sync is negotiated — the fd poll below would
                        // return "ready" immediately on an in-progress
                        // buffer. Drop this frame to avoid presenting
                        // unfinished GPU content (static noise).
                        debug!("dropping frame without acquire_point (explicit sync not yet established)");
                        pending_frame = None;
                        continue;
                    } else {
                        // Implicit sync: poll the DMA-BUF fd.
                        use std::os::unix::io::AsFd;
                        backend::gpu::vulkan_blitter::VulkanBlitter::poll_dmabuf_ready(
                            planes[0].fd.as_fd(),
                            0, // Non-blocking
                        )
                    }
                }
                // SHM buffers are CPU-side — always ready.
                wayland::protocols::CommittedBuffer::Shm { .. } => true,
            };

            if !ready {
                trace!("buffer not ready, waiting for acquire eventfd");
                continue;
            }

            // Acquire fence signaled — the eventfd callback already
            // removed the source (PostAction::Remove), so just drop
            // the token.
            acquire_token.take();
            loop_state.acquire_ready = false;
            acquire_registered_at = None;
        }

        // --- Startup grace period ---
        // Keep the black modeset frame on screen until the grace period
        // expires. Signal release points for dropped frames so the client
        // keeps its buffer pool flowing and GPU channel warm.
        if !startup_done {
            if let Some(ref grace_start) = startup_grace_start {
                if grace_start.elapsed() < std::time::Duration::from_millis(STARTUP_GRACE_MS) {
                    if let Some(frame) = pending_frame.take() {
                        if let wayland::protocols::CommittedBuffer::DmaBuf { release_point: Some(rp), .. } = &frame.app
                            && let Some(ref device) = syncobj_device
                        {
                            let _ = device.timeline_signal(rp.handle(), rp.point);
                        }
                        info!(
                            elapsed_ms = grace_start.elapsed().as_millis() as u64,
                            "dropping startup frame (grace period active)"
                        );
                    }
                    continue;
                }
                // Grace period expired — present next real frame.
                startup_done = true;
            } else if pending_frame.is_some() {
                // First client frame arrived — start the grace period.
                startup_grace_start = Some(std::time::Instant::now());
                if let Some(frame) = pending_frame.take() {
                    if let wayland::protocols::CommittedBuffer::DmaBuf { release_point: Some(rp), .. } = &frame.app
                        && let Some(ref device) = syncobj_device
                    {
                        let _ = device.timeline_signal(rp.handle(), rp.point);
                    }
                    info!("dropping first startup frame, starting grace period");
                }
                continue;
            }
        }

        if let Some(mut frame) = pending_frame.take() {
            let present_us = crate::wayland::monotonic_ns() / 1000;
            eprintln!("[GC_FENCE] present_frame: t={present_us}us");
            let present_start = std::time::Instant::now();
            let recv_ns = crate::wayland::monotonic_ns();

            // Check overlay acquire fences (non-blocking).
            //
            // Matching gamescope: overlay acquire fences are NEVER blocking
            // dependencies. The GPU may read incomplete overlay data, but
            // the overlay is alpha-blended on top and brief glitches are
            // invisible. This prevents a slow overlay fence from starving
            // the game pipeline (attempt 6 failure: 5s overlay fence stall
            // → black screen because render thread was blocked).
            if let Some(ref device) = syncobj_device {
                check_overlay_acquire(&frame.overlay, device);
                check_overlay_acquire(&frame.external_overlay, device);
            }

            // Extract explicit sync release points before consuming the frame.
            let frame_release_point = match &mut frame.app {
                wayland::protocols::CommittedBuffer::DmaBuf {
                    width,
                    height,
                    release_point,
                    ..
                } => {
                    if *width > 0 {
                        last_client_w = *width;
                    }
                    if *height > 0 {
                        last_client_h = *height;
                    }
                    release_point.take()
                }
                wayland::protocols::CommittedBuffer::Shm { width, height, .. } => {
                    if *width > 0 {
                        last_client_w = *width;
                    }
                    if *height > 0 {
                        last_client_h = *height;
                    }
                    None
                }
            };
            // Take overlay release points — signaled after the blit
            // composite finishes reading the overlay DMA-BUFs.
            let overlay_release = take_overlay_release(&mut frame.overlay);
            let ext_overlay_release = take_overlay_release(&mut frame.external_overlay);

            match present_committed_frame(
                drm,
                frame,
                &mut blitter,
                &mut output_fb_cache,
            ) {
                Ok((is_async, buf_idx)) => {
                    // Track last presented FB for VT resume.
                    if let Some(idx) = buf_idx {
                        // Blit path: the GBM output buffer was presented.
                        last_blit_output_idx = Some(idx);
                        if let Some(Some(fb)) = output_fb_cache.get(idx) {
                            last_presented_fb = Some(*fb);
                        }
                    }
                    let present_us = present_start.elapsed().as_micros();
                    let submit_ns = crate::wayland::monotonic_ns();
                    last_present_submit_ns = submit_ns;
                    let recv_to_present_us = (submit_ns - recv_ns) / 1000;
                    diag_sum_recv_to_present_us += recv_to_present_us;
                    diag_frame_count += 1;

                    // Signal release points immediately post-blit.
                    // The Vulkan blitter has finished reading all DMA-BUFs
                    // — the display scans from the GBM output copy. Signal
                    // app + overlay release points so clients can recycle.
                    if let Some(ref device) = syncobj_device {
                        if let Some(rp) = frame_release_point
                            && let Err(e) = device.timeline_signal(rp.handle(), rp.point)
                        {
                            warn!(?e, point = rp.point, "failed to signal app release");
                        }
                        if let Some(rp) = overlay_release
                            && let Err(e) = device.timeline_signal(rp.handle(), rp.point)
                        {
                            warn!(?e, point = rp.point, "failed to signal overlay release");
                        }
                        if let Some(rp) = ext_overlay_release
                            && let Err(e) = device.timeline_signal(rp.handle(), rp.point)
                        {
                            warn!(?e, point = rp.point, "failed to signal ext overlay release");
                        }
                    }

                    flip_pending = is_async;
                    if is_async {
                        // Track in-flight presents. Decremented on page flip
                        // completion. The main loop gates frame callbacks on
                        // this reaching 0 (gamescope-style backpressure).
                        presents_in_flight.fetch_add(1, Ordering::Release);
                    } else {
                        // Synchronous flip (e.g., modeset on VT resume) —
                        // no PAGE_FLIP_EVENT will arrive, so send a
                        // synthetic vblank to the main thread. Without
                        // this the client never receives wl_callback.done
                        // and stalls for seconds until the next commit.
                        let _ = vblank_tx.send(submit_ns);
                    }
                    last_present_time = std::time::Instant::now();
                    frame_count += 1;
                    consecutive_failures = 0;
                    trace!(
                        frame_count,
                        present_us,
                        is_async,
                        "render: frame presented"
                    );

                    // ── Periodic diagnostic summary (every 2s) ──
                    if diag_last_report_ns == 0 {
                        diag_last_report_ns = submit_ns;
                    }
                    let diag_elapsed_ms = (submit_ns - diag_last_report_ns) / 1_000_000;
                    if diag_elapsed_ms >= 2000 && diag_frame_count > 0 {
                        let avg_recv_to_present = diag_sum_recv_to_present_us / diag_frame_count;
                        let avg_present_to_flip = diag_sum_present_to_flip_us / diag_frame_count;
                        let avg_flip_interval = diag_sum_flip_interval_us / diag_frame_count;
                        info!(
                            frames = diag_frame_count,
                            avg_recv_to_present_us = avg_recv_to_present,
                            avg_present_to_flip_us = avg_present_to_flip,
                            avg_flip_interval_us = avg_flip_interval,
                            acquire_deferred = acquire_not_ready_count,
                            "render_diag: periodic summary"
                        );
                        diag_frame_count = 0;
                        diag_sum_recv_to_present_us = 0;
                        diag_sum_present_to_flip_us = 0;
                        diag_sum_flip_interval_us = 0;
                        acquire_not_ready_count = 0;
                        diag_last_report_ns = submit_ns;
                    }

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
        } else if config.vrr
            && startup_done
            && !flip_pending
            && last_present_time.elapsed()
                > std::time::Duration::from_millis(VRR_KEEPALIVE_MS)
        {
            // VRR idle keepalive: no new client content, but the display's
            // VRR panel will lower its refresh rate if we stop sending
            // flips. Re-present the last GBM output buffer (no GPU blit
            // needed — the pixels are still in VRAM) to keep the panel at
            // its active refresh rate. This eliminates the ~180ms VRR
            // wake-up penalty that causes the idle→resume stutter.
            if let Some(idx) = last_blit_output_idx
                && let Some(Some(fb)) = output_fb_cache.get(idx)
            {
                match drm.present(fb) {
                    Ok(backend::FlipResult::Queued) => {
                        flip_pending = true;
                        presents_in_flight.fetch_add(1, Ordering::Release);
                        last_present_time = std::time::Instant::now();
                        trace!("VRR keepalive: re-presented last frame");
                    }
                    Ok(_) => {
                        last_present_time = std::time::Instant::now();
                    }
                    Err(e) => {
                        trace!(?e, "VRR keepalive: re-present failed (harmless)");
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
/// Returns `(is_async, output_buffer_index)` where `is_async` is true if a
/// page flip is pending, and `output_buffer_index` is the GBM output buffer
/// used (blit path) or `None` (SHM).
fn present_committed_frame(
    drm: &mut backend::drm::DrmBackend,
    frame: wayland::protocols::CommittedFrame,
    blitter: &mut backend::gpu::vulkan_blitter::VulkanBlitter,
    output_fb_cache: &mut [Option<backend::Framebuffer>],
) -> anyhow::Result<(bool, Option<usize>)> {
    use std::os::unix::io::AsFd;

    let has_overlays = frame.overlay.is_some() || frame.external_overlay.is_some();

    match frame.app {
        wayland::protocols::CommittedBuffer::DmaBuf {
            ref planes,
            width,
            height,
            format,
            modifier,
            ..
        } => {
            // --- GPU composition path (always blit) ---
            //
            // Direct scanout is intentionally disabled. While it saves
            // ~0.5ms of GPU blit time per frame, it makes the client's
            // DMA-BUF the display scanout buffer. This means:
            //   1. Release points can't be signaled until the NEXT
            //      pageflip, creating pipeline-dry deadlocks when
            //      frame flow stops.
            //   2. Cold GPU contexts (5s+ NVIDIA fence delays after
            //      app transitions) cascade into infinite stalls
            //      because each frame waits for the previous release.
            //   3. Buffer management with held_buffers becomes fragile.
            //
            // With blit, the Vulkan blitter copies the client buffer
            // to a GBM output buffer. The client buffer is released
            // immediately after the blit, breaking the dependency chain.

            // --- GPU composition path ---
            let first_plane = &planes[0];
            let exported = if has_overlays {
                // Build overlay layer descriptors for blit_composite().
                let mut overlay_layers = Vec::new();
                collect_overlay_layer(&frame.overlay, frame.overlay_opacity, &mut overlay_layers);
                collect_overlay_layer(
                    &frame.external_overlay,
                    frame.external_overlay_opacity,
                    &mut overlay_layers,
                );

                blitter
                    .blit_composite(
                        first_plane.fd.as_fd(),
                        width,
                        height,
                        format,
                        modifier,
                        first_plane.offset,
                        first_plane.stride,
                        &overlay_layers,
                    )
                    .context("Vulkan blit_composite failed")?
            } else {
                blitter
                    .blit(
                        first_plane.fd.as_fd(),
                        width,
                        height,
                        format,
                        modifier,
                        first_plane.offset,
                        first_plane.stride,
                    )
                    .context("Vulkan blit failed")?
            };

            // Use the pre-created DRM framebuffer for this output image.
            // FBs were created at startup,
            // so we just look up by buffer index.
            let out_fb = output_fb_cache[exported.buffer_index]
                .as_ref()
                .expect("output FB must be pre-created at startup");

            debug!(
                buffer_index = exported.buffer_index,
                fb = ?out_fb.handle,
                has_overlays,
                "present: using DRM framebuffer"
            );

            let buf_idx = exported.buffer_index;

            match drm.present(out_fb)? {
                backend::FlipResult::Queued => Ok((true, Some(buf_idx))),
                backend::FlipResult::DirectScanout => Ok((false, Some(buf_idx))),
                backend::FlipResult::Failed(e) => Err(e.context("blitted frame flip failed")),
            }
        }
        wayland::protocols::CommittedBuffer::Shm { .. } => {
            // TODO: SHM buffers require GPU-side blit (Vulkan compositor).
            trace!("skipping SHM buffer on DRM path (not yet supported)");
            Ok((false, None))
        }
    }
}

/// Extract an overlay `CommittedBuffer` into an `OverlayLayer` descriptor.
#[inline(always)]
fn collect_overlay_layer<'a>(
    buffer: &'a Option<wayland::protocols::CommittedBuffer>,
    opacity: f32,
    layers: &mut Vec<backend::gpu::vulkan_blitter::OverlayLayer<'a>>,
) {
    use std::os::unix::io::AsFd;

    if let Some(wayland::protocols::CommittedBuffer::DmaBuf {
        planes,
        width,
        height,
        format,
        modifier,
        ..
    }) = buffer
    {
        let first = &planes[0];
        layers.push(backend::gpu::vulkan_blitter::OverlayLayer {
            fd: first.fd.as_fd(),
            width: *width,
            height: *height,
            format: *format,
            modifier: *modifier,
            offset: first.offset,
            stride: first.stride,
            opacity,
        });
    }
}

/// Check an overlay buffer's explicit sync acquire fence (non-blocking).
///
/// Matching gamescope's approach: overlays are never blocking dependencies.
/// If the fence is ready, great. If not, proceed anyway — the GPU may
/// read partially-rendered data, but the overlay is alpha-blended on top
/// and any glitch is brief and nearly invisible. This prevents the
/// overlay's slow acquire fence from blocking the game pipeline.
fn check_overlay_acquire(
    buffer: &Option<wayland::protocols::CommittedBuffer>,
    device: &wayland::protocols::drm_syncobj::SyncobjDevice,
) {
    if let Some(wayland::protocols::CommittedBuffer::DmaBuf {
        acquire_point: Some(ap),
        ..
    }) = buffer
        && !device.is_acquire_ready(ap.handle(), ap.point)
    {
        debug!(
            point = ap.point,
            "overlay acquire fence not ready, proceeding anyway (non-blocking)"
        );
    }
}

/// Take the release point from an overlay buffer (if present).
#[inline(always)]
fn take_overlay_release(
    buffer: &mut Option<wayland::protocols::CommittedBuffer>,
) -> Option<wayland::protocols::drm_syncobj::SyncPoint> {
    if let Some(wayland::protocols::CommittedBuffer::DmaBuf {
        release_point, ..
    }) = buffer
    {
        release_point.take()
    } else {
        None
    }
}
