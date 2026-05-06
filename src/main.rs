//! Gamecomp — high-performance single-app fullscreen Wayland compositor.
//!
//! Entrypoint for the compositor binary. Handles:
//! 1. CLI argument parsing and configuration
//! 2. Logging initialization (tracing)
//! 3. Backend selection and initialization
//! 4. Thread spawning (render thread, XWM thread)
//! 5. Main event loop (calloop) orchestration

// Allow dead code during early development — many types and traits are
// defined ahead of their integration. Remove once the pipeline is fully wired.
#![allow(dead_code)]

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod backend;
mod compositor;
mod config;
mod focus;
mod frame_pacer;
mod input;
mod render;
mod render_thread;
mod retry;
mod stats;
#[cfg(test)]
mod test_harness;
mod vblank_timer;
mod wayland;
mod xwayland_mgr;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::thread;

use anyhow::Context;
use tracing::{debug, error, info, trace, warn};
use tracing_subscriber::EnvFilter;

use crate::compositor::scene::FrameInfo;
use crate::config::Config;
use crate::focus::{FocusArbiter, FocusResult, ServerFocusState};
use crate::frame_pacer::{FpsLimiter, FramePacer};
use crate::input::InputHandler;
use crate::input::keyboard::{KeyAction, KeyboardMonitor};
use crate::input::pointer::PointerMonitor;
use crate::retry::{RetryPolicy, retry_with_backoff};
use crate::stats::StatsTracker;
use crate::wayland::WaylandServer;
use crate::xwayland_mgr::XWaylandInstance;

/// Global shutdown flag. Set by signal handlers or error paths.
/// Ordering: Relaxed is sufficient — all threads poll this periodically.
static RUNNING: AtomicBool = AtomicBool::new(true);

fn main() {
    // Parse config first (before logging, since it sets the log level).
    let config = Config::from_args(std::env::args());

    // Initialize tracing.
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .compact()
        .init();

    info!(version = env!("CARGO_PKG_VERSION"), "gamecomp starting");
    info!(?config, "configuration");

    if let Err(e) = run(config) {
        error!(?e, "fatal error");
        std::process::exit(1);
    }

    info!("gamecomp exited cleanly");
}

/// Main compositor loop.
fn run(config: Config) -> anyhow::Result<()> {
    // Install signal handler for graceful shutdown.
    install_signal_handlers()?;

    // --- Initialize backend ---
    let refresh_hz = config.refresh_rate.unwrap_or(60);

    // --- Initialize frame pacer ---
    let mut pacer = FramePacer::new(refresh_hz);
    pacer.set_red_zone(config.red_zone_us * 1000);
    pacer.set_vrr(config.vrr);

    // --- Initialize FPS limiter ---
    // Determines when to release frame callbacks to clients.
    // 0 = match display refresh (no explicit cap beyond VSync).
    let target_fps = if config.fps_limit > 0 {
        config.fps_limit
    } else {
        refresh_hz
    };
    let mut fps_limiter = FpsLimiter::new(target_fps, refresh_hz);
    fps_limiter.set_vrr(config.vrr);
    info!(
        target_fps,
        refresh_hz,
        explicit_limit = config.fps_limit,
        "FPS limiter configured"
    );

    // --- Initialize VBlank timer ---
    // Free-running timer at the display refresh rate. On the DRM path,
    // the real refresh rate is detected once the render thread reads the
    // connector mode — DON'T arm at the fallback 60Hz to avoid sending
    // clients a wrong vsync cadence. The timer will be armed by
    // update_refresh_rate() once the real rate is known.
    let mut vblank_timer = vblank_timer::VBlankTimer::new();
    if config.refresh_rate.is_some() {
        // Explicit refresh rate: arm immediately.
        if let Some(ref mut timer) = vblank_timer {
            timer.arm(refresh_hz);
        }
    }

    // --- Initialize input handler ---
    let _input_handler = InputHandler::new().context("failed to initialize input handler")?;

    // --- Initialize stats tracker ---
    let _stats = StatsTracker::new(config.stats_pipe.clone());

    // --- Save host WAYLAND_DISPLAY ---
    // The wayland backend needs to connect to the *host* compositor. We must
    // capture the original WAYLAND_DISPLAY before overwriting it with our own.
    let host_wayland_display = std::env::var("WAYLAND_DISPLAY").ok();

    // Immediately remove WAYLAND_DISPLAY from the process env so that no
    // library (e.g., NVIDIA Vulkan driver) accidentally opens a second
    // connection to the host compositor. The wayland backend receives the
    // host display via its config; child processes receive their own socket
    // via Command::env().
    //
    // SAFETY: No other threads exist yet (called at the start of main before
    // spawning any threads), so modifying the environment is safe.
    unsafe {
        std::env::remove_var("WAYLAND_DISPLAY");
        std::env::remove_var("DISPLAY");
    }

    // --- Initialize Wayland server ---
    // Game resolution (-w×-h): what clients render at.
    // Output resolution (-W×-H): the physical display or nested window size.
    // If game resolution is unset, it falls back to the output resolution.
    let (output_w, output_h) = config.resolution.unwrap_or((1280, 720));
    let (game_w, game_h) = config.game_resolution.unwrap_or((output_w, output_h));
    let mut wayland_server = WaylandServer::new(Vec::new(), game_w, game_h)
        .context("failed to initialize Wayland server")?;
    let mut wayland_state = wayland::WaylandState::new(Vec::new(), game_w, game_h);
    wayland_state.drm_mode = matches!(config.backend, crate::config::BackendKind::Drm);

    // Shared host DMA-BUF format→modifier map. Written by the wayland
    // backend's event thread during its initial roundtrip, read by the
    // client-facing dmabuf module to advertise formats that enable zero-copy.
    let host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>> =
        Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    wayland_state.host_dmabuf_formats = host_dmabuf_formats.clone();

    // Create frame channel: main thread → wayland backend (committed buffers).
    let (frame_tx, frame_rx) = std::sync::mpsc::channel::<wayland::protocols::CommittedFrame>();
    wayland_state.frame_channel = Some(frame_tx);

    // Create cursor channel: main thread → wayland backend (cursor images).
    let (cursor_tx, cursor_rx) =
        std::sync::mpsc::channel::<crate::backend::wayland::CursorUpdate>();
    wayland_state.cursor_tx = Some(cursor_tx);

    let socket_name = wayland_server.socket_name().to_string();
    info!(socket = %socket_name, "Wayland socket ready");

    // --- Spawn render thread FIRST ---
    // The render thread starts the wayland backend event loop, which connects
    // to the host compositor and collects DMA-BUF format/modifier information
    // during its initial roundtrip. We must spawn it before XWayland so that
    // by the time XWayland connects and binds zwp_linux_dmabuf_v1, our
    // client-facing dmabuf module can advertise the real host formats instead
    // of the hardcoded fallback list. This enables zero-copy DMA-BUF forwarding.
    let (calloop_frame_tx, _frame_rx) = calloop::channel::channel::<FrameInfo>();
    let (vblank_tx, vblank_rx) = std::sync::mpsc::channel::<u64>(); // VBlank timestamp (ns).

    // Shared atomic: number of frames sent to render thread but not yet
    // page-flipped (or host-committed). The main loop increments this on
    // forward; the render thread decrements on page flip completion.
    // This creates gamescope-style backpressure: frame callbacks are only
    // fired when presents_in_flight == 0 (display consumed previous frame).
    let presents_in_flight = Arc::new(AtomicU32::new(0));
    let presents_in_flight_render = presents_in_flight.clone();

    // Shared atomic for detected host display refresh rate (millihertz).
    // Written by the wayland backend's event thread, read by the main loop.
    // Ordering: Release on write, Relaxed on read — main loop polls periodically.
    let detected_refresh_mhz = Arc::new(AtomicU32::new(0));
    let detected_refresh_mhz_render = detected_refresh_mhz.clone();

    // Shared atomics for host window physical size. Written by the render
    // thread when xdg_toplevel configure events arrive, read by the main
    // loop to advertise output resolution to clients.
    let host_physical_width = Arc::new(AtomicU32::new(0));
    let host_physical_height = Arc::new(AtomicU32::new(0));
    let host_physical_width_render = host_physical_width.clone();
    let host_physical_height_render = host_physical_height.clone();

    // Channel for host input events (nested mode only). The render thread
    // forwards keyboard/pointer events from the host compositor to the main
    // thread so they can be sent to Wayland clients.
    let (host_input_tx, host_input_rx) =
        std::sync::mpsc::channel::<crate::backend::wayland::WaylandEvent>();

    // Shared atomics for cursor position. Written by the main thread after
    // pointer motion events, read by the render thread to position the
    // hardware cursor plane. Ordering: Relaxed — render thread polls each
    // iteration, no ordering guarantee needed.
    let cursor_x = Arc::new(AtomicI32::new(0));
    let cursor_y = Arc::new(AtomicI32::new(0));
    let cursor_x_render = cursor_x.clone();
    let cursor_y_render = cursor_y.clone();

    // --- DRM path: open session and GPU device on main thread ---
    // Session management and device discovery must happen before the render
    // thread is spawned so the DRM fd can be transferred.
    let mut session: Option<backend::session::Session> = None;
    let drm_device: Option<(std::path::PathBuf, std::os::unix::io::OwnedFd)> =
        if matches!(config.backend, crate::config::BackendKind::Drm) {
            let mut sess =
                backend::session::Session::open().context("failed to open seat session")?;
            let gpus = backend::gpu_discovery::discover_gpus(sess.seat_name())
                .context("GPU discovery failed")?;
            let gpu = backend::gpu_discovery::select_primary_gpu(&gpus)
                .ok_or_else(|| anyhow::anyhow!("no usable GPU found"))?;
            let path = gpu.dev_path.clone();
            let fd = sess
                .open_device(&path)
                .context("failed to open GPU device via session")?;
            info!(path = %path.display(), "DRM device opened via session");
            session = Some(sess);
            Some((path, fd))
        } else {
            None
        };

    // --- Keyboard monitor for VT switching + input forwarding (DRM path only) ---
    let mut keyboard_monitor: Option<KeyboardMonitor> = None;
    let mut pointer_monitor: Option<PointerMonitor> = None;
    if let Some(ref mut sess) = session {
        let mut kbd = KeyboardMonitor::new();
        kbd.open_from_session(sess);
        keyboard_monitor = Some(kbd);

        match PointerMonitor::new() {
            Ok(ptr) => pointer_monitor = Some(ptr),
            Err(e) => warn!(?e, "failed to create libinput pointer monitor"),
        }
    }

    // Track session active→inactive→active transitions so we can
    // re-open keyboard devices after VT switch restore. Logind revokes
    // evdev fds via EVIOCREVOKE when the session goes inactive, making
    // the old fds permanently dead.
    let mut session_was_active = true;

    let session_active_flag: Option<Arc<AtomicBool>> = session.as_ref().map(|s| s.active_flag());

    let config_clone = config.clone();
    let host_display_clone = host_wayland_display.clone();
    let host_dmabuf_formats_render = host_dmabuf_formats.clone();

    // --- Create DRM syncobj device for explicit sync ---
    // Opens a render node fd for DRM syncobj operations (timeline import,
    // signal, and eventfd). Shared between main thread (protocol dispatch)
    // and render thread (release signaling on pageflip).
    let syncobj_device: Option<wayland::protocols::SyncobjDevice> = if matches!(
        config.backend,
        crate::config::BackendKind::Drm
    ) {
        let render_node_path = wayland::protocols::wl_drm::find_render_node();
        match std::fs::File::open(&render_node_path) {
            Ok(file) => {
                let syncobj_fd: std::os::unix::io::OwnedFd = file.into();
                let device = wayland::protocols::SyncobjDevice::new(syncobj_fd);
                if device.supports_eventfd() {
                    info!("DRM syncobj_eventfd supported, explicit sync available");
                    Some(device)
                } else {
                    info!("DRM syncobj_eventfd not supported, explicit sync disabled");
                    None
                }
            }
            Err(e) => {
                warn!(?e, path = %render_node_path.display(), "failed to open render node for syncobj");
                None
            }
        }
    } else {
        None
    };
    let syncobj_device_render = syncobj_device.clone();

    let render_thread = thread::Builder::new()
        .name("gamecomp-render".to_string())
        .spawn(move || {
            render_thread::render_thread_main(
                &config_clone,
                host_display_clone,
                frame_rx,
                cursor_rx,
                detected_refresh_mhz_render,
                host_dmabuf_formats_render,
                drm_device,
                vblank_tx,
                session_active_flag,
                host_physical_width_render,
                host_physical_height_render,
                host_input_tx,
                cursor_x_render,
                cursor_y_render,
                presents_in_flight_render,
                syncobj_device_render,
            );
        })
        .context("failed to spawn render thread")?;

    // --- Wait for host DMA-BUF formats ---
    // Block until the wayland backend's event thread has completed its host
    // roundtrips and published the host's DMA-BUF format/modifier pairs.
    // XWayland (and all subsequent clients) will then be advertised the real
    // host formats, enabling zero-copy DMA-BUF forwarding.
    if matches!(config.backend, crate::config::BackendKind::Wayland) {
        xwayland_mgr::wait_for_host_formats(&host_dmabuf_formats);
    }

    // Wait for the render thread to publish the display resolution before
    // launching XWayland. On the Wayland path this comes from the host
    // window configure; on DRM it comes from the connector mode.
    // Resolution is applied to clients via propagate_host_resolution()
    // in the main loop — we only need the synchronization barrier here.
    xwayland_mgr::wait_for_host_configure(&host_physical_width, &host_physical_height);

    // Initialize cursor position to center of screen (like gamescope).
    let pw = host_physical_width.load(Ordering::Acquire);
    let ph = host_physical_height.load(Ordering::Acquire);
    cursor_x.store((pw / 2) as i32, Ordering::Relaxed);
    cursor_y.store((ph / 2) as i32, Ordering::Relaxed);

    // --- Register DRM syncobj global (explicit sync for XWayland) ---
    // Must happen after the DRM device is opened but before XWayland
    // starts, so XWayland sees the global during its initial roundtrip.
    if let Some(ref device) = syncobj_device {
        wayland_state.syncobj_device = Some(device.clone());
        wayland_server.register_syncobj_global(device.clone());
        info!("DRM explicit sync (syncobj) global registered");
    }

    // Track the last-seen detected refresh rate to avoid redundant updates.
    let mut last_detected_hz: u32 = 0;

    // Apply detected refresh rate to wl_output, frame pacer, and FPS limiter
    // BEFORE launching XWayland. The render thread has already published
    // detected_refresh_mhz by the time wait_for_host_configure returns
    // (same init code path). Without this, XWayland binds wl_output at 60Hz
    // and Flutter targets the wrong vsync interval.
    update_refresh_rate(
        &detected_refresh_mhz,
        &mut last_detected_hz,
        &config,
        &mut pacer,
        &mut fps_limiter,
        &mut wayland_state,
        &mut vblank_timer,
    );

    // --- Launch XWayland servers ---
    // Spawn `xwayland_count` instances. Server 0 is the platform display
    // (Steam client, etc.) and gets the full output resolution. Servers 1+
    // are game displays and get the game resolution.
    //
    // Following gamescope's convention:
    //   DISPLAY        = server 0 (platform)
    //   STEAM_GAME_DISPLAY_0 = server 1 (first game)
    //   STEAM_GAME_DISPLAY_1 = server 2 (second game)
    //   ...etc.
    let xwayland_count = config.xwayland_count.max(1);

    let (xwm_event_tx, xwm_event_rx) = calloop::channel::channel::<wayland::xwayland::XwmEvent>();

    // Global "winning" focus state — the main loop aggregates per-server
    // atomics and writes the winner here for the commit handler to read.
    let focused_app_id = Arc::new(AtomicU32::new(0));
    let focused_wl_surface_id = Arc::new(AtomicU32::new(0));
    let focused_server_index = Arc::new(AtomicU32::new(u32::MAX));

    // Wire focused surface into wayland state so the commit handler
    // can gate presentation per-surface.
    wayland_state.focused_wl_surface_id = Arc::clone(&focused_wl_surface_id);
    wayland_state.focused_server_index = Arc::clone(&focused_server_index);

    let mut xwayland_servers: Vec<XWaylandInstance> = Vec::with_capacity(xwayland_count as usize);

    for server_idx in 0..xwayland_count {
        let allocated: Vec<String> = xwayland_servers.iter().map(|s| s.display.clone()).collect();
        let display_str = xwayland_mgr::find_free_x11_display(&allocated)?;

        let child = xwayland_mgr::spawn_xwayland(
            &display_str,
            &socket_name,
            &mut wayland_server,
            &mut wayland_state,
            server_idx,
        )?;

        // Server 0 gets the full output resolution (platform client).
        // Servers 1+ get the game resolution.
        let (srv_w, srv_h) = if xwayland_count > 1 && server_idx == 0 {
            (output_w, output_h)
        } else {
            // Use host physical dims if available, otherwise game resolution.
            let pw = host_physical_width.load(Ordering::Acquire);
            let ph = host_physical_height.load(Ordering::Acquire);
            if pw > 0 && ph > 0 {
                (pw, ph)
            } else {
                (game_w, game_h)
            }
        };

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<wayland::xwayland::XwmCommand>();
        let evt_tx = xwm_event_tx.clone();
        let xwm_display = display_str.clone();
        let srv_focused_app = Arc::new(AtomicU32::new(0));
        let srv_focused_surface = Arc::new(AtomicU32::new(0));
        let xwm_focused_app = Arc::clone(&srv_focused_app);
        let xwm_focused_surface = Arc::clone(&srv_focused_surface);
        let xwm_steam_mode = config.steam_mode;
        let thread = thread::Builder::new()
            .name(format!("gamecomp-xwm-{server_idx}"))
            .spawn(move || {
                let result = retry_with_backoff("XWM", &RetryPolicy::DEFAULT, &RUNNING, || {
                    wayland::xwayland::run_xwm(
                        &xwm_display,
                        &evt_tx,
                        &cmd_rx,
                        server_idx,
                        &xwm_focused_app,
                        &xwm_focused_surface,
                        srv_w,
                        srv_h,
                        xwm_steam_mode,
                    )
                });
                if let Err(e) = result {
                    error!(
                        server_idx,
                        ?e,
                        "XWM thread exiting after exhausting retries"
                    );
                }
            })
            .context("failed to spawn XWM thread")?;

        info!(
            server_idx,
            display = %display_str,
            width = srv_w,
            height = srv_h,
            "XWayland server ready"
        );

        xwayland_servers.push(XWaylandInstance {
            display: display_str,
            child,
            cmd_tx,
            thread,
            index: server_idx,
            focused_app_id: srv_focused_app,
            focused_wl_surface_id: srv_focused_surface,
            respawn_failures: 0,
            permanently_failed: false,
        });
    }

    // --- Launch child command ---
    let mut child_process =
        launch_child_command(&config, &xwayland_servers, xwayland_count, &socket_name)?;

    // --- Main event loop ---
    info!("entering main event loop");

    let wayland_fd = wayland_server.display_fd();

    // Track last-propagated resolution to avoid redundant SetResolution
    // commands and update_output_resolution calls every loop iteration.
    let mut last_propagated_w: u32 = 0;
    let mut last_propagated_h: u32 = 0;

    // Cross-server focus arbiter — picks the global winning server
    // and gates commits so only the focused surface gets frame callbacks.
    // +1 for the native Wayland virtual server slot.
    let mut focus_arbiter = FocusArbiter::new(xwayland_count as usize + 1);

    // Native Wayland focus state — participates in arbitration alongside
    // XWayland servers so native clients (e.g., Grid) can win focus.
    let native_focus = focus::ServerFocusState {
        index: u32::MAX,
        focused_app_id: Arc::clone(&wayland_state.native_focused_app_id),
        focused_wl_surface_id: Arc::clone(&wayland_state.native_focused_surface_id),
    };

    // Presentation focus — tracks which server's content is actually being
    // shown. Separate from the logical focus (commit gate atomics) so that
    // the old content stays visible until the new target commits a frame.
    let mut presentation_server_index: u32 = u32::MAX;
    let mut presentation_app_id: u32 = 0;
    let mut loop_start_ns: u64 = 0;

    // VT resume grace period: number of vblank ticks remaining where we
    // repeatedly send xdg configure events to keep the client rendering.
    // After VT switch, Flutter's internal frame clock can go dormant; a
    // single configure event triggers one commit but the client may not
    // sustain continuous rendering on its own. Repeated configures ensure
    // the commit→callback→commit cycle stays alive until the client's own
    // scheduler catches up. 120 ticks @ 60 Hz ≈ 2 seconds.
    const EXPOSE_GRACE_TICKS: u32 = 120;
    let mut expose_grace_remaining: u32 = 0;

    // Idle keepalive: track the last time a commit was received. When
    // the focused client goes idle (no commits for >1s), continuously
    // send Expose events on each vblank to nudge Flutter into rendering.
    // This combats Flutter's internal optimization of skipping frames
    // when the widget tree is unchanged.
    let mut last_commit_ns: u64 = 0;
    const IDLE_KEEPALIVE_NS: u64 = 1_000_000_000; // 1 second

    // VT resume watchdog: if the focused server doesn't produce a frame
    // within this window after VT resume, force focus to server 0 (Grid)
    // so the UI remains interactive. NVIDIA GPU contexts can die after
    // long VT switches (>10s), leaving the focused game server permanently
    // unresponsive while the commit gate blocks all other servers.
    const VT_RESUME_WATCHDOG: std::time::Duration = std::time::Duration::from_secs(3);
    let mut vt_resume_at: Option<std::time::Instant> = None;
    let mut vt_fallback_active = false;
    // Periodic stats counters for summary logging.
    let mut stats_last_ns: u64 = 0;
    let mut stats_commits: u64 = 0;
    let mut stats_vblank_ticks: u64 = 0;
    let mut stats_callbacks_fired: u64 = 0;
    let mut stats_vblank_callbacks_fired: u64 = 0;
    let mut stats_frames_forwarded: u64 = 0;
    let mut stats_dispatch_ns: u64 = 0;
    let mut stats_iterations: u64 = 0;
    let mut stats_presented: u64 = 0;
    let mut stats_feedbacks_queued: u64 = 0;

    // ── End-to-end pipeline diagnostic counters ──
    // Tracks: callback_fire → commit → forward → page_flip_complete
    let mut diag_sum_callback_to_commit_us: u64 = 0;
    let mut diag_sum_forward_to_pageflip_us: u64 = 0;
    let mut diag_pipeline_frames: u64 = 0;
    let mut diag_last_forward_ns: u64 = 0;
    let mut diag_pageflip_count: u64 = 0;
    let mut diag_sum_pageflip_interval_us: u64 = 0;
    let mut diag_last_pageflip_ns: u64 = 0;

    while RUNNING.load(Ordering::Relaxed) {
        let iter_start = wayland::monotonic_ns();
        stats_iterations += 1;
        if loop_start_ns > 0 {
            let full_iteration_us = (iter_start - loop_start_ns) / 1_000;
            if full_iteration_us > 10_000 {
                trace!(full_iteration_us, "main_loop: slow iteration (>10ms)");
            }
        }
        loop_start_ns = iter_start;

        // Periodic summary stats (every ~2 seconds)
        if stats_last_ns == 0 {
            stats_last_ns = iter_start;
        } else if iter_start - stats_last_ns > 2_000_000_000 {
            let elapsed_ms = (iter_start - stats_last_ns) / 1_000_000;
            let avg_dispatch_us = if stats_iterations > 0 {
                stats_dispatch_ns / stats_iterations / 1_000
            } else {
                0
            };
            let commits = wayland_state.commit_count - stats_commits;
            let feedbacks_queued = wayland_state.presentation_requests - stats_feedbacks_queued;
            let inflight = presents_in_flight.load(Ordering::Relaxed);
            let foc_id = focused_app_id.load(Ordering::Relaxed);
            let foc_surf = focused_wl_surface_id.load(Ordering::Relaxed);
            let foc_srv = focused_server_index.load(Ordering::Relaxed);
            let ov_id = wayland_state.overlay_wl_surface_id.load(Ordering::Relaxed);
            let idle_ms = if last_commit_ns > 0 {
                (iter_start.saturating_sub(last_commit_ns)) / 1_000_000
            } else {
                0
            };
            info!(
                elapsed_ms,
                commits,
                vblank_ticks = stats_vblank_ticks,
                callbacks_fired = stats_callbacks_fired,
                vblank_cb = stats_vblank_callbacks_fired,
                frames_forwarded = stats_frames_forwarded,
                iterations = stats_iterations,
                avg_dispatch_us,
                held = wayland_state.held_buffers.len(),
                deferred = wayland_state.deferred_callback_count(),
                presented = stats_presented,
                feedbacks_queued,
                staged_feedbacks = wayland_state.staged_feedbacks.len(),
                inflight_feedbacks = wayland_state.inflight_feedbacks.len(),
                inflight,
                foc_id,
                foc_surf,
                foc_srv,
                ov_id,
                has_staged = wayland_state.staged_buffer.is_some(),
                pending_cb = wayland_state.pending_callback_count(),
                idle_ms,
                expose_grace = expose_grace_remaining,
                "frame_stats: periodic summary"
            );
            // ── Pipeline latency diagnostics ──
            if diag_pipeline_frames > 0 {
                let avg_cb_to_commit = diag_sum_callback_to_commit_us / diag_pipeline_frames;
                let avg_fwd_to_flip = if diag_pageflip_count > 0 {
                    diag_sum_forward_to_pageflip_us / diag_pageflip_count
                } else {
                    0
                };
                let avg_flip_interval = if diag_pageflip_count > 1 {
                    diag_sum_pageflip_interval_us / (diag_pageflip_count - 1)
                } else {
                    0
                };
                info!(
                    pipeline_frames = diag_pipeline_frames,
                    avg_callback_to_commit_us = avg_cb_to_commit,
                    avg_forward_to_pageflip_us = avg_fwd_to_flip,
                    avg_pageflip_interval_us = avg_flip_interval,
                    pageflips = diag_pageflip_count,
                    "pipeline_diag: end-to-end latency"
                );
            }
            diag_sum_callback_to_commit_us = 0;
            diag_sum_forward_to_pageflip_us = 0;
            diag_pipeline_frames = 0;
            diag_pageflip_count = 0;
            diag_sum_pageflip_interval_us = 0;

            stats_last_ns = iter_start;
            stats_commits = wayland_state.commit_count;
            stats_vblank_ticks = 0;
            stats_callbacks_fired = 0;
            stats_vblank_callbacks_fired = 0;
            stats_frames_forwarded = 0;
            stats_dispatch_ns = 0;
            stats_iterations = 0;
            stats_presented = 0;
            stats_feedbacks_queued = wayland_state.presentation_requests;
        }

        let vt_just_resumed = dispatch_session(
            &mut session,
            &mut session_was_active,
            &mut keyboard_monitor,
            &mut pointer_monitor,
            &mut wayland_state,
        );
        if vt_just_resumed {
            expose_grace_remaining = EXPOSE_GRACE_TICKS;
            vt_resume_at = Some(std::time::Instant::now());
            vt_fallback_active = false;
        }
        poll_keyboard(&mut keyboard_monitor, &mut session, &mut wayland_state);
        poll_pointer(
            &mut pointer_monitor,
            &mut wayland_state,
            &cursor_x,
            &cursor_y,
            &host_physical_width,
            &host_physical_height,
            config.pointer_sensitivity,
        );
        drain_host_input(&host_input_rx, &mut wayland_state);

        // ── Gamescope-style VRR frame pipeline ────────────────────────
        //
        // Gamescope's VRR loop:
        //   1. Every iteration: `vblank = true` (VRR skips the timer)
        //   2. `flush_frame_done()` fires callbacks immediately
        //   3. `handle_done_commits_xwayland()` picks up newly committed
        //      frames
        //   4. `paint_all()` is gated on `PresentsInFlight == 0`
        //
        // We match this by:
        //   1. Draining page flip completions (clears presents_in_flight)
        //   2. Releasing ALL held buffers on page flip (not one-at-a-time)
        //   3. Forwarding staged frame when presents_in_flight == 0
        //   4. Firing callbacks IMMEDIATELY in forward_staged_frame()
        //
        // Callbacks fire on forward, NOT after the page flip. Flutter uses
        // its own vsync timer and ignores callback timing for scheduling.
        // The callback just tells Wayland "you may commit again." Holding
        // it back after the flip provides zero benefit (callback_to_commit
        // was always 0μs) and adds latency.

        // Drain page flip completions from the render thread.
        while let Ok(vblank_ns) = vblank_rx.try_recv() {
            pacer.mark_vblank(vblank_ns);

            // Diagnostic: page flip interval + forward-to-flip latency
            if diag_last_pageflip_ns > 0 {
                diag_sum_pageflip_interval_us += (vblank_ns - diag_last_pageflip_ns) / 1000;
            }
            diag_last_pageflip_ns = vblank_ns;
            diag_pageflip_count += 1;
            if diag_last_forward_ns > 0 {
                let forward_to_flip_us = vblank_ns.saturating_sub(diag_last_forward_ns) / 1000;
                diag_sum_forward_to_pageflip_us += forward_to_flip_us;
            }

            // Release ALL held buffers on page flip — blit already copied
            // pixels into the output buffer. This matches gamescope's
            // behavior where wlr_buffer_unlock runs after paint_all.
            //
            // NOTE: Do NOT release here in the DRM direct-scanout path!
            // With direct scanout the client's DMA-BUF IS the scanout
            // buffer — releasing it tells the client to reuse it while
            // the display is still scanning from it, causing flashing.
            // In the blit path, pixels are copied to a separate GBM
            // output buffer, so releasing the client buffer is safe.
            //
            // Buffer release is handled by fire_all_surface_callbacks()
            // at vblank time — synchronized with wl_callback.done delivery.
            // while wayland_state.release_one_buffer() {}

            // Fire wp_presentation_feedback.presented for any in-flight
            // feedbacks. This is critical: XWayland's X11 Present extension
            // uses these events to drive accurate vsync timing for X11
            // clients (GTK, Flutter). Without it, X11 clients fall back
            // to a software 60Hz timer.
            if !wayland_state.inflight_feedbacks.is_empty() {
                let refresh_ns = if last_detected_hz > 0 {
                    1_000_000_000u32 / last_detected_hz
                } else {
                    16_666_667 // 60Hz fallback
                };
                let seq = wayland_state.presentation_sequence.wrapping_add(1);
                wayland_state.presentation_sequence = seq;
                stats_presented += wayland_state.inflight_feedbacks.len() as u64;
                wayland::protocols::presentation::fire_presented(
                    &mut wayland_state,
                    vblank_ns,
                    refresh_ns,
                    seq,
                );
            }
        }

        // Consume VBlank timer ticks.
        // Fire deferred frame callbacks on EVERY vblank tick — this
        // provides XWayland with a steady MSC clock, matching gamescope's
        // behaviour of firing flush_frame_done on every paint_all iteration.
        // Without this, callbacks only fire when content is forwarded,
        // creating a circular timing dependency with the X11 client.
        if let Some(ref timer) = vblank_timer {
            let ticks = timer.read_ticks();
            if ticks > 0 {
                stats_vblank_ticks += ticks;
                if wayland_state.has_surface_callbacks()
                    && wayland_state.fire_all_surface_callbacks()
                {
                    let now_ns = wayland::monotonic_ns();
                    wayland_state.last_callback_fire_ns = now_ns;
                    fps_limiter.mark_released(now_ns);
                    stats_callbacks_fired += 1;
                    stats_vblank_callbacks_fired += 1;
                    trace!(ticks, "vblank_timer: fired deferred callbacks");
                    // Flush immediately so XWayland receives the
                    // wl_callback.done event without waiting for the
                    // end-of-iteration flush. This minimises the
                    // round-trip time from callback to next commit.
                    wayland_server.flush();
                }

                // VT resume / focus-change grace: send X11 Expose events
                // on each vblank tick to keep the client's frame clock
                // alive. XWayland surfaces don't use xdg_shell, so xdg
                // configure events have no effect. Instead, Expose events
                // force the X11 client (Flutter) to repaint, sustaining
                // its vsync timer through the recovery window.
                if expose_grace_remaining > 0 {
                    expose_grace_remaining = expose_grace_remaining.saturating_sub(ticks as u32);
                    for srv in &xwayland_servers {
                        let _ = srv
                            .cmd_tx
                            .send(wayland::xwayland::XwmCommand::RefreshWindows);
                    }
                }

                // Idle keepalive: when the focused client hasn't committed
                // for >1s, send Expose events on every vblank to nudge
                // Flutter into rendering. Flutter stops painting when its
                // widget tree is unchanged; Expose events force a repaint.
                if last_commit_ns > 0
                    && iter_start.saturating_sub(last_commit_ns) > IDLE_KEEPALIVE_NS
                    && expose_grace_remaining == 0
                {
                    for srv in &xwayland_servers {
                        let _ = srv
                            .cmd_tx
                            .send(wayland::xwayland::XwmCommand::RefreshWindows);
                    }
                }

                // Overlay keepalive: while any overlay is active, send
                // Expose events to the overlay server (server 0) on
                // every vblank. This is independent of the idle keepalive
                // above, which tracks game commits and never fires while
                // a game is running at normal frame rates.
                //
                // Without this, Flutter's GDK frame clock goes idle when
                // the widget tree is clean. No eglSwapBuffers → no
                // PresentPixmap → no wl_surface.commit() → dead cycle.
                // Gamescope avoids this because its main loop fires
                // callbacks for ALL windows unconditionally; gamecomp's
                // commit gate prevents server 0 from receiving stimuli.
                if expose_grace_remaining == 0 {
                    let has_overlay = wayland_state.overlay_wl_surface_id.load(Ordering::Relaxed)
                        != 0
                        || wayland_state
                            .external_overlay_wl_surface_id
                            .load(Ordering::Relaxed)
                            != 0;
                    if has_overlay && let Some(srv) = xwayland_servers.first() {
                        let _ = srv
                            .cmd_tx
                            .send(wayland::xwayland::XwmCommand::RefreshWindows);
                    }
                }

                // Do NOT release held buffers here. Buffers must stay
                // held until page flip completion (vblank_rx path above).
                // Releasing on vblank ticks caused the texture flashing
                // bug: the client recycled the buffer for new rendering
                // while the GPU was still scanning out from it.
            }
        }

        update_refresh_rate(
            &detected_refresh_mhz,
            &mut last_detected_hz,
            &config,
            &mut pacer,
            &mut fps_limiter,
            &mut wayland_state,
            &mut vblank_timer,
        );

        propagate_host_resolution(
            &host_physical_width,
            &host_physical_height,
            &mut wayland_state,
            &xwayland_servers,
            &mut last_propagated_w,
            &mut last_propagated_h,
        );

        // Monitor all XWayland instances and respawn crashed ones.
        for srv in &mut xwayland_servers {
            xwayland_mgr::monitor_xwayland(
                srv,
                &socket_name,
                &mut wayland_server,
                &mut wayland_state,
            );
        }

        // Accept new Wayland client connections.
        if let Some(stream) = wayland_server.accept()
            && let Err(e) = wayland_server.insert_client(stream, &mut wayland_state)
        {
            warn!(?e, "failed to insert Wayland client");
        }

        // Drain XWM events and run the 4-phase focus arbiter. The arbiter
        // picks the global winning server and signals focus changes.
        let focus_events = focus_arbiter.drain_events(&xwm_event_rx);

        // Propagate overlay surface IDs from server 0 (the platform/root
        // server). Overlays always come from the platform
        // server and are composited on top of whichever game has focus,
        // regardless of which server the game runs on.
        for evt in &focus_events {
            if let wayland::xwayland::XwmEvent::FocusChanged(srv_idx, focus_state) = evt
                && *srv_idx == 0
            {
                let prev_overlay = wayland_state.overlay_wl_surface_id.load(Ordering::Relaxed);
                let prev_ext_overlay = wayland_state
                    .external_overlay_wl_surface_id
                    .load(Ordering::Relaxed);

                update_overlay_atomics(&wayland_state, focus_state, 0);

                // When an overlay becomes active, the overlay client
                // (server 0 / Grid) has likely been idle since losing
                // app focus — no pending frame callbacks, not rendering.
                // Fire callbacks and send Expose events to wake it up
                // so it commits its first overlay frame.
                let overlay_activated = prev_overlay == 0 && focus_state.overlay_wl_surface_id != 0;
                let ext_overlay_activated =
                    prev_ext_overlay == 0 && focus_state.external_overlay_wl_surface_id != 0;
                if overlay_activated || ext_overlay_activated {
                    debug!("overlay activated, waking server 0");
                    wayland_state.fire_server_callbacks(0);
                    wayland_state.reconfigure_toplevels();
                    if let Some(srv) = xwayland_servers.first() {
                        let _ = srv
                            .cmd_tx
                            .send(wayland::xwayland::XwmCommand::RefreshWindows);
                    }
                    expose_grace_remaining = EXPOSE_GRACE_TICKS;
                }

                // When overlay deactivates, no special cleanup needed.
                // The vblank-deferred callback cycle handles itself.
            }
        }

        let mut focus_states: Vec<_> = xwayland_servers.iter().map(|s| s.focus_state()).collect();
        focus_states.push(ServerFocusState {
            index: native_focus.index,
            focused_app_id: Arc::clone(&native_focus.focused_app_id),
            focused_wl_surface_id: Arc::clone(&native_focus.focused_wl_surface_id),
        });
        let mut result = focus_arbiter.update(&focus_states);

        // ── VT resume watchdog ──────────────────────────────────────
        // After VT resume, if the focused server hasn't produced a frame
        // within VT_RESUME_WATCHDOG, its GPU context likely died (NVIDIA
        // power-manages idle contexts after ~10s). Force focus to server 0
        // (Grid) so the user has an interactive UI instead of a black screen.
        if let Some(resume_at) = vt_resume_at
            && resume_at.elapsed() >= VT_RESUME_WATCHDOG
        {
            warn!(
                elapsed_ms = resume_at.elapsed().as_millis() as u64,
                focused_server = result.server_index,
                "VT resume watchdog: no frames from focused server, falling back to server 0"
            );
            vt_resume_at = None;
            vt_fallback_active = true;
            // Force focus to server 0.
            if let Some(srv0) = xwayland_servers.first() {
                let state_0 = srv0.focus_state();
                result = FocusResult {
                    app_id: state_0.focused_app_id.load(Ordering::Relaxed),
                    surface_id: state_0.focused_wl_surface_id.load(Ordering::Relaxed),
                    server_index: 0,
                    changed: true,
                };
            }
            wayland_state.fire_all_callbacks();
            for srv in &xwayland_servers {
                let _ = srv
                    .cmd_tx
                    .send(wayland::xwayland::XwmCommand::RefreshWindows);
            }
        }
        // While VT fallback is active, keep overriding focus to server 0
        // on every tick. The arbiter's normal logic would switch back to
        // the dead server because baselayer_app_ids still points to it.
        if vt_fallback_active
            && result.server_index != 0
            && let Some(srv0) = xwayland_servers.first()
        {
            let state_0 = srv0.focus_state();
            result = FocusResult {
                app_id: state_0.focused_app_id.load(Ordering::Relaxed),
                surface_id: state_0.focused_wl_surface_id.load(Ordering::Relaxed),
                server_index: 0,
                changed: false,
            };
        }

        // Update commit gate atomics immediately so the new target's
        // commits are accepted. The old target's commits get rejected
        // (and their buffers released).
        focused_app_id.store(result.app_id, Ordering::Relaxed);
        focused_wl_surface_id.store(result.surface_id, Ordering::Relaxed);
        focused_server_index.store(result.server_index, Ordering::Relaxed);

        if result.changed {
            info!(
                app_id = result.app_id,
                surface_id = result.surface_id,
                server_index = result.server_index,
                "logical focus changed"
            );

            if result.surface_id != 0 {
                let pending = wayland_state.pending_callback_count();
                let deferred = wayland_state.deferred_callback_count();
                let held = wayland_state.held_buffers.len();
                trace!(pending, deferred, held, "focus changed: waking new client");
                wayland_state.fire_all_callbacks();
                // Release ALL held wl_buffers from the previous focus
                // owner so we don't leak FDs.
                wayland_state.release_all_buffers();
                // Do NOT clear staged_buffer — the old client's last
                // frame stays visible until the new client commits.

                // Re-configure toplevels so the newly-focused client
                // receives a configure event and commits, restarting
                // its frame callback cycle.
                wayland_state.reconfigure_toplevels();

                // Send X11 Expose events to ALL XWayland servers.
                // When focus switches between servers, the new client
                // likely has no pending frame callbacks (it was not
                // rendering while unfocused), so fire_all_callbacks
                // above has no effect. Expose events force the X11
                // client (Flutter) to repaint, bootstrapping its
                // frame callback cycle. Without this, the client
                // stalls for seconds until some unrelated event
                // (e.g., cursor update) happens to trigger a render.
                for srv in &xwayland_servers {
                    let _ = srv
                        .cmd_tx
                        .send(wayland::xwayland::XwmCommand::RefreshWindows);
                }

                // Start an Expose grace period so the newly focused
                // client receives repeated Expose events on each
                // vblank tick. A single Expose may not be enough for
                // Flutter to fully restart its rendering pipeline after
                // being idle for a long time.
                expose_grace_remaining = EXPOSE_GRACE_TICKS;
            }
        }

        // Dispatch Wayland requests BEFORE forwarding so newly-arrived
        // commits are processed in the same iteration. Without this the
        // staged_buffer is always one iteration stale, adding ~1 poll
        // cycle of latency per frame and breaking continuous render.
        let dispatch_before = wayland::monotonic_ns();
        if let Err(e) = wayland_server.dispatch(&mut wayland_state) {
            warn!(?e, "Wayland dispatch error");
        }
        stats_dispatch_ns += wayland::monotonic_ns() - dispatch_before;

        // Forward staged buffer — gamescope-style VRR gating.
        // Only forward when PresentsInFlight == 0, matching gamescope's
        // `bShouldPaint = false` when `PresentsInFlight != 0`.
        // This ensures only ONE frame is in the display pipeline at a time,
        // giving clean VRR timing without stacking page flips.
        let has_focus = focused_app_id.load(Ordering::Relaxed) != 0
            || wayland_state.overlay_wl_surface_id.load(Ordering::Relaxed) != 0
            || wayland_state
                .external_overlay_wl_surface_id
                .load(Ordering::Relaxed)
                != 0;
        let can_present = presents_in_flight.load(Ordering::Acquire) == 0;
        if wayland_state.staged_buffer.is_some() && can_present {
            stats_frames_forwarded += 1;
            last_commit_ns = wayland_state.staged_at_ns;
            // VT watchdog satisfied — a frame arrived, clear the timer.
            if vt_resume_at.is_some() {
                vt_resume_at = None;
            }
            if vt_fallback_active {
                info!("VT fallback cleared: frame arrived from focused server");
                vt_fallback_active = false;
            }
            // Diagnostic: callback→commit timing
            let commit_ns = wayland_state.staged_at_ns;
            let callback_to_commit_us = if wayland_state.last_callback_fire_ns > 0
                && commit_ns > wayland_state.last_callback_fire_ns
            {
                (commit_ns - wayland_state.last_callback_fire_ns) / 1000
            } else {
                0
            };
            diag_sum_callback_to_commit_us += callback_to_commit_us;
            diag_pipeline_frames += 1;
        }
        if can_present {
            forward_staged_frame(&mut wayland_state, has_focus);
        }
        if stats_frames_forwarded > 0 && wayland_state.staged_buffer.is_none() {
            // Just forwarded — record timestamp for forward-to-flip measurement
            diag_last_forward_ns = wayland::monotonic_ns();
        }

        // ── Commit-based presentation switch ───────────────────────
        // The presentation focus only updates when the new logical
        // focus target has actually committed a frame. This prevents
        // flicker: the old content stays on screen until the new
        // window renders.
        let logical_server = result.server_index;
        if presentation_server_index != logical_server
            && wayland_state.staged_buffer.is_none()
            && wayland_state.staged_buffer_server_index == logical_server
        {
            presentation_server_index = logical_server;
            presentation_app_id = result.app_id;
            info!(
                presentation_app_id,
                presentation_server_index, "presentation focus switched"
            );

            // NOW publish focus feedback — the new client is visible.
            if let Some(primary) = xwayland_servers.first() {
                let _ = primary
                    .cmd_tx
                    .send(wayland::xwayland::XwmCommand::SetGlobalFocus {
                        app_id: presentation_app_id,
                    });
            }
        }

        // Also handle first-time presentation (boot) and focus-to-no-focus.
        if presentation_server_index == logical_server
            && presentation_app_id != result.app_id
            && result.app_id != 0
        {
            presentation_app_id = result.app_id;
            if let Some(primary) = xwayland_servers.first() {
                let _ = primary
                    .cmd_tx
                    .send(wayland::xwayland::XwmCommand::SetGlobalFocus {
                        app_id: presentation_app_id,
                    });
            }
        }

        // Dispatch Wayland requests. Unlike the old approach (which blocked
        // dispatch when staged_buffer was occupied), we always accept commits.
        // New commits overwrite the staging slot — only the latest frame is
        // forwarded to the render thread. Backpressure comes from frame
        // callbacks (VBlank-driven) and buffer release, not dispatch gating.
        //
        // NOTE: dispatch already ran above (before forward) so any commits
        // arrived in this iteration are already staged. This second dispatch
        // catches any additional events (e.g. follow-up requests after a
        // forward freed up dispatch capacity).
        let dispatch_before = wayland::monotonic_ns();
        if let Err(e) = wayland_server.dispatch(&mut wayland_state) {
            warn!(?e, "Wayland dispatch error");
        }
        stats_dispatch_ns += wayland::monotonic_ns() - dispatch_before;

        // Send keyboard/pointer enter to the focused surface (once).
        wayland_state.update_input_focus();

        wayland_server.flush();

        // Sleep until the next event or VBlank tick.
        poll_or_sleep(wayland_fd, &wayland_state, &vblank_timer);
    }

    info!("main loop exited, cleaning up");

    // Kill the child application (e.g., vkcube). Without this, the child
    // inherits our stdio and keeps the terminal session alive after exit.
    if let Some(ref mut child) = child_process {
        info!("killing child process");
        let _ = child.kill();
        let _ = child.wait();
    }

    // Shut down all XWM threads and XWayland instances.
    for srv in xwayland_servers.into_iter().rev() {
        let _ = srv.cmd_tx.send(wayland::xwayland::XwmCommand::Shutdown);
        let _ = srv.thread.join();
        xwayland_mgr::terminate_xwayland(srv.child, &srv.display);
    }

    // Wait for render thread to finish.
    drop(calloop_frame_tx);
    let _ = render_thread.join();

    Ok(())
}

// ─── Event loop helpers ─────────────────────────────────────────────

/// Launch the child process (e.g., Steam, Grid, vkcube) with the correct
/// `DISPLAY`, `WAYLAND_DISPLAY`, and `STEAM_GAME_DISPLAY_N` env vars.
///
/// Both `WAYLAND_DISPLAY` (our compositor socket) and `DISPLAY` (XWayland
/// server 0) are set so that native Wayland clients (e.g., Flutter/Grid)
/// connect directly while X11 games route through XWayland.
fn launch_child_command(
    config: &Config,
    xwayland_servers: &[XWaylandInstance],
    xwayland_count: u32,
    socket_name: &str,
) -> anyhow::Result<Option<std::process::Child>> {
    let Some(ref cmd) = config.child_command else {
        return Ok(None);
    };

    info!(command = %cmd, "launching child process");
    let mut child_cmd = std::process::Command::new("sh");
    child_cmd
        .arg("-c")
        .arg(cmd)
        .env("WAYLAND_DISPLAY", socket_name)
        .env("DISPLAY", &xwayland_servers[0].display);

    // Set STEAM_GAME_DISPLAY_N env vars for game servers (1+).
    if xwayland_count > 1 {
        for server in xwayland_servers.iter().skip(1) {
            let env_name = format!("STEAM_GAME_DISPLAY_{}", server.index - 1);
            child_cmd.env(&env_name, &server.display);
        }
    }

    child_cmd
        .spawn()
        .map(Some)
        .context("failed to launch child command")
}

/// Dispatch libseat events and handle VT switch recovery.
///
/// On session restore (inactive → active), re-opens keyboard devices since
/// logind revokes evdev fds via `EVIOCREVOKE` on VT switch. The pointer
/// monitor uses libinput's suspend/resume mechanism instead.
fn dispatch_session(
    session: &mut Option<backend::session::Session>,
    was_active: &mut bool,
    keyboard_monitor: &mut Option<KeyboardMonitor>,
    pointer_monitor: &mut Option<PointerMonitor>,
    wayland_state: &mut wayland::WaylandState,
) -> bool {
    let Some(sess) = session.as_mut() else {
        return false;
    };

    if let Err(e) = sess.dispatch() {
        warn!(?e, "seat dispatch error");
    }

    let is_active = sess.is_active();
    let mut resumed = false;
    // Session going inactive — suspend libinput so it closes devices.
    // Release all held buffers so the client isn't permanently blocked.
    // Page flips won't happen while inactive, so buffers would leak.
    if *was_active && !is_active {
        if let Some(ptr) = pointer_monitor.as_mut() {
            ptr.suspend();
        }
        while wayland_state.release_one_buffer() {}
        // Discard pending presentation feedbacks — these frames will not
        // be shown while the session is inactive.
        wayland::protocols::presentation::discard_staged(wayland_state);
        wayland::protocols::presentation::discard_inflight(wayland_state);
    }
    // Session restored — re-open keyboard devices and resume libinput.
    if !*was_active && is_active {
        resumed = true;
        info!("session restored, re-opening input devices");
        if let Some(kbd) = keyboard_monitor.as_mut() {
            kbd.reopen_after_vt_switch(sess);
        }
        if let Some(ptr) = pointer_monitor.as_mut() {
            ptr.resume();
        }
        // Fire frame callbacks so clients repaint after VT switch back.
        // The render thread drains stale frames while paused and needs
        // a fresh commit to present after modeset.
        info!("firing frame callbacks to request client repaint after VT switch");
        wayland_state.fire_all_callbacks();
    }
    *was_active = is_active;
    resumed
}

/// Poll evdev keyboard devices and forward key events to Wayland clients.
fn poll_keyboard(
    keyboard_monitor: &mut Option<KeyboardMonitor>,
    session: &mut Option<backend::session::Session>,
    wayland_state: &mut wayland::WaylandState,
) {
    if let Some(kbd) = keyboard_monitor.as_mut()
        && let Some(sess) = session.as_mut()
    {
        for action in kbd.poll(sess) {
            if let KeyAction::Key {
                key,
                pressed,
                time_ms,
            } = action
            {
                wayland_state.send_key(key, pressed, time_ms);
            }
        }
    }
}

/// Poll pointer devices via libinput and forward events to Wayland clients.
#[allow(clippy::too_many_arguments)]
fn poll_pointer(
    pointer_monitor: &mut Option<PointerMonitor>,
    wayland_state: &mut wayland::WaylandState,
    cursor_x: &AtomicI32,
    cursor_y: &AtomicI32,
    display_width: &AtomicU32,
    display_height: &AtomicU32,
    sensitivity: f64,
) {
    use crate::input::pointer::PointerEvent;

    if let Some(ptr) = pointer_monitor.as_mut() {
        for event in ptr.poll() {
            match event {
                PointerEvent::Motion { dx, dy, time_ms } => {
                    // Apply sensitivity multiplier to libinput's DPI-normalized
                    // unaccelerated deltas. 1.0 = 1:1 mapping.
                    let dx = dx * sensitivity;
                    let dy = dy * sensitivity;
                    wayland_state.send_pointer_motion(dx, dy, time_ms);
                    // Scale cursor position from client coords to display coords
                    // for the hardware cursor plane. When the client renders at a
                    // lower resolution than the display, the blitter upscales the
                    // frame, so the cursor must be positioned in display space.
                    let client_x = wayland_state.pointer_x;
                    let client_y = wayland_state.pointer_y;
                    let dw = display_width.load(Ordering::Relaxed);
                    let dh = display_height.load(Ordering::Relaxed);
                    let cw = wayland_state.output_width;
                    let ch = wayland_state.output_height;
                    let (sx, sy) = if dw > 0 && dh > 0 && cw > 0 && ch > 0 {
                        (
                            (client_x * dw as f64 / cw as f64) as i32,
                            (client_y * dh as f64 / ch as f64) as i32,
                        )
                    } else {
                        (client_x as i32, client_y as i32)
                    };
                    cursor_x.store(sx, Ordering::Relaxed);
                    cursor_y.store(sy, Ordering::Relaxed);
                }
                PointerEvent::Button(btn) => {
                    let time_ms = (btn.time_usec / 1000) as u32;
                    wayland_state.send_pointer_button(btn.button, btn.pressed, time_ms);
                }
                PointerEvent::Scroll(scroll) => {
                    let time_ms = (scroll.time_usec / 1000) as u32;
                    wayland_state.send_pointer_axis(scroll.dx, scroll.dy, time_ms);
                }
            }
        }
    }
}

/// Drain host compositor input events (nested/wayland mode) and forward
/// them to Wayland clients.
fn drain_host_input(
    host_input_rx: &std::sync::mpsc::Receiver<crate::backend::wayland::WaylandEvent>,
    wayland_state: &mut wayland::WaylandState,
) {
    use crate::backend::wayland::WaylandEvent;

    let now_ms = (wayland::monotonic_ns() / 1_000_000) as u32;
    while let Ok(event) = host_input_rx.try_recv() {
        match event {
            WaylandEvent::Key { key, pressed } => {
                wayland_state.send_key(key, pressed, now_ms);
            }
            WaylandEvent::Modifiers {
                mods_depressed,
                mods_latched,
                mods_locked,
                group,
            } => {
                wayland_state.send_modifiers(mods_depressed, mods_latched, mods_locked, group);
            }
            WaylandEvent::Keymap { format, fd, size } => {
                wayland_state.send_keymap(format, fd, size);
            }
            WaylandEvent::PointerMotion { x, y } => {
                wayland_state.send_pointer_motion_absolute(x, y, now_ms);
            }
            WaylandEvent::PointerButton { button, pressed } => {
                wayland_state.send_pointer_button(button, pressed, now_ms);
            }
            WaylandEvent::Scroll { dx, dy } => {
                wayland_state.send_pointer_axis(dx, dy, now_ms);
            }
            WaylandEvent::FocusIn => {
                // Host window regained focus (e.g., VT switch back, workspace
                // switch). Flutter only renders on state changes, so fire
                // frame callbacks + reconfigure toplevels to prompt a repaint.
                info!("host focus regained, requesting client repaint");
                wayland_state.fire_all_callbacks();
                wayland_state.reconfigure_toplevels();
            }
            _ => {}
        }
    }
}

/// Propagate host window physical size to Wayland clients and XWM threads.
///
/// Only sends updates when the resolution has actually changed since the
/// last call, avoiding redundant wl_output.mode events and XWM commands.
fn propagate_host_resolution(
    host_physical_width: &AtomicU32,
    host_physical_height: &AtomicU32,
    wayland_state: &mut wayland::WaylandState,
    xwayland_servers: &[XWaylandInstance],
    last_w: &mut u32,
    last_h: &mut u32,
) {
    let pw = host_physical_width.load(Ordering::Acquire);
    let ph = host_physical_height.load(Ordering::Acquire);
    if pw > 0 && ph > 0 && (pw != *last_w || ph != *last_h) {
        *last_w = pw;
        *last_h = ph;
        wayland_state.update_output_resolution(pw, ph);
        for srv in xwayland_servers {
            let _ = srv
                .cmd_tx
                .send(wayland::xwayland::XwmCommand::SetResolution {
                    width: pw,
                    height: ph,
                });
        }
    }
}

/// Update overlay commit gate atomics from a server's focus state.
///
/// Writes the overlay and external overlay wl_surface_ids so the commit
/// handler accepts their commits alongside the focused app surface.
fn update_overlay_atomics(
    state: &wayland::WaylandState,
    focus: &wayland::window_tracker::FocusState,
    server_index: u32,
) {
    state
        .overlay_wl_surface_id
        .store(focus.overlay_wl_surface_id, Ordering::Relaxed);
    state.overlay_server_index.store(
        if focus.overlay_wl_surface_id != 0 {
            server_index
        } else {
            u32::MAX
        },
        Ordering::Relaxed,
    );
    state
        .external_overlay_wl_surface_id
        .store(focus.external_overlay_wl_surface_id, Ordering::Relaxed);
    state.external_overlay_server_index.store(
        if focus.external_overlay_wl_surface_id != 0 {
            server_index
        } else {
            u32::MAX
        },
        Ordering::Relaxed,
    );
    *state.overlay_opacity.lock() = focus.overlay_opacity;
    *state.external_overlay_opacity.lock() = focus.external_overlay_opacity;
    state
        .overlay_input_focus_mode
        .store(focus.overlay_input_focus_mode, Ordering::Relaxed);
    state
        .external_overlay_input_focus_mode
        .store(focus.external_overlay_input_focus_mode, Ordering::Relaxed);
}

/// Check if the wayland backend detected the host display refresh rate and
/// update the frame pacer and FPS limiter accordingly.
///
/// Called once per main-loop iteration. Does nothing if the rate hasn't
/// changed since the last check. An explicit `--fps-limit` override
/// prevents the target FPS from being updated, but the display refresh
/// is always synced.
fn update_refresh_rate(
    detected_mhz: &AtomicU32,
    last_hz: &mut u32,
    config: &Config,
    pacer: &mut FramePacer,
    limiter: &mut FpsLimiter,
    wayland_state: &mut wayland::WaylandState,
    vblank_timer: &mut Option<vblank_timer::VBlankTimer>,
) {
    let raw_mhz = detected_mhz.load(Ordering::Relaxed);
    if raw_mhz == 0 {
        return;
    }
    // Allow forcing 60 Hz output for testing. When set, XWayland/XRandR
    // reports 60 Hz to GDK and GTK's frame clock targets 60fps, matching
    // Flutter's hardcoded 60fps VsyncWaiterFallback timer.
    let mhz = if std::env::var("GAMECOMP_FORCE_60HZ").is_ok() {
        60_000
    } else {
        raw_mhz
    };
    let hz = (mhz + 500) / 1000;
    if hz == *last_hz {
        return;
    }
    *last_hz = hz;
    info!(
        detected_hz = hz,
        detected_mhz = mhz,
        raw_mhz = raw_mhz,
        forced_60hz = std::env::var("GAMECOMP_FORCE_60HZ").is_ok(),
        "host display refresh rate detected"
    );
    *pacer = FramePacer::new(hz);
    pacer.set_red_zone(config.red_zone_us * 1000);
    pacer.set_vrr(config.vrr);

    if config.fps_limit == 0 {
        limiter.set_target_fps(hz);
    }
    limiter.set_display_refresh(hz);

    // Re-arm the VBlank timer at the new refresh rate.
    if let Some(timer) = vblank_timer {
        timer.arm(hz);
    }

    // Push the new refresh rate to all bound `wl_output` clients so they
    // (e.g. Flutter via XWayland RandR) target the correct vsync interval.
    wayland_state.update_output_refresh(mhz);
}

/// Forward the staged buffer to the render thread.
///
/// This is gamescope's equivalent of `paint_all()` — the frame is accepted
/// for presentation. Since we gate this on `presents_in_flight == 0`, it only
/// runs when the display is ready for a new frame.
///
/// Callbacks are fired by the vblank timer, not here.
fn forward_staged_frame(state: &mut wayland::WaylandState, has_focus: bool) {
    let Some(buffer) = state.staged_buffer.take() else {
        return;
    };

    // Buffer release is now handled by fire_all_surface_callbacks()
    // at vblank time, so wl_buffer.release and wl_callback.done arrive
    // in the same Wayland flush — matching gamescope's paint_all().

    // Move staged presentation feedbacks into in-flight. They'll be sent
    // `presented(...)` when the page flip completes.
    state.inflight_feedbacks.append(&mut state.staged_feedbacks);

    let now_ns = wayland::monotonic_ns();
    let staging_duration_ms = if state.staged_at_ns > 0 {
        (now_ns - state.staged_at_ns) / 1_000_000
    } else {
        0
    };
    trace!(
        staging_duration_ms,
        held_count = state.held_buffers.len(),
        deferred_count = state.deferred_callback_count(),
        "forward: sending frame to render thread"
    );

    // Only present if a window actually has focus. Without a
    // focused window, buffers are dropped so the host window
    // stays blank.
    if has_focus && let Some(ref tx) = state.frame_channel {
        // Overlay buffers persist across frames: the overlay client
        // may commit infrequently while the game commits at 60 fps.
        // Dup the fds so the render thread gets its own copy while
        // the original stays in staged_*_buffer for the next frame.
        //
        // When the overlay is inactive (surface ID == 0), we return
        // None so no overlay is composited, but keep the buffer cached
        // for instant display when the overlay reactivates.
        let overlay =
            dup_or_clear_overlay(&state.staged_overlay_buffer, &state.overlay_wl_surface_id);
        let external_overlay = dup_or_clear_overlay(
            &state.staged_external_overlay_buffer,
            &state.external_overlay_wl_surface_id,
        );
        let overlay_opacity = *state.overlay_opacity.lock();
        let external_overlay_opacity = *state.external_overlay_opacity.lock();

        if state.overlay_wl_surface_id.load(Ordering::Relaxed) != 0 {
            debug!(
                has_overlay_buffer = overlay.is_some(),
                has_staged_overlay = state.staged_overlay_buffer.is_some(),
                overlay_id = state.overlay_wl_surface_id.load(Ordering::Relaxed),
                overlay_opacity,
                "forward: overlay state"
            );
        }

        let frame = wayland::protocols::CommittedFrame {
            app: buffer,
            overlay,
            external_overlay,
            overlay_opacity,
            external_overlay_opacity,
        };
        let _ = tx.send(frame);
    }

    // Deferred callbacks are fired on vblank ticks (see main loop).
    // This decouples callback timing from commit timing, giving XWayland
    // a steady MSC clock aligned with the display refresh rate.
    // Do NOT fire here — let the vblank timer be the sole driver.
}

/// Dup an overlay buffer for the render thread, or return None if the
/// overlay is not active.
///
/// The original stays in `slot` so subsequent game frames can reuse it
/// without the overlay client having to re-commit every frame.
///
/// The cached buffer persists across deactivation/reactivation:
/// - When overlay is active: returns a dup of the cached buffer
/// - When overlay is inactive: returns None (no overlay composited)
/// - When overlay reactivates: the cached buffer from the previous session
///   is immediately available while waiting for fresh content
///
/// This matches gamescope's approach where overlay buffers persist
/// indefinitely in the commit_queue.
#[inline(always)]
fn dup_or_clear_overlay(
    slot: &Option<wayland::protocols::CommittedBuffer>,
    surface_id: &std::sync::atomic::AtomicU32,
) -> Option<wayland::protocols::CommittedBuffer> {
    if surface_id.load(Ordering::Relaxed) == 0 {
        return None;
    }
    slot.as_ref().and_then(|buf| buf.try_dup().ok())
}

/// Sleep until the next Wayland event or VBlank timer tick.
///
/// Polls both the Wayland fd (client events) and the VBlank timerfd
/// (display refresh ticks). Wakes on whichever fires first.
fn poll_or_sleep(
    wayland_fd: i32,
    state: &wayland::WaylandState,
    vblank_timer: &Option<vblank_timer::VBlankTimer>,
) {
    // Wake on Wayland events (client commits, protocol requests) or
    // VBlank timer ticks (for firing frame callbacks). Use a very short
    // baseline timeout so we quickly notice page flip completions on
    // the vblank_rx channel (which isn't pollable as an fd).
    let timeout = std::time::Duration::from_millis(1);

    let timer_fd = vblank_timer.as_ref().map(|t| t.raw_fd()).unwrap_or(-1);
    let nfds: libc::nfds_t = if timer_fd >= 0 { 2 } else { 1 };

    let mut fds = [
        libc::pollfd {
            fd: wayland_fd,
            events: libc::POLLIN,
            revents: 0,
        },
        libc::pollfd {
            fd: timer_fd,
            events: libc::POLLIN,
            revents: 0,
        },
    ];

    let ts = libc::timespec {
        tv_sec: timeout.as_secs() as i64,
        tv_nsec: timeout.subsec_nanos() as i64,
    };
    // SAFETY: fds points to a valid pollfd array on the stack.
    // timespec is on the stack. null sigmask keeps the current signal mask.
    let ret = unsafe { libc::ppoll(fds.as_mut_ptr(), nfds, &ts, std::ptr::null()) };
    if ret < 0 {
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::Interrupted {
            warn!(?err, "ppoll error");
        }
    }

    let _ = state; // Used for has_pending check in future if needed.
}

/// Install signal handlers for SIGTERM and SIGINT.
fn install_signal_handlers() -> anyhow::Result<()> {
    // SAFETY: Signal handlers only set an atomic bool — async-signal-safe.
    unsafe {
        libc_signal(libc::SIGINT);
        libc_signal(libc::SIGTERM);
    }
    Ok(())
}

unsafe fn libc_signal(sig: libc::c_int) {
    // SAFETY: We register a minimal signal handler that only writes an atomic.
    unsafe {
        libc::signal(sig, signal_handler as *const () as libc::sighandler_t);
    }
}

extern "C" fn signal_handler(_sig: libc::c_int) {
    // Ordering: Relaxed — all threads poll this flag periodically.
    RUNNING.store(false, Ordering::Relaxed);
}
