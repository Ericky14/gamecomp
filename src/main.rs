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
mod wayland;
mod xwayland_mgr;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;

use anyhow::Context;
use tracing::{error, info, trace, warn};
use tracing_subscriber::EnvFilter;

use crate::compositor::scene::FrameInfo;
use crate::config::Config;
use crate::focus::{FocusArbiter, ServerFocusState};
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

    // Shared host DMA-BUF format→modifier map. Written by the wayland
    // backend's event thread during its initial roundtrip, read by the
    // client-facing dmabuf module to advertise formats that enable zero-copy.
    let host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>> =
        Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    wayland_state.host_dmabuf_formats = host_dmabuf_formats.clone();

    // Create frame channel: main thread → wayland backend (committed buffers).
    let (frame_tx, frame_rx) = std::sync::mpsc::channel::<wayland::protocols::CommittedBuffer>();
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

        let mut ptr = PointerMonitor::new();
        ptr.open_from_session(sess);
        pointer_monitor = Some(ptr);
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

        // Also wait for the host window configure so we know the actual
        // window dimensions before launching XWayland. Without this,
        // clients start at the CLI-supplied resolution (e.g., 2560×1440)
        // instead of the host-constrained physical dimensions, causing
        // buffer size mismatches on the first frames.
        xwayland_mgr::wait_for_host_configure(&host_physical_width, &host_physical_height);
        let pw = host_physical_width.load(Ordering::Acquire);
        let ph = host_physical_height.load(Ordering::Acquire);
        if pw > 0 && ph > 0 {
            wayland_state.update_output_resolution(pw, ph);
        }
    }

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
        let display_str = xwayland_mgr::find_free_x11_display()?;

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
        });
    }

    // --- Launch child command ---
    let mut child_process = launch_child_command(&config, &xwayland_servers, xwayland_count)?;

    // --- Main event loop ---
    info!("entering main event loop");

    let wayland_fd = wayland_server.display_fd();

    // Track the last-seen detected refresh rate to avoid redundant updates.
    let mut last_detected_hz: u32 = 0;

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

    while RUNNING.load(Ordering::Relaxed) {
        dispatch_session(
            &mut session,
            &mut session_was_active,
            &mut keyboard_monitor,
            &mut pointer_monitor,
        );
        poll_keyboard(&mut keyboard_monitor, &mut session, &mut wayland_state);
        poll_pointer(&mut pointer_monitor, &mut session, &mut wayland_state);
        drain_host_input(&host_input_rx, &mut wayland_state);

        // Drain vblank timestamps from the render thread and feed them to
        // the frame pacer. This synchronizes frame callback releases with
        // the display's actual refresh cycle on the DRM path.
        while let Ok(vblank_ns) = vblank_rx.try_recv() {
            pacer.mark_vblank(vblank_ns);
        }

        update_refresh_rate(
            &detected_refresh_mhz,
            &mut last_detected_hz,
            &config,
            &mut pacer,
            &mut fps_limiter,
        );

        propagate_host_resolution(
            &host_physical_width,
            &host_physical_height,
            &mut wayland_state,
            &xwayland_servers,
        );

        // Monitor all XWayland instances and respawn crashed ones.
        for srv in &mut xwayland_servers {
            xwayland_mgr::monitor_xwayland(
                &mut srv.child,
                &srv.display,
                &socket_name,
                &mut wayland_server,
                &mut wayland_state,
                srv.index,
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
        focus_arbiter.drain_events(&xwm_event_rx);
        let mut focus_states: Vec<_> = xwayland_servers.iter().map(|s| s.focus_state()).collect();
        focus_states.push(ServerFocusState {
            index: native_focus.index,
            focused_app_id: Arc::clone(&native_focus.focused_app_id),
            focused_wl_surface_id: Arc::clone(&native_focus.focused_wl_surface_id),
        });
        let result = focus_arbiter.update(&focus_states);

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
                let pending = wayland_state.pending_frame_callbacks.len();
                let deferred = wayland_state.deferred_frame_callbacks.len();
                let held = wayland_state.held_buffers.len();
                trace!(pending, deferred, held, "focus changed: waking new client");
                wayland_state.fire_frame_callbacks();
                wayland_state.fire_deferred_callbacks();
                // Release ALL held wl_buffers from the previous focus
                // owner so we don't leak FDs.
                wayland_state.release_all_buffers();
                // Do NOT clear staged_buffer — the old client's last
                // frame stays visible until the new client commits.

                // Re-configure toplevels so the newly-focused client
                // receives a configure event and commits, restarting
                // its frame callback cycle.
                wayland_state.reconfigure_toplevels();
            }
        }

        // Forward staged buffer if the FPS limiter allows it.
        // Suppress presentation when no app has focus (focused_app_id == 0
        // means no window is mapped or focusable yet).
        let has_focus = focused_app_id.load(Ordering::Relaxed) != 0;
        forward_staged_frame(&mut wayland_state, &mut fps_limiter, has_focus);

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

        // Dispatch Wayland requests only when the staging slot is empty
        // (backpressure: don't accept new commits while one is pending).
        if wayland_state.staged_buffer.is_none()
            && let Err(e) = wayland_server.dispatch(&mut wayland_state)
        {
            warn!(?e, "Wayland dispatch error");
        }

        // Send keyboard/pointer enter to the focused surface (once).
        wayland_state.send_focus_enter();

        wayland_server.flush();

        // Sleep until the next event or FPS tick.
        poll_or_sleep(wayland_fd, &wayland_state, &fps_limiter);
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
        let mut child = srv.child;
        let _ = child.kill();
        let _ = child.wait();
    }

    // Wait for render thread to finish.
    drop(calloop_frame_tx);
    let _ = render_thread.join();

    Ok(())
}

// ─── Event loop helpers ─────────────────────────────────────────────

/// Launch the child process (e.g., Steam, vkcube) with the correct
/// `DISPLAY` and `STEAM_GAME_DISPLAY_N` env vars.
///
/// `WAYLAND_DISPLAY` is explicitly removed so all apps route through
/// XWayland (X11), where window tracking and focus gating operate.
fn launch_child_command(
    config: &Config,
    xwayland_servers: &[XWaylandInstance],
    xwayland_count: u32,
) -> anyhow::Result<Option<std::process::Child>> {
    let Some(ref cmd) = config.child_command else {
        return Ok(None);
    };

    info!(command = %cmd, "launching child process");
    let mut child_cmd = std::process::Command::new("sh");
    child_cmd
        .arg("-c")
        .arg(cmd)
        .env_remove("WAYLAND_DISPLAY")
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
/// On session restore (inactive → active), re-opens input devices since
/// logind revokes evdev fds via `EVIOCREVOKE` on VT switch.
fn dispatch_session(
    session: &mut Option<backend::session::Session>,
    was_active: &mut bool,
    keyboard_monitor: &mut Option<KeyboardMonitor>,
    pointer_monitor: &mut Option<PointerMonitor>,
) {
    let Some(sess) = session.as_mut() else { return };

    if let Err(e) = sess.dispatch() {
        warn!(?e, "seat dispatch error");
    }

    let is_active = sess.is_active();
    if !*was_active && is_active {
        info!("session restored, re-opening input devices");
        if let Some(kbd) = keyboard_monitor.as_mut() {
            kbd.reopen_after_vt_switch(sess);
        }
        if let Some(ptr) = pointer_monitor.as_mut() {
            ptr.reopen_after_vt_switch(sess);
        }
    }
    *was_active = is_active;
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

/// Poll evdev pointer devices and forward events to Wayland clients.
fn poll_pointer(
    pointer_monitor: &mut Option<PointerMonitor>,
    session: &mut Option<backend::session::Session>,
    wayland_state: &mut wayland::WaylandState,
) {
    use crate::input::pointer::PointerEvent;

    if let Some(ptr) = pointer_monitor.as_mut()
        && let Some(sess) = session.as_mut()
    {
        for event in ptr.poll(sess) {
            match event {
                PointerEvent::Motion { dx, dy, time_ms } => {
                    wayland_state.send_pointer_motion(dx, dy, time_ms);
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
            _ => {}
        }
    }
}

/// Propagate host window physical size to Wayland clients and XWM threads.
fn propagate_host_resolution(
    host_physical_width: &AtomicU32,
    host_physical_height: &AtomicU32,
    wayland_state: &mut wayland::WaylandState,
    xwayland_servers: &[XWaylandInstance],
) {
    let pw = host_physical_width.load(Ordering::Acquire);
    let ph = host_physical_height.load(Ordering::Acquire);
    if pw > 0 && ph > 0 {
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
) {
    let mhz = detected_mhz.load(Ordering::Relaxed);
    if mhz == 0 {
        return;
    }
    let hz = (mhz + 500) / 1000;
    if hz == *last_hz {
        return;
    }
    *last_hz = hz;
    info!(
        detected_hz = hz,
        detected_mhz = mhz,
        "host display refresh rate detected"
    );
    *pacer = FramePacer::new(hz);
    pacer.set_red_zone(config.red_zone_us * 1000);
    pacer.set_vrr(config.vrr);

    if config.fps_limit == 0 {
        limiter.set_target_fps(hz);
    }
    limiter.set_display_refresh(hz);
}

/// Forward the staged buffer to the render thread if the FPS limiter allows it.
///
/// This is the throttle valve: it releases held wl_buffers back to the client,
/// fires deferred frame callbacks, and sends the committed buffer to the render
/// thread — all gated by the FPS limiter's timestamp check.
///
/// Must be called BEFORE Wayland dispatch so the staged slot is empty when we
/// accept new commits.
fn forward_staged_frame(
    state: &mut wayland::WaylandState,
    limiter: &mut FpsLimiter,
    has_focus: bool,
) {
    let now_ns = wayland::monotonic_ns();
    if !limiter.should_release(now_ns) {
        return;
    }

    let mut did_release = false;

    if let Some(buffer) = state.staged_buffer.take() {
        // Only present if a window actually has focus. Without a
        // focused window, buffers are dropped so the host window
        // stays blank.
        if has_focus && let Some(ref tx) = state.frame_channel {
            let _ = tx.send(buffer);
        }
        did_release = true;
    }

    // Release exactly ONE held wl_buffer back to the client.
    //
    // By releasing only one buffer per tick, the client can acquire
    // one free swapchain image, render one frame, and commit — then
    // it blocks until the next tick releases another buffer.
    if state.release_one_buffer() {
        did_release = true;
    }

    if state.fire_deferred_callbacks() {
        did_release = true;
    }

    if did_release {
        limiter.mark_released(now_ns);
    }
}

/// Sleep until the next Wayland event or FPS limiter deadline.
///
/// Uses `ppoll()` for sub-millisecond precision. When a staged buffer is
/// pending, this acts as a pure timer (no fd polling) — we won't accept new
/// commits until the limiter fires. When idle, we poll the Wayland fd to
/// wake immediately on client events.
fn poll_or_sleep(wayland_fd: i32, state: &wayland::WaylandState, limiter: &FpsLimiter) {
    let now_ns = wayland::monotonic_ns();
    let has_pending = state.staged_buffer.is_some() || !state.deferred_frame_callbacks.is_empty();

    let timeout = if has_pending {
        limiter.time_until_release(now_ns)
    } else {
        // Idle: wake on Wayland events or periodically to accept
        // connections and check XWayland status.
        std::time::Duration::from_millis(4)
    };

    let mut fds = [libc::pollfd {
        fd: wayland_fd,
        events: libc::POLLIN,
        revents: 0,
    }];
    let (fds_ptr, nfds) = if has_pending {
        (std::ptr::null_mut(), 0)
    } else {
        (fds.as_mut_ptr(), 1)
    };

    let ts = libc::timespec {
        tv_sec: timeout.as_secs() as i64,
        tv_nsec: timeout.subsec_nanos() as i64,
    };
    // SAFETY: fds_ptr is either null (nfds=0, pure timer) or points to
    // a valid pollfd array. timespec is on the stack. null sigmask
    // keeps the current signal mask.
    let ret = unsafe { libc::ppoll(fds_ptr, nfds, &ts, std::ptr::null()) };
    if ret < 0 {
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::Interrupted {
            warn!(?err, "ppoll error");
        }
    }
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
