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
mod frame_pacer;
mod input;
mod render;
mod retry;
mod stats;
#[cfg(test)]
mod test_harness;
mod wayland;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;

use anyhow::Context;
use tracing::{debug, error, info, trace, warn};
use tracing_subscriber::EnvFilter;

use crate::backend::Backend;
use crate::compositor::scene::FrameInfo;
use crate::config::Config;
use crate::frame_pacer::{FpsLimiter, FramePacer};
use crate::input::InputHandler;
use crate::input::keyboard::{KeyAction, KeyboardMonitor};
use crate::input::pointer::PointerMonitor;
use crate::retry::{RetryPolicy, retry_with_backoff};
use crate::stats::StatsTracker;
use crate::wayland::WaylandServer;

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

    // Determine the XWayland display number early so we can set both
    // WAYLAND_DISPLAY and DISPLAY before spawning any threads.
    let xwayland_display = find_free_x11_display()?;

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
            render_thread_main(
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
        wait_for_host_formats(&host_dmabuf_formats);

        // Also wait for the host window configure so we know the actual
        // window dimensions before launching XWayland. Without this,
        // clients start at the CLI-supplied resolution (e.g., 2560×1440)
        // instead of the host-constrained physical dimensions, causing
        // buffer size mismatches on the first frames.
        wait_for_host_configure(&host_physical_width, &host_physical_height);
        let pw = host_physical_width.load(Ordering::Acquire);
        let ph = host_physical_height.load(Ordering::Acquire);
        if pw > 0 && ph > 0 {
            wayland_state.update_output_resolution(pw, ph);
        }
    }

    // --- Launch XWayland ---
    // Spawned after the render thread so that when XWayland connects and binds
    // zwp_linux_dmabuf_v1, the host format map is already populated.
    let mut xwayland_child = spawn_xwayland(
        &xwayland_display,
        &socket_name,
        &mut wayland_server,
        &mut wayland_state,
    )?;
    info!(display = %xwayland_display, "XWayland ready");

    // --- Spawn XWM thread ---
    // The XWM thread owns the channels and runs a retry loop: if the X11
    // connection drops (e.g., XWayland restart), it reconnects with
    // exponential backoff instead of silently dying.
    let (xwm_event_tx, _xwm_event_rx) = calloop::channel::channel::<wayland::xwayland::XwmEvent>();
    let (xwm_cmd_tx, xwm_cmd_rx) = std::sync::mpsc::channel::<wayland::xwayland::XwmCommand>();
    let xwm_display = xwayland_display.clone();
    let xwm_thread = thread::Builder::new()
        .name("gamecomp-xwm".to_string())
        .spawn({
            // Use host physical dims if available, otherwise fall back to
            // game_w/game_h. After wait_for_host_configure, the atomics
            // should be set for the Wayland backend.
            let xwm_w = {
                let v = host_physical_width.load(Ordering::Acquire);
                if v > 0 { v } else { game_w }
            };
            let xwm_h = {
                let v = host_physical_height.load(Ordering::Acquire);
                if v > 0 { v } else { game_h }
            };
            move || {
                let result = retry_with_backoff("XWM", &RetryPolicy::DEFAULT, &RUNNING, || {
                    wayland::xwayland::run_xwm(
                        &xwm_display,
                        &xwm_event_tx,
                        &xwm_cmd_rx,
                        xwm_w,
                        xwm_h,
                    )
                });
                if let Err(e) = result {
                    error!(?e, "XWM thread exiting after exhausting retries");
                }
            }
        })
        .context("failed to spawn XWM thread")?;

    // --- Launch child command ---
    let mut child_process: Option<std::process::Child> = None;
    if let Some(ref cmd) = config.child_command {
        info!(command = %cmd, "launching child process");
        child_process = Some(
            std::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .env("WAYLAND_DISPLAY", &socket_name)
                .env("DISPLAY", &xwayland_display)
                .spawn()
                .context("failed to launch child command")?,
        );
    }

    // --- Main event loop ---
    info!("entering main event loop");

    let wayland_fd = wayland_server.display_fd();

    // Track the last-seen detected refresh rate to avoid redundant updates.
    let mut last_detected_hz: u32 = 0;

    while RUNNING.load(Ordering::Relaxed) {
        // Dispatch libseat events (VT switch notifications) for DRM sessions.
        if let Some(ref mut sess) = session
            && let Err(e) = sess.dispatch()
        {
            warn!(?e, "seat dispatch error");
        }

        // Detect session restore (inactive → active) and re-open input
        // devices. Logind revokes evdev fds via EVIOCREVOKE on VT switch,
        // making the old fds permanently dead. We must re-open them.
        if let Some(ref mut sess) = session {
            let is_active = sess.is_active();
            if !session_was_active && is_active {
                info!("session restored, re-opening input devices");
                if let Some(ref mut kbd) = keyboard_monitor {
                    kbd.reopen_after_vt_switch(sess);
                }
                if let Some(ref mut ptr) = pointer_monitor {
                    ptr.reopen_after_vt_switch(sess);
                }
            }
            session_was_active = is_active;
        }

        // Poll keyboard: hotplug, VT switch hotkeys, and key events.
        if let Some(ref mut kbd) = keyboard_monitor
            && let Some(ref mut sess) = session
        {
            let key_events = kbd.poll(sess);
            for action in key_events {
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

        // Poll pointer: hotplug, motion, button, and scroll events.
        if let Some(ref mut ptr) = pointer_monitor
            && let Some(ref mut sess) = session
        {
            use crate::input::pointer::PointerEvent;
            let pointer_events = ptr.poll(sess);
            for event in pointer_events {
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

        // Drain host input events (nested/wayland mode). The render thread
        // forwards keyboard/pointer events from the host compositor.
        {
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
                        wayland_state.send_modifiers(
                            mods_depressed,
                            mods_latched,
                            mods_locked,
                            group,
                        );
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

        // Drain vblank timestamps from the render thread and feed them to
        // the frame pacer. This synchronizes frame callback releases with
        // the display's actual refresh cycle on the DRM path.
        while let Ok(vblank_ns) = vblank_rx.try_recv() {
            pacer.mark_vblank(vblank_ns);
        }

        // Sync frame pacer and FPS limiter to the host display refresh rate.
        update_refresh_rate(
            &detected_refresh_mhz,
            &mut last_detected_hz,
            &config,
            &mut pacer,
            &mut fps_limiter,
        );

        // Propagate host window physical pixel size as our advertised output
        // resolution. Clients may render at this size or choose their own.
        // The wayland backend's viewport applies contain-fit scaling to
        // fit the game buffer within the host window, preserving aspect ratio.
        {
            let pw = host_physical_width.load(Ordering::Acquire);
            let ph = host_physical_height.load(Ordering::Acquire);
            if pw > 0 && ph > 0 {
                wayland_state.update_output_resolution(pw, ph);
                let _ = xwm_cmd_tx.send(wayland::xwayland::XwmCommand::SetResolution {
                    width: pw,
                    height: ph,
                });
            }
        }

        // Monitor XWayland and respawn if it crashed.
        monitor_xwayland(
            &mut xwayland_child,
            &xwayland_display,
            &socket_name,
            &mut wayland_server,
            &mut wayland_state,
        );

        // Accept new Wayland client connections.
        if let Some(stream) = wayland_server.accept()
            && let Err(e) = wayland_server.insert_client(stream, &mut wayland_state)
        {
            warn!(?e, "failed to insert Wayland client");
        }

        // Forward staged buffer if the FPS limiter allows it.
        forward_staged_frame(&mut wayland_state, &mut fps_limiter);

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

    // Shut down XWM.
    let _ = xwm_cmd_tx.send(wayland::xwayland::XwmCommand::Shutdown);
    let _ = xwm_thread.join();

    // Shut down XWayland.
    let _ = xwayland_child.kill();
    let _ = xwayland_child.wait();

    // Wait for render thread to finish.
    drop(calloop_frame_tx);
    let _ = render_thread.join();

    Ok(())
}

// ─── Event loop helpers ─────────────────────────────────────────────

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

/// Check if XWayland exited and respawn it if necessary.
///
/// If XWayland crashed or was terminated, spawns a fresh instance.
/// The XWM thread's retry loop will re-establish the window manager
/// connection automatically.
fn monitor_xwayland(
    child: &mut std::process::Child,
    display: &str,
    socket: &str,
    server: &mut WaylandServer,
    state: &mut wayland::WaylandState,
) {
    match child.try_wait() {
        Ok(Some(status)) => {
            warn!(?status, "XWayland exited, respawning");
            match spawn_xwayland(display, socket, server, state) {
                Ok(new_child) => {
                    *child = new_child;
                    info!("XWayland respawned successfully");
                }
                Err(e) => {
                    error!(?e, "failed to respawn XWayland");
                }
            }
        }
        Ok(None) => {} // Still running.
        Err(e) => {
            warn!(?e, "error checking XWayland status");
        }
    }
}

/// Forward the staged buffer to the render thread if the FPS limiter allows it.
///
/// This is the throttle valve: it releases held wl_buffers back to the client,
/// fires deferred frame callbacks, and sends the committed buffer to the render
/// thread — all gated by the FPS limiter's timestamp check.
///
/// Must be called BEFORE Wayland dispatch so the staged slot is empty when we
/// accept new commits.
fn forward_staged_frame(state: &mut wayland::WaylandState, limiter: &mut FpsLimiter) {
    let now_ns = wayland::monotonic_ns();
    if !limiter.should_release(now_ns) {
        return;
    }

    let mut did_release = false;

    if let Some(buffer) = state.staged_buffer.take() {
        if let Some(ref tx) = state.frame_channel {
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

// ─── Host format readiness ──────────────────────────────────────────

/// Wait for the wayland backend's event thread to populate host DMA-BUF
/// format/modifier pairs.
///
/// The render thread starts the wayland backend event loop, which performs
/// two roundtrips to the host compositor to discover DMA-BUF formats.
/// We block here so that XWayland (and subsequent clients) can be
/// advertised the real host formats when they bind `zwp_linux_dmabuf_v1`,
/// enabling zero-copy DMA-BUF forwarding.
///
/// Times out after 5 s — if the host compositor doesn't support DMA-BUF,
/// clients fall back to the hardcoded format list.
fn wait_for_host_formats(
    host_formats: &Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
) {
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);

    info!("waiting for host DMA-BUF formats before launching XWayland");

    let start = std::time::Instant::now();
    loop {
        let formats = host_formats.lock();
        if !formats.is_empty() {
            let elapsed = start.elapsed();
            info!(
                formats = formats.len(),
                elapsed_ms = elapsed.as_millis(),
                "host DMA-BUF formats ready, proceeding with XWayland launch"
            );
            return;
        }
        if start.elapsed() >= TIMEOUT {
            warn!(
                "timeout waiting for host DMA-BUF formats — \
                 clients will use fallback format list"
            );
            return;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Wait for the host compositor to send its first `xdg_toplevel.configure`
/// so that `host_physical_width` / `host_physical_height` are non-zero.
///
/// Without this, XWayland and the game start with the CLI-supplied
/// resolution instead of the host-constrained size, causing buffer
/// dimensions to mismatch the first viewport commit.
fn wait_for_host_configure(
    host_physical_width: &Arc<AtomicU32>,
    host_physical_height: &Arc<AtomicU32>,
) {
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);

    info!("waiting for host window configure before launching XWayland");

    let start = std::time::Instant::now();
    loop {
        let pw = host_physical_width.load(Ordering::Acquire);
        let ph = host_physical_height.load(Ordering::Acquire);
        if pw > 0 && ph > 0 {
            info!(
                physical_w = pw,
                physical_h = ph,
                elapsed_ms = start.elapsed().as_millis(),
                "host configure received, proceeding with XWayland launch"
            );
            return;
        }
        if start.elapsed() >= TIMEOUT {
            warn!(
                "timeout waiting for host configure — \
                 XWayland will start with CLI resolution"
            );
            return;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

// ─── XWayland management ────────────────────────────────────────────

/// Find a free X11 display number and return the display string (e.g., ":1").
fn find_free_x11_display() -> anyhow::Result<String> {
    let display_num = (0..33)
        .find(|n| !std::path::Path::new(&format!("/tmp/.X11-unix/X{n}")).exists())
        .context("no free X11 display number found")?;
    Ok(format!(":{display_num}"))
}

/// Spawn XWayland on the given display and wait for readiness.
///
/// Dispatches the Wayland server while waiting so XWayland can complete
/// its connection handshake. Returns the child process handle for
/// lifecycle monitoring.
fn spawn_xwayland(
    display_str: &str,
    wayland_socket: &str,
    wayland_server: &mut WaylandServer,
    wayland_state: &mut wayland::WaylandState,
) -> anyhow::Result<std::process::Child> {
    // Create a pipe for readiness notification. XWayland writes to the write-end
    // when it's ready to accept connections (replaces SIGUSR1 in modern Xwayland).
    let (read_fd, write_fd) = rustix::pipe::pipe().context("failed to create readiness pipe")?;

    info!(display = %display_str, "launching XWayland");

    let mut cmd = std::process::Command::new("Xwayland");
    cmd.arg(display_str)
        .arg("-rootless")
        .arg("-terminate")
        .arg("-displayfd")
        .arg(format!("{}", rustix::fd::AsRawFd::as_raw_fd(&write_fd)))
        .env("WAYLAND_DISPLAY", wayland_socket);

    // Keep the write-fd open in the child; close read-fd.
    use std::os::unix::process::CommandExt;
    let write_raw = rustix::fd::AsRawFd::as_raw_fd(&write_fd);
    // SAFETY: Called after fork() in the child process. Only async-signal-safe
    // functions (fcntl) are used. No heap allocation or mutex interaction.
    unsafe {
        cmd.pre_exec(move || {
            // Unset CLOEXEC on the write fd so the child inherits it.
            let flags = libc::fcntl(write_raw, libc::F_GETFD);
            libc::fcntl(write_raw, libc::F_SETFD, flags & !libc::FD_CLOEXEC);
            Ok(())
        });
    }

    let child = cmd
        .spawn()
        .context("failed to launch Xwayland \u{2014} is it installed?")?;

    // Close the write end in the parent.
    drop(write_fd);

    // Wait for XWayland to signal readiness while dispatching Wayland events.
    // XWayland connects to our server during startup, so we must accept and
    // dispatch for it to complete initialization.
    use std::io::Read;
    let mut read_file = std::fs::File::from(read_fd);
    let mut buf = [0u8; 64];
    let raw_fd = std::os::unix::io::AsRawFd::as_raw_fd(&read_file);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);

    loop {
        // Accept any pending client connections (XWayland connecting to us).
        if let Some(stream) = wayland_server.accept()
            && let Err(e) = wayland_server.insert_client(stream, wayland_state)
        {
            warn!(?e, "failed to insert Wayland client during XWayland launch");
        }

        // Dispatch Wayland events so XWayland can complete its handshake.
        let _ = wayland_server.dispatch(wayland_state);
        wayland_server.flush();

        // Poll the readiness pipe with a short timeout.
        let mut fds = [libc::pollfd {
            fd: raw_fd,
            events: libc::POLLIN,
            revents: 0,
        }];
        // SAFETY: Valid pollfd, single fd, short timeout.
        let poll_ret = unsafe { libc::poll(fds.as_mut_ptr(), 1, 50) };

        if poll_ret > 0 {
            // XWayland wrote its display number.
            let n = read_file.read(&mut buf).unwrap_or(0);
            if n > 0 {
                let reported = std::str::from_utf8(&buf[..n]).unwrap_or("").trim();
                info!(reported_display = reported, "XWayland reported ready");
            }
            return Ok(child);
        }

        if std::time::Instant::now() >= deadline {
            warn!("XWayland readiness timeout -- proceeding anyway");
            return Ok(child);
        }
    }
}

/// Render thread entry point.
///
/// Owns the Vulkan compositor and DRM backend exclusively.
/// Receives FrameInfo from the main thread, composites or performs direct
/// scanout, and signals flip completion back.
#[allow(clippy::too_many_arguments)]
fn render_thread_main(
    config: &Config,
    host_wayland_display: Option<String>,
    committed_frames: std::sync::mpsc::Receiver<wayland::protocols::CommittedBuffer>,
    cursor_updates: std::sync::mpsc::Receiver<crate::backend::wayland::CursorUpdate>,
    detected_refresh_mhz: Arc<AtomicU32>,
    host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
    drm_device: Option<(std::path::PathBuf, std::os::unix::io::OwnedFd)>,
    vblank_tx: std::sync::mpsc::Sender<u64>,
    session_active: Option<Arc<AtomicBool>>,
    host_physical_width: Arc<AtomicU32>,
    host_physical_height: Arc<AtomicU32>,
    host_input_tx: std::sync::mpsc::Sender<crate::backend::wayland::WaylandEvent>,
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
    let mut _backend: Option<crate::backend::wayland::WaylandBackend> = None;
    let mut _drm_backend: Option<crate::backend::drm::DrmBackend> = None;
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
            let mut backend = crate::backend::wayland::WaylandBackend::new(wayland_config);
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
                let mut drm = crate::backend::drm::DrmBackend::new(path.clone(), fd);
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
                    crate::backend::wayland::WaylandEvent::Resized {
                        width: _,
                        height: _,
                        physical_width,
                        physical_height,
                    } => {
                        host_physical_width.store(physical_width, Ordering::Release);
                        host_physical_height.store(physical_height, Ordering::Release);
                    }
                    crate::backend::wayland::WaylandEvent::FrameCallback => {
                        // Handled by the backend internally.
                    }
                    crate::backend::wayland::WaylandEvent::CloseRequested => {
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
    drm: &mut crate::backend::drm::DrmBackend,
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
    let mut blitter = crate::backend::gpu::vulkan_blitter::VulkanBlitter::new_for_import()
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
    let gbm_dmabufs: Vec<crate::backend::DmaBuf> =
        gbm_outputs.iter().map(|o| o.dmabuf.clone()).collect();
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
    let mut output_fb_cache: Vec<Option<crate::backend::Framebuffer>> =
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
                    crate::backend::gpu::vulkan_blitter::VulkanBlitter::poll_dmabuf_ready(
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
                        tracing::debug!(frame_count, "DRM frame presented");
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
    drm: &mut crate::backend::drm::DrmBackend,
    buffer: wayland::protocols::CommittedBuffer,
    blitter: &mut crate::backend::gpu::vulkan_blitter::VulkanBlitter,
    display_w: u32,
    display_h: u32,
    output_fb_cache: &mut [Option<crate::backend::Framebuffer>],
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
                let dmabuf = crate::backend::DmaBuf {
                    width,
                    height,
                    format: fourcc,
                    modifier: drm_fourcc::DrmModifier::from(modifier),
                    planes: planes
                        .iter()
                        .map(|p| crate::backend::DmaBufPlane {
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
                        crate::backend::FlipResult::Queued => return Ok(true),
                        crate::backend::FlipResult::DirectScanout => return Ok(false),
                        crate::backend::FlipResult::Failed(e) => {
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
                crate::backend::FlipResult::Queued => Ok(true),
                crate::backend::FlipResult::DirectScanout => Ok(false),
                crate::backend::FlipResult::Failed(e) => {
                    Err(e.context("blitted frame flip failed"))
                }
            }
        }
        wayland::protocols::CommittedBuffer::Shm { .. } => {
            // TODO: SHM buffers require GPU-side blit (Vulkan compositor).
            tracing::trace!("skipping SHM buffer on DRM path (not yet supported)");
            Ok(false)
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
