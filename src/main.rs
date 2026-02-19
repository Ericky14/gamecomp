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
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use crate::backend::Backend;
use crate::compositor::scene::FrameInfo;
use crate::config::Config;
use crate::frame_pacer::{FpsLimiter, FramePacer};
use crate::input::InputHandler;
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

    // --- Initialize Wayland server ---
    let (output_w, output_h) = config.resolution.unwrap_or((1280, 720));
    let mut wayland_server = WaylandServer::new(Vec::new(), output_w, output_h)
        .context("failed to initialize Wayland server")?;
    let mut wayland_state = wayland::WaylandState::new(Vec::new(), output_w, output_h);

    // Shared host DMA-BUF format→modifier map. Written by the wayland
    // backend's event thread during its initial roundtrip, read by the
    // client-facing dmabuf module to advertise formats that enable zero-copy.
    let host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>> =
        Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new()));
    wayland_state.host_dmabuf_formats = host_dmabuf_formats.clone();

    // Create frame channel: main thread → wayland backend (committed buffers).
    let (frame_tx, frame_rx) = std::sync::mpsc::channel::<wayland::protocols::CommittedBuffer>();
    wayland_state.frame_channel = Some(frame_tx);

    let socket_name = wayland_server.socket_name().to_string();
    info!(socket = %socket_name, "Wayland socket ready");

    // Determine the XWayland display number early so we can set both
    // WAYLAND_DISPLAY and DISPLAY before spawning any threads.
    let xwayland_display = find_free_x11_display()?;

    // Set environment variables for child processes.
    // SAFETY: No other threads are reading environment variables at this point —
    // the render thread, XWM thread, and child processes have not been spawned yet.
    unsafe { std::env::set_var("WAYLAND_DISPLAY", &socket_name) };
    unsafe { std::env::set_var("DISPLAY", &xwayland_display) };

    // --- Spawn render thread FIRST ---
    // The render thread starts the wayland backend event loop, which connects
    // to the host compositor and collects DMA-BUF format/modifier information
    // during its initial roundtrip. We must spawn it before XWayland so that
    // by the time XWayland connects and binds zwp_linux_dmabuf_v1, our
    // client-facing dmabuf module can advertise the real host formats instead
    // of the hardcoded fallback list. This enables zero-copy DMA-BUF forwarding.
    let (calloop_frame_tx, _frame_rx) = calloop::channel::channel::<FrameInfo>();
    let (_flip_tx, _flip_rx) = calloop::channel::channel::<u64>(); // VBlank timestamp.

    // Shared atomic for detected host display refresh rate (millihertz).
    // Written by the wayland backend's event thread, read by the main loop.
    // Ordering: Release on write, Relaxed on read — main loop polls periodically.
    let detected_refresh_mhz = Arc::new(AtomicU32::new(0));
    let detected_refresh_mhz_render = detected_refresh_mhz.clone();

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
                detected_refresh_mhz_render,
                host_dmabuf_formats_render,
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
        .spawn(move || {
            let result = retry_with_backoff("XWM", &RetryPolicy::DEFAULT, &RUNNING, || {
                wayland::xwayland::run_xwm(
                    &xwm_display,
                    &xwm_event_tx,
                    &xwm_cmd_rx,
                    output_w,
                    output_h,
                )
            });
            if let Err(e) = result {
                error!(?e, "XWM thread exiting after exhausting retries");
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
        // Sync frame pacer and FPS limiter to the host display refresh rate.
        update_refresh_rate(
            &detected_refresh_mhz,
            &mut last_detected_hz,
            &config,
            &mut pacer,
            &mut fps_limiter,
        );

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
fn render_thread_main(
    config: &Config,
    host_wayland_display: Option<String>,
    committed_frames: std::sync::mpsc::Receiver<wayland::protocols::CommittedBuffer>,
    detected_refresh_mhz: Arc<AtomicU32>,
    host_dmabuf_formats: Arc<parking_lot::Mutex<std::collections::HashMap<u32, Vec<u64>>>>,
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

    match config.backend {
        crate::config::BackendKind::Wayland => {
            let mut wayland_config = config.to_wayland_config();
            wayland_config.host_wayland_display = host_wayland_display;
            wayland_config.committed_frame_rx = Some(committed_frames);
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
        _ => {
            // TODO: Initialize DRM / headless backends.
        }
    }
    // TODO: Set up calloop on this thread for DRM page flip fd + frame channel.

    // Monitor the backend until shutdown.
    while RUNNING.load(Ordering::Relaxed) {
        if let Some(ref backend) = _backend
            && !backend.is_alive()
        {
            info!("wayland backend closed, initiating shutdown");
            RUNNING.store(false, Ordering::Relaxed);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    info!("render thread exited");
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
