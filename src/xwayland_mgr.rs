//! XWayland server lifecycle management.
//!
//! Handles spawning, monitoring, and respawning XWayland instances.
//! Each instance gets a dedicated X11 display number and XWM thread.
//! The main thread uses [`XWaylandInstance`] to track process state
//! and communicate with the XWM via command channels.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Context;
use tracing::{error, info, warn};

use crate::focus::ServerFocusState;
use crate::wayland::{self, WaylandServer};

/// Per-XWayland server state managed by the main thread.
pub struct XWaylandInstance {
    /// X11 display string (e.g., ":1").
    pub display: String,
    /// XWayland child process.
    pub child: std::process::Child,
    /// Command sender to the XWM thread.
    pub cmd_tx: std::sync::mpsc::Sender<wayland::xwayland::XwmCommand>,
    /// XWM thread handle.
    pub thread: std::thread::JoinHandle<()>,
    /// Server index (0 = platform, 1+ = game).
    pub index: u32,
    /// Per-server focused app ID (XWM thread writes, main loop reads).
    pub focused_app_id: Arc<AtomicU32>,
    /// Per-server focused surface protocol ID (XWM thread writes, main loop reads).
    pub focused_wl_surface_id: Arc<AtomicU32>,
}

impl XWaylandInstance {
    /// Build a [`ServerFocusState`] view for the focus arbiter.
    pub fn focus_state(&self) -> ServerFocusState {
        ServerFocusState {
            index: self.index,
            focused_app_id: Arc::clone(&self.focused_app_id),
            focused_wl_surface_id: Arc::clone(&self.focused_wl_surface_id),
        }
    }
}

/// Find a free X11 display number and return the display string (e.g., ":1").
pub fn find_free_x11_display() -> anyhow::Result<String> {
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
pub fn spawn_xwayland(
    display_str: &str,
    wayland_socket: &str,
    wayland_server: &mut WaylandServer,
    wayland_state: &mut wayland::WaylandState,
    server_index: u32,
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
        if let Some(stream) = wayland_server.accept() {
            match wayland_server.insert_client(stream, wayland_state) {
                Ok(client_id) => {
                    // Tag this client as belonging to this XWayland server
                    // so the commit handler can disambiguate surfaces.
                    wayland_state
                        .xwayland_client_map
                        .insert(client_id, server_index);
                }
                Err(e) => {
                    warn!(?e, "failed to insert Wayland client during XWayland launch");
                }
            }
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

/// Check if XWayland exited and respawn it if necessary.
///
/// If XWayland crashed or was terminated, spawns a fresh instance.
/// The XWM thread's retry loop will re-establish the window manager
/// connection automatically.
pub fn monitor_xwayland(
    child: &mut std::process::Child,
    display: &str,
    socket: &str,
    server: &mut WaylandServer,
    state: &mut wayland::WaylandState,
    server_index: u32,
) {
    match child.try_wait() {
        Ok(Some(status)) => {
            warn!(?status, "XWayland exited, respawning");
            match spawn_xwayland(display, socket, server, state, server_index) {
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

// ─── Host readiness waits ───────────────────────────────────────────

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
pub fn wait_for_host_formats(
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
pub fn wait_for_host_configure(
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
