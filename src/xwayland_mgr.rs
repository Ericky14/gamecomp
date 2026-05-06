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

/// Maximum number of consecutive respawn attempts before giving up.
const MAX_RESPAWNS: u32 = 10;
/// Initial respawn backoff delay.
const RESPAWN_BACKOFF_BASE: std::time::Duration = std::time::Duration::from_millis(100);

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
    /// Consecutive respawn failures (reset on successful stable run).
    pub respawn_failures: u32,
    /// Whether this server has permanently failed and should not be respawned.
    pub permanently_failed: bool,
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

/// Find a free X11 display number and return the display string (e.g., ":0").
///
/// A display is considered free if neither its socket
/// (`/tmp/.X11-unix/X<N>`) nor its lock file (`/tmp/.X<N>-lock`) exists.
/// Stale lock files (whose PID is no longer running) are cleaned up so
/// display numbers can be reused across gamecomp restarts.
///
/// `exclude` lists display strings already allocated by this compositor
/// instance (e.g., `[":2"]`), preventing TOCTOU races between
/// sequential server launches.
pub fn find_free_x11_display(exclude: &[String]) -> anyhow::Result<String> {
    let display_num = (0..64)
        .find(|n| {
            // Skip displays already allocated to our own servers.
            let candidate = format!(":{n}");
            if exclude.contains(&candidate) {
                return false;
            }
            let socket = format!("/tmp/.X11-unix/X{n}");
            let lock = format!("/tmp/.X{n}-lock");
            let socket_exists = std::path::Path::new(&socket).exists();
            let lock_exists = std::path::Path::new(&lock).exists();

            if !socket_exists && !lock_exists {
                return true;
            }

            // Check if the lock holder is still alive.
            if lock_exists
                && let Ok(contents) = std::fs::read_to_string(&lock)
                && let Ok(pid) = contents.trim().parse::<i32>()
            {
                // SAFETY: kill(pid, 0) only probes process existence.
                let ret = unsafe { libc::kill(pid, 0) };
                if ret == 0 {
                    return false; // Lock holder alive — display in use.
                }
                // kill() returned -1. Check errno: EPERM means the process
                // exists but is owned by another user (e.g., display manager).
                // Only ESRCH means the process is truly dead.
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EPERM) {
                    return false; // Process alive, different user — display in use.
                }
            }
            // Lock holder dead or unreadable — clean up stale files.
            if lock_exists {
                let _ = std::fs::remove_file(&lock);
            }
            if socket_exists {
                let _ = std::fs::remove_file(&socket);
            }
            // Verify cleanup succeeded. On sticky-bit directories like /tmp,
            // only the file owner (or root) can delete. If the socket or lock
            // still exists after our remove attempt, we cannot use this display.
            if std::path::Path::new(&socket).exists() || std::path::Path::new(&lock).exists() {
                return false;
            }
            true
        })
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

    // Remove stale client map entries for this server index. On respawn,
    // the old XWayland's client ID may still be in the map; if wayland-server
    // recycles that ID for a new client, it would incorrectly
    // appear as belonging to this XWayland server.
    wayland_state
        .xwayland_client_map
        .retain(|_, idx| *idx != server_index);

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
    instance: &mut XWaylandInstance,
    socket: &str,
    server: &mut WaylandServer,
    state: &mut wayland::WaylandState,
) {
    if instance.permanently_failed {
        return;
    }

    match instance.child.try_wait() {
        Ok(Some(status)) => {
            instance.respawn_failures += 1;

            if instance.respawn_failures > MAX_RESPAWNS {
                error!(
                    server_index = instance.index,
                    display = %instance.display,
                    attempts = instance.respawn_failures,
                    "XWayland exceeded max respawn attempts, giving up"
                );
                instance.permanently_failed = true;
                return;
            }

            let backoff = RESPAWN_BACKOFF_BASE * instance.respawn_failures;
            warn!(
                ?status,
                server_index = instance.index,
                attempt = instance.respawn_failures,
                max = MAX_RESPAWNS,
                backoff_ms = backoff.as_millis() as u64,
                "XWayland exited, respawning"
            );
            std::thread::sleep(backoff);

            // Clean up stale socket/lock files before respawning.
            let display_num = instance.display.trim_start_matches(':');
            let socket_path = format!("/tmp/.X11-unix/X{display_num}");
            let lock_path = format!("/tmp/.X{display_num}-lock");
            if std::path::Path::new(&socket_path).exists() {
                let _ = std::fs::remove_file(&socket_path);
            }
            if std::path::Path::new(&lock_path).exists() {
                let _ = std::fs::remove_file(&lock_path);
            }

            match spawn_xwayland(&instance.display, socket, server, state, instance.index) {
                Ok(new_child) => {
                    instance.child = new_child;
                    info!(
                        server_index = instance.index,
                        display = %instance.display,
                        "XWayland respawned successfully"
                    );
                }
                Err(e) => {
                    error!(
                        ?e,
                        server_index = instance.index,
                        "failed to respawn XWayland"
                    );
                }
            }
        }
        Ok(None) => {
            // Still running — reset failure counter so transient crashes
            // don't accumulate across long healthy stretches.
            if instance.respawn_failures > 0 {
                instance.respawn_failures = 0;
            }
        }
        Err(e) => {
            warn!(?e, "error checking XWayland status");
        }
    }
}

/// Gracefully terminate an XWayland child process.
///
/// Sends SIGTERM first so XWayland can clean up its lock files and
/// `/tmp/.X11-unix/X<N>` sockets. Falls back to SIGKILL after 1 second.
/// Removes the socket file as a safety net in case XWayland didn't
/// clean it up (e.g., if SIGKILL was required).
pub fn terminate_xwayland(mut child: std::process::Child, display: &str) {
    let pid = child.id() as i32;
    // SAFETY: Sending a signal to a known child PID.
    unsafe { libc::kill(pid, libc::SIGTERM) };

    // Wait up to 1 s for graceful exit.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    warn!(pid, "XWayland did not exit after SIGTERM, sending SIGKILL");
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => break,
        }
    }

    // Clean up the socket file in case XWayland didn't remove it
    // (happens when SIGKILL is used or XWayland crashes).
    let display_num = display.trim_start_matches(':');
    let socket_path = format!("/tmp/.X11-unix/X{display_num}");
    if std::path::Path::new(&socket_path).exists() {
        let _ = std::fs::remove_file(&socket_path);
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
