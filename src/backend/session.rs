//! Seat session management via libseat.
//!
//! Provides GPU device access by opening a session with the system's
//! seat manager (logind, seatd, etc.). The session grants permission
//! to open DRM and input devices, and notifies us of VT switches so
//! we can pause/resume hardware access.

use std::collections::HashMap;
use std::os::unix::io::{AsFd, AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, bail};
use libseat::{Seat, SeatEvent};
use tracing::{debug, info, warn};

/// A managed session that owns the libseat connection.
///
/// Tracks opened devices and session active state. The session must be
/// polled periodically (via [`Session::dispatch`]) to process seat events
/// like VT switches.
pub struct Session {
    seat: Seat,
    /// Whether the session is currently active (not VT-switched away).
    /// Ordering: Release on write (callback), Acquire on read (render thread).
    active: Arc<AtomicBool>,
    /// Seat name (e.g., "seat0").
    seat_name: String,
    /// Open devices tracked for cleanup. Maps raw fd → libseat Device.
    devices: HashMap<RawFd, libseat::Device>,
}

impl Session {
    /// Open a new seat session.
    ///
    /// Blocks briefly while libseat negotiates with the session manager.
    /// Returns an error if no seat is available (e.g., not running from
    /// a login session or seatd is not running).
    pub fn open() -> anyhow::Result<Self> {
        let active = Arc::new(AtomicBool::new(false));
        let active_cb = active.clone();

        let mut seat = Seat::open(move |seat_ref, event| {
            match event {
                SeatEvent::Enable => {
                    // Ordering: Release so the render thread's Acquire sees it.
                    active_cb.store(true, Ordering::Release);
                    info!("session enabled");
                }
                SeatEvent::Disable => {
                    active_cb.store(false, Ordering::Release);
                    info!("session disabled (VT switch away)");
                    // Must acknowledge disable or the session manager hangs.
                    if let Err(e) = seat_ref.disable() {
                        warn!(?e, "failed to acknowledge session disable");
                    }
                }
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to open seat session: {e}"))?;

        // Dispatch immediately — the Enable event may already be queued.
        seat.dispatch(0)
            .map_err(|e| anyhow::anyhow!("initial seat dispatch failed: {e}"))?;

        let seat_name = seat.name().to_owned();
        let is_active = active.load(Ordering::Acquire);
        info!(seat = %seat_name, active = is_active, "seat session opened");

        if !is_active {
            bail!("seat session opened but not active — is another session in the foreground?");
        }

        Ok(Self {
            seat,
            active,
            seat_name,
            devices: HashMap::new(),
        })
    }

    /// Open a device (DRM or input) through the session.
    ///
    /// The session manager grants access to the device fd. The returned
    /// [`OwnedFd`] is valid for the lifetime of the session (until paused
    /// or the device is closed).
    pub fn open_device(&mut self, path: &Path) -> anyhow::Result<OwnedFd> {
        let device = self
            .seat
            .open_device(&path)
            .map_err(|e| anyhow::anyhow!("failed to open device {}: {e}", path.display()))?;

        let raw_fd = device.as_fd().as_raw_fd();
        // SAFETY: libseat::Device does not close the fd on drop — we take
        // ownership via OwnedFd. The raw fd is valid because open_device
        // succeeded.
        let owned = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        info!(path = %path.display(), fd = raw_fd, "opened device via session");
        self.devices.insert(raw_fd, device);

        Ok(owned)
    }

    /// Close a previously opened device.
    pub fn close_device(&mut self, raw_fd: RawFd) -> anyhow::Result<()> {
        let device = self
            .devices
            .remove(&raw_fd)
            .context("device not tracked by session")?;
        self.seat
            .close_device(device)
            .map_err(|e| anyhow::anyhow!("failed to close device: {e}"))?;
        debug!(fd = raw_fd, "closed device via session");
        Ok(())
    }

    /// Whether the session is currently active (not VT-switched away).
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        // Ordering: Acquire to pair with the callback's Release store.
        self.active.load(Ordering::Acquire)
    }

    /// Get a clone of the active flag for use in other threads.
    pub fn active_flag(&self) -> Arc<AtomicBool> {
        self.active.clone()
    }

    /// Seat name (e.g., "seat0").
    pub fn seat_name(&self) -> &str {
        &self.seat_name
    }

    /// Dispatch pending seat events (VT switch notifications).
    ///
    /// Non-blocking (timeout = 0). Should be called from the event loop
    /// whenever the seat fd is readable.
    pub fn dispatch(&mut self) -> anyhow::Result<()> {
        self.seat
            .dispatch(0)
            .map_err(|e| anyhow::anyhow!("seat dispatch failed: {e}"))?;
        Ok(())
    }

    /// Get the seat's pollable fd for event loop registration.
    ///
    /// When this fd is readable, call [`Session::dispatch`].
    pub fn seat_fd(&mut self) -> anyhow::Result<RawFd> {
        use std::os::unix::io::AsRawFd;
        let fd = self
            .seat
            .get_fd()
            .map_err(|e| anyhow::anyhow!("failed to get seat fd: {e}"))?;
        Ok(fd.as_raw_fd())
    }

    /// Request a VT switch to the given session number.
    pub fn switch_vt(&mut self, vt: i32) -> anyhow::Result<()> {
        self.seat
            .switch_session(vt)
            .map_err(|e| anyhow::anyhow!("VT switch to {vt} failed: {e}"))?;
        info!(vt, "requested VT switch");
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Close all tracked devices before the seat is dropped.
        let fds: Vec<RawFd> = self.devices.keys().copied().collect();
        for fd in fds {
            if let Some(device) = self.devices.remove(&fd) {
                let _ = self.seat.close_device(device);
                debug!(fd, "closed device on session drop");
            }
        }
    }
}
