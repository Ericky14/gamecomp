//! Raw evdev pointer reader for mice and touchpads.
//!
//! Opens pointer devices via the session (libseat) and reads raw
//! `input_event` structs from the evdev fds. Produces motion, button,
//! and scroll events for forwarding to Wayland clients.
//!
//! Supports runtime hotplug via udev monitor.

use std::collections::HashMap;
use std::os::unix::io::{AsRawFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

use super::keyboard::set_nonblock;
use super::{ButtonEvent, ScrollEvent};

// Event types from <linux/input-event-codes.h>.
const EV_KEY: u16 = 0x01;
const EV_REL: u16 = 0x02;

// Relative axis codes.
const REL_X: u16 = 0x00;
const REL_Y: u16 = 0x01;
const REL_WHEEL: u16 = 0x08;
const REL_HWHEEL: u16 = 0x06;

/// Linux `input_event` struct layout (evdev).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct InputEvent {
    tv_sec: libc::time_t,
    tv_usec: libc::suseconds_t,
    type_: u16,
    code: u16,
    value: i32,
}

/// Events produced by the pointer monitor.
#[derive(Debug)]
pub enum PointerEvent {
    /// Relative motion delta.
    Motion { dx: f64, dy: f64, time_ms: u32 },
    /// Button press/release.
    Button(ButtonEvent),
    /// Scroll wheel.
    Scroll(ScrollEvent),
}

/// Check if a udev device is a mouse/touchpad with an `/dev/input/event*` node.
#[inline(always)]
fn is_pointer_event_node(device: &udev::Device) -> bool {
    device.devnode().is_some_and(|p| {
        p.file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("event"))
    }) && (device
        .property_value("ID_INPUT_MOUSE")
        .is_some_and(|v| v == "1")
        || device
            .property_value("ID_INPUT_TOUCHPAD")
            .is_some_and(|v| v == "1"))
}

/// Discover pointer evdev devices (mice + touchpads) via udev.
pub fn discover_pointers() -> anyhow::Result<Vec<PathBuf>> {
    let mut enumerator =
        udev::Enumerator::new().map_err(|e| anyhow::anyhow!("udev enumerator: {e}"))?;
    enumerator
        .match_subsystem("input")
        .map_err(|e| anyhow::anyhow!("udev subsystem filter: {e}"))?;

    let devices = enumerator
        .scan_devices()
        .map_err(|e| anyhow::anyhow!("udev scan: {e}"))?;

    let mut paths = Vec::new();
    for device in devices {
        if is_pointer_event_node(&device)
            && let Some(devnode) = device.devnode()
        {
            paths.push(devnode.to_path_buf());
        }
    }

    info!(count = paths.len(), "discovered pointer devices");
    for p in &paths {
        debug!(path = %p.display(), "pointer device");
    }

    Ok(paths)
}

/// Tracks pointer devices and reads evdev events.
///
/// Supports runtime hotplug via udev monitor.
pub struct PointerMonitor {
    /// Open pointer device fds.
    devices: HashMap<RawFd, OwnedFd>,
    /// Reverse map: devnode path → raw fd.
    path_to_fd: HashMap<PathBuf, RawFd>,
    /// Read buffer — reused across calls.
    buf: Vec<u8>,
    /// Udev monitor for hotplug events.
    udev_monitor: Option<udev::MonitorSocket>,
}

impl PointerMonitor {
    /// Create a new pointer monitor (no devices opened yet).
    pub fn new() -> Self {
        let udev_monitor = match udev::MonitorBuilder::new()
            .and_then(|b| b.match_subsystem("input"))
            .and_then(|b| b.listen())
        {
            Ok(socket) => {
                let raw = socket.as_raw_fd();
                if let Err(e) = set_nonblock(raw) {
                    warn!(?e, "failed to set udev monitor fd non-blocking");
                }
                info!("udev pointer hotplug monitor started");
                Some(socket)
            }
            Err(e) => {
                warn!(
                    ?e,
                    "failed to create udev pointer monitor — hotplug disabled"
                );
                None
            }
        };

        Self {
            devices: HashMap::new(),
            path_to_fd: HashMap::new(),
            buf: vec![0u8; 64 * std::mem::size_of::<InputEvent>()],
            udev_monitor,
        }
    }

    /// Add a pointer device by its owned fd and devnode path.
    pub fn add_device(&mut self, fd: OwnedFd, path: &Path) {
        let raw = fd.as_raw_fd();
        info!(fd = raw, path = %path.display(), "added pointer device to monitor");
        self.path_to_fd.insert(path.to_path_buf(), raw);
        self.devices.insert(raw, fd);
    }

    /// Remove a pointer device by its devnode path.
    fn remove_device(&mut self, path: &Path, session: &mut crate::backend::session::Session) {
        if let Some(raw) = self.path_to_fd.remove(path)
            && let Some(_fd) = self.devices.remove(&raw)
        {
            if let Err(e) = session.close_device(raw) {
                debug!(?e, fd = raw, "failed to close removed pointer fd");
            }
            info!(fd = raw, path = %path.display(), "removed pointer device");
        }
    }

    /// Discover and open all pointer devices via the session.
    pub fn open_from_session(&mut self, session: &mut crate::backend::session::Session) {
        match discover_pointers() {
            Ok(paths) => {
                for path in paths {
                    self.open_one(session, &path);
                }
            }
            Err(e) => {
                warn!(?e, "failed to discover pointer devices");
            }
        }
    }

    /// Close all tracked fds, re-discover, and re-open.
    pub fn reopen_after_vt_switch(&mut self, session: &mut crate::backend::session::Session) {
        for old_fd in self.fds() {
            if let Err(e) = session.close_device(old_fd) {
                debug!(
                    ?e,
                    fd = old_fd,
                    "failed to close old pointer fd (expected after revoke)"
                );
            }
        }
        self.devices.clear();
        self.path_to_fd.clear();

        if let Ok(paths) = discover_pointers() {
            for path in paths {
                self.open_one(session, &path);
            }
        }
    }

    /// Poll udev for hotplug events.
    pub fn dispatch_hotplug(&mut self, session: &mut crate::backend::session::Session) {
        let Some(ref monitor) = self.udev_monitor else {
            return;
        };

        let mut to_add: Vec<PathBuf> = Vec::new();
        let mut to_remove: Vec<PathBuf> = Vec::new();

        for event in monitor.iter() {
            let device = event.device();
            let Some(devnode) = device.devnode() else {
                continue;
            };
            if !is_pointer_event_node(&device) {
                continue;
            }
            match event.event_type() {
                udev::EventType::Add => {
                    if !self.path_to_fd.contains_key(devnode) {
                        to_add.push(devnode.to_path_buf());
                    }
                }
                udev::EventType::Remove => {
                    to_remove.push(devnode.to_path_buf());
                }
                _ => {}
            }
        }

        for path in to_remove {
            self.remove_device(&path, session);
        }
        for path in to_add {
            self.open_one(session, &path);
        }
    }

    /// Open a single pointer device via the session.
    fn open_one(&mut self, session: &mut crate::backend::session::Session, path: &Path) {
        match session.open_device(path) {
            Ok(fd) => {
                let raw = fd.as_raw_fd();
                if let Err(e) = set_nonblock(raw) {
                    warn!(
                        path = %path.display(), ?e,
                        "failed to set pointer fd non-blocking"
                    );
                }
                self.add_device(fd, path);
            }
            Err(e) => {
                warn!(
                    path = %path.display(), ?e,
                    "failed to open pointer device"
                );
            }
        }
    }

    /// Get the raw fds.
    pub fn fds(&self) -> Vec<RawFd> {
        self.devices.keys().copied().collect()
    }

    /// Read and process pending events from all pointer devices.
    pub fn dispatch(&mut self) -> Vec<PointerEvent> {
        let mut events = Vec::new();
        let event_size = std::mem::size_of::<InputEvent>();
        let fds: Vec<RawFd> = self.devices.keys().copied().collect();

        for fd in fds {
            loop {
                // SAFETY: We read into a properly sized buffer and the fd
                // is valid (owned by us). Using O_NONBLOCK.
                let n = unsafe { libc::read(fd, self.buf.as_mut_ptr().cast(), self.buf.len()) };

                if n <= 0 {
                    break;
                }

                let n = n as usize;
                let num_events = n / event_size;

                for i in 0..num_events {
                    let offset = i * event_size;
                    // SAFETY: Verified bounds, InputEvent is repr(C).
                    let ev: InputEvent =
                        unsafe { std::ptr::read_unaligned(self.buf.as_ptr().add(offset).cast()) };

                    let time_ms = (ev.tv_sec as u32)
                        .wrapping_mul(1000)
                        .wrapping_add((ev.tv_usec as u32) / 1000);

                    match ev.type_ {
                        EV_REL => match ev.code {
                            REL_X => {
                                events.push(PointerEvent::Motion {
                                    dx: ev.value as f64,
                                    dy: 0.0,
                                    time_ms,
                                });
                            }
                            REL_Y => {
                                events.push(PointerEvent::Motion {
                                    dx: 0.0,
                                    dy: ev.value as f64,
                                    time_ms,
                                });
                            }
                            REL_WHEEL => {
                                events.push(PointerEvent::Scroll(ScrollEvent {
                                    dx: 0.0,
                                    // Wheel up = negative scroll in Wayland convention.
                                    dy: -(ev.value as f64) * 15.0,
                                    time_usec: ev.tv_sec as u64 * 1_000_000 + ev.tv_usec as u64,
                                }));
                            }
                            REL_HWHEEL => {
                                events.push(PointerEvent::Scroll(ScrollEvent {
                                    dx: (ev.value as f64) * 15.0,
                                    dy: 0.0,
                                    time_usec: ev.tv_sec as u64 * 1_000_000 + ev.tv_usec as u64,
                                }));
                            }
                            _ => {}
                        },
                        EV_KEY => {
                            // Mouse buttons are in the BTN_MOUSE range (0x110–0x11f).
                            if ev.code >= 0x110 && ev.code <= 0x11f {
                                events.push(PointerEvent::Button(ButtonEvent {
                                    button: ev.code as u32,
                                    pressed: ev.value == 1,
                                    time_usec: ev.tv_sec as u64 * 1_000_000 + ev.tv_usec as u64,
                                }));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        events
    }

    /// Poll hotplug and dispatch events. Returns pointer events for the main loop.
    pub fn poll(&mut self, session: &mut crate::backend::session::Session) -> Vec<PointerEvent> {
        self.dispatch_hotplug(session);
        self.dispatch()
    }
}
