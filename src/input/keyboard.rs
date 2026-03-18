//! Raw evdev keyboard reader for VT switching and compositor hotkeys.
//!
//! Opens keyboard devices via the session (libseat) and reads raw
//! `input_event` structs from the evdev fds. Tracks modifier state
//! (Ctrl, Alt) and detects VT switch key combos (Ctrl+Alt+F1–F12).
//!
//! This does NOT use libinput — we read evdev directly because:
//! 1. VT switching must work even when libinput is not initialized.
//! 2. We need raw keycodes, not XKB-processed symbols.
//! 3. Minimal code — no need for a full input library just for hotkeys.

use std::collections::HashMap;
use std::os::unix::io::{AsRawFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};

use tracing::{debug, info, warn};

/// Linux `input_event` struct layout (evdev).
/// Matches `struct input_event` from `<linux/input.h>`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct InputEvent {
    tv_sec: libc::time_t,
    tv_usec: libc::suseconds_t,
    type_: u16,
    code: u16,
    value: i32,
}

// Event types from <linux/input-event-codes.h>.
const EV_KEY: u16 = 0x01;

// Key codes for modifiers.
const KEY_LEFTCTRL: u16 = 29;
const KEY_RIGHTCTRL: u16 = 97;
const KEY_LEFTALT: u16 = 56;
const KEY_RIGHTALT: u16 = 100;

// Function key codes (F1–F10 are sequential, F11–F12 are separate).
const KEY_F1: u16 = 59;
const KEY_F10: u16 = 68;
const KEY_F11: u16 = 87;
const KEY_F12: u16 = 88;

/// Map an evdev keycode to a VT number (1–12), or `None` if not an F-key.
#[inline(always)]
fn keycode_to_vt(code: u16) -> Option<i32> {
    match code {
        KEY_F1..=KEY_F10 => Some((code - KEY_F1 + 1) as i32),
        KEY_F11 => Some(11),
        KEY_F12 => Some(12),
        _ => None,
    }
}

/// Discover keyboard evdev devices via udev.
///
/// Scans the `input` subsystem for devices tagged as keyboards
/// (`ID_INPUT_KEYBOARD=1`) and returns their `/dev/input/event*` paths.
pub fn discover_keyboards() -> anyhow::Result<Vec<PathBuf>> {
    let mut enumerator =
        udev::Enumerator::new().map_err(|e| anyhow::anyhow!("udev enumerator: {e}"))?;
    enumerator
        .match_subsystem("input")
        .map_err(|e| anyhow::anyhow!("udev subsystem filter: {e}"))?;
    enumerator
        .match_property("ID_INPUT_KEYBOARD", "1")
        .map_err(|e| anyhow::anyhow!("udev property filter: {e}"))?;

    let devices = enumerator
        .scan_devices()
        .map_err(|e| anyhow::anyhow!("udev scan: {e}"))?;

    let mut paths = Vec::new();
    for device in devices {
        if let Some(devnode) = device.devnode() {
            // Only take /dev/input/event* nodes (skip /dev/input/mouse* etc.).
            if let Some(name) = devnode.file_name()
                && name.to_string_lossy().starts_with("event")
            {
                paths.push(devnode.to_path_buf());
            }
        }
    }

    info!(count = paths.len(), "discovered keyboard devices");
    for p in &paths {
        debug!(path = %p.display(), "keyboard device");
    }

    Ok(paths)
}

/// Set a file descriptor to non-blocking mode.
///
/// Required for keyboard fds so `dispatch()` doesn't block the main loop.
pub fn set_nonblock(fd: RawFd) -> anyhow::Result<()> {
    // SAFETY: fd is valid (just opened via session).
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(anyhow::anyhow!(
            "fcntl F_GETFL failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: Setting O_NONBLOCK on a valid fd.
    let ret = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if ret < 0 {
        return Err(anyhow::anyhow!(
            "fcntl F_SETFL failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// Action detected from keyboard input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Request VT switch to the given terminal number (1-based).
    SwitchVt(i32),
}

/// Check if a udev device is a keyboard with an `/dev/input/event*` node.
#[inline(always)]
fn is_keyboard_event_node(device: &udev::Device) -> bool {
    device.devnode().is_some_and(|p| {
        p.file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("event"))
    }) && device
        .property_value("ID_INPUT_KEYBOARD")
        .is_some_and(|v| v == "1")
}

/// Tracks modifier state and reads keyboard events from evdev devices.
///
/// Supports runtime hotplug: an optional udev monitor watches for
/// keyboard add/remove events so newly connected keyboards are
/// picked up automatically.
pub struct KeyboardMonitor {
    /// Open keyboard device fds. Maps raw fd → owned fd.
    devices: HashMap<RawFd, OwnedFd>,
    /// Reverse map: devnode path → raw fd, for removal on unplug.
    path_to_fd: HashMap<PathBuf, RawFd>,
    /// Whether left/right Ctrl is currently held.
    ctrl_held: bool,
    /// Whether left/right Alt is currently held.
    alt_held: bool,
    /// Per-modifier key tracking (to handle left+right independently).
    left_ctrl: bool,
    right_ctrl: bool,
    left_alt: bool,
    right_alt: bool,
    /// Read buffer — reused across calls to avoid allocation.
    buf: Vec<u8>,
    /// Udev monitor for keyboard hotplug events.
    udev_monitor: Option<udev::MonitorSocket>,
}

impl KeyboardMonitor {
    /// Create a new keyboard monitor (no devices opened yet).
    pub fn new() -> Self {
        // Start the udev monitor early so we don't miss hotplug events
        // between initial enumeration and the first dispatch_hotplug call.
        let udev_monitor = match udev::MonitorBuilder::new()
            .and_then(|b| b.match_subsystem("input"))
            .and_then(|b| b.listen())
        {
            Ok(socket) => {
                // Set the monitor fd non-blocking so dispatch_hotplug
                // never stalls the main loop.
                let raw = socket.as_raw_fd();
                if let Err(e) = set_nonblock(raw) {
                    warn!(?e, "failed to set udev monitor fd non-blocking");
                }
                info!("udev keyboard hotplug monitor started");
                Some(socket)
            }
            Err(e) => {
                warn!(
                    ?e,
                    "failed to create udev keyboard monitor — hotplug disabled"
                );
                None
            }
        };

        Self {
            devices: HashMap::new(),
            path_to_fd: HashMap::new(),
            ctrl_held: false,
            alt_held: false,
            left_ctrl: false,
            right_ctrl: false,
            left_alt: false,
            right_alt: false,
            // Pre-allocate for 64 events per read (~1.5KB).
            buf: vec![0u8; 64 * std::mem::size_of::<InputEvent>()],
            udev_monitor,
        }
    }

    /// Add a keyboard device by its owned fd and devnode path.
    ///
    /// The fd should have been opened via `Session::open_device()` on
    /// an evdev node like `/dev/input/event0`.
    pub fn add_device(&mut self, fd: OwnedFd, path: &Path) {
        let raw = fd.as_raw_fd();
        info!(fd = raw, path = %path.display(), "added keyboard device to monitor");
        self.path_to_fd.insert(path.to_path_buf(), raw);
        self.devices.insert(raw, fd);
    }

    /// Remove a keyboard device by its devnode path.
    ///
    /// Called on hotplug removal. Closes the fd via the session and
    /// drops it from tracking. No-op if the path is not tracked.
    fn remove_device(&mut self, path: &Path, session: &mut crate::backend::session::Session) {
        if let Some(raw) = self.path_to_fd.remove(path)
            && let Some(_fd) = self.devices.remove(&raw)
        {
            if let Err(e) = session.close_device(raw) {
                debug!(?e, fd = raw, "failed to close removed keyboard fd");
            }
            info!(fd = raw, path = %path.display(), "removed keyboard device");
        }
    }

    /// Replace all tracked devices with new ones.
    ///
    /// Used after a VT switch: logind revokes evdev fds via `EVIOCREVOKE`
    /// when the session goes inactive, so the old fds are permanently dead.
    /// The caller must re-open devices via `Session::open_device()` and
    /// pass the new `(path, fd)` pairs here.
    pub fn replace_devices(&mut self, devices: Vec<(PathBuf, OwnedFd)>) {
        // Drop old (revoked) fds.
        let old_count = self.devices.len();
        self.devices.clear();
        self.path_to_fd.clear();
        for (path, fd) in devices {
            let raw = fd.as_raw_fd();
            self.path_to_fd.insert(path, raw);
            self.devices.insert(raw, fd);
        }
        info!(
            old_count,
            new_count = self.devices.len(),
            "keyboard devices replaced after session restore"
        );
    }

    /// Reset all modifier tracking state.
    ///
    /// Called after VT switch restore because the Ctrl/Alt release events
    /// happened on another VT and were never seen by this monitor.
    pub fn reset_modifiers(&mut self) {
        self.ctrl_held = false;
        self.alt_held = false;
        self.left_ctrl = false;
        self.right_ctrl = false;
        self.left_alt = false;
        self.right_alt = false;
        debug!("keyboard modifier state reset");
    }

    /// Discover and open all keyboard devices via the session.
    ///
    /// Scans udev for keyboard devices, opens each through the session
    /// (which grants evdev access via logind/seatd), and sets them
    /// non-blocking. Devices that fail to open are logged and skipped.
    pub fn open_from_session(&mut self, session: &mut crate::backend::session::Session) {
        match discover_keyboards() {
            Ok(paths) => {
                for path in paths {
                    self.open_one(session, &path);
                }
            }
            Err(e) => {
                warn!(?e, "failed to discover keyboard devices");
            }
        }
    }

    /// Close all tracked fds via the session, then re-discover and re-open.
    ///
    /// Used after VT switch restore: logind revokes evdev fds via
    /// `EVIOCREVOKE` when the session goes inactive, making old fds
    /// permanently dead. This closes them, re-discovers keyboard nodes,
    /// opens fresh fds, and resets modifier state.
    pub fn reopen_after_vt_switch(&mut self, session: &mut crate::backend::session::Session) {
        // Close old (revoked) device fds via libseat.
        for old_fd in self.fds() {
            if let Err(e) = session.close_device(old_fd) {
                debug!(
                    ?e,
                    fd = old_fd,
                    "failed to close old keyboard fd (expected after revoke)"
                );
            }
        }

        // Re-discover and re-open.
        let mut new_devices = Vec::new();
        if let Ok(paths) = discover_keyboards() {
            for path in paths {
                match session.open_device(&path) {
                    Ok(fd) => {
                        let raw = fd.as_raw_fd();
                        if let Err(e) = set_nonblock(raw) {
                            warn!(
                                path = %path.display(), ?e,
                                "failed to set keyboard fd non-blocking"
                            );
                        }
                        new_devices.push((path, fd));
                    }
                    Err(e) => {
                        warn!(
                            path = %path.display(), ?e,
                            "failed to re-open keyboard device"
                        );
                    }
                }
            }
        }
        self.replace_devices(new_devices);
        self.reset_modifiers();
    }

    /// Poll the udev monitor for keyboard hotplug events.
    ///
    /// Opens newly connected keyboards and closes disconnected ones.
    /// Non-blocking — returns immediately if no events are pending.
    pub fn dispatch_hotplug(&mut self, session: &mut crate::backend::session::Session) {
        let Some(ref monitor) = self.udev_monitor else {
            return;
        };

        // Collect events into a local vec to avoid borrowing self.udev_monitor
        // while mutating self.devices.
        let mut to_add: Vec<PathBuf> = Vec::new();
        let mut to_remove: Vec<PathBuf> = Vec::new();

        for event in monitor.iter() {
            let device = event.device();
            let Some(devnode) = device.devnode() else {
                continue;
            };
            if !is_keyboard_event_node(&device) {
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

    /// Open a single keyboard device via the session.
    fn open_one(&mut self, session: &mut crate::backend::session::Session, path: &Path) {
        match session.open_device(path) {
            Ok(fd) => {
                let raw = fd.as_raw_fd();
                if let Err(e) = set_nonblock(raw) {
                    warn!(
                        path = %path.display(), ?e,
                        "failed to set keyboard fd non-blocking"
                    );
                }
                self.add_device(fd, path);
            }
            Err(e) => {
                warn!(
                    path = %path.display(), ?e,
                    "failed to open keyboard device"
                );
            }
        }
    }

    /// Get the raw fds for poll registration.
    pub fn fds(&self) -> Vec<RawFd> {
        self.devices.keys().copied().collect()
    }

    /// Poll for hotplug events, read keyboard input, and execute actions.
    ///
    /// Combines hotplug dispatch, event reading, and action handling
    /// into a single call. VT switch requests are forwarded directly
    /// to the session.
    pub fn poll(&mut self, session: &mut crate::backend::session::Session) {
        self.dispatch_hotplug(session);
        for action in self.dispatch() {
            match action {
                KeyAction::SwitchVt(vt) => {
                    if let Err(e) = session.switch_vt(vt) {
                        warn!(?e, vt, "VT switch failed");
                    }
                }
            }
        }
    }

    /// Read and process pending events from all keyboard devices.
    ///
    /// Returns any detected actions (e.g., VT switch requests).
    /// Non-blocking — returns an empty vec if no events are available.
    pub fn dispatch(&mut self) -> Vec<KeyAction> {
        let mut actions = Vec::new();
        let event_size = std::mem::size_of::<InputEvent>();
        let fds: Vec<RawFd> = self.devices.keys().copied().collect();

        for fd in fds {
            loop {
                // SAFETY: We read into a properly sized buffer and the fd
                // is valid (owned by us). Using O_NONBLOCK so this won't
                // block.
                let n = unsafe { libc::read(fd, self.buf.as_mut_ptr().cast(), self.buf.len()) };

                if n <= 0 {
                    // EAGAIN/EWOULDBLOCK = no more events, or error.
                    break;
                }

                let n = n as usize;
                let num_events = n / event_size;

                for i in 0..num_events {
                    let offset = i * event_size;
                    // SAFETY: We verified `offset + event_size <= n` and
                    // InputEvent is repr(C) with no padding issues.
                    let event: InputEvent =
                        unsafe { std::ptr::read_unaligned(self.buf.as_ptr().add(offset).cast()) };

                    if let Some(action) = self.process_event(&event) {
                        actions.push(action);
                    }
                }
            }
        }

        actions
    }

    /// Process a single evdev event. Returns an action if a hotkey was detected.
    #[inline(always)]
    fn process_event(&mut self, event: &InputEvent) -> Option<KeyAction> {
        if event.type_ != EV_KEY {
            return None;
        }

        let pressed = event.value == 1; // 1 = press, 0 = release, 2 = repeat

        // Update modifier tracking (press and release only, ignore repeat).
        if event.value != 2 {
            match event.code {
                KEY_LEFTCTRL => self.left_ctrl = pressed,
                KEY_RIGHTCTRL => self.right_ctrl = pressed,
                KEY_LEFTALT => self.left_alt = pressed,
                KEY_RIGHTALT => self.right_alt = pressed,
                _ => {}
            }
            self.ctrl_held = self.left_ctrl || self.right_ctrl;
            self.alt_held = self.left_alt || self.right_alt;
        }

        // Detect Ctrl+Alt+Fn on key press only.
        if pressed
            && self.ctrl_held
            && self.alt_held
            && let Some(vt) = keycode_to_vt(event.code)
        {
            debug!(vt, "VT switch hotkey detected");
            return Some(KeyAction::SwitchVt(vt));
        }

        None
    }
}

#[cfg(test)]
#[path = "keyboard_tests.rs"]
mod tests;
