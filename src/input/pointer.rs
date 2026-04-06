//! Pointer input via libinput.
//!
//! Uses libinput for DPI-normalized pointer motion with automatic device
//! discovery and hotplug via udev. Follows gamescope's approach: consume
//! unaccelerated (but normalized) deltas and apply a flat sensitivity
//! multiplier configured by the user.
//!
//! This replaces the raw evdev reader. libinput normalizes motion deltas
//! across different mouse DPI values, so a 400 DPI mouse and a 1600 DPI
//! mouse produce the same cursor speed at the same sensitivity setting.

use std::ffi::CString;
use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;

use input::event::Event;
use input::event::pointer::{Axis, ButtonState, PointerScrollEvent};
use input::event::pointer::{PointerEvent as LibPointerEvent, PointerEventTrait};
use input::{Libinput, LibinputInterface};
use tracing::{info, warn};

use super::{ButtonEvent, ScrollEvent};

/// Events produced by the pointer monitor.
#[derive(Debug)]
pub enum PointerEvent {
    /// Relative motion delta (DPI-normalized, unaccelerated).
    Motion { dx: f64, dy: f64, time_ms: u32 },
    /// Button press/release.
    Button(ButtonEvent),
    /// Scroll wheel or finger scroll.
    Scroll(ScrollEvent),
}

/// libinput interface that opens/closes input devices via libc.
///
/// This works on all session managers: logind sets ACLs on device nodes
/// for the active session user, seatd relies on udev rules / `input`
/// group membership, and root always has access.
struct InputInterface;

impl LibinputInterface for InputInterface {
    fn open_restricted(&mut self, path: &Path, flags: i32) -> Result<OwnedFd, i32> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| libc::EINVAL)?;
        // SAFETY: c_path is a valid null-terminated string, flags are from libinput.
        let fd = unsafe { libc::open(c_path.as_ptr(), flags) };
        if fd < 0 {
            Err(std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::ENODEV))
        } else {
            // SAFETY: open() succeeded, fd is valid.
            Ok(unsafe { OwnedFd::from_raw_fd(fd) })
        }
    }

    fn close_restricted(&mut self, _fd: OwnedFd) {
        // OwnedFd closes on drop.
    }
}

/// Pointer input monitor backed by libinput.
///
/// Uses libinput's udev backend for automatic device discovery and
/// hotplug. Produces DPI-normalized unaccelerated deltas matching
/// gamescope's input model.
pub struct PointerMonitor {
    libinput: Libinput,
}

impl PointerMonitor {
    /// Create a new pointer monitor using libinput with udev discovery.
    ///
    /// Discovers all input devices on `seat0`. Pointer events are
    /// extracted; keyboard and other events are ignored (the keyboard
    /// monitor reads from evdev separately).
    pub fn new() -> anyhow::Result<Self> {
        let mut libinput = Libinput::new_with_udev(InputInterface);
        libinput
            .udev_assign_seat("seat0")
            .map_err(|()| anyhow::anyhow!("libinput: failed to assign seat"))?;
        info!("libinput pointer monitor initialized");
        Ok(Self { libinput })
    }

    /// Get the libinput fd for event loop polling.
    #[inline(always)]
    pub fn fd(&self) -> RawFd {
        self.libinput.as_raw_fd()
    }

    /// Suspend libinput (call on VT switch away).
    ///
    /// Closes all open devices. Call [`resume`] when the session
    /// becomes active again.
    pub fn suspend(&mut self) {
        self.libinput.suspend();
        info!("libinput pointer monitor suspended");
    }

    /// Resume libinput after suspend (call on VT switch back).
    ///
    /// Re-opens all devices via the `InputInterface`.
    pub fn resume(&mut self) {
        if self.libinput.resume().is_err() {
            warn!("libinput resume failed");
        } else {
            info!("libinput pointer monitor resumed");
        }
    }

    /// Dispatch pending libinput events and return pointer events.
    ///
    /// Non-blocking. Returns an empty vec if no events are pending.
    pub fn poll(&mut self) -> Vec<PointerEvent> {
        if let Err(e) = self.libinput.dispatch() {
            warn!(?e, "libinput dispatch error");
            return Vec::new();
        }

        let mut events = Vec::new();
        for event in &mut self.libinput {
            if let Event::Pointer(ptr_event) = event {
                Self::handle_pointer_event(ptr_event, &mut events);
            }
        }
        events
    }

    /// Convert a libinput pointer event into our event type.
    #[inline(always)]
    fn handle_pointer_event(event: LibPointerEvent, out: &mut Vec<PointerEvent>) {
        match event {
            LibPointerEvent::Motion(ev) => {
                out.push(PointerEvent::Motion {
                    dx: ev.dx_unaccelerated(),
                    dy: ev.dy_unaccelerated(),
                    time_ms: ev.time(),
                });
            }
            LibPointerEvent::Button(ev) => {
                out.push(PointerEvent::Button(ButtonEvent {
                    button: ev.button(),
                    pressed: ev.button_state() == ButtonState::Pressed,
                    time_usec: ev.time_usec(),
                }));
            }
            LibPointerEvent::ScrollWheel(ev) => {
                Self::push_scroll(&ev, out);
            }
            LibPointerEvent::ScrollFinger(ev) => {
                Self::push_scroll(&ev, out);
            }
            LibPointerEvent::ScrollContinuous(ev) => {
                Self::push_scroll(&ev, out);
            }
            _ => {}
        }
    }

    /// Extract scroll deltas from any scroll event type.
    #[inline(always)]
    fn push_scroll<E: PointerScrollEvent + PointerEventTrait>(ev: &E, out: &mut Vec<PointerEvent>) {
        let dx = if ev.has_axis(Axis::Horizontal) {
            ev.scroll_value(Axis::Horizontal)
        } else {
            0.0
        };
        let dy = if ev.has_axis(Axis::Vertical) {
            ev.scroll_value(Axis::Vertical)
        } else {
            0.0
        };
        out.push(PointerEvent::Scroll(ScrollEvent {
            dx,
            dy,
            time_usec: ev.time_usec(),
        }));
    }
}
