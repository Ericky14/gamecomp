//! `wl_seat`, `wl_pointer`, `wl_keyboard`, and `wl_touch` dispatch.

use std::os::unix::io::{AsFd, AsRawFd};

use wayland_server::protocol::{
    wl_keyboard::{self, WlKeyboard},
    wl_pointer::{self, WlPointer},
    wl_seat::{self, WlSeat},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource};

use super::SurfaceData;
use crate::wayland::WaylandState;

impl GlobalDispatch<WlSeat, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlSeat>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let seat = data_init.init(resource, ());
        seat.capabilities(wl_seat::Capability::Pointer | wl_seat::Capability::Keyboard);
        seat.name("seat0".into());
    }
}

impl Dispatch<WlSeat, ()> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _seat: &WlSeat,
        request: wl_seat::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_seat::Request::GetPointer { id } => {
                let ptr = data_init.init(id, ());
                state.pointers.push(ptr);
            }
            wl_seat::Request::GetKeyboard { id } => {
                let kb = data_init.init(id, ());
                // XWayland requires a keymap event before it will process
                // keyboard input and initialize properly.
                let keymap_str = b"xkb_keymap {\n\
                    \txkb_keycodes  { include \"evdev+aliases(qwerty)\" };\n\
                    \txkb_types     { include \"complete\" };\n\
                    \txkb_compat    { include \"complete\" };\n\
                    \txkb_symbols   { include \"pc+us\" };\n\
                    \txkb_geometry  { include \"pc(pc105)\" };\n\
                    };\n\0";
                let size = keymap_str.len();
                let name = c"gamecomp-keymap";
                if let Ok(fd) = rustix::fs::memfd_create(name, rustix::fs::MemfdFlags::CLOEXEC) {
                    let _ = rustix::fs::ftruncate(&fd, size as u64);
                    // SAFETY: Valid fd and size, no concurrent access.
                    let ptr = unsafe {
                        libc::mmap(
                            std::ptr::null_mut(),
                            size,
                            libc::PROT_READ | libc::PROT_WRITE,
                            libc::MAP_SHARED,
                            fd.as_raw_fd(),
                            0,
                        )
                    };
                    if ptr != libc::MAP_FAILED {
                        // SAFETY: ptr valid for `size` bytes.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                keymap_str.as_ptr(),
                                ptr as *mut u8,
                                size,
                            );
                            libc::munmap(ptr, size);
                        }
                        kb.keymap(wl_keyboard::KeymapFormat::XkbV1, fd.as_fd(), size as u32);
                    }
                }
                // XWayland requires an initial modifiers event after the keymap.
                let serial = state.next_serial();
                kb.modifiers(serial, 0, 0, 0, 0);
                // GTK4/Flutter need repeat_info to initialise keyboard handling;
                // without it GTK falls back to GSettings and hits a fatal GLib error.
                kb.repeat_info(25, 600);
                state.keyboards.push(kb);
            }
            wl_seat::Request::GetTouch { id } => {
                data_init.init(id, ());
            }
            wl_seat::Request::Release => {}
            _ => {}
        }
    }
}

impl Dispatch<WlPointer, ()> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WlPointer,
        request: wl_pointer::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_pointer::Request::SetCursor {
                surface,
                hotspot_x,
                hotspot_y,
                ..
            } => {
                if let Some(ref s) = surface {
                    if let Some(sd) = s.data::<SurfaceData>() {
                        sd.is_cursor
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        sd.hotspot_x
                            .store(hotspot_x, std::sync::atomic::Ordering::Relaxed);
                        sd.hotspot_y
                            .store(hotspot_y, std::sync::atomic::Ordering::Relaxed);
                    }
                } else {
                    // surface=None means hide cursor.
                    if let Some(ref tx) = state.cursor_tx {
                        let _ = tx.send(crate::backend::wayland::CursorUpdate::Hide);
                    }
                }
            }
            wl_pointer::Request::Release => {}
            _ => {}
        }
    }
}

impl Dispatch<WlKeyboard, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlKeyboard,
        _request: wl_keyboard::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle Release.
    }
}

impl Dispatch<wayland_server::protocol::wl_touch::WlTouch, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &wayland_server::protocol::wl_touch::WlTouch,
        _request: wayland_server::protocol::wl_touch::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle Release.
    }
}
