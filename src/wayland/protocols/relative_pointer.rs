//! `zwp_relative_pointer_v1` protocol implementation.
//!
//! Provides unaccelerated relative pointer motion events to clients
//! that request them. These events are emitted alongside (or instead of)
//! `wl_pointer.motion` events — critically, they continue to flow even
//! when the pointer is locked (no absolute motion).

use tracing::debug;
use wayland_protocols::wp::relative_pointer::zv1::server::{
    zwp_relative_pointer_manager_v1::{self, ZwpRelativePointerManagerV1},
    zwp_relative_pointer_v1::{self, ZwpRelativePointerV1},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New};

use crate::wayland::WaylandState;

// ─── Manager global ─────────────────────────────────────────────────

impl GlobalDispatch<ZwpRelativePointerManagerV1, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwpRelativePointerManagerV1>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<ZwpRelativePointerManagerV1, ()> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &ZwpRelativePointerManagerV1,
        request: zwp_relative_pointer_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_relative_pointer_manager_v1::Request::GetRelativePointer { id, .. } => {
                let rel_ptr = data_init.init(id, ());
                debug!("client created relative pointer");
                state.relative_pointers.push(rel_ptr);
            }
            zwp_relative_pointer_manager_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

// ─── Relative pointer object ────────────────────────────────────────

impl Dispatch<ZwpRelativePointerV1, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZwpRelativePointerV1,
        request: zwp_relative_pointer_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        if let zwp_relative_pointer_v1::Request::Destroy = request {
            debug!("client destroyed relative pointer");
        }
    }
}
