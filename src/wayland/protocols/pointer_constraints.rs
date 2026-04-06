//! `zwp_pointer_constraints_v1` protocol implementation.
//!
//! Allows clients to request pointer lock (disable cursor motion) or
//! pointer confinement (restrict cursor to a region). In a single-app
//! fullscreen compositor like gamecomp, lock requests are granted
//! immediately and the hardware cursor is hidden. Confinement is
//! accepted but treated as a no-op since the client already owns the
//! entire display.

use tracing::debug;
use wayland_protocols::wp::pointer_constraints::zv1::server::{
    zwp_confined_pointer_v1::{self, ZwpConfinedPointerV1},
    zwp_locked_pointer_v1::{self, ZwpLockedPointerV1},
    zwp_pointer_constraints_v1::{self, ZwpPointerConstraintsV1},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New};

use crate::wayland::WaylandState;

// ─── Manager global ─────────────────────────────────────────────────

impl GlobalDispatch<ZwpPointerConstraintsV1, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwpPointerConstraintsV1>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<ZwpPointerConstraintsV1, ()> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &ZwpPointerConstraintsV1,
        request: zwp_pointer_constraints_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_pointer_constraints_v1::Request::LockPointer { id, lifetime, .. } => {
                let locked_ptr = data_init.init(id, ());
                debug!(?lifetime, "client requested pointer lock");

                // Grant immediately — single-app compositor, client owns
                // the entire display.
                state.lock_pointer(locked_ptr);
            }
            zwp_pointer_constraints_v1::Request::ConfinePointer { id, .. } => {
                let confined_ptr = data_init.init(id, ());
                debug!("client requested pointer confinement (no-op in fullscreen)");
                // Immediately confirm confinement. The client already
                // occupies the full display so there is nothing to confine.
                confined_ptr.confined();
            }
            zwp_pointer_constraints_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

// ─── Locked pointer ─────────────────────────────────────────────────

impl Dispatch<ZwpLockedPointerV1, ()> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwpLockedPointerV1,
        request: zwp_locked_pointer_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_locked_pointer_v1::Request::Destroy => {
                debug!("client destroyed locked pointer — unlocking");
                state.unlock_pointer(resource);
            }
            zwp_locked_pointer_v1::Request::SetCursorPositionHint { .. } => {
                // Ignored — we don't warp the cursor on unlock.
            }
            zwp_locked_pointer_v1::Request::SetRegion { .. } => {
                // Ignored — the entire surface is always valid.
            }
            _ => {}
        }
    }
}

// ─── Confined pointer ───────────────────────────────────────────────

impl Dispatch<ZwpConfinedPointerV1, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZwpConfinedPointerV1,
        request: zwp_confined_pointer_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_confined_pointer_v1::Request::Destroy => {
                debug!("client destroyed confined pointer");
            }
            zwp_confined_pointer_v1::Request::SetRegion { .. } => {
                // Ignored — fullscreen confinement means the entire output.
            }
            _ => {}
        }
    }
}
