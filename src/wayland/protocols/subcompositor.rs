//! `wl_subcompositor` and `wl_subsurface` dispatch.

use wayland_server::protocol::{
    wl_subcompositor::{self, WlSubcompositor},
    wl_subsurface::{self, WlSubsurface},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New};

use crate::wayland::WaylandState;

impl GlobalDispatch<WlSubcompositor, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlSubcompositor>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<WlSubcompositor, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlSubcompositor,
        request: wl_subcompositor::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_subcompositor::Request::GetSubsurface { id, .. } => {
                data_init.init(id, ());
            }
            wl_subcompositor::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<WlSubsurface, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlSubsurface,
        _request: wl_subsurface::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle SetPosition, PlaceAbove, PlaceBelow, SetSync, SetDesync, Destroy.
    }
}
