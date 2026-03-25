//! `xdg_wm_base`, `xdg_surface`, `xdg_toplevel`, `xdg_positioner`,
//! and `xdg_popup` dispatch.

use tracing::debug;
use wayland_protocols::xdg::shell::server::{
    xdg_popup::{self, XdgPopup},
    xdg_positioner::{self, XdgPositioner},
    xdg_surface::{self, XdgSurface},
    xdg_toplevel::{self, XdgToplevel},
    xdg_wm_base::{self, XdgWmBase},
};
use wayland_server::protocol::wl_surface::WlSurface;
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New};

use crate::wayland::WaylandState;

/// Per-xdg_surface data: remembers the associated `wl_surface` so that
/// `get_toplevel` can register the correct surface for focus enter.
pub struct XdgSurfaceData {
    pub wl_surface: WlSurface,
}

/// Per-xdg_toplevel data: remembers the parent `xdg_surface` so we can
/// send re-configure events when the output resolution changes.
pub struct XdgToplevelData {
    pub xdg_surface: XdgSurface,
}

/// Build the `Activated | Fullscreen` states byte array for
/// `xdg_toplevel.configure`.
pub fn activated_fullscreen_states() -> Vec<u8> {
    let mut states: Vec<u8> = Vec::new();
    states.extend_from_slice(&(xdg_toplevel::State::Activated as u32).to_ne_bytes());
    states.extend_from_slice(&(xdg_toplevel::State::Fullscreen as u32).to_ne_bytes());
    states
}

impl GlobalDispatch<XdgWmBase, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<XdgWmBase>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<XdgWmBase, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _wm_base: &XdgWmBase,
        request: xdg_wm_base::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_wm_base::Request::GetXdgSurface { id, surface } => {
                data_init.init(
                    id,
                    XdgSurfaceData {
                        wl_surface: surface,
                    },
                );
            }
            xdg_wm_base::Request::CreatePositioner { id } => {
                data_init.init(id, ());
            }
            xdg_wm_base::Request::Pong { .. } => {}
            xdg_wm_base::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<XdgSurface, XdgSurfaceData> for WaylandState {
    fn request(
        state: &mut Self,
        _client: &Client,
        xdg_surface: &XdgSurface,
        request: xdg_surface::Request,
        data: &XdgSurfaceData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_surface::Request::GetToplevel { id } => {
                let toplevel = data_init.init(
                    id,
                    XdgToplevelData {
                        xdg_surface: xdg_surface.clone(),
                    },
                );
                // Register the underlying wl_surface as a toplevel surface
                // so focus enter goes to the correct surface, not a cursor
                // or subsurface.
                state.toplevel_surfaces.push(data.wl_surface.clone());
                // Track the toplevel for re-configure on resize.
                state.toplevels.push(toplevel.clone());
                // Send initial configure with Activated so clients accept input.
                let (w, h) = state.output_resolution();
                let states = crate::wayland::protocols::xdg_shell::activated_fullscreen_states();
                toplevel.configure(w as i32, h as i32, states);
                xdg_surface.configure(state.next_serial());
                debug!("xdg_toplevel: configured {}x{}", w, h);
            }
            xdg_surface::Request::GetPopup { id, .. } => {
                data_init.init(id, ());
            }
            xdg_surface::Request::AckConfigure { .. } => {}
            xdg_surface::Request::SetWindowGeometry { .. } => {}
            xdg_surface::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<XdgToplevel, XdgToplevelData> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _toplevel: &XdgToplevel,
        request: xdg_toplevel::Request,
        _data: &XdgToplevelData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            xdg_toplevel::Request::SetTitle { title } => {
                debug!(title, "xdg_toplevel: set_title");
            }
            xdg_toplevel::Request::SetAppId { app_id } => {
                debug!(app_id, "xdg_toplevel: set_app_id");
            }
            xdg_toplevel::Request::SetFullscreen { .. } => {}
            xdg_toplevel::Request::UnsetFullscreen => {}
            xdg_toplevel::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<XdgPositioner, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &XdgPositioner,
        _request: xdg_positioner::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle SetSize, SetAnchorRect, SetOffset, etc.
    }
}

impl Dispatch<XdgPopup, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &XdgPopup,
        _request: xdg_popup::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle Grab, Reposition, Destroy.
    }
}
