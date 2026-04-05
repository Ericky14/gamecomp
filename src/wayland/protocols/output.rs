//! `wl_output` dispatch.

use wayland_server::protocol::wl_output::{self, WlOutput};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New};

use crate::wayland::WaylandState;

/// Per-global data for wl_output (carries resolution info).
#[derive(Debug, Clone)]
pub struct OutputData {
    pub width: u32,
    pub height: u32,
}

impl GlobalDispatch<WlOutput, OutputData> for WaylandState {
    fn bind(
        state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlOutput>,
        _data: &OutputData,
        data_init: &mut DataInit<'_, Self>,
    ) {
        let output = data_init.init(resource, ());
        // Use current runtime resolution, not the stale global data from
        // creation time. The DRM lease backend may have updated the output
        // resolution after the global was registered.
        let width = state.output_width;
        let height = state.output_height;
        output.geometry(
            0,
            0,
            0,
            0,
            wl_output::Subpixel::Unknown,
            "gamecomp".into(),
            "gamecomp".into(),
            wl_output::Transform::Normal,
        );
        output.mode(
            wl_output::Mode::Current | wl_output::Mode::Preferred,
            width as i32,
            height as i32,
            60_000, // 60 Hz in mHz
        );
        output.scale(1);
        output.done();
        state.bound_outputs.push(output);
    }
}

impl Dispatch<WlOutput, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlOutput,
        _request: wl_output::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle Release.
    }
}
