//! `wl_data_device_manager`, `wl_data_device`, `wl_data_source`,
//! and `wl_data_offer` dispatch.

use wayland_server::protocol::{
    wl_data_device::{self, WlDataDevice},
    wl_data_device_manager::{self, WlDataDeviceManager},
    wl_data_offer::WlDataOffer,
    wl_data_source::{self, WlDataSource},
};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New};

use crate::wayland::WaylandState;

impl GlobalDispatch<WlDataDeviceManager, ()> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WlDataDeviceManager>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<WlDataDeviceManager, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataDeviceManager,
        request: wl_data_device_manager::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_data_device_manager::Request::CreateDataSource { id } => {
                data_init.init(id, ());
            }
            wl_data_device_manager::Request::GetDataDevice { id, .. } => {
                data_init.init(id, ());
            }
            _ => {}
        }
    }
}

impl Dispatch<WlDataDevice, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataDevice,
        _request: wl_data_device::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle StartDrag, SetSelection, Release.
    }
}

impl Dispatch<WlDataSource, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataSource,
        _request: wl_data_source::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle Offer, Destroy, SetActions.
    }
}

impl Dispatch<WlDataOffer, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WlDataOffer,
        _request: wayland_server::protocol::wl_data_offer::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Handle Accept, Receive, Destroy, Finish, SetActions.
    }
}
