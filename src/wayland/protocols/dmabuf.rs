//! `zwp_linux_dmabuf_v1` and `zwp_linux_buffer_params_v1` dispatch,
//! plus DMA-BUF buffer infrastructure.

use parking_lot::Mutex;
use std::os::unix::io::OwnedFd;

use tracing::{info, warn, trace};
use wayland_protocols::wp::linux_dmabuf::zv1::server::{
    zwp_linux_buffer_params_v1::{self, ZwpLinuxBufferParamsV1},
    zwp_linux_dmabuf_v1::{self, ZwpLinuxDmabufV1},
};
use wayland_server::protocol::wl_buffer::WlBuffer;
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New};

use super::BufferData;
use crate::wayland::WaylandState;

// ─── DMA-BUF buffer infrastructure ─────────────────────────────────

/// Per-plane metadata for a DMA-BUF import.
pub struct DmaBufPlaneInfo {
    /// File descriptor for this plane's DMA-BUF.
    pub fd: OwnedFd,
    /// Plane index (0–3).
    pub plane_idx: u32,
    /// Byte offset into the GEM object.
    pub offset: u32,
    /// Row stride in bytes.
    pub stride: u32,
    /// DRM format modifier.
    pub modifier: u64,
}

/// DMA-BUF buffer data stored on `WlBuffer`.
pub struct DmaBufBufferData {
    /// Per-plane info (sorted by plane index).
    pub planes: Vec<DmaBufPlaneInfo>,
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
    /// DRM fourcc format code.
    pub format: u32,
    /// DRM modifier (all planes share one modifier).
    pub modifier: u64,
}

/// User data on `ZwpLinuxBufferParamsV1` — collects planes during creation.
pub struct DmaBufParamsData {
    /// Collected planes (protected for interior mutability).
    planes: Mutex<Vec<DmaBufPlaneInfo>>,
    /// Whether Create/CreateImmed has been called (params are single-use).
    used: Mutex<bool>,
}

/// DMA-BUF format/modifier pairs we advertise to clients.
///
/// LINEAR is universally importable across GPUs and compositors.
/// INVALID tells the driver to choose the best tiling autonomously.
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// (fourcc, modifier) pairs advertised to clients.
///
/// We advertise both LINEAR and INVALID. NVIDIA's proprietary EGL requires
/// DRM_FORMAT_MOD_INVALID to allocate DMA-BUFs (it does not support LINEAR
/// allocation via the modifier negotiation path). When a client submits a
/// buffer with INVALID modifier, the wayland backend falls back to mmap +
/// wl_shm presentation instead of forwarding the raw DMA-BUF fd.
const DMABUF_FORMATS: &[(u32, u64)] = &[
    // ARGB8888
    (0x3432_5241, DRM_FORMAT_MOD_LINEAR),
    (0x3432_5241, DRM_FORMAT_MOD_INVALID),
    // XRGB8888
    (0x3432_5258, DRM_FORMAT_MOD_LINEAR),
    (0x3432_5258, DRM_FORMAT_MOD_INVALID),
    // ABGR8888
    (0x3432_4241, DRM_FORMAT_MOD_LINEAR),
    (0x3432_4241, DRM_FORMAT_MOD_INVALID),
    // XBGR8888
    (0x3432_4258, DRM_FORMAT_MOD_LINEAR),
    (0x3432_4258, DRM_FORMAT_MOD_INVALID),
];

// ─── Dispatch impls ─────────────────────────────────────────────────

impl GlobalDispatch<ZwpLinuxDmabufV1, ()> for WaylandState {
    fn bind(
        state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwpLinuxDmabufV1>,
        _data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let dmabuf = data_init.init(resource, ());

        // Try host format passthrough first: advertise exactly the formats
        // the host compositor accepts so clients allocate buffers that can
        // be zero-copy forwarded without a GPU blit.
        let formats = state.host_dmabuf_formats.lock();
        let host_count = formats.values().map(|v| v.len()).sum::<usize>();

        if host_count > 0 {
            let mut total = 0usize;
            for (&fourcc, modifiers) in formats.iter() {
                for &modifier in modifiers {
                    let hi = (modifier >> 32) as u32;
                    let lo = modifier as u32;
                    dmabuf.modifier(fourcc, hi, lo);
                    total += 1;
                }
                // Always include DRM_FORMAT_MOD_INVALID for each format.
                // Some legacy clients may request INVALID when they don't
                // negotiate explicit modifiers. Including it ensures those
                // clients can still allocate buffers.
                if !modifiers.contains(&DRM_FORMAT_MOD_INVALID) {
                    let hi = (DRM_FORMAT_MOD_INVALID >> 32) as u32;
                    let lo = DRM_FORMAT_MOD_INVALID as u32;
                    dmabuf.modifier(fourcc, hi, lo);
                    total += 1;
                }
            }
            info!(
                total,
                host_formats = formats.len(),
                "zwp_linux_dmabuf_v1: bound, advertised host format/modifier pairs (zero-copy)"
            );
        } else {
            // Fallback: host formats not available (headless, no dmabuf
            // support on host, or roundtrip not yet complete). Use the
            // hardcoded set that covers common RGB888 variants.
            for &(fourcc, modifier) in DMABUF_FORMATS {
                let hi = (modifier >> 32) as u32;
                let lo = modifier as u32;
                dmabuf.modifier(fourcc, hi, lo);
            }
            info!(
                "zwp_linux_dmabuf_v1: bound, advertised {} fallback format/modifier pairs",
                DMABUF_FORMATS.len()
            );
        }
    }
}

impl Dispatch<ZwpLinuxDmabufV1, ()> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ZwpLinuxDmabufV1,
        request: zwp_linux_dmabuf_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_linux_dmabuf_v1::Request::CreateParams { params_id } => {
                data_init.init(
                    params_id,
                    DmaBufParamsData {
                        planes: Mutex::new(Vec::with_capacity(4)),
                        used: Mutex::new(false),
                    },
                );
            }
            zwp_linux_dmabuf_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwpLinuxBufferParamsV1, DmaBufParamsData> for WaylandState {
    fn request(
        _state: &mut Self,
        client: &Client,
        params: &ZwpLinuxBufferParamsV1,
        request: zwp_linux_buffer_params_v1::Request,
        data: &DmaBufParamsData,
        dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwp_linux_buffer_params_v1::Request::Add {
                fd,
                plane_idx,
                offset,
                stride,
                modifier_hi,
                modifier_lo,
            } => {
                let modifier = ((modifier_hi as u64) << 32) | modifier_lo as u64;
                {
                    let mut planes = data.planes.lock();
                    planes.push(DmaBufPlaneInfo {
                        fd,
                        plane_idx,
                        offset,
                        stride,
                        modifier,
                    });
                }
            }
            zwp_linux_buffer_params_v1::Request::CreateImmed {
                buffer_id,
                width,
                height,
                format,
                flags: _,
            } => {
                {
                    let mut used = data.used.lock();
                    if *used {
                        warn!("dmabuf params used twice");
                        return;
                    }
                    *used = true;
                }
                let mut planes = data.planes.lock();
                let taken = std::mem::take(&mut *planes);
                let modifier = taken.first().map(|p| p.modifier).unwrap_or(0);
                trace!(
                    width,
                    height,
                    format = format!("0x{:08x}", format),
                    modifier = format!("0x{:016x}", modifier),
                    num_planes = taken.len(),
                    "dmabuf: create_immed buffer"
                );
                let buf_data = BufferData::DmaBuf(DmaBufBufferData {
                    planes: taken,
                    width,
                    height,
                    format,
                    modifier,
                });
                data_init.init(buffer_id, buf_data);
            }
            zwp_linux_buffer_params_v1::Request::Create {
                width,
                height,
                format,
                flags: _,
            } => {
                {
                    let mut used = data.used.lock();
                    if *used {
                        params.failed();
                        return;
                    }
                    *used = true;
                }
                let mut planes = data.planes.lock();
                let taken = std::mem::take(&mut *planes);
                let modifier = taken.first().map(|p| p.modifier).unwrap_or(0);
                info!(
                    width,
                    height,
                    format = format!("0x{:08x}", format),
                    num_planes = taken.len(),
                    "dmabuf: async create buffer"
                );
                // Async create: allocate a WlBuffer server-side via the
                // client connection and send the `created` event back.
                let buf_data = BufferData::DmaBuf(DmaBufBufferData {
                    planes: taken,
                    width,
                    height,
                    format,
                    modifier,
                });
                match client.create_resource::<WlBuffer, BufferData, WaylandState>(dh, 1, buf_data)
                {
                    Ok(buffer) => {
                        params.created(&buffer);
                    }
                    Err(err) => {
                        warn!(?err, "dmabuf: failed to create wl_buffer for async create");
                        params.failed();
                    }
                }
            }
            zwp_linux_buffer_params_v1::Request::Destroy => {}
            _ => {}
        }
    }
}
