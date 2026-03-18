//! Legacy `wl_drm` protocol implementation.
//!
//! This protocol is required by XWayland's glamor (GPU-accelerated rendering)
//! to discover the DRM render node. Without it, glamor falls back to software
//! rendering even if `zwp_linux_dmabuf_v1` is available.
//!
//! Flow:
//! 1. Client binds `wl_drm` global
//! 2. Server sends `device("/dev/dri/renderDXXX")`, `capabilities(PRIME)`,
//!    and `format(...)` for each supported format
//! 3. Client sends `authenticate(0)` — server immediately responds `authenticated`
//! 4. Client opens the render node, creates GBM + EGL context (glamor init)
//! 5. Buffers are submitted via `zwp_linux_dmabuf_v1` (modern) or
//!    `create_prime_buffer` (legacy fallback)
//!
//! The `authenticate` request is a no-op because render nodes don't require
//! DRM authentication. Flink-based `create_buffer`/`create_planar_buffer`
//! are rejected — only PRIME fd passing is supported.

// Generate protocol types from the XML definition.
#[allow(
    non_snake_case,
    non_upper_case_globals,
    non_camel_case_types,
    unused_qualifications
)]
pub mod generated {
    // Re-export wayland_server so the generated code can find it as `super::wayland_server`.
    pub(crate) use wayland_server;
    use wayland_server::protocol::*;

    pub mod __interfaces {
        use wayland_backend;
        use wayland_server::protocol::__interfaces::*;

        wayland_scanner::generate_interfaces!("protocols/wayland-drm.xml");
    }
    use self::__interfaces::*;

    wayland_scanner::generate_server_code!("protocols/wayland-drm.xml");
}

pub use generated::wl_drm;

use std::os::unix::io::AsFd;
use std::path::PathBuf;

use tracing::{info, warn};
use wayland_server::{Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource};

use super::BufferData;
use crate::wayland::WaylandState;

// ─── Data types ─────────────────────────────────────────────────────

/// Per-global data stored on the `wl_drm` global.
pub struct WlDrmGlobalData {
    /// Path to the DRM render node (e.g., `/dev/dri/renderD128`).
    pub device_path: PathBuf,
    /// DRM fourcc format codes to advertise.
    pub formats: Vec<u32>,
}

/// Per-instance data stored on each bound `wl_drm` resource.
pub struct WlDrmInstanceData {
    /// Fourcc formats (copied from global).
    pub formats: Vec<u32>,
}

// ─── Format constants ───────────────────────────────────────────────

/// DRM formats advertised via `wl_drm`. These are the common formats
/// that XWayland/glamor uses.
pub const WL_DRM_FORMATS: &[u32] = &[
    0x3432_5241, // ARGB8888
    0x3432_5258, // XRGB8888
    0x3432_4241, // ABGR8888
    0x3432_4258, // XBGR8888
    0x3432_4752, // RGB888
    0x3432_4742, // BGR888
];

// ─── GlobalDispatch — handle client binding ─────────────────────────

impl GlobalDispatch<wl_drm::WlDrm, WlDrmGlobalData> for WaylandState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<wl_drm::WlDrm>,
        global_data: &WlDrmGlobalData,
        data_init: &mut DataInit<'_, Self>,
    ) {
        let instance_data = WlDrmInstanceData {
            formats: global_data.formats.clone(),
        };
        let drm = data_init.init(resource, instance_data);

        // Send device path — this is the critical piece glamor needs.
        drm.device(global_data.device_path.to_string_lossy().into_owned());

        // Send PRIME capability (v2).
        if drm.version() >= 2 {
            drm.capabilities(wl_drm::Capability::Prime as u32);
        }

        // Advertise supported formats.
        for &fourcc in &global_data.formats {
            drm.format(fourcc);
        }

        info!(
            device = %global_data.device_path.display(),
            num_formats = global_data.formats.len(),
            "wl_drm: bound, sent device + {} formats",
            global_data.formats.len()
        );
    }
}

// ─── Dispatch — handle client requests ──────────────────────────────

impl Dispatch<wl_drm::WlDrm, WlDrmInstanceData> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        drm: &wl_drm::WlDrm,
        request: wl_drm::Request,
        data: &WlDrmInstanceData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wl_drm::Request::Authenticate { .. } => {
                // Render nodes don't need DRM authentication — immediately confirm.
                drm.authenticated();
                info!("wl_drm: authenticated (no-op for render nodes)");
            }

            wl_drm::Request::CreateBuffer { .. } => {
                // Flink handles are legacy and insecure — reject.
                drm.post_error(
                    wl_drm::Error::InvalidName,
                    String::from("Flink handles are unsupported, use PRIME"),
                );
            }

            wl_drm::Request::CreatePlanarBuffer { .. } => {
                // Flink planar buffers — also reject.
                drm.post_error(
                    wl_drm::Error::InvalidName,
                    String::from("Flink handles are unsupported, use PRIME"),
                );
            }

            wl_drm::Request::CreatePrimeBuffer {
                id,
                name: fd,
                width,
                height,
                format,
                offset0,
                stride0,
                ..
            } => {
                // Validate format.
                if !data.formats.contains(&format) {
                    drm.post_error(
                        wl_drm::Error::InvalidFormat,
                        String::from("format not advertised by wl_drm"),
                    );
                    return;
                }

                if width < 1 || height < 1 {
                    drm.post_error(
                        wl_drm::Error::InvalidFormat,
                        String::from("width or height not positive"),
                    );
                    return;
                }

                info!(
                    width,
                    height,
                    format = format!("0x{:08x}", format),
                    "wl_drm: create_prime_buffer"
                );

                // Import as a single-plane DMA-BUF with INVALID modifier
                // (wl_drm doesn't negotiate modifiers).
                use super::{DmaBufBufferData, DmaBufPlaneInfo};
                let owned_fd = match rustix::io::dup(fd.as_fd()) {
                    Ok(f) => f,
                    Err(e) => {
                        warn!(?e, "wl_drm: failed to dup prime fd");
                        drm.post_error(
                            wl_drm::Error::InvalidName,
                            String::from("failed to dup prime fd"),
                        );
                        return;
                    }
                };

                let plane = DmaBufPlaneInfo {
                    fd: owned_fd,
                    plane_idx: 0,
                    offset: offset0 as u32,
                    stride: stride0 as u32,
                    modifier: 0x00ff_ffff_ffff_ffff, // DRM_FORMAT_MOD_INVALID
                };

                let buf_data = BufferData::DmaBuf(DmaBufBufferData {
                    planes: vec![plane],
                    width,
                    height,
                    format,
                    modifier: 0x00ff_ffff_ffff_ffff,
                });

                data_init.init(id, buf_data);
            }
        }
    }
}

// Buffer dispatch for wl_drm-created buffers uses the same BufferData
// dispatch that is already registered in protocols.rs for WlBuffer.

// ─── Render node discovery ──────────────────────────────────────────

/// Discover the DRM render node path for the system's primary GPU.
///
/// Uses udev GPU discovery to find the boot GPU's render node. Falls back
/// to scanning `/dev/dri/renderD128`–`renderD143` if udev enumeration fails.
pub fn find_render_node() -> PathBuf {
    // Try udev-based discovery for an accurate result on multi-GPU systems.
    if let Ok(gpus) = crate::backend::gpu_discovery::discover_gpus("seat0")
        && let Some(primary) = crate::backend::gpu_discovery::select_primary_gpu(&gpus)
        && let Some(render) = crate::backend::gpu_discovery::render_node_for(&primary.dev_path)
    {
        info!(path = %render.display(), "found DRM render node for primary GPU");
        return render;
    }

    // Fallback: scan render nodes directly.
    for i in 128..144 {
        let path = PathBuf::from(format!("/dev/dri/renderD{}", i));
        if path.exists() {
            info!(path = %path.display(), "found DRM render node (fallback scan)");
            return path;
        }
    }
    warn!("no DRM render node found, defaulting to /dev/dri/renderD128");
    PathBuf::from("/dev/dri/renderD128")
}
