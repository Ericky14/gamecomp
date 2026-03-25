//! Wayland protocol implementations for the compositor server.
//!
//! Registers the minimum set of globals needed for XWayland and native
//! Wayland clients. Each global requires a `GlobalDispatch` impl (for
//! binding) and `Dispatch` impls for every child object it creates.
//!
//! Globals registered:
//! - `wl_compositor` / `wl_surface` / `wl_region` — surface management
//! - `wl_subcompositor` / `wl_subsurface` — sub-surface stacking
//! - `wl_shm` / `wl_shm_pool` / `wl_buffer` — shared memory buffers
//! - `wl_seat` / `wl_pointer` / `wl_keyboard` — input devices
//! - `wl_output` — display information
//! - `xdg_wm_base` / `xdg_surface` / `xdg_toplevel` — window management
//! - `wl_data_device_manager` — clipboard (required by XWayland)

mod compositor;
mod data_device;
mod dmabuf;
mod output;
mod seat;
mod shm;
mod subcompositor;
pub mod wl_drm;
pub mod xdg_shell;

use parking_lot::Mutex;
use std::os::unix::io::{AsFd, OwnedFd};
use std::sync::atomic::{AtomicBool, AtomicI32};

use tracing::{debug, info};
use wayland_protocols::wp::linux_dmabuf::zv1::server::zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1;
use wayland_protocols::xdg::shell::server::xdg_wm_base::XdgWmBase;
use wayland_server::DisplayHandle;
use wayland_server::backend::GlobalId;
use wayland_server::protocol::{
    wl_buffer::{self, WlBuffer},
    wl_compositor::WlCompositor,
    wl_data_device_manager::WlDataDeviceManager,
    wl_output::WlOutput,
    wl_seat::WlSeat,
    wl_shm::WlShm,
    wl_subcompositor::WlSubcompositor,
};

use super::WaylandState;

pub use dmabuf::{DmaBufBufferData, DmaBufPlaneInfo};
pub use output::OutputData;
pub use shm::ShmBufferData;

// ─── Buffer infrastructure ──────────────────────────────────────────

/// Per-surface state stored as user data on `WlSurface`.
pub struct SurfaceData {
    /// The currently attached buffer (pending until commit).
    ///
    /// Uses `Mutex` for interior mutability — wayland-server requires `Sync`
    /// on user data. This is only accessed from the main thread so the lock
    /// is uncontended.
    pub attached_buffer: Mutex<Option<WlBuffer>>,
    /// `true` if this surface has the cursor role (set via
    /// `wl_pointer.set_cursor`). Cursor surfaces must not be staged
    /// as application frames.
    pub is_cursor: AtomicBool,
    /// Cursor hotspot X, set by `wl_pointer.set_cursor`.
    pub hotspot_x: AtomicI32,
    /// Cursor hotspot Y, set by `wl_pointer.set_cursor`.
    pub hotspot_y: AtomicI32,
}

/// Wrapper enum for `WlBuffer` user data — either SHM or DMA-BUF.
pub enum BufferData {
    /// Shared-memory buffer.
    Shm(ShmBufferData),
    /// DMA-BUF zero-copy buffer.
    DmaBuf(DmaBufBufferData),
}

/// Plane info for a committed DMA-BUF, with dup'd file descriptors.
pub struct CommittedDmaBufPlane {
    /// Duplicated fd (owned by the receiver).
    pub fd: OwnedFd,
    /// Byte offset into the GEM object.
    pub offset: u32,
    /// Row stride in bytes.
    pub stride: u32,
}

/// A committed buffer from a client, sent through the channel to the
/// wayland event thread for zero-copy presentation on the host window.
pub enum CommittedBuffer {
    /// DMA-BUF zero-copy path — forward fds directly to host compositor.
    DmaBuf {
        /// Per-plane info with dup'd fds.
        planes: Vec<CommittedDmaBufPlane>,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// DRM fourcc format code.
        format: u32,
        /// DRM modifier.
        modifier: u64,
    },
    /// SHM CPU-copy fallback.
    Shm {
        /// Pixel data (ARGB8888 or XRGB8888).
        pixels: Vec<u8>,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Stride in bytes.
        stride: u32,
    },
}

// ─── Fence sync ─────────────────────────────────────────────────────

/// Wait for the GPU to finish writing to a DMA-BUF by exporting and waiting
/// on its implicit sync fence. This prevents the host compositor from reading
/// incomplete renders.
///
/// Uses `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` to get a fence fd, then `poll()` on
/// it. Fails silently if the ioctl is unsupported (kernel < 5.17).
pub(crate) fn sync_dma_buf_fence(fd: &impl AsFd) {
    // DMA_BUF_IOCTL_EXPORT_SYNC_FILE = _IOWR('b', 2, struct dma_buf_export_sync_file)
    // struct dma_buf_export_sync_file { __u32 flags; __s32 fd; }
    // DMA_BUF_SYNC_READ = 1
    #[repr(C)]
    struct DmaBufExportSyncFile {
        flags: u32,
        fd: i32,
    }
    const DMA_BUF_SYNC_READ: u32 = 1;
    // _IOWR('b', 2, 8) = 0xC0086202
    const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: libc::c_ulong = 0xC008_6202;

    let mut args = DmaBufExportSyncFile {
        flags: DMA_BUF_SYNC_READ,
        fd: -1,
    };

    // SAFETY: Valid fd, valid pointer to stack struct, ioctl is well-defined
    // for DMA-BUF fds. Returns -1 on error (kernel too old, not a DMA-BUF, etc.).
    let ret = unsafe {
        libc::ioctl(
            std::os::unix::io::AsRawFd::as_raw_fd(&fd.as_fd()),
            DMA_BUF_IOCTL_EXPORT_SYNC_FILE,
            &mut args as *mut DmaBufExportSyncFile,
        )
    };
    if ret < 0 {
        // Not supported or not a DMA-BUF — skip sync.
        debug!(
            "sync_dma_buf_fence: ioctl failed (errno={}, not a DMA-BUF or kernel too old)",
            std::io::Error::last_os_error()
        );
        return;
    }
    if args.fd >= 0 {
        // Poll/wait on the sync fence — blocks until GPU rendering completes.
        let mut pfd = libc::pollfd {
            fd: args.fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: Valid fd, valid pollfd struct, 1000ms timeout.
        let poll_ret = unsafe { libc::poll(&mut pfd, 1, 1000) };
        if poll_ret <= 0 {
            debug!(
                fence_fd = args.fd,
                poll_ret, "sync_dma_buf_fence: fence wait failed or timed out"
            );
        }
        unsafe {
            libc::close(args.fd);
        }
    }
}

// ─── Globals registration ───────────────────────────────────────────

/// Registered global IDs for cleanup.
pub struct Globals {
    pub compositor: GlobalId,
    pub subcompositor: GlobalId,
    pub shm: GlobalId,
    pub seat: GlobalId,
    pub output: GlobalId,
    pub xdg_shell: GlobalId,
    pub data_device_manager: GlobalId,
    pub linux_dmabuf: GlobalId,
    pub wl_drm: GlobalId,
}

/// Register all protocol globals on the display.
pub fn register_globals(dh: &DisplayHandle, output_width: u32, output_height: u32) -> Globals {
    let compositor = dh.create_global::<WaylandState, WlCompositor, ()>(6, ());
    let subcompositor = dh.create_global::<WaylandState, WlSubcompositor, ()>(1, ());
    let shm = dh.create_global::<WaylandState, WlShm, ()>(2, ());
    let seat = dh.create_global::<WaylandState, WlSeat, ()>(9, ());
    let output = dh.create_global::<WaylandState, WlOutput, OutputData>(
        4,
        OutputData {
            width: output_width,
            height: output_height,
        },
    );
    let xdg_shell = dh.create_global::<WaylandState, XdgWmBase, ()>(6, ());
    let data_device_manager = dh.create_global::<WaylandState, WlDataDeviceManager, ()>(3, ());
    let linux_dmabuf = dh.create_global::<WaylandState, ZwpLinuxDmabufV1, ()>(3, ());

    // Register wl_drm global — required for XWayland glamor (GPU acceleration).
    // Glamor needs this to discover the DRM render node path.
    let render_node = wl_drm::find_render_node();
    let wl_drm = dh.create_global::<WaylandState, wl_drm::wl_drm::WlDrm, wl_drm::WlDrmGlobalData>(
        2,
        wl_drm::WlDrmGlobalData {
            device_path: render_node,
            formats: wl_drm::WL_DRM_FORMATS.to_vec(),
        },
    );

    info!("registered Wayland protocol globals");

    Globals {
        compositor,
        subcompositor,
        shm,
        seat,
        output,
        xdg_shell,
        data_device_manager,
        linux_dmabuf,
        wl_drm,
    }
}

// ─── wl_buffer dispatch ─────────────────────────────────────────────

impl wayland_server::Dispatch<WlBuffer, BufferData> for WaylandState {
    fn request(
        _state: &mut Self,
        _client: &wayland_server::Client,
        _buffer: &WlBuffer,
        _request: wl_buffer::Request,
        _data: &BufferData,
        _dh: &DisplayHandle,
        _data_init: &mut wayland_server::DataInit<'_, Self>,
    ) {
        // Handle Destroy.
    }
}
