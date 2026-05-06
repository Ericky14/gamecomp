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
pub mod drm_syncobj;
mod output;
mod pointer_constraints;
pub mod presentation;
mod relative_pointer;
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
use wayland_protocols::wp::linux_drm_syncobj::v1::server::wp_linux_drm_syncobj_manager_v1::WpLinuxDrmSyncobjManagerV1;
use wayland_protocols::wp::pointer_constraints::zv1::server::zwp_pointer_constraints_v1::ZwpPointerConstraintsV1;
use wayland_protocols::wp::presentation_time::server::wp_presentation::WpPresentation;
use wayland_protocols::wp::relative_pointer::zv1::server::zwp_relative_pointer_manager_v1::ZwpRelativePointerManagerV1;
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
pub use drm_syncobj::{SyncPoint, SyncobjDevice, SyncobjGlobalData};
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
    /// XWayland server index that owns this surface. `u32::MAX` for
    /// non-XWayland surfaces. Set at creation time from the client→server
    /// mapping so the commit handler can disambiguate surfaces with the
    /// same protocol_id across different XWayland servers.
    pub server_index: u32,
    /// Pending `wp_presentation_feedback` callbacks for this surface.
    /// Filled by `wp_presentation.feedback` requests, drained on commit
    /// and moved into `WaylandState.staged_feedbacks`.
    pub pending_feedbacks: Mutex<
        Vec<wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::WpPresentationFeedback>,
    >,
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
        /// Explicit sync acquire point. When set, the render thread must
        /// wait for this timeline point before reading the buffer.
        acquire_point: Option<SyncPoint>,
        /// Explicit sync release point. When set, the render thread signals
        /// this timeline point instead of relying on `wl_buffer.release`.
        release_point: Option<SyncPoint>,
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

impl CommittedBuffer {
    /// Duplicate the buffer by dup'ing all owned file descriptors.
    ///
    /// Used to keep an overlay buffer alive across frames while sending
    /// a copy to the render thread.
    pub fn try_dup(&self) -> std::io::Result<Self> {
        match self {
            Self::DmaBuf {
                planes,
                width,
                height,
                format,
                modifier,
                acquire_point,
                release_point,
            } => {
                let mut dup_planes = Vec::with_capacity(planes.len());
                for p in planes {
                    dup_planes.push(CommittedDmaBufPlane {
                        fd: p.fd.try_clone()?,
                        offset: p.offset,
                        stride: p.stride,
                    });
                }
                Ok(Self::DmaBuf {
                    planes: dup_planes,
                    width: *width,
                    height: *height,
                    format: *format,
                    modifier: *modifier,
                    acquire_point: acquire_point.clone(),
                    release_point: release_point.clone(),
                })
            }
            Self::Shm {
                pixels,
                width,
                height,
                stride,
            } => Ok(Self::Shm {
                pixels: pixels.clone(),
                width: *width,
                height: *height,
                stride: *stride,
            }),
        }
    }
}

impl CommittedBuffer {
    /// Signal the explicit sync release point (if any) and discard it.
    ///
    /// **Must** be called before dropping a `CommittedBuffer` whose frame
    /// will never be presented (e.g., coalesced / replaced). Without this,
    /// the client永远 waits for the release signal and its buffer pool gets
    /// exhausted, causing a rendering freeze.
    pub fn signal_release(&mut self, device: &SyncobjDevice) {
        if let Self::DmaBuf { release_point, .. } = self
            && let Some(rp) = release_point.take()
        {
            let _ = device.timeline_signal(rp.handle(), rp.point);
        }
    }
}

/// A complete frame with optional overlay layers.
///
/// Sent through the channel to the render thread / wayland backend.
/// Contains the primary app buffer plus optional overlay buffers for
/// multi-layer composition.
pub struct CommittedFrame {
    /// Primary application buffer (always present).
    pub app: CommittedBuffer,
    /// Steam overlay buffer (LAYER_OVERLAY). `None` if no overlay visible.
    pub overlay: Option<CommittedBuffer>,
    /// External overlay buffer (LAYER_EXTERNAL_OVERLAY). `None` if no external
    /// overlay visible.
    pub external_overlay: Option<CommittedBuffer>,
    /// Overlay opacity (0.0–1.0, from `_NET_WM_OPACITY`).
    pub overlay_opacity: f32,
    /// External overlay opacity (0.0–1.0).
    pub external_overlay_opacity: f32,
}

impl CommittedFrame {
    /// Signal all explicit sync release points in this frame.
    ///
    /// Called when discarding a frame that will never be presented
    /// (coalesced in the render thread or replaced in the staging area).
    pub fn signal_release_points(&mut self, device: &SyncobjDevice) {
        self.app.signal_release(device);
        if let Some(ref mut buf) = self.overlay {
            buf.signal_release(device);
        }
        if let Some(ref mut buf) = self.external_overlay {
            buf.signal_release(device);
        }
    }
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
    pub pointer_constraints: GlobalId,
    pub relative_pointer_manager: GlobalId,
    pub presentation: GlobalId,
    pub drm_syncobj: Option<GlobalId>,
}

/// Register all protocol globals on the display.
pub fn register_globals(
    dh: &DisplayHandle,
    output_width: u32,
    output_height: u32,
    syncobj_device: Option<SyncobjDevice>,
) -> Globals {
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

    // Pointer constraints — allows clients to lock/confine the pointer.
    let pointer_constraints = dh.create_global::<WaylandState, ZwpPointerConstraintsV1, ()>(1, ());

    // Relative pointer — provides raw dx/dy while pointer is locked.
    let relative_pointer_manager =
        dh.create_global::<WaylandState, ZwpRelativePointerManagerV1, ()>(1, ());

    // wp_presentation_time — accurate frame presentation feedback.
    // Critical for XWayland's X11 Present extension; without it,
    // GTK/Flutter clients fall back to a software 60Hz timer.
    let presentation = dh.create_global::<WaylandState, WpPresentation, ()>(2, ());

    // wp_linux_drm_syncobj_manager_v1 — explicit GPU synchronization.
    // Only available in DRM mode when the kernel supports syncobj_eventfd.
    // XWayland uses this for zero-copy flip mode: acquire fences tell the
    // compositor when the client GPU finishes, release fences tell the
    // client when the display is done scanning out the buffer.
    let drm_syncobj = syncobj_device.map(|device| {
        let global = dh.create_global::<WaylandState, WpLinuxDrmSyncobjManagerV1, SyncobjGlobalData>(
            1,
            SyncobjGlobalData {
                device: Some(device),
            },
        );
        info!("wp_linux_drm_syncobj_manager_v1: registered");
        global
    });

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
        pointer_constraints,
        relative_pointer_manager,
        presentation,
        drm_syncobj,
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
